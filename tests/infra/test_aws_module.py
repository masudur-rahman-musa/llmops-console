r"""`infra/terraform/aws`: the security and cost claims `terraform validate` accepts either way.

The same trade as `test_hetzner_module.py`, for the same two reasons: `terraform validate`
accepts a publicly reachable database, a bucket-wide IAM policy and a US region with equal
calm, and terraform is not installed on the development machine. So the
claims a syntactically perfect module can still break are pinned here as plain text, and
`fmt`, `init` and `validate` are proven by the `tf-aws` CI job.

Every path is taken from this file's own `__file__`, never from the working directory, and
every read goes through `read()`, so a file the module does not have yet fails the one case
that needs it instead of collapsing the whole file at collection.

The helpers and the token-scan patterns are imported from `test_hetzner_module.py` rather
than copied: one reader, one definition of a token-shaped literal. One helper is local.
`named_blocks` keys a block on its FIRST quoted label, so `resource "aws_security_group"
"postgres"` and `resource "aws_security_group" "redis"` collide under one key; `resources()`
below keys on the `(type, name)` pair instead, and `resource_list()` keeps duplicates so a
count can see them.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from infra.test_hetzner_module import (
    ASSIGNMENT,
    SECRETISH_NAME,
    TOKEN_LITERAL,
    WORLD,
    argument,
    body_after,
    inner_blocks,
    load_yaml,
    named_blocks,
    provider_block,
    read,
    recipe,
    steps,
    steps_running,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE = REPO_ROOT / "infra" / "terraform" / "aws"
VERSIONS_TF = MODULE / "versions.tf"
PROVIDERS_TF = MODULE / "providers.tf"
VARIABLES_TF = MODULE / "variables.tf"
NETWORK_TF = MODULE / "network.tf"
EKS_TF = MODULE / "eks.tf"
DATA_TF = MODULE / "data.tf"
BACKUPS_TF = MODULE / "backups.tf"
BUDGET_TF = MODULE / "budget.tf"
OUTPUTS_TF = MODULE / "outputs.tf"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
MAKEFILE = REPO_ROOT / "Makefile"

#: Section 1's validation criterion, in the order the plan names them.
VALIDATED_VARIABLES = ("admin_ips", "region", "name", "kubernetes_version", "budget_email")

#: Section 1's output list.
EXPECTED_OUTPUTS = {
    "region",
    "cluster_name",
    "cluster_endpoint",
    "rds_endpoint",
    "redis_endpoint",
    "backup_bucket",
}

#: Never in `outputs.tf`: a project rule and decision E.
FORBIDDEN_IN_OUTPUTS = ("password", "token", "secret", "kubeconfig")

#: The exact conditions the plan agrees. Written out literally.
REGION_CONDITION = 'can(regex("^eu-[a-z]+-[0-9]$", var.region))'
NAME_CONDITION = 'can(regex("^[a-z][a-z0-9-]{0,19}$", var.name))'

#: The node group's `2` that section 1 names three times.
NODE_COUNT = "2"

#: The one IRSA policy's whole reach: the bucket and the objects in it.
BACKUP_RESOURCES = {"aws_s3_bucket.backups.arn", '"${aws_s3_bucket.backups.arn}/*"'}

#: Criterion 42: both halves of the plan condition, `&&` and not `||`.
PLAN_CONDITION = "env.TF_API_TOKEN_PRESENT == 'true' && vars.AWS_PLAN_ROLE_ARN != ''"

#: Criterion 45, each its own recipe line.
AWS_INFRA_CHECK_LINES = (
    "terraform -chdir=infra/terraform/aws fmt -check -recursive",
    "terraform -chdir=infra/terraform/aws init -backend=false -input=false",
    "terraform -chdir=infra/terraform/aws validate",
)

#: An argument NAMED `password`, not the substring: `manage_master_user_password = true`
#: is the criterion's own required line and must not count as a finding.
PASSWORD_ARGUMENT = re.compile(r"(?m)^\s*password\s*=")

RESOURCE_HEADER = re.compile(r'\bresource\s+"([^"]+)"\s+"([^"]+)"\s*\{')

#: An inline `ingress {` block or `ingress = [...]` attribute, not the word in a description.
INLINE_INGRESS = re.compile(r"(?m)^\s*ingress\s*[={]")


def is_ingress_rule(kind: str, body: str) -> bool:
    """A standalone ingress rule: the VPC rule resource, or a legacy rule of `type` ingress."""
    if kind == "aws_vpc_security_group_ingress_rule":
        return True
    return kind == "aws_security_group_rule" and argument(body, "type") == '"ingress"'


def resource_list(text: str) -> list[tuple[str, str, str]]:
    """Every `resource "type" "name" {` in `text` as `(type, name, body)`, duplicates kept."""
    return [
        (match.group(1), match.group(2), body_after(text, match.end() - 1))
        for match in RESOURCE_HEADER.finditer(text)
    ]


def resources(text: str) -> dict[tuple[str, str], str]:
    """Every resource body in `text`, keyed on `(type, name)`."""
    return {(kind, name): body for kind, name, body in resource_list(text)}


def resource(path: Path, kind: str, name: str) -> str:
    """The body of one named resource in `path`, failing this case alone when absent."""
    found = resources(read(path))
    assert (kind, name) in found, f"{path.name} declares no {kind}.{name}, only {sorted(found)}"
    return found[(kind, name)]


def module_block(path: Path, name: str) -> str:
    """The body of `module "<name>" {` in `path`."""
    found = named_blocks(read(path), "module")
    assert name in found, f"{path.name} declares no module {name}, only {sorted(found)}"
    return found[name]


def tf_files() -> list[Path]:
    """Every `.tf` file of the module, asserting there is at least one to scan."""
    assert MODULE.is_dir(), f"{MODULE} does not exist"
    found = sorted(MODULE.rglob("*.tf"))
    assert found != [], f"{MODULE} holds no .tf file"
    return found


def all_resources() -> list[tuple[str, str, str]]:
    """Every resource across every `.tf` file of the module, duplicates kept."""
    return [entry for path in tf_files() for entry in resource_list(read(path))]


def top_level(body: str) -> str:
    """`body` with every nested `{ ... }` emptied, so only its own keys remain."""
    kept: list[str] = []
    depth = 0
    for character in body:
        if character == "}":
            depth -= 1
        if depth == 0:
            kept.append(character)
        if character == "{":
            depth += 1
    return "".join(kept)


def map_body(body: str, key: str) -> str:
    """The body of `key = { ... }` in `body`."""
    match = re.search(rf"(?m)^\s*{key}\s*=\s*\{{", body)
    assert match is not None, f"no `{key} = {{` map"
    return body_after(body, match.end() - 1)


def regex_literal(condition: str) -> str:
    """The pattern inside `regex("...")` in a condition, HCL's `\\` escape undone."""
    match = re.search(r'regex\("((?:[^"\\]|\\.)*)"', condition)
    assert match is not None, f"no regex( literal in {condition}"
    return match.group(1).replace("\\\\", "\\")


def conditions(variable: str) -> list[str]:
    """Every `condition` of `variable`'s validation blocks, as written."""
    variables = named_blocks(read(VARIABLES_TF), "variable")
    assert variable in variables, f"variables.tf declares no {variable}"
    return [
        argument(block, "condition") for block in inner_blocks(variables[variable], "validation")
    ]


def tf_aws_job() -> dict[str, Any]:
    """The `tf-aws` job of `ci.yml`."""
    jobs = load_yaml(WORKFLOW)["jobs"]
    assert isinstance(jobs, dict)
    assert "tf-aws" in jobs, f"ci.yml declares no `tf-aws` job, only {sorted(jobs)}"
    job = jobs["tf-aws"]
    assert isinstance(job, dict)
    return job


def is_printer(step: dict[str, Any]) -> bool:
    """A step whose script is an `echo` only: it names commands, runs none."""
    return str(step.get("run", "")).strip().startswith("echo")


# ---------------------------------------------------------------------------------------------
# Inputs


def test_region_defaults_to_frankfurt_and_its_condition_is_the_eu_pattern() -> None:
    """Criterion 6. Catches a `region` default of `us-east-1` (the provider's habit), which puts
    every resource outside the residency boundary a project rule is written around; and catches a
    condition widened to `^(eu|us)-` or dropped to `length(var.region) > 0`, where a US region
    plans cleanly. Equality on the whole condition also catches `|| true` appended to it."""
    variables = named_blocks(read(VARIABLES_TF), "variable")
    assert "region" in variables, "variables.tf declares no region"

    assert argument(variables["region"], "default") == '"eu-central-1"'
    assert conditions("region") == [REGION_CONDITION]


@pytest.mark.parametrize(
    ("value", "accepted"),
    [
        ("eu-central-1", True),
        ("eu-west-3", True),
        ("eu-north-1", True),
        ("us-east-1", False),
        ("ap-south-1", False),
        ("xeu-central-1", False),
        ("eu-central-12", False),
        ("eu-central", False),
        ("EU-central-1", False),
    ],
)
def test_the_region_pattern_refuses_every_non_eu_region(value: str, accepted: bool) -> None:
    """Criterion 40's non-EU row, read off the file's own pattern. Catches the pattern losing an
    anchor (`eu-[a-z]+-[0-9]` unanchored accepts `xeu-central-1`) or its `^eu-` prefix, which
    the row set above makes visible as a US region accepted rather than a string mismatch."""
    variables = named_blocks(read(VARIABLES_TF), "variable")
    assert "region" in variables, "variables.tf declares no region"
    pattern = regex_literal(" ".join(conditions("region")))

    assert (re.search(pattern, value) is not None) is accepted


def test_the_name_condition_is_the_agreed_pattern_on_the_name_variable() -> None:
    """Criterion 39. Catches a `name` validation checking only `length(var.name) > 0`, where a
    21-character or upper-case name reaches EKS and RDS identifiers that refuse it mid-apply;
    and catches the right regex applied to the wrong variable (`var.region` pasted from above)."""
    assert conditions("name") == [NAME_CONDITION]


@pytest.mark.parametrize(
    ("value", "accepted"),
    [
        ("a", True),
        ("aioc-evidence", True),
        ("a" + "b" * 19, True),
        ("a" + "b" * 20, False),
        ("", False),
        ("1abc", False),
        ("-abc", False),
        ("Abc", False),
        ("ab_c", False),
        ("ab c", False),
    ],
)
def test_the_name_pattern_holds_its_twenty_character_boundary(value: str, accepted: bool) -> None:
    """Criterion 39's unhappy path with the inclusive boundary: 20 characters
    in, 21 out. Catches `{0,20}` or `{1,19}` in the pattern, and a pattern that lets a leading
    digit, a hyphen, a capital or an underscore through."""
    pattern = regex_literal(" ".join(conditions("name")))

    assert (re.search(pattern, value) is not None) is accepted


def test_every_input_is_validated_and_the_admin_cidrs_have_no_default() -> None:
    """Criteria 15, 38 and 40. Catches a variable declared with no `validation` block, parsed
    per variable so one variable's block cannot stand in for a neighbour's; catches `admin_ips`
    acquiring a default, where an unset variable opens the EKS API to that default instead of
    refusing to plan; and catches an `admin_ips` validation that checks neither the empty list
    nor that each entry is a CIDR."""
    variables = named_blocks(read(VARIABLES_TF), "variable")
    missing = [name for name in VALIDATED_VARIABLES if name not in variables]
    unvalidated = [
        name
        for name in VALIDATED_VARIABLES
        if name in variables and not inner_blocks(variables[name], "validation")
    ]

    assert missing == [], f"variables.tf declares no {missing}"
    assert unvalidated == [], f"no validation block on {unvalidated}"
    assert argument(variables["admin_ips"], "default") == "", "admin_ips carries a default"

    joined = " ".join(conditions("admin_ips"))
    assert "length(var.admin_ips) > 0" in joined, "an empty admin_ips is accepted"
    assert "cidrhost(" in joined, "admin_ips entries are not checked as CIDRs"


# ---------------------------------------------------------------------------------------------
# Terraform and provider pins


def test_the_terraform_block_pins_core_provider_and_its_own_workspace() -> None:
    """Criteria 33 and 34. Catches the `cloud` workspace copied from the Hetzner module as
    `ai-operations-console-hetzner`, where this module's `destroy` would plan against the live
    Hetzner state; catches a missing `cloud {` block (state in a file CI throws away); and
    catches an unconstrained `hashicorp/aws`, where the next major lands on the next `init`."""
    text = read(VERSIONS_TF)
    cloud = re.search(r"(?m)^\s*cloud\s*\{", text)
    assert cloud is not None, "no Terraform Cloud block"
    workspaces = inner_blocks(body_after(text, cloud.end() - 1), "workspaces")
    aws = provider_block(text, "aws")

    assert re.search(r'required_version\s*=\s*">= 1\.9"', text) is not None
    assert [argument(block, "name") for block in workspaces] == ['"ai-operations-console-aws"']
    assert argument(aws, "source") == '"hashicorp/aws"'
    assert argument(aws, "version") == '"~> 6.64"'


def test_the_provider_uses_the_validated_region_and_tags_everything_as_evidence() -> None:
    """Criterion 32. Catches `region = "eu-central-1"` hard-coded in the provider, which makes
    the validated variable decorative, and catches `default_tags` missing or carrying a
    different value, which is how a forgotten evidence resource hides from the cost explorer."""
    providers = named_blocks(read(PROVIDERS_TF), "provider")
    assert "aws" in providers, "providers.tf declares no aws provider"
    default_tags = inner_blocks(providers["aws"], "default_tags")
    assert len(default_tags) == 1, f"{len(default_tags)} default_tags blocks"
    tags = map_body(default_tags[0], "tags")

    assert argument(providers["aws"], "region") == "var.region"
    assert argument(tags, "project") == '"ai-operations-console"'
    assert argument(tags, "purpose") == '"evidence"'


@pytest.mark.parametrize(
    ("path_name", "module", "source", "version"),
    [
        ("network.tf", "vpc", "terraform-aws-modules/vpc/aws", "6.7.2"),
        ("eks.tf", "eks", "terraform-aws-modules/eks/aws", "21.25.0"),
    ],
)
def test_the_community_modules_are_pinned_exactly(
    path_name: str, module: str, source: str, version: str
) -> None:
    """Criterion 35 and decision A. Catches `version = "~> 21.0"` or no version at all, where
    the next `init` pulls a release with renamed inputs (v21 already renamed `cluster_*`), and
    catches a module sourced from a fork or a git URL rather than the approved registry path."""
    body = module_block(MODULE / path_name, module)

    assert argument(body, "source") == f'"{source}"'
    assert argument(body, "version") == f'"{version}"'


# ---------------------------------------------------------------------------------------------
# Network and cluster


def test_the_vpc_spans_two_zones_with_public_and_private_subnets() -> None:
    """Criterion 1. Catches `azs` taking every zone in the region (three NAT-bearing subnets
    where the budget was drawn for two) or a single zone (RDS subnet groups refuse one), and
    catches either subnet list missing or empty, where EKS nodes or the load balancer have
    nowhere to land."""
    vpc = module_block(NETWORK_TF, "vpc")

    assert argument(vpc, "azs") == "slice(data.aws_availability_zones.available.names, 0, 2)"
    for subnets in ("private_subnets", "public_subnets"):
        value = argument(vpc, subnets)
        assert value not in ("", "[]"), f"{subnets} is {value or 'absent'}"


def test_the_cluster_api_is_reachable_only_from_the_admin_cidrs() -> None:
    """Criterion 14. Catches `endpoint_public_access_cidrs` omitted, where the module's default
    is `0.0.0.0/0` and the Kubernetes API is open to the internet, and catches it set to a
    literal list that drifts from the validated `admin_ips`."""
    eks = module_block(EKS_TF, "eks")

    assert argument(eks, "endpoint_public_access_cidrs") == "var.admin_ips"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("instance_types", '["t4g.medium"]'),
        ("min_size", NODE_COUNT),
        ("desired_size", NODE_COUNT),
        ("max_size", NODE_COUNT),
    ],
)
def test_the_one_node_group_is_two_arm_mediums(name: str, expected: str) -> None:
    """Criterion 2, one row per value. Catches a second instance type (an x86 `t3.medium` beside
    the arm one, which mixes architectures under one AMI), a `max_size` above 2 that lets the
    autoscaler outrun the 10 USD budget, and a `min_size` of 1 where one node cannot hold the
    workload the evidence run deploys."""
    groups = map_body(module_block(EKS_TF, "eks"), "eks_managed_node_groups")
    names = re.findall(r"(?m)^\s*([A-Za-z0-9_-]+)\s*=\s*\{", top_level(groups))
    assert len(names) == 1, f"{len(names)} managed node groups: {names}"
    group = map_body(groups, names[0])

    assert argument(group, name) == expected


def test_the_node_group_runs_an_arm64_ami() -> None:
    """Criterion 2's AMI. Catches `AL2023_x86_64_STANDARD` beside `t4g.medium`, which fails at
    node launch with an architecture mismatch that `terraform validate` cannot see."""
    groups = map_body(module_block(EKS_TF, "eks"), "eks_managed_node_groups")
    names = re.findall(r"(?m)^\s*([A-Za-z0-9_-]+)\s*=\s*\{", top_level(groups))
    assert len(names) == 1, f"{len(names)} managed node groups: {names}"

    assert "ARM_64" in argument(map_body(groups, names[0]), "ami_type")


# ---------------------------------------------------------------------------------------------
# Data stores


def test_postgres_is_version_16_on_the_smallest_arm_class() -> None:
    """Criterion 3. Catches `engine_version = "15"` (the major the app was never tested on) or
    an engine of `aurora-postgresql`, and catches `db.t3.micro` or a larger class, which the
    evidence budget was not drawn for."""
    rds = resource(DATA_TF, "aws_db_instance", "postgres")

    assert argument(rds, "engine") == '"postgres"'
    assert re.fullmatch(r'"16(\.\d+)*"', argument(rds, "engine_version")) is not None
    assert argument(rds, "instance_class") == '"db.t4g.micro"'


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("publicly_accessible", "false"),
        ("storage_encrypted", "true"),
        ("manage_master_user_password", "true"),
        ("skip_final_snapshot", "true"),
        ("deletion_protection", "false"),
    ],
)
def test_postgres_carries_each_agreed_flag(name: str, expected: str) -> None:
    """Criteria 7 and 9, one row per flag. Catches `publicly_accessible = true` (a public DNS
    name for the database), an unencrypted volume, a password handled outside Secrets Manager,
    and the two destroy blockers: a final snapshot that outlives the run and bills, and deletion
    protection that makes `destroy` fail halfway."""
    rds = resource(DATA_TF, "aws_db_instance", "postgres")

    assert argument(rds, name) == expected


def test_no_tf_file_carries_a_password_argument() -> None:
    """Criterion 8 and decision E. Catches `password = var.db_password` or a `random_password`
    result wired into RDS, which puts the credential into state and plan output. Matched as an
    argument NAME at line start, so the required `manage_master_user_password = true` is not a
    finding; the directory and its `.tf` files are asserted first, so a missing module cannot
    pass as a clean scan."""
    findings = [path.name for path in tf_files() if PASSWORD_ARGUMENT.search(read(path))]

    assert findings == [], f"a password argument in {findings}"


def test_redis_is_a_single_node_7_replication_group_encrypted_both_ways() -> None:
    """Criteria 4 and 10. Catches `aws_elasticache_cluster` in place of the replication group
    (which cannot carry transit encryption), `num_cache_clusters = 2` doubling the node bill,
    a Redis 6 engine, a larger node type, and either encryption flag left `false` - the only
    protection decision E leaves an auth-token-less Redis."""
    redis = resource(DATA_TF, "aws_elasticache_replication_group", "redis")

    assert argument(redis, "engine") == '"redis"'
    assert re.fullmatch(r'"7(\.\d+)*"', argument(redis, "engine_version")) is not None
    assert argument(redis, "node_type") == '"cache.t4g.micro"'
    assert argument(redis, "num_cache_clusters") == "1"
    assert argument(redis, "transit_encryption_enabled") == "true"
    assert argument(redis, "at_rest_encryption_enabled") == "true"


@pytest.mark.parametrize(
    ("store_kind", "store", "group_argument", "group_kind", "group"),
    [
        ("aws_db_instance", "postgres", "db_subnet_group_name", "aws_db_subnet_group", "postgres"),
        (
            "aws_elasticache_replication_group",
            "redis",
            "subnet_group_name",
            "aws_elasticache_subnet_group",
            "redis",
        ),
    ],
)
def test_each_data_store_lives_in_the_private_subnets(
    store_kind: str, store: str, group_argument: str, group_kind: str, group: str
) -> None:
    """Criterion 10's placement and criterion 3's by the same reading. Catches a subnet group
    built on `module.vpc.public_subnets`, which puts the store one security-group edit away from
    the internet, and catches a store that names no subnet group at all and lands in the
    default VPC instead of this one."""
    body = resource(DATA_TF, store_kind, store)
    subnet_group = resource(DATA_TF, group_kind, group)

    assert argument(body, group_argument) == f"{group_kind}.{group}.name"
    assert argument(subnet_group, "subnet_ids") == "module.vpc.private_subnets"


@pytest.mark.parametrize(
    ("group", "port"),
    [("postgres", "5432"), ("redis", "6379")],
)
def test_each_store_group_admits_only_the_node_group_on_its_own_port(group: str, port: str) -> None:
    """Criteria 11 and 12. Catches a second ingress rule on the group (a CIDR rule "for
    debugging", in any file of the module), an inline `ingress` on the security group itself,
    a rule sourced from a CIDR or a prefix list instead of `module.eks.node_security_group_id`,
    a port range wider than the one port, and `ip_protocol = "-1"` (all traffic)."""
    security_group = resource(DATA_TF, "aws_security_group", group)
    target = f"aws_security_group.{group}.id"
    rules = [
        body
        for kind, _name, body in all_resources()
        if is_ingress_rule(kind, body) and argument(body, "security_group_id") == target
    ]

    assert INLINE_INGRESS.search(security_group) is None, f"inline ingress on {group}"
    assert len(rules) == 1, f"{len(rules)} ingress rules on aws_security_group.{group}"
    rule = rules[0]
    assert argument(rule, "referenced_security_group_id") == "module.eks.node_security_group_id"
    assert argument(rule, "from_port") == port
    assert argument(rule, "to_port") == port
    assert argument(rule, "ip_protocol") == '"tcp"'
    for widening in ("cidr_ipv4", "cidr_ipv6", "prefix_list_id", "cidr_blocks"):
        assert argument(rule, widening) == "", f"{group} rule also admits {widening}"


@pytest.mark.parametrize(
    ("store_kind", "store", "list_argument", "group"),
    [
        ("aws_db_instance", "postgres", "vpc_security_group_ids", "postgres"),
        ("aws_elasticache_replication_group", "redis", "security_group_ids", "redis"),
    ],
)
def test_each_store_is_attached_to_its_own_security_group(
    store_kind: str, store: str, list_argument: str, group: str
) -> None:
    """Criteria 11 and 12 made load-bearing. Catches a store that never names its group, which
    AWS then attaches to the VPC default group, so the node-only rule pinned above guards a
    group with nothing behind it and the test beside it stays green on an unguarded store."""
    body = resource(DATA_TF, store_kind, store)

    assert argument(body, list_argument) == f"[aws_security_group.{group}.id]"


def test_no_ingress_anywhere_in_the_module_names_the_world() -> None:
    """Criterion 13, over every `.tf` file. Catches `cidr_ipv4 = "0.0.0.0/0"` on a standalone
    ingress-rule resource, which an inline-block scan alone would miss, and catches an inline
    `ingress {` block naming `::/0` on any group, the IPv6 half that is easy to leave open."""
    findings: list[str] = []
    for path in tf_files():
        text = read(path)
        bodies = inner_blocks(text, "ingress") + [
            body for kind, _name, body in resource_list(text) if is_ingress_rule(kind, body)
        ]
        findings += [f"{path.name}: {net}" for body in bodies for net in WORLD if net in body]

    assert findings == [], f"world-open ingress: {findings}"


# ---------------------------------------------------------------------------------------------
# Backups and IRSA


def test_the_backup_bucket_destroys_with_its_objects() -> None:
    """Criterion 29. Catches `force_destroy` omitted, where the first backup the evidence run
    writes makes `terraform destroy` fail on a non-empty bucket and leaves the cluster billing."""
    bucket = resource(BACKUPS_TF, "aws_s3_bucket", "backups")

    assert argument(bucket, "force_destroy") == "true"


@pytest.mark.parametrize(
    "flag",
    ["block_public_acls", "block_public_policy", "ignore_public_acls", "restrict_public_buckets"],
)
def test_the_backup_bucket_blocks_every_public_path(flag: str) -> None:
    """Criterion 27, one row per flag. Catches any one of the four left `false` or omitted, and
    catches the block attached to some other bucket, since `bucket` must name this one."""
    block = resource(BACKUPS_TF, "aws_s3_bucket_public_access_block", "backups")

    assert "aws_s3_bucket.backups" in argument(block, "bucket")
    assert argument(block, flag) == "true"


def test_the_backup_bucket_is_encrypted_and_versioned() -> None:
    """Criterion 28. Catches the encryption resource with no `sse_algorithm`, versioning set to
    `Suspended`, and either resource pointing at a different bucket than the backups one."""
    encryption = resource(
        BACKUPS_TF, "aws_s3_bucket_server_side_encryption_configuration", "backups"
    )
    versioning = resource(BACKUPS_TF, "aws_s3_bucket_versioning", "backups")

    assert "aws_s3_bucket.backups" in argument(encryption, "bucket")
    assert re.search(r"(?m)^\s*sse_algorithm\s*=", encryption) is not None
    assert "aws_s3_bucket.backups" in argument(versioning, "bucket")
    assert argument(versioning, "status") == '"Enabled"'


def test_there_is_one_irsa_role_trusted_through_the_cluster_oidc_provider() -> None:
    """Criterion 5's "one role". Catches a second IAM role added in any file (a node-wide role
    with S3 access is the shortcut), and catches a trust policy that does not go through the
    cluster's OIDC provider, which is what makes the role IRSA rather than assumable by anyone
    the policy happens to name."""
    roles = [(name, body) for kind, name, body in all_resources() if kind == "aws_iam_role"]
    names = [name for name, _body in roles]

    assert names == ["backups"], f"IAM roles: {names}"
    assert "module.eks.oidc_provider_arn" in roles[0][1]


def test_the_irsa_policy_reaches_the_backup_bucket_and_nothing_else() -> None:
    """Criterion 5's "that bucket only". Catches `Resource = "*"`, a second bucket's ARN beside
    this one, a `NotResource` that grants everything except the bucket, and a managed-policy
    attachment (`AmazonS3FullAccess`) that widens the role without touching the inline policy."""
    policies = [
        (name, body) for kind, name, body in all_resources() if kind == "aws_iam_role_policy"
    ]
    attachments = [
        f"{kind}.{name}"
        for kind, name, _body in all_resources()
        if kind in ("aws_iam_role_policy_attachment", "aws_iam_policy_attachment")
    ]
    backups_text = read(BACKUPS_TF)

    assert [name for name, _body in policies] == ["backups"]
    assert attachments == [], f"managed policies attached: {attachments}"
    policy = policies[0][1]
    assert "aws_iam_role.backups" in argument(policy, "role")
    assert "notresource" not in policy.lower()
    assert "managed_policy_arns" not in backups_text
    assert '"*"' not in backups_text, 'backups.tf names a "*" resource'

    reached: set[str] = set()
    for match in re.finditer(r"(?i)\bresources?\s*=\s*", policy):
        rest = policy[match.end() :]
        if rest.startswith("["):
            closing = rest.index("]")
            reached |= {item.strip() for item in rest[1:closing].split(",") if item.strip()}
        else:
            reached.add(rest.splitlines()[0].strip().rstrip(","))
    assert reached == BACKUP_RESOURCES


# ---------------------------------------------------------------------------------------------
# Budget and outputs


def test_the_budget_is_ten_dollars_a_month_of_cost() -> None:
    """Criterion 30. Catches `limit_amount = "100"`, a `USAGE` budget that measures hours rather
    than money, and a `QUARTERLY` period that lets a forgotten cluster run for weeks first."""
    budget = resource(BUDGET_TF, "aws_budgets_budget", "evidence")

    assert argument(budget, "budget_type") == '"COST"'
    assert argument(budget, "limit_amount") == '"10"'
    assert argument(budget, "limit_unit") == '"USD"'
    assert argument(budget, "time_unit") == '"MONTHLY"'


def test_the_budget_mails_the_owner_at_eighty_percent_of_actual_spend() -> None:
    """Criterion 31, the guard against a forgotten destroy. Catches `FORECASTED` (which fires on
    a projection, or not at all in the first hours of a month), `threshold = 100` (the alert
    after the money is gone), an `ABSOLUTE_VALUE` threshold type that reads 80 as 80 USD, and a
    subscriber hard-coded instead of the validated `budget_email`."""
    budget = resource(BUDGET_TF, "aws_budgets_budget", "evidence")
    notifications = inner_blocks(budget, "notification")
    assert len(notifications) == 1, f"{len(notifications)} notification blocks"
    notification = notifications[0]

    assert argument(notification, "comparison_operator") == '"GREATER_THAN"'
    assert argument(notification, "threshold") == "80"
    assert argument(notification, "threshold_type") == '"PERCENTAGE"'
    assert argument(notification, "notification_type") == '"ACTUAL"'
    assert argument(notification, "subscriber_email_addresses") == "[var.budget_email]"


def test_the_outputs_are_the_six_agreed_facts_and_no_credential() -> None:
    """Criteria 36 and 37. Catches a seventh output added for convenience - a kubeconfig, a
    cluster auth token, the RDS master secret ARN - which Terraform Cloud stores and prints in
    every run log; and catches one of the six dropped. Equality on the set fails both ways."""
    text = read(OUTPUTS_TF)
    declared = set(named_blocks(text, "output"))
    leaked = [word for word in FORBIDDEN_IN_OUTPUTS if word in text.lower()]

    assert declared == EXPECTED_OUTPUTS
    assert leaked == [], f"outputs.tf names {leaked}"


def test_no_file_in_the_module_carries_a_token_shaped_literal() -> None:
    r"""Criterion 47 and a project rule. Catches a real AWS or Terraform Cloud credential pasted into
    `terraform.tfvars.example` while trying the plan out, which is why every file is scanned
    and the example is not exempted. The patterns are the Hetzner module's, imported: a
    `(token|secret|password|api_key)`-ish name, matched with `re.fullmatch`,
    assigned a quoted run of 20+ `[A-Za-z0-9+/=_-]` characters. The directory is asserted
    first, because an empty scan of a missing directory passes vacuously."""
    assert MODULE.is_dir(), f"{MODULE} does not exist"
    findings: list[str] = []
    for path in sorted(MODULE.rglob("*")):
        if not path.is_file():
            continue
        for name, value in ASSIGNMENT.findall(path.read_text(encoding="utf-8")):
            if SECRETISH_NAME.fullmatch(name) and TOKEN_LITERAL.fullmatch(value):
                findings.append(f"{path.name}: {name}")

    assert findings == [], f"token-shaped literals: {findings}"


# ---------------------------------------------------------------------------------------------
# CI and Makefile


def test_the_tf_aws_job_holds_only_the_oidc_and_read_permissions() -> None:
    """Criterion 44 and decision C. Catches `permissions` omitted, where the job inherits the
    repository default and cannot mint an OIDC token at all, and catches `contents: write` or an
    extra scope, which hands a job that runs on pull requests more than it reads."""
    job = tf_aws_job()

    assert job.get("permissions") == {"id-token": "write", "contents": "read"}


@pytest.mark.parametrize(
    "fragment",
    ["terraform fmt -check", "terraform init", "terraform validate"],
)
def test_the_tf_aws_job_checks_terraform_unconditionally(fragment: str) -> None:
    """Criterion 41. Catches the plan's `if` copied onto `fmt`, `init` or `validate`, which makes
    the gate inert on exactly the pull requests with no credentials, and catches a job whose
    working directory is still the Hetzner module, so it checks the wrong code green. The
    `echo`-only skip step is excluded: it may name these commands."""
    job = tf_aws_job()
    matched = [step for step in steps_running(job, fragment) if not is_printer(step)]

    assert job.get("defaults", {}).get("run", {}).get("working-directory") == (
        "infra/terraform/aws"
    )
    assert len(matched) == 1, f"{len(matched)} steps run `{fragment}`"
    assert "if" not in matched[0], f"`{fragment}` is conditional"


def test_the_tf_aws_init_falls_back_to_no_backend_without_a_token() -> None:
    """Criterion 41's `init -backend=false`. Catches an init that always contacts Terraform Cloud,
    which fails every fork pull request before `validate` runs, and catches the opposite, an
    init that never uses the backend, so the plan step has no state to plan against."""
    job = tf_aws_job()
    inits = [step for step in steps_running(job, "terraform init") if not is_printer(step)]
    assert len(inits) == 1, f"{len(inits)} steps run `terraform init`"
    script = str(inits[0]["run"])

    assert "terraform init -backend=false -input=false" in script
    assert re.search(r"(?m)^\s*terraform init -input=false\s*$", script) is not None


def test_the_tf_aws_plan_runs_only_with_both_credentials_after_assuming_the_role() -> None:
    """Criterion 42 and decision C. Catches `||` in place of `&&` (a plan with a token and no
    role fails on credentials), an unconditional plan, a condition on an env flag the job never
    defines (always false, so the plan silently never runs), and the credentials step placed
    after the plan or gated differently from it. The `echo` skip is excluded."""
    job = tf_aws_job()
    listed = steps(job)
    planners = [step for step in steps_running(job, "terraform plan") if not is_printer(step)]
    credentials = [
        step
        for step in listed
        if str(step.get("uses", "")).startswith("aws-actions/configure-aws-credentials@v4")
    ]

    assert "TF_API_TOKEN_PRESENT" in job.get("env", {}), "the plan's env flag is never defined"
    assert len(planners) == 1, f"{len(planners)} steps run `terraform plan`"
    assert len(credentials) == 1, f"{len(credentials)} configure-aws-credentials steps"
    assert planners[0].get("if") == PLAN_CONDITION
    assert credentials[0].get("if") == PLAN_CONDITION
    assert listed.index(credentials[0]) < listed.index(planners[0]), "credentials after plan"


def test_the_tf_aws_job_says_so_when_the_plan_is_skipped() -> None:
    """Criterion 43. Catches the silent skip, where a green tick reads as "the plan was checked",
    and catches a skip conditioned on the token alone, which logs nothing on the commonest
    case: the token set and the read-only role not yet created by the owner."""
    job = tf_aws_job()
    skips = [
        step
        for step in steps(job)
        if is_printer(step)
        and "TF_API_TOKEN" in str(step.get("if", ""))
        and "AWS_PLAN_ROLE_ARN" in str(step.get("if", ""))
        and step.get("if") != PLAN_CONDITION
    ]

    assert len(skips) == 1, f"{len(skips)} visible skip steps naming both credentials"


def test_infra_check_runs_the_aws_module_lines_each_on_its_own() -> None:
    """Criterion 45 and a project rule. Catches an aws line missing from the recipe (the local
    gate then never reads the module), the three chained onto one line with `&&` so only the
    last exit code counts, and any terraform line swallowing failure with `||`."""
    lines = recipe(read(MAKEFILE), "infra-check")
    missing = [line for line in AWS_INFRA_CHECK_LINES if line not in lines]
    swallowed = [line for line in lines if "terraform" in line and "||" in line]

    assert missing == [], f"infra-check lacks {missing}"
    assert swallowed == [], f"a swallowed failure: {swallowed}"
