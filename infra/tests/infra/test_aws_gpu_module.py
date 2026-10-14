r"""`infra/terraform/aws-gpu`: the cost and exposure claims `terraform validate` accepts either way.

The same trade as `test_aws_module.py`: `terraform validate` accepts a GPU instance with SSH
open to the world, no shutdown timer and a US region with equal calm, and terraform is not
installed on the development machine. So the claims a syntactically perfect
module can still break are pinned here as plain text, and `fmt`, `init` and `validate` are
proven by the `tf-aws-gpu` CI job.

Every path is taken from this file's own `__file__`, and every read goes through `read()`, so a
file the module does not have yet fails the one case that needs it instead of collapsing the
whole file at collection.

The block readers and the token-scan patterns are imported from `test_hetzner_module.py`. `entries()`, `regex_literal()`, `conditions()` and `is_printer()` are
local copies of `test_aws_module.py`'s, because importing between the two module test files
is not the pattern this directory uses.
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

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE = REPO_ROOT / "infra" / "terraform" / "aws-gpu"
VERSIONS_TF = MODULE / "versions.tf"
PROVIDERS_TF = MODULE / "providers.tf"
VARIABLES_TF = MODULE / "variables.tf"
MAIN_TF = MODULE / "main.tf"
CLOUD_INIT = MODULE / "cloud-init.yaml"
BUDGET_TF = MODULE / "budget.tf"
OUTPUTS_TF = MODULE / "outputs.tf"
HETZNER_CLOUD_INIT = REPO_ROOT / "infra" / "terraform" / "hetzner" / "cloud-init.yaml"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
MAKEFILE = REPO_ROOT / "Makefile"

#: Section 1's input-validation criterion, in the order the plan names them.
VALIDATED_VARIABLES = (
    "admin_ips",
    "region",
    "name",
    "instance_type",
    "ssh_public_key",
    "budget_email",
    "auto_terminate_minutes",
)

EXPECTED_OUTPUTS = {"region", "instance_id", "public_ip"}
FORBIDDEN_IN_OUTPUTS = ("password", "token", "secret", "kubeconfig")

#: The exact conditions the plan agrees, written out literally. Raw strings:
#: the HCL source carries `\\.`, two characters, and so does each literal here.
REGION_CONDITION = r'can(regex("^eu-[a-z]+-[0-9]$", var.region))'
NAME_CONDITION = r'can(regex("^[a-z][a-z0-9-]{0,19}$", var.name))'
INSTANCE_TYPE_CONDITION = r'can(regex("^g[56]\\.xlarge$", var.instance_type))'
SSH_KEY_CONDITION = r'startswith(var.ssh_public_key, "ssh-ed25519 ")'
AUTO_TERMINATE_CONDITION = (
    "var.auto_terminate_minutes >= 30 && var.auto_terminate_minutes <= 480"
)

AMI_NAME_FILTER = '["Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) *"]'

PLAN_CONDITION = "env.TF_API_TOKEN_PRESENT == 'true' && vars.AWS_PLAN_ROLE_ARN != ''"

AWS_GPU_INFRA_CHECK_LINES = (
    "terraform -chdir=infra/terraform/aws-gpu fmt -check -recursive",
    "terraform -chdir=infra/terraform/aws-gpu init -backend=false -input=false",
    "terraform -chdir=infra/terraform/aws-gpu validate",
)

#: Ports no ingress may open: vLLM's API and the k3s API server.
FORBIDDEN_INGRESS_PORTS = (8000, 6443)

RESOURCE_HEADER = re.compile(r'\bresource\s+"([^"]+)"\s+"([^"]+)"\s*\{')
DATA_HEADER = re.compile(r'\bdata\s+"([^"]+)"\s+"([^"]+)"\s*\{')


def entries(pattern: re.Pattern[str], text: str) -> list[tuple[str, str, str]]:
    """Every `<keyword> "type" "name" {` in `text` as `(type, name, body)`, duplicates kept."""
    return [
        (match.group(1), match.group(2), body_after(text, match.end() - 1))
        for match in pattern.finditer(text)
    ]


def tf_files() -> list[Path]:
    """Every `.tf` file of the module, asserting there is at least one to scan."""
    assert MODULE.is_dir(), f"{MODULE} does not exist"
    found = sorted(MODULE.rglob("*.tf"))
    assert found != [], f"{MODULE} holds no .tf file"
    return found


def all_resources() -> list[tuple[str, str, str]]:
    return [
        entry for path in tf_files() for entry in entries(RESOURCE_HEADER, read(path))
    ]


def resources_of(kind: str) -> list[tuple[str, str]]:
    """`(name, body)` for every resource of `kind` anywhere in the module."""
    return [(name, body) for found, name, body in all_resources() if found == kind]


def sole_resource(kind: str) -> str:
    found = resources_of(kind)
    assert len(found) == 1, (
        f"{len(found)} {kind} resources: {[name for name, _ in found]}"
    )
    return found[0][1]


def variable(name: str) -> str:
    variables = named_blocks(read(VARIABLES_TF), "variable")
    assert name in variables, (
        f"variables.tf declares no {name}, only {sorted(variables)}"
    )
    return variables[name]


def conditions(name: str) -> list[str]:
    """Every `condition` of `name`'s validation blocks, as written."""
    return [
        argument(block, "condition")
        for block in inner_blocks(variable(name), "validation")
    ]


def regex_literal(condition: str) -> str:
    """The pattern inside `regex("...")` in a condition, HCL's `\\` escape undone."""
    match = re.search(r'regex\("((?:[^"\\]|\\.)*)"', condition)
    assert match is not None, f"no regex( literal in {condition}"
    return match.group(1).replace("\\\\", "\\")


def is_printer(step: dict[str, Any]) -> bool:
    """A step whose script is an `echo` only: it names commands, runs none."""
    return str(step.get("run", "")).strip().startswith("echo")


def is_ingress_rule(kind: str, body: str) -> bool:
    """A standalone ingress rule: the VPC rule resource, or a legacy rule of `type` ingress."""
    if kind == "aws_vpc_security_group_ingress_rule":
        return True
    return kind == "aws_security_group_rule" and argument(body, "type") == '"ingress"'


def tf_aws_gpu_job() -> dict[str, Any]:
    jobs = load_yaml(WORKFLOW)["jobs"]
    assert isinstance(jobs, dict)
    assert "tf-aws-gpu" in jobs, (
        f"ci.yml declares no `tf-aws-gpu` job, only {sorted(jobs)}"
    )
    job = jobs["tf-aws-gpu"]
    assert isinstance(job, dict)
    return job


# ---------------------------------------------------------------------------------------------
# The instance


def test_one_instance_of_the_validated_type_defaulting_to_g5_xlarge() -> None:
    """Criterion 30. Catches a hard-coded `instance_type = "g5.xlarge"` that makes the validated
    variable decorative, a default of `g5.2xlarge` (twice the hourly bill for no extra GPU), a
    second instance, and the type condition widened past the two single-GPU sizes."""
    instance = sole_resource("aws_instance")

    assert argument(instance, "instance_type") == "var.instance_type"
    assert argument(variable("instance_type"), "default") == '"g5.xlarge"'
    assert conditions("instance_type") == [INSTANCE_TYPE_CONDITION]


def test_the_ami_is_amazons_latest_deep_learning_gpu_base() -> None:
    """Criterion 31. Catches `owners` naming a marketplace or a wildcard account (anyone can
    publish an AMI with that name), `most_recent` dropped (the lookup then fails on more than one
    match), a filter on a plain Ubuntu AMI that carries no NVIDIA driver, and the instance not
    using the looked-up AMI at all."""
    amis = [
        (name, body)
        for path in tf_files()
        for kind, name, body in entries(DATA_HEADER, read(path))
        if kind == "aws_ami"
    ]
    assert len(amis) == 1, f"{len(amis)} aws_ami data sources"
    name, ami = amis[0]
    name_filters = [
        block
        for block in inner_blocks(ami, "filter")
        if argument(block, "name") == '"name"'
    ]

    assert argument(ami, "owners") == '["amazon"]'
    assert argument(ami, "most_recent") == "true"
    assert [argument(block, "values") for block in name_filters] == [AMI_NAME_FILTER]
    assert argument(sole_resource("aws_instance"), "ami") == f"data.aws_ami.{name}.id"


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("volume_type", '"gp3"'),
        ("volume_size", "100"),
        ("delete_on_termination", "true"),
        ("encrypted", "true"),
    ],
)
def test_the_root_volume_carries_each_agreed_value(key: str, expected: str) -> None:
    """Criterion 32, one row per value. Catches `delete_on_termination = false`, where a 100GB
    volume outlives the destroy and bills monthly unseen; an unencrypted volume holding model
    weights; `gp2`; and a size too small for the image plus the 15GB of weights."""
    blocks = inner_blocks(sole_resource("aws_instance"), "root_block_device")
    assert len(blocks) == 1, f"{len(blocks)} root_block_device blocks"

    assert argument(blocks[0], key) == expected


def test_the_security_group_has_exactly_one_ingress_ssh_from_the_admins() -> None:
    """Criterion 33. Catches a second inline `ingress {` in any file (8000 opened "to test the
    API from a laptop"), SSH admitted from a literal CIDR list that drifts from the validated
    `admin_ips`, a port range around 22, and `protocol = "-1"`."""
    blocks = [
        block for path in tf_files() for block in inner_blocks(read(path), "ingress")
    ]
    assert len(blocks) == 1, f"{len(blocks)} ingress blocks in the module"
    rule = blocks[0]

    assert argument(rule, "from_port") == "22"
    assert argument(rule, "to_port") == "22"
    assert argument(rule, "protocol") == '"tcp"'
    assert argument(rule, "cidr_blocks") == "var.admin_ips"


def ingress_bodies() -> list[tuple[str, str]]:
    """Every inline `ingress {` body and every standalone ingress-rule resource, by file."""
    found: list[tuple[str, str]] = []
    for path in tf_files():
        text = read(path)
        found += [(path.name, body) for body in inner_blocks(text, "ingress")]
        found += [
            (path.name, body)
            for kind, _name, body in entries(RESOURCE_HEADER, text)
            if is_ingress_rule(kind, body)
        ]
    return found


def opens(body: str, port: int) -> bool:
    """Whether an ingress body's port range covers `port`; all-protocol and unparsed count."""
    for protocol_key in ("protocol", "ip_protocol"):
        if argument(body, protocol_key) in ('"-1"', '"all"', "-1"):
            return True
    low = argument(body, "from_port")
    high = argument(body, "to_port")
    if low == "" and high == "":
        return False
    if not (low.isdigit() and high.isdigit()):
        return True
    return int(low) <= port <= int(high)


def test_no_ingress_anywhere_names_the_world_or_opens_8000_or_6443() -> None:
    """Criterion 34, reading ONLY ingress bodies: egress legitimately names `0.0.0.0/0`. Catches
    a standalone `aws_vpc_security_group_ingress_rule` with `cidr_ipv4 = "0.0.0.0/0"` (the
    inline-block count above does not see it), `ipv6_cidr_blocks = ["::/0"]` added beside the
    admin CIDRs on the SSH rule, and any ingress whose port range covers vLLM's 8000 or the
    k3s API's 6443, including an all-protocol rule."""
    bodies = ingress_bodies()
    world = [f"{name}: {net}" for name, body in bodies for net in WORLD if net in body]
    ports = [
        f"{name}: {port}"
        for name, body in bodies
        for port in FORBIDDEN_INGRESS_PORTS
        if opens(body, port)
    ]

    assert bodies != [], "no ingress found to scan"
    assert world == [], f"world-open ingress: {world}"
    assert ports == [], f"ingress opening a forbidden port: {ports}"


def test_the_instance_terminates_when_it_shuts_down() -> None:
    """Criterion 35. Catches the default `stop`, where the cloud-init shutdown timer leaves a
    stopped instance with a 100GB volume and an Elastic address still billing."""
    assert argument(
        sole_resource("aws_instance"), "instance_initiated_shutdown_behavior"
    ) == ('"terminate"')


def test_cloud_init_arms_the_shutdown_timer_from_the_variable() -> None:
    """Criterion 36 and decision F. Catches the timer missing, a literal minute count that drifts
    from the validated variable, a default other than 240, and `user_data = file(...)`, which
    ships the placeholder unrendered so `shutdown` refuses it and the guard never arms."""
    instance = sole_resource("aws_instance")

    assert "shutdown -h +${auto_terminate_minutes}" in read(CLOUD_INIT)
    assert argument(variable("auto_terminate_minutes"), "default") == "240"
    assert "templatefile(" in instance
    assert "cloud-init.yaml" in instance
    assert re.search(
        r"auto_terminate_minutes\s*=\s*var\.auto_terminate_minutes", instance
    )


def test_cloud_init_installs_the_same_k3s_as_hetzner() -> None:
    """Criterion 37. Catches the GPU host on a different k3s release from the long-lived cluster,
    so a manifest valid on one refuses on the other, and catches an install with no version,
    which takes the channel's latest. Every occurrence in both files is read."""
    pattern = re.compile(r"INSTALL_K3S_VERSION=(\S+)")
    hetzner = set(pattern.findall(read(HETZNER_CLOUD_INIT)))
    gpu = pattern.findall(read(CLOUD_INIT))

    assert len(hetzner) == 1, hetzner
    assert gpu != [], "cloud-init.yaml pins no INSTALL_K3S_VERSION"
    assert set(gpu) == hetzner


# ---------------------------------------------------------------------------------------------
# Budget, provider and outputs


def test_the_budget_is_fifteen_dollars_of_cost_mailed_at_eighty_percent_actual() -> (
    None
):
    """Criteria 38 and 39. Catches a `USAGE` budget that counts hours, `limit_amount = "150"`,
    `FORECASTED` (silent in the first hours of a month, which is exactly a forgotten GPU),
    `threshold = 100`, and a subscriber hard-coded instead of the validated `budget_email`."""
    budgets = [
        body
        for kind, _name, body in entries(RESOURCE_HEADER, read(BUDGET_TF))
        if kind == "aws_budgets_budget"
    ]
    assert len(budgets) == 1, f"{len(budgets)} aws_budgets_budget in budget.tf"
    budget = budgets[0]
    notifications = inner_blocks(budget, "notification")
    assert len(notifications) == 1, f"{len(notifications)} notification blocks"
    notification = notifications[0]

    assert argument(budget, "budget_type") == '"COST"'
    assert argument(budget, "limit_amount") == '"15"'
    assert argument(budget, "limit_unit") == '"USD"'
    assert argument(notification, "threshold") == "80"
    assert argument(notification, "notification_type") == '"ACTUAL"'
    assert argument(notification, "subscriber_email_addresses") == "[var.budget_email]"


def test_the_provider_tags_everything_as_the_gpu_measurement() -> None:
    """Criterion 40. Catches `purpose = "evidence"` copied from the `aws` module, which files the
    GPU spend under the wrong run in the cost explorer, and `default_tags` dropped entirely."""
    providers = named_blocks(read(PROVIDERS_TF), "provider")
    assert "aws" in providers, "providers.tf declares no aws provider"
    default_tags = inner_blocks(providers["aws"], "default_tags")
    assert len(default_tags) == 1, f"{len(default_tags)} default_tags blocks"
    match = re.search(r"(?m)^\s*tags\s*=\s*\{", default_tags[0])
    assert match is not None, "default_tags carries no tags map"
    tags = body_after(default_tags[0], match.end() - 1)

    assert argument(tags, "project") == '"ai-operations-console"'
    assert argument(tags, "purpose") == '"gpu-measurement"'


def test_the_terraform_block_pins_core_provider_and_its_own_workspace() -> None:
    """Criterion 41. Catches the workspace copied from the `aws` module as
    `ai-operations-console-aws`, where this module's destroy plans against the evidence stack's
    state, a missing `cloud {` block, and an unconstrained `hashicorp/aws`."""
    text = read(VERSIONS_TF)
    cloud = re.search(r"(?m)^\s*cloud\s*\{", text)
    assert cloud is not None, "no Terraform Cloud block"
    workspaces = inner_blocks(body_after(text, cloud.end() - 1), "workspaces")
    aws = provider_block(text, "aws")

    assert re.search(r'required_version\s*=\s*">= 1\.9"', text) is not None
    assert [argument(block, "name") for block in workspaces] == [
        '"ai-operations-console-aws-gpu"'
    ]
    assert argument(aws, "source") == '"hashicorp/aws"'
    assert argument(aws, "version") == '"~> 6.64"'


def test_the_outputs_are_the_three_agreed_facts_and_no_credential() -> None:
    """Criteria 42 and 43. Catches a fourth output added for convenience (a kubeconfig fetched
    over SSH, which Terraform Cloud then stores), one of the three dropped, and any of the four
    forbidden words anywhere in `outputs.tf`."""
    text = read(OUTPUTS_TF)
    leaked = [word for word in FORBIDDEN_IN_OUTPUTS if word in text.lower()]

    assert set(named_blocks(text, "output")) == EXPECTED_OUTPUTS
    assert leaked == [], f"outputs.tf names {leaked}"


# ---------------------------------------------------------------------------------------------
# Inputs


def test_every_input_is_validated_and_admin_ips_has_no_default() -> None:
    """Criterion 44. Catches a variable declared with no `validation` block, parsed per variable
    so one block cannot stand in for a neighbour's; `admin_ips` acquiring a default, where an
    unset variable opens SSH to that default instead of refusing to plan; and a `region`
    default outside the EU."""
    variables = named_blocks(read(VARIABLES_TF), "variable")
    missing = [name for name in VALIDATED_VARIABLES if name not in variables]
    unvalidated = [
        name
        for name in VALIDATED_VARIABLES
        if name in variables and not inner_blocks(variables[name], "validation")
    ]

    assert missing == [], f"variables.tf declares no {missing}"
    assert unvalidated == [], f"no validation block on {unvalidated}"
    assert argument(variables["admin_ips"], "default") == "", (
        "admin_ips carries a default"
    )
    assert argument(variables["region"], "default") == '"eu-central-1"'


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("region", REGION_CONDITION),
        ("name", NAME_CONDITION),
        ("instance_type", INSTANCE_TYPE_CONDITION),
        ("ssh_public_key", SSH_KEY_CONDITION),
        ("auto_terminate_minutes", AUTO_TERMINATE_CONDITION),
    ],
)
def test_each_condition_is_exactly_the_agreed_one(name: str, expected: str) -> None:
    """Criterion 44, one row per variable. Catches a condition dropped to `length(...) > 0`, the
    right condition pasted onto the wrong variable, `|| true` appended, `ssh-rsa` accepted, and
    the auto-terminate bound written `> 30` so the boundary moves. Equality on the one condition
    each variable carries sees a second, looser validation block too."""
    assert conditions(name) == [expected]


def test_admin_ips_refuses_an_empty_list_and_a_non_cidr_entry() -> None:
    """Criterion 45's first two rows. Catches an `admin_ips` validation that checks only length
    (a hostname or a bare address then reaches `cidr_blocks` and fails mid-apply) or only shape
    (an empty list plans a group nobody can SSH into), and `anytrue` in place of `alltrue`."""
    joined = " ".join(conditions("admin_ips"))

    assert "length(var.admin_ips) > 0" in joined
    assert "alltrue([for cidr in var.admin_ips : can(cidrhost(cidr, 0))])" in joined


@pytest.mark.parametrize(
    ("name", "value", "accepted"),
    [
        ("region", "eu-central-1", True),
        ("region", "us-east-1", False),
        ("region", "xeu-central-1", False),
        ("instance_type", "g5.xlarge", True),
        ("instance_type", "g6.xlarge", True),
        ("instance_type", "p4d.24xlarge", False),
        ("instance_type", "g5.2xlarge", False),
        ("instance_type", "g5xxlarge", False),
    ],
)
def test_the_region_and_type_patterns_refuse_the_unhappy_rows(
    name: str, value: str, accepted: bool
) -> None:
    """Criterion 45's region and type rows, read off the file's own pattern. The exact-condition
    case above already reddens on any edit to the file; this one catches the AGREED pattern
    itself being wrong, a region pattern without its anchor that admits `xeu-central-1`, or a
    type pattern with an unescaped dot that admits `g5xxlarge`, where both files agree."""
    pattern = regex_literal(" ".join(conditions(name)))

    assert (re.search(pattern, value) is not None) is accepted


@pytest.mark.parametrize(
    ("value", "accepted"),
    [(29, False), (30, True), (480, True), (481, False)],
)
def test_auto_terminate_bounds_hold_their_inclusive_edges(
    value: int, accepted: bool
) -> None:
    """Criterion 45's `29` and `481` rows with the inclusive edges. Parsed from
    the condition, so it catches the agreed bounds themselves being off by one (`> 30` agreed as
    `>= 31`), which the exact-condition case cannot see when both files carry the same typo."""
    joined = " ".join(conditions("auto_terminate_minutes"))
    low = re.findall(r"var\.auto_terminate_minutes\s*>=\s*(\d+)", joined)
    high = re.findall(r"var\.auto_terminate_minutes\s*<=\s*(\d+)", joined)
    assert len(low) == 1 and len(high) == 1, (low, high)

    assert (int(low[0]) <= value <= int(high[0])) is accepted


# ---------------------------------------------------------------------------------------------
# CI and Makefile


def test_the_tf_aws_gpu_job_holds_only_the_oidc_and_read_permissions() -> None:
    """Criterion 48. Catches `permissions` omitted (no OIDC token, so the plan never assumes the
    role) and `contents: write` or any extra scope on a job that runs on pull requests."""
    assert tf_aws_gpu_job().get("permissions") == {
        "id-token": "write",
        "contents": "read",
    }


@pytest.mark.parametrize(
    "fragment",
    ["terraform fmt -check", "terraform init", "terraform validate"],
)
def test_the_tf_aws_gpu_job_checks_terraform_unconditionally(fragment: str) -> None:
    """Criterion 46. Catches the plan's `if` copied onto `fmt`, `init` or `validate`, which makes
    the gate inert on every pull request without credentials, and a working directory still
    pointed at the `aws` module, which checks the wrong code green. The `echo` skip step is
    excluded."""
    job = tf_aws_gpu_job()
    matched = [step for step in steps_running(job, fragment) if not is_printer(step)]

    assert job.get("defaults", {}).get("run", {}).get("working-directory") == (
        "infra/terraform/aws-gpu"
    )
    assert len(matched) == 1, f"{len(matched)} steps run `{fragment}`"
    assert "if" not in matched[0], f"`{fragment}` is conditional"


def test_the_tf_aws_gpu_init_falls_back_to_no_backend_without_a_token() -> None:
    """Criterion 46's `-backend=false`. Catches an init that always contacts Terraform Cloud and
    fails every fork pull request before `validate` runs."""
    job = tf_aws_gpu_job()
    inits = [
        step for step in steps_running(job, "terraform init") if not is_printer(step)
    ]
    assert len(inits) == 1, f"{len(inits)} steps run `terraform init`"

    assert "-backend=false" in str(inits[0]["run"])


def test_the_tf_aws_gpu_plan_runs_only_with_both_credentials() -> None:
    """Criterion 47. Catches `||` in place of `&&`, an unconditional plan that fails on every
    credential-less pull request, and a plan conditioned on the token alone."""
    job = tf_aws_gpu_job()
    planners = [
        step for step in steps_running(job, "terraform plan") if not is_printer(step)
    ]

    assert "TF_API_TOKEN_PRESENT" in job.get("env", {}), (
        "the plan's env flag is never defined"
    )
    assert len(planners) == 1, f"{len(planners)} steps run `terraform plan`"
    assert planners[0].get("if") == PLAN_CONDITION


def test_the_tf_aws_gpu_job_says_so_when_the_plan_is_skipped() -> None:
    """Criterion 47's skip. Catches the silent skip, where a green tick reads as "the plan was
    checked", and a skip conditioned on one credential only."""
    skips = [
        step
        for step in steps(tf_aws_gpu_job())
        if is_printer(step)
        and "TF_API_TOKEN" in str(step.get("if", ""))
        and "AWS_PLAN_ROLE_ARN" in str(step.get("if", ""))
        and step.get("if") != PLAN_CONDITION
    ]

    assert len(skips) == 1, f"{len(skips)} visible skip steps naming both credentials"


def test_infra_check_runs_the_aws_gpu_lines_each_on_its_own() -> None:
    """Criterion 51 and a project rule. Catches an aws-gpu line missing (the local gate never
    reads the module), the three chained onto one line so only the last exit code counts, and
    any terraform line swallowing failure with `||`."""
    lines = recipe(read(MAKEFILE), "infra-check")
    missing = [line for line in AWS_GPU_INFRA_CHECK_LINES if line not in lines]
    swallowed = [line for line in lines if "terraform" in line and "||" in line]

    assert missing == [], f"infra-check lacks {missing}"
    assert swallowed == [], f"a swallowed failure: {swallowed}"


def test_no_file_in_the_module_carries_a_token_shaped_literal() -> None:
    """Criterion 52 and a project rule. Catches an AWS or Terraform Cloud credential pasted into
    `terraform.tfvars.example`, which is why every file is scanned and none exempted. The
    directory is asserted first, because an empty scan of a missing directory passes."""
    assert MODULE.is_dir(), f"{MODULE} does not exist"
    findings: list[str] = []
    for path in sorted(MODULE.rglob("*")):
        if not path.is_file():
            continue
        for name, value in ASSIGNMENT.findall(path.read_text(encoding="utf-8")):
            if SECRETISH_NAME.fullmatch(name) and TOKEN_LITERAL.fullmatch(value):
                findings.append(f"{path.name}: {name}")

    assert findings == [], f"token-shaped literals: {findings}"
