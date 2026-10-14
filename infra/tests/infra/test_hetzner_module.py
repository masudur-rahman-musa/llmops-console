r"""`infra/terraform/hetzner`: the security claims `terraform validate` would accept either way.

Decision C of an earlier change keeps this file on plain text rather than an HCL parser, so what is
asserted here is deliberately narrow: `terraform fmt -check`, `init` and `validate` are the
semantic checks and they run in the `tf` job, while the claims that a syntactically perfect
module can still break - the firewall's exact rule set, the pinned k3s version, the absent
`admin_ips` default, an output that leaks a token - have no other guard anywhere in the repo.
Terraform is not installed on the development machine (a project rule forbids installing it),
which is the second reason these live as file assertions.

Every path is taken from this file's own `__file__`, never from the working directory: a test
that finds the module only when pytest runs from the repo root passes in CI and misleads
anyone running it from `tests/`. Every read goes through `read()`, so a file that does not
exist yet fails the one case that needs it instead of collapsing the module.

Two notes on the helpers. The block readers below balance braces rather than matching
`\{[^}]*\}`, because a `rule {` block and a `variable {` block both nest. And the token scan
anchors its identifier check with `re.fullmatch`: `$` in Python also matches
before a trailing newline, so `^...$` with `.match` would accept `token\n` as a clean name.

PyYAML quirk handled in `trigger()`: the bare key `on` is YAML 1.1's boolean, so
`yaml.safe_load` returns the trigger under the key `True` and `doc["on"]` would raise
`KeyError` on a perfectly correct workflow - the same reader as
`infra/tests/ci/test_eval_gate_workflow.py`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE = REPO_ROOT / "infra" / "terraform" / "hetzner"
MAIN_TF = MODULE / "main.tf"
VARIABLES_TF = MODULE / "variables.tf"
VERSIONS_TF = MODULE / "versions.tf"
OUTPUTS_TF = MODULE / "outputs.tf"
CLOUD_INIT = MODULE / "cloud-init.yaml"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
MAKEFILE = REPO_ROOT / "Makefile"

#: The exact k3s release the server installs. a project rule: a pinned version is a
#: constant, so it is asserted by value and not by "some version string is present".
K3S_VERSION = "v1.31.5+k3s1"

#: Every public-facing rule carries both families; IPv6-only omission is the quiet half.
WORLD = ("0.0.0.0/0", "::/0")

#: Section 1's validation criterion, in the order the plan names them.
VALIDATED_VARIABLES = (
    "admin_ips",
    "zone",
    "subdomains",
    "server_type",
    "location",
    "ssh_public_key",
)

#: Section 1's output list: two server facts and the four FQDNs.
EXPECTED_OUTPUTS = {
    "server_ipv4",
    "server_id",
    "api_fqdn",
    "console_fqdn",
    "helpdesk_fqdn",
    "grafana_fqdn",
}

#: Never an output, never a variable default: a project rule.
FORBIDDEN_IN_OUTPUTS = ("token", "private_key", "kubeconfig")

SECRETISH_NAME = re.compile(
    r"[A-Za-z0-9_.-]*(token|secret|password|api_key)[A-Za-z0-9_.-]*", re.IGNORECASE
)
TOKEN_LITERAL = re.compile(r"[A-Za-z0-9+/=_-]{20,}")
ASSIGNMENT = re.compile(r"""([A-Za-z0-9_.-]+)\s*[:=]\s*["']([^"'\n]*)["']""")


def read(path: Path) -> str:
    """Read a file that the module owes, failing this case alone when it is absent."""
    assert path.is_file(), f"{path} does not exist"
    return path.read_text(encoding="utf-8")


def body_after(text: str, opening: int) -> str:
    """The brace-balanced body that starts at the `{` at or after `opening`."""
    start = text.index("{", opening)
    depth = 0
    for position in range(start, len(text)):
        character = text[position]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : position]
    raise AssertionError("unbalanced braces")


def named_blocks(text: str, keyword: str) -> dict[str, str]:
    """Every `<keyword> "<name>" ... {` body in `text`, keyed by the FIRST quoted label.

    `resource "hcloud_firewall" "edge"` keys on the type, `variable "zone"` on the name.
    """
    found: dict[str, str] = {}
    for match in re.finditer(rf'\b{keyword}\s+"([^"]+)"[^{{\n]*\{{', text):
        found[match.group(1)] = body_after(text, match.end() - 1)
    return found


def inner_blocks(text: str, keyword: str) -> list[str]:
    """Every bare `<keyword> {` body in `text`, in file order."""
    return [
        body_after(text, match.end() - 1)
        for match in re.finditer(rf"\b{keyword}\s*\{{", text)
    ]


def provider_block(text: str, name: str) -> str:
    """The body of `<name> = { ... }` inside `required_providers`, or the empty string."""
    match = re.search(rf"(?m)^\s*{name}\s*=\s*\{{", text)
    return "" if match is None else body_after(text, match.end() - 1)


def argument(body: str, name: str) -> str:
    """The right-hand side of `name = ...`, as written, or the empty string when absent."""
    match = re.search(rf"(?m)^\s*{name}\s*=\s*(.+?)\s*$", body)
    return "" if match is None else match.group(1)


def load_yaml(path: Path) -> dict[str, Any]:
    """Parse a YAML document that must be a mapping."""
    document = yaml.safe_load(read(path))
    assert isinstance(document, dict), f"{path} is not a YAML mapping"
    return document


def trigger(document: dict[str, Any]) -> dict[str, Any]:
    """The `on:` block, found under either the string key or PyYAML's boolean `True`."""
    for key in (True, "on"):
        if key in document:
            block = document[key]
            assert isinstance(block, dict), "the trigger block is not a mapping"
            return block
    raise AssertionError("the workflow declares no trigger")


def steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    """The job's steps, each a mapping."""
    listed = job["steps"]
    assert isinstance(listed, list)
    return [step for step in listed if isinstance(step, dict)]


def steps_running(job: dict[str, Any], fragment: str) -> list[dict[str, Any]]:
    """The job's steps whose `run` script contains `fragment`."""
    return [step for step in steps(job) if fragment in str(step.get("run", ""))]


def recipe(makefile: str, target: str) -> list[str]:
    """The tab-indented recipe lines of a Makefile target, comments and blanks dropped."""
    lines = makefile.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(f"{target}:"):
            collected: list[str] = []
            for following in lines[index + 1 :]:
                if not following.startswith("\t"):
                    break
                stripped = following.strip()
                if stripped and not stripped.startswith("#"):
                    collected.append(stripped)
            return collected
    raise AssertionError(f"the Makefile declares no `{target}` target")


def test_the_firewall_opens_ssh_to_the_admins_and_nothing_else_to_them() -> None:
    """Criterion 2, the whole reason this file exists. Catches a fourth `rule` block (the
    k3s API on 6443, a debug port) added to a firewall whose default-deny nobody re-reads;
    catches port 22 taking `["0.0.0.0/0", "::/0"]` like its neighbours, which is the exact
    mistake decision F's missing default is meant to make impossible; and catches 80 or 443
    carrying only the IPv4 family, which leaves a dual-stacked box unreachable over IPv6.
    Nothing else in the repo reads this file - `terraform validate` accepts any rule set."""
    firewalls = named_blocks(read(MAIN_TF), "resource")
    assert "hcloud_firewall" in firewalls, "main.tf declares no hcloud_firewall"
    rules = inner_blocks(firewalls["hcloud_firewall"], "rule")

    assert len(rules) == 3, f"the firewall carries {len(rules)} rule blocks, not 3"

    by_port = {argument(rule, "port").strip('"'): rule for rule in rules}
    assert set(by_port) == {"22", "80", "443"}, f"ports {sorted(by_port)}"
    assert "var.admin_ips" in argument(by_port["22"], "source_ips")
    for port in ("80", "443"):
        sources = argument(by_port[port], "source_ips")
        assert all(network in sources for network in WORLD), (
            f"port {port} sources: {sources}"
        )
        assert "admin_ips" not in sources, (
            f"port {port} is restricted to the admin CIDRs"
        )


def test_ci_reaches_ssh_through_a_declared_list_that_defaults_to_empty() -> None:
    """An earlier change criterion 5 and decision A: CD cannot reach port 22 while it is `admin_ips`
    alone, so `var.ci_ips` is concatenated into that one rule. Catches the variable declared
    with NO default, which makes every `plan` and every `apply` fail until someone invents a
    value for a list that is meant to be empty until the owner sets it - and catches the
    default written as anything but the empty list, which is the whole point of the decision:
    a default of `["0.0.0.0/0"]` would open SSH to the internet the moment the variable is
    forgotten, exactly the failure `admin_ips` has no default to avoid. The rule COUNT is
    re-asserted here as well as in the port-22 case above, because the tempting alternative
    implementation is a FOURTH rule opening 22 to the CI list separately, which satisfies
    "22 still names admin_ips" and quietly widens the firewall; and the port-22 source is
    re-read so the concat cannot land on 80 or 443 instead. Read as text, as everything in
    this module is: `terraform validate` accepts every one of those shapes."""
    variables = named_blocks(read(VARIABLES_TF), "variable")
    firewalls = named_blocks(read(MAIN_TF), "resource")
    assert "ci_ips" in variables, (
        f"variables.tf declares no ci_ips, only {sorted(variables)}"
    )
    assert "hcloud_firewall" in firewalls, "main.tf declares no hcloud_firewall"
    rules = inner_blocks(firewalls["hcloud_firewall"], "rule")
    by_port = {argument(rule, "port").strip('"'): rule for rule in rules}

    assert argument(variables["ci_ips"], "default") == "[]", (
        "ci_ips does not default to empty"
    )
    assert len(rules) == 3, f"the firewall carries {len(rules)} rule blocks, not 3"
    assert "var.ci_ips" in argument(by_port["22"], "source_ips")
    assert "var.admin_ips" in argument(by_port["22"], "source_ips")


def test_the_server_is_the_arm_box_in_the_agreed_region() -> None:
    """Criterion 1's hardware, pinned by value. Catches a `server_type`
    default of `cx22`, which is x86 and would run an arm64 image nowhere, and catches a
    `location` outside the EU - `fsn1` is still in region, `ash` and `hil` are the US and
    silently move the node out of the residency boundary a project rule is written around.
    Read from `variables.tf`'s own block, so a default moved into a `.tfvars` is a failure."""
    variables = named_blocks(read(VARIABLES_TF), "variable")

    assert argument(variables["server_type"], "default") == '"cax21"'
    assert argument(variables["location"], "default") == '"nbg1"'


def test_the_cloud_init_pins_k3s_once_and_disables_nothing() -> None:
    """Criterion 3. Catches `INSTALL_K3S_VERSION` set to an empty string or omitted, where the
    installer takes `latest` and two applies a month apart build different clusters; catches
    the version being pasted a second time (a comment, a second export) where one copy is
    then bumped and the other is not; and catches `--disable traefik` or
    `--disable servicelb`, which would leave every Ingress unrouted with no error
    from this module at all. Counted over the raw text, so a commented-out copy still counts."""
    text = read(CLOUD_INIT)

    assert text.count(K3S_VERSION) == 1, (
        f"{K3S_VERSION} occurs {text.count(K3S_VERSION)} times"
    )
    assert text.count("latest") == 0, "cloud-init.yaml names `latest`"
    assert text.count("--disable") == 0, (
        "cloud-init.yaml disables a bundled k3s component"
    )


def test_the_terraform_block_pins_the_core_and_both_providers() -> None:
    """Criterion 4. Catches a missing `cloud {` block, where state falls back to a local
    `terraform.tfstate` that CI throws away and an `apply` then rebuilds the server; catches
    an unconstrained provider (`version` dropped), where a 2.x hcloud release lands on the
    next `init` with renamed arguments; and catches the two constraints being swapped between
    providers, which the paired reads below make visible where two separate greps would not."""
    text = read(VERSIONS_TF)
    required = {name: provider_block(text, name) for name in ("hcloud", "cloudflare")}

    assert re.search(r'required_version\s*=\s*">= 1\.9"', text) is not None
    assert re.search(r"(?m)^\s*cloud\s*\{", text) is not None, (
        "no Terraform Cloud block"
    )
    assert '"hetznercloud/hcloud"' in required["hcloud"]
    assert '"~> 1.49"' in required["hcloud"]
    assert '"cloudflare/cloudflare"' in required["cloudflare"]
    assert '"~> 4.40"' in required["cloudflare"]


def test_every_input_is_validated_and_the_admin_cidrs_have_no_default() -> None:
    """Criteria 6 and 7, and decision F. Catches a variable declared with a description and no
    `validation` block, where a typo'd region or a bare IP reaches the API instead of failing
    at plan time; and catches `admin_ips` acquiring a default, which is the failure mode the
    decision is named for - an unset variable would then open 22 to the internet quietly
    rather than refusing to plan. Parsed per variable, so a `validation` block belonging to
    one variable cannot stand in for a neighbour that has none."""
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

    # Criterion 7 names the two cases by hand, so the conditions are pinned by what
    # they call and not only by their presence: `length(...) > 0` is the empty list,
    # `cidrhost` is what refuses a bare address. A validation block that checked
    # neither would satisfy the loop above.
    conditions = " ".join(
        argument(block, "condition")
        for block in inner_blocks(variables["admin_ips"], "validation")
    )
    assert "length(var.admin_ips) > 0" in conditions, "an empty admin_ips is accepted"
    assert "cidrhost(" in conditions, "admin_ips entries are not checked as CIDRs"


def test_the_outputs_are_the_six_agreed_facts_and_no_credential() -> None:
    """Criterion 5. Catches a seventh output added for convenience - a `kubeconfig`, the
    hcloud token, the generated private key - which Terraform Cloud then stores in state and
    prints in every run's log; and catches one of the four FQDNs being dropped, which is what
    the Ingresses read to point their hosts once the app manifests land. Equality on the name
    set is what
    makes both directions fail."""
    text = read(OUTPUTS_TF)
    declared = set(named_blocks(text, "output"))
    leaked = [word for word in FORBIDDEN_IN_OUTPUTS if word in text.lower()]

    assert declared == EXPECTED_OUTPUTS
    assert leaked == [], f"outputs.tf names {leaked}"


def test_no_file_in_the_module_carries_a_token_shaped_literal() -> None:
    r"""Criterion 10 and a project rule. Catches a real credential pasted into
    `terraform.tfvars.example` while "filling it in to test the plan", which is why the
    example file is scanned like every other file and not exempted. A finding is a
    `(token|secret|password|api_key)`-ish name assigned a quoted run of 20+ characters from
    `[A-Za-z0-9+/=_-]` - long enough that `hcloud_token = "changeme"` and a `${var.x}`
    interpolation both stay clean, short enough that every real hcloud and Cloudflare token
    matches. The identifier half uses `re.fullmatch`, never `^...$` with `.match`: `$` also
    matches before a trailing newline. The directory's existence is
    asserted first, because an empty scan of a missing directory would pass vacuously."""
    assert MODULE.is_dir(), f"{MODULE} does not exist"
    findings: list[str] = []
    for path in sorted(MODULE.rglob("*")):
        if not path.is_file():
            continue
        for name, value in ASSIGNMENT.findall(path.read_text(encoding="utf-8")):
            if SECRETISH_NAME.fullmatch(name) and TOKEN_LITERAL.fullmatch(value):
                findings.append(f"{path.name}: {name}")

    assert findings == [], f"token-shaped literals: {findings}"


def test_the_ci_job_checks_terraform_always_and_plans_only_with_credentials() -> None:
    """Criterion 8 and decision B. Catches an `if` copied from the plan step onto `fmt`, `init`
    or `validate`, which would make the whole terraform gate inert on exactly the fork pull
    requests it exists to check - green because it never ran; catches an unconditional `plan`,
    which fails every PR that has no TFC credentials; and catches the silent skip, where a
    contributor reads a green tick as "the plan was checked". `init -backend=false` is what
    lets `validate` run without touching Terraform Cloud at all."""
    jobs = load_yaml(WORKFLOW)["jobs"]
    assert isinstance(jobs, dict)
    assert "pull_request" in trigger(load_yaml(WORKFLOW)), "ci.yml does not run on PRs"
    assert "tf" in jobs, f"ci.yml declares no `tf` job, only {sorted(jobs)}"
    job = jobs["tf"]

    for fragment in ("terraform fmt -check", "terraform init", "terraform validate"):
        matched = steps_running(job, fragment)
        assert len(matched) == 1, f"{len(matched)} steps run `{fragment}`"
        assert "if" not in matched[0], f"`{fragment}` is conditional"
    assert "-backend=false" in str(steps_running(job, "terraform init")[0]["run"])

    # `echo` excluded deliberately: the visible-skip step below says the words
    # "terraform plan" in its message, and counting it as a planner would make this
    # assertion fail on the correct workflow.
    planners = [
        step
        for step in steps_running(job, "terraform plan")
        if "echo" not in str(step.get("run", ""))
    ]
    assert len(planners) == 1
    assert "TF_API_TOKEN" in str(planners[0].get("if", ""))

    skips = [
        step
        for step in steps(job)
        if "TF_API_TOKEN" in str(step.get("if", ""))
        and "echo" in str(step.get("run", ""))
    ]
    assert skips != [], "the skipped plan leaves nothing in the log"


def test_infra_check_runs_terraform_for_real_and_skips_kustomize_when_absent() -> None:
    """Decision H and a project rule. Three terraform lines, each its own line so its exit
    code is the target's, then a guarded kustomize half. That half no longer carries the
    pipeline itself: it delegates to the `k8s-check` target, which CI runs on its own without
    a terraform toolchain, so the invocation exists in one place. The guard stays because it
    is a skip-when-absent, not a skip-on-failure - a checkout without a kustomization skips
    rather than failing the whole target.

    Catches the two obvious repairs: chaining the kustomize half onto the terraform half with
    `||` so the target reports success when TERRAFORM fails (rule 9's `A && B || C`), and
    deleting the guarded half outright, which hides the manifest gate instead of skipping it.
    What `k8s-check` itself runs - the kubeconform flags, the single invocation - is pinned by
    infra/tests/infra/test_k8s_base.py, not here."""
    lines = recipe(read(MAKEFILE), "infra-check")
    terraform_lines = [line for line in lines if "terraform" in line]

    assert terraform_lines != [], "infra-check runs no terraform"
    for line in terraform_lines:
        assert "||" not in line, f"a swallowed failure: {line}"
        assert "kustomize" not in line, f"the two halves share a line: {line}"
    assert any("if [ -f infra/k8s/kustomization.yaml ]" in line for line in lines), (
        "the kustomize half is not guarded by a skip-when-absent conditional"
    )
    assert any("k8s-check" in line for line in lines), "no kustomize half at all"
