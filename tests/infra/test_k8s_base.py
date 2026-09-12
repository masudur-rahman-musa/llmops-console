r"""`infra/k8s`: the resource envelope an earlier change agrees, read off what kustomize actually emits.

Decision D of an earlier change chose "build and parse" over reading the raw files under `base/`: a
`resources:` entry that is mis-listed, or a `base/` the root never references, produces a
perfectly valid set of files and an output stream missing the object. So the one session-scoped
fixture here runs `kubectl kustomize <repo>/infra/k8s` once (a project rule - built once, no
per-test rebuild) and every case reads the parsed stream. `kubectl` v1.32.2 with embedded
kustomize v5.5.0 is what this is probed against. That fixture now lives in
`tests/infra/conftest.py`, shared with `test_k8s_platform.py`; it is
requested by name here, and no assertion in this file changed when it moved.

The fixture skips ONLY when `kubectl` is absent from `PATH`. A non-zero exit is a failure
carrying kubectl's stderr, never a skip: a broken kustomization that skipped itself green would
be exactly the `A && B || C` shape a project rule exists to refuse. The skip is survivable
because CI's `k8s` job runs the same build for real on every pull request, which is the half
this file's last two cases pin.

Two constants below are quantities, not numbers: `2Gi` and `800m` are asserted by string
equality against the built objects, because the API server compares quantities and a test that
parsed first would accept `2048Mi` for a table that says `2Gi`. The totals case is the one
place parsing happens, and it DERIVES its sums from those same built objects rather than
re-stating the per-namespace literals. CPU totals are summed in whole
millicores rather than as floats: `0.8 + 1.5 + 0.3 + 0.4` is `2.9999999999999996` in binary
floating point, so a float comparison against `3.0` would be red on a correct manifest.

Paths come from this file's own `__file__`, never the working directory. Every read goes through
`read()` so a file that does not exist yet fails the one case needing it.

PyYAML quirk handled in `trigger()`: the bare key `on` is YAML 1.1's boolean, so
`yaml.safe_load` returns the trigger under the key `True` and `document["on"]` would raise
`KeyError` on a perfectly correct workflow - the same reader as `test_hetzner_module.py` and
`tests/ci/test_eval_gate_workflow.py`, copied rather than imported.

How decision E is resolved, which is the shape the last two cases pin. The Makefile grows a
second target, `k8s-check`, holding the one `kustomize build infra/k8s | kubeconform ...`
invocation in the repo; `infra-check` keeps its `if [ -f ... ]` guard exactly where it is and
delegates to `$(MAKE) k8s-check` from inside it; the `k8s` job runs `make k8s-check`. All three
halves of that are forced. The job cannot run `make infra-check`, because that pulls in the
terraform steps decision E split the job to avoid. The guard cannot move into `k8s-check`,
because `tests/infra/test_hetzner_module.py`'s
`test_infra_check_runs_terraform_for_real_and_skips_kustomize_when_absent` asserts the guard
line inside `infra-check`'s recipe, and §8 says no existing test's expectation changes. And the
pipeline cannot be repeated in the CI step, because two copies drift, which is the thing
decision E exists to prevent.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
K8S_ROOT = REPO_ROOT / "infra" / "k8s"
ROOT_KUSTOMIZATION = K8S_ROOT / "kustomization.yaml"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
MAKEFILE = REPO_ROOT / "Makefile"

#: Criterion 2, in the order §1 names them. Asserted by set EQUALITY, so a fifth namespace
#: and a missing one both redden.
NAMESPACES = ("platform", "apps", "evals", "monitoring")

#: The root kustomization's labels, set once with `includeSelectors: false`. Asserted by
#: equality on each Namespace's label mapping: a superset check would not see one dropped.
COMMON_LABELS = {
    "app.kubernetes.io/part-of": "ai-operations-console",
    "app.kubernetes.io/managed-by": "kustomize",
}

#: §1's quota table, decisions B and C, by value. `limits.cpu` is absent
#: from every row ON PURPOSE - see `FORBIDDEN_QUOTA_KEYS`.
QUOTAS: dict[str, dict[str, str]] = {
    "platform": {"requests.memory": "2Gi", "requests.cpu": "800m", "limits.memory": "3Gi"},
    "apps": {"requests.memory": "3Gi", "requests.cpu": "1500m", "limits.memory": "7Gi"},
    "evals": {"requests.memory": "512Mi", "requests.cpu": "300m", "limits.memory": "1Gi"},
    "monitoring": {"requests.memory": "512Mi", "requests.cpu": "400m", "limits.memory": "1Gi"},
}

#: Decision C: a CPU limit throttles rather than kills, so forcing one onto every container
#: would cost the gateway tail latency for no safety gained.
FORBIDDEN_QUOTA_KEYS = ("limits.cpu",)

#: Criterion 5, both resources, both directions of the LimitRange.
LIMIT_RANGE_DEFAULT_REQUEST = {"memory": "64Mi", "cpu": "50m"}
LIMIT_RANGE_DEFAULT = {"memory": "256Mi", "cpu": "500m"}

#: The LimitRange pins exactly one limit type; `Pod` or `PersistentVolumeClaim` entries would
#: change what admission rejects.
LIMIT_RANGE_TYPE = "Container"

#: Criterion 4's totals. Derived in the test from the four built quotas, stated here once as
#: the target: `6Gi` of requests inside the CAX21's 8 GiB, and 3 of its 4 vCPU.
EXPECTED_MEMORY_REQUEST_BYTES = 6 * 1024**3
EXPECTED_CPU_REQUEST_MILLICORES = 3000

#: Criterion 1's other half: a document is legal without `metadata.namespace` only when its
#: kind is cluster-scoped. Anything else with no namespace is the unhappy path.
CLUSTER_SCOPED_KINDS = frozenset(
    {
        "APIService",
        "ClusterIssuer",
        "ClusterRole",
        "ClusterRoleBinding",
        "CustomResourceDefinition",
        "IngressClass",
        "MutatingWebhookConfiguration",
        "Namespace",
        "PersistentVolume",
        "PriorityClass",
        "StorageClass",
        "ValidatingWebhookConfiguration",
    }
)

#: §5's kubeconform invocation, set now so 20b inherits a working one instead of discovering
#: that cert-manager's and Traefik's CRDs have no stock schema.
KUBECONFORM_FLAGS = ("-strict", "-summary", "-schema-location")

#: The flag that turns §5's risk into a silent green: unknown kinds are skipped, and 20b's
#: `ClusterIssuer` would be validated by nothing at all.
FORBIDDEN_KUBECONFORM_FLAG = "-ignore-missing-schemas"

#: The one target CI and the Makefile share (decision E), and the only place the kustomize
#: pipeline is written. The `k8s` job runs THIS, not `infra-check`, so it needs no Terraform.
K8S_CHECK_TARGET = "k8s-check"
SHARED_INFRA_COMMAND = "make k8s-check"

#: The two tools the `k8s` job installs, pinned by value (a project rule and 37 - neither
#: version was named by the plan, so the implementation invented them). `KUSTOMIZE_VERSION`
#: is the kustomize embedded in the kubectl the fixture above shells out to, so CI and a
#: developer build the same stream; an unpinned kubeconform would change what the gate
#: accepts underneath a diff that never touched it.
CI_TOOL_VERSIONS = {"KUSTOMIZE_VERSION": "5.5.0", "KUBECONFORM_VERSION": "0.6.7"}

#: The skip-when-absent guard `test_hetzner_module.py` also pins, inside `infra-check`'s own
#: recipe. It stays where it is; what changes is that the file it tests for now exists, so the
#: guarded branch delegates to `k8s-check` for real instead of echoing a skip.
KUSTOMIZE_GUARD = "if [ -f infra/k8s/kustomization.yaml ]"

#: How the guarded branch reaches the new target. `$(MAKE)` rather than a bare `make` so a
#: sub-make inherits the flags; matched loosely on `MAKE) k8s-check` for that reason.
GUARD_DELEGATION = f"MAKE) {K8S_CHECK_TARGET}"

#: Binary and decimal suffixes a `resources.requests.memory` may carry.
MEMORY_SUFFIXES = {
    "Ki": 1024,
    "Mi": 1024**2,
    "Gi": 1024**3,
    "Ti": 1024**4,
    "k": 1000,
    "M": 1000**2,
    "G": 1000**3,
    "T": 1000**4,
}

#: A step's `if:` is "conditional on a secret" when it reads the `secrets` context or a
#: variable lifted from it, the way the `tf` job's `TF_API_TOKEN_PRESENT` does.
SECRET_CONDITION = re.compile(r"secrets\.|TOKEN|SECRET|API_KEY", re.IGNORECASE)


def read(path: Path) -> str:
    """Read a file the plan owes, failing the one case that needs it when it is absent."""
    assert path.is_file(), f"{path} does not exist"
    return path.read_text(encoding="utf-8")


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


def memory_bytes(quantity: str) -> int:
    """`512Mi` as an integer count of bytes. Raises on a quantity this repo does not write."""
    for suffix, multiplier in MEMORY_SUFFIXES.items():
        if quantity.endswith(suffix):
            return int(quantity[: -len(suffix)]) * multiplier
    return int(quantity)


def cpu_millicores(quantity: str) -> int:
    """`800m` as 800, `2` as 2000. Whole millicores, never floats - see the module docstring."""
    if quantity.endswith("m"):
        return int(quantity[:-1])
    return round(float(quantity) * 1000)


def objects_of_kind(documents: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    """Every built document of `kind`, in build order."""
    return [document for document in documents if document.get("kind") == kind]


def name_of(document: dict[str, Any]) -> str:
    """`metadata.name`, as a string."""
    metadata = document.get("metadata", {})
    assert isinstance(metadata, dict)
    return str(metadata.get("name", ""))


def namespace_of(document: dict[str, Any]) -> str | None:
    """`metadata.namespace`, or None when the document declares none."""
    metadata = document.get("metadata", {})
    assert isinstance(metadata, dict)
    value = metadata.get("namespace")
    return None if value is None else str(value)


def one_per_namespace(documents: list[dict[str, Any]], kind: str) -> dict[str, dict[str, Any]]:
    """The single object of `kind` in each namespace, asserting there is exactly one each."""
    found: dict[str, list[dict[str, Any]]] = {namespace: [] for namespace in NAMESPACES}
    for document in objects_of_kind(documents, kind):
        namespace = namespace_of(document)
        assert namespace in found, f"{kind}/{name_of(document)} is in namespace {namespace!r}"
        found[str(namespace)].append(document)
    counts = {namespace: len(objects) for namespace, objects in found.items()}
    assert all(count == 1 for count in counts.values()), f"{kind} per namespace: {counts}"
    return {namespace: objects[0] for namespace, objects in found.items()}


def quota_hard(quota: dict[str, Any]) -> dict[str, str]:
    """A ResourceQuota's `spec.hard` mapping, values as written."""
    spec = quota.get("spec", {})
    assert isinstance(spec, dict), f"{name_of(quota)} has no spec"
    hard = spec.get("hard", {})
    assert isinstance(hard, dict), f"{name_of(quota)} has no spec.hard"
    return {str(key): str(value) for key, value in hard.items()}


def test_the_build_succeeds_and_emits_only_the_four_namespaces(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 1 and the unhappy path. Catches a fifth `Namespace` (a `staging` copied in
    with a directory) and a missing one, because the assertion is EQUALITY on the name set;
    and catches a namespaced object that names a namespace outside the four, or declares none
    at all while its kind is namespaced - a Deployment with no `metadata.namespace` lands in
    `default`, outside every quota this plan writes, which is the one way "requests and limits
    on everything" quietly stops being true."""
    names = {name_of(document) for document in objects_of_kind(built, "Namespace")}
    assert names == set(NAMESPACES), f"the build emits namespaces {sorted(names)}"

    stray = [
        f"{document.get('kind')}/{name_of(document)} -> {namespace_of(document)!r}"
        for document in built
        if str(document.get("kind")) not in CLUSTER_SCOPED_KINDS
        and namespace_of(document) not in NAMESPACES
    ]
    assert stray == [], f"objects outside the four namespaces: {stray}"


def test_every_namespace_carries_the_common_labels(built: list[dict[str, Any]]) -> None:
    """Criterion 2's second half. Catches one of the two labels dropped from the root
    kustomization's `labels:` block, and catches the block being moved into `base/` where a
    later directory (20b's `platform/`) would not inherit it. Equality on each Namespace's
    label mapping, not containment: a superset assertion passes with a label missing as long
    as something else is there."""
    carried = {
        name_of(document): document.get("metadata", {}).get("labels")
        for document in objects_of_kind(built, "Namespace")
    }
    assert carried == {name: COMMON_LABELS for name in NAMESPACES}


@pytest.mark.parametrize(("namespace", "expected"), sorted(QUOTAS.items()))
def test_each_namespace_has_one_resource_quota_with_the_agreed_values(
    namespace: str, expected: dict[str, str], built: list[dict[str, Any]]
) -> None:
    """Criterion 3, decisions B and C, one row per namespace. Catches a quantity written as
    the wrong value (`4Gi` for `apps`, which the totals case then catches a second way);
    catches `limits.memory` dropped from a row, which is what makes a container with no memory
    limit schedulable again and retires half of #55's claim; and catches `limits.cpu` creeping
    in, which decision C refuses because a CPU limit throttles rather than kills. Equality on
    the whole `spec.hard` key set, so an extra key is a failure and not a silent addition."""
    quota = one_per_namespace(built, "ResourceQuota")[namespace]
    hard = quota_hard(quota)

    assert hard == expected, f"{namespace} quota is {hard}"
    for key in FORBIDDEN_QUOTA_KEYS:
        assert key not in hard, f"{namespace} quota names {key} (decision C)"


def test_the_memory_requests_total_six_gibibytes_and_the_cpu_requests_total_three(
    built: list[dict[str, Any]],
) -> None:
    """Criterion 4, derived and never re-stated: the four quantities are read
    off the BUILT quotas and parsed, so the only literals here are the totals themselves.
    Catches a row raised in isolation - `apps` going `3Gi` to `4Gi` because one workload did
    not fit - which leaves each per-namespace assertion arguable and blows the 6Gi envelope the
    8 GiB box has to hold with k3s still running. Millicores, not floats: `0.8 + 1.5 + 0.3 +
    0.4` is `2.9999999999999996`."""
    quotas = one_per_namespace(built, "ResourceQuota")
    memory = [quota_hard(quota)["requests.memory"] for quota in quotas.values()]
    cpu = [quota_hard(quota)["requests.cpu"] for quota in quotas.values()]

    assert len(memory) == len(NAMESPACES) and len(cpu) == len(NAMESPACES)
    assert sum(memory_bytes(quantity) for quantity in memory) == EXPECTED_MEMORY_REQUEST_BYTES
    assert sum(cpu_millicores(quantity) for quantity in cpu) == EXPECTED_CPU_REQUEST_MILLICORES


@pytest.mark.parametrize("namespace", NAMESPACES)
def test_each_namespace_has_one_limit_range_with_the_agreed_defaults(
    namespace: str, built: list[dict[str, Any]]
) -> None:
    """Criterion 5. The LimitRange is the backstop under the quota: without it a third-party
    manifest that declares no resources is REJECTED at admission rather than defaulted, so an
    unedited upstream chart stops being installable at all. Neither of this repo's two
    candidates is an example of that any more: 20b's decision A installs cert-manager with all
    three quantities set on the command line, and an earlier change's metrics stack is hand-written
    rather than the upstream chart, with all three declared on every container.
    Catches the `Container` entry retyped as `Pod` (which defaults nothing per container),
    catches `default` and `defaultRequest` swapped - a 256Mi request against a 64Mi ceiling is
    unschedulable - and catches either resource dropped from either side."""
    limit_range = one_per_namespace(built, "LimitRange")[namespace]
    spec = limit_range.get("spec", {})
    assert isinstance(spec, dict), f"{namespace} LimitRange has no spec"
    entries = spec.get("limits", [])
    assert isinstance(entries, list) and len(entries) == 1, f"{namespace} limits: {entries}"
    entry = entries[0]
    assert isinstance(entry, dict)

    assert entry.get("type") == LIMIT_RANGE_TYPE, f"{namespace} limit type {entry.get('type')}"
    assert entry.get("defaultRequest") == LIMIT_RANGE_DEFAULT_REQUEST
    assert entry.get("default") == LIMIT_RANGE_DEFAULT


def test_infra_check_runs_the_kustomize_half_for_real_now_that_the_file_exists() -> None:
    """Criterion 6's Makefile half, across BOTH recipes, plus §5's flags. The conditional stays
    inside `infra-check` - it is a skip-when-absent, and deleting it would turn a missing file
    into a target failure rather than a skip - so what this pins is that the guard is now
    SATISFIED and DELEGATES: `kustomization.yaml` exists and the guarded branch reaches
    `k8s-check`. Catches the plan shipping its CI job while `infra/k8s` still holds only
    `.gitkeep`, leaving both gates green having validated nothing; catches the split being made
    with the guard left pointing at a target that no longer exists or at nothing at all, which
    `make` reports as success from a `@if` that ran no command; catches `k8s-check` missing
    (`recipe()` fails on it by name); catches the kubeconform invocation left bare, which §5
    says fails the moment 20b adds a `ClusterIssuer` kubeconform has no schema for; and catches
    the repair being `-ignore-missing-schemas`, which makes that failure a silent pass
    forever."""
    makefile = read(MAKEFILE)
    guarded = recipe(makefile, "infra-check")

    guards = [line for line in guarded if KUSTOMIZE_GUARD in line]
    assert len(guards) == 1, f"`{KUSTOMIZE_GUARD}` appears {len(guards)} times in infra-check"
    assert GUARD_DELEGATION in guards[0], f"the guard reaches no `{K8S_CHECK_TARGET}`: {guards[0]}"
    assert ROOT_KUSTOMIZATION.is_file(), f"{ROOT_KUSTOMIZATION} does not exist, so the half skips"

    # Selected by fragment, with the `echo` printer of the skip branch excluded: it repeats
    # the words "kustomize build" in its message, and counting it as the real invocation
    # would pass this case against a Makefile that only ever prints them.
    invocations = [
        line
        for line in recipe(makefile, K8S_CHECK_TARGET)
        if "kustomize build infra/k8s" in line and "echo" not in line
    ]
    assert len(invocations) == 1, f"{len(invocations)} real kustomize invocations: {invocations}"
    invocation = invocations[0]

    assert "kubeconform" in invocation, f"nothing validates the build: {invocation}"
    missing = [flag for flag in KUBECONFORM_FLAGS if flag not in invocation]
    assert missing == [], f"kubeconform runs without {missing}: {invocation}"
    assert FORBIDDEN_KUBECONFORM_FLAG not in invocation, (
        f"{FORBIDDEN_KUBECONFORM_FLAG} turns an unknown CRD into a green tick: {invocation}"
    )


def test_the_k8s_job_builds_and_validates_on_every_pull_request() -> None:
    """Criterion 6's CI half and decision E, pinned as `make k8s-check` (see the module
    docstring for why that target and not `infra-check`). Catches the job being given an `if:`
    on a secret, the way the `tf` job's `plan` step legitimately is - kustomize and kubeconform
    need no credential, so a gated `k8s` job would be inert on exactly the fork pull requests it
    exists to check, and green because it never ran. Catches the job being dropped or renamed,
    catches the workflow's `pull_request` trigger going away so the gate only runs post-merge,
    catches the step re-spelling the pipeline inline instead of calling the shared target, which
    is the drift decision E refuses, and catches it calling `make infra-check` instead, which
    drags the terraform steps - and a Terraform setup this job does not install - into a
    kustomize check."""
    document = load_yaml(WORKFLOW)
    jobs = document["jobs"]
    assert isinstance(jobs, dict)

    assert "pull_request" in trigger(document), "ci.yml does not run on pull requests"
    assert "k8s" in jobs, f"ci.yml declares no `k8s` job, only {sorted(jobs)}"
    job = jobs["k8s"]
    assert isinstance(job, dict)

    assert not SECRET_CONDITION.search(str(job.get("if", ""))), f"the k8s job is gated: {job['if']}"

    running = steps_running(job, SHARED_INFRA_COMMAND)
    assert len(running) == 1, f"{len(running)} steps run `{SHARED_INFRA_COMMAND}` in the k8s job"
    condition = str(running[0].get("if", ""))
    assert not SECRET_CONDITION.search(condition), (
        f"the build step is gated on a secret: {condition}"
    )


def test_the_k8s_job_pins_both_tool_versions() -> None:
    """a project rule: two constants the plan never named, so no case above reaches them.
    Catches either version becoming `latest` or a floating tag, which is how a gate moves
    under a diff that did not touch it - a newer kustomize emits a different stream, a newer
    kubeconform accepts a different set of manifests, and both would land as an unexplained
    red on whatever pull request happened to be open. Read from the job's `env` block rather
    than from the install script, because that is where the value a reader would change is."""
    job = load_yaml(WORKFLOW)["jobs"]["k8s"]
    assert isinstance(job, dict)
    environment = job.get("env", {})
    assert isinstance(environment, dict), "the k8s job pins no tool versions at all"

    pinned = {name: str(environment.get(name, "")) for name in CI_TOOL_VERSIONS}
    assert pinned == CI_TOOL_VERSIONS, f"the k8s job pins {pinned}"

    # The install step has to READ those names, or pinning them in `env` changes nothing.
    installs = [step for step in steps(job) if "curl" in str(step.get("run", ""))]
    assert len(installs) == 1, f"{len(installs)} install steps in the k8s job"
    script = str(installs[0]["run"])
    unused = [name for name in CI_TOOL_VERSIONS if f"${{{name}}}" not in script]
    assert unused == [], f"the install script ignores {unused}"
