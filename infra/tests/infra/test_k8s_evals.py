r"""`infra/k8s/evals/`: the Pushgateway. The nightly-eval CronJob is gone.

The stream comes from the session-scoped `built` fixture in `infra/tests/infra/conftest.py`: one
`kubectl kustomize infra/k8s` for the whole session, shared with every other
`infra/tests/infra/test_k8s_*.py` module. `infra/k8s/evals/` does not exist yet, so the fixture
still builds (the root `resources:` list has not been asked to include it), and every case below
that reads FOR the new objects therefore fails on "found none" rather than on a build error - the
distinction the suite relies on.

The parsing helpers (`read`, `objects_of_kind`, `name_of`, `namespace_of`, `memory_bytes`,
`cpu_millicores`, `quota_hard`, `owner_spec`, `pod_owners`, `pod_spec`, `all_containers`,
`resource_quantities`, `replicas_of`, `MEMORY_SUFFIXES`) are COPIES of
`infra/tests/infra/test_k8s_platform.py`'s, not imports - the convention this whole suite already
follows (`infra/tests/infra/test_k8s_alerting.py`'s own docstring says so explicitly, and
`test_k8s_platform.py`'s says the same of `test_k8s_base.py`'s).

`base/`'s three kinds in this namespace (`Namespace`, `ResourceQuota`, `LimitRange`) are NOT
re-asserted here - `infra/tests/infra/test_k8s_base.py` owns them, including the `evals-envelope`
`ResourceQuota`'s own hard values, which this file only READS for the subtotal case.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
K8S_ROOT = REPO_ROOT / "infra" / "k8s"
EVALS_DIR = K8S_ROOT / "evals"

EVALS = "evals"

#: `base/`'s three kinds, excluded the same way `test_k8s_platform.py`'s `BASE_KINDS` is:
#: this file owns what `evals/` ADDS, not what `base/` already put in the namespace.
BASE_KINDS = frozenset({"Namespace", "ResourceQuota", "LimitRange"})

#: The exact three objects in the `evals`
#: namespace. Set equality, so a manifest left out of `evals/kustomization.yaml`'s
#: `resources:` and an object nobody agreed to are both visible.
EVALS_OBJECTS: frozenset[tuple[str, str]] = frozenset(
    {
        ("Deployment", "pushgateway"),
        ("Service", "pushgateway"),
    }
)

#: The three keys every container in this namespace must declare (mirrors
#: `test_k8s_platform.py`'s `REQUIRED_CONTAINER_KEYS`); no `limits.cpu` value is pinned here
#: because (a CPU limit throttles rather than kills) is `platform/`'s own choice
#: and nothing here restates a value for it either way.
REQUIRED_CONTAINER_KEYS = ("requests.memory", "requests.cpu", "limits.memory")

#: Binary and decimal suffixes a `resources.requests.memory` may carry. Copied from
#: `test_k8s_platform.py`'s `MEMORY_SUFFIXES`.
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

#: Copied from `test_k8s_platform.py`'s credential scan, verbatim.
INLINE_CREDENTIAL = re.compile(
    r"^\s*-?\s*(password|passwd|postgres_password|token|apikey|api_key)\s*:\s*(\S.*)$",
    re.IGNORECASE,
)
CREDENTIAL_ENV_NAME = re.compile(
    r"^\s*-?\s*name\s*:\s*[\"']?(\w*(password|passwd|token|apikey|api_key|secret)\w*)[\"']?\s*$",
    re.IGNORECASE,
)
ENV_VALUE = re.compile(r"^\s*value\s*:\s*(\S.*)$")
ENV_VALUE_LOOKAHEAD = 3


def read(path: Path) -> str:
    """Read a file the tests need, failing the one case that needs it when it is absent."""
    assert path.is_file(), f"{path} does not exist"
    return path.read_text(encoding="utf-8")


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


def memory_bytes(quantity: str) -> int:
    """`512Mi` as an integer count of bytes. Raises on a quantity this repo does not write."""
    for suffix, multiplier in MEMORY_SUFFIXES.items():
        if quantity.endswith(suffix):
            return int(quantity[: -len(suffix)]) * multiplier
    return int(quantity)


def cpu_millicores(quantity: str) -> int:
    """`200m` as 200, `2` as 2000. Whole millicores, never floats."""
    if quantity.endswith("m"):
        return int(quantity[:-1])
    return round(float(quantity) * 1000)


def quota_hard(quota: dict[str, Any]) -> dict[str, str]:
    """A ResourceQuota's `spec.hard` mapping, values as written."""
    spec = quota.get("spec", {})
    assert isinstance(spec, dict), f"{name_of(quota)} has no spec"
    hard = spec.get("hard", {})
    assert isinstance(hard, dict), f"{name_of(quota)} has no spec.hard"
    return {str(key): str(value) for key, value in hard.items()}


def owner_spec(owner: dict[str, Any]) -> dict[str, Any]:
    """The mapping that carries a `template` key: `spec` itself for every kind here except a
    CronJob, whose template sits one level deeper at `spec.jobTemplate.spec`."""
    spec = owner["spec"]
    assert isinstance(spec, dict)
    job_template = spec.get("jobTemplate")
    if isinstance(job_template, dict) and isinstance(job_template.get("spec"), dict):
        return dict(job_template["spec"])
    return spec


def pod_owners(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every built document carrying a pod template, found by the PRESENCE of
    `spec.template.spec` (or, for a CronJob, `spec.jobTemplate.spec.template.spec`)."""
    owners: list[dict[str, Any]] = []
    for document in documents:
        spec = document.get("spec")
        if not isinstance(spec, dict):
            continue
        template = owner_spec(document).get("template")
        if isinstance(template, dict) and isinstance(template.get("spec"), dict):
            owners.append(document)
    return owners


def pod_spec(owner: dict[str, Any]) -> dict[str, Any]:
    """The pod template's `spec` of a document `pod_owners` selected."""
    spec = owner_spec(owner)["template"]["spec"]
    assert isinstance(spec, dict)
    return dict(spec)


def all_containers(owner: dict[str, Any]) -> list[dict[str, Any]]:
    """Every container of a pod template, `initContainers` included."""
    spec = pod_spec(owner)
    collected: list[dict[str, Any]] = []
    for key in ("initContainers", "containers"):
        listed = spec.get(key, [])
        assert isinstance(listed, list), f"{name_of(owner)}.{key} is not a list"
        collected.extend(container for container in listed if isinstance(container, dict))
    return collected


def resource_quantities(container: dict[str, Any]) -> dict[str, str]:
    """A container's `resources` flattened to the `requests.memory`-style keys, values
    exactly as written."""
    resources = container.get("resources", {})
    if not isinstance(resources, dict):
        return {}
    flattened: dict[str, str] = {}
    for side in ("requests", "limits"):
        values = resources.get(side, {})
        if not isinstance(values, dict):
            continue
        for resource, quantity in values.items():
            flattened[f"{side}.{resource}"] = str(quantity)
    return flattened


def replicas_of(owner: dict[str, Any]) -> int:
    """`spec.replicas`, defaulting to Kubernetes' own default of 1."""
    declared = owner["spec"].get("replicas", 1)
    return int(declared)


def evals_owners(built: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every pod-owning object in the `evals` namespace."""
    return [owner for owner in pod_owners(built) if namespace_of(owner) == EVALS]


def evals_object(built: list[dict[str, Any]], kind: str, name: str) -> dict[str, Any]:
    """The one object of `kind` named `name` in `evals`, asserting the build emits exactly
    one - a clear failure while `infra/k8s/evals/` does not exist at all."""
    found = [
        document
        for document in objects_of_kind(built, kind)
        if name_of(document) == name and namespace_of(document) == EVALS
    ]
    emitted = sorted(
        name_of(document)
        for document in objects_of_kind(built, kind)
        if namespace_of(document) == EVALS
    )
    assert len(found) == 1, f"expected one evals/{kind}/{name}; evals has {kind}s {emitted}"
    return found[0]


# --------------------------------------------------------------------------
# the object inventory
# --------------------------------------------------------------------------


def test_the_evals_directory_emits_exactly_the_objects_the_plan_names(
    built: list[dict[str, Any]],
) -> None:
    """Catches `infra/k8s/evals/` never wired into the root
    `kustomization.yaml`'s `resources:` list - the manifests can be perfectly valid and
    simply absent from the built stream, which set equality over `(kind, name)` in the
    `evals` namespace (minus `base/`'s three kinds) is the only thing that sees. Catches a
    fourth object nobody agreed to (a second Service, a stray ConfigMap) with the same
    equality. This is the ONE case in this module the whole rest of the file depends on: every
    other case below asserts SHAPE, and a build with none of these three objects fails here
    first, with a readable diff, rather than failing sixty cases down with "0 objects found"."""
    present = {
        (str(document.get("kind")), name_of(document))
        for document in built
        if namespace_of(document) == EVALS and str(document.get("kind")) not in BASE_KINDS
    }
    assert present == EVALS_OBJECTS, (
        f"unexpected: {sorted(present - EVALS_OBJECTS)}; missing: {sorted(EVALS_OBJECTS - present)}"
    )


# --------------------------------------------------------------------------
# the CronJob's shape
# --------------------------------------------------------------------------


def test_there_is_no_nightly_eval_cronjob(built: list[dict[str, Any]]) -> None:
    """The in-cluster nightly CronJob goes, because its cases would be a copy of the
    help desk's file baked into the image. `nightly-eval.yml` is the one nightly path. Catches
    a CronJob still built into the cluster, its manifest still on disk, and a kustomization
    that still lists it (which would fail the build the day the file is deleted)."""
    cronjobs = [
        name_of(obj) for obj in objects_of_kind(built, "CronJob") if namespace_of(obj) == EVALS
    ]

    assert cronjobs == []
    assert not (EVALS_DIR / "nightly-eval-cronjob.yaml").exists()
    assert "nightly-eval" not in read(EVALS_DIR / "kustomization.yaml")


def test_no_credential_is_written_inline_under_evals() -> None:
    """Catches a literal secret in any manifest under `infra/k8s/evals/`. A
    literal a later patch strips from the built output is still in git history, so the scan
    is of what is written, and it asserts the directory exists first so a missing folder is a
    named failure and not a pass over zero files."""
    assert EVALS_DIR.is_dir(), f"{EVALS_DIR} does not exist, so this scan reads no files"
    files = sorted(EVALS_DIR.glob("*.yaml"))
    assert files, f"{EVALS_DIR} holds no YAML at all"
    offenders: list[str] = []
    for path in files:
        lines = read(path).splitlines()
        for index, line in enumerate(lines):
            if INLINE_CREDENTIAL.match(line):
                offenders.append(f"{path.name}:{index + 1}: {line.strip()}")
                continue
            if not CREDENTIAL_ENV_NAME.match(line):
                continue
            for offset, following in enumerate(lines[index + 1 : index + 1 + ENV_VALUE_LOOKAHEAD]):
                if ENV_VALUE.match(following):
                    offenders.append(f"{path.name}:{index + offset + 2}: {line.strip()}")
                    break
    assert offenders == [], f"inline credentials under {EVALS_DIR.name}/: {offenders}"


# --------------------------------------------------------------------------
# every container declares its resources
# --------------------------------------------------------------------------


def test_every_container_in_evals_declares_requests_and_a_memory_limit(
    built: list[dict[str, Any]],
) -> None:
    """Criterion (the LimitRange would silently default an incomplete container rather than
    reject it, exactly `test_k8s_platform.py`'s reasoning). Catches the Pushgateway or the
    CronJob's container shipped with no `resources` block at all, and catches one of the
    three required quantities dropped while the other two survive - which the namespace
    subtotal case below cannot see on its own, since a missing key reads as zero rather than
    as a failure to a naive sum."""
    owners = evals_owners(built)
    assert owners, "the build emits no pod templates in the evals namespace"

    missing: list[str] = []
    for owner in owners:
        label = f"{owner.get('kind')}/{name_of(owner)}"
        for container in all_containers(owner):
            quantities = resource_quantities(container)
            where = f"{label}:{container.get('name')}"
            missing.extend(
                f"{where} has no {key}" for key in REQUIRED_CONTAINER_KEYS if key not in quantities
            )
    assert missing == [], f"containers missing resources: {missing}"


# --------------------------------------------------------------------------
# the namespace subtotal fits the existing quota
# --------------------------------------------------------------------------


def test_the_evals_subtotals_fit_inside_the_namespace_quota(
    built: list[dict[str, Any]],
) -> None:
    """Criterion: the new Pushgateway Deployment (by its replica count) and the CronJob's one
    run's containers must fit inside the ALREADY-DECLARED `evals-envelope`
    `512Mi`/`300m`/`1Gi` (`infra/k8s/base/quotas.yaml`), derived from the BUILT pod templates
    and compared to the quota read from the SAME build (never restated).
    Catches the arithmetic failing even though every individual container above is
    individually correctly-shaped - a Pushgateway given two replicas, or either workload
    sized generously enough on its own but not against the OTHER one sharing the same small
    envelope, which is the failure mode `platform/`'s equivalent case exists to catch and
    `evals/` inherits unchanged."""
    quotas = [
        quota for quota in objects_of_kind(built, "ResourceQuota") if namespace_of(quota) == EVALS
    ]
    assert len(quotas) == 1, f"{len(quotas)} ResourceQuotas in {EVALS}"
    hard = quota_hard(quotas[0])

    owners = evals_owners(built)
    assert owners, "the build emits no pod templates in the evals namespace"

    totals = {"requests.memory": 0, "requests.cpu": 0, "limits.memory": 0}
    for owner in owners:
        replicas = replicas_of(owner)
        for container in all_containers(owner):
            quantities = resource_quantities(container)
            totals["requests.memory"] += replicas * memory_bytes(quantities["requests.memory"])
            totals["requests.cpu"] += replicas * cpu_millicores(quantities["requests.cpu"])
            totals["limits.memory"] += replicas * memory_bytes(quantities["limits.memory"])

    assert totals["requests.memory"] <= memory_bytes(hard["requests.memory"]), (
        f"evals requests {totals['requests.memory']} bytes against {hard['requests.memory']}"
    )
    assert totals["requests.cpu"] <= cpu_millicores(hard["requests.cpu"]), (
        f"evals requests {totals['requests.cpu']}m against {hard['requests.cpu']}"
    )
    assert totals["limits.memory"] <= memory_bytes(hard["limits.memory"]), (
        f"evals limits {totals['limits.memory']} bytes against {hard['limits.memory']}"
    )
