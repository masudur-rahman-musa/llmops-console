r"""`infra/k8s/gpu`: the standalone vLLM deployment for one GPU host, read off what kustomize emits.

The stream comes from the session-scoped `built_gpu` fixture in `tests/infra/conftest.py`: one
`kubectl kustomize infra/k8s/gpu` for the whole session. The root build never
includes this directory, so `built` cannot stand in for it. A handful of cases read plain files
instead: the root kustomization, `routing.yaml`, the Makefile, `ci.yml`, and the two committed
manifests whose values this build must equal by value (`monitoring/prometheus.yaml`'s image and
`base/networkpolicies.yaml`'s `gateway-egress` except list).

NetworkPolicies are compared as a set of `(peer, port)` grants rather than as raw rule lists.
Kubernetes unions allows, so two rules or one rule with two ports grant the same thing, and a
comparison on the grant set is blind to that grouping while still seeing an extra peer or port.
A peer with no `namespaceSelector` belongs to the policy's own namespace; a rule with no `to` or
`from` is any peer; a rule with no `ports` is every port.

The helpers are local, and none is imported from `test_hetzner_module.py` except the Makefile
and workflow readers and the token-scan patterns, which exist there once.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest
import yaml

from infra.test_hetzner_module import (
    ASSIGNMENT,
    SECRETISH_NAME,
    TOKEN_LITERAL,
    load_yaml,
    read,
    recipe,
    steps,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
K8S = REPO_ROOT / "infra" / "k8s"
GPU_DIR = K8S / "gpu"
ROOT_KUSTOMIZATION = K8S / "kustomization.yaml"
MONITORING_PROMETHEUS = K8S / "monitoring" / "prometheus.yaml"
BASE_POLICIES = K8S / "base" / "networkpolicies.yaml"
ROUTING = REPO_ROOT / "routing.yaml"
MAKEFILE = REPO_ROOT / "Makefile"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"

GPU = "gpu"
APPS = "apps"
KUBE_SYSTEM = "kube-system"
NAME_LABEL = "app.kubernetes.io/name"
NAMESPACE_NAME_LABEL = "kubernetes.io/metadata.name"
NETWORK_POLICY = "NetworkPolicy"

#: The pinned model revision, written out literally wherever it is compared.
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
MARKER = f"/models/.complete-{REVISION}"

VLLM_IMAGE = "vllm/vllm-openai:v0.29.0"
DEVICE_PLUGIN_IMAGE = "nvcr.io/nvidia/k8s-device-plugin:v0.20.0"
ADAPTER_IMAGE = "registry.k8s.io/prometheus-adapter/prometheus-adapter:v0.12.0"
PROMETHEUS_IMAGE = "prom/prometheus:v3.1.0"
INIT_IMAGE = "busybox:1.37.0"

#: Section 1's quota, compared as strings so YAML's `1` and `"1"` read the same.
ENVELOPE = {
    "requests.memory": "11Gi",
    "requests.cpu": "2500m",
    "limits.memory": "16Gi",
    "requests.nvidia.com/gpu": "1",
}

VLLM_REQUESTS = {"nvidia.com/gpu": "1", "cpu": "2", "memory": "10Gi"}
VLLM_LIMITS = {"nvidia.com/gpu": "1", "memory": "14Gi"}

#: Section 1's argument list, each flag and each value its own item.
VLLM_ARGS = [
    "--model",
    "/models/Qwen2.5-7B-Instruct",
    "--served-model-name",
    "qwen2.5-7b-instruct",
    "--max-model-len",
    "8192",
    "--max-num-seqs",
    "16",
    "--gpu-memory-utilization",
    "0.90",
    "--enable-auto-tool-choice",
    "--tool-call-parser",
    "hermes",
    "--host",
    "0.0.0.0",
    "--port",
    "8000",
]

INIT_COMMAND = ["sh", "-c", f"until [ -f {MARKER} ]; do sleep 10; done"]

VLLM_HOST = "vllm.gpu.svc.cluster.local"

#: The node's own address range as `vllm-ingress` admits it: AWS's
#: default VPC, which is where `infra/terraform/aws-gpu` puts the instance. Written out
#: literally here and in the manifest, never derived.
NODE_CIDR = "172.31.0.0/16"

#: The NodePort the measurement harness dials from the host, and the Service publishing it.
NODE_PORT = 30800
NODE_SERVICE = "vllm-node"

#: The port vLLM serves on, behind both Services.
VLLM_PORT = 8000

GPU_POLICY_NAMES = {
    "default-deny",
    "allow-dns",
    "vllm-ingress",
    "vllm-model-fetch-egress",
    "prometheus-egress",
    # The adapter's queries into Prometheus. `adapter-egress` allows the call out
    # and nothing admitted it, so `default-deny` refused every one of them.
    "prometheus-ingress",
    "adapter-ingress",
    "adapter-egress",
}

#: Section 1's refusal list for image tags: moving names, never a release.
FLOATING_TAGS = frozenset({"latest", "main", "stable", "server", "nightly"})

#: The pod-template kinds this build can emit.
POD_OWNERS = frozenset({"Deployment", "DaemonSet", "StatefulSet", "Job", "ReplicaSet"})

ANY_PEER: tuple[str, ...] = ("any",)
ALL_PORTS: tuple[str | int | None, ...] = ("all", None, None)

Peer = tuple[Any, ...]
Port = tuple[Any, ...]
Grant = tuple[Peer, Port]


def pods(namespace: str, name: str) -> Peer:
    """A normalised peer: pods labelled `name` in `namespace`."""
    return ("peer", namespace, frozenset({(NAME_LABEL, name)}))


def namespace_peer(namespace: str) -> Peer:
    """A normalised peer: every pod in `namespace`."""
    return ("peer", namespace, None)


def tcp(port: int) -> Port:
    return ("TCP", port, None)


def udp(port: int) -> Port:
    return ("UDP", port, None)


# ---------------------------------------------------------------------------------------------
# Stream readers


def objects(built: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [document for document in built if document.get("kind") == kind]


def the_one(
    built: list[dict[str, Any]], kind: str, name: str, namespace: str | None = None
) -> dict[str, Any]:
    """The single object of `kind` and `name` (and `namespace`, when given) in the build."""
    found = [
        document
        for document in objects(built, kind)
        if document.get("metadata", {}).get("name") == name
        and (namespace is None or document.get("metadata", {}).get("namespace") == namespace)
    ]
    assert len(found) == 1, f"{len(found)} {kind} {namespace}/{name} in the gpu build"
    return found[0]


def pod_spec(document: dict[str, Any]) -> dict[str, Any]:
    spec = document.get("spec", {}).get("template", {}).get("spec")
    assert isinstance(spec, dict), f"{document.get('kind')} carries no pod template"
    return spec


def pod_labels(document: dict[str, Any]) -> dict[str, str]:
    labels = document.get("spec", {}).get("template", {}).get("metadata", {}).get("labels", {})
    assert isinstance(labels, dict)
    return {str(key): str(value) for key, value in labels.items()}


def container(spec: dict[str, Any], name: str) -> dict[str, Any]:
    found = [entry for entry in spec.get("containers", []) if entry.get("name") == name]
    assert len(found) == 1, f"{len(found)} containers named {name}"
    return dict(found[0])


def sole(entries: list[dict[str, Any]], what: str) -> dict[str, Any]:
    assert len(entries) == 1, f"{len(entries)} {what}"
    return entries[0]


def as_strings(mapping: dict[str, Any]) -> dict[str, str]:
    return {str(key): str(value) for key, value in mapping.items()}


def vllm(built: list[dict[str, Any]]) -> dict[str, Any]:
    return the_one(built, "Deployment", "vllm", GPU)


def fetch_job(built: list[dict[str, Any]]) -> dict[str, Any]:
    return the_one(built, "Job", "vllm-model-fetch", GPU)


def fetch_script(built: list[dict[str, Any]]) -> str:
    """The fetch Job's one Python script: `command: [python3, -c]`, `args: [<script>]`."""
    job_container = sole(pod_spec(fetch_job(built)).get("containers", []), "fetch containers")
    assert job_container.get("command") == ["python3", "-c"], job_container.get("command")
    arguments = job_container.get("args")
    assert isinstance(arguments, list) and len(arguments) == 1, arguments
    return str(arguments[0])


# ---------------------------------------------------------------------------------------------
# NetworkPolicy normalisation


def policies(built: list[dict[str, Any]], namespace: str) -> dict[str, dict[str, Any]]:
    return {
        str(document["metadata"]["name"]): document
        for document in objects(built, NETWORK_POLICY)
        if document.get("metadata", {}).get("namespace") == namespace
    }


def policy(built: list[dict[str, Any]], namespace: str, name: str) -> dict[str, Any]:
    found = policies(built, namespace)
    assert name in found, f"no NetworkPolicy {namespace}/{name}, only {sorted(found)}"
    return found[name]


def selector_key(selector: dict[str, Any] | None) -> frozenset[tuple[str, str]] | str | None:
    """`None` for every pod, a frozenset of label pairs, or the raw text when not plain labels."""
    if selector is None or selector == {}:
        return None
    if set(selector) == {"matchLabels"}:
        return frozenset(as_strings(selector["matchLabels"]).items())
    return repr(selector)


def peer_key(peer: dict[str, Any], own_namespace: str) -> Peer:
    if "ipBlock" in peer:
        block = peer["ipBlock"]
        return (
            "ipBlock",
            str(block.get("cidr")),
            tuple(str(net) for net in block.get("except", [])),
        )
    namespace_selector = peer.get("namespaceSelector")
    if namespace_selector is None:
        namespace: str = own_namespace
    elif namespace_selector == {}:
        namespace = "*"
    else:
        labels = namespace_selector.get("matchLabels", {})
        plain = set(namespace_selector) == {"matchLabels"} and set(labels) == {NAMESPACE_NAME_LABEL}
        namespace = str(labels[NAMESPACE_NAME_LABEL]) if plain else repr(namespace_selector)
    return ("peer", namespace, selector_key(peer.get("podSelector")))


def port_key(port: dict[str, Any]) -> Port:
    raw = port.get("port")
    number = int(raw) if raw is not None and str(raw).isdigit() else raw
    return (str(port.get("protocol", "TCP")), number, port.get("endPort"))


def grants(document: dict[str, Any], direction: str) -> set[Grant]:
    """Every `(peer, port)` pair the policy's `ingress` or `egress` rules allow."""
    field = "from" if direction == "ingress" else "to"
    own = str(document["metadata"]["namespace"])
    found: set[Grant] = set()
    for rule in document.get("spec", {}).get(direction, []) or []:
        peers = [peer_key(peer, own) for peer in rule.get(field) or []] or [ANY_PEER]
        ports = [port_key(port) for port in rule.get("ports") or []] or [ALL_PORTS]
        found |= {(peer, port) for peer in peers for port in ports}
    return found


def match_labels(document: dict[str, Any]) -> dict[str, str]:
    selector = document.get("spec", {}).get("podSelector", {})
    assert set(selector) <= {"matchLabels"}, f"not a plain label selector: {selector}"
    return as_strings(selector.get("matchLabels", {}))


# ---------------------------------------------------------------------------------------------
# Structure


def test_the_root_kustomization_does_not_list_the_gpu_directory() -> None:
    """Criterion 1. Catches `- gpu` added to the root `resources`, which drags a 10Gi GPU pod,
    a second Prometheus and a cluster-wide APIService into every Hetzner apply and every CI
    build of the long-lived cluster that has no GPU at all."""
    resources = load_yaml(ROOT_KUSTOMIZATION).get("resources", [])
    assert isinstance(resources, list)

    assert [entry for entry in resources if "gpu" in str(entry)] == []


def test_the_namespace_and_its_quota_are_the_agreed_envelope(
    built_gpu: list[dict[str, Any]],
) -> None:
    """Criterion 2. Catches the quota copied from `apps-envelope` (3Gi of requests, where the
    vLLM pod alone asks 10Gi and is refused at admission), a missing `requests.nvidia.com/gpu`
    line that lets a second GPU pod be admitted, and any fifth key. Equality on `hard` sees both
    a missing and an extra key."""
    the_one(built_gpu, "Namespace", GPU)
    quota = the_one(built_gpu, "ResourceQuota", "gpu-envelope", GPU)

    assert as_strings(quota["spec"]["hard"]) == ENVELOPE


def test_the_namespace_has_a_limit_range(built_gpu: list[dict[str, Any]]) -> None:
    """Criterion 3. Catches the LimitRange left out, where a quota on `requests.memory` makes the
    API server refuse any container that states no request, the initContainer included."""
    ranges = [
        document
        for document in objects(built_gpu, "LimitRange")
        if document.get("metadata", {}).get("namespace") == GPU
    ]

    assert len(ranges) >= 1, "no LimitRange in gpu"


def test_the_runtime_class_and_device_plugin_are_pinned(built_gpu: list[dict[str, Any]]) -> None:
    """Criteria 4 and 5. Catches a RuntimeClass whose `handler` is not `nvidia` (k3s registers
    the NVIDIA containerd runtime under that name, so any other value fails every sandbox), the
    device plugin in a namespace other than `kube-system`, a plugin on the default runtime
    (it then cannot see the GPU and advertises zero `nvidia.com/gpu`), and its image drifting."""
    runtime = the_one(built_gpu, "RuntimeClass", "nvidia")
    plugins = [
        document
        for document in objects(built_gpu, "DaemonSet")
        if document.get("metadata", {}).get("namespace") == KUBE_SYSTEM
    ]
    plugin = sole(plugins, "DaemonSets in kube-system")
    spec = pod_spec(plugin)

    assert runtime.get("handler") == "nvidia"
    assert spec.get("runtimeClassName") == "nvidia"
    assert [entry.get("image") for entry in spec.get("containers", [])] == [DEVICE_PLUGIN_IMAGE]


# ---------------------------------------------------------------------------------------------
# vLLM


def test_the_vllm_deployment_is_one_recreated_gpu_pod(built_gpu: list[dict[str, Any]]) -> None:
    """Criteria 6 and 7. Catches `RollingUpdate` (the new pod waits forever for the one GPU the
    old pod holds), `replicas: 2`, the default runtime class (no GPU device inside the
    container), a floating or different vLLM image, and resources that differ by any key: a
    `limits.cpu` added throttles token generation, a missing GPU limit schedules without a GPU."""
    deployment = vllm(built_gpu)
    spec = pod_spec(deployment)
    server = container(spec, "vllm")
    resources = server.get("resources", {})

    assert deployment["spec"].get("replicas") == 1
    assert deployment["spec"].get("strategy", {}).get("type") == "Recreate"
    assert spec.get("runtimeClassName") == "nvidia"
    assert server.get("image") == VLLM_IMAGE
    assert as_strings(resources.get("requests", {})) == VLLM_REQUESTS
    assert as_strings(resources.get("limits", {})) == VLLM_LIMITS


def test_the_vllm_args_are_exactly_the_agreed_list(built_gpu: list[dict[str, Any]]) -> None:
    """Criterion 9. Catches a dropped `--tool-call-parser hermes` (tool calls come back as plain
    text), a `--max-model-len` that no longer fits 14Gi, a served name the gateway's alias does
    not dial, and any extra flag. Also catches a `command` override, which replaces the image's
    entrypoint and turns the args into arguments to something else."""
    server = container(pod_spec(vllm(built_gpu)), "vllm")

    assert "command" not in server, f"vllm overrides command: {server.get('command')}"
    assert [str(item) for item in server.get("args", [])] == VLLM_ARGS


def test_vllm_runs_offline_with_a_memory_backed_shm(built_gpu: list[dict[str, Any]]) -> None:
    """Criteria 8 and 10. Catches `HF_HUB_OFFLINE` absent or `"0"`, where vLLM reaches for the
    hub at start and hangs behind the default-deny egress; and catches `/dev/shm` left as the
    container's 64Mi default or on disk, which crashes tensor transfer workers under load, or
    the volume declared and never mounted at `/dev/shm`."""
    spec = pod_spec(vllm(built_gpu))
    server = container(spec, "vllm")
    offline = [entry for entry in server.get("env", []) if entry.get("name") == "HF_HUB_OFFLINE"]
    shm_mounts = [
        mount for mount in server.get("volumeMounts", []) if mount.get("mountPath") == "/dev/shm"
    ]
    mount = sole(shm_mounts, "mounts at /dev/shm")
    volume = sole(
        [entry for entry in spec.get("volumes", []) if entry.get("name") == mount.get("name")],
        f"volumes named {mount.get('name')}",
    )

    assert [str(entry.get("value")) for entry in offline] == ["1"]
    assert as_strings(volume.get("emptyDir", {})) == {"medium": "Memory", "sizeLimit": "2Gi"}


def test_vllm_waits_for_the_download_marker(built_gpu: list[dict[str, Any]]) -> None:
    """Criterion 11. Catches an initContainer that waits on the model DIRECTORY (it exists from
    the first file the Job writes, so vLLM starts on a partial download), one that checks the
    marker once and exits instead of looping, a marker of a different revision, and a floating
    busybox image."""
    init_containers = pod_spec(vllm(built_gpu)).get("initContainers", [])
    waiter = sole(init_containers, "initContainers on vllm")

    assert waiter.get("image") == INIT_IMAGE
    assert waiter.get("command") == INIT_COMMAND


def test_the_fetch_job_downloads_the_pinned_revision_to_the_model_path(
    built_gpu: list[dict[str, Any]],
) -> None:
    """Criteria 12 and 13. Catches `snapshot_download` without `revision=` (the branch head, so
    two runs a week apart measure two different models), a different repository, a `local_dir`
    other than the path vLLM's `--model` reads, and a Job on a different image from the server.
    The three arguments are read inside the ONE call's parentheses, so a revision named in a
    comment elsewhere in the script does not count."""
    job = fetch_job(built_gpu)
    image = sole(pod_spec(job).get("containers", []), "fetch containers").get("image")
    script = fetch_script(built_gpu)
    starts = [match.end() for match in re.finditer(r"snapshot_download\(", script)]
    assert len(starts) == 1, f"{len(starts)} snapshot_download( calls"
    call = script[starts[0] : script.index(")", starts[0])]

    assert image == VLLM_IMAGE
    assert "from huggingface_hub import snapshot_download" in script
    assert '"Qwen/Qwen2.5-7B-Instruct"' in call
    assert f'revision="{REVISION}"' in call
    assert 'local_dir="/models/Qwen2.5-7B-Instruct"' in call


def test_the_fetch_job_writes_the_marker_only_after_the_download(
    built_gpu: list[dict[str, Any]],
) -> None:
    """Criterion 14. Catches the marker touched before `snapshot_download(`, where a Job killed
    mid-download leaves a marker the initContainer trusts and vLLM loads half a model. Every
    occurrence of the marker is checked, not the first."""
    script = fetch_script(built_gpu)
    call = script.find("snapshot_download(")
    markers = [match.start() for match in re.finditer(re.escape(MARKER), script)]

    assert call >= 0, "no snapshot_download( call"
    assert markers != [], "the script never names the marker"
    assert [position for position in markers if position < call] == []


def test_the_model_volume_claim_is_forty_gigabytes_on_the_default_class(
    built_gpu: list[dict[str, Any]],
) -> None:
    """Criterion 15. Catches a claim below the 15GB of weights plus headroom, `ReadWriteMany`
    (k3s's local-path provisioner refuses it and the claim stays Pending), and a hard-coded
    `storageClassName` that exists on one cluster and not on the next."""
    claim = the_one(built_gpu, "PersistentVolumeClaim", "vllm-models", GPU)
    spec = claim["spec"]

    assert str(spec.get("resources", {}).get("requests", {}).get("storage")) == "40Gi"
    assert spec.get("accessModes") == ["ReadWriteOnce"]
    assert "storageClassName" not in spec


def test_the_service_is_where_routing_yaml_dials(built_gpu: list[dict[str, Any]]) -> None:
    """Criteria 16 and 17. Catches `routing.yaml` left at the bare `vllm:8000`, which does not
    resolve from the gateway's `apps` namespace, a Service on a port other than the one the URL
    names, and a Service named or namespaced differently from the host the URL names. Both
    halves are read from their files, never written down twice."""
    providers = load_yaml(ROUTING)["providers"]
    url = urlparse(str(providers["vllm"]["base_url"]))
    service = the_one(built_gpu, "Service", "vllm", GPU)
    ports = [entry.get("port") for entry in service["spec"].get("ports", [])]

    assert url.hostname == VLLM_HOST
    assert url.hostname == f"{service['metadata']['name']}.{GPU}.svc.cluster.local"
    assert url.port is not None
    assert ports == [url.port]


def resolved_target_port(
    service: dict[str, Any], entry: dict[str, Any], pod: dict[str, Any]
) -> int:
    """`entry`'s `targetPort` as a number, resolving a named port through the pod's containers.

    Every container port is scanned and the name is required alongside the number (standing
    rule 44): taking the first port named `http` would read a sidecar's.
    """
    target = entry.get("targetPort", entry.get("port"))
    if isinstance(target, int):
        return target
    if str(target).isdigit():
        return int(str(target))
    numbers = {
        int(port["containerPort"])
        for container_spec in pod_spec(pod).get("containers", [])
        for port in container_spec.get("ports", []) or []
        if port.get("name") == target
    }
    assert len(numbers) == 1, f"{service['metadata']['name']}: {target} resolves to {numbers}"
    return numbers.pop()


def test_the_node_port_service_publishes_vllm_on_the_port_the_harness_dials(
    built_gpu: list[dict[str, Any]],
) -> None:
    """An earlier change criterion 56 and decision C. The measurement harness runs in compose on the GPU
    host and reaches the in-cluster model server through this Service, so its `nodePort` is the
    port `project-evals/tradeoff/gpu-vllm.yaml`'s `base_url` names. Catches `type: NodePort`
    with no `nodePort` (Kubernetes then allocates a random one in the 30000-32767 range and the
    overlay's URL is wrong on every apply), a `nodePort` that is not 30800, a selector that does
    not match the vLLM pod (the port then answers nothing), and a target port other than the
    8000 the container serves on - a named `targetPort` is resolved through the pod, so `http`
    and `8000` are both accepted and a name pointing elsewhere is not."""
    service = the_one(built_gpu, "Service", NODE_SERVICE, GPU)
    deployment = vllm(built_gpu)
    ports = service["spec"].get("ports", [])

    assert service["spec"].get("type") == "NodePort"
    assert as_strings(service["spec"].get("selector", {})) == {NAME_LABEL: "vllm"}
    assert as_strings(service["spec"]["selector"]).items() <= pod_labels(deployment).items()
    assert len(ports) == 1, f"{NODE_SERVICE} publishes {len(ports)} ports: {ports}"
    assert int(ports[0]["nodePort"]) == NODE_PORT
    assert resolved_target_port(service, ports[0], deployment) == VLLM_PORT


def test_the_cluster_internal_vllm_service_stays_the_only_one_without_a_node_port(
    built_gpu: list[dict[str, Any]],
) -> None:
    """An earlier change criterion 56's other half. The gateway inside the cluster still dials the
    ClusterIP Service `routing.yaml` names, and the host-facing one is an ADDITION. Catches the
    existing `vllm` Service flipped to `NodePort` in place of adding a second (every node in a
    future multi-node cluster would then answer on 30800 with no policy admitting it), and a
    third NodePort left behind from a trial."""
    internal = the_one(built_gpu, "Service", "vllm", GPU)
    node_ports = sorted(
        str(document["metadata"]["name"])
        for document in objects(built_gpu, "Service")
        if document["spec"].get("type") == "NodePort"
    )

    assert internal["spec"].get("type") == "ClusterIP"
    assert node_ports == [NODE_SERVICE], node_ports


def test_routing_keeps_the_vllm_provider_disabled() -> None:
    """Criterion 17. Catches `enabled: true` slipped in with the URL edit, which routes live
    traffic to a host that exists only during a one-off measurement run and 502s otherwise."""
    providers = load_yaml(ROUTING)["providers"]

    assert providers["vllm"].get("enabled") is False


def test_the_hpa_scales_one_to_two_on_waiting_requests(built_gpu: list[dict[str, Any]]) -> None:
    """Criterion 18. Catches a `Resource` CPU metric (a GPU-bound server's CPU says nothing about
    its queue), `maxReplicas` above 2, a target other than 4 waiting requests, a metric name the
    adapter never publishes, and an HPA aimed at anything but the vllm Deployment."""
    hpa = the_one(built_gpu, "HorizontalPodAutoscaler", "vllm", GPU)
    spec = hpa["spec"]
    metric = sole(spec.get("metrics", []), "HPA metrics")

    assert hpa.get("apiVersion") == "autoscaling/v2"
    assert spec.get("scaleTargetRef", {}).get("name") == "vllm"
    assert spec.get("minReplicas") == 1
    assert spec.get("maxReplicas") == 2
    assert metric.get("type") == "Pods"
    assert metric["pods"]["metric"]["name"] == "vllm_num_requests_waiting"
    assert metric["pods"]["target"]["type"] == "AverageValue"
    assert str(metric["pods"]["target"]["averageValue"]) == "4"


def test_the_hpa_comment_names_the_one_gpu_pending_replica() -> None:
    """Criterion 19. Catches the comment dropped, which leaves the next reader to discover by
    incident that replica 2 can never schedule on a host with one GPU."""
    text = read(GPU_DIR / "vllm.yaml")
    comments = " ".join(line for line in text.splitlines() if line.lstrip().startswith("#"))

    assert "Pending" in comments, "no comment on vllm.yaml says replica 2 stays Pending"


# ---------------------------------------------------------------------------------------------
# Metrics path


def test_the_adapter_publishes_the_waiting_requests_metric(built_gpu: list[dict[str, Any]]) -> None:
    """Criteria 20 and 21. Catches a floating adapter image, a rule on a series vLLM does not
    export (the HPA then reads `<unknown>` and never scales), a published name that differs
    from the HPA's, and a rule without `namespace` and `pod` overrides, which the Pods metric
    API cannot attribute to a pod."""
    adapter = the_one(built_gpu, "Deployment", "prometheus-adapter", GPU)
    images = [entry.get("image") for entry in pod_spec(adapter).get("containers", [])]
    config_map = the_one(built_gpu, "ConfigMap", "prometheus-adapter", GPU)
    config = yaml.safe_load(config_map["data"]["config.yaml"])
    rule = config["rules"][0]

    assert images == [ADAPTER_IMAGE]
    assert "vllm:num_requests_waiting" in rule["seriesQuery"]
    assert {"namespace", "pod"} <= set(rule["resources"]["overrides"])
    assert rule["name"]["as"] == "vllm_num_requests_waiting"


def test_the_custom_metrics_api_is_registered(built_gpu: list[dict[str, Any]]) -> None:
    """Criterion 22. Catches the APIService left out, where the adapter runs, answers nothing,
    and the HPA reports `FailedGetPodsMetric` forever."""
    the_one(built_gpu, "APIService", "v1beta1.custom.metrics.k8s.io")


def test_the_local_prometheus_image_equals_the_monitoring_one(
    built_gpu: list[dict[str, Any]],
) -> None:
    """Criterion 24 and a project rule. Catches the GPU Prometheus bumped or pinned apart from
    `monitoring/`'s, where the next image bump updates one of two copies and the scrape config
    format drifts silently between them."""
    monitoring = [
        document
        for document in yaml.safe_load_all(read(MONITORING_PROMETHEUS))
        if isinstance(document, dict)
        and document.get("kind") in ("Deployment", "StatefulSet")
        and document.get("metadata", {}).get("name") == "prometheus"
    ]
    reference = container(pod_spec(sole(monitoring, "monitoring Prometheus")), "prometheus")
    local = container(pod_spec(the_one(built_gpu, "Deployment", "prometheus", GPU)), "prometheus")

    assert reference.get("image") == PROMETHEUS_IMAGE
    assert local.get("image") == reference.get("image")


def test_the_local_prometheus_scrapes_only_vllm_pods_in_gpu(
    built_gpu: list[dict[str, Any]],
) -> None:
    """Criterion 23. Catches `kubernetes_sd_configs` with no namespace limit (the pod then lists
    pods cluster-wide and needs RBAC it should not have), a second scrape job, and a relabel
    that keeps every pod rather than those labelled `vllm`."""
    config_map = the_one(built_gpu, "ConfigMap", "prometheus", GPU)
    config = yaml.safe_load(config_map["data"]["prometheus.yml"])
    job = sole(config.get("scrape_configs", []), "scrape jobs")
    keep = {
        "action": "keep",
        "source_labels": ["__meta_kubernetes_pod_label_app_kubernetes_io_name"],
        "regex": "vllm",
    }

    assert job.get("kubernetes_sd_configs") == [{"role": "pod", "namespaces": {"names": [GPU]}}]
    assert keep in job.get("relabel_configs", [])


# ---------------------------------------------------------------------------------------------
# Network policy


def test_the_policy_names_are_exactly_the_agreed_eight(built_gpu: list[dict[str, Any]]) -> None:
    """Criteria 25 and 26. Catches a policy dropped (the pod it gave traffic back to is dark
    behind `default-deny`), a `vllm-egress` added "for the hub", and anything this build adds to
    `apps` beyond `gateway-egress-vllm`. Equality fails both ways."""
    assert set(policies(built_gpu, GPU)) == GPU_POLICY_NAMES
    assert set(policies(built_gpu, APPS)) == {"gateway-egress-vllm"}


@pytest.mark.parametrize(
    ("name", "selector", "direction", "expected"),
    [
        (
            "allow-dns",
            None,
            "egress",
            {(namespace_peer(KUBE_SYSTEM), udp(53)), (namespace_peer(KUBE_SYSTEM), tcp(53))},
        ),
        (
            "prometheus-egress",
            {NAME_LABEL: "prometheus"},
            "egress",
            {(pods(GPU, "vllm"), tcp(8000)), (ANY_PEER, tcp(6443)), (ANY_PEER, tcp(443))},
        ),
        ("adapter-ingress", {NAME_LABEL: "prometheus-adapter"}, "ingress", {(ANY_PEER, tcp(6443))}),
        (
            "adapter-egress",
            {NAME_LABEL: "prometheus-adapter"},
            "egress",
            {(pods(GPU, "prometheus"), tcp(9090)), (ANY_PEER, tcp(6443)), (ANY_PEER, tcp(443))},
        ),
    ],
)
def test_each_support_policy_grants_exactly_its_agreed_pairs(
    built_gpu: list[dict[str, Any]],
    name: str,
    selector: dict[str, str] | None,
    direction: str,
    expected: set[Grant],
) -> None:
    """Criterion 25's support rows. Catches DNS on UDP alone (fails intermittently on large
    answers), the API-server rules given a `to` peer that matches nothing, a Prometheus egress
    widened to every `gpu` pod or to every port, and the adapter reaching Prometheus on a port
    other than 9090. Grants are compared as a set, so rule grouping is free and extras are not."""
    document = policy(built_gpu, GPU, name)

    if selector is not None:
        assert match_labels(document) == selector
    assert direction.capitalize() in document["spec"].get("policyTypes", [])
    assert grants(document, direction) == expected


def test_default_deny_covers_every_pod_both_ways(built_gpu: list[dict[str, Any]]) -> None:
    """Criterion 25's first row. Catches `default-deny` with only `Ingress`, which leaves every
    `gpu` pod, the model server included, free to call anywhere, and catches it scoped to one
    label so the adapter and Prometheus run unrestricted."""
    document = policy(built_gpu, GPU, "default-deny")
    spec = document["spec"]

    assert spec.get("podSelector") == {}
    assert set(spec.get("policyTypes", [])) == {"Ingress", "Egress"}
    assert grants(document, "ingress") == set()
    assert grants(document, "egress") == set()


def test_no_policy_selecting_vllm_grants_egress_beyond_dns(built_gpu: list[dict[str, Any]]) -> None:
    """Criterion 27, over EVERY policy in the build, not the named ones. Catches `vllm-ingress`
    given an `Egress` type and a 443 rule (its name and its ingress grants unchanged, so no
    other case sees it), and any `{}`-selector policy granting a port other than 53, which
    reaches the model server too. A rule with no `ports` is every port and is a finding. The
    scan must find `default-deny` and `allow-dns` among the selecting policies, so a vllm pod
    with different labels cannot make it vacuous."""
    labels = pod_labels(vllm(built_gpu))
    assert labels.get(NAME_LABEL) == "vllm", labels
    selecting: list[str] = []
    findings: list[str] = []
    for document in objects(built_gpu, NETWORK_POLICY):
        if document["metadata"].get("namespace") != GPU:
            continue
        selector = document["spec"].get("podSelector", {})
        wanted = as_strings(selector.get("matchLabels", {}))
        plain = set(selector) <= {"matchLabels"}
        if plain and not wanted.items() <= labels.items():
            continue
        name = str(document["metadata"]["name"])
        selecting.append(name)
        if "Egress" not in document["spec"].get("policyTypes", []):
            continue
        for rule in document["spec"].get("egress", []) or []:
            ports = rule.get("ports") or []
            if not ports or any(str(port.get("port")) != "53" for port in ports):
                findings.append(f"{name}: {ports or 'every port'}")

    assert {"default-deny", "allow-dns"} <= set(selecting), selecting
    assert findings == [], f"egress beyond DNS for the vllm pod: {findings}"


def test_vllm_ingress_admits_the_gateway_prometheus_and_the_node_on_8000_only(
    built_gpu: list[dict[str, Any]],
) -> None:
    """Criterion 25's `vllm-ingress` and an earlier change decision C. Catches the gateway peer written
    without its `namespaceSelector` (it then means a `gateway` pod in `gpu`, which does not
    exist), every `apps` pod admitted, a FOURTH source, and a port other than TCP 8000. The
    third source is the node's own VPC range, which is the address the NodePort SNATs the
    measurement harness's traffic to; without it k3s's controller drops every measured turn
    against a pod-selector rule that cannot match a host-sourced packet. Written as an equality
    on the whole grant set, so the node peer widened to `0.0.0.0/0` is red too. The policy must
    select the vllm pod, or the grants protect nothing."""
    document = policy(built_gpu, GPU, "vllm-ingress")

    assert match_labels(document) == {NAME_LABEL: "vllm"}
    assert grants(document, "ingress") == {
        (pods(APPS, "gateway"), tcp(8000)),
        (pods(GPU, "prometheus"), tcp(8000)),
        (("ipBlock", NODE_CIDR, ()), tcp(8000)),
    }


def test_exactly_two_policies_carry_an_ip_block_each_on_its_own_port(
    built_gpu: list[dict[str, Any]],
) -> None:
    """Criteria 25 and 28, and an earlier change decision C. Two rules in this build name a raw address
    range and no more: the fetch Job reaching the model registry on 443, and the node reaching
    vLLM on 8000. Catches an `ipBlock` policy written OUTSIDE `gpu` and `apps` (a `kube-system`
    policy, which the name-set case never reads), a third one added later, the node block put
    on the fetch policy or on 443, the fetch policy selecting the vllm pod rather than the Job's
    own label, and either block granted every port - a rule with no `ports` normalises to
    `ALL_PORTS`, so it cannot pass as `{tcp(443)}`. The Job template's own label is checked too,
    so a typo that selects nothing is red."""
    with_blocks: dict[str, set[Port]] = {}
    for document in objects(built_gpu, NETWORK_POLICY):
        name = f"{document['metadata'].get('namespace')}/{document['metadata']['name']}"
        for direction, field in (("ingress", "from"), ("egress", "to")):
            for rule in document["spec"].get(direction, []) or []:
                if not any("ipBlock" in peer for peer in rule.get(field) or []):
                    continue
                ports = [port_key(port) for port in rule.get("ports") or []] or [ALL_PORTS]
                with_blocks.setdefault(name, set()).update(ports)
    document = policy(built_gpu, GPU, "vllm-model-fetch-egress")
    fetch_labels = pod_labels(fetch_job(built_gpu))
    fetch_ports = {port for _peer, port in grants(document, "egress")}
    peers = {peer[:2] for peer, _port in grants(document, "egress")}

    assert with_blocks == {
        f"{GPU}/vllm-model-fetch-egress": {tcp(443)},
        f"{GPU}/vllm-ingress": {tcp(VLLM_PORT)},
    }
    assert fetch_labels.get(NAME_LABEL) == "vllm-model-fetch"
    assert match_labels(document) == {NAME_LABEL: "vllm-model-fetch"}
    assert fetch_ports == {tcp(443)}
    assert peers == {("ipBlock", "0.0.0.0/0")}


def test_the_fetch_except_list_equals_the_gateway_egress_one(
    built_gpu: list[dict[str, Any]],
) -> None:
    """Criterion 25 and a project rule. Catches the except list missing `169.254.0.0/16`, which
    lets the fetch pod read the cloud metadata endpoint and the instance role behind it, and
    catches either copy edited without the other. Read from `base/networkpolicies.yaml` by
    value, never from a shared constant."""
    reference = [
        document
        for document in yaml.safe_load_all(read(BASE_POLICIES))
        if isinstance(document, dict)
        and document.get("kind") == NETWORK_POLICY
        and document.get("metadata", {}).get("name") == "gateway-egress"
        and document.get("metadata", {}).get("namespace") == APPS
    ]
    reference_blocks = {
        peer for peer, _port in grants(sole(reference, "apps/gateway-egress"), "egress")
    }
    expected = [peer[2] for peer in reference_blocks if peer[0] == "ipBlock"]
    document = policy(built_gpu, GPU, "vllm-model-fetch-egress")
    found = [peer[2] for peer, _port in grants(document, "egress") if peer[0] == "ipBlock"]

    assert len(expected) == 1, expected
    assert expected[0] != ()
    assert set(found) == {expected[0]}


def test_the_gateway_may_reach_vllm_on_8000(built_gpu: list[dict[str, Any]]) -> None:
    """Criterion 26. Catches the policy scoped to every `apps` pod (a Helpdesk workload then
    dials the model directly, bypassing the gateway's residency and budget checks), a peer
    missing its `gpu` namespaceSelector, and a port other than TCP 8000."""
    document = policy(built_gpu, APPS, "gateway-egress-vllm")

    assert match_labels(document) == {NAME_LABEL: "gateway"}
    assert "Egress" in document["spec"].get("policyTypes", [])
    assert grants(document, "egress") == {(pods(GPU, "vllm"), tcp(8000))}


# ---------------------------------------------------------------------------------------------
# Images


def test_every_image_in_the_build_has_an_exact_tag(built_gpu: list[dict[str, Any]]) -> None:
    """Criterion 29, over every container and initContainer of every pod template. Catches an
    untagged `busybox` (resolved as `latest`), `:latest`, and a moving name such as `:stable`,
    where two applies run two different images. A tag must carry a digit to count as a
    release. The scan asserts it found the vllm image, so an empty build cannot pass."""
    images: list[str] = []
    for document in built_gpu:
        if document.get("kind") not in POD_OWNERS:
            continue
        spec = pod_spec(document)
        for entry in [*spec.get("containers", []), *spec.get("initContainers", [])]:
            images.append(str(entry.get("image", "")))
    findings: list[str] = []
    for image in images:
        last = image.rsplit("/", 1)[-1]
        tag = last.rpartition(":")[2] if ":" in last else ""
        if tag == "" or tag in FLOATING_TAGS or re.search(r"\d", tag) is None:
            findings.append(image)

    assert VLLM_IMAGE in images, images
    assert findings == [], f"images without an exact tag: {findings}"


# ---------------------------------------------------------------------------------------------
# Makefile, CI and secrets


def test_k8s_gpu_check_builds_and_validates_with_the_k8s_check_flags() -> None:
    """Criterion 49 and a project rule. Catches the gpu build piped to nothing (kustomize alone
    accepts an unknown field), a `|| true` that turns a schema failure green, `-strict` dropped,
    and kubeconform flags drifting from `k8s-check`'s. Exactly one line names the build, so a
    second, unpiped copy cannot sit beside a correct one."""
    makefile = read(MAKEFILE)
    lines = recipe(makefile, "k8s-gpu-check")
    base = recipe(makefile, "k8s-check")
    building = [line for line in lines if "kustomize build infra/k8s/gpu" in line]
    assert len(building) == 1, f"{len(building)} lines build infra/k8s/gpu: {lines}"
    assert len(base) == 1, base
    line = building[0]

    assert "kustomize build infra/k8s/gpu | kubeconform" in line
    assert "-strict" in line
    assert "||" not in line
    assert line.split("| ", 1)[1] == base[0].split("| ", 1)[1]


def is_printer(step: dict[str, Any]) -> bool:
    """A step whose script is an `echo` only: it names commands, runs none."""
    return str(step.get("run", "")).strip().startswith("echo")


def test_the_k8s_job_runs_k8s_gpu_check_unconditionally() -> None:
    """Criterion 50. Catches the step missing (the gpu build is then checked by nobody, since it
    is applied only from the owner's shell), a second copy, and an `if` that skips it on the
    pull requests where a manifest first arrives. An `echo` naming the target is excluded."""
    jobs = load_yaml(WORKFLOW)["jobs"]
    running = [
        step
        for step in steps(jobs["k8s"])
        if "make k8s-gpu-check" in str(step.get("run", "")) and not is_printer(step)
    ]

    assert len(running) == 1, f"{len(running)} steps run make k8s-gpu-check"
    assert "if" not in running[0]


def test_no_file_in_the_gpu_directory_carries_a_token_shaped_literal() -> None:
    """Criterion 52 and a project rule. Catches a Hugging Face or registry credential pasted into a
    manifest while trying the fetch out. The patterns are the Hetzner module's, imported. The
    directory is asserted first, because an empty scan of a missing directory passes."""
    assert GPU_DIR.is_dir(), f"{GPU_DIR} does not exist"
    findings: list[str] = []
    for path in sorted(GPU_DIR.rglob("*")):
        if not path.is_file():
            continue
        for name, value in ASSIGNMENT.findall(path.read_text(encoding="utf-8")):
            if SECRETISH_NAME.fullmatch(name) and TOKEN_LITERAL.fullmatch(value):
                findings.append(f"{path.name}: {name}")

    assert findings == [], f"token-shaped literals: {findings}"
