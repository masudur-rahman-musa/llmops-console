# `infra/k8s/gpu`

vLLM serving Qwen2.5-7B-Instruct on one GPU host, with everything it needs to
run on a bare k3s. The root build does not include this directory: the
long-lived cluster has no GPU. The host comes from `infra/terraform/aws-gpu`.

```
namespace.yaml         the `gpu` namespace, its quota and limit range
device-plugin.yaml     RuntimeClass `nvidia` and the NVIDIA device plugin
vllm.yaml              model claim, fetch Job, vLLM Deployment, two Services, HPA
metrics.yaml           a local Prometheus, prometheus-adapter, the APIService, RBAC
networkpolicies.yaml   default deny, then what each pod needs
```

## Applying it

A human applies it, through the SSH tunnel the module README describes:

    kubectl apply -k infra/k8s/gpu
    kubectl -n gpu wait --for=condition=complete job/vllm-model-fetch --timeout=30m
    kubectl -n gpu rollout status deploy/vllm --timeout=20m

The `apps/gateway-egress-vllm` policy needs an `apps` namespace. On a bare host,
create it first (`kubectl create namespace apps`), or apply the gateway there.

Validate without applying:

    make k8s-gpu-check

## The fetch Job is the one pod with internet access

`vllm-model-fetch` downloads a pinned revision of the weights from Hugging Face
on TCP 443, with private ranges and the metadata endpoint excluded. It writes a
marker file only when the download has finished, and the vLLM pod's
initContainer waits for that marker. vLLM itself runs offline and has no egress
but DNS. The Job never receives a prompt.

## The NodePort and its one CIDR rule exist for the measurement run

`vllm-node` publishes the model server on the node's port 30800, and
`vllm-ingress` admits one extra source on TCP 8000: the `172.31.0.0/16` range,
which is the default VPC the GPU instance runs in and therefore the node's own
address.

Both are there because the run that measures this deployment is a compose stack
on the host itself. The gateway's container dials `host.docker.internal:30800`;
that traffic is source-translated to the node, which no pod or namespace
selector can name. `kubectl port-forward` is not an alternative: its traffic is
sourced from the node too, and k3s drops it against a pod-selector rule.

It is a widening, and a deliberate one. The host lives about four hours and is
destroyed with the run, and the rule names one port. The
long-lived cluster carries neither object: the root build does not include this
directory.

## The HPA cannot add a replica here

The HPA scales vLLM between 1 and 2 on `vllm_num_requests_waiting`. On a
one-GPU host the second replica stays Pending: there is no second GPU, and the
quota allows one. It is there to prove the metrics path works end to end, from
vLLM's `/metrics` through the adapter to a scaling decision.
