# `infra/k8s`

The manifests for the k3s cluster, built with kustomize. Nothing here is
applied by CI or by any tool in this repo: `kustomize build infra/k8s |
kubeconform` is the whole automated check, and `kubectl apply` is a human's.

```
kustomization.yaml     the single root; `resources` grows as directories land
base/                  namespaces, quotas, limit ranges
```

Build it:

```
make k8s-check
```

## The resource envelope

The cluster is one CAX21 (4 vCPU, 8 GiB). Requests are what the scheduler has
to fit; they total exactly 6Gi and 3 CPU, leaving k3s itself a margin in both.

| namespace | requests.memory | requests.cpu | limits.memory | holds |
|---|---|---|---|---|
| `platform` | 2Gi | 800m | 3Gi | Postgres, Redis, Qdrant, cert-manager, the Traefik middlewares |
| `apps` | 3Gi | 1500m | 5Gi | the gateway, the console, llama.cpp, the Helpdesk services |
| `evals` | 512Mi | 300m | 1Gi | the nightly eval CronJob |
| `monitoring` | 512Mi | 400m | 1Gi | Prometheus, Grafana, Alertmanager |
| **total** | **6Gi** | **3** | 10Gi | |

The limits column is oversubscribed against an 8 GiB box on purpose. Requests
are what the scheduler packs and limits are the ceiling a single container may
burst to; llama.cpp alone carries a 3 GiB limit, and it is not resident while
the eval CronJob is running.

`limits.cpu` is deliberately not in the table. A CPU limit throttles rather
than kills, so requiring one on every container would cost the gateway tail
latency and buy no safety. Memory is the opposite case: a container without a
memory limit is what takes the node down, which is why the quota names
`limits.memory` and admission rejects a workload that omits it.

## What enforces it

A `ResourceQuota` that names a resource makes the API server reject any pod in
that namespace whose containers do not declare it. So "requests and limits on
everything" is a property admission enforces here, not one a reviewer checks
workload by workload.

A `LimitRange` per namespace is the backstop underneath: `64Mi`/`50m` as the
default request and `256Mi`/`500m` as the default limit. Without it, the quota
would make any third-party manifest that declares no resources unschedulable,
and upstream charts would have to be edited before they could be installed.

Raising any row means re-totalling this table and the comment in
`base/quotas.yaml` by hand. Only the manifests are checked mechanically:
`tests/infra/test_k8s_base.py` pins every quantity above against what kustomize
actually emits, and derives the two totals from those same built objects rather
than restating them - so a row raised in isolation fails there. This prose is
not in that loop.

## Still to land

- `platform/` - Postgres, Redis and Qdrant, cert-manager and its issuer, and
  the Traefik middlewares for basic auth, rate limiting and security headers.
- `apps/` - the gateway, the console, llama.cpp, the Helpdesk services, the
  evals CronJob, and the TLS Ingresses for the four subdomains the Terraform
  module publishes.
- NetworkPolicies keyed off exactly these four namespace names, and SOPS-
  encrypted secrets.

Each of those is admitted against the envelope above, so a workload that does
not fit is a conversation about this table rather than a silent edit to it.
