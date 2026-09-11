# `infra/k8s`

The manifests for the k3s cluster, built with kustomize. Nothing here is
applied by CI or by any tool in this repo: `kustomize build infra/k8s |
kubeconform` is the whole automated check, and `kubectl apply` is a human's.

```
kustomization.yaml     the single root; `resources` grows as directories land
base/                  namespaces, quotas, limit ranges
platform/              Postgres, Redis, Qdrant, the ACME issuers, the middlewares
apps/                  the gateway, its config and the two `api.` Ingresses
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
| `apps` | 3Gi | 1500m | 7Gi | the gateway, the console, llama.cpp, the Helpdesk services |
| `evals` | 512Mi | 300m | 1Gi | the nightly eval CronJob |
| `monitoring` | 512Mi | 400m | 1Gi | Prometheus, Grafana, Alertmanager |
| **total** | **6Gi** | **3** | 12Gi | |

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
than restating them - so a row raised in isolation fails there.

Two rows of prose below ARE in that loop, because a table nobody checks is a
table that quietly stops being true. `tests/infra/test_k8s_platform.py` reads
the cert-manager row of the platform budget, which has no manifest under it to
check instead; `tests/infra/test_k8s_apps.py` reads the `apps` row of the table
above, whose limits column is the one a workload still to land depends on.
Everything else here is prose.

## The platform budget

What `platform/` spends of the 2Gi / 800m / 3Gi the table above gives it. Every
row is one pod; cert-manager is three pods this repo does not write a manifest
for (see below), held as a reserve rather than discovered at install time.

| workload | requests.memory | requests.cpu | limits.memory |
|---|---|---|---|
| `postgres` | 512Mi | 200m | 1Gi |
| `redis` | 256Mi | 100m | 512Mi |
| `qdrant` | 512Mi | 200m | 768Mi |
| cert-manager, 3 pods at 32Mi / 10m / 128Mi each | 96Mi | 30m | 384Mi |
| **total** | **1376Mi** | **530m** | **2688Mi** |

`tests/infra/test_k8s_platform.py` derives those three totals from the built pod
templates plus the same reserve and checks them against the `platform` quota, so
a row raised in isolation fails there rather than at `helm install`.

## The apps budget

What `apps/` spends of the 3Gi / 1500m / 7Gi the envelope gives it. The gateway
is priced at its HPA CEILING rather than at its two resting replicas: a quota
sized to the floor admits the cluster happily and then rejects the first
scale-up at admission, which is the one event an autoscaler exists to serve. The
other three rows are workloads this directory does not write yet, held as named
reserves so the namespace is not re-planned when each lands.

| workload | requests.memory | requests.cpu | limits.memory |
|---|---|---|---|
| `gateway`, 3 pods at HPA max (256Mi / 150m / 512Mi) | 768Mi | 450m | 1536Mi |
| llama.cpp reserve, 1 pod | 1024Mi | 300m | 3072Mi |
| Helpdesk reserve, 3 pods at 192Mi / 100m / 384Mi | 576Mi | 300m | 1152Mi |
| console reserve, 1 pod | 256Mi | 100m | 512Mi |
| **total** | **2624Mi** | **1150m** | **6272Mi** |

`tests/infra/test_k8s_apps.py` derives those three totals the same way - the
built pod templates, the gateway multiplied by the `maxReplicas` of the HPA in
the same build, plus the three reserves - and checks them against the `apps`
quota.

The limits column is why that quota holds 7Gi rather than the 5Gi the envelope
first gave it: llama.cpp is sized for 3 GiB on its own, and trimming its ceiling
to fit would be shrinking the model to fit the table. The requests column, the
one the scheduler actually packs, did not move.

## The human steps

Neither of these is in any gate: `kubectl apply` and `helm install` are a
human's, and this repo only prepares and validates.

cert-manager is installed from its chart into `platform`, not from the upstream
static manifest. The manifest creates its own `cert-manager` namespace, which
would sit outside the four namespaces above with no quota and no LimitRange on
it - so the one component issuing our certificates would be the one thing
without requests and limits. The chart takes both:

```
helm repo add jetstack https://charts.jetstack.io && helm repo update
helm install cert-manager jetstack/cert-manager \
  --version v1.16.2 --namespace platform \
  --set crds.enabled=true \
  --set resources.requests.memory=32Mi \
  --set resources.requests.cpu=10m \
  --set resources.limits.memory=128Mi \
  --set webhook.resources.requests.memory=32Mi \
  --set webhook.resources.requests.cpu=10m \
  --set webhook.resources.limits.memory=128Mi \
  --set cainjector.resources.requests.memory=32Mi \
  --set cainjector.resources.requests.cpu=10m \
  --set cainjector.resources.limits.memory=128Mi
```

If the chart's real footprint is larger than that, the quota refuses the install
with the namespace named, which is the intended direction: raising it means
re-totalling the table above and the reserve constant in the test together.

The three Secrets nothing here writes, each supplied SOPS-encrypted before the
first apply:

| Secret | namespace | keys | read by |
|---|---|---|---|
| `postgres-credentials` | `platform` | `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` | the Postgres StatefulSet, by `envFrom` |
| `traefik-basic-auth` | `platform`, `apps` | `users` (htpasswd lines) | the `basic-auth` middleware in each namespace |
| `gateway-secrets` | `apps` | the database URL, the provider keys `routing.yaml` reads by `key_env`, the Langfuse pair, the tenant key digests | the gateway Deployment, by `envFrom` |

`traefik-basic-auth` is supplied once per routing namespace, not once per
cluster: a `Middleware` reads its Secret from its own namespace, and a design note is
why each namespace carries its own copy of the three middlewares. The TLS
Secrets the Ingresses name are not in this table - cert-manager writes those.

Until they exist an apply fails on the missing Secret, which is the point: a
placeholder with a password in git would come up working and never mention it
again.

## Still to land

- The rest of `apps/` - llama.cpp, the Helpdesk services and the console, with
  the `helpdesk.` and `console.` Ingresses.
- The nightly eval CronJob in `evals/`, and the metrics stack in `monitoring/`
  with its `grafana.` Ingress.
- NetworkPolicies keyed off exactly these four namespace names, and SOPS-
  encrypted secrets, including the three named above.

Each of those is admitted against the envelope above, so a workload that does
not fit is a conversation about this table rather than a silent edit to it.
