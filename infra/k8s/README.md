# `infra/k8s`

The manifests for the k3s cluster, built with kustomize. Nothing here is
applied by CI or by any tool in this repo: `kustomize build infra/k8s |
kubeconform` is the whole automated check, and `kubectl apply` is a human's.

```
kustomization.yaml     the single root; `resources` grows as directories land
base/                  namespaces, quotas, limit ranges, network policies
platform/              Postgres, Redis, Qdrant, the ACME issuers, the middlewares
apps/                  the gateway, llama.cpp, the Helpdesk, their config and the four public Ingresses
monitoring/            Prometheus, node-exporter, kube-state-metrics, Grafana and the `grafana.` Ingress
gpu/                   vLLM on one GPU host; standalone, not in the root build (`make k8s-gpu-check`)
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
| `postgres-backup`, `pg-dump` init + `rclone-sync` (64Mi/50m/96Mi each) | 128Mi | 100m | 192Mi |
| `qdrant-backup`, `qdrant-snapshot` init (32Mi/25m/64Mi) + `rclone-sync` (64Mi/50m/96Mi) | 96Mi | 75m | 160Mi |
| **total** | **1600Mi** | **705m** | **3040Mi** |

`tests/infra/test_k8s_platform.py` derives those three totals from the built pod
templates plus the same reserve and checks them against the `platform` quota, so
a row raised in isolation fails there rather than at `helm install`. Row 23d's two
CronJobs leave the namespace 448Mi / 95m / 32Mi of headroom - the limits column is
now the tight one.

## The apps budget

What `apps/` spends of the 3Gi / 1500m / 7Gi the envelope gives it. The gateway
is priced at its HPA CEILING rather than at its two resting replicas: a quota
sized to the floor admits the cluster happily and then rejects the first
scale-up at admission, which is the one event an autoscaler exists to serve.

| workload | requests.memory | requests.cpu | limits.memory |
|---|---|---|---|
| `gateway`, 3 pods at HPA max (256Mi / 150m / 512Mi) | 768Mi | 450m | 1536Mi |
| `llama`, 1 pod | 1024Mi | 300m | 3072Mi |
| `llama-embed`, 1 pod | 256Mi | 100m | 512Mi |
| `helpdesk-agent-api`, 1 pod | 192Mi | 100m | 384Mi |
| `helpdesk-mcp-server`, 1 pod | 128Mi | 75m | 256Mi |
| `helpdesk-web`, 1 pod | 128Mi | 75m | 256Mi |
| `helpdesk-scheduler`, 1 pod | 128Mi | 50m | 256Mi |
| `console`, 1 pod | 256Mi | 100m | 512Mi |
| **total** | **2880Mi** | **1250m** | **6784Mi** |

`tests/infra/test_k8s_apps.py` derives those three totals the same way - the
built pod templates, the gateway multiplied by the `maxReplicas` of the HPA in
the same build - and checks them against the `apps` quota. llama.cpp and the
console each moved from a reserve row to the built side when their manifests
landed, and the totals did not move: the manifest was written to the
reserve, not the reserve rewritten to the manifest.

The embedding server is the one row that DID move them. No reserve was ever held
for it - it exists because one llama.cpp process serves one model, so the
`embed` alias needs a second address - and pretending otherwise by carving its
256Mi out of the chat model's row would be shrinking that model to fit the
table.

The four Helpdesk rows divide the envelope that was held for three. The reserve
priced three pods at 192Mi / 100m / 384Mi and the service list turned out to be
four, so the four rows above were written to sum to exactly the same 576Mi,
300m and 1152Mi rather than the quota being raised to hold a fourth pod at the
reserve's own size - which would have put the requests column at exactly 3Gi
against a 3Gi hard limit, no headroom at all, and left the console with nowhere
to land. What it costs is honest: `helpdesk-agent-api` keeps the reserve's own
row and is the only Helpdesk pod sized for real traffic, while the other three
run in 128Mi with a 256Mi ceiling. If one of them needs more, the conversation
is this table and not a quiet edit - and the direction of the error is visible,
because an out-of-memory kill on `helpdesk-web` is a CrashLoopBackOff rather
than a silent degradation.

With the console built there is no reserve left: it was the last workload this
envelope holds, and a new one is a conversation about this table.

The limits column is why that quota holds 7Gi rather than the 5Gi the envelope
first gave it: llama.cpp is sized for 3 GiB on its own, and trimming its ceiling
to fit would be shrinking the model to fit the table. The requests column, the
one the scheduler actually packs, did not move.

## The monitoring budget

What `monitoring/` spends of the 512Mi / 400m / 1Gi the envelope gives it. One
row per pod; `node-exporter` is a DaemonSet and this cluster is one node, so it
is one pod here and would be one per node on a larger one.

| workload | requests.memory | requests.cpu | limits.memory |
|---|---|---|---|
| `prometheus`, 1 pod | 256Mi | 150m | 512Mi |
| `node-exporter`, 1 pod per node | 32Mi | 50m | 64Mi |
| `kube-state-metrics`, 1 pod | 64Mi | 50m | 128Mi |
| `grafana`, 1 pod | 128Mi | 100m | 256Mi |
| `alertmanager`, 1 pod | 32Mi | 50m | 64Mi |
| **total** | **512Mi** | **400m** | **1024Mi** |

The total now reads the envelope exactly, with nothing left over: Alertmanager is the exact remainder the four rows above it left.

`tests/infra/test_k8s_grafana.py` derives those three totals from the built pod
templates and checks them against the `monitoring` quota, the same way the two
budgets above are checked; `tests/infra/test_k8s_monitoring.py` holds the three
scrape rows and `tests/infra/test_k8s_alerting.py` holds Alertmanager's own row
against the values they are priced at here.

What used to be left - 32Mi of requests, 50m of CPU and 64Mi of the limits
ceiling - is now Alertmanager's own row, spent to the byte. There is
no slack left in this namespace at all, so a future Alertmanager change that
needs more is a conversation about this table rather than a quiet edit here -
which is the direction the test enforces, because it asserts the total as an
exact equality against the quota and not as a `<=`.

There is no operator and no CustomResourceDefinition: this stack is written by
hand rather than installed from `kube-prometheus-stack`, because the whole build
has to pass `kubeconform -strict` offline and a custom resource has no stock
schema. Prometheus scrapes five static targets by Service DNS and
holds no RBAC at all; `kube-state-metrics` is the one subject in the repo with a
ClusterRole, and its verbs are `get`, `list` and `watch`.

## The public hosts

Five Ingresses, three hosts. Two of the hosts carry a PAIR, because a Traefik
middleware annotation is per Ingress and not per path: one Ingress cannot exempt
`/healthz` from basic auth, and `/healthz` behind a 401 is an uptime monitor
reporting an outage that is not happening. The third, `grafana.`, is a single
Ingress — it is not behind basic auth at all, so there is nothing to carve out.

| host | Ingress | path | backend Service | middlewares | TLS Secret |
|---|---|---|---|---|---|
| `api.example.com` | `gateway` | `/v1` (Prefix) | `gateway` | basic-auth, rate-limit, security-headers | `gateway-tls` |
| `api.example.com` | `gateway-healthz` | `/healthz` (Exact) | `gateway` | rate-limit, security-headers | `gateway-healthz-tls` |
| `helpdesk.example.com` | `helpdesk` | `/` (Prefix) | `helpdesk-web` | basic-auth, rate-limit, security-headers | `helpdesk-tls` |
| `helpdesk.example.com` | `helpdesk-healthz` | `/healthz` (Exact) | `helpdesk-web` | rate-limit, security-headers | `helpdesk-healthz-tls` |
| `console.example.com` | `console` | `/` (Prefix) | `console` | basic-auth, rate-limit, security-headers | `console-tls` |
| `console.example.com` | `console-healthz` | `/healthz` (Exact) | `console` | rate-limit, security-headers | `console-healthz-tls` |
| `console.example.com` | `console-login` | `/api/login` (Exact) | `console` | basic-auth, console-login-rate-limit, security-headers | `console-login-tls` |
| `grafana.example.com` | `grafana` | `/` (Prefix) | `grafana` | rate-limit, security-headers | `grafana-tls` |

`grafana.` is the one public host with no basic auth in front of it, and the
reason is specific rather than an oversight. Grafana authenticates its own
callers against the `grafana-admin` Secret below, so the host is not open; what
basic auth would add is a second credential to rotate in front of the first. It
would also sit in front of `/api/annotations`, which CD POSTs a release marker to
with a Bearer token — a 401 there breaks the deploy for a credential Grafana is
already asking for. The trade is that this host's access control is the
application's rather than the proxy's, and that the rate limit and the security
headers are therefore the only middlewares doing anything on it.

`Exact` on the `/healthz` rows is what keeps the split honest: `Prefix` there
would put everything under `/healthz/...` outside basic auth. On the Helpdesk
host it also settles an overlap - Traefik orders routers by rule length, so the
exact path outranks its neighbour's `PathPrefix(/)`. On the gateway host the two
routers no longer overlap at all.

`helpdesk-web` is the only public Helpdesk surface. The agent API is the shape
a project rule rejects for the console, and the MCP server is a tool endpoint with
no host - both are reachable in-cluster by their ClusterIP Services and from
nowhere else.

The gateway host publishes `/v1` and `/healthz` and nothing else. `/metrics`,
`/admin/routing`, `/admin/routing/reload` and `/admin/directory` are matched by
no router, so they are not served publicly at all rather than served behind
basic auth: a path with no router gets Traefik's own 404, which discloses less
than a 403 does. The admin key still guards the three admin routes for anything
that reaches the pod another way - the router rule removed a surface, it did not
replace a control. `/readyz` is published nowhere either, because its body names
which dependencies are reachable and the kubelet dials the pod directly. Every
path the app mounts still answers on `Service/gateway` in-cluster, which is what
"cluster-internal" means here and how the Prometheus in `monitoring/` scrapes
`/metrics`: a scrape target in its config and a `monitoring` peer on
`gateway-ingress`, neither of them a router.

The console reads `console-api` in-cluster, through `Service/gateway`, so those
routes need no router on `api.`. A future public surface that is not under `/v1`
needs a router of its own on this table, or it is a 404 on a host that is up.

Every middleware reference is namespace-qualified — `apps-<name>@kubernetescrd`
on the seven `apps` routers, `monitoring-<name>@kubernetescrd` on the Grafana one
— and each resolves to that namespace's own copy of the middleware.
Traefik treats a reference it cannot resolve as no middleware at all rather than
as an error, so a `platform-` prefix here, or an `apps-` prefix copied onto the
Grafana row, would serve unprotected traffic silently;
`tests/infra/test_k8s_apps.py` and `tests/infra/test_k8s_grafana.py` each
resolve every parsed reference against the `Middleware`s the build actually
emits.

The rate limit is instantiated PER ROUTER, so these five do not share one
budget: each gets the middleware's own 50/s average and 100 burst. Nobody has
measured a browser page load against that number, and a Next.js page fanning out
past the burst takes a 429 on assets, which looks like a broken page rather than
like a rate limit. The fix, if it ever bites, is a second middleware and a row
here rather than a value quietly widened.

Both `/healthz` paths are a promise the pod behind them makes. The gateway
serves `/healthz` from this repo's code; `helpdesk-web`'s is the companion
repo's, dialled by its own liveness probe, and if it is never served the one
router exempt from basic auth is a public 404 - an exposed path that answers
nothing rather than one that answers too much.

## What may reach what

Twenty-one `NetworkPolicy` objects in `base/networkpolicies.yaml`: ten in
`apps`, three in `platform` and eight in `monitoring`. `apps` and `monitoring` are denied
in both directions and every row below is an additive allow on top of that;
`platform` is denied inbound only, because
cert-manager lives there and a cluster that cannot reach the ACME endpoint is a
cluster with no certificate on any of the four public hosts. What that costs is
that a compromised data pod can still talk out. The namespace holding the
provider keys is the one closed both ways.

| policy | namespace | applies to | allows |
|---|---|---|---|
| `default-deny` | `apps` | every pod | nothing, in either direction |
| `allow-dns` | `apps` | every pod | egress to `kube-system` on UDP 53 and TCP 53 |
| `gateway-ingress` | `apps` | `gateway` | ingress from `kube-system` (Traefik), from any pod in `apps` and from any pod in `monitoring`, on its serving port |
| `gateway-egress` | `apps` | `gateway` | egress to the public internet on TCP 443, to llama 8082, llama-embed 8083, to Postgres, Redis and Qdrant in `platform`, and to `prometheus:9090` in `monitoring` |
| `helpdesk-web-ingress` | `apps` | `helpdesk-web` | ingress from `kube-system` on its serving port |
| `helpdesk-api-ingress` | `apps` | `helpdesk-agent-api`, `helpdesk-mcp-server` | ingress from the four Helpdesk pods, each on its own serving port |
| `helpdesk-egress` | `apps` | the four Helpdesk pods | egress to `gateway:8080`, `helpdesk-agent-api:8000`, `helpdesk-mcp-server:8081`, `postgres:5432`, `redis:6379`, `qdrant:6333` - and no route off the cluster at all |
| `llama-ingress` | `apps` | `llama`, `llama-embed` | ingress from `gateway`, each on its own serving port |
| `console-ingress` | `apps` | `console` | ingress from `kube-system` (Traefik) on its serving port, and from nowhere else |
| `console-egress` | `apps` | `console` | egress to `gateway:8080` - and no route off the cluster at all |
| `default-deny-ingress` | `platform` | every pod | nothing inbound; egress is not constrained |
| `data-ingress-from-apps` | `platform` | `postgres`, `redis`, `qdrant` | ingress from any pod in `apps` on 5432, 6379 and 6333 |
| `metrics-ingress-from-monitoring` | `platform` | `qdrant` | ingress from any pod in `monitoring` on 6333 |
| `default-deny` | `monitoring` | every pod | nothing, in either direction |
| `allow-dns` | `monitoring` | every pod | egress to `kube-system` on UDP 53 and TCP 53 |
| `metrics-ingress-from-monitoring` | `monitoring` | every pod with a `metrics` port | ingress from any pod in `monitoring`, each on its own `metrics` port |
| `prometheus-egress` | `monitoring` | `prometheus` | egress to `gateway:8080`, `qdrant:6333`, the two exporters on their `metrics` ports, `alertmanager:9093` and cert-manager on 9402 - and no other route off the cluster |
| `prometheus-ingress-from-gateway` | `monitoring` | `prometheus` | ingress from `gateway` in `apps` on its `metrics` port (9090), and from no other `apps` pod |
| `grafana-ingress` | `monitoring` | `grafana` | ingress from `kube-system` (Traefik) on its serving port, and from nowhere else |
| `grafana-egress` | `monitoring` | `grafana` | egress to `prometheus:9090` - and no route off the cluster at all |
| `alertmanager-egress` | `monitoring` | `alertmanager` | egress to the public internet on TCP 443 only |

The gateway and Alertmanager are the only two pods in the cluster with a route
to the internet, and that is the point of the whole file:
the Helpdesk holds no provider key, and with no `ipBlock` in any policy that
selects it, even a bug in the companion repo cannot dial a provider directly.
The Langfuse pair in `helpdesk-secrets` is dead config for the same reason,
until an in-cluster collector relays it.

Both public destinations are `0.0.0.0/0` minus `10.0.0.0/8`, `172.16.0.0/12`,
`192.168.0.0/16` and `169.254.0.0/16`, on TCP 443 only - `alertmanager-egress`
copies the gateway's excluded ranges exactly rather than re-deriving them. The
link-local carve-out is the one worth naming: it closes `169.254.169.254`, the
metadata endpoint an SSRF through a provider `base_url`, or a leaked credential
inside Alertmanager's SMTP/Telegram relay, would otherwise reach.

Qdrant's scrape is a separate `platform` policy rather than a `monitoring` peer
added to `data-ingress-from-apps`: that policy's guarantee is that the data
services admit `apps` and nothing else, and widening it would spend the
guarantee to save a document. Postgres and Redis are scraped by nothing until an
exporter for each lands, so the platform dashboard, whenever it lands, arrives
with two empty panels.

Every row above that reads `on its serving port`, `on its own serving port` or
`on its own metrics port` is written by NAME rather than by number. A policy has
one `podSelector` and its ports apply to every pod that selector matches, so
`llama-ingress` - one policy over two pods listening on 8082 and 8083 - cannot
say "each on its own port" with a literal. Both pods call their port `http`, and
Kubernetes resolves a named port against each target pod's own container port.
`platform`'s policy is the exception and reads in numbers: its three targets
name their ports differently.

The same mechanism is why Grafana has two policies of its own. A named port that
does not resolve on a pod is no allowance for that pod, and Grafana's port is
named `http` rather than `metrics`, because an Ingress backend points at it and
it serves a browser. So `metrics-ingress-from-monitoring` - written to cover
three scrape targets on three different numbers with one rule - reaches Grafana
not at all, and `grafana-ingress` is what admits Traefik to it. That is a real
narrowing of one rule's reach, stated here rather than discovered later: a fifth
pod added to this namespace is covered by that rule only if it publishes a
`metrics` port.

Two things these policies do NOT do. They cannot see a URL path: what takes
`/admin/*` and `/metrics` off the public side is the `gateway` Ingress routing
`/v1` and nothing wider, a router rule rather than a policy. These policies
decide only which pods may open a connection, and Traefik is one of them. And
`evals` still has no policies at all, so the nightly CronJob's allows land with
that row.

Nothing here has ever been applied, so none of it has ever denied a packet. What
the tests prove is shape, by equality in both directions, so an allow nobody
agreed to is as red as a missing one; the live proof is an egress probe against
a kind cluster, and it lands with the end-to-end tests.

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

The ten Secrets nothing here writes, each supplied SOPS-encrypted before the
first apply. `../secrets/` holds one plaintext contract per Secret
(`*.example.yaml`, every value `REPLACE_ME`), the workflow that encrypts a filled
copy, and the key procedure; `../../.sops.yaml` is the policy that decides what
gets encrypted and to whom. The key names below and the ones in those templates
are asserted equal, and equal to the names the build references, by
`tests/infra/test_secrets.py`:

| Secret | namespace | keys | read by |
|---|---|---|---|
| `postgres-credentials` | `platform` | `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` | the Postgres StatefulSet, by `envFrom` |
| `traefik-basic-auth` | `platform`, `apps` | `users` (htpasswd lines) | the `basic-auth` middleware in each namespace |
| `gateway-secrets` | `apps` | `OPSCONSOLE_DATABASE_URL`, `OPSCONSOLE_API_KEYS`, `OPSCONSOLE_ADMIN_KEY_SHA256`, `OPSCONSOLE_BUDGET_OVERRIDE_SECRET`, `OPSCONSOLE_OTLP_ENDPOINT`, `OPSCONSOLE_OTLP_HEADERS`, `ANTHROPIC_API_KEY`, `ANTHROPIC_EU_API_KEY`, `OPENAI_API_KEY`, `AZURE_OPENAI_KEY`, `AZURE_OPENAI_ENDPOINT` | the gateway Deployment, by `envFrom` |
| `helpdesk-secrets` | `apps` | `DATABASE_URL`, `APP_DATABASE_URL`, `APP_DB_PASSWORD`, `GATEWAY_API_KEYS`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`, `JWT_SECRET` | the four Helpdesk Deployments, by `envFrom` |
| `ghcr-pull` | `apps` | `.dockerconfigjson` | the four Helpdesk Deployments, by `imagePullSecrets` |
| `console-secrets` | `apps` | `CONSOLE_PASSWORD_SHA256`, `CONSOLE_SESSION_SECRET`, `CONSOLE_ADMIN_KEY` | the console Deployment, by `envFrom`; `CONSOLE_ADMIN_KEY` must hash to `gateway-secrets`' `OPSCONSOLE_ADMIN_KEY_SHA256` |
| `grafana-admin` | `monitoring` | `GF_SECURITY_ADMIN_USER`, `GF_SECURITY_ADMIN_PASSWORD` | the Grafana Deployment, by `envFrom` |
| `alertmanager-secrets` | `monitoring` | `SMTP_SMARTHOST`, `SMTP_FROM`, `SMTP_AUTH_USERNAME`, `SMTP_AUTH_PASSWORD`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | the Alertmanager Deployment, by `envFrom` |
| `backup-secrets` | `platform` | `RCLONE_CONFIG_HETZNER_ACCESS_KEY_ID`, `RCLONE_CONFIG_HETZNER_SECRET_ACCESS_KEY`, `RCLONE_CONFIG_HETZNER_ENDPOINT`, `BACKUP_BUCKET` | both backup CronJobs' `rclone-sync` container, by `envFrom` |
| `evals-secrets` | `evals` | `OPSCONSOLE_DATABASE_URL` | the nightly-eval CronJob, by `envFrom` |

`traefik-basic-auth` is supplied once per routing namespace, not once per
cluster: a `Middleware` reads its Secret from its own namespace, and a design note is
why each namespace carries its own copy of the middlewares it uses. The five TLS
Secrets the five Ingresses name are not in this table - cert-manager writes
those, and their names have to stay disjoint from the ten above, or an issued
certificate overwrites a set of credentials. `grafana-tls` and `grafana-admin`
are the pair where that is easiest to get wrong, because they sit in one
namespace and share a prefix; the disjointness is asserted.

Three of those key names are endpoints rather than credentials
(`OPSCONSOLE_OTLP_ENDPOINT`, `AZURE_OPENAI_ENDPOINT`, `LANGFUSE_HOST`).
A pod reads exactly two sources, its ConfigMap and its Secret, and each of these
three travels with a credential it would be useless apart from, so they sit with
the credential rather than in the ConfigMap.

Until they exist an apply fails on the missing Secret, which is the point: a
placeholder with a password in git would come up working and never mention it
again. Nothing under `../secrets/` is a kustomize input: no kustomization lists a
path there, the build emits no `Secret` at all, and both are asserted.

The model weights are the third thing this repo does not supply. No GGUF is
committed, no image carries one and no container downloads one: an unverified
download URL with a checksum nobody has ever compared anything to is worse than
an absence that says so. The `llama-models` claim is filled once, by hand, with
the same file `docker-compose.yml` loads locally - `qwen2.5-3b-instruct-q4_k_m.gguf`,
the upstream Q4_K_M build. A 3B rather than a 7B because the pod's ceiling is
3 GiB and a 7B Q4_K_M is ~4.4 GB before its KV cache.

Both llama.cpp claims are filled this way, and each has its own: a
`ReadWriteOnce` volume mounted by two Deployments would pin both pods to one
node forever.

```
kubectl -n apps scale deployment/llama --replicas=0
kubectl -n apps run llama-load --image=busybox:1.36 --restart=Never \
  --overrides='{"spec":{"containers":[{"name":"llama-load","image":"busybox:1.36",
  "command":["sleep","3600"],"volumeMounts":[{"name":"models","mountPath":"/models"}]}],
  "volumes":[{"name":"models","persistentVolumeClaim":{"claimName":"llama-models"}}]}}'
kubectl -n apps cp qwen2.5-3b-instruct-q4_k_m.gguf llama-load:/models/
kubectl -n apps delete pod llama-load
kubectl -n apps scale deployment/llama --replicas=1
```

The scale to zero is not optional: the claim is `ReadWriteOnce`, so the helper
pod cannot mount it while the llama.cpp pod holds it. Until the file is there
the pod fails its startup probe for 600 seconds and then restarts, which is a
visibly unready workload rather than a silent one.

The embedding claim is filled by the same five commands with three names
changed - `llama-embed`, `llama-embed-models` and
`bge-small-en-v1.5-q8_0.gguf`, the upstream Q8_0 build of a 33M-parameter
encoder, where quantising harder buys nothing and costs vector quality. Its
startup ceiling is 120 seconds rather than 600: the file is ~130 MB.

## What the Helpdesk reads

These are the companion repo's published names: bare, unprefixed, one name per
value. `helpdesk-config` carries these nine keys and the four Deployments take
the whole object by `envFrom`. A name that drifts on either side leaves the pods
starting on their localhost defaults rather than failing loudly, so a change to
that repo's contract is a change here.

| key | value | why |
|---|---|---|
| `GATEWAY_URL` | `http://gateway.apps.svc.cluster.local:8080/v1` | every model call goes through this gateway, so the Helpdesk never holds a provider key |
| `REDIS_URL` | `redis://redis.platform.svc.cluster.local:6379/1` | database `1`, not the gateway's `0`: one Redis, two tenants of it, and a shared index is one `FLUSHDB` from taking out the other repo's state |
| `QDRANT_URL` | `http://qdrant.platform.svc.cluster.local:6333` | the same vector store the gateway's semantic cache uses, in its own collections |
| `ENV` | `production` | what the pods report about themselves |
| `MCP_URL` | `http://helpdesk-mcp-server.apps.svc.cluster.local:8081/mcp` | the agent API's tool server, on its Service's port |
| `AGENT_API_URL` | `http://helpdesk-agent-api.apps.svc.cluster.local:8000` | the web pod calls the agent API from its server side only, never from the browser |
| `WEB_ORIGIN` | `https://helpdesk.example.com` | the public origin, the host of Ingress `helpdesk`; scheme and host only, no trailing slash |
| `SESSION_COOKIE_SECURE` | `"true"` | the web pod's session cookie is sent over HTTPS only |
| `POLICY_DIR` | `/srv/policies` | the corpus root the MCP server walks; its image carries no corpus and expects a read-only mount here |

Every in-cluster host is fully qualified because these pods are in `apps` and
two of the services are in `platform`, where a bare `redis` resolves to nothing.
`WEB_ORIGIN` is the one public address.

Bare names share a namespace with every other `envFrom` in a pod, so only the
four Helpdesk Deployments mount `helpdesk-config` and `helpdesk-secrets`; a
gateway pod reading this `REDIS_URL` would silently change database. The
Helpdesk images are private GHCR packages, pulled with `ghcr-pull`, which only
those four pod specs name.

`OTEL_SERVICE_NAME` is deliberately NOT in that ConfigMap. It is a
per-container `env` entry on each of the four, set to that workload's own name,
because one trace carries spans from the Helpdesk and from the gateway and a
shared service name would collapse five processes into one in the trace viewer.

The scheduler has no liveness signal and that is a decision, not an omission: it
serves no port, so a health probe would have to be a command no code in this
repo defines, and a liveness probe against a command that does not exist
restarts the pod forever. A wedged scheduler is therefore visible only as an
object in kube-state-metrics. Row 23c's `PodCrashLooping` alert pages on a
container that keeps RESTARTING; a scheduler that hangs without crashing trips
no alert at all, since nothing here reads a liveness signal it does not have -
still a gap, not one this row closes.

## Still to land

- The nightly eval CronJob in `evals/`.
- `monitoring/` carries Grafana with all six dashboards and
  Alertmanager with the rules in `../../project-obs/alerts/` - the
  namespace's envelope is now spent exactly, to the byte. Postgres and Redis
  exporters are not planned yet and are what the platform dashboard's two empty
  panels are waiting on.
- The `../secrets/*.enc.yaml` themselves, one per contract in the table above.
  The policy, the contracts and the guard are in place; the ciphertext waits on
  an age key being generated, which is one pass with `sops` installed.
- NetworkPolicies for `evals`; the other three namespaces are covered above.

Each of those is admitted against the envelope above, so a workload that does
not fit is a conversation about this table rather than a silent edit to it.
