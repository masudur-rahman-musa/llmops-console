# Dashboards

The dashboard JSON is not in this directory. It lives in
[`../../infra/k8s/monitoring/dashboards/`](../../infra/k8s/monitoring/dashboards/),
and Grafana is provisioned from there.

## Why

Kustomize refuses to read a file from above its own root. A
`configMapGenerator` in `infra/k8s/monitoring/` whose `files:` entry pointed at
this directory is rejected with

```
security; file '.../project-obs/dashboards/gateway-overview.json' is not in or below '.../infra/k8s'
```

and a symlink placed under the root is rejected with the same message against
the resolved path. Lifting the restriction means passing
`--load-restrictor LoadRestrictionsNone` to every build — `make k8s-check`,
CI's `k8s` job and the test suite would each have to carry that flag, and a
build that reads arbitrary paths is a weaker check than one that cannot.

Keeping a copy in both places was the other option, and it is two files that a
search-and-replace fixes in one of them.

So the manifests own the JSON, and this file is the pointer. `docs/adr/0016-*`
records the decision in full.

## What is there

| dashboard | shows |
|---|---|
| `gateway-overview.json` | request rate by status, p50/p95 latency, provider errors by code, fallbacks, circuit state |

Cost, residency and redaction, the Helpdesk agent, evals over time and the
platform view are still to land.

## Editing one

Edit the JSON, commit it, and apply the build. Grafana's file provider polls the
mounted directory, so a changed dashboard appears without a pod restart.

Do not edit a dashboard in the Grafana UI and expect it to survive: the provider
is configured with `allowUiUpdates: false`, and the provisioned file is the only
source. Exporting a dashboard from the UI as a starting point is fine — strip
the `__inputs` and `__requires` blocks it adds, and point every panel target at
the datasource uid `prometheus`, which is the uid the provisioned datasource
declares.

Every panel's `expr` must name a metric family the gateway actually registers in
`packages/opsconsole/opsconsole/core/metrics.py`. A typo there renders an empty
panel with no error anywhere, so `tests/infra/test_k8s_grafana.py` resolves
every metric name in every `expr` against that module.
