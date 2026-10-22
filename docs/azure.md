# Azure OpenAI

The Azure path is an adapter and recorded fixtures. There is no Azure-specific
code beyond the adapter, and there is no Azure account behind this repository.

## What is built

`backend/app/providers/azure_openai.py` holds
`AzureOpenAIAdapter`. Azure serves the OpenAI wire shape and differs on three
counts, all of them outside the request body: the key travels as an `api-key`
header rather than a bearer token, the model names a deployment that lives in
the URL path rather than a field in the body, and an `api-version` query
parameter is mandatory on every call.

So the adapter is the shared OpenAI-compatible adapter with exactly two seams
overridden, `_auth_headers` and `_path`. The mappers, the transport, the
timeouts, the retry table and the provider-message truncation are inherited, so
the two wire shapes cannot drift apart while both suites stay green.

The deployment name reaches a URL path segment, so it is checked before any
outbound call is built: a name carrying a `/`, a `..` or a trailing CRLF
addresses something other than the deployment the policy named.

It is tested against recorded fixtures under `backend/tests/providers/`, by
`backend/tests/providers/test_azure_openai.py` and `backend/tests/providers/test_factory.py`.
CI never needs a live Azure key.

## It ships disabled

The shipped provider row in `backend/config/routing.yaml` is `azure_eu`, and it carries
`enabled: false`. The factory builds no Azure adapter from it, which is pinned
by `test_the_shipped_azure_provider_is_disabled_and_builds_nothing` in
`backend/tests/providers/test_factory.py`.

## Enabling it is a configuration change

Turning that row on is the whole of enabling Azure. No code change is needed,
and none lands here.

An operator sets `enabled: true` on the `azure_eu` row in `backend/config/routing.yaml` and
supplies two values in the environment the gateway is handed:

| Variable | What it holds |
|---|---|
| `AZURE_OPENAI_ENDPOINT` | the resource's base URL, named by the row's `endpoint_env` |
| `AZURE_OPENAI_KEY` | the resource key, named by the row's `key_env` |
| `AZURE_OPENAI_API_VERSION` | optional; overrides the pinned `api-version` default |

An Azure row needs both the endpoint and the key, and neither has a default. A
row missing either is dropped with a warning rather than built half-configured,
so a misconfigured resource never becomes a candidate the resolver can pick.

The `api-version` is pinned in the adapter rather than tracked, because the
value decides which fields the resource accepts. Pointing at a resource on
another version is itself a configuration change, through that third variable.

## What is not here

No Azure region appears in the shipped residency policy beyond the `eu` region
the row declares, and no live request has ever been made. Everything above is
what the code does; what an Azure resource does in production is untested here.
