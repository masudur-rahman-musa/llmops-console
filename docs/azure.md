# Azure OpenAI

The Azure path is an adapter and recorded fixtures. There is no Azure code beyond
the adapter, and no Azure account sits behind this repository.

## What is built

`backend/app/providers/azure_openai.py` holds `AzureOpenAIAdapter`. Azure serves
the OpenAI wire shape and differs in three ways, all outside the request body.

- The key travels as an `api-key` header, not a bearer token.
- The model names a deployment, which goes in the URL path, not the body.
- Every call must carry an `api-version` query parameter.

So the adapter is the shared OpenAI-compatible adapter with two methods
overridden, `_auth_headers` and `_path`. The mappers, transport, timeouts, retry
rules and error handling are inherited, so the two wire shapes cannot drift apart
while both test suites stay green.

The deployment name ends up in the URL path, so it is checked before any call is
built. A name with a `/`, a `..` or a trailing CRLF would point at something other
than the deployment the policy named, and is refused.

It is tested against recorded fixtures under `backend/tests/providers/fixtures/azure/`,
by `backend/tests/providers/test_azure_openai.py` and
`backend/tests/providers/test_factory.py`. The fixtures include Azure's content
filter refusals, which `backend/tests/providers/test_refusals.py` checks turn into
a typed error. CI never needs a live Azure key.

## It ships disabled

The shipped folder `backend/config/providers/` has no Azure file, so the gateway
builds no Azure adapter. A file with `enabled: false` builds none either. That is
pinned by `test_a_disabled_azure_provider_file_builds_nothing` in
`backend/tests/providers/test_factory.py`.

## Enabling it is a configuration change

Dropping in one provider file is the whole job. No code change is needed. Add
`azure.yaml` to `backend/config/providers/`:

    provider: azure
    kind: azure_openai
    endpoint_env: AZURE_OPENAI_ENDPOINT
    key_env: AZURE_OPENAI_KEY
    enabled: true
    models:
      # the DEPLOYMENT name on your resource, not a vendor model id
      gpt-5-mini:
        prices:
          - effective_from: 2025-08-07
            input_usd_per_1m: "0.25"
            output_usd_per_1m: "2.00"

Then name it in an alias in `backend/config/routing.yaml`, and give the gateway
these values in `backend/.env`:

| Variable | What it holds |
|---|---|
| `AZURE_OPENAI_ENDPOINT` | the resource's base URL, such as `https://<resource>.openai.azure.com`, named by the file's `endpoint_env` |
| `AZURE_OPENAI_KEY` | the resource key, named by the file's `key_env` |
| `AZURE_OPENAI_API_VERSION` | optional, overrides the pinned default `2024-10-21` |

An Azure provider needs both the endpoint and the key, and neither has a default.
A provider missing either is left out with a warning, so a half-set resource never
becomes a candidate the resolver can pick. A file that names both an
`endpoint_env` and a `base_url` is left out too, because two possible hosts is how
a key reaches the wrong one.

The `api-version` is pinned in the adapter, because it decides which fields the
resource accepts. Pointing at a resource on another version is a configuration
change too, through the third variable.

## What is not here

No live request has ever been made to an Azure resource. Everything above is what
the code does. How a real Azure resource behaves under load is untested here.
