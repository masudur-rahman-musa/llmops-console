r"""The Azure OpenAI adapter: the same wire as the module next door, addressed differently.

Azure serves the OpenAI shape and differs on three counts, all of them outside
the body: the key is an `api-key` header rather than a bearer token, the model
names a DEPLOYMENT that lives in the URL path rather than a field in the body,
and an `api-version` query parameter is mandatory on every call. So
`AzureOpenAIAdapter` is `OpenAICompatibleAdapter` with exactly two seams
overridden - `_auth_headers` and `_path` - and nothing else. `chat`, `embed`, `stream_chat`, both
mappers, the transport,
the fixed timeouts, the retry table and the provider-message truncation are all inherited,
so the two wire shapes cannot drift apart while both suites stay green.

The body still carries `model`. Azure ignores it, and stripping it would mean a
second mapper for one dead field.

The deployment name reaches a URL PATH SEGMENT, so it is pre-checked here by
`checked_deployment` before any outbound call is built. A path has no length backstop and no
character
backstop of its own, and a name carrying a `/`, a `..` or a trailing CRLF
addresses something other than the deployment the policy named.

Cost of that check: on this wire the value comes from the
POLICY rather than from the caller, so a misconfigured deployment is reported to
the caller as a 422 naming `model` - an operator's error wearing a caller's
status code. The alternative, a non-retryable `ProviderError`, is a 502 the
routes' fallback loop treats as final anyway, and it says less about what to fix.

This module logs nothing, and no key is read from the environment here: the
factory injects it (`providers/factory.py`).
"""

from __future__ import annotations

import random
import re
import time
from collections.abc import Callable

import httpx

from app.core.validation import IdentifierInvalid
from app.providers.openai_compatible import OpenAICompatibleAdapter

AZURE_API_VERSION = "2024-10-21"
"""The api-version every request carries, pinned exactly rather than tracked.

The value decides which fields the resource accepts, so a preview string arriving
in a refactor changes the wire for every deployment at once. A resource pinned to
another version is a CONFIG change: `AZURE_OPENAI_API_VERSION` in the environment
overrides this, read by `providers/factory.py`."""

AZURE_API_VERSION_ENV = "AZURE_OPENAI_API_VERSION"
"""The variable holding that override, read by `providers/factory.py` out of the
environment it was injected rather than from `os.environ`. Unlike a
provider's `key_env` and `endpoint_env` this name is NOT policy-configurable: an
api-version is a property of the resource an operator points at, and one name for
it is one place to look."""

AZURE_DEPLOYMENT_PATH = "/openai/deployments"
"""The prefix the deployment segment hangs off. The whole URL, query included, is
the same shape on all three request paths: chat, embeddings and streamed chat."""

MAX_DEPLOYMENT_NAME_LENGTH = 64
AZURE_DEPLOYMENT_PATTERN = re.compile(r"[a-zA-Z0-9_-]{1,64}")
"""What Azure allows in a deployment name, and the cap is the vendor's rather than
ours - which is why this is not `checked_tool_name`, whose cap is 128.
We use `fullmatch` because in Python `$` also matches before a trailing
newline, and here a trailing newline in a URL path segment is a request-splitting
attempt rather than a typo. A name of exactly 64 characters is accepted and 65 is
rejected."""


def checked_deployment(value: str, field: str) -> str:
    r"""Validate one deployment name before it becomes a URL path segment.

    Raises `IdentifierInvalid` (code VALIDATION, status 422) naming `field`, and
    raises it before the outbound request is built, so no request is made for a
    bad name.

    The same shape as `providers/base.py::checked_tool_name` and deliberately not
    the same function: that one's cap is 128 and its subject is a name the MODEL
    calls back, while this one's cap is the vendor's and its subject is a path
    segment. It has ONE call site, in `_path` below, because one module addresses
    a deployment.
    """
    if len(value) > MAX_DEPLOYMENT_NAME_LENGTH:
        raise IdentifierInvalid(
            f"{field}: longer than {MAX_DEPLOYMENT_NAME_LENGTH} characters", param=field
        )
    if AZURE_DEPLOYMENT_PATTERN.fullmatch(value) is None:
        raise IdentifierInvalid(
            f"{field}: must match {AZURE_DEPLOYMENT_PATTERN.pattern}", param=field
        )
    return value


class AzureOpenAIAdapter(OpenAICompatibleAdapter):
    """One Azure OpenAI resource, addressed by its policy provider name.

    `name` is that policy name, never the kind and never the vendor: the ledger,
    the metric label and the resolver all address a provider by that string, so
    two resources in two regions stay apart exactly as `local` and `vllm` do.

    `endpoint` is the resource root as an operator pastes it out of the portal,
    with no path on it - the deployment path is this adapter's to build. There is
    no vendor default: the endpoint is per-resource, so a row without one is
    dropped by the factory rather than guessed at.

    `api_key` is MANDATORY here, unlike the parent's optional one: this talks to
    the vendor over the internet, and a keyless call is a 401 the tenant paid for.
    It travels in one header and nowhere else.

    Only `_auth_headers` and `_path` are overridden. If this class ever needs a
    third override, the subclass should be a
    standalone adapter.
    """

    def __init__(
        self,
        *,
        name: str,
        endpoint: str,
        api_key: str,
        api_version: str = AZURE_API_VERSION,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        # Set before the parent runs: `__init__` calls `_auth_headers`, and this
        # override reads no state, but `_path` reads `_api_version` on every
        # request and an attribute assigned afterwards would be missing from any
        # call the constructor itself made.
        self._api_version = api_version
        super().__init__(
            name=name,
            base_url=endpoint,
            api_key=api_key,
            transport=transport,
            sleep=sleep,
            jitter=jitter,
        )

    def _auth_headers(self, api_key: str | None) -> dict[str, str]:
        """The key as an `api-key` header, which is the only header Azure reads.

        The parent's bearer token is what an Azure resource answers 401 to, so
        this is not a stylistic difference. The key reaches no other header, no query parameter and
        no part of the body.

        `api_key` is typed optional to match the seam it overrides; the factory
        drops a keyless azure row before an adapter is constructed,
        so an empty one here sends no header rather than an empty one.
        """
        return {"api-key": api_key} if api_key else {}

    def _path(self, leaf: str, model: str) -> str:
        """`<AZURE_DEPLOYMENT_PATH>/<deployment><leaf>?api-version=<version>`.

        `model` is the deployment name on this wire, and it becomes a path
        segment, so it is checked first and the request is never built if the
        check fails. `leaf` arrives with its leading slash (`/chat/completions`,
        `/embeddings`) and is appended rather than re-spelled here, so the two
        endpoints stay the parent's.
        """
        deployment = checked_deployment(model, "model")
        return f"{AZURE_DEPLOYMENT_PATH}/{deployment}{leaf}?api-version={self._api_version}"
