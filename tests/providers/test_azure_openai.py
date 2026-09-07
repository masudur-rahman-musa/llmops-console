"""The Azure OpenAI adapter: the deployment URL, the `api-key` header, and nothing else.

`AzureOpenAIAdapter` is `OpenAICompatibleAdapter` with two seams overridden (plan
11b decision A), so the mapping is not re-asserted here: `to_openai` and
`to_openai_embedding` are the same functions
`tests/providers/test_local_mapping.py` and
`tests/providers/test_local_embeddings.py` already pin in both directions, and a
second copy of that suite would assert the same code twice and drift on the first
change (decision E). What this file asserts is exactly what Azure changes - the
request URL, the auth header - plus the one case that proves the body did NOT
change (`test_the_body_is_the_shared_openai_mapping`).

Everything runs over `httpx.MockTransport`: no socket is opened, no key is read
from the environment, and no Azure resource exists to call (#66). The OpenAI-wire
fixtures under `tests/providers/fixtures/local/` are the answers, because Azure
answers the OpenAI shape and the readers are the shared ones.

Every URL below is asserted as a WHOLE string including the query, never as a
substring: an adapter that appended the deployment but dropped `api-version`
passes every substring assertion and 400s on every real call.
"""

from __future__ import annotations

import pytest

from opsconsole.providers.azure_openai import (
    AZURE_API_VERSION,
    AZURE_DEPLOYMENT_PATH,
    AZURE_DEPLOYMENT_PATTERN,
    MAX_DEPLOYMENT_NAME_LENGTH,
    AzureOpenAIAdapter,
    checked_deployment,
)
from opsconsole.providers.base import (
    ChatMessage,
    ChatRequest,
    EmbeddingRequest,
    ResponseFormat,
    ToolChoice,
    ToolSpec,
)
from opsconsole.providers.openai_compatible import to_openai
from opsconsole.routing.errors import IdentifierInvalid

from .conftest import (
    FixedJitter,
    ScriptedTransport,
    SleepRecorder,
    Step,
    local_ok,
)
from .test_local_streaming import SseStep, SseTransport, sse_body

# The policy provider name: deliberately neither `azure_openai` (the kind) nor a
# vendor name, so an adapter keyed, named or priced by the kind is visible.
AZURE_PROVIDER_NAME = "azure-eu-sweden"

# A resource endpoint in the reserved `.test` TLD, with no trailing slash, as an
# operator would paste it out of the portal.
AZURE_ENDPOINT = "https://ops-console-eu.openai.azure.test"

AZURE_KEY = "azure-test-not-a-real-key"

# Deployment names, not model names: on Azure the caller names the deployment and
# the policy's `model` field carries it. Neither contains a `.`, because a
# deployment name cannot (`AZURE_DEPLOYMENT_PATTERN`), which is also why the
# embedding deployment is not the `bge-small-en-v1.5` the local suite uses.
CHAT_DEPLOYMENT = "gpt-4o-eu"
EMBED_DEPLOYMENT = "text-embedding-3-small-eu"

# The three URLs, written out whole. Nothing here is composed from the constants
# under test: a typo in the implementation's f-string must not be reproduced by a
# test that builds the expected value the same way.
CHAT_URL = (
    "https://ops-console-eu.openai.azure.test"
    "/openai/deployments/gpt-4o-eu/chat/completions?api-version=2024-10-21"
)
EMBEDDINGS_URL = (
    "https://ops-console-eu.openai.azure.test"
    "/openai/deployments/text-embedding-3-small-eu/embeddings?api-version=2024-10-21"
)

REQUEST = ChatRequest(
    model=CHAT_DEPLOYMENT,
    messages=(ChatMessage(role="user", content="What is the capital of Germany?"),),
)

EMBEDDING_REQUEST = EmbeddingRequest(model=EMBED_DEPLOYMENT, input=("the capital of Germany",))

WEATHER_TOOL = ToolSpec(
    name="get_weather",
    description="Look up the weather in a city.",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
)

# Deliberately the busiest request the contract allows: a system turn, tools, a
# forced choice, a response format and both numeric knobs. Every one of those is
# a field the shared mapper renders, so a subclass that rebuilt the body would
# have to reproduce all of them to stay green.
RICH_REQUEST = ChatRequest(
    model=CHAT_DEPLOYMENT,
    messages=(
        ChatMessage(role="system", content="You are terse."),
        ChatMessage(role="user", content="What is the weather in Berlin?"),
    ),
    tools=(WEATHER_TOOL,),
    tool_choice=ToolChoice(mode="tool", name="get_weather"),
    max_tokens=256,
    temperature=0.25,
    response_format=ResponseFormat(
        name="weather_answer",
        json_schema={"type": "object", "properties": {"summary": {"type": "string"}}},
    ),
)


def make_azure_adapter(
    steps: list[Step],
    *,
    name: str = AZURE_PROVIDER_NAME,
    endpoint: str = AZURE_ENDPOINT,
    api_key: str = AZURE_KEY,
) -> tuple[AzureOpenAIAdapter, ScriptedTransport]:
    """An Azure adapter over a scripted transport, with an explicit key and no clock.

    Built the way `make_local_adapter` builds its own (`conftest.py`): the key is
    passed in rather than read from the environment, and `sleep` and `jitter` are
    injected, so a retry costs microseconds.
    """
    scripted = ScriptedTransport(steps)
    adapter = AzureOpenAIAdapter(
        name=name,
        endpoint=endpoint,
        api_key=api_key,
        transport=scripted.transport,
        sleep=SleepRecorder(),
        jitter=FixedJitter(1.0),
    )
    return adapter, scripted


def make_azure_sse_adapter(steps: list[SseStep]) -> tuple[AzureOpenAIAdapter, SseTransport]:
    """The same adapter over the SSE transport `test_local_streaming.py` already scripts."""
    scripted = SseTransport(steps)
    adapter = AzureOpenAIAdapter(
        name=AZURE_PROVIDER_NAME,
        endpoint=AZURE_ENDPOINT,
        api_key=AZURE_KEY,
        transport=scripted.transport,
        sleep=SleepRecorder(),
        jitter=FixedJitter(1.0),
    )
    return adapter, scripted


# --- the three URLs ----------------------------------------------------------


def test_the_chat_url_is_the_deployment_path_with_the_api_version() -> None:
    """Catches an adapter that inherited the parent's path unchanged - `/chat/completions`
    against the resource root, which Azure 404s - and one that builds the deployment path
    but omits the mandatory `api-version` query, which Azure answers 400 to on every
    call. The whole URL is the assertion, so neither can hide behind the other."""
    adapter, scripted = make_azure_adapter([local_ok("plain")])

    response = adapter.chat(REQUEST)

    assert AZURE_DEPLOYMENT_PATH == "/openai/deployments"
    assert scripted.request_count == 1
    sent = scripted.requests[0]
    assert sent.method == "POST"
    assert str(sent.url) == CHAT_URL
    assert response.content == "Ship it."


def test_the_embeddings_url_is_the_deployment_path_with_the_api_version() -> None:
    """Catches a `_path` override applied to the chat leaf only: `embed()` would then
    POST an embedding body to the resource root's `/embeddings`, and the residency-only
    embedding route would 404 while every chat case stayed green. Catches too an override
    that hardcodes `/chat/completions` into the deployment path for both leaves."""
    adapter, scripted = make_azure_adapter([local_ok("embeddings_single")])

    response = adapter.embed(EMBEDDING_REQUEST)

    assert scripted.request_count == 1
    sent = scripted.requests[0]
    assert sent.method == "POST"
    assert str(sent.url) == EMBEDDINGS_URL
    assert len(response.embeddings) == 1


def test_the_streamed_chat_url_is_the_same_deployment_path() -> None:
    """Catches a `stream_chat` that opens its stream against the parent's constant path
    instead of `self._path(...)` - the one route no non-streaming test can see, since the
    open is a different call on the transport - which would 404 every streamed answer
    while chat and embeddings both worked."""
    adapter, scripted = make_azure_sse_adapter([sse_body("stream_text")])

    events = list(adapter.stream_chat(REQUEST))

    assert scripted.request_count == 1
    sent = scripted.requests[0]
    assert sent.method == "POST"
    assert str(sent.url) == CHAT_URL
    assert events, "the scripted stream must still be read through the seam"
    assert scripted.body()["stream"] is True


# --- the key and the body ----------------------------------------------------


def test_the_key_travels_only_as_an_api_key_header() -> None:
    """Catches the inherited bearer header, which an Azure resource ignores and answers
    401 to, and catches a key added to the query string or the body alongside the header -
    the URL carries a query already, so appending one more parameter is the easy mistake,
    and every proxy on the path would log the key."""
    adapter, scripted = make_azure_adapter([local_ok("plain")])

    adapter.chat(REQUEST)

    sent = scripted.requests[0]
    assert sent.headers.get_list("api-key") == [AZURE_KEY]
    assert "authorization" not in sent.headers
    assert [name for name, value in sent.headers.multi_items() if AZURE_KEY in value] == ["api-key"]
    assert AZURE_KEY not in str(sent.url)
    assert AZURE_KEY not in sent.content.decode("utf-8")


def test_the_body_is_the_shared_openai_mapping() -> None:
    """Catches a subclass that builds its own body - the likeliest form being one that
    strips `model` because Azure takes the deployment from the URL, or renames a field to
    the vendor's spelling. The sent body is compared against `to_openai(request)` whole,
    so any added, dropped or renamed key fails, and mapper reuse is pinned rather than
    assumed by the absence of a mapping suite here."""
    adapter, scripted = make_azure_adapter([local_ok("plain")])

    adapter.chat(RICH_REQUEST)

    assert scripted.request_count == 1
    assert scripted.body() == to_openai(RICH_REQUEST)
    assert scripted.body()["model"] == CHAT_DEPLOYMENT


# --- the pinned constants ----------------------------------------------------


def test_the_api_version_default_is_exactly_2024_10_21() -> None:
    """Catches a silent bump of the pinned api-version: the value decides which fields the
    resource accepts, so a preview string arriving in a refactor changes the wire for
    every deployment at once. A range or a `startswith` would admit exactly that, so the
    constant is asserted as an exact literal."""
    assert AZURE_API_VERSION == "2024-10-21"


def test_max_deployment_name_length_is_exactly_64() -> None:
    r"""Catches a cap borrowed from `checked_tool_name` (128, decision D), which would let
    a name Azure cannot have onto the wire as a 404 the caller cannot read, and catches a
    pattern whose own `{1,64}` quantifier drifted away from the constant beside it."""
    assert MAX_DEPLOYMENT_NAME_LENGTH == 64
    assert AZURE_DEPLOYMENT_PATTERN.pattern == "[a-zA-Z0-9_-]{1,64}"


# --- the deployment pre-check ------------------------------------------------

AT_CAP = "a" * MAX_DEPLOYMENT_NAME_LENGTH
OVER_CAP = "a" * (MAX_DEPLOYMENT_NAME_LENGTH + 1)


def test_a_deployment_name_of_exactly_64_characters_is_accepted() -> None:
    """Catches an exclusive boundary (`len(value) < 64`), which would refuse a deployment
    Azure itself allows and take the provider out of service for a name nobody can
    shorten. Without this row the rejection table below would pass against a
    `checked_deployment` that rejected everything."""
    adapter, scripted = make_azure_adapter([local_ok("plain")])

    adapter.chat(ChatRequest(model=AT_CAP, messages=REQUEST.messages))

    assert len(AT_CAP) == 64
    assert checked_deployment(AT_CAP, "model") == AT_CAP
    assert scripted.request_count == 1
    assert str(scripted.requests[0].url) == (
        f"{AZURE_ENDPOINT}/openai/deployments/{AT_CAP}/chat/completions?api-version=2024-10-21"
    )


def test_a_deployment_name_of_65_characters_is_rejected() -> None:
    """Catches an inclusive-on-the-wrong-side cap (`len(value) <= 65`) and a length check
    left out entirely: with the pattern alone anchored by `fullmatch` this row would still
    fail, so the row exists to hold the BOUNDARY at 64 rather than wherever the pattern's
    quantifier happens to sit."""
    adapter, scripted = make_azure_adapter([local_ok("plain")])

    with pytest.raises(IdentifierInvalid) as raised:
        adapter.chat(ChatRequest(model=OVER_CAP, messages=REQUEST.messages))

    assert len(OVER_CAP) == 65
    assert raised.value.http_status == 422
    assert raised.value.param == "model"
    assert scripted.request_count == 0


# The rejection table a project rule asks for. The newline and CRLF rows are the
# ones a pattern applied with `re.match` and `^...$` lets through - and here that
# is a request-splitting attempt in a URL path, not just a bad name. The `/` and
# `..` rows are the ones that make the deployment segment address a different
# path on the resource; the empty string is the row a truthiness check misses.
BAD_DEPLOYMENTS: list[tuple[str, str]] = [
    ("trailing newline", f"{CHAT_DEPLOYMENT}\n"),
    ("trailing CRLF", f"{CHAT_DEPLOYMENT}\r\n"),
    ("trailing space", f"{CHAT_DEPLOYMENT} "),
    ("a path separator", f"{CHAT_DEPLOYMENT}/completions"),
    ("a parent segment", ".."),
    ("the empty string", ""),
]


@pytest.mark.parametrize(
    ("case", "deployment"), BAD_DEPLOYMENTS, ids=[case for case, _ in BAD_DEPLOYMENTS]
)
def test_a_bad_deployment_name_raises_before_any_outbound_call(case: str, deployment: str) -> None:
    """Catches a model string interpolated into the URL path with no check at all, and a
    check written with `re.match` and `^...$` (the newline and CRLF rows pass that). The
    model field is checked NOWHERE else in the request path - `checked_tool_name` guards
    tool ids and names, not `model` - so only this adapter's own pre-check can reject
    these, and `request_count == 0` proves it ran before the request was built rather
    than after Azure answered."""
    adapter, scripted = make_azure_adapter([local_ok("plain")])

    with pytest.raises(IdentifierInvalid) as raised:
        adapter.chat(ChatRequest(model=deployment, messages=REQUEST.messages))

    assert raised.value.http_status == 422, case
    assert raised.value.param == "model", case
    assert scripted.request_count == 0, f"{case}: the provider was called first"


# --- the ledger's identity ---------------------------------------------------


def test_price_key_is_the_policy_provider_name_slash_model() -> None:
    """Catches an adapter priced by the vendor or by the kind (`azure_openai/gpt-4o-eu`
    for every resource): the ledger prices against this string, so two Azure resources -
    an EU one and a US one - would be billed into one row and one metric label, which is
    exactly the distinction residency reporting exists to make."""
    adapter, _ = make_azure_adapter([local_ok("plain")])

    assert adapter.name == AZURE_PROVIDER_NAME
    assert adapter.price_key(CHAT_DEPLOYMENT) == "azure-eu-sweden/gpt-4o-eu"
