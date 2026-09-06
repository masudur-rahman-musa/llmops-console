r"""`POST /v1/redact`, and the two pieces the inline path shares with it.

Three things live here, and the second and third are here because the chat and
embeddings routes need them too - the same reason `gateway_headers` lives in
`chat.py` and is imported rather than copied.

* `redact_batch` redacts a whole request under ONE map and ONE numbering. The
  engine is per-string; a request is a list of them, and #19's "placeholders are
  stable within a request" is exactly the difference. Called once per message,
  the engine would restart at 1 for every turn, so `<EMAIL_1>` would name a
  different person in message 1 and message 4 and the map would stop being an
  inverse (`test_one_numbering_spans_every_string_in_the_batch`).
* `redaction_applies` answers whether the tenant's rule for a data class carries
  `redact: true`. The routing policy is where that flag lives; nothing in the
  routing package reads it, because the control it configures is this one.
* The route itself: the two body shapes, the class it applied, and the id of the
  map it filed.

The endpoint always redacts, whatever the tenant's `redact` flag says for the
class (decision D). It is the ingestion path and the call is explicit, so
honouring the flag would turn a request that asked for redaction into a silent
no-op; `redaction_applies` gates the INLINE path and nothing here
(`test_the_endpoint_redacts_on_a_class_whose_rule_carries_no_redact_flag`).

No `X-Gateway-*` header is written on this route. Those six describe where a
request was served and by which provider, and this one resolves nothing
(`test_the_response_carries_no_gateway_headers`).

The cap is per STRING, applied by the request model rather than in the route body
(decision F, a project rule): a body over the cap is a 422 before anything
reaches the store, and a legal 64-item batch of at-cap strings is not refused for
its total. `MAX_REDACTION_INPUT_CHARS` is imported from the engine, never
restated, so a body and the engine cannot disagree about what is too long.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from opsconsole.core.errors import ProviderError
from opsconsole.core.metrics import record_redactions
from opsconsole.core.tracing import SpanAttributes, annotate, stage_span
from opsconsole.gateway.context import Context
from opsconsole.redaction import (
    MAX_REDACTION_INPUT_CHARS,
    Kind,
    RedactionInputTooLong,
    RedactionMapStore,
    RedactionMapStoreUnavailable,
    redact,
    taken_placeholders,
)
from opsconsole.routing.errors import IdentifierInvalid
from opsconsole.routing.policy import DataClass, Policy

REDACT = "/v1/redact"
"""The published route path. Pinned against the mounted route by
`test_the_route_path_is_the_published_one`."""


@dataclass(frozen=True, slots=True)
class RedactedBatch:
    """What one `redact_batch` call produced: the strings, and the one map behind them.

    Frozen because the two halves have to stay in step: the route writes `mapping`
    to the store and returns `texts` to the caller, so anything able to rewrite
    one of them afterwards would put the stored map and the answer out of step
    with no error anywhere (`test_the_batch_is_frozen`).
    """

    texts: tuple[str, ...]
    mapping: Mapping[str, str]
    counts: Mapping[str, int] = field(default_factory=dict)
    """How many placeholders were minted, per detector kind, keyed by `Kind.value`.

    Derived from the same walk that mints them rather than by parsing `mapping`'s
    keys, so the count and the map can never disagree about what a placeholder is.
    One entry is one redacted VALUE, which is what makes the same address in three
    turns count once (`test_a_repeated_value_across_turns_counts_once`)."""

    @property
    def redacted(self) -> bool:
        """Whether at least one value was actually replaced.

        This is what `X-Gateway-Redacted` reports (decision C): evidence about the
        payload, not about the rule. A clean body on a redact class answers False
        (`test_the_batch_reports_redaction_only_when_a_value_was_replaced`).
        """
        return bool(self.mapping)


REDACT_SPAN = "redact"
"""The stage span around a redaction pass. Emitted only where redaction actually
runs, which on the model routes is inside the `redaction_applies` branch and not
around the check: a `redact` span on an `internal` request would tell an auditor
that something was redacted when nothing touched it
(`test_redact_span_is_absent_on_a_non_redact_class`)."""


def redact_batch(texts: Sequence[str]) -> RedactedBatch:
    """Redact every string under one shared map and one shared numbering.

    Raises `RedactionInputTooLong` for any string over
    `MAX_REDACTION_INPUT_CHARS`, per string and not over the joined request
    (decision F); the caps come from `redact` itself rather than from a second
    comparison here (`test_a_string_over_the_cap_raises_wherever_it_sits`,
    `test_strings_at_the_cap_are_accepted_however_many_there_are`).

    The engine does the detection and the overlap resolution; only the
    placeholder walk is repeated here, over the whole call rather than over one
    string, because `redact` numbers from 1 every time it is called. Keys are
    `(kind, raw value)` exactly as the engine's are, so the same address in two
    turns is one placeholder and two addresses are never one.

    The taken-token set is the UNION over every input string, which is why
    `taken_placeholders` is public. A `<KIND_N>` token a caller wrote in message
    0 must not be handed out for a value in message 1: rehydrating message 0
    would otherwise inject a value nobody wrote there
    (`test_a_placeholder_written_in_one_string_is_never_handed_out_for_another`).
    """
    originals = tuple(texts)
    detected = tuple(redact(original).detections for original in originals)
    taken = set[str]().union(*(taken_placeholders(original) for original in originals))

    mapping: dict[str, str] = {}
    assigned: dict[tuple[Kind, str], str] = {}
    counters: dict[Kind, int] = {}
    minted: dict[str, int] = {}
    rewritten: list[str] = []

    for original, detections in zip(originals, detected, strict=True):
        pieces: list[str] = []
        cursor = 0
        for found in detections:
            key = (found.kind, found.value)
            placeholder = assigned.get(key)
            if placeholder is None:
                number = counters.get(found.kind, 0) + 1
                while f"<{found.kind.value}_{number}>" in taken:
                    number += 1
                counters[found.kind] = number
                placeholder = f"<{found.kind.value}_{number}>"
                assigned[key] = placeholder
                mapping[placeholder] = found.value
                minted[found.kind.value] = minted.get(found.kind.value, 0) + 1
            pieces.append(original[cursor : found.start])
            pieces.append(placeholder)
            cursor = found.end
        pieces.append(original[cursor:])
        rewritten.append("".join(pieces))

    return RedactedBatch(texts=tuple(rewritten), mapping=mapping, counts=minted)


def redact_or_reject(texts: Sequence[str], *, param: str) -> RedactedBatch:
    """`redact_batch`, with the engine's cap rendered as the 422 a client can act on.

    `RedactionInputTooLong` is a plain exception, not a `GatewayError` (an earlier change:
    `ErrorCode.VALIDATION` already belongs to `IdentifierInvalid` and
    `test_error_code_set_is_exhaustive` allows one class per code), so a route
    that let it escape would answer a 500 with no code. The inline paths have no
    request model to refuse an over-cap string on - the cap belongs to redaction,
    not to the chat or embeddings wire shape - so this is where it becomes a 422
    (`test_a_chat_message_over_the_cap_is_422_with_no_outbound_call`).

    The conversion lives in one place for both routes: two `except` blocks would
    be two chances to answer differently for the same refusal.
    """
    try:
        return redact_batch(texts)
    except RedactionInputTooLong as exc:
        raise IdentifierInvalid(
            f"{param}: a string longer than {exc.limit} characters cannot be redacted",
            param=param,
        ) from exc


def redaction_applies(policy: Policy, tenant: str, data_class: DataClass) -> bool:
    """Whether this tenant's rule for this class asks for inline redaction.

    False for a tenant or a class the policy does not define, rather than a
    `KeyError` that would surface as a 500 with no typed code. Neither is
    reachable on a served request - `resolve()` has already refused both by the
    time a route asks - so the honest answer is "no rule says to redact"
    (`test_redaction_applies_is_false_for_a_tenant_the_policy_does_not_define`).
    """
    tenant_config = policy.tenants.get(tenant)
    if tenant_config is None:
        return False
    rule = tenant_config.data_class_rules.get(data_class)
    return rule is not None and rule.redact


class _Wire(BaseModel):
    """A model on the published API boundary: frozen, and unknown fields rejected.

    Rejected rather than ignored, as on the chat and embeddings routes: a
    misspelled `texts` would otherwise be a 200 that redacted nothing the caller
    sent (`test_an_unknown_field_in_the_body_is_422`).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


def _within_cap(value: str, field: str) -> str:
    """Refuse one string the engine would refuse, before anything is stored.

    The number is the engine's own constant. Inclusive: `MAX_REDACTION_INPUT_CHARS`
    characters is accepted and one more is not
    (`test_a_string_at_the_cap_is_accepted`,
    `test_a_string_over_the_cap_is_422_before_any_store_write`).
    """
    if len(value) > MAX_REDACTION_INPUT_CHARS:
        raise ValueError(f"{field}: longer than {MAX_REDACTION_INPUT_CHARS} characters")
    return value


class RedactMessage(_Wire):
    """One turn, in the only shape this endpoint takes: a role and its text.

    `role` is a free string rather than the chat route's closed set. Nothing here
    is sent to a provider, so a role this gateway does not know is the caller's
    business; refusing it would make the ingestion path stricter than the
    completion path for no gain.
    """

    role: str
    content: str

    @model_validator(mode="after")
    def _within_caps(self) -> RedactMessage:
        """Apply the per-string cap to every turn, not only the first.

        On the message rather than on the list, so a 64-turn body is refused for
        the one turn that is too long rather than for its total
        (`test_the_cap_applies_to_a_later_message_too`,
        `test_two_messages_each_at_the_cap_are_accepted`).
        """
        _within_cap(self.content, "content")
        return self


class RedactRequest(_Wire):
    """One redaction request: exactly one of `text` or `messages`, and an optional class.

    `data_class` is the closed enum, so an unknown one is a 422 rather than a
    coercion onto the default: the class is stored alongside the map and read
    back by an earlier change, and silently filing `secret` as `confidential` labels it
    with something the caller never asked for
    (`test_an_unknown_body_data_class_is_422_before_any_store_write`).
    """

    text: str | None = None
    messages: tuple[RedactMessage, ...] | None = Field(default=None, min_length=1)
    data_class: DataClass | None = None

    @model_validator(mode="after")
    def _exactly_one_shape(self) -> RedactRequest:
        """Refuse a body carrying neither shape or both of them.

        Both is not a merge and not a silent preference for `text`: the answer
        has one `redaction_map_id`, and a caller that sent both cannot tell which
        half it describes. On the model, so the refusal happens before the route
        body and therefore before any store write
        (`test_a_body_that_is_not_exactly_one_shape_is_422_before_any_store_write`).
        """
        if (self.text is None) == (self.messages is None):
            raise ValueError("exactly one of text or messages is required")
        if self.text is not None:
            _within_cap(self.text, "text")
        return self


class RedactResponse(BaseModel):
    """The redacted payload, the class that applied, and the id of the stored map.

    The shape mirrors the request: a `text` call answers `text` with `messages`
    null, and a `messages` call the other way round, so a caller reads back what
    it sent rather than a shape it has to branch on twice.
    """

    model_config = ConfigDict(frozen=True)

    text: str | None
    messages: tuple[RedactMessage, ...] | None
    data_class: DataClass
    redaction_map_id: str


def filed_map_id(store: RedactionMapStore, tenant: str, mapping: Mapping[str, str]) -> str:
    """File one map, with an unreachable store rendered as the 502 a client can act on.

    `RedactionMapStoreUnavailable` is a plain exception, not a `GatewayError`
    (`ErrorCode.PROVIDER_ERROR` already belongs to `ProviderError` and
    `test_error_code_set_is_exhaustive` allows one concrete class per code), so a
    route that let it escape would answer a 500 with no code at all and a caller
    could not tell a dependency outage from a bug in its own request. The map
    store is an outbound dependency like a provider, which is why it lands on
    `PROVIDER_ERROR` rather than on a new code.

    The message is this route's own words. The client's carry an internal host
    and port, and they stay in the chained cause where the log line reads them
    (`test_a_store_that_cannot_be_reached_is_502_with_a_typed_code`,
    `test_a_store_failure_does_not_put_the_clients_words_in_front_of_a_caller`).

    `retryable=True`: an unreachable Redis is the transient class, and an earlier change
    reads that flag rather than re-deriving the verdict from a status it can no
    longer see.
    """
    try:
        return store.put(tenant, mapping)
    except RedactionMapStoreUnavailable as exc:
        raise ProviderError("the redaction map store is unavailable", retryable=True) from exc


router = APIRouter()


@router.post(REDACT)
def create_redaction(
    request: Request,
    context: Context,
    body: RedactRequest,
) -> RedactResponse:
    """Redact the body under one map, file the map for the tenant, answer both.

    The tenant is the AUTHENTICATED one from `context`, never the `X-Tenant`
    header on its own: the map holds raw personal data, so filing it under a name
    the client merely claimed would be the store's one unacceptable failure. The
    header is checked against the key before this runs
    (`test_a_tenant_that_does_not_match_the_key_is_403`).

    The class is the body's when it carries one and `X-Data-Class` otherwise
    (decision E), and the answer echoes whichever applied. It is a label on this
    call rather than a control: what the class decides is the INLINE path, and
    this route redacts either way.

    Declared `def` rather than `async def` like every other route here. Redaction
    is CPU work over the whole body, so a coroutine would hold the event loop for
    the length of the scan instead of running in a worker thread.
    """
    maps: RedactionMapStore = request.app.state.redaction_map_store
    data_class = body.data_class if body.data_class is not None else context.data_class

    with stage_span(request.scope, REDACT_SPAN):
        if body.text is not None:
            batch = redact_batch((body.text,))
            text: str | None = batch.texts[0]
            messages: tuple[RedactMessage, ...] | None = None
        else:
            turns = body.messages or ()
            batch = redact_batch(tuple(turn.content for turn in turns))
            text = None
            messages = tuple(
                turn.model_copy(update={"content": content})
                for turn, content in zip(turns, batch.texts, strict=True)
            )
    annotate(request.scope, SpanAttributes(redacted=batch.redacted, data_class=data_class.value))
    record_redactions(tenant=context.tenant, counts=batch.counts)

    return RedactResponse(
        text=text,
        messages=messages,
        data_class=data_class,
        redaction_map_id=filed_map_id(maps, context.tenant, batch.mapping),
    )
