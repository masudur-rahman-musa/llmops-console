"""Shared redaction logic: batch redaction, the directory, the map store."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from app.core.errors import ProviderError
from app.core.validation import IdentifierInvalid
from app.model_routing.policy import DataClass, Policy
from app.redaction import (
    DirectoryStore,
    DirectoryStoreUnavailable,
    Kind,
    NameDirectory,
    RedactionInputTooLong,
    RedactionMapStore,
    RedactionMapStoreUnavailable,
    redact,
    taken_placeholders,
)


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


def tenant_directory(store: DirectoryStore, tenant: str) -> NameDirectory:
    """This tenant's names, compiled once, with an unreachable store rendered as a 502.

    Compiled HERE and not inside `redact`: a request is up
    to 64 strings under one numbering, and compiling 2000 alternatives per string
    is the difference between microseconds and a visible pause.

    `DirectoryStoreUnavailable` becomes `ProviderError` - 502, `PROVIDER_ERROR` -
    and never an empty directory. That is decision H, and it is the same
    deny-by-default rule residency holds: a request redacted without the tenant's
    names and served anyway carries the values this control exists to remove out
    to a provider, with `X-Gateway-Redacted: true` on the answer
    (`test_an_unreachable_directory_store_is_502_with_no_outbound_call`). The
    message is this module's own words, exactly as `filed_map_id`'s is: the
    client's name an internal host and port
    (`test_a_directory_store_failure_does_not_put_the_clients_words_in_front_of_a_caller`).
    """
    try:
        names = store.get(tenant)
    except DirectoryStoreUnavailable as exc:
        raise ProviderError("the directory store is unavailable", retryable=True) from exc
    return NameDirectory.compile(names)


def redact_batch(texts: Sequence[str], directory: NameDirectory | None = None) -> RedactedBatch:
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
    detected = tuple(redact(original, directory).detections for original in originals)
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


def redact_or_reject(
    texts: Sequence[str], *, param: str, directory: NameDirectory | None = None
) -> RedactedBatch:
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
        return redact_batch(texts, directory)
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

    `retryable=True` classifies the failure rather than asking for anything: an
    unreachable Redis is the transient class. Nothing acts on it here - this route
    has one store and no second candidate to move to - and the routes' fallback
    loop reads the flag on a PROVIDER call, which this is not.
    """
    try:
        return store.put(tenant, mapping)
    except RedactionMapStoreUnavailable as exc:
        raise ProviderError("the redaction map store is unavailable", retryable=True) from exc
