"""Pydantic request and response models for /v1/redact."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.model_routing.policy import DataClass
from app.redaction import MAX_REDACTION_INPUT_CHARS


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
