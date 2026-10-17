"""Tests that pin the new shared base class and the removal of `healthcheck`.

Two controls, arranged so each test catches only the control it names:

* `test_both_adapters_subclass_the_shared_base` — catches an implementation
  that left the base class out of one adapter's MRO, leaving that adapter
  without the shared `close`/`name`/`price_key` from the base.
* `test_no_adapter_or_protocol_exposes_healthcheck` — catches an
  implementation that kept `healthcheck` on an adapter or on the `Adapter`
  protocol after the method was meant to be removed.
"""

from __future__ import annotations

# These imports are expected to FAIL before an earlier change is implemented:
# - `HttpAdapter` does not exist yet in `app.providers.base`
# The tests therefore serve as the acceptance check: they go green only
# once the implementation adds `HttpAdapter` and removes `healthcheck`.


def test_both_adapters_subclass_the_shared_base() -> None:
    """Catches an implementation that added HttpAdapter to one adapter only,
    or not at all, leaving the common close/name/price_key in two copies."""
    from app.providers.anthropic import AnthropicAdapter
    from app.providers.base import HttpAdapter
    from app.providers.openai_compatible import OpenAICompatibleAdapter

    assert issubclass(OpenAICompatibleAdapter, HttpAdapter), (
        "OpenAICompatibleAdapter must subclass HttpAdapter"
    )
    assert issubclass(AnthropicAdapter, HttpAdapter), "AnthropicAdapter must subclass HttpAdapter"


def test_no_adapter_or_protocol_exposes_healthcheck() -> None:
    """Catches an implementation that left healthcheck() on an adapter class
    or kept it in the Adapter protocol after the method was meant to be removed."""
    from app.providers.anthropic import AnthropicAdapter
    from app.providers.base import Adapter
    from app.providers.openai_compatible import OpenAICompatibleAdapter

    assert not hasattr(Adapter, "healthcheck"), "Adapter protocol must not expose healthcheck"
    assert not hasattr(OpenAICompatibleAdapter, "healthcheck"), (
        "OpenAICompatibleAdapter must not expose healthcheck"
    )
    assert not hasattr(AnthropicAdapter, "healthcheck"), (
        "AnthropicAdapter must not expose healthcheck"
    )
