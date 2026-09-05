"""Building the adapter set the gateway can actually call, from the routing policy.

The policy says which providers exist; the environment says which of them are
usable. This is where the two meet, and the rule is that an unusable provider is
absent rather than broken: an adapter constructed with an empty key would look
healthy to the resolver and 401 on the first real request a tenant paid for.

An enabled provider with no key is dropped with a warning naming both the
provider and the variable to set, because a provider that vanishes silently
looks to an operator like a routing bug
(`test_enabled_provider_without_a_key_is_omitted_with_a_warning`). A disabled one
is dropped without a warning: that is an operator's decision, and warning about
it trains people to ignore the warning that matters
(`test_disabled_provider_is_omitted_without_a_warning`).

Only `ProviderKind.ANTHROPIC` is built here. The other kinds are absent and
silent until the plans that add their adapters; a missing provider is a case the
resolver's health seam already handles.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping

from opsconsole.providers.anthropic import ANTHROPIC_BASE_URL, AnthropicAdapter
from opsconsole.providers.base import Adapter
from opsconsole.routing.policy import Policy, ProviderKind

logger = logging.getLogger(__name__)


def build_adapters(policy: Policy, environ: Mapping[str, str] | None = None) -> dict[str, Adapter]:
    """One adapter per usable provider, keyed by its policy provider name.

    That key is the same string the ledger prices against and the resolver names,
    so a policy with two Anthropic accounts produces two distinguishable adapters.
    `environ` is injected rather than read from `os.environ` inside the loop, so a
    test never needs a provider key in the real environment (#66).
    """
    source = os.environ if environ is None else environ
    adapters: dict[str, Adapter] = {}

    for name, provider in policy.providers.items():
        if not provider.enabled or provider.kind is not ProviderKind.ANTHROPIC:
            continue

        # Presence AND non-emptiness: `key_env` naming a variable set to "" is a
        # half-configured deployment, not a configured one.
        api_key = source.get(provider.key_env) if provider.key_env is not None else None
        if not api_key:
            logger.warning(
                "provider %r is enabled but its API key is not set: "
                "no adapter was built for it (expected in %s)",
                name,
                provider.key_env if provider.key_env is not None else "an unset key_env",
            )
            continue

        adapters[name] = AnthropicAdapter(
            name=name,
            api_key=api_key,
            base_url=provider.base_url if provider.base_url is not None else ANTHROPIC_BASE_URL,
        )

    return adapters
