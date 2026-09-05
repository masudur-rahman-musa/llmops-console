"""Building the adapter set the gateway can actually call, from the routing policy.

The policy says which providers exist; the environment says which of them are
usable. This is where the two meet, and the rule is that an unusable provider is
absent rather than broken: an adapter constructed with an empty key would look
healthy to the resolver and 401 on the first real request a tenant paid for.

What "usable" means is per kind, and the difference is the point:

* an `anthropic` provider needs a key, so an enabled one without it is dropped
  with a warning naming both the provider and the variable to set
  (`test_enabled_provider_without_a_key_is_omitted_with_a_warning`);
* an `openai_compatible` provider may legitimately declare NO `key_env` - the
  shipped `local` and `vllm` sit on a private network - so a missing key is not
  a fault there (an earlier change decision B,
  `test_an_openai_compatible_provider_with_no_key_env_is_built`). It needs a
  `base_url` instead, because there is no vendor default to fall back to and a
  guessed endpoint would send `self`-region traffic somewhere else
  (`test_an_openai_compatible_provider_without_a_base_url_is_dropped_with_a_warning`).

A provider that DECLARES a `key_env` which is unset is dropped whatever its kind:
that is a half-configured deployment, not a configured one. A disabled provider
is dropped without a warning either way: that is an operator's decision, and
warning about it trains people to ignore the warning that matters
(`test_disabled_provider_is_omitted_without_a_warning`).

`ProviderKind.OPENAI` and `ProviderKind.AZURE_OPENAI` are still absent and
silent, until an earlier change adds their adapters; a missing provider is a case the
resolver's health seam already handles.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping

from opsconsole.providers.anthropic import ANTHROPIC_BASE_URL, AnthropicAdapter
from opsconsole.providers.base import Adapter
from opsconsole.providers.openai_compatible import OpenAICompatibleAdapter
from opsconsole.routing.policy import Policy, Provider, ProviderKind

logger = logging.getLogger(__name__)


def build_adapters(policy: Policy, environ: Mapping[str, str] | None = None) -> dict[str, Adapter]:
    """One adapter per usable provider, keyed by its policy provider name.

    That key is the same string the ledger prices against and the resolver names,
    so a policy with two accounts on one vendor - or `local` and `vllm`, which are
    one kind on two boxes - produces two distinguishable adapters
    (`test_one_class_serves_both_openai_compatible_providers`). `environ` is
    injected rather than read from `os.environ` inside the loop, so a test never
    needs a provider key in the real environment (#66).
    """
    source = os.environ if environ is None else environ
    adapters: dict[str, Adapter] = {}

    for name, provider in policy.providers.items():
        if not provider.enabled:
            continue

        # Presence AND non-emptiness: `key_env` naming a variable set to "" is a
        # half-configured deployment, not a configured one.
        api_key = source.get(provider.key_env) if provider.key_env is not None else None

        if provider.kind is ProviderKind.ANTHROPIC:
            if not api_key:
                _warn_unusable_key(name, provider)
                continue
            adapters[name] = AnthropicAdapter(
                name=name,
                api_key=api_key,
                base_url=provider.base_url if provider.base_url is not None else ANTHROPIC_BASE_URL,
            )

        elif provider.kind is ProviderKind.OPENAI_COMPATIBLE:
            if provider.key_env is not None and not api_key:
                _warn_unusable_key(name, provider)
                continue
            if provider.base_url is None:
                logger.warning(
                    "provider %r is enabled but names no base_url: no adapter was built for "
                    "it (an openai_compatible provider has no default endpoint, and guessing "
                    "one would send its traffic off the box)",
                    name,
                )
                continue
            adapters[name] = OpenAICompatibleAdapter(
                name=name, base_url=provider.base_url, api_key=api_key
            )

        # `openai` and `azure_openai` are absent and silent until an earlier change adds
        # their adapters. `test_non_anthropic_kind_is_absent_without_a_warning`
        # pins that for `openai` specifically - it asserts one provider of that
        # kind is neither built nor warned about, not that every non-Anthropic
        # kind is dropped, which this diff makes false for `openai_compatible`.

    return adapters


def _warn_unusable_key(name: str, provider: Provider) -> None:
    """Say that a provider needing a key has none, naming the variable to set.

    A provider that vanishes silently looks to an operator like a routing bug
    (`test_enabled_provider_without_a_key_is_omitted_with_a_warning`).
    """
    logger.warning(
        "provider %r is enabled but its API key is not set: "
        "no adapter was built for it (expected in %s)",
        name,
        provider.key_env if provider.key_env is not None else "an unset key_env",
    )
