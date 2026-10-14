"""The shared identifier check, moved into `app.core.validation`.

The constants, the public `checked_identifier` and the `IdentifierInvalid` error
all resolve from one place in `core`, so the twenty-one raisers that import them
no longer reach back into the routing package to do it.
"""

from __future__ import annotations

import pytest


def test_the_identifier_check_lives_once_in_core_validation() -> None:
    """Catches a move that left the pattern, the cap or the check in
    `app.routing.resolver`: the names must resolve from `app.core.validation`,
    the cap must still be 64, and a valid slug must pass and come back unchanged
    rather than being rewritten on the way through."""
    from app.core.validation import (
        IDENTIFIER_PATTERN,
        MAX_IDENTIFIER_LENGTH,
        IdentifierInvalid,
        checked_identifier,
    )

    assert MAX_IDENTIFIER_LENGTH == 64
    assert IDENTIFIER_PATTERN.fullmatch("acme-eu") is not None
    assert issubclass(IdentifierInvalid, Exception)
    assert checked_identifier("acme-eu", "tenant") == "acme-eu"


def test_a_too_long_identifier_raises_identifier_invalid() -> None:
    """Catches a check that dropped the length cap and only kept the pattern:
    sixty-five lowercase letters match `IDENTIFIER_PATTERN` but are one over the
    inclusive limit, so the only thing that can refuse them is the cap."""
    from app.core.validation import IdentifierInvalid, checked_identifier

    too_long = "a" * 65

    with pytest.raises(IdentifierInvalid):
        checked_identifier(too_long, "tenant")
