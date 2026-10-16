"""The shared identifier check, moved into `app.core.validation`.

The constants, the public `checked_identifier` and the `IdentifierInvalid` error
all resolve from one place in `core`, so the twenty-one raisers that import them
no longer reach back into the routing package to do it.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[2] / "app"
"""`backend/app`, from this test's own path, so no absolute path is hard-coded."""


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


BAD_SLUG = "Bad Tenant!"
"""Uppercase, a space and a bang: fails `IDENTIFIER_PATTERN` so the check rejects it."""


def test_checked_identifier_carries_the_param_when_given() -> None:
    """Catches a check that drops the `param` on its way into the error: when the
    caller names the wire field the rejection came from, that name has to ride on
    the raised `IdentifierInvalid`, or the error body cannot point at the field."""
    from app.core.validation import IdentifierInvalid, checked_identifier

    # Splat the new keyword so the call type-checks before the signature gains it
    # and leaves no unused ignore to strip once it does (strict warns on those).
    param_kwarg = {"param": "tenant"}

    with pytest.raises(IdentifierInvalid) as exc:
        checked_identifier(BAD_SLUG, "tenant", **param_kwarg)

    assert exc.value.param == "tenant"


def test_checked_identifier_param_defaults_to_none() -> None:
    """Catches a check that invents a param when the caller gave none: with no
    `param` passed the raised error must carry `None`, not the field label or a
    guessed value, so the default stays the absence of a wire field."""
    from app.core.validation import IdentifierInvalid, checked_identifier

    with pytest.raises(IdentifierInvalid) as exc:
        checked_identifier(BAD_SLUG, "tenant")

    assert exc.value.param is None


def test_only_validation_holds_the_client_identifier_check() -> None:
    """Catches a collapse that left a hand-copied identifier check behind in any
    other function: exactly one function may hold all three tokens of the check,
    and it must be `checked_identifier` in `app/core/validation.py`. A function
    two modules away that still does its own cap, pattern and raise fails here."""
    tokens = ("MAX_IDENTIFIER_LENGTH", "IDENTIFIER_PATTERN.fullmatch", "raise IdentifierInvalid")
    holders: set[tuple[str, str]] = set()

    for path in sorted(APP_DIR.rglob("*.py")):
        source = path.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            segment = ast.get_source_segment(source, node)
            if segment is None:
                continue
            if all(token in segment for token in tokens):
                holders.add((str(path.relative_to(APP_DIR.parent)), node.name))

    assert holders == {("app/core/validation.py", "checked_identifier")}


def test_identifier_invalid_is_imported_from_its_home() -> None:
    """Catches an importer still reaching into `app.model_routing.errors` for
    `IdentifierInvalid`: the error now lives in `app.core.validation`, so no file
    under `app/` may import the name from the routing package, single-line or in a
    parenthesised multi-line import."""
    forbidden = re.compile(
        r"from\s+app\.model_routing\.errors\s+import\s+(?:\([^)]*)?\bIdentifierInvalid\b",
        re.DOTALL,
    )
    hits = [
        str(path.relative_to(APP_DIR.parent))
        for path in sorted(APP_DIR.rglob("*.py"))
        if forbidden.search(path.read_text())
    ]

    assert hits == []


def test_model_routing_errors_no_longer_exports_identifier_invalid() -> None:
    """Catches the re-export line surviving in `app.model_routing.errors`: once the
    name has a single home, importing it from the routing package must fail, so a
    left-behind `as IdentifierInvalid` re-export is caught by the import raising."""
    with pytest.raises(ImportError):
        from app.model_routing.errors import (  # type: ignore[attr-defined]  # noqa: F401
            IdentifierInvalid,
        )
