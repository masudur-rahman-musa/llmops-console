"""The folder move into `backend/` leaves nothing at its old place."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_the_old_tree_is_gone() -> None:
    """Catches a move that copied files forward but left the originals behind."""
    for old in (
        "packages",
        "services/gateway/main.py",
        "routing.yaml",
        "prices.yaml",
        "backend/config/prices.yaml",
    ):
        assert not (REPO_ROOT / old).exists(), f"old path still present: {old}"


def test_no_module_named_opsconsole() -> None:
    """Catches a leftover compatibility shim keeping the old import name alive."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("opsconsole")
