"""The quality page's gate rows: the newest run's scores beside the baseline thresholds.

No backing service. Each case builds the admin app with its own baseline file (a
temporary file the case writes, or none) and swaps in a reader that answers fixed
runs, the same way test_console_routes.py swaps its stub reader in after startup.

What is pinned, each case naming the wrong implementation it catches:

* The gate has two rows, `check_score` then `judge_score`, each with the newest
  run's value, the baseline's threshold and a verdict.
* A score exactly on its threshold passes.
* A baseline that is missing or unreadable never fails the page: thresholds and
  verdicts come back null.
* With no run in the window the values and verdicts are null, the thresholds not.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.schemas.dashboard import EvalRunSummary
from app.storage.dashboard_reads import QualityRead

from .conftest import (
    ADMIN_KEY,
    API_KEYS,
    NO_PROVIDER_KEYS,
    SHIPPED_POLICY_PATH,
    sha256_hex,
)

QUALITY = "/console/quality"
BASELINE = {"check_score": 48.48, "judge_score": 27.42}
NEWEST = datetime(2026, 9, 15, 2, 40, tzinfo=UTC)
SHIPPED_BASELINE = Path(__file__).resolve().parents[2] / "evals" / "conversation-baseline.json"


def run(run_id: str, started_at: datetime, check: float, judge: float) -> EvalRunSummary:
    return EvalRunSummary(
        id=run_id,
        started_at=started_at,
        run_kind="ci",
        git_sha=None,
        passed=True,
        metrics={"check_score": check, "judge_score": judge},
    )


class RunsReader:
    """A configured reader answering fixed runs on the quality read."""

    configured = True

    def __init__(self, runs: tuple[EvalRunSummary, ...]) -> None:
        self.runs = runs

    def quality(self, *, days: int, now: datetime) -> QualityRead:
        return QualityRead(
            runs=self.runs, release_markers=(), categories=(), taxonomy=(), blocked=()
        )


def gate(
    baseline_path: Path, runs: tuple[EvalRunSummary, ...]
) -> tuple[httpx.Response, list[dict[str, Any]]]:
    """The quality response for one baseline file and one set of runs, and its gate rows."""
    settings = Settings(
        api_keys=API_KEYS,
        routing_policy_path=SHIPPED_POLICY_PATH,
        admin_key_sha256=sha256_hex(ADMIN_KEY),
        eval_baseline_path=baseline_path,
    )
    with TestClient(create_app(settings, NO_PROVIDER_KEYS)) as client:
        client.app.state.console_reader = RunsReader(runs)  # type: ignore[attr-defined]
        response: httpx.Response = client.get(
            QUALITY, headers={"Authorization": f"Bearer {ADMIN_KEY}"}
        )
    assert response.status_code == 200, response.text
    return response, response.json()["gate"]


@pytest.fixture
def baseline(tmp_path: Path) -> Path:
    path = tmp_path / "conversation-baseline.json"
    path.write_text(json.dumps(BASELINE), encoding="utf-8")
    return path


@pytest.fixture
def runs() -> Iterator[tuple[EvalRunSummary, ...]]:
    # Newest first, as the reader answers; the older run passes both scores so a gate
    # that read it would show different verdicts.
    yield (
        run("run-new", NEWEST, 52.1, 25.0),
        run("run-old", NEWEST - timedelta(days=2), 60.0, 30.0),
    )


def test_the_gate_rows_compare_the_newest_run_to_the_baseline(
    baseline: Path, runs: tuple[EvalRunSummary, ...]
) -> None:
    """Catches a gate that reads the older run, swaps value and threshold, flips a verdict,
    or lists the judge row before the check row."""
    _, rows = gate(baseline, runs)

    notes = [row.pop("note") for row in rows]
    assert rows == [
        {
            "name": "check_score",
            "label": "Check score",
            "value": 52.1,
            "threshold": 48.48,
            "passed": True,
        },
        {
            "name": "judge_score",
            "label": "Judge score",
            "value": 25.0,
            "threshold": 27.42,
            "passed": False,
        },
    ]
    assert notes[0] is None
    assert "kappa 0.7" in notes[1]


@pytest.mark.parametrize(
    ("check", "passed"),
    [(48.47, False), (48.48, True), (48.49, True)],
)
def test_a_run_landing_on_the_threshold_passes(baseline: Path, check: float, passed: bool) -> None:
    """Catches a verdict that uses `>` instead of `>=`, which fails a run exactly on its
    threshold, or one that rounds so 48.47 slips through."""
    _, rows = gate(baseline, (run("run-new", NEWEST, check, 30.0),))

    assert rows[0]["value"] == check
    assert rows[0]["passed"] is passed


@pytest.mark.parametrize("content", [None, "{not json", "[1, 2]"], ids=["missing", "junk", "list"])
def test_a_missing_baseline_gives_null_thresholds(
    tmp_path: Path, runs: tuple[EvalRunSummary, ...], content: str | None
) -> None:
    """Catches a page that answers 500 when the baseline file is absent or unreadable, or
    that guesses a verdict and a threshold of zero without a baseline."""
    path = tmp_path / "conversation-baseline.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")

    _, rows = gate(path, runs)

    assert [(row["name"], row["value"], row["threshold"], row["passed"]) for row in rows] == [
        ("check_score", 52.1, None, None),
        ("judge_score", 25.0, None, None),
    ]


def test_no_runs_gives_null_values(baseline: Path) -> None:
    """Catches a gate that invents a zero value or a failed verdict when the window holds
    no run, or drops the thresholds with it."""
    _, rows = gate(baseline, ())

    assert [(row["name"], row["value"], row["threshold"], row["passed"]) for row in rows] == [
        ("check_score", None, 48.48, None),
        ("judge_score", None, 27.42, None),
    ]


def test_the_default_path_is_the_committed_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catches a default that points at a file which is not in the repository, so a fresh
    checkout shows every threshold as not set."""
    monkeypatch.delenv("OPSCONSOLE_EVAL_BASELINE_PATH", raising=False)

    path = Settings().eval_baseline_path

    assert path == SHIPPED_BASELINE
    assert path.is_file()
