"""An earlier change: the postmortem, its README links, and the runbook's incident-recording command.

Text reads over committed files; no build, no service.
"""

from __future__ import annotations

import re
from pathlib import Path

#: This file's own location, never the working directory - `pytest` runs from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[3]

POSTMORTEMS_DIR = REPO_ROOT / "dev" / "ops" / "postmortems"
TEMPLATE = POSTMORTEMS_DIR / "TEMPLATE.md"
README = REPO_ROOT / "README.md"
RUNBOOK = REPO_ROOT / "dev" / "ops" / "runbook.md"

POSTMORTEM_LINK = "dev/ops/postmortems/2026-09-17-red-tests-went-unseen.md"
SLO_LINK = "dev/ops/slo.md"
RECORD_HEADING = "## Record an incident"


def headings(text: str) -> list[str]:
    """Every level-two heading line, exactly as written."""
    return [line.rstrip() for line in text.splitlines() if line.startswith("## ")]


def section_body(text: str, heading: str) -> str:
    """The lines under `heading` up to the next level-two heading; the heading must occur once."""
    lines = text.splitlines()
    starts = [index for index, line in enumerate(lines) if line.rstrip() == heading]
    assert len(starts) == 1, f"{heading!r} occurs {len(starts)} times"
    body: list[str] = []
    for line in lines[starts[0] + 1 :]:
        if line.startswith("## "):
            break
        body.append(line)
    return "\n".join(body)


def test_one_postmortem_besides_the_template_carries_every_template_heading() -> None:
    """Catches no real postmortem (only the template), a second stray file, and a postmortem
    that drops or renames a template section - "What went well" or the action items are the
    ones a hurried write-up skips."""
    template_headings = headings(TEMPLATE.read_text(encoding="utf-8"))
    assert template_headings, (
        "TEMPLATE.md has no level-two headings; this case asserts nothing"
    )

    written = sorted(
        path for path in POSTMORTEMS_DIR.glob("*.md") if path.name != TEMPLATE.name
    )
    assert len(written) == 1, [path.name for path in written]

    present = set(headings(written[0].read_text(encoding="utf-8")))
    missing = [heading for heading in template_headings if heading not in present]
    assert missing == [], f"{written[0].name} lacks {missing}"


def test_the_readme_links_the_postmortem_and_slo_md() -> None:
    """Catches the README naming the postmortem or the SLO page as prose with no link, or
    linking a path that is not the file: each must appear as a markdown link target."""
    text = README.read_text(encoding="utf-8")
    for target in (POSTMORTEM_LINK, SLO_LINK):
        assert re.search(rf"\]\({re.escape(target)}\)", text), (
            f"README.md does not link {target}"
        )
        assert (REPO_ROOT / target).is_file(), f"{target} does not exist"


def test_the_runbook_records_an_incident_with_postmortem_url() -> None:
    """Catches the section missing, and a section that describes recording an incident without
    the runnable command: the body must carry a `curl` against `/admin/incidents` that sets
    `postmortem_url`, which is how the console timeline gets its link."""
    body = section_body(RUNBOOK.read_text(encoding="utf-8"), RECORD_HEADING)

    for needle in ("curl", "/admin/incidents", "postmortem_url"):
        assert needle in body, f"{RECORD_HEADING!r} does not contain {needle!r}"
