"""The postmortem (in docs/postmortems/) and its README link.

Text reads over committed files; no build, no service.
"""

from __future__ import annotations

import re
from pathlib import Path

#: This file's own location, never the working directory - `pytest` runs from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[3]

POSTMORTEMS_DIR = REPO_ROOT / "docs" / "postmortems"
TEMPLATE = POSTMORTEMS_DIR / "TEMPLATE.md"
README = REPO_ROOT / "README.md"

POSTMORTEM_LINK = "docs/postmortems/2026-09-17-red-tests-went-unseen.md"


def headings(text: str) -> list[str]:
    """Every level-two heading line, exactly as written."""
    return [line.rstrip() for line in text.splitlines() if line.startswith("## ")]


def test_one_postmortem_besides_the_template_carries_every_template_heading() -> None:
    """Catches no real postmortem (only the template), a second stray file, and a postmortem
    that drops or renames a template section - "What went well" or the action items are the
    ones a hurried write-up skips."""
    template_headings = headings(TEMPLATE.read_text(encoding="utf-8"))
    assert template_headings, "TEMPLATE.md has no level-two headings; this case asserts nothing"

    written = sorted(path for path in POSTMORTEMS_DIR.glob("*.md") if path.name != TEMPLATE.name)
    assert len(written) == 1, [path.name for path in written]

    present = set(headings(written[0].read_text(encoding="utf-8")))
    missing = [heading for heading in template_headings if heading not in present]
    assert missing == [], f"{written[0].name} lacks {missing}"


def test_the_readme_links_the_postmortem() -> None:
    """Catches the README naming the postmortem as prose with no link, or linking a path
    that is not the file: it must appear as a markdown link target."""
    text = README.read_text(encoding="utf-8")
    assert re.search(rf"\]\({re.escape(POSTMORTEM_LINK)}\)", text), (
        f"README.md does not link {POSTMORTEM_LINK}"
    )
    assert (REPO_ROOT / POSTMORTEM_LINK).is_file(), f"{POSTMORTEM_LINK} does not exist"
