r"""The public documents - README, Azure page and CUSTOMIZING.

Text reads over committed files; no build, no service, no network. Every case resolves
its links against the repository itself, so a renamed file is a red test rather than a
silently stale document.
"""

from __future__ import annotations

import re
from pathlib import Path

#: This file's own location, never the working directory - `pytest` runs from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[3]

README = REPO_ROOT / "README.md"
CUSTOMIZING = REPO_ROOT / "docs/customizing.md"
AZURE = REPO_ROOT / "docs" / "azure.md"

#: The thirteen level-two headings, in order. Compared as a list.
README_HEADINGS = [
    "## The problem",
    "## Video and screenshots",
    "## How it fits together",
    "## The seven guarantees",
    "## What broke",
    "## How the evals work",
    "## Self-hosted vs API tradeoff",
    "## The decisions",
    "## SLOs",
    "## Scaling",
    "## Quickstart",
    "## Azure",
    "## AWS",
]

#: The seven guarantees and the test file that pins each, all committed today.
GUARANTEE_TESTS = [
    "backend/tests/api/test_fallback.py",
    "backend/tests/redaction/test_redactor.py",
    "backend/tests/limits/test_budget.py",
    "backend/tests/limits/test_ratelimit.py",
    "backend/tests/cache/test_key.py",
    "backend/tests/evals/test_gate.py",
    "backend/tests/storage/test_cost_ledger.py",
]

AZURE_LINK = "docs/azure.md"
AWS_LINK = "docs/aws-evidence/"

#: The three claims `docs/azure.md` must make, by literal phrase.
AZURE_CLAIMS = ("recorded fixtures", "ships disabled", "configuration change")
AZURE_ADAPTER = "backend/app/providers/azure_openai.py"
AZURE_ROUTING_ROW = "azure.yaml"

CUSTOMIZING_MUST_EDIT = re.compile(r"only two things", re.IGNORECASE)
MUST_EDIT_FILES = {"backend/config/routing.yaml"}
#: The provider folder, the other must-edit thing (a folder, so not a `.yaml`).
MUST_EDIT_FOLDER = "backend/config/providers/"
#: Any `.yaml` or `.py` path written on the must-edit line, backticked or bare.
YAML_OR_PY_PATH = re.compile(r"[A-Za-z0-9_./-]+\.(?:yaml|py)")

MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
BACKTICKED = re.compile(r"`([^`]+)`")
ABSOLUTE_LINK_PREFIXES = ("http://", "https://", "mailto:", "#")


def read(path: Path) -> str:
    """The file's text; a missing file fails here with its path rather than an OSError."""
    assert path.is_file(), f"{path.relative_to(REPO_ROOT)} does not exist"
    return path.read_text(encoding="utf-8")


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


def link_targets(text: str) -> list[str]:
    """Every markdown link target, in order and with duplicates kept."""
    return [match.group(1).strip() for match in MARKDOWN_LINK.finditer(text)]


def relative_targets(text: str) -> list[str]:
    """Link targets that name a path in this repository, with any `#fragment` stripped."""
    targets: list[str] = []
    for target in link_targets(text):
        if target.startswith(ABSOLUTE_LINK_PREFIXES):
            continue
        path = target.split("#", 1)[0]
        if path:
            targets.append(path)
    return targets


def looks_like_path(token: str) -> bool:
    """A backticked token naming a file or a directory rather than a value or a header."""
    return "/" in token or token.endswith((".py", ".yaml", ".md"))


def backticked_paths(text: str) -> list[str]:
    """Every backticked token that looks like a path, in order."""
    return [token for token in BACKTICKED.findall(text) if looks_like_path(token)]


def missing_paths(targets: list[str]) -> list[str]:
    """Those targets that do not resolve to a committed path under the repository root."""
    return sorted({target for target in targets if not (REPO_ROOT / target).exists()})


def test_the_readme_carries_the_thirteen_headings_in_order() -> None:
    """Catches a README that keeps its old section order, drops a required section (the
    problem statement, the video, the decisions), or inserts a fourteenth level-two heading
    between two required ones: the comparison is a list equality, so order and extras both
    fail."""
    assert headings(read(README)) == README_HEADINGS


def test_the_seven_guarantees_link_the_seven_tests() -> None:
    """Catches a guarantees section that claims a guarantee in prose with no test behind it,
    links six or eight tests, links the same test twice to reach seven, or links a test path
    that no longer exists after a rename."""
    body = section_body(read(README), "## The seven guarantees")
    linked = [target for target in relative_targets(body) if target.startswith("backend/tests/")]

    assert sorted(linked) == sorted(GUARANTEE_TESTS), linked
    assert missing_paths(linked) == []


def test_the_azure_and_aws_sections_link_their_documents() -> None:
    """Catches an Azure or AWS section that describes the path in prose instead of linking the
    document the request asked for, and a link to a document that was never written."""
    text = read(README)

    for heading, target in (("## Azure", AZURE_LINK), ("## AWS", AWS_LINK)):
        body = section_body(text, heading)
        assert target in relative_targets(body), f"{heading} does not link {target}"
        assert (REPO_ROOT / target).exists(), f"{target} does not exist"


def test_every_relative_readme_link_resolves() -> None:
    """Catches any relative link in the README pointing at a path that is not committed - a
    screenshot never added, a document renamed, a typo in a directory name - anywhere in the
    file, including the sections no other case reads."""
    assert missing_paths(relative_targets(read(README))) == []


def test_azure_says_the_path_is_disabled_and_configuration_only() -> None:
    """Catches an Azure page that leaves the reader thinking the path is live, or that it needs
    code: all three claims (recorded fixtures, ships disabled, configuration change) must be
    written, and the page must name the adapter module and the provider file a reader drops in."""
    text = read(AZURE)

    for claim in AZURE_CLAIMS:
        assert claim in text, f"docs/azure.md does not state {claim!r}"

    assert AZURE_ADAPTER in text
    assert (REPO_ROOT / AZURE_ADAPTER).is_file()
    assert AZURE_ROUTING_ROW in text


def test_customizing_names_exactly_two_must_edit_files() -> None:
    """Catches the template's central promise quietly growing a third must-edit file: the
    must-edit line is located by its own marker and every `.yaml` or `.py` path on it is read,
    so an added module or config makes the set unequal."""
    lines = [line for line in read(CUSTOMIZING).splitlines() if CUSTOMIZING_MUST_EDIT.search(line)]
    assert len(lines) == 1, lines

    named = set(YAML_OR_PY_PATH.findall(lines[0]))
    assert named == MUST_EDIT_FILES, sorted(named)
    assert MUST_EDIT_FOLDER in lines[0], lines[0]


def test_every_path_customizing_names_exists() -> None:
    """Catches an adopter being sent to a file that does not exist - a config path from an
    earlier layout, a directory that was renamed - in either a backticked path or a link."""
    text = read(CUSTOMIZING)
    assert missing_paths(backticked_paths(text) + relative_targets(text)) == []
