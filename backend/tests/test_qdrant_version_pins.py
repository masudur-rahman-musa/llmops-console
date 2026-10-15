"""The Qdrant server pins this plan moves are checked by reading the files they live in.

Every case reads a committed file (compose, ci.yml, the k8s manifest, the
runbook) or `backend/pyproject.toml`; none runs a service, so none carries the
`db` marker.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE = REPO_ROOT / "docker-compose.yml"
CI = REPO_ROOT / ".github" / "workflows" / "ci.yml"
QDRANT_MANIFEST = REPO_ROOT / "infra" / "k8s" / "platform" / "qdrant.yaml"
RUNBOOK = REPO_ROOT / "dev" / "ops" / "runbook.md"
PYPROJECT = REPO_ROOT / "backend" / "pyproject.toml"

# The four live places the deployed Qdrant server image is pinned. Kept in one
# tuple so the pin list has a single home for both pin cases.
DEPLOYED_PIN_FILES = (COMPOSE, CI, QDRANT_MANIFEST, RUNBOOK)

WANTED_PIN = "qdrant/qdrant:v1.19.2"
OLD_PIN = "v1.12.4"


def test_every_deployed_qdrant_pin_is_v1_19_2() -> None:
    """Catches a pin left behind: any of the four deployed files not reading
    the exact `qdrant/qdrant:v1.19.2` image string.
    """
    for path in DEPLOYED_PIN_FILES:
        text = path.read_text(encoding="utf-8")
        assert WANTED_PIN in text, path


def test_no_deployed_qdrant_pin_is_left_on_v1_12_4() -> None:
    """Catches a half-done move: the old `v1.12.4` still present in any of the
    four deployed files.
    """
    for path in DEPLOYED_PIN_FILES:
        text = path.read_text(encoding="utf-8")
        assert OLD_PIN not in text, path


def test_the_gateway_declares_no_qdrant_client_dependency() -> None:
    """Catches a `qdrant-client` sneaking into the gateway's dependencies, which
    would mean a client version has to match too, not only the server image.
    """
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = data.get("project", {})
    specs: list[str] = list(project.get("dependencies", []))
    for group in project.get("optional-dependencies", {}).values():
        specs.extend(group)
    names = [spec.split(";")[0].strip().lower() for spec in specs]
    for name in names:
        assert not name.startswith("qdrant-client"), name
