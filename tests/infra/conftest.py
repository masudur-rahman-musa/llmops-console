r"""The one `kubectl kustomize infra/k8s` of the session, shared by every `infra` test module.

An earlier change decision G. Two files now read the same built stream - `test_k8s_base.py` for the
resource envelope and `test_k8s_platform.py` for what `platform/` adds - and a file-local
fixture in each would mean two builds per session (a project rule says built ONCE) and two
copies of the skip and its message that would have to stay in step. Both live here instead,
so the skip condition, the failure text and the parse exist exactly once.

The fixture body moved verbatim: it is requested by name, so its relocation is invisible to
every case, and no assertion in `test_k8s_base.py` changed with it.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

#: This file's own location, never the working directory - `pytest` is run from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: The directory the root kustomization lives in, and the only argument the build takes.
K8S_ROOT = REPO_ROOT / "infra" / "k8s"


@pytest.fixture(scope="session")
def built() -> list[dict[str, Any]]:
    """`kubectl kustomize infra/k8s`, run ONCE for the whole session.

    Skips only when `kubectl` is absent from `PATH` (decision D). A non-zero exit fails, with
    kubectl's stderr in the message, because a build that cannot run is the failure this file
    exists to report - not a reason to report nothing.
    """
    if shutil.which("kubectl") is None:
        pytest.skip("kubectl is not on PATH; CI's `k8s` job is the copy that cannot skip")
    completed = subprocess.run(
        ["kubectl", "kustomize", str(K8S_ROOT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"`kubectl kustomize {K8S_ROOT}` exited {completed.returncode}:\n{completed.stderr}"
    )
    documents = [
        document
        for document in yaml.safe_load_all(completed.stdout)
        if isinstance(document, dict) and document
    ]
    assert documents, f"`kubectl kustomize {K8S_ROOT}` emitted no objects"
    return documents
