#!/usr/bin/env python3
r"""Build one measurement option's effective routing policy and provider folder.

    infra/scripts/tradeoff_policy.py <option-id> <output path> <providers output dir>

`backend/config/routing.yaml` deep-merged with `infra/tradeoff/<option-id>.yaml`, written
to the output path. The shipped policy is read and never written: the merge works
on a copy, and the only places this script writes are the two named on the
command line (`test_routing_yaml_is_byte_identical_after_a_merge`,
`test_the_merge_does_not_mutate_the_shipped_document`).

The provider folder is the shipped `backend/config/providers/` plus any file in
`infra/tradeoff/providers/` whose provider the effective policy names.
That is how the GPU option brings its own `vllm` provider: the shipped folder has no
vLLM file, so the live system never depends on a host that exists for four hours.

Four partial overlays rather than four whole policies: a full
copy drifts from the shipped file silently, and each partial then says only what
makes its option different. `backend/config/routing.ci.yaml` set the precedent.

The merge is recursive over MAPPINGS and replaces LISTS whole. Both halves are
load-bearing. A top-level `dict.update` would drop every tenant an overlay does not
mention; a merge that extended lists would leave the shipped three candidates in
place beside the overlay's one, and every option would measure whichever provider
answered first (`test_the_merge_is_recursive_over_mappings`,
`test_the_merge_replaces_a_list_whole`).

The option id names both the overlay file and, in `tradeoff_run.sh`, the evidence
directory, so it is a store-shaping input: it is checked with `re.fullmatch`
against `opsconsole.evals.tradeoff.OPTION_ID_PATTERN` - the same constant the
manifest validates its ids with - before any path is built (a project rule and
4; `test_a_malformed_option_id_is_refused_before_any_file_is_touched`).

A pattern refusal and a missing overlay both exit 2, and the MESSAGE is what tells
them apart: the first prints the pattern, the second prints the path it looked for.
That is what makes the inclusive boundary observable, since a well-formed id with
no overlay has no other visible difference from a malformed one
(`test_the_inclusive_length_boundaries_pass_the_pre_check`).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Final, NoReturn

import yaml

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
ROUTING: Final[Path] = REPO_ROOT / "backend" / "config" / "routing.yaml"
SHIPPED_PROVIDERS: Final[Path] = REPO_ROOT / "backend" / "config" / "providers"
OVERLAY_DIR: Final[Path] = REPO_ROOT / "infra" / "tradeoff"
OPTION_PROVIDERS: Final[Path] = OVERLAY_DIR / "providers"

OPTION_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"[a-z][a-z0-9-]{0,31}")
"""The option id, matched with `re.fullmatch` and never `.match`.

Written out here rather than imported from `opsconsole.evals.tradeoff`, which
declares the same constant for the manifest: this script is run with the system
interpreter from `tradeoff_run.sh`, where the package is not importable. The test
imports the package's copy and compares the two, so a drift is red
(`test_a_malformed_option_id_is_refused_before_any_file_is_touched`)."""

EXIT_REFUSED: Final[int] = 2
"""Every refusal, as `argparse` and `ops-eval`'s own usage errors exit."""


# Any is the honest type: both sides are any parsed YAML value.
def deep_merge(base: Any, patch: Any) -> Any:  # noqa: ANN401
    """`patch` over `base`, recursively through mappings, lists replaced whole.

    Neither argument is mutated: a nested mapping present in both is merged into a
    new dict, so the caller's shipped document is still the shipped document
    afterwards (`test_the_merge_does_not_mutate_the_shipped_document`).
    """
    if not isinstance(base, dict) or not isinstance(patch, dict):
        return patch
    merged: dict[Any, Any] = dict(base)
    for key, value in patch.items():
        merged[key] = deep_merge(base[key], value) if key in base else value
    return merged


def checked_option_id(value: str) -> str:
    """The id, or a refusal naming the pattern it failed."""
    if OPTION_ID_PATTERN.fullmatch(value) is None:
        die(f"{value!r} is not an option id matching {OPTION_ID_PATTERN.pattern}")
    return value


def die(message: str) -> NoReturn:
    """One line on stderr and exit 2. Never a traceback: the caller is a shell script."""
    print(f"tradeoff-policy: {message}", file=sys.stderr)
    raise SystemExit(EXIT_REFUSED)


def load_mapping(path: Path) -> dict[str, Any]:
    """One YAML document as a mapping, or a refusal naming the path."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        die(f"{path}: cannot be read: {exc.strerror}")
    document = yaml.safe_load(text)
    if not isinstance(document, dict):
        die(f"{path}: is not a YAML mapping")
    return document


def effective_policy(option_id: str) -> dict[str, Any]:
    """The shipped policy with this option's overlay merged over it.

    Both files are read before `main` opens the output, so a missing overlay leaves
    nothing behind (`test_a_well_formed_id_with_no_overlay_names_the_path_and_writes_nothing`).
    """
    return deep_merge(load_mapping(ROUTING), load_mapping(OVERLAY_DIR / f"{option_id}.yaml"))


def named_providers(policy: dict[str, Any]) -> set[str]:
    """Every provider a candidate of `policy` names, through an alias or an override."""
    names: set[str] = set()
    for alias in (policy.get("aliases") or {}).values():
        names.update(str(c["provider"]) for c in alias.get("candidates", []))
    for tenant in (policy.get("tenants") or {}).values():
        for candidates in (tenant.get("overrides") or {}).values():
            names.update(str(c["provider"]) for c in candidates)
    return names


def provider_files(policy: dict[str, Any]) -> list[Path]:
    """The shipped provider files, plus the option's own ones the policy names."""
    shipped = sorted(SHIPPED_PROVIDERS.glob("*.yaml"))
    taken = {path.stem for path in shipped}
    names = named_providers(policy)
    own = sorted(
        path
        for path in OPTION_PROVIDERS.glob("*.yaml")
        if path.stem in names and path.stem not in taken
    )
    return shipped + own


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(
            f"usage: {Path(sys.argv[0]).name} <option-id> <output path> <providers output dir>",
            file=sys.stderr,
        )
        return EXIT_REFUSED
    option_id, output, providers = checked_option_id(argv[0]), Path(argv[1]), Path(argv[2])
    policy = effective_policy(option_id)
    files = provider_files(policy)
    output.write_text(yaml.safe_dump(policy, sort_keys=False), encoding="utf-8")
    providers.mkdir(parents=True, exist_ok=True)
    for path in files:
        (providers / path.name).write_bytes(path.read_bytes())
    print(f"tradeoff-policy: wrote {output} and {providers}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
