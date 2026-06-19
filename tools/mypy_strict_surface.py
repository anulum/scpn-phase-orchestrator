# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — mypy strict-surface inventory

"""Deterministic inventory of the mypy strict surface for regression review.

The strict type-check gate is only as strong as the relaxations it allows. This
module reads ``pyproject.toml`` and emits a normalised, sorted snapshot of the
base strictness plus every ``[[tool.mypy.overrides]]`` block and the exact
settings it relaxes. A committed snapshot plus the companion test
(``tests/test_mypy_strict_surface.py``) means any new relaxation — a fresh
``ignore_errors``, a widened ``follow_imports`` — fails CI until the snapshot is
updated on purpose, so strict-coverage erosion is visible in review rather than
silent.

Usage::

    python tools/mypy_strict_surface.py            # print the current surface
    python tools/mypy_strict_surface.py --check     # exit 1 if it drifted
    python tools/mypy_strict_surface.py --write      # rewrite the snapshot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import tomllib

_REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT_PATH = _REPO_ROOT / "pyproject.toml"
SNAPSHOT_PATH = _REPO_ROOT / "tests" / "data" / "mypy_strict_surface.json"


def compute_strict_surface(pyproject_path: Path) -> dict[str, Any]:
    """Return the normalised mypy strict surface read from ``pyproject.toml``.

    Parameters
    ----------
    pyproject_path : Path
        Path to the ``pyproject.toml`` whose ``[tool.mypy]`` table is read.

    Returns
    -------
    dict[str, Any]
        A mapping with ``strict`` (the base flag) and ``overrides`` (a sorted
        list of ``{"modules": [...], "settings": {...}}`` entries, one per
        ``[[tool.mypy.overrides]]`` block, with every key except ``module``).
    """
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    mypy = data.get("tool", {}).get("mypy", {})
    raw_overrides = mypy.get("overrides", [])
    overrides: list[dict[str, Any]] = []
    for override in raw_overrides:
        module = override.get("module", [])
        modules = sorted(module) if isinstance(module, list) else [module]
        settings = {
            key: value for key, value in sorted(override.items()) if key != "module"
        }
        overrides.append({"modules": modules, "settings": settings})
    overrides.sort(key=lambda entry: (entry["modules"], sorted(entry["settings"])))
    return {"strict": bool(mypy.get("strict", False)), "overrides": overrides}


def render_snapshot(surface: dict[str, Any]) -> str:
    """Return the canonical JSON text for a strict-surface mapping.

    Parameters
    ----------
    surface : dict[str, Any]
        A surface mapping as returned by :func:`compute_strict_surface`.

    Returns
    -------
    str
        Deterministic, sorted, newline-terminated JSON.
    """
    return json.dumps(surface, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Run the strict-surface CLI.

    Parameters
    ----------
    argv : list[str] | None
        Argument vector; ``None`` uses ``sys.argv``.

    Returns
    -------
    int
        Process exit code: ``0`` on success or a clean snapshot, ``1`` when
        ``--check`` finds drift.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if the committed snapshot is out of date",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="rewrite the committed snapshot from the current pyproject",
    )
    args = parser.parse_args(argv)
    text = render_snapshot(compute_strict_surface(PYPROJECT_PATH))
    if args.write:
        SNAPSHOT_PATH.write_text(text, encoding="utf-8")
        sys.stdout.write(f"wrote {SNAPSHOT_PATH}\n")
        return 0
    if args.check:
        current = (
            SNAPSHOT_PATH.read_text(encoding="utf-8") if SNAPSHOT_PATH.exists() else ""
        )
        if current != text:
            sys.stderr.write(
                "mypy strict surface drifted from the committed snapshot; "
                "review the relaxation and run "
                "`python tools/mypy_strict_surface.py --write` if intentional.\n"
            )
            return 1
        return 0
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
