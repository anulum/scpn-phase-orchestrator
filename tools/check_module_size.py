#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Module size review guard

"""Module-size review guard for the Python source tree.

Line count is a *proxy*, never the gate: single-responsibility code is fine at
any length, and multi-responsibility code must be split at any length (see
``docs/module_size_policy.md``). This guard therefore surfaces files
for a responsibility review rather than rejecting them on a line budget.

Two tiers drive the report:

* ``REVIEW_THRESHOLD`` (default 900) — informational; glance at the file and
  confirm it is still one responsibility.
* ``SPLIT_THRESHOLD`` (default 1200) — strong signal of multiple
  responsibilities; default-split unless an AST call-graph shows one cohesive
  cluster. A file above this threshold that has been reviewed and kept whole is
  recorded in the allowlist with a justification.

Default invocation prints a report and exits 0 (a warning, not an error).
``--check`` turns the split tier into a ratchet: it fails only when a file
exceeds ``SPLIT_THRESHOLD`` and is absent from the allowlist, so new god-modules
are blocked while reviewed-cohesive files stay grandfathered.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC_ROOT = REPO_ROOT / "src" / "scpn_phase_orchestrator"
DEFAULT_ALLOWLIST = REPO_ROOT / "tools" / "large_module_allowlist.json"

REVIEW_THRESHOLD = 900
SPLIT_THRESHOLD = 1200

# Generated or vendored trees are exempt: their size is not a design choice.
_EXEMPT_PARTS = frozenset({"grpc_gen"})


@dataclass(frozen=True)
class ModuleSize:
    """A source module path with its physical line count."""

    dotted_path: str
    relative_path: str
    line_count: int


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _is_exempt(parts: tuple[str, ...]) -> bool:
    return any(part in _EXEMPT_PARTS for part in parts)


def _dotted_path(src_root: Path, module_path: Path) -> str:
    rel = module_path.relative_to(src_root).with_suffix("")
    parts = [p for p in rel.parts if p != "__init__"]
    return ".".join(["scpn_phase_orchestrator", *parts]) or "scpn_phase_orchestrator"


def collect_modules(src_root: Path) -> list[ModuleSize]:
    """Return every non-exempt Python module under ``src_root`` with line counts."""
    display_base = src_root.parents[1]  # the directory containing ``src/``
    modules: list[ModuleSize] = []
    for path in sorted(src_root.rglob("*.py")):
        rel_parts = path.relative_to(src_root).parts
        if _is_exempt(rel_parts):
            continue
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        modules.append(
            ModuleSize(
                dotted_path=_dotted_path(src_root, path),
                relative_path=str(path.relative_to(display_base)),
                line_count=line_count,
            )
        )
    return modules


def load_allowlist(path: Path) -> dict[str, str]:
    """Return the reviewed-large-module allowlist as ``{dotted_path: reason}``."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("reviewed_large_modules", [])
    allowlist: dict[str, str] = {}
    for entry in entries:
        module = entry["module"]
        reason = entry.get("reason", "")
        if not reason.strip():
            raise ValueError(f"allowlisted module {module!r} needs a non-empty reason")
        allowlist[module] = reason
    return allowlist


def review_tier(modules: Iterable[ModuleSize]) -> list[ModuleSize]:
    return [m for m in modules if REVIEW_THRESHOLD <= m.line_count < SPLIT_THRESHOLD]


def split_tier(modules: Iterable[ModuleSize]) -> list[ModuleSize]:
    return [m for m in modules if m.line_count >= SPLIT_THRESHOLD]


def unreviewed_split_modules(
    modules: Iterable[ModuleSize], allowlist: dict[str, str]
) -> list[ModuleSize]:
    """Return split-tier modules absent from the allowlist (ratchet failures)."""
    return [m for m in split_tier(modules) if m.dotted_path not in allowlist]


def format_report(modules: list[ModuleSize], allowlist: dict[str, str]) -> str:
    review = sorted(review_tier(modules), key=lambda m: m.line_count, reverse=True)
    split = sorted(split_tier(modules), key=lambda m: m.line_count, reverse=True)
    lines = [
        "Module size review guard",
        f"  review threshold : {REVIEW_THRESHOLD} lines (confirm one responsibility)",
        f"  split threshold  : {SPLIT_THRESHOLD} lines (default-split unless cohesive)",
        f"  scanned modules  : {len(modules)}",
        "",
        f"Split tier (>= {SPLIT_THRESHOLD}): {len(split)}",
    ]
    for m in split:
        mark = "allowlisted" if m.dotted_path in allowlist else "REVIEW"
        lines.append(f"  [{mark}] {m.line_count:5} {m.relative_path}")
        if m.dotted_path in allowlist:
            lines.append(f"            reason: {allowlist[m.dotted_path]}")
    lines.append("")
    lines.append(
        f"Review tier ({REVIEW_THRESHOLD}-{SPLIT_THRESHOLD - 1}): {len(review)}"
    )
    for m in review:
        lines.append(f"            {m.line_count:5} {m.relative_path}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", default=str(DEFAULT_SRC_ROOT))
    parser.add_argument("--allowlist", default=str(DEFAULT_ALLOWLIST))
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero when a split-tier module is not allowlisted",
    )
    args = parser.parse_args(argv)

    modules = collect_modules(_resolve(args.src_root))
    allowlist = load_allowlist(_resolve(args.allowlist))
    print(format_report(modules, allowlist))

    if args.check:
        offenders = unreviewed_split_modules(modules, allowlist)
        if offenders:
            print("")
            print(
                f"FAIL: {len(offenders)} module(s) over {SPLIT_THRESHOLD} lines "
                "need a responsibility review — split them or add a justified "
                "entry to the allowlist:"
            )
            for m in offenders:
                print(f"  {m.line_count:5} {m.relative_path}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
