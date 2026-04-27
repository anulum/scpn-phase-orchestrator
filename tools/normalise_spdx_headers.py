# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SPDX header normaliser

"""Normalise SPDX headers from 6-line merged variant to canonical 7-line format.

Splits the merged opener
    <prefix> SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
into two lines
    <prefix> SPDX-License-Identifier: AGPL-3.0-or-later
    <prefix> Commercial license available

Also corrects the ``protoscience@<typo>`` Contact email typo to
``protoscience@anulum.li`` wherever it occurs in the file.

Usage::

    python tools/normalise_spdx_headers.py --dry-run
    python tools/normalise_spdx_headers.py --apply
    python tools/normalise_spdx_headers.py --verify
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

EXCLUDE_DIRS = frozenset(
    {
        ".git",
        "BACKUP",
        "ARCHIVE",
        "node_modules",
        "site",
        "wasm-pkg",
        "target",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        "htmlcov",
        ".pytest_cache",
        "build",
        "dist",
        "ml.350.context",
        "experiments_archive",
    }
)

VENV_PREFIXES = ("venv", ".venv")

INCLUDE_SUFFIXES = frozenset(
    {
        ".py",
        ".rs",
        ".sh",
        ".yml",
        ".yaml",
        ".toml",
        ".md",
        ".js",
        ".ts",
        ".v",
    }
)

PREFIXED_PATTERN = re.compile(
    r"^(?P<prefix>(?:#|//) ?)"
    r"SPDX-License-Identifier: AGPL-3\.0-or-later \| Commercial license available\s*$",
)

MERGED_BARE = (
    "SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available"
)
# Defined via concatenation so the script's own source does not contain the
# literal typo and therefore does not self-match during a sweep.
TYPO = "protoscience@" + "any" + "lum.li"
TYPO_FIX = "protoscience@anulum.li"

HEADER_SCAN_LIMIT = 10


@dataclass
class FileResult:
    path: Path
    spdx_split: bool
    typo_fixes: int

    @property
    def changed(self) -> bool:
        return self.spdx_split or self.typo_fixes > 0


def _is_excluded(rel_parts: tuple[str, ...]) -> bool:
    for part in rel_parts:
        if part in EXCLUDE_DIRS:
            return True
        if part.startswith(VENV_PREFIXES):
            return True
    return False


def iter_repo_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(root).parts
        if _is_excluded(rel_parts):
            continue
        if path.suffix not in INCLUDE_SUFFIXES:
            continue
        yield path


def split_prefixed(line: str) -> tuple[str, str] | None:
    match = PREFIXED_PATTERN.match(line)
    if match is None:
        return None
    prefix = match.group("prefix")
    return (
        f"{prefix}SPDX-License-Identifier: AGPL-3.0-or-later\n",
        f"{prefix}Commercial license available\n",
    )


def split_bare(line: str) -> tuple[str, str] | None:
    if MERGED_BARE not in line:
        return None
    if line.lstrip().startswith(("#", "//")):
        return None
    leading = line[: len(line) - len(line.lstrip())]
    return (
        f"{leading}SPDX-License-Identifier: AGPL-3.0-or-later\n",
        f"{leading}Commercial license available\n",
    )


def normalise_file(path: Path, apply: bool) -> FileResult:
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines(keepends=True)

    new_lines: list[str] = []
    spdx_split = False
    typo_fixes = 0

    for idx, line in enumerate(lines):
        if not spdx_split and idx < HEADER_SCAN_LIMIT:
            split = split_prefixed(line) or split_bare(line)
            if split is not None:
                first, second = split
                if TYPO in first:
                    first = first.replace(TYPO, TYPO_FIX)
                    typo_fixes += 1
                if TYPO in second:
                    second = second.replace(TYPO, TYPO_FIX)
                    typo_fixes += 1
                new_lines.extend([first, second])
                spdx_split = True
                continue
        if TYPO in line:
            line = line.replace(TYPO, TYPO_FIX)
            typo_fixes += 1
        new_lines.append(line)

    result = FileResult(path=path, spdx_split=spdx_split, typo_fixes=typo_fixes)

    if result.changed and apply:
        tmp = path.with_suffix(path.suffix + ".spdxtmp")
        tmp.write_text("".join(new_lines), encoding="utf-8")
        tmp.replace(path)

    return result


def report_change(result: FileResult, apply: bool) -> None:
    rel = result.path.relative_to(REPO)
    parts = []
    if result.spdx_split:
        parts.append("SPDX split")
    if result.typo_fixes:
        parts.append(f"{result.typo_fixes} typo fix")
    verb = "APPLIED" if apply else "WOULD CHANGE"
    print(f"{verb}: {rel} — {'; '.join(parts)}")


def verify(root: Path) -> int:
    remaining: list[Path] = []
    typo_files: list[Path] = []
    for path in iter_repo_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            print(f"SKIP {path}: {exc}", file=sys.stderr)
            continue
        head_lines = text.splitlines()[:HEADER_SCAN_LIMIT]
        if any(MERGED_BARE in line for line in head_lines):
            remaining.append(path)
        if TYPO in text:
            typo_files.append(path)

    if remaining:
        print(f"FAIL: {len(remaining)} files still contain merged SPDX line in header:")
        for path in remaining:
            print(f"  {path.relative_to(root)}")
    if typo_files:
        print(f"FAIL: {len(typo_files)} files still contain '{TYPO}':")
        for path in typo_files:
            print(f"  {path.relative_to(root)}")

    if not remaining and not typo_files:
        print("All files normalised (no merged SPDX line in header, no anylum typo).")
        return 0
    return 1


def run_normalise(apply: bool) -> int:
    total = 0
    typos = 0
    for path in iter_repo_files(REPO):
        try:
            result = normalise_file(path, apply=apply)
        except (OSError, UnicodeDecodeError) as exc:
            print(f"SKIP {path}: {exc}", file=sys.stderr)
            continue
        if result.changed:
            total += 1
            typos += result.typo_fixes
            report_change(result, apply)
    verb = "changed" if apply else "would change"
    print(f"\nSummary: {total} files {verb}; {typos} typo fixes.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dry-run", action="store_true", help="report changes without writing"
    )
    group.add_argument("--apply", action="store_true", help="write changes")
    group.add_argument(
        "--verify",
        action="store_true",
        help="check no merged variant remains",
    )
    args = parser.parse_args()

    if args.verify:
        return verify(REPO)
    return run_normalise(apply=args.apply)


if __name__ == "__main__":
    sys.exit(main())
