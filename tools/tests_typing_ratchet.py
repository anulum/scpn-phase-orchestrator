# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Strict-typing ratchet for the test suite

"""No-increase ratchet for ``mypy --strict`` errors in ``tests/``.

The test suite carries historical typing debt, so a blanket strict gate
would fail and a blanket exemption would hide new debt. This tool runs
``mypy --strict tests`` with the repository configuration and counts the
errors reported in each ``tests/`` file. A JSON baseline holds the ceiling
for every file that still has debt:

* a file that is not in the baseline, including every new test file, must
  have no errors;
* no file may exceed its ceiling;
* a ceiling above the current count, or the entry of a file that no
  longer exists, is stale and fails until ``--write`` lowers or drops it.

``--write`` regenerates the baseline from the current counts and refuses
to raise any ceiling, so the recorded debt can only fall. Errors that mypy
reports in modules outside ``tests/`` reached through imports are not
counted here; those modules have their own gates.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "tools" / "tests_typing_baseline.json"
_ERROR_LINE = re.compile(r"^(?P<path>tests/[^:]+\.py):\d+: error: ")


@dataclass(frozen=True)
class RatchetReport:
    """Outcome of comparing current counts with the recorded ceilings."""

    increased: dict[str, tuple[int, int]] = field(default_factory=dict)
    stale: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Return whether no ceiling was exceeded and none is stale."""
        return not self.increased and not self.stale


def count_errors(mypy_output: str) -> Counter[str]:
    """Return the number of mypy errors per ``tests/`` file.

    Parameters
    ----------
    mypy_output : str
        Combined mypy output, one diagnostic per line.

    Returns
    -------
    Counter[str]
        Error counts keyed by repository-relative POSIX path.
    """
    counts: Counter[str] = Counter()
    for line in mypy_output.splitlines():
        match = _ERROR_LINE.match(line.replace("\\", "/"))
        if match is not None:
            counts[match.group("path")] += 1
    return counts


def compare(
    current: Mapping[str, int],
    ceilings: Mapping[str, int],
    existing: set[str],
) -> RatchetReport:
    """Compare current counts with the ceilings.

    Parameters
    ----------
    current : Mapping[str, int]
        Current error counts per file.
    ceilings : Mapping[str, int]
        Recorded ceilings per file; an absent file has ceiling zero.
    existing : set[str]
        Test files present in the checkout.

    Returns
    -------
    RatchetReport
        Files above their ceiling, and ceilings that are stale.
    """
    increased = {
        path: (count, ceilings.get(path, 0))
        for path, count in current.items()
        if count > ceilings.get(path, 0)
    }
    stale = {
        path: (current.get(path, 0), ceiling)
        for path, ceiling in ceilings.items()
        if path not in existing or current.get(path, 0) < ceiling
    }
    return RatchetReport(increased=increased, stale=stale)


def lowered_baseline(
    current: Mapping[str, int], ceilings: Mapping[str, int] | None
) -> dict[str, int]:
    """Return the new ceilings, refusing any increase over ``ceilings``.

    Parameters
    ----------
    current : Mapping[str, int]
        Current error counts per file.
    ceilings : Mapping[str, int] | None
        Existing ceilings, or ``None`` when no baseline exists yet.

    Returns
    -------
    dict[str, int]
        Sorted ceilings for every file that still has errors.

    Raises
    ------
    ValueError
        If any file would get a higher ceiling than it has now.
    """
    if ceilings is not None:
        raised = {
            path: (count, ceilings.get(path, 0))
            for path, count in current.items()
            if count > ceilings.get(path, 0)
        }
        if raised:
            raise ValueError(f"refusing to raise ceilings: {_describe(raised)}")
    return {path: current[path] for path in sorted(current) if current[path] > 0}


def load_baseline(path: Path) -> dict[str, int] | None:
    """Return the recorded ceilings, or ``None`` when the file is absent.

    Raises
    ------
    ValueError
        If the file is not a mapping of paths to non-negative integers.
    """
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    files = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(files, dict) or not all(
        isinstance(key, str)
        and isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
        for key, value in files.items()
    ):
        raise ValueError(f"{path} must map test paths to non-negative integers")
    return dict(files)


def write_baseline(path: Path, ceilings: Mapping[str, int]) -> None:
    """Write the ceilings with a short description of the rule."""
    payload = {
        "description": (
            "mypy --strict error ceilings per tests/ file; maintained by "
            "tools/tests_typing_ratchet.py. Absent files must have no errors; "
            "ceilings may only fall."
        ),
        "total": sum(ceilings.values()),
        "files": dict(ceilings),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_mypy(root: Path) -> str:
    """Run ``mypy --strict tests`` in ``root`` and return its output.

    Raises
    ------
    RuntimeError
        If mypy did not complete a type check (exit status other than 0 or 1).
    """
    completed = subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", "--no-error-summary", "tests"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode not in (0, 1):
        raise RuntimeError(
            f"mypy failed with status {completed.returncode}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed.stdout + completed.stderr


def _describe(entries: Mapping[str, tuple[int, int]]) -> str:
    """Return ``path (now/ceiling)`` items joined for a message."""
    return ", ".join(f"{p} ({now}/{cap})" for p, (now, cap) in sorted(entries.items()))


def main(argv: Sequence[str] | None = None) -> int:
    """Check or rewrite the test typing baseline.

    Parameters
    ----------
    argv : Sequence[str] | None
        Command-line arguments; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        ``0`` when the ratchet holds or the baseline was written, else ``1``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument(
        "--write", action="store_true", help="lower or drop ceilings to the counts"
    )
    args = parser.parse_args(argv)
    root: Path = args.root
    baseline: Path = args.baseline or root / "tools" / "tests_typing_baseline.json"

    current = count_errors(run_mypy(root))
    ceilings = load_baseline(baseline)
    if args.write:
        try:
            new = lowered_baseline(current, ceilings)
        except ValueError as exc:
            print(f"test typing ratchet: {exc}", file=sys.stderr)
            return 1
        write_baseline(baseline, new)
        total = sum(new.values())
        print(f"test typing ratchet: wrote {len(new)} ceilings, {total} errors")
        return 0

    existing = {p.relative_to(root).as_posix() for p in (root / "tests").rglob("*.py")}
    report = compare(current, ceilings or {}, existing)
    total = sum(current.values())
    if report.increased:
        print(
            "test typing ratchet: new strict-typing errors in "
            + _describe(report.increased),
            file=sys.stderr,
        )
    if report.stale:
        print(
            "test typing ratchet: stale ceilings, run "
            "`python tools/tests_typing_ratchet.py --write`: "
            + _describe(report.stale),
            file=sys.stderr,
        )
    if report.passed:
        print(f"test typing ratchet: OK ({total} recorded errors, none new)")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
