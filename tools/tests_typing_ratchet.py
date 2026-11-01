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

The same baseline also holds a ceiling per file for ``# type: ignore``
comments, counted from comment tokens so text inside strings does not
count. A mypy error count alone would miss debt that an ignore silences,
so ignores follow the same rules: absent files must have none, no file may
gain one, and a ceiling above the count is stale.

``--write`` regenerates the baseline from the current counts and refuses
to raise any ceiling, so the recorded debt can only fall. Errors that mypy
reports in modules outside ``tests/`` reached through imports are not
counted here; those modules have their own gates.

Counts depend on the installed packages (an optional package with stubs can
remove an import error), so the baseline follows the CI environment built
from ``requirements/dev-lock.txt``; run ``--write`` in that environment.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tokenize
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "tools" / "tests_typing_baseline.json"
_ERROR_LINE = re.compile(r"^(?P<path>tests/[^:]+\.py):\d+: error: ")
_TYPE_IGNORE = re.compile(r"#\s*type:\s*ignore\b")


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


def count_ignores(root: Path) -> Counter[str]:
    """Return the number of ``# type: ignore`` comments per ``tests/`` file.

    Parameters
    ----------
    root : Path
        Repository root that holds ``tests/``.

    Returns
    -------
    Counter[str]
        Ignore-comment counts keyed by repository-relative POSIX path; files
        without ignores are absent.

    Raises
    ------
    ValueError
        If a test file cannot be tokenized.
    """
    counts: Counter[str] = Counter()
    for path in sorted((root / "tests").rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        try:
            with path.open("rb") as handle:
                tokens = list(tokenize.tokenize(handle.readline))
        except (SyntaxError, tokenize.TokenError) as exc:
            raise ValueError(f"cannot tokenize {relative}: {exc}") from exc
        found = sum(
            1
            for token in tokens
            if token.type == tokenize.COMMENT and _TYPE_IGNORE.search(token.string)
        )
        if found:
            counts[relative] = found
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


def _load_section(path: Path, key: str) -> dict[str, int] | None:
    """Return one ceiling section of the baseline, or ``None`` when absent.

    Raises
    ------
    ValueError
        If the section is not a mapping of paths to non-negative integers.
    """
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and key == "ignores" and key not in payload:
        return None
    section = payload.get(key) if isinstance(payload, dict) else None
    if not isinstance(section, dict) or not all(
        isinstance(name, str)
        and isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
        for name, value in section.items()
    ):
        raise ValueError(f"{path} must map test paths to non-negative integers")
    return dict(section)


def load_baseline(path: Path) -> dict[str, int] | None:
    """Return the recorded error ceilings, or ``None`` when the file is absent.

    Raises
    ------
    ValueError
        If the file is not a mapping of paths to non-negative integers.
    """
    return _load_section(path, "files")


def load_ignore_baseline(path: Path) -> dict[str, int] | None:
    """Return the recorded ignore-comment ceilings, or ``None`` if unrecorded.

    ``None`` means the file or its ``ignores`` section is absent; the first
    ``--write`` then records the current counts.

    Raises
    ------
    ValueError
        If the section is not a mapping of paths to non-negative integers.
    """
    return _load_section(path, "ignores")


def write_baseline(
    path: Path,
    ceilings: Mapping[str, int],
    ignores: Mapping[str, int] | None = None,
) -> None:
    """Write the error and ignore-comment ceilings with the rule they follow."""
    ignore_ceilings = dict(ignores or {})
    payload = {
        "description": (
            "mypy --strict error ceilings (files) and '# type: ignore' comment "
            "ceilings (ignores) per tests/ file; maintained by "
            "tools/tests_typing_ratchet.py. Absent files must have none; "
            "ceilings may only fall."
        ),
        "total": sum(ceilings.values()),
        "files": dict(ceilings),
        "ignore_total": sum(ignore_ceilings.values()),
        "ignores": ignore_ceilings,
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
    ignores = count_ignores(root)
    ceilings = load_baseline(baseline)
    ignore_ceilings = load_ignore_baseline(baseline)
    if args.write:
        try:
            new = lowered_baseline(current, ceilings)
        except ValueError as exc:
            print(f"test typing ratchet: {exc}", file=sys.stderr)
            return 1
        try:
            new_ignores = lowered_baseline(ignores, ignore_ceilings)
        except ValueError as exc:
            print(f"test typing ratchet: type: ignore {exc}", file=sys.stderr)
            return 1
        write_baseline(baseline, new, new_ignores)
        total = sum(new.values())
        ignore_total = sum(new_ignores.values())
        print(
            f"test typing ratchet: wrote {len(new)} ceilings, {total} errors, "
            f"{ignore_total} type: ignore comments"
        )
        return 0

    existing = {p.relative_to(root).as_posix() for p in (root / "tests").rglob("*.py")}
    report = compare(current, ceilings or {}, existing)
    ignore_report = compare(ignores, ignore_ceilings or {}, existing)
    total = sum(current.values())
    ignore_total = sum(ignores.values())
    rewrite = "run `python tools/tests_typing_ratchet.py --write`: "
    if report.increased:
        print(
            "test typing ratchet: new strict-typing errors in "
            + _describe(report.increased),
            file=sys.stderr,
        )
    if ignore_report.increased:
        print(
            "test typing ratchet: new type: ignore comments in "
            + _describe(ignore_report.increased),
            file=sys.stderr,
        )
    if report.stale:
        print(
            "test typing ratchet: stale ceilings, " + rewrite + _describe(report.stale),
            file=sys.stderr,
        )
    if ignore_report.stale:
        print(
            "test typing ratchet: stale type: ignore ceilings, "
            + rewrite
            + _describe(ignore_report.stale),
            file=sys.stderr,
        )
    passed = report.passed and ignore_report.passed
    if passed:
        print(
            f"test typing ratchet: OK ({total} recorded errors, "
            f"{ignore_total} type: ignore comments, none new)"
        )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
