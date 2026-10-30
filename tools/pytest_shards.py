# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Duration-weighted pytest module scheduling

"""Partition test modules by measured duration and prove the shards are complete.

The discovery list decides which modules run; timings only decide where. A
module without timing history gets the median measured duration, and a cold
run without any history deals the sorted list round-robin. ``verify`` refuses
a run whose shard selections omit, duplicate or invent a module.
"""

from __future__ import annotations

import argparse
import fnmatch
import heapq
import json
import math
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

SCHEMA = 1
"""Version of the timing manifest and of the selection receipt."""


def discover(root: Path, tests: str, ignored: Sequence[str]) -> list[str]:
    """Return the sorted test modules pytest collects under ``tests``.

    Parameters
    ----------
    root : pathlib.Path
        Repository root; returned paths are relative to it, POSIX style.
    tests : str
        The collected directory, relative to ``root``.
    ignored : sequence of str
        Exclusions relative to ``root``. An entry with ``*``, ``?`` or ``[``
        is an ``--ignore-glob`` pattern matched against the whole path; any
        other entry is an ``--ignore`` path that excludes itself and, as a
        directory, everything below it.

    Returns
    -------
    list of str
        Every ``test_*.py`` or ``*_test.py`` module not ignored.

    Raises
    ------
    ValueError
        If the directory holds no selected module.
    """
    base = root / tests
    found = sorted(
        path.relative_to(root).as_posix()
        for pattern in ("test_*.py", "*_test.py")
        for path in base.rglob(pattern)
        if path.is_file()
    )
    selected = [
        name for name in found if not any(_ignored(name, item) for item in ignored)
    ]
    if not selected:
        raise ValueError(f"no test modules discovered under {tests}")
    return selected


def _ignored(name: str, item: str) -> bool:
    """Return whether one exclusion removes ``name``, as pytest applies it."""
    if any(character in item for character in "*?["):
        return fnmatch.fnmatchcase(name, item)
    prefix = item.rstrip("/")
    return name == prefix or name.startswith(prefix + "/")


def read_timings(path: Path | None) -> dict[str, float]:
    """Read scheduling hints; refuse malformed or nonfinite measurements.

    Parameters
    ----------
    path : pathlib.Path or None
        Timing manifest. ``None`` or a missing file is a cold run.

    Returns
    -------
    dict of str to float
        Positive seconds keyed by repository-relative module path.

    Raises
    ------
    ValueError
        If the manifest has another schema or any entry is not a positive
        finite number of seconds.
    """
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ValueError("unsupported pytest timing manifest")
    durations = data.get("durations")
    if not isinstance(durations, dict):
        raise ValueError("timing durations must be an object")
    result: dict[str, float] = {}
    for name, seconds in cast(dict[str, object], durations).items():
        if (
            not name
            or isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            raise ValueError("timing entries need module names and positive seconds")
        result[name] = float(seconds)
    return result


def partition(
    modules: Sequence[str], count: int, timings: dict[str, float]
) -> list[list[str]]:
    """Assign every module to exactly one shard, longest measured first.

    Parameters
    ----------
    modules : sequence of str
        The complete discovery list; timings never change membership.
    count : int
        Number of shards, each nonempty.
    timings : dict of str to float
        Measured seconds per module. Unknown modules get the median.

    Returns
    -------
    list of list of str
        Sorted module lists, one per shard, deterministic for equal inputs.

    Raises
    ------
    ValueError
        If ``count`` does not fit the list or a module repeats or is empty.
    """
    if not 1 <= count <= len(modules) or len(set(modules)) != len(modules):
        raise ValueError("shard count must fit a unique discovery list")
    if not all(modules):
        raise ValueError("module names must be nonempty")
    if not timings:
        ordered = sorted(modules)
        return [ordered[index::count] for index in range(count)]
    default = statistics.median(timings.values())
    shards: list[list[str]] = [[] for _ in range(count)]
    loads = [(0.0, index) for index in range(count)]
    heapq.heapify(loads)
    for name in sorted(modules, key=lambda item: (-timings.get(item, default), item)):
        load, index = heapq.heappop(loads)
        shards[index].append(name)
        heapq.heappush(loads, (load + timings.get(name, default), index))
    return [sorted(shard) for shard in shards]


def verify(modules: Sequence[str], receipts: Sequence[Path], count: int) -> None:
    """Refuse unless the shard receipts select every module exactly once.

    Parameters
    ----------
    modules : sequence of str
        The complete discovery list of this run.
    receipts : sequence of pathlib.Path
        One selection receipt per shard, as written by ``select``.
    count : int
        The number of shards the run declared.

    Raises
    ------
    ValueError
        If a shard is missing, repeated or foreign, or the union of the
        selections differs from the discovery list.
    """
    seen: dict[str, int] = {}
    shards: set[int] = set()
    for receipt in receipts:
        data = json.loads(receipt.read_text(encoding="utf-8"))
        if (
            not isinstance(data, dict)
            or data.get("schema") != SCHEMA
            or data.get("count") != count
        ):
            raise ValueError(f"{receipt}: foreign or malformed selection receipt")
        index = data.get("shard")
        selected = data.get("modules")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 1 <= index <= count
            or index in shards
        ):
            raise ValueError(f"{receipt}: missing, repeated or foreign shard index")
        if not isinstance(selected, list) or not selected:
            raise ValueError(f"{receipt}: shard selected no module")
        shards.add(index)
        for name in cast(list[object], selected):
            if not isinstance(name, str) or name in seen:
                raise ValueError(f"{receipt}: module {name!r} is duplicated or invalid")
            seen[name] = index
    if shards != set(range(1, count + 1)):
        raise ValueError(f"shards {sorted(set(range(1, count + 1)) - shards)} missing")
    expected = set(modules)
    omitted = sorted(expected - set(seen))
    invented = sorted(set(seen) - expected)
    if omitted or invented:
        raise ValueError(f"selection differs: omitted {omitted}, invented {invented}")


def record(
    reports: Sequence[Path], modules: Sequence[str], previous: Path | None
) -> str:
    """Return the next timing manifest from this run's measured durations.

    Parameters
    ----------
    reports : sequence of pathlib.Path
        Per-shard manifests written by :mod:`tools.pytest_timing_plugin`
        during this run.
    modules : sequence of str
        The live discovery list; history for removed modules is dropped.
    previous : pathlib.Path or None
        The manifest the run scheduled with; modules this run did not time
        keep their earlier measurement.

    Returns
    -------
    str
        The serialised manifest, newline-terminated.

    Raises
    ------
    ValueError
        If a report is malformed, times a module twice, or no live module
        was timed.
    """
    live = set(modules)
    durations = {k: v for k, v in read_timings(previous).items() if k in live}
    measured: dict[str, float] = {}
    for report in reports:
        for name, seconds in read_timings(report).items():
            if name in measured:
                raise ValueError(f"{report}: module {name} timed by two shards")
            if name in live:
                measured[name] = seconds
    if not measured:
        raise ValueError("no live module durations found in the shard reports")
    durations.update(measured)
    return json.dumps({"schema": SCHEMA, "durations": durations}, sort_keys=True) + "\n"


def _arguments() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--tests", default="tests")
    parser.add_argument("--ignore", action="append", default=[])
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select", help="write one shard's modules")
    select.add_argument("--shard", required=True, help="K/N, 1 <= K <= N")
    select.add_argument("--timings", type=Path)
    select.add_argument("--receipt", type=Path, required=True)
    check = commands.add_parser("verify", help="prove the shards are complete")
    check.add_argument("--count", type=int, required=True)
    check.add_argument("receipts", type=Path, nargs="+")
    timing = commands.add_parser("record", help="write the next timing manifest")
    timing.add_argument("--previous", type=Path)
    timing.add_argument("--output", type=Path, required=True)
    timing.add_argument("reports", type=Path, nargs="+")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one scheduler command.

    Parameters
    ----------
    argv : sequence of str or None
        Command arguments; ``None`` reads this process's arguments.

    Returns
    -------
    int
        Zero on success, two when an input is refused.
    """
    args = _arguments().parse_args(argv)
    try:
        modules = discover(args.root, args.tests, args.ignore)
        if args.command == "select":
            index_text, _, count_text = args.shard.partition("/")
            index, count = int(index_text), int(count_text)
            if not 1 <= index <= count:
                raise ValueError("shard must be K/N with 1 <= K <= N")
            selected = partition(modules, count, read_timings(args.timings))[index - 1]
            receipt = {
                "schema": SCHEMA,
                "shard": index,
                "count": count,
                "modules": selected,
            }
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            args.receipt.write_text(
                json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
            )
            sys.stdout.write("\n".join(selected) + "\n")
        elif args.command == "verify":
            verify(modules, args.receipts, args.count)
            total = len(modules)
            print(f"{total} modules selected exactly once across {args.count} shards")
        else:
            manifest = record(args.reports, modules, args.previous)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(manifest, encoding="utf-8")
    except (OSError, ValueError) as error:
        print(f"pytest shard refusal: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["discover", "main", "partition", "read_timings", "record", "verify"]
