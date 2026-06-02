# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Baseline comparison

"""Compare benchmark results against a stored baseline.

Usage:
    python bench/compare_baseline.py bench/baseline.json bench/current.json

Exit code 1 if any configuration regresses >20%.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REGRESSION_THRESHOLD_PCT = 20.0
BENCHMARK_KEY_FIELDS = ("n_osc", "method", "backend")
BENCHMARK_VALUE_FIELD = "us_per_step"


def _positive_finite_float(value: Any, *, field: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"benchmark field {field!r} must be numeric") from exc
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"benchmark field {field!r} must be finite and > 0")
    return out


def _is_benchmark_entry(entry: Any) -> bool:
    return (
        isinstance(entry, dict)
        and all(field in entry for field in BENCHMARK_KEY_FIELDS)
        and BENCHMARK_VALUE_FIELD in entry
    )


def _extract_results(data: Any) -> list[dict[str, Any]]:
    """Return comparable benchmark entries from supported JSON layouts.

    Historical baselines are category maps such as ``{"upde": [...]}``,
    whereas fresh benchmark runs use ``{"results": [...]}``.  Only records
    with the full regression key are comparable; unrelated science-kernel
    sections stay in the file but are not part of this guard.
    """
    if isinstance(data, list):
        return [entry for entry in data if _is_benchmark_entry(entry)]
    if not isinstance(data, dict):
        return []
    if isinstance(data.get("results"), list):
        return [entry for entry in data["results"] if _is_benchmark_entry(entry)]

    results: list[dict[str, Any]] = []
    for value in data.values():
        if isinstance(value, list):
            results.extend(entry for entry in value if _is_benchmark_entry(entry))
    return results


def _benchmark_key(entry: dict[str, Any]) -> tuple[int, str, str]:
    return (int(entry["n_osc"]), str(entry["method"]), str(entry["backend"]))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail closed on benchmark regressions against a baseline.",
    )
    parser.add_argument("baseline", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument(
        "--threshold-pct",
        type=float,
        default=REGRESSION_THRESHOLD_PCT,
        help="Allowed slowdown percentage before a comparison fails.",
    )
    parser.add_argument(
        "--allow-missing-current",
        action="store_true",
        help=(
            "Allow baseline entries missing from the current run. "
            "Use only for deliberate local partial benchmark runs."
        ),
    )
    return parser.parse_args(argv)


def main() -> int:
    try:
        args = _parse_args(sys.argv[1:])
    except SystemExit as exc:
        return int(exc.code)

    if not math.isfinite(args.threshold_pct) or args.threshold_pct < 0.0:
        print("threshold percentage must be finite and >= 0", file=sys.stderr)
        return 2

    with args.baseline.open() as f:
        baseline = _extract_results(json.load(f))
    with args.current.open() as f:
        current = _extract_results(json.load(f))
    if not baseline:
        print("\nNo comparable baseline benchmark entries found.")
        return 1
    if not current:
        print("\nNo comparable current benchmark entries found.")
        return 1

    base_map: dict[tuple[int, str, str], float] = {}
    for entry in baseline:
        key = _benchmark_key(entry)
        if key in base_map:
            label = f"N={key[0]:4d} {key[1]:>5s} {key[2]:>6s}"
            print(f"\nDuplicate baseline benchmark entry: {label}")
            return 1
        base_map[key] = _positive_finite_float(
            entry[BENCHMARK_VALUE_FIELD],
            field=BENCHMARK_VALUE_FIELD,
        )

    failures = []
    comparisons = 0
    current_keys: set[tuple[int, str, str]] = set()
    for entry in current:
        key = _benchmark_key(entry)
        if key in current_keys:
            label = f"N={key[0]:4d} {key[1]:>5s} {key[2]:>6s}"
            print(f"\nDuplicate current benchmark entry: {label}")
            return 1
        current_keys.add(key)
        if key not in base_map:
            continue
        base_val = base_map[key]
        cur_val = _positive_finite_float(
            entry[BENCHMARK_VALUE_FIELD],
            field=BENCHMARK_VALUE_FIELD,
        )
        comparisons += 1
        pct = (cur_val - base_val) / base_val * 100.0
        status = "PASS" if pct <= args.threshold_pct else "FAIL"
        label = f"N={key[0]:4d} {key[1]:>5s} {key[2]:>6s}"
        msg = (
            f"  {status}  {label}  {base_val:8.1f} -> {cur_val:8.1f}"
            f" us/step  ({pct:+.1f}%)"
        )
        print(msg)
        if pct > args.threshold_pct:
            failures.append((label, pct))

    missing_current = sorted(set(base_map) - current_keys)
    if missing_current and not args.allow_missing_current:
        print("\nCurrent benchmark run is missing baseline configurations:")
        for key in missing_current:
            print(f"  N={key[0]:4d} {key[1]:>5s} {key[2]:>6s}")
        return 1

    if failures:
        n = len(failures)
        print(f"\n{n} regression(s) exceeded {args.threshold_pct}% threshold:")
        for label, pct in failures:
            print(f"  {label}: {pct:+.1f}%")
        return 1

    if comparisons == 0:
        print("\nNo overlapping benchmark entries found.")
        return 1

    print("\nAll benchmarks within regression threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
