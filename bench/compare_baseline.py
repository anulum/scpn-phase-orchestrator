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

import json
import sys
from pathlib import Path
from typing import Any

REGRESSION_THRESHOLD_PCT = 20.0
BENCHMARK_KEY_FIELDS = ("n_osc", "method", "backend")
BENCHMARK_VALUE_FIELD = "us_per_step"


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


def main() -> int:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <baseline.json> <current.json>", file=sys.stderr)
        return 2

    with Path(sys.argv[1]).open() as f:
        baseline = _extract_results(json.load(f))
    with Path(sys.argv[2]).open() as f:
        current = _extract_results(json.load(f))

    base_map: dict[tuple[int, str, str], float] = {}
    for entry in baseline:
        key = _benchmark_key(entry)
        base_map[key] = float(entry["us_per_step"])

    failures = []
    comparisons = 0
    for entry in current:
        key = _benchmark_key(entry)
        if key not in base_map:
            continue
        base_val = base_map[key]
        cur_val = float(entry["us_per_step"])
        if base_val <= 0:
            continue
        comparisons += 1
        pct = (cur_val - base_val) / base_val * 100.0
        status = "PASS" if pct <= REGRESSION_THRESHOLD_PCT else "FAIL"
        label = f"N={key[0]:4d} {key[1]:>5s} {key[2]:>6s}"
        msg = (
            f"  {status}  {label}  {base_val:8.1f} -> {cur_val:8.1f}"
            f" us/step  ({pct:+.1f}%)"
        )
        print(msg)
        if pct > REGRESSION_THRESHOLD_PCT:
            failures.append((label, pct))

    if failures:
        n = len(failures)
        print(f"\n{n} regression(s) exceeded {REGRESSION_THRESHOLD_PCT}% threshold:")
        for label, pct in failures:
            print(f"  {label}: {pct:+.1f}%")
        return 1

    if comparisons == 0:
        print("\nNo overlapping benchmark entries found.")
        return 0

    print("\nAll benchmarks within regression threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
