# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
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

REGRESSION_THRESHOLD_PCT = 20.0


def _extract_results(data):
    if isinstance(data, list):
        return data
    return data.get("results", data)


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
        key = (entry["n_osc"], entry["method"], entry["backend"])
        base_map[key] = entry["us_per_step"]

    failures = []
    for entry in current:
        key = (entry["n_osc"], entry["method"], entry["backend"])
        if key not in base_map:
            continue
        base_val = base_map[key]
        cur_val = entry["us_per_step"]
        if base_val <= 0:
            continue
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

    print("\nAll benchmarks within regression threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
