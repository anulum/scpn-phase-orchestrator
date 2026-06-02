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

Exit code 1 if any configuration regresses beyond the configured relative
and absolute significance budgets.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REGRESSION_THRESHOLD_PCT = 20.0
REGRESSION_MIN_ABSOLUTE_US = 100.0
BENCHMARK_KEY_FIELDS = ("n_osc", "method", "backend")
BENCHMARK_VALUE_FIELD = "us_per_step"


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"benchmark JSON must not contain non-finite token {value!r}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"benchmark JSON contains duplicate key {key!r}")
        out[key] = value
    return out


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(
            f,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )


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
    raw_n = entry["n_osc"]
    if isinstance(raw_n, bool) or not isinstance(raw_n, int):
        raise ValueError("benchmark field 'n_osc' must be a positive integer")
    if raw_n <= 0:
        raise ValueError("benchmark field 'n_osc' must be a positive integer")

    method = entry["method"]
    backend = entry["backend"]
    if not isinstance(method, str) or not method.strip():
        raise ValueError("benchmark field 'method' must be a non-empty string")
    if not isinstance(backend, str) or not backend.strip():
        raise ValueError("benchmark field 'backend' must be a non-empty string")
    return (raw_n, method, backend)


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
        "--min-absolute-us",
        type=float,
        default=REGRESSION_MIN_ABSOLUTE_US,
        help=(
            "Minimum absolute slowdown in microseconds required before a "
            "relative regression is considered significant."
        ),
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
    if not math.isfinite(args.min_absolute_us) or args.min_absolute_us < 0.0:
        print("minimum absolute slowdown must be finite and >= 0", file=sys.stderr)
        return 2

    try:
        baseline = _extract_results(_load_json(args.baseline))
        current = _extract_results(_load_json(args.current))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
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
        delta_us = cur_val - base_val
        pct = delta_us / base_val * 100.0
        significant = pct > args.threshold_pct and delta_us >= args.min_absolute_us
        status = "FAIL" if significant else "PASS"
        label = f"N={key[0]:4d} {key[1]:>5s} {key[2]:>6s}"
        msg = (
            f"  {status}  {label}  {base_val:8.1f} -> {cur_val:8.1f}"
            f" us/step  ({pct:+.1f}%, {delta_us:+.1f} us)"
        )
        print(msg)
        if pct > args.threshold_pct and not significant:
            print(
                f"        tolerated: absolute slowdown below "
                f"{args.min_absolute_us:.1f} us significance floor"
            )
        if significant:
            failures.append((label, pct, delta_us))

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
