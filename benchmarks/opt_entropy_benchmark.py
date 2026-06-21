# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — OPT-entropy multi-backend benchmark

"""Per-backend wall-clock benchmark for ``monitor/opt_entropy.py``."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor import opt_entropy as oe_mod
from scpn_phase_orchestrator.monitor.opt_entropy import (
    ACTIVE_BACKEND,
    AVAILABLE_BACKENDS,
    ordinal_pattern_sequence,
    transition_entropy,
)

BACKEND_ORDER = ("rust", "mojo", "julia", "go", "python")
PARITY_TOLERANCES = {
    "rust": 1.0e-12,
    "mojo": 1.0e-9,
    "julia": 1.0e-12,
    "go": 1.0e-12,
    "python": 0.0,
}


def _bench_with_outputs(
    backend: str,
    series: NDArray[np.floating],
    dimension: int,
    delay: int,
    calls: int,
) -> tuple[float, NDArray[np.int64], float]:
    saved = oe_mod.ACTIVE_BACKEND
    try:
        oe_mod.ACTIVE_BACKEND = backend
        codes = ordinal_pattern_sequence(series, dimension, delay)
        entropy = transition_entropy(series, dimension, delay)
        t0 = time.perf_counter()
        for _ in range(calls):
            codes = ordinal_pattern_sequence(series, dimension, delay)
            entropy = transition_entropy(series, dimension, delay)
        return time.perf_counter() - t0, codes, float(entropy)
    finally:
        oe_mod.ACTIVE_BACKEND = saved


def _bench(backend: str, series: NDArray[np.floating], calls: int) -> float:
    elapsed, _, _ = _bench_with_outputs(backend, series, 3, 1, calls)
    return elapsed


def bench_at(n: int, calls: int) -> dict:
    rng = np.random.default_rng(42)
    series = rng.standard_normal(n)
    row: dict = {"n": n, "calls": calls, "available": AVAILABLE_BACKENDS}
    for backend in AVAILABLE_BACKENDS:
        elapsed = _bench(backend, series, calls)
        row[f"{backend}_ms_per_call"] = (elapsed / calls) * 1000.0
    return row


def _array_sha256(array: NDArray[np.integer]) -> str:
    contiguous = np.ascontiguousarray(array, dtype=np.int64)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _scalar_sha256(value: float) -> str:
    payload = json.dumps(float(value), separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _backend_status(backend: str) -> tuple[bool, str]:
    if backend in AVAILABLE_BACKENDS:
        return True, ""
    return False, f"{backend} backend was not resolved by monitor.opt_entropy"


def _validate_int_control(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return parsed


def benchmark_opt_entropy_polyglot_parity_gate(
    *,
    n: int = 256,
    calls: int = 1,
    seed: int = 2026,
    dimension: int = 3,
    delay: int = 1,
) -> dict[str, object]:
    """Record OPT-entropy parity across every declared language backend slot."""

    n = _validate_int_control(n, name="n", minimum=4)
    calls = _validate_int_control(calls, name="calls", minimum=1)
    seed = _validate_int_control(seed, name="seed", minimum=0)
    dimension = _validate_int_control(dimension, name="dimension", minimum=2)
    delay = _validate_int_control(delay, name="delay", minimum=1)

    rng = np.random.default_rng(seed)
    series = np.ascontiguousarray(rng.standard_normal(n), dtype=np.float64)

    t0 = time.perf_counter()
    _, reference_codes, reference_entropy = _bench_with_outputs(
        "python", series, dimension, delay, 1
    )

    records: list[dict[str, object]] = []
    parity_pass_count = 0
    parity_checked_count = 0
    available_backend_count = 0

    for backend in BACKEND_ORDER:
        tolerance = PARITY_TOLERANCES[backend]
        available, reason = _backend_status(backend)
        if not available:
            records.append(
                {
                    "backend": backend,
                    "status": "unavailable",
                    "ms_per_call": None,
                    "codes_sha256": None,
                    "entropy_sha256": None,
                    "max_code_abs_error": None,
                    "entropy_abs_error": None,
                    "max_abs_error": None,
                    "tolerance": tolerance,
                    "parity_passed": False,
                    "unavailable_reason": reason,
                }
            )
            continue

        available_backend_count += 1
        elapsed, codes, entropy_value = _bench_with_outputs(
            backend, series, dimension, delay, calls
        )
        max_code_error = float(np.max(np.abs(codes - reference_codes)))
        entropy_error = abs(float(entropy_value) - reference_entropy)
        max_abs_error = max(max_code_error, entropy_error)
        parity_passed = max_code_error == 0.0 and entropy_error <= tolerance
        parity_checked_count += 1
        parity_pass_count += int(parity_passed)
        records.append(
            {
                "backend": backend,
                "status": "available",
                "ms_per_call": (elapsed / calls) * 1000.0,
                "codes_sha256": _array_sha256(codes),
                "entropy_sha256": _scalar_sha256(entropy_value),
                "max_code_abs_error": max_code_error,
                "entropy_abs_error": entropy_error,
                "max_abs_error": max_abs_error,
                "tolerance": tolerance,
                "parity_passed": parity_passed,
                "unavailable_reason": "",
            }
        )

    wall_time = time.perf_counter() - t0
    thresholds = {
        "backend_order": list(BACKEND_ORDER),
        "max_mojo_abs_error": PARITY_TOLERANCES["mojo"],
        "max_native_abs_error": PARITY_TOLERANCES["rust"],
        "require_all_available_parity": True,
        "require_all_declared_backend_records": True,
        "require_exact_ordinal_codes": True,
        "require_python_reference": True,
        "require_unit_interval_entropy": True,
    }
    acceptance_passed = (
        len(records) == len(BACKEND_ORDER)
        and any(
            record["backend"] == "python" and record["status"] == "available"
            for record in records
        )
        and parity_pass_count == parity_checked_count
        and 0.0 <= reference_entropy <= 1.0
    )
    benchmark_payload = {
        "n": n,
        "calls": calls,
        "seed": seed,
        "dimension": dimension,
        "delay": delay,
        "records": records,
        "thresholds": thresholds,
        "reference_codes_sha256": _array_sha256(reference_codes),
        "reference_entropy_sha256": _scalar_sha256(reference_entropy),
    }
    benchmark_sha = hashlib.sha256(
        json.dumps(benchmark_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    return {
        "suite": "opt_entropy_polyglot_parity_gate",
        "backend_count": len(records),
        "available_backend_count": available_backend_count,
        "unavailable_backend_count": len(records) - available_backend_count,
        "parity_checked_count": parity_checked_count,
        "parity_pass_count": parity_pass_count,
        "all_available_passed": int(parity_pass_count == parity_checked_count),
        "python_reference_present": 1,
        "n": n,
        "calls": calls,
        "seed": seed,
        "dimension": dimension,
        "delay": delay,
        "reference_entropy": reference_entropy,
        "reference_codes_sha256": _array_sha256(reference_codes),
        "reference_entropy_sha256": _scalar_sha256(reference_entropy),
        "benchmark_sha256": benchmark_sha,
        "wall_time_s": wall_time,
        "steps_per_second": parity_checked_count / wall_time if wall_time else 0.0,
        "acceptance_passed": int(acceptance_passed),
        "acceptance_thresholds_json": json.dumps(thresholds, sort_keys=True),
        "backend_records_json": json.dumps(records, sort_keys=True),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 1024, 4096])
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument(
        "--parity-gate",
        action="store_true",
        help="emit deterministic all-backend parity-gate JSON",
    )
    args = parser.parse_args()

    if args.parity_gate:
        result = benchmark_opt_entropy_polyglot_parity_gate(
            n=args.sizes[0],
            calls=args.calls,
        )
        text = json.dumps(result, indent=2, sort_keys=True)
        print(text)
        if args.output:
            args.output.write_text(text + "\n", encoding="utf-8")
        return 0

    print(f"Active: {ACTIVE_BACKEND}  Available: {AVAILABLE_BACKENDS}\n")
    header = f"{'N':>6}"
    for b in AVAILABLE_BACKENDS:
        header += f" {b + '_ms':>12}"
    print(header)
    print("-" * len(header))
    results: list[dict] = []
    for n in args.sizes:
        row = bench_at(n, args.calls)
        results.append(row)
        line = f"{n:>6}"
        for b in AVAILABLE_BACKENDS:
            line += f" {row[f'{b}_ms_per_call']:>12.4f}"
        print(line)
    if args.output:
        args.output.write_text(
            json.dumps({"results": results}, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
