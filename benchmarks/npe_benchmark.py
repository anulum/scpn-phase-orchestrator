# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — NPE multi-backend benchmark

"""Per-backend wall-clock benchmark for ``monitor/npe.py``
``compute_npe``."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor import npe as npe_mod
from scpn_phase_orchestrator.monitor.npe import (
    ACTIVE_BACKEND,
    AVAILABLE_BACKENDS,
    compute_npe,
    phase_distance_matrix,
)

TWO_PI = 2.0 * np.pi
BACKEND_ORDER = ("rust", "mojo", "julia", "go", "python")
PARITY_TOLERANCES = {
    "rust": 1.0e-12,
    "mojo": 1.0e-9,
    "julia": 1.0e-12,
    "go": 1.0e-12,
    "python": 0.0,
}


def _bench(backend: str, phases: NDArray[np.floating], calls: int) -> float:
    elapsed, _, _ = _bench_with_outputs(backend, phases, calls, max_radius=None)
    return elapsed


def _bench_with_outputs(
    backend: str,
    phases: NDArray[np.floating],
    calls: int,
    *,
    max_radius: float | None,
) -> tuple[float, NDArray[np.float64], float]:
    saved = npe_mod.ACTIVE_BACKEND
    try:
        npe_mod.ACTIVE_BACKEND = backend
        distance = phase_distance_matrix(phases)
        npe = compute_npe(phases, max_radius=max_radius)
        t0 = time.perf_counter()
        for _ in range(calls):
            distance = phase_distance_matrix(phases)
            npe = compute_npe(phases, max_radius=max_radius)
        return time.perf_counter() - t0, distance, float(npe)
    finally:
        npe_mod.ACTIVE_BACKEND = saved


def bench_at(n: int, calls: int) -> dict:
    rng = np.random.default_rng(42)
    phases = rng.uniform(0.0, TWO_PI, size=n)
    row: dict = {"n": n, "calls": calls, "available": AVAILABLE_BACKENDS}
    for backend in AVAILABLE_BACKENDS:
        t = _bench(backend, phases, calls)
        row[f"{backend}_ms_per_call"] = (t / calls) * 1000.0
    return row


def _array_sha256(array: NDArray[np.floating]) -> str:
    contiguous = np.ascontiguousarray(array, dtype=np.float64)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _scalar_sha256(value: float) -> str:
    payload = json.dumps(float(value), separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _backend_status(backend: str) -> tuple[bool, str]:
    if backend in AVAILABLE_BACKENDS:
        return True, ""
    return False, f"{backend} backend was not resolved by monitor.npe"


def _validate_int_control(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{name} must be an integer")
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return parsed


def _validate_radius_control(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError("max_radius must be a finite non-negative real")
    radius = float(value)
    if not np.isfinite(radius) or radius < 0.0 or radius > np.pi:
        raise ValueError("max_radius must be finite and lie in [0, pi]")
    return radius


def benchmark_npe_polyglot_parity_gate(
    *,
    n: int = 20,
    calls: int = 1,
    seed: int = 2026,
    max_radius: float | None = None,
) -> dict[str, object]:
    """Record NPE parity across every declared language backend slot."""

    n = _validate_int_control(n, name="n", minimum=2)
    calls = _validate_int_control(calls, name="calls", minimum=1)
    seed = _validate_int_control(seed, name="seed", minimum=0)
    radius = _validate_radius_control(max_radius)

    rng = np.random.default_rng(seed)
    phases = np.ascontiguousarray(rng.uniform(0.0, TWO_PI, size=n), dtype=np.float64)

    t0 = time.perf_counter()
    _, reference_distance, reference_npe = _bench_with_outputs(
        "python",
        phases,
        1,
        max_radius=radius,
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
                    "distance_matrix_sha256": None,
                    "npe_sha256": None,
                    "max_distance_abs_error": None,
                    "npe_abs_error": None,
                    "max_abs_error": None,
                    "tolerance": tolerance,
                    "parity_passed": False,
                    "unavailable_reason": reason,
                }
            )
            continue

        available_backend_count += 1
        elapsed, distance, npe_value = _bench_with_outputs(
            backend,
            phases,
            calls,
            max_radius=radius,
        )
        max_distance_error = float(np.max(np.abs(distance - reference_distance)))
        npe_error = abs(float(npe_value) - reference_npe)
        max_abs_error = max(max_distance_error, npe_error)
        parity_passed = max_abs_error <= tolerance
        parity_checked_count += 1
        parity_pass_count += int(parity_passed)
        records.append(
            {
                "backend": backend,
                "status": "available",
                "ms_per_call": (elapsed / calls) * 1000.0,
                "distance_matrix_sha256": _array_sha256(distance),
                "npe_sha256": _scalar_sha256(npe_value),
                "max_distance_abs_error": max_distance_error,
                "npe_abs_error": npe_error,
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
        "require_distance_matrix_contract": True,
        "require_python_reference": True,
        "require_unit_interval_npe": True,
    }
    acceptance_passed = (
        len(records) == len(BACKEND_ORDER)
        and any(
            record["backend"] == "python" and record["status"] == "available"
            for record in records
        )
        and parity_pass_count == parity_checked_count
        and 0.0 <= reference_npe <= 1.0
    )
    benchmark_payload = {
        "n": n,
        "calls": calls,
        "seed": seed,
        "max_radius": radius,
        "records": records,
        "thresholds": thresholds,
        "reference_distance_sha256": _array_sha256(reference_distance),
        "reference_npe_sha256": _scalar_sha256(reference_npe),
    }
    benchmark_sha = hashlib.sha256(
        json.dumps(benchmark_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    return {
        "suite": "npe_polyglot_parity_gate",
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
        "max_radius": radius if radius is not None else float(np.pi),
        "reference_npe": reference_npe,
        "reference_distance_sha256": _array_sha256(reference_distance),
        "reference_npe_sha256": _scalar_sha256(reference_npe),
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
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--calls", type=int, default=50)
    parser.add_argument(
        "--parity-gate",
        action="store_true",
        help="emit deterministic all-backend parity-gate JSON",
    )
    args = parser.parse_args()

    if args.parity_gate:
        result = benchmark_npe_polyglot_parity_gate(
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
