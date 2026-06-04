# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Fractal-dimension multi-backend benchmark

"""Per-backend wall-clock benchmark for
``monitor.dimension.correlation_integral`` (full-pairs mode)."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor import dimension as dim_mod
from scpn_phase_orchestrator.monitor.dimension import (
    ACTIVE_BACKEND,
    AVAILABLE_BACKENDS,
    correlation_integral,
)

BACKEND_ORDER = ("rust", "mojo", "julia", "go", "python")
PARITY_TOLERANCES = {
    "rust": 1.0e-12,
    "mojo": 1.0e-9,
    "julia": 1.0e-12,
    "go": 1.0e-12,
    "python": 0.0,
}


def _bench(
    backend: str,
    traj: NDArray[np.floating],
    eps: NDArray[np.floating],
    calls: int,
) -> float:
    saved = dim_mod.ACTIVE_BACKEND
    try:
        dim_mod.ACTIVE_BACKEND = backend
        correlation_integral(traj, eps, max_pairs=1_000_000)
        t0 = time.perf_counter()
        for _ in range(calls):
            correlation_integral(traj, eps, max_pairs=1_000_000)
        return time.perf_counter() - t0
    finally:
        dim_mod.ACTIVE_BACKEND = saved


def bench_at(t: int, d: int, n_k: int, calls: int) -> dict:
    rng = np.random.default_rng(42)
    traj = rng.normal(0, 1, (t, d))
    eps = np.logspace(-1, 0.5, n_k)
    row: dict = {
        "T": t,
        "d": d,
        "n_k": n_k,
        "calls": calls,
        "available": AVAILABLE_BACKENDS,
        "boundary_contract": "exact_numpy_dimension_reference_validated",
    }
    for backend in AVAILABLE_BACKENDS:
        tt = _bench(backend, traj, eps, calls)
        row[f"{backend}_ms_per_call"] = (tt / calls) * 1000.0
    return row


def _validate_int_control(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _problem(
    t: int,
    d: int,
    n_k: int,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, 6.0 * np.pi, t, dtype=np.float64)
    basis = []
    for idx in range(d):
        freq = 0.45 + 0.23 * idx
        phase = rng.uniform(0.0, 2.0 * np.pi)
        basis.append(np.sin(freq * grid + phase) + 0.35 * np.cos(1.7 * freq * grid))
    trajectory = np.ascontiguousarray(np.column_stack(basis), dtype=np.float64)
    epsilons = np.ascontiguousarray(np.logspace(-1.3, 0.45, n_k), dtype=np.float64)
    spectrum = np.ascontiguousarray(
        np.array([0.42, 0.03, -0.20, -1.10, -2.40], dtype=np.float64),
        dtype=np.float64,
    )
    return trajectory, epsilons, spectrum


def _array_sha256(values: NDArray[np.floating]) -> str:
    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _scalar_sha256(value: float) -> str:
    payload = json.dumps(float(value), separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _backend_status(backend: str) -> tuple[bool, str]:
    if backend in AVAILABLE_BACKENDS:
        return True, ""
    return False, f"{backend} backend was not resolved by monitor.dimension"


def _pair_indices(t: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    pairs = dim_mod._prepare_pair_indices(t, t * (t - 1) // 2, seed=0)
    if pairs is None:
        raise ValueError("dimension parity gate requires at least two samples")
    return pairs


def _direct_outputs(
    backend: str,
    trajectory: NDArray[np.float64],
    epsilons: NDArray[np.float64],
    spectrum: NDArray[np.float64],
    *,
    idx_i: NDArray[np.int64],
    idx_j: NDArray[np.int64],
) -> tuple[NDArray[np.float64], float]:
    t, d = int(trajectory.shape[0]), int(trajectory.shape[1])
    flat = np.ascontiguousarray(trajectory.ravel(), dtype=np.float64)
    reference_ci = dim_mod._correlation_integral_exact_reference(
        trajectory,
        idx_i,
        idx_j,
        epsilons,
    )
    reference_ky = dim_mod._kaplan_yorke_exact_reference(spectrum)
    if backend == "python":
        return reference_ci, reference_ky

    loaded = dim_mod._load_backend(backend)
    ci_fn = loaded.get("ci")
    ky_fn = loaded.get("ky")
    if not callable(ci_fn) or not callable(ky_fn):
        raise ValueError(f"{backend} backend must expose ci and ky kernels")

    if backend == "rust":
        ci_raw = ci_fn(
            flat,
            t,
            d,
            epsilons,
            t * (t - 1) // 2,
            0,
        )
    else:
        ci_raw = ci_fn(flat, t, d, idx_i, idx_j, epsilons)
    ci = dim_mod._validate_ci_exact_reference(
        ci_raw,
        expected=reference_ci,
        atol=PARITY_TOLERANCES[backend],
    )
    ky = dim_mod._validate_ky_dimension(
        ky_fn(spectrum),
        n_exponents=int(spectrum.size),
        expected=reference_ky,
        atol=PARITY_TOLERANCES[backend],
    )
    return ci, ky


def _bench_with_output(
    backend: str,
    trajectory: NDArray[np.float64],
    epsilons: NDArray[np.float64],
    spectrum: NDArray[np.float64],
    *,
    idx_i: NDArray[np.int64],
    idx_j: NDArray[np.int64],
    calls: int,
) -> tuple[float, NDArray[np.float64], float]:
    _direct_outputs(
        backend,
        trajectory,
        epsilons,
        spectrum,
        idx_i=idx_i,
        idx_j=idx_j,
    )
    t0 = time.perf_counter()
    ci: NDArray[np.float64] | None = None
    ky: float | None = None
    for _ in range(calls):
        ci, ky = _direct_outputs(
            backend,
            trajectory,
            epsilons,
            spectrum,
            idx_i=idx_i,
            idx_j=idx_j,
        )
    if ci is None or ky is None:
        raise RuntimeError("benchmark calls must be positive")
    return time.perf_counter() - t0, ci, ky


def benchmark_dimension_polyglot_parity_gate(
    *,
    t: int = 64,
    d: int = 3,
    n_k: int = 12,
    calls: int = 1,
    seed: int = 2026,
) -> dict[str, object]:
    """Record exact fractal-dimension parity across declared backends."""

    t = _validate_int_control(t, name="t", minimum=2)
    d = _validate_int_control(d, name="d", minimum=1)
    n_k = _validate_int_control(n_k, name="n_k", minimum=2)
    calls = _validate_int_control(calls, name="calls", minimum=1)
    seed = _validate_int_control(seed, name="seed", minimum=0)

    trajectory, epsilons, spectrum = _problem(t, d, n_k, seed)
    idx_i, idx_j = _pair_indices(t)
    reference_ci, reference_ky = _direct_outputs(
        "python",
        trajectory,
        epsilons,
        spectrum,
        idx_i=idx_i,
        idx_j=idx_j,
    )

    records: list[dict[str, object]] = []
    parity_checked_count = 0
    parity_pass_count = 0
    available_backend_count = 0
    t0 = time.perf_counter()

    for backend in BACKEND_ORDER:
        tolerance = PARITY_TOLERANCES[backend]
        available, reason = _backend_status(backend)
        if not available:
            records.append(
                {
                    "backend": backend,
                    "status": "unavailable",
                    "ms_per_call": None,
                    "correlation_integral_sha256": None,
                    "kaplan_yorke_sha256": None,
                    "ci_max_abs_error": None,
                    "ky_abs_error": None,
                    "max_abs_error": None,
                    "ci_monotonic": False,
                    "ci_unit_interval": False,
                    "ky_bounds_passed": False,
                    "parity_passed": False,
                    "tolerance": tolerance,
                    "unavailable_reason": reason,
                }
            )
            continue

        available_backend_count += 1
        elapsed, ci, ky = _bench_with_output(
            backend,
            trajectory,
            epsilons,
            spectrum,
            idx_i=idx_i,
            idx_j=idx_j,
            calls=calls,
        )
        ci_max_abs_error = float(np.max(np.abs(ci - reference_ci)))
        ky_abs_error = abs(float(ky) - float(reference_ky))
        max_abs_error = max(ci_max_abs_error, ky_abs_error)
        ci_monotonic = bool(np.all(np.diff(ci) >= -1.0e-12))
        ci_unit_interval = bool(np.all(ci >= -1.0e-12) and np.all(ci <= 1.0 + 1.0e-12))
        ky_bounds_passed = bool(0.0 <= ky <= float(spectrum.size) + 1.0e-12)
        parity_passed = (
            max_abs_error <= tolerance
            and ci_monotonic
            and ci_unit_interval
            and ky_bounds_passed
        )
        parity_checked_count += 1
        parity_pass_count += int(parity_passed)
        records.append(
            {
                "backend": backend,
                "status": "available",
                "ms_per_call": (elapsed / calls) * 1000.0,
                "correlation_integral_sha256": _array_sha256(ci),
                "kaplan_yorke_sha256": _scalar_sha256(ky),
                "ci_max_abs_error": ci_max_abs_error,
                "ky_abs_error": ky_abs_error,
                "max_abs_error": max_abs_error,
                "ci_monotonic": ci_monotonic,
                "ci_unit_interval": ci_unit_interval,
                "ky_bounds_passed": ky_bounds_passed,
                "parity_passed": parity_passed,
                "tolerance": tolerance,
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
        "require_correlation_integral_monotonic": True,
        "require_exact_full_pairs_reference": True,
        "require_kaplan_yorke_bounds": True,
        "require_kaplan_yorke_contract": True,
        "require_python_reference": True,
        "require_unit_interval_correlation_integral": True,
    }
    acceptance_passed = (
        len(records) == len(BACKEND_ORDER)
        and any(
            record["backend"] == "python" and record["status"] == "available"
            for record in records
        )
        and parity_pass_count == parity_checked_count
        and np.all(np.diff(reference_ci) >= -1.0e-12)
        and np.all(reference_ci >= -1.0e-12)
        and np.all(reference_ci <= 1.0 + 1.0e-12)
        and 0.0 <= reference_ky <= float(spectrum.size) + 1.0e-12
    )
    benchmark_payload = {
        "T": t,
        "d": d,
        "n_k": n_k,
        "calls": calls,
        "seed": seed,
        "records": records,
        "thresholds": thresholds,
        "reference_correlation_integral_sha256": _array_sha256(reference_ci),
        "reference_kaplan_yorke_sha256": _scalar_sha256(reference_ky),
    }
    benchmark_sha = hashlib.sha256(
        json.dumps(benchmark_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    return {
        "suite": "dimension_polyglot_parity_gate",
        "backend_count": len(records),
        "available_backend_count": available_backend_count,
        "unavailable_backend_count": len(records) - available_backend_count,
        "parity_checked_count": parity_checked_count,
        "parity_pass_count": parity_pass_count,
        "all_available_passed": int(parity_pass_count == parity_checked_count),
        "python_reference_present": 1,
        "T": t,
        "d": d,
        "n_k": n_k,
        "calls": calls,
        "seed": seed,
        "reference_ci_min": float(np.min(reference_ci)),
        "reference_ci_max": float(np.max(reference_ci)),
        "reference_kaplan_yorke": float(reference_ky),
        "reference_correlation_integral_sha256": _array_sha256(reference_ci),
        "reference_kaplan_yorke_sha256": _scalar_sha256(reference_ky),
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
    parser.add_argument("--T-list", type=int, nargs="+", default=[50, 150, 400])
    parser.add_argument("--d", type=int, default=3)
    parser.add_argument("--n-k", type=int, default=20)
    parser.add_argument("--calls", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--parity-gate",
        action="store_true",
        help="emit deterministic all-backend dimension parity-gate JSON",
    )
    args = parser.parse_args()

    if args.parity_gate:
        result = benchmark_dimension_polyglot_parity_gate(
            t=args.T_list[0],
            d=args.d,
            n_k=args.n_k,
            calls=args.calls,
            seed=args.seed,
        )
        text = json.dumps(result, indent=2, sort_keys=True)
        print(text)
        if args.output:
            args.output.write_text(text + "\n", encoding="utf-8")
        return 0

    print(f"Active: {ACTIVE_BACKEND}  Available: {AVAILABLE_BACKENDS}")
    print("Boundary contract: exact NumPy dimension reference validated\n")
    header = f"{'T':>5} {'d':>3} {'K':>3} {'calls':>6}"
    for b in AVAILABLE_BACKENDS:
        header += f" {b + '_ms':>12}"
    print(header)
    print("-" * len(header))
    results: list[dict] = []
    for t in args.T_list:
        row = bench_at(t, args.d, args.n_k, args.calls)
        results.append(row)
        line = f"{t:>5} {args.d:>3} {args.n_k:>3} {args.calls:>6}"
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
