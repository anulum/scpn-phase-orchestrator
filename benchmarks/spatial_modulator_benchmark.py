# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — spatial modulator benchmark gate

"""Per-backend parity gate for ``coupling.spatial_modulator``."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.coupling import spatial_modulator as sm_mod
from scpn_phase_orchestrator.coupling.spatial_modulator import (
    AVAILABLE_BACKENDS,
    SpatialCouplingModulator,
)

FloatArray: TypeAlias = NDArray[np.float64]
BenchmarkResult: TypeAlias = dict[str, object]


class ReferenceContracts(TypedDict):
    """Reference invariants enforced by the spatial-modulator parity gate."""

    manual_formula_abs_error: float
    translation_abs_error: float
    permutation_abs_error: float
    symmetry_abs_error: float
    zero_diagonal_abs_error: float
    nearer_pair_weight_exceeds_far_pair: bool


BACKEND_ORDER: tuple[str, ...] = ("rust", "mojo", "julia", "go", "python")
PARITY_TOLERANCES: dict[str, float] = {
    "rust": 1.0e-12,
    "mojo": 1.0e-9,
    "julia": 1.0e-10,
    "go": 1.0e-10,
    "python": 0.0,
}
BENCHMARK_EVIDENCE_KIND = "local_regression_non_isolated"


def _validate_int_control(value: object, *, name: str, minimum: int) -> int:
    """Return a validated integer benchmark control value."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return parsed


def _problem(n: int, dim: int, seed: int) -> tuple[FloatArray, FloatArray]:
    """Return a deterministic symmetric coupling problem for the benchmark."""
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.2, 1.4, size=(n, n))
    knm = 0.5 * (raw + raw.T)
    np.fill_diagonal(knm, 0.0)
    positions = rng.normal(0.0, 0.7, size=(n, dim))
    return np.ascontiguousarray(knm, dtype=np.float64), np.ascontiguousarray(
        positions, dtype=np.float64
    )


def _force_backend(backend: str) -> str:
    """Set the active backend and return the previous backend name."""
    previous = sm_mod.ACTIVE_BACKEND
    sm_mod.ACTIVE_BACKEND = backend
    return previous


def _restore_backend(previous: str) -> None:
    """Restore the active backend after an isolated benchmark call."""
    sm_mod.ACTIVE_BACKEND = previous


def _bench_backend(
    backend: str,
    knm: FloatArray,
    positions: FloatArray,
    calls: int,
) -> tuple[float, FloatArray]:
    """Return elapsed wall time and output matrix for one backend."""
    modulator = SpatialCouplingModulator(K_base=0.75)
    previous = _force_backend(backend)
    try:
        out = modulator.modulate(knm, positions)
        t0 = time.perf_counter()
        for _ in range(calls):
            out = modulator.modulate(knm, positions)
        return time.perf_counter() - t0, np.ascontiguousarray(out, dtype=np.float64)
    finally:
        _restore_backend(previous)


def _array_sha256(array: FloatArray) -> str:
    """Return a stable SHA-256 digest for a float64 matrix."""
    values = np.ascontiguousarray(array, dtype=np.float64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _backend_status(backend: str) -> tuple[bool, str]:
    """Return whether the named backend is available and why not when absent."""
    if backend == "python" or backend in AVAILABLE_BACKENDS:
        return True, ""
    return False, f"{backend} backend was not resolved by coupling.spatial_modulator"


def _reference_contracts(
    knm: FloatArray,
    positions: FloatArray,
    reference: FloatArray,
) -> ReferenceContracts:
    """Return mathematical contract errors for the Python reference output."""
    modulator = SpatialCouplingModulator(K_base=0.75)
    distances = modulator.distance_matrix(positions)
    manual = 0.75 * knm / (1.0 + distances)
    np.fill_diagonal(manual, 0.0)
    translated = modulator.modulate(
        knm, positions + np.array([3.0] * positions.shape[1])
    )
    perm = np.array(list(reversed(range(knm.shape[0]))), dtype=int)
    permuted = modulator.modulate(knm[np.ix_(perm, perm)], positions[perm])
    near_far = SpatialCouplingModulator(K_base=1.0).modulate(
        np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]),
        np.array([[0.0], [0.5], [4.0]]),
    )
    return {
        "manual_formula_abs_error": float(np.max(np.abs(reference - manual))),
        "translation_abs_error": float(np.max(np.abs(reference - translated))),
        "permutation_abs_error": float(
            np.max(np.abs(reference[np.ix_(perm, perm)] - permuted))
        ),
        "symmetry_abs_error": float(np.max(np.abs(reference - reference.T))),
        "zero_diagonal_abs_error": float(np.max(np.abs(np.diag(reference)))),
        "nearer_pair_weight_exceeds_far_pair": bool(near_far[0, 1] > near_far[0, 2]),
    }


def benchmark_spatial_modulator_polyglot_parity_gate(
    *,
    n: int = 10,
    dim: int = 2,
    calls: int = 1,
    seed: int = 2026,
) -> BenchmarkResult:
    """Record spatial-modulator parity across declared backend slots."""

    n = _validate_int_control(n, name="n", minimum=2)
    dim = _validate_int_control(dim, name="dim", minimum=1)
    calls = _validate_int_control(calls, name="calls", minimum=1)
    seed = _validate_int_control(seed, name="seed", minimum=0)
    knm, positions = _problem(n, dim, seed)
    t0 = time.perf_counter()
    _, reference = _bench_backend("python", knm, positions, 1)
    contracts = _reference_contracts(knm, positions, reference)
    records: list[dict[str, object]] = []
    available_count = 0
    parity_checked_count = 0
    parity_pass_count = 0
    for backend in BACKEND_ORDER:
        tolerance = PARITY_TOLERANCES[backend]
        available, reason = _backend_status(backend)
        if not available:
            records.append(
                {
                    "backend": backend,
                    "status": "unavailable",
                    "ms_per_call": None,
                    "matrix_sha256": None,
                    "max_abs_error": None,
                    "tolerance": tolerance,
                    "parity_passed": False,
                    "unavailable_reason": reason,
                }
            )
            continue
        available_count += 1
        elapsed, got = _bench_backend(backend, knm, positions, calls)
        error = float(np.max(np.abs(got - reference)))
        passed = error <= tolerance
        parity_checked_count += 1
        parity_pass_count += int(passed)
        records.append(
            {
                "backend": backend,
                "status": "available",
                "ms_per_call": (elapsed / calls) * 1000.0,
                "matrix_sha256": _array_sha256(got),
                "max_abs_error": error,
                "tolerance": tolerance,
                "parity_passed": passed,
                "unavailable_reason": "",
            }
        )
    wall_time = time.perf_counter() - t0
    thresholds = {
        "backend_order": list(BACKEND_ORDER),
        "max_mojo_abs_error": PARITY_TOLERANCES["mojo"],
        "max_native_abs_error": PARITY_TOLERANCES["rust"],
        "production_timing_claim": False,
        "require_all_available_parity": True,
        "require_all_declared_backend_records": True,
        "require_inverse_plus_one_formula": True,
        "require_near_far_monotonicity": True,
        "require_permutation_equivariance": True,
        "require_python_reference": True,
        "require_translation_invariance": True,
        "require_zero_diagonal": True,
    }
    acceptance_passed = (
        len(records) == len(BACKEND_ORDER)
        and parity_pass_count == parity_checked_count
        and contracts["manual_formula_abs_error"] <= 1.0e-12
        and contracts["translation_abs_error"] <= 1.0e-12
        and contracts["permutation_abs_error"] <= 1.0e-12
        and contracts["symmetry_abs_error"] <= 1.0e-12
        and contracts["zero_diagonal_abs_error"] <= 1.0e-12
        and bool(contracts["nearer_pair_weight_exceeds_far_pair"])
    )
    benchmark_payload = {
        "n": n,
        "dim": dim,
        "calls": calls,
        "seed": seed,
        "reference_sha256": _array_sha256(reference),
        "records": records,
        "contracts": contracts,
        "thresholds": thresholds,
    }
    benchmark_sha = hashlib.sha256(
        json.dumps(benchmark_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "suite": "spatial_modulator_polyglot_parity_gate",
        "backend_count": len(records),
        "available_backend_count": available_count,
        "unavailable_backend_count": len(records) - available_count,
        "parity_checked_count": parity_checked_count,
        "parity_pass_count": parity_pass_count,
        "all_available_passed": int(parity_pass_count == parity_checked_count),
        "python_reference_present": 1,
        "n": n,
        "dim": dim,
        "calls": calls,
        "seed": seed,
        "reference_matrix_sha256": _array_sha256(reference),
        "manual_formula_abs_error": contracts["manual_formula_abs_error"],
        "translation_abs_error": contracts["translation_abs_error"],
        "permutation_abs_error": contracts["permutation_abs_error"],
        "symmetry_abs_error": contracts["symmetry_abs_error"],
        "zero_diagonal_abs_error": contracts["zero_diagonal_abs_error"],
        "nearer_pair_weight_exceeds_far_pair": int(
            bool(contracts["nearer_pair_weight_exceeds_far_pair"])
        ),
        "benchmark_sha256": benchmark_sha,
        "benchmark_evidence_kind": BENCHMARK_EVIDENCE_KIND,
        "isolation_method": "none",
        "production_timing_claim": 0,
        "wall_time_s": wall_time,
        "steps_per_second": parity_checked_count / wall_time if wall_time else 0.0,
        "acceptance_passed": int(acceptance_passed),
        "acceptance_thresholds_json": json.dumps(thresholds, sort_keys=True),
        "backend_records_json": json.dumps(records, sort_keys=True),
    }


def main() -> int:
    """Run the spatial-modulator benchmark CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sizes", type=int, nargs="+", default=[16])
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--calls", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--parity-gate", action="store_true")
    args = parser.parse_args()
    if args.parity_gate:
        result = benchmark_spatial_modulator_polyglot_parity_gate(
            n=args.sizes[0],
            dim=args.dim,
            calls=args.calls,
            seed=args.seed,
        )
        payload = json.dumps(result, indent=2, sort_keys=True)
        if args.output:
            args.output.write_text(payload + "\n", encoding="utf-8")
        print(payload)
        return 0
    results = [
        benchmark_spatial_modulator_polyglot_parity_gate(
            n=size,
            dim=args.dim,
            calls=args.calls,
            seed=args.seed,
        )
        for size in args.sizes
    ]
    if args.output:
        args.output.write_text(
            json.dumps({"results": results}, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps({"results": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
