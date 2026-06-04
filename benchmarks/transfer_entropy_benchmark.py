# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Transfer entropy polyglot parity benchmark gate

"""Polyglot parity and local wall-clock gates for transfer entropy.

The reference contract is the exact NumPy histogram estimator in
``monitor.transfer_entropy``. Backend timings emitted here are local regression
and parity-execution evidence only unless the run metadata separately records
CPU/core isolation and host-load controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor import transfer_entropy as te_mod
from scpn_phase_orchestrator.monitor.transfer_entropy import (
    ACTIVE_BACKEND,
    AVAILABLE_BACKENDS,
    phase_transfer_entropy,
    transfer_entropy_matrix,
)

FloatArray = NDArray[np.float64]

TWO_PI = 2.0 * np.pi
BACKEND_ORDER = ("rust", "mojo", "julia", "go", "python")
SCALAR_TOLERANCES = {
    "rust": 1.0e-12,
    "mojo": 1.0e-9,
    "julia": 1.0e-12,
    "go": 1.0e-12,
    "python": 0.0,
}
MATRIX_TOLERANCES = SCALAR_TOLERANCES
CONTRACT_TOLERANCE = 1.0e-12
MIN_CAUSAL_DIRECTION_MARGIN = 1.0e-2
BENCHMARK_EVIDENCE_KIND = "local_regression_non_isolated"


def _validate_int_control(value: int, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _phase_pair(n: int, seed: int) -> tuple[FloatArray, FloatArray]:
    n = _validate_int_control(n, name="n", minimum=48)
    seed = _validate_int_control(seed, name="seed", minimum=0)
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 8.0 * np.pi, n, dtype=np.float64)
    source = np.mod(
        np.cumsum(0.37 + 0.11 * np.sin(t) + 0.025 * rng.normal(size=n)),
        TWO_PI,
    )
    carrier = np.mod(
        np.cumsum(
            0.19 + 0.07 * np.cos(0.7 * t + 0.2) + 0.035 * rng.normal(size=n),
        ),
        TWO_PI,
    )
    target = np.empty(n, dtype=np.float64)
    target[0] = carrier[0]
    target[1:] = np.mod(
        0.86 * source[:-1] + 0.14 * carrier[1:] + 0.02 * rng.normal(size=n - 1),
        TWO_PI,
    )
    return (
        np.ascontiguousarray(source, dtype=np.float64),
        np.ascontiguousarray(target, dtype=np.float64),
    )


def _phase_series(source: FloatArray, target: FloatArray, seed: int) -> FloatArray:
    rng = np.random.default_rng(seed + 7919)
    n = source.size
    t = np.linspace(0.0, 5.0 * np.pi, n, dtype=np.float64)
    background = np.mod(
        np.cumsum(0.23 + 0.05 * np.sin(1.7 * t) + 0.04 * rng.normal(size=n)),
        TWO_PI,
    )
    return np.ascontiguousarray(
        np.vstack([source, target, background]).astype(np.float64, copy=False),
        dtype=np.float64,
    )


def _force_backend(backend: str) -> str:
    previous = te_mod.ACTIVE_BACKEND
    te_mod.ACTIVE_BACKEND = backend
    return previous


def _restore_backend(previous: str) -> None:
    te_mod.ACTIVE_BACKEND = previous


def _python_scalar(source: FloatArray, target: FloatArray, n_bins: int) -> float:
    previous = _force_backend("python")
    try:
        return float(phase_transfer_entropy(source, target, n_bins))
    finally:
        _restore_backend(previous)


def _python_matrix(series: FloatArray, n_bins: int) -> FloatArray:
    previous = _force_backend("python")
    try:
        return np.ascontiguousarray(transfer_entropy_matrix(series, n_bins))
    finally:
        _restore_backend(previous)


def _array_sha256(value: FloatArray) -> str:
    array = np.ascontiguousarray(value, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("utf-8"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _float_sha256(value: float) -> str:
    return hashlib.sha256(repr(float(value)).encode("ascii")).hexdigest()


def _max_abs_error(actual: FloatArray, expected: FloatArray) -> float:
    if actual.shape != expected.shape:
        return float("inf")
    if actual.size == 0:
        return 0.0
    return float(np.max(np.abs(actual - expected)))


def _load_backend(backend: str) -> dict[str, object] | None:
    if backend == "python":
        return None
    return te_mod._load_backend(backend)  # noqa: SLF001


def _call_direct_scalar(
    funcs: dict[str, object] | None,
    source: FloatArray,
    target: FloatArray,
    n_bins: int,
) -> float:
    if funcs is None:
        return _python_scalar(source, target, n_bins)
    fn = funcs["phase_te"]
    if not callable(fn):
        raise RuntimeError("phase transfer-entropy primitive is not callable")
    return float(fn(source, target, n_bins))


def _call_direct_matrix(
    funcs: dict[str, object] | None,
    series: FloatArray,
    n_bins: int,
) -> FloatArray:
    if funcs is None:
        return _python_matrix(series, n_bins)
    fn = funcs["te_matrix"]
    if not callable(fn):
        raise RuntimeError("transfer-entropy matrix primitive is not callable")
    n_osc, n_time = series.shape
    raw = fn(
        np.ascontiguousarray(series.ravel(), dtype=np.float64),
        n_osc,
        n_time,
        n_bins,
    )
    matrix = np.asarray(raw, dtype=np.float64)
    if matrix.shape == (n_osc * n_osc,):
        matrix = matrix.reshape(n_osc, n_osc)
    return np.ascontiguousarray(matrix, dtype=np.float64)


def _reference_contracts(
    source: FloatArray,
    target: FloatArray,
    series: FloatArray,
    n_bins: int,
) -> dict[str, float | int]:
    forward = _python_scalar(source, target, n_bins)
    reverse = _python_scalar(target, source, n_bins)
    matrix = _python_matrix(series, n_bins)
    shifted_series = np.ascontiguousarray(
        series + np.array([[TWO_PI], [-TWO_PI], [4.0 * TWO_PI]], dtype=np.float64),
    )
    shifted_matrix = _python_matrix(shifted_series, n_bins)
    shifted_forward = _python_scalar(source + TWO_PI, target - TWO_PI, n_bins)
    short_zero = _python_scalar(source[:2], target[:2], n_bins)
    max_entropy = float(np.log(n_bins))
    diagonal = np.diag(matrix)

    return {
        "finite_reference_outputs": int(
            np.isfinite(forward)
            and np.isfinite(reverse)
            and np.all(np.isfinite(matrix))
            and np.isfinite(shifted_forward)
            and np.all(np.isfinite(shifted_matrix))
            and np.isfinite(short_zero),
        ),
        "forward_te": float(forward),
        "reverse_te": float(reverse),
        "causal_direction_margin": float(forward - reverse),
        "scalar_matrix_forward_error": abs(float(matrix[0, 1]) - forward),
        "scalar_matrix_reverse_error": abs(float(matrix[1, 0]) - reverse),
        "diagonal_max_abs": float(np.max(np.abs(diagonal))),
        "matrix_min_value": float(np.min(matrix)),
        "matrix_max_entropy_excess": max(0.0, float(np.max(matrix)) - max_entropy),
        "phase_wrap_scalar_abs_error": abs(shifted_forward - forward),
        "phase_wrap_matrix_max_abs_error": _max_abs_error(shifted_matrix, matrix),
        "short_series_abs_error": abs(short_zero),
    }


def _contracts_passed(contracts: dict[str, float | int]) -> bool:
    if int(contracts["finite_reference_outputs"]) != 1:
        return False
    if float(contracts["causal_direction_margin"]) <= MIN_CAUSAL_DIRECTION_MARGIN:
        return False
    if float(contracts["matrix_min_value"]) < -CONTRACT_TOLERANCE:
        return False
    checked = (
        "scalar_matrix_forward_error",
        "scalar_matrix_reverse_error",
        "diagonal_max_abs",
        "matrix_max_entropy_excess",
        "phase_wrap_scalar_abs_error",
        "phase_wrap_matrix_max_abs_error",
        "short_series_abs_error",
    )
    return all(float(contracts[key]) <= CONTRACT_TOLERANCE for key in checked)


def _backend_record(
    backend: str,
    source: FloatArray,
    target: FloatArray,
    series: FloatArray,
    *,
    n_bins: int,
    calls: int,
    reference_forward: float,
    reference_reverse: float,
    reference_matrix: FloatArray,
    reference_contracts: dict[str, float | int],
) -> dict[str, Any]:
    tolerance = SCALAR_TOLERANCES[backend]
    matrix_tolerance = MATRIX_TOLERANCES[backend]
    base: dict[str, Any] = {
        "backend": backend,
        "status": "unavailable",
        "parity_passed": False,
        "public_dispatch_parity_passed": False,
        "contracts_passed": False,
        "causal_direction_preserved": False,
        "tolerance": tolerance,
        "matrix_tolerance": matrix_tolerance,
        "forward_te": None,
        "reverse_te": None,
        "scalar_forward_abs_error": None,
        "scalar_reverse_abs_error": None,
        "matrix_max_abs_error": None,
        "forward_te_sha256": None,
        "reverse_te_sha256": None,
        "matrix_sha256": None,
        "ms_per_call": None,
        "unavailable_reason": "",
    }
    try:
        funcs = _load_backend(backend)
    except (ImportError, ModuleNotFoundError, OSError, RuntimeError) as exc:
        reason = str(exc).strip() or exc.__class__.__name__
        base["unavailable_reason"] = (
            f"{backend} backend unavailable for monitor.transfer_entropy: {reason}"
        )
        return base

    try:
        direct_forward = reference_forward
        direct_reverse = reference_reverse
        direct_matrix = reference_matrix
        t0 = time.perf_counter()
        for _ in range(calls):
            direct_forward = _call_direct_scalar(funcs, source, target, n_bins)
            direct_reverse = _call_direct_scalar(funcs, target, source, n_bins)
            direct_matrix = _call_direct_matrix(funcs, series, n_bins)
        elapsed = time.perf_counter() - t0
    except (TypeError, ValueError, RuntimeError) as exc:
        base["status"] = "invalid"
        base["unavailable_reason"] = (
            f"{backend} backend returned invalid transfer-entropy output: {exc}"
        )
        return base

    previous = _force_backend(backend)
    try:
        public_forward = phase_transfer_entropy(source, target, n_bins)
        public_reverse = phase_transfer_entropy(target, source, n_bins)
        public_matrix = transfer_entropy_matrix(series, n_bins)
    finally:
        _restore_backend(previous)

    scalar_forward_error = abs(float(direct_forward) - reference_forward)
    scalar_reverse_error = abs(float(direct_reverse) - reference_reverse)
    matrix_error = _max_abs_error(direct_matrix, reference_matrix)
    public_dispatch_parity = (
        abs(float(public_forward) - reference_forward) <= tolerance
        and abs(float(public_reverse) - reference_reverse) <= tolerance
        and _max_abs_error(public_matrix, reference_matrix) <= matrix_tolerance
    )
    causal_direction_preserved = bool(direct_forward > direct_reverse)
    contracts_passed = _contracts_passed(reference_contracts)
    parity_passed = (
        scalar_forward_error <= tolerance
        and scalar_reverse_error <= tolerance
        and matrix_error <= matrix_tolerance
        and public_dispatch_parity
        and causal_direction_preserved
        and contracts_passed
    )

    base.update(
        {
            "status": "available",
            "parity_passed": parity_passed,
            "public_dispatch_parity_passed": public_dispatch_parity,
            "contracts_passed": contracts_passed,
            "causal_direction_preserved": causal_direction_preserved,
            "forward_te": float(direct_forward),
            "reverse_te": float(direct_reverse),
            "scalar_forward_abs_error": scalar_forward_error,
            "scalar_reverse_abs_error": scalar_reverse_error,
            "matrix_max_abs_error": matrix_error,
            "forward_te_sha256": _float_sha256(float(direct_forward)),
            "reverse_te_sha256": _float_sha256(float(direct_reverse)),
            "matrix_sha256": _array_sha256(direct_matrix),
            "ms_per_call": (elapsed / calls) * 1000.0,
        },
    )
    return base


def _deterministic_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8",
    )
    return hashlib.sha256(encoded).hexdigest()


def _clear_backend_cache() -> None:
    te_mod._BACKEND_CACHE.clear()  # noqa: SLF001


def benchmark_transfer_entropy_polyglot_parity_gate(
    *,
    n: int = 160,
    calls: int = 1,
    seed: int = 2026,
    n_bins: int = 16,
) -> dict[str, float | int | str]:
    """Run the transfer-entropy Rust/Mojo/Julia/Go/Python parity gate."""
    n = _validate_int_control(n, name="n", minimum=48)
    calls = _validate_int_control(calls, name="calls", minimum=1)
    seed = _validate_int_control(seed, name="seed", minimum=0)
    n_bins = _validate_int_control(n_bins, name="n_bins", minimum=2)
    source, target = _phase_pair(n, seed)
    series = _phase_series(source, target, seed)
    reference_forward = _python_scalar(source, target, n_bins)
    reference_reverse = _python_scalar(target, source, n_bins)
    reference_matrix = _python_matrix(series, n_bins)
    reference_contracts = _reference_contracts(source, target, series, n_bins)

    _clear_backend_cache()
    t0 = time.perf_counter()
    records = [
        _backend_record(
            backend,
            source,
            target,
            series,
            n_bins=n_bins,
            calls=calls,
            reference_forward=reference_forward,
            reference_reverse=reference_reverse,
            reference_matrix=reference_matrix,
            reference_contracts=reference_contracts,
        )
        for backend in BACKEND_ORDER
    ]
    elapsed = time.perf_counter() - t0

    available_records = [
        record for record in records if record["status"] == "available"
    ]
    unavailable_records = [
        record for record in records if record["status"] == "unavailable"
    ]
    invalid_records = [record for record in records if record["status"] == "invalid"]
    all_available_passed = all(
        record["parity_passed"] is True for record in available_records
    )
    python_reference_present = any(
        record["backend"] == "python" and record["status"] == "available"
        for record in records
    )
    reference_contracts_passed = _contracts_passed(reference_contracts)
    acceptance_passed = (
        python_reference_present
        and reference_contracts_passed
        and all_available_passed
        and not invalid_records
    )
    thresholds = {
        "backend_order": list(BACKEND_ORDER),
        "max_mojo_abs_error": SCALAR_TOLERANCES["mojo"],
        "max_native_abs_error": SCALAR_TOLERANCES["rust"],
        "max_reference_contract_abs_error": CONTRACT_TOLERANCE,
        "min_causal_direction_margin": MIN_CAUSAL_DIRECTION_MARGIN,
        "require_all_available_parity": True,
        "require_all_declared_backend_records": True,
        "require_causal_direction_preservation": True,
        "require_entropy_bound": True,
        "require_matrix_scalar_consistency": True,
        "require_phase_wrapping_invariance": True,
        "require_public_dispatch_parity": True,
        "require_python_reference": True,
        "require_zero_diagonal_matrix": True,
        "production_timing_claim": False,
    }
    deterministic_payload = {
        "suite": "transfer_entropy_polyglot_parity_gate",
        "backend_records": [
            {
                key: value
                for key, value in record.items()
                if key not in {"ms_per_call", "unavailable_reason"}
            }
            for record in records
        ],
        "reference_contracts": reference_contracts,
        "reference_forward_sha256": _float_sha256(reference_forward),
        "reference_reverse_sha256": _float_sha256(reference_reverse),
        "reference_matrix_sha256": _array_sha256(reference_matrix),
    }
    return {
        "suite": "transfer_entropy_polyglot_parity_gate",
        "n": n,
        "calls": calls,
        "seed": seed,
        "n_bins": n_bins,
        "backend_count": len(BACKEND_ORDER),
        "available_backend_count": len(available_records),
        "unavailable_backend_count": len(unavailable_records),
        "invalid_backend_count": len(invalid_records),
        "parity_pass_count": sum(
            1 for record in available_records if record["parity_passed"] is True
        ),
        "python_reference_present": int(python_reference_present),
        "all_available_passed": int(all_available_passed),
        "reference_contracts_passed": int(reference_contracts_passed),
        "acceptance_passed": int(acceptance_passed),
        "reference_forward_te": reference_forward,
        "reference_reverse_te": reference_reverse,
        "reference_direction_margin": float(reference_forward - reference_reverse),
        "reference_forward_te_sha256": _float_sha256(reference_forward),
        "reference_reverse_te_sha256": _float_sha256(reference_reverse),
        "reference_matrix_sha256": _array_sha256(reference_matrix),
        "finite_reference_outputs": int(
            reference_contracts["finite_reference_outputs"],
        ),
        "scalar_matrix_forward_error": float(
            reference_contracts["scalar_matrix_forward_error"],
        ),
        "scalar_matrix_reverse_error": float(
            reference_contracts["scalar_matrix_reverse_error"],
        ),
        "diagonal_max_abs": float(reference_contracts["diagonal_max_abs"]),
        "matrix_min_value": float(reference_contracts["matrix_min_value"]),
        "matrix_max_entropy_excess": float(
            reference_contracts["matrix_max_entropy_excess"],
        ),
        "phase_wrap_scalar_abs_error": float(
            reference_contracts["phase_wrap_scalar_abs_error"],
        ),
        "phase_wrap_matrix_max_abs_error": float(
            reference_contracts["phase_wrap_matrix_max_abs_error"],
        ),
        "short_series_abs_error": float(reference_contracts["short_series_abs_error"]),
        "backend_records_json": json.dumps(records, sort_keys=True),
        "acceptance_thresholds_json": json.dumps(thresholds, sort_keys=True),
        "benchmark_sha256": _deterministic_hash(deterministic_payload),
        "benchmark_evidence_kind": BENCHMARK_EVIDENCE_KIND,
        "isolation_method": "none",
        "production_timing_claim": 0,
        "wall_time_s": elapsed,
        "steps_per_second": (n * calls * max(len(available_records), 1))
        / max(elapsed, 1.0e-12),
    }


def _bench(
    backend: str,
    source: FloatArray,
    target: FloatArray,
    calls: int,
    n_bins: int,
) -> float:
    previous = _force_backend(backend)
    try:
        t0 = time.perf_counter()
        for _ in range(calls):
            phase_transfer_entropy(source, target, n_bins)
        return time.perf_counter() - t0
    finally:
        _restore_backend(previous)


def bench_at(n: int, calls: int, *, seed: int = 42, n_bins: int = 16) -> dict:
    """Return legacy-compatible per-backend local timing rows."""
    n = _validate_int_control(n, name="n", minimum=48)
    calls = _validate_int_control(calls, name="calls", minimum=1)
    seed = _validate_int_control(seed, name="seed", minimum=0)
    n_bins = _validate_int_control(n_bins, name="n_bins", minimum=2)
    source, target = _phase_pair(n, seed)
    row: dict[str, object] = {
        "n": n,
        "calls": calls,
        "seed": seed,
        "n_bins": n_bins,
        "available": AVAILABLE_BACKENDS,
        "boundary_contract": "exact_numpy_histogram_estimator_validated",
        "benchmark_evidence_kind": BENCHMARK_EVIDENCE_KIND,
        "isolation_method": "none",
        "production_timing_claim": 0,
    }
    for backend in AVAILABLE_BACKENDS:
        elapsed = _bench(backend, source, target, calls, n_bins)
        row[f"{backend}_ms_per_call"] = (elapsed / calls) * 1000.0
    return row


def bench_sizes(
    sizes: Sequence[int],
    calls: int,
    *,
    seed: int = 42,
    n_bins: int = 16,
) -> list[dict]:
    return [bench_at(size, calls, seed=seed, n_bins=n_bins) for size in sizes]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sizes", type=int, nargs="+", default=[200, 1000, 5000])
    parser.add_argument("--calls", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-bins", type=int, default=16)
    parser.add_argument("--parity-gate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.parity_gate:
        payload: dict[str, object] = benchmark_transfer_entropy_polyglot_parity_gate(
            n=int(args.sizes[0]),
            calls=args.calls,
            seed=args.seed,
            n_bins=args.n_bins,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Active: {ACTIVE_BACKEND}  Available: {AVAILABLE_BACKENDS}")
        print("Boundary contract: exact NumPy histogram estimator validated")
        print("Timing evidence: local regression, non-isolated; not throughput claim\n")
        header = f"{'N':>6}"
        for backend in AVAILABLE_BACKENDS:
            header += f" {backend + '_ms':>12}"
        print(header)
        print("-" * len(header))
        results = bench_sizes(
            args.sizes,
            args.calls,
            seed=args.seed,
            n_bins=args.n_bins,
        )
        for row in results:
            line = f"{row['n']:>6}"
            for backend in AVAILABLE_BACKENDS:
                line += f" {row[f'{backend}_ms_per_call']:>12.4f}"
            print(line)
        payload = {"results": results}
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
