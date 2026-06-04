# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Entropy production polyglot parity benchmark gate

"""Polyglot parity and local wall-clock gates for entropy production.

The reference contract is the exact NumPy overdamped-Kuramoto dissipation
formula in ``monitor.entropy_prod``. Backend timings emitted here are local
regression and parity-execution evidence only unless the run metadata
separately records CPU/core isolation and host-load controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Sequence
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor import entropy_prod as ep_mod
from scpn_phase_orchestrator.monitor.entropy_prod import (
    ACTIVE_BACKEND,
    AVAILABLE_BACKENDS,
    entropy_production_rate,
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
CONTRACT_TOLERANCE = 1.0e-12
BENCHMARK_EVIDENCE_KIND = "local_regression_non_isolated"


def _validate_int_control(value: int, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _validate_positive_float_control(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must not be a boolean value")
    if not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _problem(n: int, seed: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    n = _validate_int_control(n, name="n", minimum=2)
    seed = _validate_int_control(seed, name="seed", minimum=0)
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, TWO_PI, size=n)
    omegas = rng.normal(0.0, 0.2, size=n)
    knm = rng.uniform(0.3, 0.9, size=(n, n))
    np.fill_diagonal(knm, 0.0)
    return (
        np.ascontiguousarray(phases, dtype=np.float64),
        np.ascontiguousarray(omegas, dtype=np.float64),
        np.ascontiguousarray(knm, dtype=np.float64),
    )


def _force_backend(backend: str) -> str:
    previous = ep_mod.ACTIVE_BACKEND
    ep_mod.ACTIVE_BACKEND = backend
    return previous


def _restore_backend(previous: str) -> None:
    ep_mod.ACTIVE_BACKEND = previous


def _python_rate(
    phases: FloatArray,
    omegas: FloatArray,
    knm: FloatArray,
    alpha: float,
    dt: float,
) -> float:
    previous = _force_backend("python")
    try:
        return float(entropy_production_rate(phases, omegas, knm, alpha, dt))
    finally:
        _restore_backend(previous)


def _manual_rate(
    phases: FloatArray,
    omegas: FloatArray,
    knm: FloatArray,
    alpha: float,
    dt: float,
) -> float:
    if phases.size == 0 or dt == 0.0:
        return 0.0
    n = int(phases.size)
    diff = phases[np.newaxis, :] - phases[:, np.newaxis]
    coupling = np.sum(knm * np.sin(diff), axis=1)
    dtheta_dt = omegas + (alpha / n) * coupling
    return float(np.sum(dtheta_dt**2) * dt)


def _float_sha256(value: float) -> str:
    return hashlib.sha256(repr(float(value)).encode("ascii")).hexdigest()


def _reference_contracts(
    phases: FloatArray,
    omegas: FloatArray,
    knm: FloatArray,
    alpha: float,
    dt: float,
) -> dict[str, float | int]:
    reference = _python_rate(phases, omegas, knm, alpha, dt)
    manual = _manual_rate(phases, omegas, knm, alpha, dt)
    fixed_point = _python_rate(np.zeros(4), np.zeros(4), np.zeros((4, 4)), alpha, dt)
    zero_dt = _python_rate(phases, omegas, knm, alpha, 0.0)
    double_dt = _python_rate(phases, omegas, knm, alpha, 2.0 * dt)
    shifted = _python_rate(phases + 3.0 * TWO_PI, omegas, knm, alpha, dt)
    permutation = np.arange(phases.size - 1, -1, -1, dtype=np.int64)
    permuted = _python_rate(
        phases[permutation],
        omegas[permutation],
        knm[np.ix_(permutation, permutation)],
        alpha,
        dt,
    )
    zero_omegas = np.zeros_like(omegas)
    alpha_base = _python_rate(phases, zero_omegas, knm, alpha, dt)
    alpha_double = _python_rate(phases, zero_omegas, knm, 2.0 * alpha, dt)
    rates = np.array(
        [
            reference,
            manual,
            fixed_point,
            zero_dt,
            double_dt,
            shifted,
            permuted,
            alpha_base,
            alpha_double,
        ],
        dtype=np.float64,
    )
    return {
        "finite_reference_outputs": int(np.all(np.isfinite(rates))),
        "reference_rate": reference,
        "manual_formula_abs_error": abs(reference - manual),
        "fixed_point_abs_error": abs(fixed_point),
        "zero_dt_abs_error": abs(zero_dt),
        "dt_scaling_abs_error": abs(double_dt - 2.0 * reference),
        "phase_shift_abs_error": abs(shifted - reference),
        "permutation_abs_error": abs(permuted - reference),
        "alpha_quadratic_abs_error": abs(alpha_double - 4.0 * alpha_base),
        "minimum_observed_rate": float(np.min(rates)),
    }


def _contracts_passed(contracts: dict[str, float | int]) -> bool:
    if int(contracts["finite_reference_outputs"]) != 1:
        return False
    if float(contracts["minimum_observed_rate"]) < -CONTRACT_TOLERANCE:
        return False
    checked = (
        "manual_formula_abs_error",
        "fixed_point_abs_error",
        "zero_dt_abs_error",
        "dt_scaling_abs_error",
        "phase_shift_abs_error",
        "permutation_abs_error",
        "alpha_quadratic_abs_error",
    )
    return all(float(contracts[key]) <= CONTRACT_TOLERANCE for key in checked)


def _load_backend(backend: str) -> Any | None:
    if backend == "python":
        return None
    return ep_mod._load_backend(backend)  # noqa: SLF001


def _call_direct_rate(
    func: Any | None,
    phases: FloatArray,
    omegas: FloatArray,
    knm: FloatArray,
    alpha: float,
    dt: float,
) -> float:
    if func is None:
        return _python_rate(phases, omegas, knm, alpha, dt)
    if not callable(func):
        raise RuntimeError("entropy-production primitive is not callable")
    return float(func(phases, omegas, knm, alpha, dt))


def _backend_record(
    backend: str,
    phases: FloatArray,
    omegas: FloatArray,
    knm: FloatArray,
    *,
    alpha: float,
    dt: float,
    calls: int,
    reference_rate: float,
    reference_contracts: dict[str, float | int],
) -> dict[str, Any]:
    tolerance = SCALAR_TOLERANCES[backend]
    base: dict[str, Any] = {
        "backend": backend,
        "status": "unavailable",
        "parity_passed": False,
        "public_dispatch_parity_passed": False,
        "contracts_passed": False,
        "non_negative_rate": False,
        "tolerance": tolerance,
        "rate": None,
        "rate_abs_error": None,
        "public_rate_abs_error": None,
        "rate_sha256": None,
        "ms_per_call": None,
        "unavailable_reason": "",
    }
    try:
        func = _load_backend(backend)
    except (ImportError, ModuleNotFoundError, OSError, RuntimeError) as exc:
        reason = str(exc).strip() or exc.__class__.__name__
        base["unavailable_reason"] = (
            f"{backend} backend unavailable for monitor.entropy_prod: {reason}"
        )
        return base

    try:
        rate = reference_rate
        t0 = time.perf_counter()
        for _ in range(calls):
            rate = _call_direct_rate(func, phases, omegas, knm, alpha, dt)
        elapsed = time.perf_counter() - t0
    except (TypeError, ValueError, RuntimeError) as exc:
        base["status"] = "invalid"
        base["unavailable_reason"] = (
            f"{backend} backend returned invalid entropy-production output: {exc}"
        )
        return base

    previous = _force_backend(backend)
    try:
        public_rate = float(entropy_production_rate(phases, omegas, knm, alpha, dt))
    finally:
        _restore_backend(previous)

    rate_error = abs(rate - reference_rate)
    public_error = abs(public_rate - reference_rate)
    non_negative_rate = bool(rate >= -CONTRACT_TOLERANCE)
    contracts_passed = _contracts_passed(reference_contracts)
    public_dispatch_parity = public_error <= tolerance
    parity_passed = (
        rate_error <= tolerance
        and public_dispatch_parity
        and non_negative_rate
        and contracts_passed
    )
    base.update(
        {
            "status": "available",
            "parity_passed": parity_passed,
            "public_dispatch_parity_passed": public_dispatch_parity,
            "contracts_passed": contracts_passed,
            "non_negative_rate": non_negative_rate,
            "rate": float(rate),
            "rate_abs_error": rate_error,
            "public_rate_abs_error": public_error,
            "rate_sha256": _float_sha256(rate),
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
    ep_mod._BACKEND_CACHE.clear()  # noqa: SLF001


def benchmark_entropy_production_polyglot_parity_gate(
    *,
    n: int = 16,
    calls: int = 1,
    seed: int = 2026,
    alpha: object = 0.5,
    dt: object = 0.01,
) -> dict[str, float | int | str]:
    """Run the entropy-production Rust/Mojo/Julia/Go/Python parity gate."""
    n = _validate_int_control(n, name="n", minimum=2)
    calls = _validate_int_control(calls, name="calls", minimum=1)
    seed = _validate_int_control(seed, name="seed", minimum=0)
    alpha_value = _validate_positive_float_control(alpha, name="alpha")
    dt_value = _validate_positive_float_control(dt, name="dt")
    phases, omegas, knm = _problem(n, seed)
    reference_rate = _python_rate(phases, omegas, knm, alpha_value, dt_value)
    reference_contracts = _reference_contracts(
        phases,
        omegas,
        knm,
        alpha_value,
        dt_value,
    )

    _clear_backend_cache()
    t0 = time.perf_counter()
    records = [
        _backend_record(
            backend,
            phases,
            omegas,
            knm,
            alpha=alpha_value,
            dt=dt_value,
            calls=calls,
            reference_rate=reference_rate,
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
        "require_all_available_parity": True,
        "require_all_declared_backend_records": True,
        "require_alpha_quadratic_scaling": True,
        "require_dt_linear_scaling": True,
        "require_exact_formula_parity": True,
        "require_fixed_point_zero": True,
        "require_non_negative_rate": True,
        "require_permutation_invariance": True,
        "require_phase_shift_invariance": True,
        "require_public_dispatch_parity": True,
        "require_python_reference": True,
        "production_timing_claim": False,
    }
    deterministic_payload = {
        "suite": "entropy_production_polyglot_parity_gate",
        "backend_records": [
            {
                key: value
                for key, value in record.items()
                if key not in {"ms_per_call", "unavailable_reason"}
            }
            for record in records
        ],
        "reference_contracts": reference_contracts,
        "reference_rate_sha256": _float_sha256(reference_rate),
    }
    return {
        "suite": "entropy_production_polyglot_parity_gate",
        "n": n,
        "calls": calls,
        "seed": seed,
        "alpha": alpha_value,
        "dt": dt_value,
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
        "reference_rate": reference_rate,
        "reference_rate_sha256": _float_sha256(reference_rate),
        "finite_reference_outputs": int(
            reference_contracts["finite_reference_outputs"],
        ),
        "manual_formula_abs_error": float(
            reference_contracts["manual_formula_abs_error"],
        ),
        "fixed_point_abs_error": float(reference_contracts["fixed_point_abs_error"]),
        "zero_dt_abs_error": float(reference_contracts["zero_dt_abs_error"]),
        "dt_scaling_abs_error": float(reference_contracts["dt_scaling_abs_error"]),
        "phase_shift_abs_error": float(
            reference_contracts["phase_shift_abs_error"],
        ),
        "permutation_abs_error": float(
            reference_contracts["permutation_abs_error"],
        ),
        "alpha_quadratic_abs_error": float(
            reference_contracts["alpha_quadratic_abs_error"],
        ),
        "minimum_observed_rate": float(reference_contracts["minimum_observed_rate"]),
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
    phases: FloatArray,
    omegas: FloatArray,
    knm: FloatArray,
    calls: int,
    alpha: float,
    dt: float,
) -> float:
    previous = _force_backend(backend)
    try:
        entropy_production_rate(phases, omegas, knm, alpha, dt)
        t0 = time.perf_counter()
        for _ in range(calls):
            entropy_production_rate(phases, omegas, knm, alpha, dt)
        return time.perf_counter() - t0
    finally:
        _restore_backend(previous)


def bench_at(
    n: int,
    calls: int,
    *,
    seed: int = 42,
    alpha: object = 0.5,
    dt: object = 0.01,
) -> dict:
    """Return legacy-compatible per-backend local timing rows."""
    n = _validate_int_control(n, name="n", minimum=2)
    calls = _validate_int_control(calls, name="calls", minimum=1)
    seed = _validate_int_control(seed, name="seed", minimum=0)
    alpha_value = _validate_positive_float_control(alpha, name="alpha")
    dt_value = _validate_positive_float_control(dt, name="dt")
    phases, omegas, knm = _problem(n, seed)
    row: dict[str, object] = {
        "n": n,
        "calls": calls,
        "seed": seed,
        "alpha": alpha_value,
        "dt": dt_value,
        "available": AVAILABLE_BACKENDS,
        "boundary_contract": "exact_overdamped_kuramoto_dissipation_validated",
        "benchmark_evidence_kind": BENCHMARK_EVIDENCE_KIND,
        "isolation_method": "none",
        "production_timing_claim": 0,
    }
    for backend in AVAILABLE_BACKENDS:
        elapsed = _bench(backend, phases, omegas, knm, calls, alpha_value, dt_value)
        row[f"{backend}_ms_per_call"] = (elapsed / calls) * 1000.0
    return row


def bench_sizes(
    sizes: Sequence[int],
    calls: int,
    *,
    seed: int = 42,
    alpha: object = 0.5,
    dt: object = 0.01,
) -> list[dict]:
    return [
        bench_at(size, calls, seed=seed, alpha=alpha, dt=dt)
        for size in sizes
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 64, 256, 1024])
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--parity-gate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.parity_gate:
        payload: dict[str, object] = benchmark_entropy_production_polyglot_parity_gate(
            n=int(args.sizes[0]),
            calls=args.calls,
            seed=args.seed,
            alpha=args.alpha,
            dt=args.dt,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Active: {ACTIVE_BACKEND}  Available: {AVAILABLE_BACKENDS}")
        print("Boundary contract: exact overdamped-Kuramoto dissipation validated")
        print("Timing evidence: local regression, non-isolated; not throughput claim\n")
        header = f"{'N':>5} {'calls':>6}"
        for backend in AVAILABLE_BACKENDS:
            header += f" {backend + '_ms':>12}"
        print(header)
        print("-" * len(header))
        results = bench_sizes(
            args.sizes,
            args.calls,
            seed=args.seed,
            alpha=args.alpha,
            dt=args.dt,
        )
        for row in results:
            line = f"{row['n']:>5} {args.calls:>6}"
            for backend in AVAILABLE_BACKENDS:
                line += f" {row[f'{backend}_ms_per_call']:>12.4f}"
            print(line)
        payload = {"results": results}
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
