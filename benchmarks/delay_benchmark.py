# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Delayed Kuramoto multi-backend benchmark

"""Per-backend wall-clock benchmark and polyglot parity gate for
``upde.delay`` (time-delayed Kuramoto integration)."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.upde import delay as delay_mod
from scpn_phase_orchestrator.upde.delay import (
    ACTIVE_BACKEND,
    AVAILABLE_BACKENDS,
    DelayedEngine,
)

TWO_PI = 2.0 * np.pi
BACKEND_ORDER = ("rust", "mojo", "julia", "go", "python")
PARITY_TOLERANCES = {
    "rust": 1.0e-9,
    "mojo": 1.0e-6,
    "julia": 1.0e-9,
    "go": 1.0e-9,
    "python": 0.0,
}
REFERENCE_TOLERANCE = 1.0e-9
BENCHMARK_EVIDENCE_KIND = "local_regression_non_isolated"
_DT = 0.05
_ZETA = 0.3
_PSI = 0.7


def _validate_int_control(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _problem(
    n: int, seed: int
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    rng = np.random.default_rng(seed)
    phases = np.ascontiguousarray(rng.uniform(0.0, TWO_PI, n))
    omegas = np.ascontiguousarray(rng.uniform(-1.0, 1.0, n))
    knm = rng.uniform(0.0, 0.5, (n, n))
    np.fill_diagonal(knm, 0.0)
    alpha = rng.uniform(0.0, 0.3, (n, n))
    return phases, omegas, np.ascontiguousarray(knm), np.ascontiguousarray(alpha)


def _run(
    backend: str,
    problem: tuple[NDArray[np.float64], ...],
    delay_steps: int,
    n_steps: int,
) -> NDArray[np.float64]:
    phases, omegas, knm, alpha = problem
    saved = delay_mod.ACTIVE_BACKEND
    try:
        delay_mod.ACTIVE_BACKEND = backend
        eng = DelayedEngine(int(phases.size), dt=_DT, delay_steps=delay_steps)
        return eng.run(phases, omegas, knm, _ZETA, _PSI, alpha, n_steps=n_steps)
    finally:
        delay_mod.ACTIVE_BACKEND = saved


def _vector_sha256(values: NDArray[np.float64]) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(values, dtype=np.float64).tobytes()
    ).hexdigest()


def _reference_contracts(
    backend: str, n: int, delay_steps: int
) -> dict[str, float | int]:
    """Pure-rotation identity for zero coupling/forcing and phase-range check."""
    rng = np.random.default_rng(11)
    phases = np.ascontiguousarray(rng.uniform(0.0, TWO_PI, n))
    omegas = np.ascontiguousarray(rng.uniform(-1.0, 1.0, n))
    zero = np.zeros((n, n), dtype=np.float64)
    saved = delay_mod.ACTIVE_BACKEND
    try:
        delay_mod.ACTIVE_BACKEND = backend
        eng = DelayedEngine(n, dt=_DT, delay_steps=delay_steps)
        rotated = eng.run(phases, omegas, zero, 0.0, 0.0, zero, n_steps=20)
    finally:
        delay_mod.ACTIVE_BACKEND = saved
    expected = (phases + omegas * _DT * 20) % TWO_PI
    in_range = bool(np.all(rotated >= 0.0) and np.all(rotated < TWO_PI))
    return {
        "pure_rotation_max_abs_error": float(np.max(np.abs(rotated - expected))),
        "phases_in_range": int(in_range),
    }


def _contracts_passed(contracts: Mapping[str, float | int]) -> bool:
    return (
        float(contracts["pure_rotation_max_abs_error"]) <= REFERENCE_TOLERANCE
        and int(contracts["phases_in_range"]) == 1
    )


def _backend_status(backend: str) -> tuple[bool, str]:
    if backend in AVAILABLE_BACKENDS:
        return True, ""
    return False, f"{backend} backend was not resolved by upde.delay"


def _bench_with_output(
    backend: str,
    problem: tuple[NDArray[np.float64], ...],
    delay_steps: int,
    n_steps: int,
    *,
    calls: int,
) -> tuple[float, NDArray[np.float64]]:
    _run(backend, problem, delay_steps, n_steps)
    output: NDArray[np.float64] | None = None
    t0 = time.perf_counter()
    for _ in range(calls):
        output = _run(backend, problem, delay_steps, n_steps)
    elapsed = time.perf_counter() - t0
    if output is None:
        raise RuntimeError("benchmark calls must be positive")
    return elapsed, output


def benchmark_delay_polyglot_parity_gate(
    *,
    n: int = 16,
    delay_steps: int = 3,
    n_steps: int = 80,
    calls: int = 1,
    seed: int = 2026,
) -> dict[str, object]:
    """Record delayed-Kuramoto final-phase parity across declared backends.

    Every available backend must reproduce the Python reference final phases and
    satisfy the contracts: with zero coupling and forcing the integrator reduces
    to pure rotation ``(θ + ω·dt·n_steps) mod 2π``, and all output phases lie in
    ``[0, 2π)``.
    """
    n = _validate_int_control(n, name="n", minimum=1)
    delay_steps = _validate_int_control(delay_steps, name="delay_steps", minimum=1)
    n_steps = _validate_int_control(n_steps, name="n_steps", minimum=1)
    calls = _validate_int_control(calls, name="calls", minimum=1)
    seed = _validate_int_control(seed, name="seed", minimum=0)

    problem = _problem(n, seed)
    reference = _run("python", problem, delay_steps, n_steps)
    reference_contracts = _reference_contracts("python", n, delay_steps)
    reference_contracts_passed = _contracts_passed(reference_contracts)

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
                    "max_abs_error": None,
                    "phases_sha256": None,
                    "reference_contracts_passed": False,
                    "tolerance": tolerance,
                    "parity_passed": False,
                    "unavailable_reason": reason,
                }
            )
            continue

        available_backend_count += 1
        elapsed, phases = _bench_with_output(
            backend, problem, delay_steps, n_steps, calls=calls
        )
        max_err = float(np.max(np.abs(phases - reference))) if phases.size else 0.0
        contracts_passed = _contracts_passed(
            _reference_contracts(backend, n, delay_steps)
        )
        parity_passed = bool(max_err <= tolerance and contracts_passed)
        parity_checked_count += 1
        parity_pass_count += int(parity_passed)
        records.append(
            {
                "backend": backend,
                "status": "available",
                "ms_per_call": (elapsed / calls) * 1000.0,
                "max_abs_error": max_err,
                "phases_sha256": _vector_sha256(phases),
                "reference_contracts_passed": contracts_passed,
                "tolerance": tolerance,
                "parity_passed": parity_passed,
                "unavailable_reason": "",
            }
        )

    wall_time = time.perf_counter() - t0
    thresholds = {
        "backend_order": list(BACKEND_ORDER),
        "parity_tolerances": PARITY_TOLERANCES,
        "max_reference_contract_abs_error": REFERENCE_TOLERANCE,
        "require_all_available_parity": True,
        "require_all_declared_backend_records": True,
        "require_python_reference": True,
        "require_pure_rotation_limit": True,
        "require_phases_in_range": True,
        "production_timing_claim": False,
    }
    acceptance_passed = (
        len(records) == len(BACKEND_ORDER)
        and any(
            r["backend"] == "python" and r["status"] == "available" for r in records
        )
        and parity_checked_count > 0
        and parity_pass_count == parity_checked_count
        and reference_contracts_passed
    )
    payload: dict[str, Any] = {
        "n": n,
        "delay_steps": delay_steps,
        "n_steps": n_steps,
        "calls": calls,
        "seed": seed,
        "records": [
            {
                k: v
                for k, v in r.items()
                if k not in {"ms_per_call", "unavailable_reason"}
            }
            for r in records
        ],
        "reference_contracts": reference_contracts,
        "reference_phases_sha256": _vector_sha256(reference),
    }
    benchmark_sha = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    return {
        "suite": "delay_polyglot_parity_gate",
        "backend_count": len(records),
        "available_backend_count": available_backend_count,
        "unavailable_backend_count": len(records) - available_backend_count,
        "parity_checked_count": parity_checked_count,
        "parity_pass_count": parity_pass_count,
        "all_available_passed": int(parity_pass_count == parity_checked_count),
        "python_reference_present": 1,
        "reference_contracts_passed": int(reference_contracts_passed),
        "acceptance_passed": int(acceptance_passed),
        "n": n,
        "delay_steps": delay_steps,
        "n_steps": n_steps,
        "calls": calls,
        "seed": seed,
        "pure_rotation_max_abs_error": float(
            reference_contracts["pure_rotation_max_abs_error"]
        ),
        "reference_phases_sha256": _vector_sha256(reference),
        "benchmark_sha256": benchmark_sha,
        "benchmark_evidence_kind": BENCHMARK_EVIDENCE_KIND,
        "isolation_method": "none",
        "production_timing_claim": 0,
        "wall_time_s": wall_time,
        "steps_per_second": parity_checked_count / wall_time if wall_time else 0.0,
        "acceptance_thresholds_json": json.dumps(thresholds, sort_keys=True),
        "backend_records_json": json.dumps(records, sort_keys=True),
    }


def bench_at(
    n: int, delay_steps: int, n_steps: int, calls: int, *, seed: int = 2026
) -> dict:
    problem = _problem(n, seed)
    row: dict = {
        "n": n,
        "delay_steps": delay_steps,
        "n_steps": n_steps,
        "calls": calls,
        "available": AVAILABLE_BACKENDS,
    }
    for backend in AVAILABLE_BACKENDS:
        elapsed, _ = _bench_with_output(
            backend, problem, delay_steps, n_steps, calls=calls
        )
        row[f"{backend}_ms_per_call"] = (elapsed / calls) * 1000.0
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--delay-steps", type=int, default=3)
    parser.add_argument("--n-steps", type=int, default=80)
    parser.add_argument("--calls", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--parity-gate", action="store_true")
    args = parser.parse_args()

    if args.parity_gate:
        result = benchmark_delay_polyglot_parity_gate(
            n=args.n,
            delay_steps=args.delay_steps,
            n_steps=args.n_steps,
            calls=args.calls,
            seed=args.seed,
        )
        text = json.dumps(result, indent=2, sort_keys=True)
        print(text)
        if args.output:
            args.output.write_text(text + "\n", encoding="utf-8")
        return 0

    print(f"Active: {ACTIVE_BACKEND}  Available: {AVAILABLE_BACKENDS}\n")
    row = bench_at(args.n, args.delay_steps, args.n_steps, args.calls, seed=args.seed)
    for backend in AVAILABLE_BACKENDS:
        print(f"{backend:>8}: {row[f'{backend}_ms_per_call']:.4f} ms/call")
    if args.output:
        args.output.write_text(json.dumps(row, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
