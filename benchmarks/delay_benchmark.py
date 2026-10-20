# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Delayed Kuramoto multi-backend benchmark

"""Measure and qualify actual delayed Kuramoto runtime owners."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

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
    """Execute the requested public owner without changing process-wide dispatch."""
    phases, omegas, knm, alpha = problem
    eng = DelayedEngine(int(phases.size), dt=_DT, delay_steps=delay_steps)
    return eng.run(
        phases, omegas, knm, _ZETA, _PSI, alpha, n_steps=n_steps, backend=backend
    )


def _independent_reference(
    problem: tuple[NDArray[np.float64], ...], delay: int, steps: int
) -> NDArray[np.float64]:
    """Evaluate scalar simultaneous Euler updates using full saved history."""
    phases, omega, coupling, alpha = problem
    current = [float(value) for value in phases]
    history: list[list[float]] = []
    for step in range(steps):
        history.append(current.copy())
        delayed = history[step - delay] if step >= delay else current
        candidate = []
        for i, phase in enumerate(current):
            force = sum(
                float(coupling[i, j])
                * math.sin(delayed[j] - phase - float(alpha[i, j]))
                for j in range(len(current))
            )
            raw = phase + _DT * (
                float(omega[i]) + force + _ZETA * math.sin(_PSI - phase)
            )
            wrapped = raw % math.tau
            candidate.append(0.0 if wrapped >= math.tau or wrapped == 0.0 else wrapped)
        current = candidate
    return np.array(current, dtype=np.float64)


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
    eng = DelayedEngine(n, dt=_DT, delay_steps=delay_steps)
    rotated = eng.run(phases, omegas, zero, 0.0, 0.0, zero, n_steps=20, backend=backend)
    expected = (phases + omegas * _DT * 20) % TWO_PI
    in_range = bool(np.all(rotated >= 0.0) and np.all(rotated < TWO_PI))
    interior = np.nextafter(TWO_PI, 0.0)
    endpoint = DelayedEngine(3, dt=_DT).run(
        np.array([-np.nextafter(0.0, 1.0), -0.0, interior]),
        np.zeros(3),
        np.zeros((3, 3)),
        n_steps=1,
        backend=backend,
    )
    endpoint_passed = bool(
        np.array_equal(endpoint, [0.0, 0.0, interior])
        and not np.any(np.signbit(endpoint[:2]))
    )
    return {
        "pure_rotation_max_abs_error": float(np.max(np.abs(rotated - expected))),
        "phases_in_range": int(in_range),
        "endpoint_contract_passed": int(endpoint_passed),
    }


def _contracts_passed(contracts: Mapping[str, float | int]) -> bool:
    return (
        float(contracts["pure_rotation_max_abs_error"]) <= REFERENCE_TOLERANCE
        and int(contracts["phases_in_range"]) == 1
        and int(contracts["endpoint_contract_passed"]) == 1
    )


def _backend_status(backend: str) -> tuple[bool, str]:
    if backend in delay_mod.AVAILABLE_BACKENDS:
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
    require_backends: Sequence[str] = (),
) -> dict[str, object]:
    """Record delayed-Kuramoto final-phase parity across declared backends.

    Every available owner executes explicitly and must match both the public
    Python owner and an independent scalar-history recurrence. Pure rotation,
    half-open endpoints and positive zero are also required. Named required
    owners must be present and execute successfully; unavailable owners remain
    explicit records and never acquire successful execution credit.

    Parameters
    ----------
    n : int
        Positive oscillator count in the generated directed coupling problem.
    delay_steps, n_steps : int
        Positive discrete source delay and number of simultaneous Euler steps.
    calls : int
        Positive repeated calls per owner after its warmup call.
    seed : int
        Nonnegative seed binding identical input arrays across owners.
    require_backends : Sequence[str]
        Owners whose absence or failure rejects the complete qualification.

    Returns
    -------
    dict[str, object]
        Explicit owner records, independent reference hashes/errors, repeated
        diagnostic timings and acceptance flags. Failed owners retain no timing
        or successful phase hash. These shared-host timings claim no throughput.

    Raises
    ------
    ValueError
        If controls or required owner names are invalid, or the reference fails.
    """
    n = _validate_int_control(n, name="n", minimum=1)
    delay_steps = _validate_int_control(delay_steps, name="delay_steps", minimum=1)
    n_steps = _validate_int_control(n_steps, name="n_steps", minimum=1)
    calls = _validate_int_control(calls, name="calls", minimum=1)
    seed = _validate_int_control(seed, name="seed", minimum=0)
    if isinstance(require_backends, (str, bytes)) or any(
        not isinstance(name, str) or name not in BACKEND_ORDER
        for name in require_backends
    ):
        raise ValueError("required delay backends must name declared owners")
    required = tuple(dict.fromkeys(require_backends))

    problem = _problem(n, seed)
    reference = _run("python", problem, delay_steps, n_steps)
    independent = _independent_reference(problem, delay_steps, n_steps)
    independent_error = float(np.max(np.abs(reference - independent)))
    reference_contracts = _reference_contracts("python", n, delay_steps)
    reference_contracts_passed = (
        _contracts_passed(reference_contracts)
        and independent_error <= REFERENCE_TOLERANCE
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
        try:
            elapsed, phases = _bench_with_output(
                backend, problem, delay_steps, n_steps, calls=calls
            )
            contracts_passed = _contracts_passed(
                _reference_contracts(backend, n, delay_steps)
            )
        except (ImportError, RuntimeError, OSError, ValueError) as exc:
            records.append(
                {
                    "backend": backend,
                    "status": "failed",
                    "ms_per_call": None,
                    "max_abs_error": None,
                    "phases_sha256": None,
                    "reference_contracts_passed": False,
                    "tolerance": tolerance,
                    "parity_passed": False,
                    "unavailable_reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        max_err = float(np.max(np.abs(phases - reference))) if phases.size else 0.0
        oracle_error = float(np.max(np.abs(phases - independent)))
        parity_passed = bool(
            max_err <= tolerance
            and oracle_error <= max(tolerance, REFERENCE_TOLERANCE)
            and contracts_passed
        )
        parity_checked_count += 1
        parity_pass_count += int(parity_passed)
        records.append(
            {
                "backend": backend,
                "status": "available",
                "ms_per_call": (elapsed / calls) * 1000.0,
                "max_abs_error": max_err,
                "independent_max_abs_error": oracle_error,
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
        and parity_pass_count == available_backend_count
        and reference_contracts_passed
        and all(
            any(
                record["backend"] == name and record["parity_passed"]
                for record in records
            )
            for name in required
        )
    )
    payload: dict[str, object] = {
        "n": n,
        "delay_steps": delay_steps,
        "n_steps": n_steps,
        "calls": calls,
        "seed": seed,
        "required_backends": list(required),
        "independent_reference_sha256": _vector_sha256(independent),
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
        "all_available_passed": int(parity_pass_count == available_backend_count),
        "python_reference_present": 1,
        "reference_contracts_passed": int(reference_contracts_passed),
        "acceptance_passed": int(acceptance_passed),
        "required_backends": list(required),
        "independent_reference_sha256": _vector_sha256(independent),
        "independent_reference_max_abs_error": independent_error,
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
) -> dict[str, object]:
    """Measure explicit available owners after warmup.

    Parameters
    ----------
    n, delay_steps, n_steps, calls : int
        Positive oscillator, discrete-delay, integration and repeat counts.
    seed : int
        Nonnegative seed of the shared directed coupling problem.

    Returns
    -------
    dict[str, object]
        Controls, available owner names and per-owner mean milliseconds per call.

    Raises
    ------
    ValueError
        If controls or an owner's numerical result are invalid.
    ImportError
        If an advertised owner cannot execute its requested batch.
    """
    n = _validate_int_control(n, name="n", minimum=1)
    delay_steps = _validate_int_control(delay_steps, name="delay_steps", minimum=1)
    n_steps = _validate_int_control(n_steps, name="n_steps", minimum=1)
    calls = _validate_int_control(calls, name="calls", minimum=1)
    seed = _validate_int_control(seed, name="seed", minimum=0)
    problem = _problem(n, seed)
    row: dict[str, object] = {
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
    """Run measured comparison CLI and return nonzero for failed qualification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--delay-steps", type=int, default=3)
    parser.add_argument("--n-steps", type=int, default=80)
    parser.add_argument("--calls", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--parity-gate", action="store_true")
    parser.add_argument(
        "--require-backends", choices=BACKEND_ORDER, nargs="+", default=[]
    )
    args = parser.parse_args()

    if args.parity_gate:
        result = benchmark_delay_polyglot_parity_gate(
            n=args.n,
            delay_steps=args.delay_steps,
            n_steps=args.n_steps,
            calls=args.calls,
            seed=args.seed,
            require_backends=args.require_backends,
        )
        text = json.dumps(result, indent=2, sort_keys=True)
        print(text)
        if args.output:
            args.output.write_text(text + "\n", encoding="utf-8")
        return 0 if result["acceptance_passed"] == 1 else 1

    print(f"Active: {ACTIVE_BACKEND}  Available: {AVAILABLE_BACKENDS}\n")
    row = bench_at(args.n, args.delay_steps, args.n_steps, args.calls, seed=args.seed)
    for backend in AVAILABLE_BACKENDS:
        print(f"{backend:>8}: {row[f'{backend}_ms_per_call']:.4f} ms/call")
    if args.output:
        args.output.write_text(json.dumps(row, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
