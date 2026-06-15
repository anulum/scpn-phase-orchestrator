# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Poincaré-section multi-backend benchmark

"""Per-backend wall-clock benchmark and polyglot parity gate for
``monitor.poincare`` (hyperplane ``poincare_section`` and phase-specific
``phase_poincare`` crossing detectors)."""

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

from scpn_phase_orchestrator.monitor import poincare as pc_mod
from scpn_phase_orchestrator.monitor.poincare import (
    ACTIVE_BACKEND,
    AVAILABLE_BACKENDS,
    phase_poincare,
    poincare_section,
)

TWO_PI = 2.0 * np.pi
BACKEND_ORDER = ("rust", "mojo", "julia", "go", "python")
PARITY_TOLERANCES = {
    "rust": 1.0e-10,
    "mojo": 1.0e-7,
    "julia": 1.0e-10,
    "go": 1.0e-10,
    "python": 0.0,
}
REFERENCE_TOLERANCE = 1.0e-9
BENCHMARK_EVIDENCE_KIND = "local_regression_non_isolated"
# Generic-hyperplane section: positive crossings of the first coordinate.
_NORMAL = np.array([1.0, 0.0], dtype=np.float64)
_OFFSET = 0.0
_DIRECTION_ID = 0  # "positive"
_OSC_IDX = 0
_SECTION_PHASE = 0.0


def _validate_int_control(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _problem(
    n_steps: int,
    n_osc: int,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build a deterministic section trajectory and a phase history with a
    monotonically advancing section oscillator, both with clean crossings."""
    n_steps = _validate_int_control(n_steps, name="n_steps", minimum=8)
    n_osc = _validate_int_control(n_osc, name="n_osc", minimum=2)
    seed = _validate_int_control(seed, name="seed", minimum=0)
    idx = np.arange(n_steps, dtype=np.float64)
    # 2-D section trajectory: three loops → clean positive x-axis crossings.
    traj = np.column_stack(
        (
            np.sin(2.0 * np.pi * 3.0 * idx / n_steps),
            np.cos(2.0 * np.pi * 3.0 * idx / n_steps),
        )
    ).astype(np.float64)
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, TWO_PI, (n_steps, n_osc)).astype(np.float64)
    # Section oscillator advances smoothly through several 2π wraps.
    phases[:, _OSC_IDX] = 0.21 * idx
    return np.ascontiguousarray(traj), np.ascontiguousarray(phases)


def _direct_bundle(
    backend: str,
    traj: NDArray[np.float64],
    phases: NDArray[np.float64],
) -> dict[str, Any]:
    """Crossings, times, and counts for both primitives from one backend."""
    t_s, d = int(traj.shape[0]), int(traj.shape[1])
    t_p, n = int(phases.shape[0]), int(phases.shape[1])
    if backend == "python":
        saved = pc_mod.ACTIVE_BACKEND
        try:
            pc_mod.ACTIVE_BACKEND = "python"
            sec = poincare_section(traj, _NORMAL, _OFFSET, direction="positive")
            ph = phase_poincare(phases, _OSC_IDX, _SECTION_PHASE)
        finally:
            pc_mod.ACTIVE_BACKEND = saved
        return {
            "section_crossings": np.ascontiguousarray(sec.crossings, dtype=np.float64),
            "section_times": np.ascontiguousarray(sec.crossing_times, dtype=np.float64),
            "section_count": int(sec.crossing_times.shape[0]),
            "phase_crossings": np.ascontiguousarray(ph.crossings, dtype=np.float64),
            "phase_times": np.ascontiguousarray(ph.crossing_times, dtype=np.float64),
            "phase_count": int(ph.crossing_times.shape[0]),
        }
    fns = pc_mod._load_backend(backend)  # noqa: SLF001
    sec_fn = fns["section"]
    ph_fn = fns["phase"]
    cr_flat, times, n_cr = sec_fn(  # type: ignore[operator]
        traj.ravel(), t_s, d, _NORMAL, _OFFSET, _DIRECTION_ID
    )
    n_cr = int(n_cr)
    sec_cr = np.asarray(cr_flat, dtype=np.float64)[: n_cr * d].reshape(n_cr, d)
    sec_t = np.asarray(times, dtype=np.float64)[:n_cr]
    cr_flat_p, times_p, n_cr_p = ph_fn(  # type: ignore[operator]
        phases.ravel(), t_p, n, _OSC_IDX, _SECTION_PHASE
    )
    n_cr_p = int(n_cr_p)
    ph_cr = np.asarray(cr_flat_p, dtype=np.float64)[: n_cr_p * n].reshape(n_cr_p, n)
    ph_t = np.asarray(times_p, dtype=np.float64)[:n_cr_p]
    return {
        "section_crossings": np.ascontiguousarray(sec_cr, dtype=np.float64),
        "section_times": np.ascontiguousarray(sec_t, dtype=np.float64),
        "section_count": n_cr,
        "phase_crossings": np.ascontiguousarray(ph_cr, dtype=np.float64),
        "phase_times": np.ascontiguousarray(ph_t, dtype=np.float64),
        "phase_count": n_cr_p,
    }


def _array_sha256(values: NDArray[np.float64]) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(values, dtype=np.float64).tobytes()
    ).hexdigest()


def _array_error(actual: NDArray[np.float64], expected: NDArray[np.float64]) -> float:
    if actual.shape != expected.shape:
        return float("inf")
    if actual.size == 0:
        return 0.0
    return float(np.max(np.abs(actual - expected)))


def _bundle_errors(
    actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, float]:
    errors: dict[str, float] = {}
    for key in ("section_crossings", "section_times", "phase_crossings", "phase_times"):
        errors[key] = _array_error(actual[key], expected[key])
    count_mismatch = (
        actual["section_count"] != expected["section_count"]
        or actual["phase_count"] != expected["phase_count"]
    )
    errors["count"] = float("inf") if count_mismatch else 0.0
    return errors


def _strictly_increasing(values: NDArray[np.float64]) -> bool:
    return bool(values.size <= 1 or np.all(np.diff(values) > 0.0))


def _reference_contracts(bundle: Mapping[str, Any]) -> dict[str, float | int]:
    n_hat = _NORMAL / np.linalg.norm(_NORMAL)
    sec_cr = bundle["section_crossings"]
    plane_residual = (
        float(np.max(np.abs(sec_cr @ n_hat - _OFFSET))) if sec_cr.size else 0.0
    )
    ph_cr = bundle["phase_crossings"]
    if ph_cr.size:
        wrapped = (ph_cr[:, _OSC_IDX] - _SECTION_PHASE) % TWO_PI
        phase_residual = float(np.max(np.minimum(wrapped, TWO_PI - wrapped)))
    else:
        phase_residual = 0.0
    return {
        "section_plane_residual": plane_residual,
        "phase_value_residual": phase_residual,
        "section_times_increasing": int(_strictly_increasing(bundle["section_times"])),
        "phase_times_increasing": int(_strictly_increasing(bundle["phase_times"])),
        "section_count": int(bundle["section_count"]),
        "phase_count": int(bundle["phase_count"]),
    }


def _reference_contracts_passed(contracts: Mapping[str, float | int]) -> bool:
    return (
        float(contracts["section_plane_residual"]) <= REFERENCE_TOLERANCE
        and float(contracts["phase_value_residual"]) <= REFERENCE_TOLERANCE
        and int(contracts["section_times_increasing"]) == 1
        and int(contracts["phase_times_increasing"]) == 1
        and int(contracts["section_count"]) > 0
        and int(contracts["phase_count"]) > 0
    )


def _backend_status(backend: str) -> tuple[bool, str]:
    if backend in AVAILABLE_BACKENDS:
        return True, ""
    return False, f"{backend} backend was not resolved by monitor.poincare"


def _bench_with_output(
    backend: str,
    traj: NDArray[np.float64],
    phases: NDArray[np.float64],
    *,
    calls: int,
) -> tuple[float, dict[str, Any]]:
    _direct_bundle(backend, traj, phases)
    output: dict[str, Any] | None = None
    t0 = time.perf_counter()
    for _ in range(calls):
        output = _direct_bundle(backend, traj, phases)
    elapsed = time.perf_counter() - t0
    if output is None:
        raise RuntimeError("benchmark calls must be positive")
    return elapsed, output


def _bundle_shas(bundle: Mapping[str, Any]) -> dict[str, str]:
    return {
        "section_crossings_sha256": _array_sha256(bundle["section_crossings"]),
        "section_times_sha256": _array_sha256(bundle["section_times"]),
        "phase_crossings_sha256": _array_sha256(bundle["phase_crossings"]),
        "phase_times_sha256": _array_sha256(bundle["phase_times"]),
    }


def benchmark_poincare_polyglot_parity_gate(
    *,
    n_steps: int = 240,
    n_osc: int = 4,
    calls: int = 1,
    seed: int = 2026,
) -> dict[str, object]:
    """Record Poincaré-section parity across declared backend slots.

    Every available backend must reproduce the Python reference crossing
    points, crossing times, and crossing counts for both the generic
    hyperplane section and the phase-specific section, and the reference must
    satisfy the geometric contracts: section crossings lie on the hyperplane,
    phase crossings recover the section phase, both crossing-time sequences
    are strictly increasing, and both produce at least one crossing.
    """
    n_steps = _validate_int_control(n_steps, name="n_steps", minimum=8)
    n_osc = _validate_int_control(n_osc, name="n_osc", minimum=2)
    calls = _validate_int_control(calls, name="calls", minimum=1)
    seed = _validate_int_control(seed, name="seed", minimum=0)

    traj, phases = _problem(n_steps, n_osc, seed)
    reference = _direct_bundle("python", traj, phases)
    reference_contracts = _reference_contracts(reference)
    reference_contracts_passed = _reference_contracts_passed(reference_contracts)

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
                    "section_crossings_abs_error": None,
                    "section_times_abs_error": None,
                    "phase_crossings_abs_error": None,
                    "phase_times_abs_error": None,
                    "count_match": False,
                    "section_count": None,
                    "phase_count": None,
                    "section_crossings_sha256": None,
                    "section_times_sha256": None,
                    "phase_crossings_sha256": None,
                    "phase_times_sha256": None,
                    "reference_contracts_passed": False,
                    "tolerance": tolerance,
                    "parity_passed": False,
                    "unavailable_reason": reason,
                }
            )
            continue

        available_backend_count += 1
        elapsed, bundle = _bench_with_output(backend, traj, phases, calls=calls)
        errors = _bundle_errors(bundle, reference)
        max_abs_error = max(errors.values())
        contracts_passed = _reference_contracts_passed(_reference_contracts(bundle))
        parity_passed = bool(max_abs_error <= tolerance and contracts_passed)
        parity_checked_count += 1
        parity_pass_count += int(parity_passed)
        shas = _bundle_shas(bundle)
        records.append(
            {
                "backend": backend,
                "status": "available",
                "ms_per_call": (elapsed / calls) * 1000.0,
                "max_abs_error": max_abs_error,
                "section_crossings_abs_error": errors["section_crossings"],
                "section_times_abs_error": errors["section_times"],
                "phase_crossings_abs_error": errors["phase_crossings"],
                "phase_times_abs_error": errors["phase_times"],
                "count_match": errors["count"] == 0.0,
                "section_count": int(bundle["section_count"]),
                "phase_count": int(bundle["phase_count"]),
                **shas,
                "reference_contracts_passed": contracts_passed,
                "tolerance": tolerance,
                "parity_passed": parity_passed,
                "unavailable_reason": "",
            }
        )

    wall_time = time.perf_counter() - t0
    thresholds = {
        "backend_order": list(BACKEND_ORDER),
        "max_reference_contract_abs_error": REFERENCE_TOLERANCE,
        "parity_tolerances": PARITY_TOLERANCES,
        "require_all_available_parity": True,
        "require_all_declared_backend_records": True,
        "require_python_reference": True,
        "require_section_on_plane": True,
        "require_phase_recovers_section": True,
        "require_strictly_increasing_times": True,
        "require_nonzero_crossings": True,
        "require_matching_crossing_counts": True,
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
    reference_shas = _bundle_shas(reference)
    benchmark_payload = {
        "n_steps": n_steps,
        "n_osc": n_osc,
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
        "reference_shas": reference_shas,
    }
    benchmark_sha = hashlib.sha256(
        json.dumps(benchmark_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    return {
        "suite": "poincare_polyglot_parity_gate",
        "backend_count": len(records),
        "available_backend_count": available_backend_count,
        "unavailable_backend_count": len(records) - available_backend_count,
        "parity_checked_count": parity_checked_count,
        "parity_pass_count": parity_pass_count,
        "all_available_passed": int(parity_pass_count == parity_checked_count),
        "python_reference_present": 1,
        "reference_contracts_passed": int(reference_contracts_passed),
        "acceptance_passed": int(acceptance_passed),
        "n_steps": n_steps,
        "n_osc": n_osc,
        "calls": calls,
        "seed": seed,
        "reference_section_count": int(reference_contracts["section_count"]),
        "reference_phase_count": int(reference_contracts["phase_count"]),
        "section_plane_residual": float(reference_contracts["section_plane_residual"]),
        "phase_value_residual": float(reference_contracts["phase_value_residual"]),
        "reference_section_crossings_sha256": reference_shas[
            "section_crossings_sha256"
        ],
        "reference_phase_crossings_sha256": reference_shas["phase_crossings_sha256"],
        "benchmark_sha256": benchmark_sha,
        "benchmark_evidence_kind": BENCHMARK_EVIDENCE_KIND,
        "isolation_method": "none",
        "production_timing_claim": 0,
        "wall_time_s": wall_time,
        "steps_per_second": parity_checked_count / wall_time if wall_time else 0.0,
        "acceptance_thresholds_json": json.dumps(thresholds, sort_keys=True),
        "backend_records_json": json.dumps(records, sort_keys=True),
    }


def _bench(
    backend: str,
    traj: NDArray[np.float64],
    phases: NDArray[np.float64],
    calls: int,
) -> float:
    saved = pc_mod.ACTIVE_BACKEND
    try:
        pc_mod.ACTIVE_BACKEND = backend
        poincare_section(traj, _NORMAL, _OFFSET, direction="positive")
        t0 = time.perf_counter()
        for _ in range(calls):
            poincare_section(traj, _NORMAL, _OFFSET, direction="positive")
            phase_poincare(phases, _OSC_IDX, _SECTION_PHASE)
        return time.perf_counter() - t0
    finally:
        pc_mod.ACTIVE_BACKEND = saved


def bench_at(n_steps: int, n_osc: int, calls: int, *, seed: int = 2026) -> dict:
    traj, phases = _problem(n_steps, n_osc, seed)
    row: dict = {
        "n_steps": n_steps,
        "n_osc": n_osc,
        "calls": calls,
        "available": AVAILABLE_BACKENDS,
    }
    for backend in AVAILABLE_BACKENDS:
        t = _bench(backend, traj, phases, calls)
        row[f"{backend}_ms_per_call"] = (t / calls) * 1000.0
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--n-steps", type=int, default=240)
    parser.add_argument("--n-osc", type=int, default=4)
    parser.add_argument("--calls", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--parity-gate", action="store_true")
    args = parser.parse_args()

    if args.parity_gate:
        result = benchmark_poincare_polyglot_parity_gate(
            n_steps=args.n_steps,
            n_osc=args.n_osc,
            calls=args.calls,
            seed=args.seed,
        )
        text = json.dumps(result, indent=2, sort_keys=True)
        print(text)
        if args.output:
            args.output.write_text(text + "\n", encoding="utf-8")
        return 0

    print(f"Active: {ACTIVE_BACKEND}  Available: {AVAILABLE_BACKENDS}\n")
    row = bench_at(args.n_steps, args.n_osc, args.calls, seed=args.seed)
    for backend in AVAILABLE_BACKENDS:
        print(f"{backend:>8}: {row[f'{backend}_ms_per_call']:.4f} ms/call")
    if args.output:
        args.output.write_text(json.dumps(row, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
