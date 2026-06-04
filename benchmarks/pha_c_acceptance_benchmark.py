# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — PHA-C acceptance benchmark gate

"""End-to-end PHA-C source-contract and subgate benchmark evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _pha_c_acceptance_go,
    _pha_c_acceptance_julia,
    _pha_c_acceptance_mojo,
    _pha_c_acceptance_rust,
)
from scpn_phase_orchestrator.upde.pha_c_acceptance import (
    PHA_C_ACCEPTANCE_CLAIM_BOUNDARY,
    PHACAcceptanceRecord,
    build_pha_c_acceptance_record,
)

ROOT = Path(__file__).resolve().parents[1]

BACKEND_ORDER = ("rust", "mojo", "julia", "go", "python")
PARITY_TOLERANCES = {
    "rust": 1.0e-12,
    "mojo": 1.0e-12,
    "julia": 1.0e-12,
    "go": 1.0e-12,
    "python": 0.0,
}
BENCHMARK_EVIDENCE_KIND = "local_regression_non_isolated"
BackendFn = Callable[..., PHACAcceptanceRecord]
BACKEND_FUNCTIONS: dict[str, BackendFn] = {
    "rust": _pha_c_acceptance_rust.build_pha_c_acceptance_record_rust,
    "mojo": _pha_c_acceptance_mojo.build_pha_c_acceptance_record_mojo,
    "julia": _pha_c_acceptance_julia.build_pha_c_acceptance_record_julia,
    "go": _pha_c_acceptance_go.build_pha_c_acceptance_record_go,
    "python": build_pha_c_acceptance_record,
}


def _validate_int_control(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return parsed


def _problem(
    n: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    phases = np.linspace(-0.002, 0.002, n, dtype=np.float64)
    positions = np.linspace(-0.0006, 0.0006, n, dtype=np.float64)
    knm = np.full((n, n), 0.04, dtype=np.float64)
    np.fill_diagonal(knm, 0.0)
    steps = 4
    omega = np.zeros((steps, n), dtype=np.float64)
    velocity_base = np.linspace(0.10, 0.12, n, dtype=np.float64)
    velocities = np.vstack(
        [velocity_base + 1.0e-3 * step for step in range(steps)],
    ).astype(np.float64, copy=False)
    return phases, positions, omega, knm, velocities


def _record_sha256(record: PHACAcceptanceRecord) -> str:
    return hashlib.sha256(
        json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")).encode(),
    ).hexdigest()


def _record_max_abs_error(
    got: PHACAcceptanceRecord,
    reference: PHACAcceptanceRecord,
) -> float:
    got_dict = got.to_dict()
    ref_dict = reference.to_dict()
    numeric_error = max(
        abs(float(got_dict[field]) - float(ref_dict[field]))
        for field in (
            "start_time",
            "end_time",
            "dt",
            "first_lock_time",
            "max_abs_doppler_term",
            "max_abs_spatial_coupling",
            "min_phase_order_parameter",
            "max_distance_to_reference_m",
            "tolerance_profile_multiplier",
        )
    )
    discrete_error = max(
        int(got_dict[field] != ref_dict[field])
        for field in (
            "sample_count",
            "step_count",
            "oscillator_count",
            "first_lock_index",
            "first_lock_observed",
            "final_lock_achieved",
            "lock_sample_count",
            "lock_loss_count",
            "reset_count",
            "tolerance_profile_name",
            "moving_frame_backend_request",
            "claim_boundary",
            "execution_disabled",
            "actuating",
            "omega_schedule_sha256",
            "velocity_schedule_sha256",
            "phase_trajectory_sha256",
            "position_trajectory_sha256",
            "timeline_sha256",
            "acceptance_sha256",
        )
    )
    return max(numeric_error, float(discrete_error))


def _bench_backend(
    backend: str,
    phases: NDArray[np.float64],
    positions: NDArray[np.float64],
    omega: NDArray[np.float64],
    knm: NDArray[np.float64],
    velocities: NDArray[np.float64],
    calls: int,
) -> tuple[float, PHACAcceptanceRecord]:
    fn = BACKEND_FUNCTIONS[backend]
    kwargs = {
        "dt": 1.0e-3,
        "required_consecutive_samples": 3,
        "tolerance_profile": "baseline_1x",
        "backend": "python",
    }
    record = fn(phases, positions, omega, knm, velocities, **kwargs)
    t0 = time.perf_counter()
    for _ in range(calls):
        record = fn(phases, positions, omega, knm, velocities, **kwargs)
    return time.perf_counter() - t0, record


def _gate_passed(payload: dict[str, object]) -> bool:
    for key in ("acceptance_passed", "all_available_passed"):
        if key in payload:
            return int(payload[key]) == 1
    return False


def _subgate_records() -> list[dict[str, object]]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    gate_paths = (
        (
            "benchmarks.spatial_modulator_benchmark",
            "benchmark_spatial_modulator_polyglot_parity_gate",
        ),
        (
            "benchmarks.upde_time_varying_omega_benchmark",
            "benchmark_upde_time_varying_omega_polyglot_gate",
        ),
        ("benchmarks.upde_doppler_benchmark", "benchmark_upde_doppler_polyglot_gate"),
        (
            "benchmarks.upde_moving_frame_benchmark",
            "benchmark_upde_moving_frame_polyglot_gate",
        ),
        (
            "benchmarks.merge_window_benchmark",
            "benchmark_merge_window_polyglot_parity_gate",
        ),
        (
            "benchmarks.pha_c_handoff_benchmark",
            "benchmark_pha_c_handoff_polyglot_parity_gate",
        ),
        (
            "benchmarks.pha_c_timeline_benchmark",
            "benchmark_pha_c_timeline_polyglot_parity_gate",
        ),
    )
    records: list[dict[str, object]] = []
    for module_name, attr_name in gate_paths:
        module = importlib.import_module(module_name)
        gate = getattr(module, attr_name)
        payload = gate()
        records.append(
            {
                "suite": payload.get("suite", "unknown"),
                "passed": int(_gate_passed(payload)),
                "backend_count": int(payload.get("backend_count", 0)),
                "parity_pass_count": int(payload.get("parity_pass_count", 0)),
                "benchmark_evidence_kind": payload.get(
                    "benchmark_evidence_kind",
                    BENCHMARK_EVIDENCE_KIND,
                ),
            },
        )
    return records


def _reference_contracts(record: PHACAcceptanceRecord) -> dict[str, Any]:
    return {
        "first_lock_observed": int(record.first_lock_observed),
        "first_lock_index": record.first_lock_index,
        "final_lock_achieved": int(record.final_lock_achieved),
        "lock_loss_count": record.lock_loss_count,
        "reset_count": record.reset_count,
        "non_actuating": int(not record.actuating),
        "execution_disabled": int(record.execution_disabled),
        "claim_boundary": record.claim_boundary,
        "tolerance_profile_name": record.tolerance_profile_name,
        "tolerance_profile_multiplier": record.tolerance_profile_multiplier,
        "has_acceptance_hash": int(len(record.acceptance_sha256) == 64),
        "has_timeline_hash": int(len(record.timeline_sha256) == 64),
    }


def benchmark_pha_c_acceptance_polyglot_gate(
    *,
    n: int = 8,
    calls: int = 2,
    include_subgates: bool = True,
) -> dict[str, object]:
    """Record end-to-end PHA-C parity and subgate acceptance evidence."""

    n = _validate_int_control(n, name="n", minimum=3)
    calls = _validate_int_control(calls, name="calls", minimum=1)
    phases, positions, omega, knm, velocities = _problem(n)
    t0 = time.perf_counter()
    _, reference = _bench_backend(
        "python",
        phases,
        positions,
        omega,
        knm,
        velocities,
        1,
    )
    contracts = _reference_contracts(reference)
    records: list[dict[str, object]] = []
    parity_pass_count = 0
    for backend in BACKEND_ORDER:
        tolerance = PARITY_TOLERANCES[backend]
        elapsed, got = _bench_backend(
            backend,
            phases,
            positions,
            omega,
            knm,
            velocities,
            calls,
        )
        error = _record_max_abs_error(got, reference)
        passed = error <= tolerance
        parity_pass_count += int(passed)
        records.append(
            {
                "backend": backend,
                "status": "available",
                "ms_per_call": (elapsed / calls) * 1000.0,
                "acceptance_sha256": got.acceptance_sha256,
                "payload_sha256": _record_sha256(got),
                "timeline_sha256": got.timeline_sha256,
                "max_abs_error": error,
                "tolerance": tolerance,
                "parity_passed": passed,
                "unavailable_reason": "",
            },
        )
    subgates = _subgate_records() if include_subgates else []
    subgate_pass_count = sum(int(record["passed"]) for record in subgates)
    wall_time = time.perf_counter() - t0
    thresholds = {
        "backend_order": list(BACKEND_ORDER),
        "max_abs_error": 1.0e-12,
        "production_timing_claim": False,
        "require_all_declared_backend_records": True,
        "require_first_lock_observed": True,
        "require_final_lock": True,
        "require_subgates": bool(include_subgates),
        "require_non_actuating": True,
        "require_execution_disabled": True,
        "require_acceptance_hash": True,
        "require_python_reference": True,
    }
    acceptance_passed = (
        len(records) == len(BACKEND_ORDER)
        and parity_pass_count == len(BACKEND_ORDER)
        and contracts["first_lock_observed"] == 1
        and contracts["first_lock_index"] == 2
        and contracts["final_lock_achieved"] == 1
        and contracts["lock_loss_count"] == 0
        and contracts["reset_count"] == 0
        and contracts["non_actuating"] == 1
        and contracts["execution_disabled"] == 1
        and contracts["claim_boundary"] == PHA_C_ACCEPTANCE_CLAIM_BOUNDARY
        and contracts["tolerance_profile_name"] == "baseline_1x"
        and contracts["tolerance_profile_multiplier"] == 1.0
        and contracts["has_acceptance_hash"] == 1
        and contracts["has_timeline_hash"] == 1
        and (not include_subgates or subgate_pass_count == len(subgates))
    )
    benchmark_payload = {
        "n": n,
        "calls": calls,
        "reference_acceptance_sha256": reference.acceptance_sha256,
        "records": [
            {
                "backend": record["backend"],
                "parity_passed": record["parity_passed"],
                "max_abs_error": record["max_abs_error"],
                "acceptance_sha256": record["acceptance_sha256"],
            }
            for record in records
        ],
        "contracts": contracts,
        "subgates": subgates,
        "thresholds": thresholds,
    }
    benchmark_sha = hashlib.sha256(
        json.dumps(benchmark_payload, sort_keys=True, separators=(",", ":")).encode(),
    ).hexdigest()
    return {
        "suite": "pha_c_acceptance_polyglot_gate",
        "backend_count": len(records),
        "available_backend_count": len(records),
        "unavailable_backend_count": 0,
        "parity_checked_count": len(records),
        "parity_pass_count": parity_pass_count,
        "subgate_count": len(subgates),
        "subgate_pass_count": subgate_pass_count,
        "all_available_passed": int(parity_pass_count == len(records)),
        "python_reference_present": 1,
        "n": n,
        "calls": calls,
        "reference_acceptance_sha256": reference.acceptance_sha256,
        "reference_timeline_sha256": reference.timeline_sha256,
        "first_lock_observed": contracts["first_lock_observed"],
        "first_lock_index": contracts["first_lock_index"],
        "final_lock_achieved": contracts["final_lock_achieved"],
        "lock_loss_count": contracts["lock_loss_count"],
        "reset_count": contracts["reset_count"],
        "non_actuating": contracts["non_actuating"],
        "execution_disabled": contracts["execution_disabled"],
        "tolerance_profile_name": contracts["tolerance_profile_name"],
        "tolerance_profile_multiplier": contracts["tolerance_profile_multiplier"],
        "benchmark_sha256": benchmark_sha,
        "benchmark_evidence_kind": BENCHMARK_EVIDENCE_KIND,
        "isolation_method": "none",
        "production_timing_claim": 0,
        "wall_time_s": wall_time,
        "steps_per_second": len(records) / wall_time if wall_time else 0.0,
        "acceptance_passed": int(acceptance_passed),
        "acceptance_thresholds_json": json.dumps(thresholds, sort_keys=True),
        "backend_records_json": json.dumps(records, sort_keys=True),
        "subgate_records_json": json.dumps(subgates, sort_keys=True),
    }


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--calls", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--parity-gate", action="store_true")
    parser.add_argument("--skip-subgates", action="store_true")
    args = parser.parse_args()

    payload = benchmark_pha_c_acceptance_polyglot_gate(
        n=args.n,
        calls=args.calls,
        include_subgates=not args.skip_subgates,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.parity_gate and int(payload["acceptance_passed"]) != 1:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
