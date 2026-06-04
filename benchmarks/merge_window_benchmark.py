# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — merge-window benchmark gate

"""Per-backend parity gate for ``monitor.merge_window``."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _merge_window_go,
    _merge_window_julia,
    _merge_window_mojo,
    _merge_window_rust,
)
from scpn_phase_orchestrator.monitor.merge_window import (
    MergeReport,
    evaluate_merge_window,
)

BACKEND_ORDER = ("rust", "mojo", "julia", "go", "python")
PARITY_TOLERANCES = {
    "rust": 1.0e-12,
    "mojo": 1.0e-12,
    "julia": 1.0e-12,
    "go": 1.0e-12,
    "python": 0.0,
}
BENCHMARK_EVIDENCE_KIND = "local_regression_non_isolated"
BackendFn = Callable[..., MergeReport]
BACKEND_FUNCTIONS: dict[str, BackendFn] = {
    "rust": _merge_window_rust.evaluate_merge_window_rust,
    "mojo": _merge_window_mojo.evaluate_merge_window_mojo,
    "julia": _merge_window_julia.evaluate_merge_window_julia,
    "go": _merge_window_go.evaluate_merge_window_go,
    "python": evaluate_merge_window,
}


def _validate_int_control(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return parsed


def _problem(n: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    phases = np.linspace(-0.006, 0.006, n, dtype=np.float64)
    positions = np.linspace(-0.001, 0.001, n, dtype=np.float64)
    return phases, positions


def _bench_backend(
    backend: str,
    phases: NDArray[np.float64],
    positions: NDArray[np.float64],
    calls: int,
) -> tuple[float, MergeReport]:
    fn = BACKEND_FUNCTIONS[backend]
    report = fn(
        phases,
        positions,
        t=2.5,
        phase_tol_rad=0.01,
        spatial_tol_m=0.002,
        required_consecutive_samples=3,
        prior_consecutive_lock_samples=2,
    )
    t0 = time.perf_counter()
    for _ in range(calls):
        report = fn(
            phases,
            positions,
            t=2.5,
            phase_tol_rad=0.01,
            spatial_tol_m=0.002,
            required_consecutive_samples=3,
            prior_consecutive_lock_samples=2,
        )
    return time.perf_counter() - t0, report


def _report_sha256(report: MergeReport) -> str:
    return hashlib.sha256(
        json.dumps(report.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _report_max_abs_error(got: MergeReport, reference: MergeReport) -> float:
    got_dict = got.to_dict()
    ref_dict = reference.to_dict()
    numeric_error = max(
        abs(float(got_dict[field]) - float(ref_dict[field]))
        for field in ("t", "phase_dispersion_rad", "spatial_dispersion_m")
    )
    discrete_error = max(
        int(got_dict[field] != ref_dict[field])
        for field in (
            "phase_locked",
            "spatial_locked",
            "lock_achieved",
            "consecutive_lock_samples",
        )
    )
    return max(numeric_error, float(discrete_error))


def _reference_contracts() -> dict[str, Any]:
    wrapped = evaluate_merge_window(
        np.array([2.0 * np.pi - 0.004, 0.003], dtype=np.float64),
        np.array([0.0, 0.0], dtype=np.float64),
        phase_tol_rad=0.005,
        spatial_tol_m=0.001,
        required_consecutive_samples=1,
    )
    phase_fail = evaluate_merge_window(
        np.array([0.0, 0.02], dtype=np.float64),
        np.array([0.0, 0.001], dtype=np.float64),
        phase_tol_rad=0.01,
        spatial_tol_m=0.002,
        required_consecutive_samples=1,
    )
    spatial_fail = evaluate_merge_window(
        np.array([0.0, 0.002], dtype=np.float64),
        np.array([0.0, 0.003], dtype=np.float64),
        phase_tol_rad=0.01,
        spatial_tol_m=0.002,
        required_consecutive_samples=1,
    )
    consecutive_pass = evaluate_merge_window(
        np.array([0.0, 0.002], dtype=np.float64),
        np.array([0.0, 0.001], dtype=np.float64),
        phase_tol_rad=0.01,
        spatial_tol_m=0.002,
        required_consecutive_samples=3,
        prior_consecutive_lock_samples=2,
    )
    profile_pass = evaluate_merge_window(
        np.array([0.0, 0.024], dtype=np.float64),
        np.array([0.0, 0.0045], dtype=np.float64),
        phase_tol_rad=0.01,
        spatial_tol_m=0.002,
        required_consecutive_samples=1,
        tolerance_profile="buffer_3x",
    )
    profile_fail = evaluate_merge_window(
        np.array([0.0, 0.024], dtype=np.float64),
        np.array([0.0, 0.0045], dtype=np.float64),
        phase_tol_rad=0.01,
        spatial_tol_m=0.002,
        required_consecutive_samples=1,
    )
    return {
        "wrapped_phase_locked": bool(wrapped.phase_locked),
        "wrapped_phase_dispersion_rad": wrapped.phase_dispersion_rad,
        "phase_failure_resets": int(phase_fail.consecutive_lock_samples == 0),
        "spatial_failure_resets": int(spatial_fail.consecutive_lock_samples == 0),
        "joint_lock_required": int(
            not phase_fail.lock_achieved and not spatial_fail.lock_achieved
        ),
        "consecutive_gate_passes_at_threshold": int(consecutive_pass.lock_achieved),
        "buffer_profile_accepts_within_3x": int(profile_pass.lock_achieved),
        "explicit_profile_rejects_same_sample": int(not profile_fail.lock_achieved),
    }


def benchmark_merge_window_polyglot_parity_gate(
    *,
    n: int = 8,
    calls: int = 3,
) -> dict[str, object]:
    """Record merge-window parity across declared backend slots."""

    n = _validate_int_control(n, name="n", minimum=2)
    calls = _validate_int_control(calls, name="calls", minimum=1)
    phases, positions = _problem(n)
    t0 = time.perf_counter()
    _, reference = _bench_backend("python", phases, positions, 1)
    contracts = _reference_contracts()
    records: list[dict[str, object]] = []
    parity_pass_count = 0
    for backend in BACKEND_ORDER:
        tolerance = PARITY_TOLERANCES[backend]
        elapsed, got = _bench_backend(backend, phases, positions, calls)
        error = _report_max_abs_error(got, reference)
        passed = error <= tolerance
        parity_pass_count += int(passed)
        records.append(
            {
                "backend": backend,
                "status": "available",
                "ms_per_call": (elapsed / calls) * 1000.0,
                "report_sha256": _report_sha256(got),
                "max_abs_error": error,
                "tolerance": tolerance,
                "parity_passed": passed,
                "unavailable_reason": "",
            }
        )
    wall_time = time.perf_counter() - t0
    thresholds = {
        "backend_order": list(BACKEND_ORDER),
        "max_abs_error": 1.0e-12,
        "production_timing_claim": False,
        "require_all_declared_backend_records": True,
        "require_consecutive_gate": True,
        "require_joint_phase_spatial_lock": True,
        "require_tolerance_profile_contract": True,
        "require_python_reference": True,
        "require_wrapped_phase": True,
    }
    acceptance_passed = (
        len(records) == len(BACKEND_ORDER)
        and parity_pass_count == len(BACKEND_ORDER)
        and bool(contracts["wrapped_phase_locked"])
        and contracts["wrapped_phase_dispersion_rad"] <= 0.004 + 1.0e-12
        and contracts["phase_failure_resets"] == 1
        and contracts["spatial_failure_resets"] == 1
        and contracts["joint_lock_required"] == 1
        and contracts["consecutive_gate_passes_at_threshold"] == 1
        and contracts["buffer_profile_accepts_within_3x"] == 1
        and contracts["explicit_profile_rejects_same_sample"] == 1
    )
    benchmark_payload = {
        "n": n,
        "calls": calls,
        "reference_sha256": _report_sha256(reference),
        "records": records,
        "contracts": contracts,
        "thresholds": thresholds,
    }
    benchmark_sha = hashlib.sha256(
        json.dumps(benchmark_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "suite": "merge_window_polyglot_parity_gate",
        "backend_count": len(records),
        "available_backend_count": len(records),
        "unavailable_backend_count": 0,
        "parity_checked_count": len(records),
        "parity_pass_count": parity_pass_count,
        "all_available_passed": int(parity_pass_count == len(records)),
        "python_reference_present": 1,
        "n": n,
        "calls": calls,
        "reference_report_sha256": _report_sha256(reference),
        "wrapped_phase_locked": int(bool(contracts["wrapped_phase_locked"])),
        "phase_failure_resets": contracts["phase_failure_resets"],
        "spatial_failure_resets": contracts["spatial_failure_resets"],
        "joint_lock_required": contracts["joint_lock_required"],
        "consecutive_gate_passes_at_threshold": contracts[
            "consecutive_gate_passes_at_threshold"
        ],
        "buffer_profile_accepts_within_3x": contracts[
            "buffer_profile_accepts_within_3x"
        ],
        "explicit_profile_rejects_same_sample": contracts[
            "explicit_profile_rejects_same_sample"
        ],
        "benchmark_sha256": benchmark_sha,
        "benchmark_evidence_kind": BENCHMARK_EVIDENCE_KIND,
        "isolation_method": "none",
        "production_timing_claim": 0,
        "wall_time_s": wall_time,
        "steps_per_second": len(records) / wall_time if wall_time else 0.0,
        "acceptance_passed": int(acceptance_passed),
        "acceptance_thresholds_json": json.dumps(thresholds, sort_keys=True),
        "backend_records_json": json.dumps(records, sort_keys=True),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8])
    parser.add_argument("--calls", type=int, default=3)
    parser.add_argument("--parity-gate", action="store_true")
    args = parser.parse_args()
    if args.parity_gate:
        result = benchmark_merge_window_polyglot_parity_gate(
            n=args.sizes[0],
            calls=args.calls,
        )
        payload = json.dumps(result, indent=2, sort_keys=True)
        if args.output:
            args.output.write_text(payload + "\n", encoding="utf-8")
        print(payload)
        return 0
    results = [
        benchmark_merge_window_polyglot_parity_gate(n=size, calls=args.calls)
        for size in args.sizes
    ]
    if args.output:
        args.output.write_text(
            json.dumps({"results": results}, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"results": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
