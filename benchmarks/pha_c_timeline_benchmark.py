# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — PHA-C timeline benchmark gate

"""Per-backend source-contract parity gate for ``upde.pha_c_timeline``."""

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

from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _pha_c_timeline_go,
    _pha_c_timeline_julia,
    _pha_c_timeline_mojo,
    _pha_c_timeline_rust,
    _pha_c_timeline_validation,
)
from scpn_phase_orchestrator.upde.pha_c_timeline import (
    PHA_C_TIMELINE_CLAIM_BOUNDARY,
    PHA_C_TIMELINE_MARGIN_REPLAY_TOLERANCE,
    PHACTimelineRecord,
    build_pha_c_event_timeline,
    verify_pha_c_event_timeline,
)

BACKEND_ORDER = ("rust", "mojo", "julia", "go", "python")
BACKEND_EXECUTION_MODES = {
    "rust": "source_contract_reference_validation",
    "mojo": "source_contract_reference_validation",
    "julia": "source_contract_reference_validation",
    "go": "source_contract_reference_validation",
    "python": "python_reference",
}
POLYGLOT_CLAIM_BOUNDARY = "source_contract_not_native_kernel"
PARITY_TOLERANCES = {
    "rust": 1.0e-12,
    "mojo": 1.0e-12,
    "julia": 1.0e-12,
    "go": 1.0e-12,
    "python": 0.0,
}
BENCHMARK_EVIDENCE_KIND = "local_regression_non_isolated"
BackendFn = Callable[..., PHACTimelineRecord]
BACKEND_FUNCTIONS: dict[str, BackendFn] = {
    "rust": _pha_c_timeline_rust.build_pha_c_event_timeline_rust,
    "mojo": _pha_c_timeline_mojo.build_pha_c_event_timeline_mojo,
    "julia": _pha_c_timeline_julia.build_pha_c_event_timeline_julia,
    "go": _pha_c_timeline_go.build_pha_c_event_timeline_go,
    "python": build_pha_c_event_timeline,
}


def _validate_int_control(value: object, *, name: str, minimum: int) -> int:
    """Return a benchmark integer control after fail-closed validation."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return parsed


def _payload_int(value: object, *, name: str) -> int:
    """Return a JSON-like payload integer after rejecting booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _problem(
    n: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Build the deterministic timeline benchmark trajectory."""
    locked_phase = np.linspace(-0.002, 0.002, n, dtype=np.float64)
    locked_position = np.linspace(-0.0005, 0.0005, n, dtype=np.float64)
    failed_phase = np.linspace(-0.02, 0.02, n, dtype=np.float64)
    failed_position = np.linspace(-0.003, 0.003, n, dtype=np.float64)
    phases = np.vstack(
        (
            failed_phase,
            locked_phase,
            locked_phase * 0.8,
            locked_phase * 0.6,
            failed_phase,
        ),
    ).astype(np.float64, copy=False)
    positions = np.vstack(
        (
            failed_position,
            locked_position,
            locked_position * 0.8,
            locked_position * 0.6,
            failed_position,
        ),
    ).astype(np.float64, copy=False)
    times = np.arange(phases.shape[0], dtype=np.float64) * 0.5
    return phases, positions, times


def _timeline_sha256(record: PHACTimelineRecord) -> str:
    """Return the SHA-256 digest of a canonical timeline payload."""
    return hashlib.sha256(
        json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")).encode(),
    ).hexdigest()


def _record_max_abs_error(
    got: PHACTimelineRecord,
    reference: PHACTimelineRecord,
) -> float:
    """Return the strict field-level parity error for timeline records."""
    return _pha_c_timeline_validation.pha_c_timeline_record_max_abs_error(
        got,
        reference,
    )


def _bench_backend(
    backend: str,
    phases: NDArray[np.float64],
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    calls: int,
) -> tuple[float, PHACTimelineRecord]:
    """Run one backend and return elapsed time plus verified timeline evidence."""
    fn = BACKEND_FUNCTIONS[backend]
    record = fn(
        phases,
        positions,
        times=times,
        phase_tol_rad=0.01,
        spatial_tol_m=0.002,
        required_consecutive_samples=3,
        tolerance_profile="baseline_1x",
    )
    t0 = time.perf_counter()
    for _ in range(calls):
        record = fn(
            phases,
            positions,
            times=times,
            phase_tol_rad=0.01,
            spatial_tol_m=0.002,
            required_consecutive_samples=3,
            tolerance_profile="baseline_1x",
        )
    return time.perf_counter() - t0, verify_pha_c_event_timeline(record)


def _margin_equation_contracts(record: PHACTimelineRecord) -> dict[str, object]:
    """Return signed-margin replay flags for a timeline record."""
    phase_validated = (
        abs(
            record.min_phase_margin_rad
            - (record.phase_tol_rad - record.max_phase_dispersion_rad)
        )
        <= PHA_C_TIMELINE_MARGIN_REPLAY_TOLERANCE
    )
    spatial_validated = (
        abs(
            record.min_spatial_margin_m
            - (record.spatial_tol_m - record.max_spatial_dispersion_m)
        )
        <= PHA_C_TIMELINE_MARGIN_REPLAY_TOLERANCE
    )
    return {
        "phase_margin_equation_validated": int(phase_validated),
        "spatial_margin_equation_validated": int(spatial_validated),
        "signed_margin_equations_validated": int(phase_validated and spatial_validated),
        "margin_replay_tolerance": PHA_C_TIMELINE_MARGIN_REPLAY_TOLERANCE,
    }


def _reference_contracts(record: PHACTimelineRecord) -> dict[str, Any]:
    """Return acceptance-contract flags derived from the reference timeline."""
    margin_contracts = _margin_equation_contracts(record)
    return {
        "first_lock_observed": int(record.first_lock_observed),
        "first_lock_index": record.first_lock_index,
        "final_lock_achieved": int(record.final_lock_achieved),
        "lock_loss_count": record.lock_loss_count,
        "reset_count": record.reset_count,
        "phase_margin_loss_observed": int(record.min_phase_margin_rad < 0.0),
        "spatial_margin_loss_observed": int(record.min_spatial_margin_m < 0.0),
        **margin_contracts,
        "non_actuating": int(not record.actuating),
        "execution_disabled": int(record.execution_disabled),
        "claim_boundary": record.claim_boundary,
        "tolerance_profile_name": record.tolerance_profile_name,
        "tolerance_profile_multiplier": record.tolerance_profile_multiplier,
        "has_sample_records_hash": int(len(record.sample_records_sha256) == 64),
        "has_timeline_hash": int(len(record.timeline_sha256) == 64),
        "hash_replay_validated": int(verify_pha_c_event_timeline(record) is record),
    }


def benchmark_pha_c_timeline_polyglot_parity_gate(
    *,
    n: int = 8,
    calls: int = 3,
) -> dict[str, object]:
    """Record PHA-C timeline parity across declared backend slots."""

    n = _validate_int_control(n, name="n", minimum=2)
    calls = _validate_int_control(calls, name="calls", minimum=1)
    phases, positions, times = _problem(n)
    t0 = time.perf_counter()
    _, reference = _bench_backend("python", phases, positions, times, 1)
    contracts = _reference_contracts(reference)
    records: list[dict[str, object]] = []
    parity_pass_count = 0
    for backend in BACKEND_ORDER:
        tolerance = PARITY_TOLERANCES[backend]
        elapsed, got = _bench_backend(backend, phases, positions, times, calls)
        error = _record_max_abs_error(got, reference)
        margin_contracts = _margin_equation_contracts(got)
        passed = error <= tolerance
        parity_pass_count += int(passed)
        records.append(
            {
                "backend": backend,
                "status": "available",
                "execution_mode": BACKEND_EXECUTION_MODES[backend],
                "native_kernel_present": 0,
                "source_contract_validation": int(backend != "python"),
                "ms_per_call": (elapsed / calls) * 1000.0,
                "timeline_sha256": got.timeline_sha256,
                "payload_sha256": _timeline_sha256(got),
                "sample_records_sha256": got.sample_records_sha256,
                "transition_table_sha256": got.transition_table_sha256,
                "min_phase_margin_rad": got.min_phase_margin_rad,
                "min_spatial_margin_m": got.min_spatial_margin_m,
                **margin_contracts,
                "hash_replay_validated": 1,
                "max_abs_error": error,
                "tolerance": tolerance,
                "parity_passed": passed,
                "unavailable_reason": "",
            },
        )
    wall_time = time.perf_counter() - t0
    thresholds = {
        "backend_order": list(BACKEND_ORDER),
        "max_abs_error": 1.0e-12,
        "production_timing_claim": False,
        "require_all_declared_backend_records": True,
        "require_first_lock_observed": True,
        "require_lock_loss_observed": True,
        "require_non_actuating": True,
        "require_execution_disabled": True,
        "require_hash_chain": True,
        "require_hash_replay_validation": True,
        "require_signed_margin_contract": True,
        "require_signed_margin_equations": True,
        "margin_replay_tolerance": PHA_C_TIMELINE_MARGIN_REPLAY_TOLERANCE,
        "require_python_reference": True,
        "require_source_contract_disclosure": True,
        "require_no_native_kernel_claim": True,
    }
    source_contract_count = sum(
        _payload_int(
            record["source_contract_validation"], name="source_contract_validation"
        )
        for record in records
    )
    native_kernel_count = sum(
        _payload_int(record["native_kernel_present"], name="native_kernel_present")
        for record in records
    )
    acceptance_passed = (
        len(records) == len(BACKEND_ORDER)
        and parity_pass_count == len(BACKEND_ORDER)
        and source_contract_count == len(BACKEND_ORDER) - 1
        and native_kernel_count == 0
        and contracts["first_lock_observed"] == 1
        and contracts["first_lock_index"] == 3
        and contracts["final_lock_achieved"] == 0
        and contracts["lock_loss_count"] == 1
        and contracts["reset_count"] == 1
        and contracts["phase_margin_loss_observed"] == 1
        and contracts["spatial_margin_loss_observed"] == 1
        and contracts["signed_margin_equations_validated"] == 1
        and contracts["non_actuating"] == 1
        and contracts["execution_disabled"] == 1
        and contracts["claim_boundary"] == PHA_C_TIMELINE_CLAIM_BOUNDARY
        and contracts["tolerance_profile_name"] == "baseline_1x"
        and contracts["tolerance_profile_multiplier"] == 1.0
        and contracts["has_sample_records_hash"] == 1
        and contracts["has_timeline_hash"] == 1
        and contracts["hash_replay_validated"] == 1
        and all(
            _payload_int(
                record["signed_margin_equations_validated"],
                name="signed_margin_equations_validated",
            )
            == 1
            for record in records
        )
        and all(
            _payload_int(record["hash_replay_validated"], name="hash_replay_validated")
            == 1
            for record in records
        )
    )
    benchmark_payload = {
        "n": n,
        "calls": calls,
        "reference_timeline_sha256": reference.timeline_sha256,
        "records": records,
        "contracts": contracts,
        "thresholds": thresholds,
    }
    benchmark_sha = hashlib.sha256(
        json.dumps(benchmark_payload, sort_keys=True, separators=(",", ":")).encode(),
    ).hexdigest()
    return {
        "suite": "pha_c_timeline_polyglot_parity_gate",
        "backend_count": len(records),
        "available_backend_count": len(records),
        "unavailable_backend_count": 0,
        "parity_checked_count": len(records),
        "parity_pass_count": parity_pass_count,
        "source_contract_backend_count": source_contract_count,
        "native_kernel_count": native_kernel_count,
        "polyglot_claim_boundary": POLYGLOT_CLAIM_BOUNDARY,
        "all_available_passed": int(parity_pass_count == len(records)),
        "python_reference_present": 1,
        "n": n,
        "calls": calls,
        "reference_timeline_sha256": reference.timeline_sha256,
        "reference_sample_records_sha256": reference.sample_records_sha256,
        "hash_replay_validated": contracts["hash_replay_validated"],
        "first_lock_observed": contracts["first_lock_observed"],
        "first_lock_index": contracts["first_lock_index"],
        "final_lock_achieved": contracts["final_lock_achieved"],
        "lock_loss_count": contracts["lock_loss_count"],
        "reset_count": contracts["reset_count"],
        "phase_margin_loss_observed": contracts["phase_margin_loss_observed"],
        "spatial_margin_loss_observed": contracts["spatial_margin_loss_observed"],
        "phase_margin_equation_validated": contracts["phase_margin_equation_validated"],
        "spatial_margin_equation_validated": contracts[
            "spatial_margin_equation_validated"
        ],
        "signed_margin_equations_validated": contracts[
            "signed_margin_equations_validated"
        ],
        "margin_replay_tolerance": contracts["margin_replay_tolerance"],
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
    }


def _main() -> int:
    """Run the command-line PHA-C timeline parity gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--calls", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--parity-gate", action="store_true")
    args = parser.parse_args()

    payload = benchmark_pha_c_timeline_polyglot_parity_gate(
        n=args.n,
        calls=args.calls,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if (
        args.parity_gate
        and _payload_int(payload["acceptance_passed"], name="acceptance_passed") != 1
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
