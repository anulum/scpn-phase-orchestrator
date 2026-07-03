# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — PHA-C handoff benchmark gate

"""Per-backend source-contract parity gate for ``upde.pha_c_handoff``."""

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
    _pha_c_handoff_go,
    _pha_c_handoff_julia,
    _pha_c_handoff_mojo,
    _pha_c_handoff_rust,
    _pha_c_handoff_validation,
)
from scpn_phase_orchestrator.upde.pha_c_handoff import (
    PHA_C_HANDOFF_CLAIM_BOUNDARY,
    PHA_C_HANDOFF_MARGIN_REPLAY_TOLERANCE,
    PHACHandoffRecord,
    build_pha_c_handoff_record,
    verify_pha_c_handoff_record,
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
BackendFn = Callable[..., PHACHandoffRecord]
BACKEND_FUNCTIONS: dict[str, BackendFn] = {
    "rust": _pha_c_handoff_rust.build_pha_c_handoff_record_rust,
    "mojo": _pha_c_handoff_mojo.build_pha_c_handoff_record_mojo,
    "julia": _pha_c_handoff_julia.build_pha_c_handoff_record_julia,
    "go": _pha_c_handoff_go.build_pha_c_handoff_record_go,
    "python": build_pha_c_handoff_record,
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


def _problem(n: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build the deterministic handoff benchmark sample vectors."""
    phases = np.linspace(-0.004, 0.004, n, dtype=np.float64)
    positions = np.linspace(-0.001, 0.001, n, dtype=np.float64)
    return phases, positions


def _record_sha256(record: PHACHandoffRecord) -> str:
    """Return the SHA-256 digest of a canonical handoff payload."""
    return hashlib.sha256(
        json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _record_max_abs_error(
    got: PHACHandoffRecord,
    reference: PHACHandoffRecord,
) -> float:
    """Return the strict field-level parity error for handoff records."""
    return _pha_c_handoff_validation.pha_c_handoff_record_max_abs_error(
        got,
        reference,
    )


def _bench_backend(
    backend: str,
    phases: NDArray[np.float64],
    positions: NDArray[np.float64],
    calls: int,
) -> tuple[float, PHACHandoffRecord]:
    """Run one backend and return elapsed time plus verified handoff evidence."""
    fn = BACKEND_FUNCTIONS[backend]
    record = fn(
        phases,
        positions,
        t=2.5,
        phase_tol_rad=0.01,
        spatial_tol_m=0.002,
        required_consecutive_samples=3,
        prior_consecutive_lock_samples=2,
        tolerance_profile="buffer_3x",
    )
    t0 = time.perf_counter()
    for _ in range(calls):
        record = fn(
            phases,
            positions,
            t=2.5,
            phase_tol_rad=0.01,
            spatial_tol_m=0.002,
            required_consecutive_samples=3,
            prior_consecutive_lock_samples=2,
            tolerance_profile="buffer_3x",
        )
    return time.perf_counter() - t0, verify_pha_c_handoff_record(record)


def _reference_contracts(record: PHACHandoffRecord) -> dict[str, Any]:
    """Return acceptance-contract flags derived from the reference record."""
    margin_contracts = _margin_equation_contracts(record)
    return {
        "lock_achieved": int(record.lock_achieved),
        "joint_lock_required": int(record.phase_locked and record.spatial_locked),
        "phase_margin_positive": int(record.phase_margin_rad >= 0.0),
        "spatial_margin_positive": int(record.spatial_margin_m >= 0.0),
        "non_actuating": int(not record.actuating),
        "execution_disabled": int(record.execution_disabled),
        "claim_boundary": record.claim_boundary,
        "tolerance_profile_name": record.tolerance_profile_name,
        "tolerance_profile_multiplier": record.tolerance_profile_multiplier,
        "has_source_chain_hash": int(len(record.source_chain_sha256) == 64),
        "has_record_hash": int(len(record.record_sha256) == 64),
        "hash_replay_validated": int(verify_pha_c_handoff_record(record) is record),
        **margin_contracts,
    }


def _margin_equation_contracts(record: PHACHandoffRecord) -> dict[str, object]:
    """Return signed-margin replay flags for a handoff record."""
    phase_margin_equation_validated = (
        abs(
            record.phase_margin_rad
            - (record.phase_tol_rad - record.phase_dispersion_rad)
        )
        <= PHA_C_HANDOFF_MARGIN_REPLAY_TOLERANCE
    )
    spatial_margin_equation_validated = (
        abs(
            record.spatial_margin_m
            - (record.spatial_tol_m - record.spatial_dispersion_m)
        )
        <= PHA_C_HANDOFF_MARGIN_REPLAY_TOLERANCE
    )
    return {
        "phase_margin_equation_validated": int(phase_margin_equation_validated),
        "spatial_margin_equation_validated": int(spatial_margin_equation_validated),
        "signed_margin_equations_validated": int(
            phase_margin_equation_validated and spatial_margin_equation_validated
        ),
        "margin_replay_tolerance": PHA_C_HANDOFF_MARGIN_REPLAY_TOLERANCE,
    }


def benchmark_pha_c_handoff_polyglot_parity_gate(
    *,
    n: int = 8,
    calls: int = 3,
) -> dict[str, object]:
    """Record PHA-C handoff parity across declared backend slots."""

    n = _validate_int_control(n, name="n", minimum=2)
    calls = _validate_int_control(calls, name="calls", minimum=1)
    phases, positions = _problem(n)
    t0 = time.perf_counter()
    _, reference = _bench_backend("python", phases, positions, 1)
    contracts = _reference_contracts(reference)
    records: list[dict[str, object]] = []
    parity_pass_count = 0
    for backend in BACKEND_ORDER:
        tolerance = PARITY_TOLERANCES[backend]
        elapsed, got = _bench_backend(backend, phases, positions, calls)
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
                "record_sha256": got.record_sha256,
                "payload_sha256": _record_sha256(got),
                "source_chain_sha256": got.source_chain_sha256,
                "phase_margin_rad": got.phase_margin_rad,
                "spatial_margin_m": got.spatial_margin_m,
                **margin_contracts,
                "hash_replay_validated": 1,
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
        "require_non_actuating": True,
        "require_execution_disabled": True,
        "require_hash_chain": True,
        "require_hash_replay_validation": True,
        "require_signed_margin_contract": True,
        "require_signed_margin_equations": True,
        "margin_replay_tolerance": PHA_C_HANDOFF_MARGIN_REPLAY_TOLERANCE,
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
        and contracts["lock_achieved"] == 1
        and contracts["joint_lock_required"] == 1
        and contracts["phase_margin_positive"] == 1
        and contracts["spatial_margin_positive"] == 1
        and contracts["non_actuating"] == 1
        and contracts["execution_disabled"] == 1
        and contracts["claim_boundary"] == PHA_C_HANDOFF_CLAIM_BOUNDARY
        and contracts["tolerance_profile_name"] == "buffer_3x"
        and contracts["tolerance_profile_multiplier"] == 3.0
        and contracts["has_source_chain_hash"] == 1
        and contracts["has_record_hash"] == 1
        and contracts["hash_replay_validated"] == 1
        and contracts["signed_margin_equations_validated"] == 1
        and all(
            _payload_int(record["hash_replay_validated"], name="hash_replay_validated")
            == 1
            for record in records
        )
        and all(
            _payload_int(
                record["signed_margin_equations_validated"],
                name="signed_margin_equations_validated",
            )
            == 1
            for record in records
        )
    )
    benchmark_payload = {
        "n": n,
        "calls": calls,
        "reference_record_sha256": reference.record_sha256,
        "records": records,
        "contracts": contracts,
        "thresholds": thresholds,
    }
    benchmark_sha = hashlib.sha256(
        json.dumps(benchmark_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "suite": "pha_c_handoff_polyglot_parity_gate",
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
        "reference_record_sha256": reference.record_sha256,
        "reference_source_chain_sha256": reference.source_chain_sha256,
        "hash_replay_validated": contracts["hash_replay_validated"],
        "lock_achieved": contracts["lock_achieved"],
        "joint_lock_required": contracts["joint_lock_required"],
        "phase_margin_positive": contracts["phase_margin_positive"],
        "spatial_margin_positive": contracts["spatial_margin_positive"],
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
    """Run the command-line PHA-C handoff parity gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--calls", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--parity-gate", action="store_true")
    args = parser.parse_args()

    payload = benchmark_pha_c_handoff_polyglot_parity_gate(
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
