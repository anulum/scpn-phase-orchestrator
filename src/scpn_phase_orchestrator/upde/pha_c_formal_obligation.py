# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — PHA-C Lean proof obligations

"""Deterministic Lean proof-obligation manifests for PHA-C acceptance records.

The PHA-C acceptance chain is runtime evidence. The Lean kinematic proofs are
formal evidence. This module binds the two surfaces by projecting a verified
``PHACAcceptanceRecord`` into fixed-point natural-number obligations that match
``SPOFormal.Kinematic.KinematicBounds``. The resulting manifest remains
review-only and non-actuating; it is a reproducible bridge for release review,
MIF/FRC specialisation, and benchmark gating.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal
from math import isfinite
from typing import Any

import numpy as np

from scpn_phase_orchestrator.upde.pha_c_acceptance import (
    PHA_C_ACCEPTANCE_CLAIM_BOUNDARY,
    PHA_C_ACCEPTANCE_KINEMATIC_SUMMARY_REPLAY_TOLERANCE,
    PHACAcceptanceRecord,
    verify_pha_c_acceptance_record,
)

PHA_C_FORMAL_OBLIGATION_CLAIM_BOUNDARY = "pha_c_lean_kinematic_obligation_review_only"
PHA_C_FORMAL_OBLIGATION_EVIDENCE_KIND = "deterministic_lean_kinematic_obligation"
PHA_C_FORMAL_OBLIGATION_SCHEMA = "pha_c_lean_kinematic_obligation_v1"
PHA_C_FORMAL_LEAN_MODULE = "SPOFormal.Kinematic"
PHA_C_FORMAL_CERTIFICATE_PREDICATE = "KinematicBounds.budgetCertificate"
PHA_C_FORMAL_CERTIFICATE_THEOREM = "budget_certificate_discharges_budget"
PHA_C_FORMAL_ZERO_GAIN_CERTIFICATE_PREDICATE = "KinematicBounds.zeroGainCertificate"
PHA_C_FORMAL_ZERO_GAIN_CERTIFICATE_THEOREM = "zero_gain_certificate_discharges_budget"
PHA_C_FORMAL_CONTINUOUS_LEAN_MODULE = "SPOFormal.Continuous"
PHA_C_FORMAL_CONTINUOUS_CERTIFICATE_PREDICATE = (
    "ContinuousEnvelopeBounds.budgetCertificate"
)
PHA_C_FORMAL_CONTINUOUS_CERTIFICATE_THEOREM = (
    "continuous_envelope_certificate_discharges_horizon"
)
PHA_C_FORMAL_PHASE_LEAN_MODULE = "SPOFormal.Kinematic"
PHA_C_FORMAL_PHASE_CERTIFICATE_PREDICATE = "PhaseBudgetBounds.budgetCertificate"
PHA_C_FORMAL_PHASE_CERTIFICATE_THEOREM = (
    "phase_budget_certificate_discharges_phase_lock"
)
PHA_C_FORMAL_ACCEPTANCE_CERTIFICATE_PREDICATE = "KinematicBounds.acceptanceCertificate"
PHA_C_FORMAL_ACCEPTANCE_CERTIFICATE_THEOREM = (
    "acceptance_certificate_discharges_runtime_preconditions"
)
PHA_C_FORMAL_DEFAULT_SCALE_M = 1.0e-6
PHA_C_FORMAL_DEFAULT_SCALE_RAD = 1.0e-6
PHA_C_FORMAL_DEFAULT_TIME_SCALE_S = 1.0e-6

__all__ = [
    "PHA_C_FORMAL_ACCEPTANCE_CERTIFICATE_PREDICATE",
    "PHA_C_FORMAL_ACCEPTANCE_CERTIFICATE_THEOREM",
    "PHA_C_FORMAL_CERTIFICATE_PREDICATE",
    "PHA_C_FORMAL_CERTIFICATE_THEOREM",
    "PHA_C_FORMAL_CONTINUOUS_CERTIFICATE_PREDICATE",
    "PHA_C_FORMAL_CONTINUOUS_CERTIFICATE_THEOREM",
    "PHA_C_FORMAL_CONTINUOUS_LEAN_MODULE",
    "PHA_C_FORMAL_DEFAULT_SCALE_M",
    "PHA_C_FORMAL_DEFAULT_SCALE_RAD",
    "PHA_C_FORMAL_DEFAULT_TIME_SCALE_S",
    "PHA_C_FORMAL_LEAN_MODULE",
    "PHA_C_FORMAL_OBLIGATION_CLAIM_BOUNDARY",
    "PHA_C_FORMAL_OBLIGATION_EVIDENCE_KIND",
    "PHA_C_FORMAL_OBLIGATION_SCHEMA",
    "PHA_C_FORMAL_PHASE_CERTIFICATE_PREDICATE",
    "PHA_C_FORMAL_PHASE_CERTIFICATE_THEOREM",
    "PHA_C_FORMAL_PHASE_LEAN_MODULE",
    "PHA_C_FORMAL_ZERO_GAIN_CERTIFICATE_PREDICATE",
    "PHA_C_FORMAL_ZERO_GAIN_CERTIFICATE_THEOREM",
    "PHACKinematicProofObligation",
    "build_pha_c_kinematic_proof_obligation",
    "pha_c_kinematic_proof_obligation_to_dict",
    "verify_pha_c_kinematic_proof_obligation",
]


@dataclass(frozen=True, slots=True)
class PHACKinematicProofObligation:
    """Review-only fixed-point obligations linked to the Lean kinematic proof."""

    schema_version: str
    evidence_kind: str
    claim_boundary: str
    acceptance_claim_boundary: str
    execution_disabled: bool
    actuating: bool
    lean_module: str
    lean_certificate_predicate: str
    lean_theorem: str
    continuous_lean_module: str
    continuous_certificate_predicate: str
    continuous_theorem: str
    phase_lean_module: str
    phase_certificate_predicate: str
    phase_theorem: str
    acceptance_certificate_predicate: str
    acceptance_certificate_theorem: str
    fixed_point_scale_m: float
    fixed_point_scale_rad: float
    fixed_point_time_scale_s: float
    time_step_s: float
    time_scale_units_per_second: int
    time_step_units: int
    horizon_time_units: int
    initial_tolerance_units: int
    lipschitz_step_gain_units: int
    relative_velocity_rate_bound_units_per_second: int
    relative_velocity_step_bound_units: int
    configured_coupling_residual_step_bound_units: int
    coupling_residual_rate_bound_units_per_second: int
    coupling_residual_step_bound_units: int
    continuous_drive_rate_bound_units_per_second: int
    continuous_horizon_drive_bound_units: int
    continuous_linear_budget_units: int
    continuous_margin_units: int
    drive_bound_units: int
    merge_window_tolerance_units: int
    horizon_steps: int
    linear_budget_units: int
    gronwall_budget_units: int
    gronwall_budget_margin_units: int
    gronwall_budget_trace_sha256: str
    window_budget_margin_units: int
    phase_tolerance_units: int
    max_phase_dispersion_units: int
    configured_phase_drift_bound_units: int
    phase_budget_units: int
    phase_margin_units: int
    phase_budget_discharged: bool
    acceptance_kinematic_equations_validated: bool
    acceptance_kinematic_summary_replay_tolerance: float
    acceptance_kinematic_summary_replay_tolerance_units: int
    acceptance_kinematic_summary_replay_tolerance_limit_units: int
    acceptance_replay_certificate_discharged: bool
    acceptance_certificate_discharged: bool
    observed_velocity_step_units: int
    kinematic_residual_units: int
    path_length_units: int
    max_spatial_dispersion_units: int
    continuous_envelope_discharged: bool
    proof_obligations_discharged: bool
    acceptance_sha256: str
    timeline_sha256: str
    record_sha256: str

    def to_dict(self) -> dict[str, bool | float | int | str]:
        """Return a JSON-safe canonical representation.

        Returns
        -------
        dict[str, bool | float | int | str]
            Return a JSON-safe canonical representation.
        """
        return pha_c_kinematic_proof_obligation_to_dict(self)


_SHA256_HEX_DIGITS = frozenset("0123456789abcdef")


def _sha256_json(payload: object) -> str:
    """Return the canonical-JSON SHA-256 hash of a payload."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _gronwall_budget_trace(
    *,
    initial_tolerance_units: int,
    lipschitz_step_gain_units: int,
    drive_bound_units: int,
    horizon_steps: int,
) -> tuple[int, ...]:
    """Return the Gronwall error-budget trace for the obligation."""
    budget = initial_tolerance_units
    trace = [budget]
    for _ in range(horizon_steps):
        budget = budget + lipschitz_step_gain_units * budget + drive_bound_units
        trace.append(budget)
    return tuple(trace)


def _gronwall_budget_trace_sha256(
    *,
    trace_units: tuple[int, ...],
    horizon_steps: int,
) -> str:
    """Return the SHA-256 hash of the Gronwall budget trace."""
    return _sha256_json(
        {
            "budget_trace_units": list(trace_units),
            "horizon_steps": horizon_steps,
            "lean_recurrence": "previous + gain * previous + drive",
        }
    )


def _validate_positive_scale(value: object, *, name: str) -> float:
    """Return ``value`` as a strictly positive scale, else raise."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite positive scalar")
    try:
        parsed = float(value)  # type: ignore[arg-type]  # type ignore: runtime validator coerces object input
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive scalar") from exc
    if not isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return parsed


def _validate_nonnegative_scalar(value: object, *, name: str) -> float:
    """Return ``value`` as a non-negative scalar, else raise."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite non-negative scalar")
    try:
        parsed = float(value)  # type: ignore[arg-type]  # type ignore: runtime validator coerces object input
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative scalar") from exc
    if not isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return parsed


def _nonnegative_units(value: object, *, scale: float, name: str) -> int:
    """Return ``value`` as a non-negative integer unit count, else raise."""
    parsed = _validate_nonnegative_scalar(value, name=name)
    return int(
        (Decimal(str(parsed)) / Decimal(str(scale))).to_integral_value(
            rounding=ROUND_CEILING,
        )
    )


def _ceil_div_units(numerator: int, denominator: int, *, name: str) -> int:
    """Return the ceiling division of two unit counts."""
    if denominator <= 0:
        raise ValueError(f"{name} denominator must be positive")
    return (numerator + denominator - 1) // denominator


def _ceil_positive_ratio_units(
    numerator: float,
    denominator: float,
    *,
    name: str,
) -> int:
    """Return the ceiling of a positive unit ratio."""
    if denominator <= 0.0:
        raise ValueError(f"{name} denominator must be positive")
    return int(
        (Decimal(str(numerator)) / Decimal(str(denominator))).to_integral_value(
            rounding=ROUND_CEILING,
        )
    )


def _validate_int(value: object, *, name: str, minimum: int) -> int:
    """Return ``value`` as a validated integer, else raise ``ValueError``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return parsed


def _validate_bool(value: object, *, name: str) -> bool:
    """Return ``value`` if it is a real boolean, else raise ``ValueError``."""
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _validate_sha256_hex(value: object, *, name: str) -> str:
    """Return ``value`` if it is a SHA-256 hex digest, else raise."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in _SHA256_HEX_DIGITS for char in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _dict_without_record_hash(
    obligation: PHACKinematicProofObligation,
) -> dict[str, bool | float | int | str]:
    """Return the record mapping without its hash field."""
    return {
        "schema_version": obligation.schema_version,
        "evidence_kind": obligation.evidence_kind,
        "claim_boundary": obligation.claim_boundary,
        "acceptance_claim_boundary": obligation.acceptance_claim_boundary,
        "execution_disabled": obligation.execution_disabled,
        "actuating": obligation.actuating,
        "lean_module": obligation.lean_module,
        "lean_certificate_predicate": obligation.lean_certificate_predicate,
        "lean_theorem": obligation.lean_theorem,
        "continuous_lean_module": obligation.continuous_lean_module,
        "continuous_certificate_predicate": (
            obligation.continuous_certificate_predicate
        ),
        "continuous_theorem": obligation.continuous_theorem,
        "phase_lean_module": obligation.phase_lean_module,
        "phase_certificate_predicate": obligation.phase_certificate_predicate,
        "phase_theorem": obligation.phase_theorem,
        "acceptance_certificate_predicate": (
            obligation.acceptance_certificate_predicate
        ),
        "acceptance_certificate_theorem": obligation.acceptance_certificate_theorem,
        "fixed_point_scale_m": obligation.fixed_point_scale_m,
        "fixed_point_scale_rad": obligation.fixed_point_scale_rad,
        "fixed_point_time_scale_s": obligation.fixed_point_time_scale_s,
        "time_step_s": obligation.time_step_s,
        "time_scale_units_per_second": obligation.time_scale_units_per_second,
        "time_step_units": obligation.time_step_units,
        "horizon_time_units": obligation.horizon_time_units,
        "initial_tolerance_units": obligation.initial_tolerance_units,
        "lipschitz_step_gain_units": obligation.lipschitz_step_gain_units,
        "relative_velocity_rate_bound_units_per_second": (
            obligation.relative_velocity_rate_bound_units_per_second
        ),
        "relative_velocity_step_bound_units": (
            obligation.relative_velocity_step_bound_units
        ),
        "configured_coupling_residual_step_bound_units": (
            obligation.configured_coupling_residual_step_bound_units
        ),
        "coupling_residual_rate_bound_units_per_second": (
            obligation.coupling_residual_rate_bound_units_per_second
        ),
        "coupling_residual_step_bound_units": (
            obligation.coupling_residual_step_bound_units
        ),
        "continuous_drive_rate_bound_units_per_second": (
            obligation.continuous_drive_rate_bound_units_per_second
        ),
        "continuous_horizon_drive_bound_units": (
            obligation.continuous_horizon_drive_bound_units
        ),
        "continuous_linear_budget_units": obligation.continuous_linear_budget_units,
        "continuous_margin_units": obligation.continuous_margin_units,
        "drive_bound_units": obligation.drive_bound_units,
        "merge_window_tolerance_units": obligation.merge_window_tolerance_units,
        "horizon_steps": obligation.horizon_steps,
        "linear_budget_units": obligation.linear_budget_units,
        "gronwall_budget_units": obligation.gronwall_budget_units,
        "gronwall_budget_margin_units": obligation.gronwall_budget_margin_units,
        "gronwall_budget_trace_sha256": obligation.gronwall_budget_trace_sha256,
        "window_budget_margin_units": obligation.window_budget_margin_units,
        "phase_tolerance_units": obligation.phase_tolerance_units,
        "max_phase_dispersion_units": obligation.max_phase_dispersion_units,
        "configured_phase_drift_bound_units": (
            obligation.configured_phase_drift_bound_units
        ),
        "phase_budget_units": obligation.phase_budget_units,
        "phase_margin_units": obligation.phase_margin_units,
        "phase_budget_discharged": obligation.phase_budget_discharged,
        "acceptance_kinematic_equations_validated": (
            obligation.acceptance_kinematic_equations_validated
        ),
        "acceptance_kinematic_summary_replay_tolerance": (
            obligation.acceptance_kinematic_summary_replay_tolerance
        ),
        "acceptance_kinematic_summary_replay_tolerance_units": (
            obligation.acceptance_kinematic_summary_replay_tolerance_units
        ),
        "acceptance_kinematic_summary_replay_tolerance_limit_units": (
            obligation.acceptance_kinematic_summary_replay_tolerance_limit_units
        ),
        "acceptance_replay_certificate_discharged": (
            obligation.acceptance_replay_certificate_discharged
        ),
        "acceptance_certificate_discharged": (
            obligation.acceptance_certificate_discharged
        ),
        "observed_velocity_step_units": obligation.observed_velocity_step_units,
        "kinematic_residual_units": obligation.kinematic_residual_units,
        "path_length_units": obligation.path_length_units,
        "max_spatial_dispersion_units": obligation.max_spatial_dispersion_units,
        "continuous_envelope_discharged": obligation.continuous_envelope_discharged,
        "proof_obligations_discharged": obligation.proof_obligations_discharged,
        "acceptance_sha256": obligation.acceptance_sha256,
        "timeline_sha256": obligation.timeline_sha256,
    }


def pha_c_kinematic_proof_obligation_to_dict(
    obligation: PHACKinematicProofObligation,
) -> dict[str, bool | float | int | str]:
    """Return a canonical JSON-safe proof-obligation manifest.

    Parameters
    ----------
    obligation : PHACKinematicProofObligation
        The PHA-C kinematic proof obligation to operate on.

    Returns
    -------
    dict[str, bool | float | int | str]
        The canonical JSON-safe proof-obligation manifest.
    """
    payload = _dict_without_record_hash(obligation)
    payload["record_sha256"] = obligation.record_sha256
    return payload


def build_pha_c_kinematic_proof_obligation(
    record: PHACAcceptanceRecord,
    *,
    fixed_point_scale_m: float = PHA_C_FORMAL_DEFAULT_SCALE_M,
    fixed_point_scale_rad: float = PHA_C_FORMAL_DEFAULT_SCALE_RAD,
    fixed_point_time_scale_s: float = PHA_C_FORMAL_DEFAULT_TIME_SCALE_S,
    relative_velocity_step_bound_m: float = 0.0,
    coupling_residual_step_bound_m: float = 0.0,
    phase_drift_bound_rad: float = 0.0,
    lipschitz_step_gain_units: int = 0,
) -> PHACKinematicProofObligation:
    """Project a verified PHA-C acceptance record into Lean proof obligations.

    The default obligation is a replay certificate: the maximum observed
    spatial dispersion is already measured over the accepted trajectory, so the
    Lean drive term only includes explicitly supplied future relative-velocity
    slack and the signed moving-frame residual. MIF/FRC specialisations can
    provide non-zero ``relative_velocity_step_bound_m`` and
    ``coupling_residual_step_bound_m``, ``phase_drift_bound_rad``, and
    ``lipschitz_step_gain_units`` values when they want a predictive
    finite-horizon Gronwall certificate instead of a replay-only envelope.

    Parameters
    ----------
    record : PHACAcceptanceRecord
        The PHA-C record to operate on.
    fixed_point_scale_m : float
        Spatial fixed-point scale in metres.
    fixed_point_scale_rad : float
        Phase fixed-point scale in radians.
    fixed_point_time_scale_s : float
        Temporal fixed-point scale in seconds.
    relative_velocity_step_bound_m : float
        Per-step relative-velocity bound in metres.
    coupling_residual_step_bound_m : float
        Per-step coupling-residual bound in metres.
    phase_drift_bound_rad : float
        Per-step phase-drift bound in radians.
    lipschitz_step_gain_units : int
        Lipschitz step-gain bound in dimensionless integer units.

    Returns
    -------
    PHACKinematicProofObligation
        The Lean proof-obligation projection of the acceptance record.
    """
    verified_record = verify_pha_c_acceptance_record(record)
    scale_m = _validate_positive_scale(fixed_point_scale_m, name="fixed_point_scale_m")
    scale_rad = _validate_positive_scale(
        fixed_point_scale_rad,
        name="fixed_point_scale_rad",
    )
    time_scale = _validate_positive_scale(
        fixed_point_time_scale_s,
        name="fixed_point_time_scale_s",
    )
    time_step_s = _validate_positive_scale(verified_record.dt, name="time_step_s")
    time_scale_units_per_second = _validate_int(
        _ceil_positive_ratio_units(
            1.0,
            time_scale,
            name="time_scale_units_per_second",
        ),
        name="time_scale_units_per_second",
        minimum=1,
    )
    time_step_units = _validate_int(
        _ceil_positive_ratio_units(
            time_step_s,
            time_scale,
            name="time_step_units",
        ),
        name="time_step_units",
        minimum=1,
    )
    horizon_steps = _validate_int(
        verified_record.step_count,
        name="step_count",
        minimum=1,
    )
    horizon_time_units = horizon_steps * time_step_units
    gain_units = _validate_int(
        lipschitz_step_gain_units,
        name="lipschitz_step_gain_units",
        minimum=0,
    )
    raw_relative_velocity_units = _nonnegative_units(
        relative_velocity_step_bound_m,
        scale=scale_m,
        name="relative_velocity_step_bound_m",
    )
    configured_residual_m = _validate_nonnegative_scalar(
        coupling_residual_step_bound_m,
        name="coupling_residual_step_bound_m",
    )
    observed_residual_m = _validate_nonnegative_scalar(
        verified_record.kinematic_residual_max_m,
        name="kinematic_residual_max_m",
    )
    configured_residual_units = _nonnegative_units(
        configured_residual_m,
        scale=scale_m,
        name="coupling_residual_step_bound_m",
    )
    observed_residual_units = _nonnegative_units(
        observed_residual_m,
        scale=scale_m,
        name="kinematic_residual_max_m",
    )
    raw_residual_units = max(configured_residual_units, observed_residual_units)
    residual_bound_m = max(configured_residual_m, observed_residual_m)
    relative_velocity_rate_units = max(
        _nonnegative_units(
            relative_velocity_step_bound_m / time_step_s,
            scale=scale_m,
            name="relative_velocity_rate_bound_m_per_s",
        ),
        _ceil_div_units(
            raw_relative_velocity_units * time_scale_units_per_second,
            time_step_units,
            name="relative_velocity_rate_bound_units_per_second",
        ),
    )
    residual_rate_units = max(
        _nonnegative_units(
            residual_bound_m / time_step_s,
            scale=scale_m,
            name="coupling_residual_rate_bound_m_per_s",
        ),
        _ceil_div_units(
            raw_residual_units * time_scale_units_per_second,
            time_step_units,
            name="coupling_residual_rate_bound_units_per_second",
        ),
    )
    relative_velocity_units = _ceil_div_units(
        relative_velocity_rate_units * time_step_units,
        time_scale_units_per_second,
        name="relative_velocity_step_bound_units",
    )
    residual_units = _ceil_div_units(
        residual_rate_units * time_step_units,
        time_scale_units_per_second,
        name="coupling_residual_step_bound_units",
    )
    drive_units = relative_velocity_units + residual_units
    initial_units = _nonnegative_units(
        verified_record.max_spatial_dispersion_m,
        scale=scale_m,
        name="max_spatial_dispersion_m",
    )
    merge_tolerance_units = _nonnegative_units(
        verified_record.spatial_tol_m,
        scale=scale_m,
        name="spatial_tol_m",
    )
    continuous_drive_rate_units = relative_velocity_rate_units + residual_rate_units
    continuous_horizon_drive_units = _ceil_div_units(
        continuous_drive_rate_units * horizon_time_units,
        time_scale_units_per_second,
        name="continuous_horizon_drive_bound_units",
    )
    continuous_linear_budget_units = initial_units + continuous_horizon_drive_units
    continuous_margin_units = merge_tolerance_units - continuous_linear_budget_units
    continuous_envelope_discharged = continuous_margin_units >= 0
    linear_budget_units = initial_units + horizon_steps * drive_units
    gronwall_trace_units = _gronwall_budget_trace(
        initial_tolerance_units=initial_units,
        lipschitz_step_gain_units=gain_units,
        drive_bound_units=drive_units,
        horizon_steps=horizon_steps,
    )
    gronwall_budget_units = gronwall_trace_units[-1]
    gronwall_budget_margin_units = merge_tolerance_units - gronwall_budget_units
    gronwall_trace_sha256 = _gronwall_budget_trace_sha256(
        trace_units=gronwall_trace_units,
        horizon_steps=horizon_steps,
    )
    window_margin_units = gronwall_budget_margin_units
    phase_tolerance_units = _nonnegative_units(
        verified_record.phase_tol_rad,
        scale=scale_rad,
        name="phase_tol_rad",
    )
    phase_dispersion_units = _nonnegative_units(
        verified_record.max_phase_dispersion_rad,
        scale=scale_rad,
        name="max_phase_dispersion_rad",
    )
    configured_phase_drift_units = _nonnegative_units(
        phase_drift_bound_rad,
        scale=scale_rad,
        name="phase_drift_bound_rad",
    )
    phase_budget_units = phase_dispersion_units + configured_phase_drift_units
    phase_margin_units = phase_tolerance_units - phase_budget_units
    phase_budget_discharged = phase_margin_units >= 0
    acceptance_tolerance_units = _nonnegative_units(
        verified_record.kinematic_summary_replay_tolerance,
        scale=scale_m,
        name="acceptance_kinematic_summary_replay_tolerance",
    )
    acceptance_tolerance_limit_units = _nonnegative_units(
        PHA_C_ACCEPTANCE_KINEMATIC_SUMMARY_REPLAY_TOLERANCE,
        scale=scale_m,
        name="acceptance_kinematic_summary_replay_tolerance_limit",
    )
    acceptance_replay_certificate_discharged = (
        verified_record.kinematic_equations_validated
        and acceptance_tolerance_units <= acceptance_tolerance_limit_units
    )
    acceptance_certificate_discharged = (
        gronwall_budget_margin_units >= 0
        and phase_budget_discharged
        and acceptance_replay_certificate_discharged
    )
    observed_velocity_step_units = _nonnegative_units(
        verified_record.max_abs_velocity_m_per_s * verified_record.dt,
        scale=scale_m,
        name="observed_velocity_step_m",
    )
    path_length_units = _nonnegative_units(
        verified_record.path_length_max_m,
        scale=scale_m,
        name="path_length_max_m",
    )
    discharged = (
        window_margin_units >= 0
        and phase_margin_units >= 0
        and phase_budget_discharged
        and acceptance_certificate_discharged
        and continuous_envelope_discharged
        and verified_record.execution_disabled
        and not verified_record.actuating
        and verified_record.claim_boundary == PHA_C_ACCEPTANCE_CLAIM_BOUNDARY
    )
    payload_without_hash: dict[str, Any] = {
        "schema_version": PHA_C_FORMAL_OBLIGATION_SCHEMA,
        "evidence_kind": PHA_C_FORMAL_OBLIGATION_EVIDENCE_KIND,
        "claim_boundary": PHA_C_FORMAL_OBLIGATION_CLAIM_BOUNDARY,
        "acceptance_claim_boundary": verified_record.claim_boundary,
        "execution_disabled": True,
        "actuating": False,
        "lean_module": PHA_C_FORMAL_LEAN_MODULE,
        "lean_certificate_predicate": PHA_C_FORMAL_CERTIFICATE_PREDICATE,
        "lean_theorem": PHA_C_FORMAL_CERTIFICATE_THEOREM,
        "continuous_lean_module": PHA_C_FORMAL_CONTINUOUS_LEAN_MODULE,
        "continuous_certificate_predicate": (
            PHA_C_FORMAL_CONTINUOUS_CERTIFICATE_PREDICATE
        ),
        "continuous_theorem": PHA_C_FORMAL_CONTINUOUS_CERTIFICATE_THEOREM,
        "phase_lean_module": PHA_C_FORMAL_PHASE_LEAN_MODULE,
        "phase_certificate_predicate": PHA_C_FORMAL_PHASE_CERTIFICATE_PREDICATE,
        "phase_theorem": PHA_C_FORMAL_PHASE_CERTIFICATE_THEOREM,
        "acceptance_certificate_predicate": (
            PHA_C_FORMAL_ACCEPTANCE_CERTIFICATE_PREDICATE
        ),
        "acceptance_certificate_theorem": (PHA_C_FORMAL_ACCEPTANCE_CERTIFICATE_THEOREM),
        "fixed_point_scale_m": scale_m,
        "fixed_point_scale_rad": scale_rad,
        "fixed_point_time_scale_s": time_scale,
        "time_step_s": time_step_s,
        "time_scale_units_per_second": time_scale_units_per_second,
        "time_step_units": time_step_units,
        "horizon_time_units": horizon_time_units,
        "initial_tolerance_units": initial_units,
        "lipschitz_step_gain_units": gain_units,
        "relative_velocity_rate_bound_units_per_second": relative_velocity_rate_units,
        "relative_velocity_step_bound_units": relative_velocity_units,
        "configured_coupling_residual_step_bound_units": configured_residual_units,
        "coupling_residual_rate_bound_units_per_second": residual_rate_units,
        "coupling_residual_step_bound_units": residual_units,
        "continuous_drive_rate_bound_units_per_second": continuous_drive_rate_units,
        "continuous_horizon_drive_bound_units": continuous_horizon_drive_units,
        "continuous_linear_budget_units": continuous_linear_budget_units,
        "continuous_margin_units": continuous_margin_units,
        "drive_bound_units": drive_units,
        "merge_window_tolerance_units": merge_tolerance_units,
        "horizon_steps": horizon_steps,
        "linear_budget_units": linear_budget_units,
        "gronwall_budget_units": gronwall_budget_units,
        "gronwall_budget_margin_units": gronwall_budget_margin_units,
        "gronwall_budget_trace_sha256": gronwall_trace_sha256,
        "window_budget_margin_units": window_margin_units,
        "phase_tolerance_units": phase_tolerance_units,
        "max_phase_dispersion_units": phase_dispersion_units,
        "configured_phase_drift_bound_units": configured_phase_drift_units,
        "phase_budget_units": phase_budget_units,
        "phase_margin_units": phase_margin_units,
        "phase_budget_discharged": phase_budget_discharged,
        "acceptance_kinematic_equations_validated": (
            verified_record.kinematic_equations_validated
        ),
        "acceptance_kinematic_summary_replay_tolerance": (
            verified_record.kinematic_summary_replay_tolerance
        ),
        "acceptance_kinematic_summary_replay_tolerance_units": (
            acceptance_tolerance_units
        ),
        "acceptance_kinematic_summary_replay_tolerance_limit_units": (
            acceptance_tolerance_limit_units
        ),
        "acceptance_replay_certificate_discharged": (
            acceptance_replay_certificate_discharged
        ),
        "acceptance_certificate_discharged": acceptance_certificate_discharged,
        "observed_velocity_step_units": observed_velocity_step_units,
        "kinematic_residual_units": observed_residual_units,
        "path_length_units": path_length_units,
        "max_spatial_dispersion_units": initial_units,
        "continuous_envelope_discharged": continuous_envelope_discharged,
        "proof_obligations_discharged": discharged,
        "acceptance_sha256": verified_record.acceptance_sha256,
        "timeline_sha256": verified_record.timeline_sha256,
    }
    return PHACKinematicProofObligation(
        **payload_without_hash,
        record_sha256=_sha256_json(payload_without_hash),
    )


def verify_pha_c_kinematic_proof_obligation(
    obligation: PHACKinematicProofObligation,
) -> PHACKinematicProofObligation:
    """Validate a PHA-C Lean proof-obligation manifest fail-closed.

    Parameters
    ----------
    obligation : PHACKinematicProofObligation
        The PHA-C kinematic proof obligation to operate on.

    Returns
    -------
    PHACKinematicProofObligation
        The same obligation after fail-closed validation.

    Raises
    ------
    TypeError
        If the manifest has the wrong type.
    ValueError
        If the manifest fails validation.
    """
    if not isinstance(obligation, PHACKinematicProofObligation):
        raise TypeError("obligation must be a PHACKinematicProofObligation")
    exact_strings = {
        "schema_version": PHA_C_FORMAL_OBLIGATION_SCHEMA,
        "evidence_kind": PHA_C_FORMAL_OBLIGATION_EVIDENCE_KIND,
        "claim_boundary": PHA_C_FORMAL_OBLIGATION_CLAIM_BOUNDARY,
        "acceptance_claim_boundary": PHA_C_ACCEPTANCE_CLAIM_BOUNDARY,
        "lean_module": PHA_C_FORMAL_LEAN_MODULE,
        "lean_certificate_predicate": PHA_C_FORMAL_CERTIFICATE_PREDICATE,
        "lean_theorem": PHA_C_FORMAL_CERTIFICATE_THEOREM,
        "continuous_lean_module": PHA_C_FORMAL_CONTINUOUS_LEAN_MODULE,
        "continuous_certificate_predicate": (
            PHA_C_FORMAL_CONTINUOUS_CERTIFICATE_PREDICATE
        ),
        "continuous_theorem": PHA_C_FORMAL_CONTINUOUS_CERTIFICATE_THEOREM,
        "phase_lean_module": PHA_C_FORMAL_PHASE_LEAN_MODULE,
        "phase_certificate_predicate": PHA_C_FORMAL_PHASE_CERTIFICATE_PREDICATE,
        "phase_theorem": PHA_C_FORMAL_PHASE_CERTIFICATE_THEOREM,
        "acceptance_certificate_predicate": (
            PHA_C_FORMAL_ACCEPTANCE_CERTIFICATE_PREDICATE
        ),
        "acceptance_certificate_theorem": (PHA_C_FORMAL_ACCEPTANCE_CERTIFICATE_THEOREM),
    }
    for field, expected in exact_strings.items():
        got = getattr(obligation, field)
        if got != expected:
            raise ValueError(f"{field} must be {expected!r}")

    if not _validate_bool(
        obligation.execution_disabled,
        name="execution_disabled",
    ):
        raise ValueError("execution_disabled must be true")
    if _validate_bool(obligation.actuating, name="actuating"):
        raise ValueError("actuating must be false")
    _validate_positive_scale(
        obligation.fixed_point_scale_m,
        name="fixed_point_scale_m",
    )
    _validate_positive_scale(
        obligation.fixed_point_scale_rad,
        name="fixed_point_scale_rad",
    )
    _validate_positive_scale(
        obligation.fixed_point_time_scale_s,
        name="fixed_point_time_scale_s",
    )
    _validate_positive_scale(obligation.time_step_s, name="time_step_s")
    nat_fields = (
        "time_scale_units_per_second",
        "time_step_units",
        "horizon_time_units",
        "initial_tolerance_units",
        "lipschitz_step_gain_units",
        "relative_velocity_rate_bound_units_per_second",
        "relative_velocity_step_bound_units",
        "configured_coupling_residual_step_bound_units",
        "coupling_residual_rate_bound_units_per_second",
        "coupling_residual_step_bound_units",
        "continuous_drive_rate_bound_units_per_second",
        "continuous_horizon_drive_bound_units",
        "continuous_linear_budget_units",
        "drive_bound_units",
        "merge_window_tolerance_units",
        "horizon_steps",
        "linear_budget_units",
        "gronwall_budget_units",
        "phase_tolerance_units",
        "max_phase_dispersion_units",
        "configured_phase_drift_bound_units",
        "phase_budget_units",
        "acceptance_kinematic_summary_replay_tolerance_units",
        "acceptance_kinematic_summary_replay_tolerance_limit_units",
        "observed_velocity_step_units",
        "kinematic_residual_units",
        "path_length_units",
        "max_spatial_dispersion_units",
    )
    for field in nat_fields:
        _validate_int(getattr(obligation, field), name=field, minimum=0)
    _validate_int(
        obligation.gronwall_budget_margin_units,
        name="gronwall_budget_margin_units",
        minimum=-(10**18),
    )
    _validate_int(
        obligation.window_budget_margin_units,
        name="window_budget_margin_units",
        minimum=-(10**18),
    )
    _validate_int(
        obligation.continuous_margin_units,
        name="continuous_margin_units",
        minimum=-(10**18),
    )
    _validate_sha256_hex(
        obligation.gronwall_budget_trace_sha256,
        name="gronwall_budget_trace_sha256",
    )
    _validate_int(
        obligation.phase_margin_units,
        name="phase_margin_units",
        minimum=-(10**18),
    )
    _validate_bool(
        obligation.phase_budget_discharged,
        name="phase_budget_discharged",
    )
    if not _validate_bool(
        obligation.acceptance_kinematic_equations_validated,
        name="acceptance_kinematic_equations_validated",
    ):
        raise ValueError("acceptance_kinematic_equations_validated must be true")
    kinematic_replay_tolerance = _validate_positive_scale(
        obligation.acceptance_kinematic_summary_replay_tolerance,
        name="acceptance_kinematic_summary_replay_tolerance",
    )
    if (
        kinematic_replay_tolerance
        != PHA_C_ACCEPTANCE_KINEMATIC_SUMMARY_REPLAY_TOLERANCE
    ):
        raise ValueError(
            "acceptance_kinematic_summary_replay_tolerance must match "
            "the acceptance constant"
        )
    expected_acceptance_tolerance_units = _nonnegative_units(
        kinematic_replay_tolerance,
        scale=obligation.fixed_point_scale_m,
        name="acceptance_kinematic_summary_replay_tolerance",
    )
    if (
        obligation.acceptance_kinematic_summary_replay_tolerance_units
        != expected_acceptance_tolerance_units
    ):
        raise ValueError(
            "acceptance_kinematic_summary_replay_tolerance_units must replay",
        )
    expected_acceptance_tolerance_limit_units = _nonnegative_units(
        PHA_C_ACCEPTANCE_KINEMATIC_SUMMARY_REPLAY_TOLERANCE,
        scale=obligation.fixed_point_scale_m,
        name="acceptance_kinematic_summary_replay_tolerance_limit",
    )
    if (
        obligation.acceptance_kinematic_summary_replay_tolerance_limit_units
        != expected_acceptance_tolerance_limit_units
    ):
        raise ValueError(
            "acceptance_kinematic_summary_replay_tolerance_limit_units must replay",
        )
    _validate_bool(
        obligation.acceptance_replay_certificate_discharged,
        name="acceptance_replay_certificate_discharged",
    )
    _validate_bool(
        obligation.acceptance_certificate_discharged,
        name="acceptance_certificate_discharged",
    )
    _validate_bool(
        obligation.continuous_envelope_discharged,
        name="continuous_envelope_discharged",
    )
    _validate_bool(
        obligation.proof_obligations_discharged,
        name="proof_obligations_discharged",
    )
    _validate_sha256_hex(obligation.acceptance_sha256, name="acceptance_sha256")
    _validate_sha256_hex(obligation.timeline_sha256, name="timeline_sha256")
    _validate_sha256_hex(obligation.record_sha256, name="record_sha256")

    expected_time_scale_units = _ceil_positive_ratio_units(
        1.0,
        obligation.fixed_point_time_scale_s,
        name="time_scale_units_per_second",
    )
    if obligation.time_scale_units_per_second != expected_time_scale_units:
        raise ValueError("time_scale_units_per_second must match time scale")
    expected_time_step_units = _ceil_positive_ratio_units(
        obligation.time_step_s,
        obligation.fixed_point_time_scale_s,
        name="time_step_units",
    )
    if obligation.time_step_units != expected_time_step_units:
        raise ValueError("time_step_units must match time step")
    expected_horizon_time_units = obligation.horizon_steps * obligation.time_step_units
    if obligation.horizon_time_units != expected_horizon_time_units:
        raise ValueError("horizon_time_units must match horizon and time step")

    expected_relative_velocity_units = _ceil_div_units(
        obligation.relative_velocity_rate_bound_units_per_second
        * obligation.time_step_units,
        obligation.time_scale_units_per_second,
        name="relative_velocity_step_bound_units",
    )
    if (
        obligation.relative_velocity_step_bound_units
        != expected_relative_velocity_units
    ):
        raise ValueError(
            "relative_velocity_step_bound_units must match sampled rate bound",
        )
    expected_residual_units = _ceil_div_units(
        obligation.coupling_residual_rate_bound_units_per_second
        * obligation.time_step_units,
        obligation.time_scale_units_per_second,
        name="coupling_residual_step_bound_units",
    )
    if obligation.coupling_residual_step_bound_units != expected_residual_units:
        raise ValueError(
            "coupling_residual_step_bound_units must match sampled rate bound",
        )

    expected_continuous_drive_rate = (
        obligation.relative_velocity_rate_bound_units_per_second
        + obligation.coupling_residual_rate_bound_units_per_second
    )
    if (
        obligation.continuous_drive_rate_bound_units_per_second
        != expected_continuous_drive_rate
    ):
        raise ValueError(
            "continuous_drive_rate_bound_units_per_second must equal rate sum",
        )
    expected_continuous_horizon_drive = _ceil_div_units(
        expected_continuous_drive_rate * obligation.horizon_time_units,
        obligation.time_scale_units_per_second,
        name="continuous_horizon_drive_bound_units",
    )
    if (
        obligation.continuous_horizon_drive_bound_units
        != expected_continuous_horizon_drive
    ):
        raise ValueError(
            "continuous_horizon_drive_bound_units must match sampled horizon rate",
        )
    expected_continuous_linear_budget = (
        obligation.initial_tolerance_units + expected_continuous_horizon_drive
    )
    if obligation.continuous_linear_budget_units != (expected_continuous_linear_budget):
        raise ValueError(
            "continuous_linear_budget_units must match continuous envelope",
        )
    expected_continuous_margin = (
        obligation.merge_window_tolerance_units - expected_continuous_linear_budget
    )
    if obligation.continuous_margin_units != expected_continuous_margin:
        raise ValueError("continuous_margin_units must match continuous envelope")
    expected_continuous_discharged = expected_continuous_margin >= 0
    if obligation.continuous_envelope_discharged != expected_continuous_discharged:
        raise ValueError(
            "continuous_envelope_discharged does not match certificate math",
        )

    expected_drive = (
        obligation.relative_velocity_step_bound_units
        + obligation.coupling_residual_step_bound_units
    )
    if obligation.drive_bound_units != expected_drive:
        raise ValueError("drive_bound_units must equal relative velocity plus residual")
    if obligation.kinematic_residual_units > (
        obligation.coupling_residual_step_bound_units
    ):
        raise ValueError("kinematic_residual_units must fit sampled residual bound")
    if obligation.configured_coupling_residual_step_bound_units > (
        obligation.coupling_residual_step_bound_units
    ):
        raise ValueError(
            "configured_coupling_residual_step_bound_units must fit residual bound",
        )
    if obligation.max_spatial_dispersion_units != obligation.initial_tolerance_units:
        raise ValueError("max_spatial_dispersion_units must mirror initial tolerance")
    expected_budget = (
        obligation.initial_tolerance_units
        + obligation.horizon_steps * obligation.drive_bound_units
    )
    if obligation.linear_budget_units != expected_budget:
        raise ValueError("linear_budget_units must match the Lean linear budget")
    expected_gronwall_trace = _gronwall_budget_trace(
        initial_tolerance_units=obligation.initial_tolerance_units,
        lipschitz_step_gain_units=obligation.lipschitz_step_gain_units,
        drive_bound_units=obligation.drive_bound_units,
        horizon_steps=obligation.horizon_steps,
    )
    expected_gronwall_budget = expected_gronwall_trace[-1]
    if obligation.gronwall_budget_units != expected_gronwall_budget:
        raise ValueError("gronwall_budget_units must match the Lean Gronwall budget")
    expected_gronwall_margin = (
        obligation.merge_window_tolerance_units - expected_gronwall_budget
    )
    if obligation.gronwall_budget_margin_units != expected_gronwall_margin:
        raise ValueError("gronwall_budget_margin_units must match the Lean margin")
    expected_trace_hash = _gronwall_budget_trace_sha256(
        trace_units=expected_gronwall_trace,
        horizon_steps=obligation.horizon_steps,
    )
    if obligation.gronwall_budget_trace_sha256 != expected_trace_hash:
        raise ValueError("gronwall_budget_trace_sha256 does not replay")
    expected_margin = expected_gronwall_margin
    if obligation.window_budget_margin_units != expected_margin:
        raise ValueError("window_budget_margin_units must match the Lean margin")
    expected_phase_budget = (
        obligation.max_phase_dispersion_units
        + obligation.configured_phase_drift_bound_units
    )
    if obligation.phase_budget_units != expected_phase_budget:
        raise ValueError(
            "phase_budget_units must match dispersion plus configured drift",
        )
    expected_phase_margin = obligation.phase_tolerance_units - expected_phase_budget
    if obligation.phase_margin_units != expected_phase_margin:
        raise ValueError(
            "phase_margin_units must match phase tolerance minus phase budget",
        )
    expected_phase_budget_discharged = expected_phase_margin >= 0
    if obligation.phase_budget_discharged != expected_phase_budget_discharged:
        raise ValueError(
            "phase_budget_discharged does not match phase certificate math",
        )
    expected_acceptance_replay_discharged = (
        obligation.acceptance_kinematic_equations_validated
        and obligation.acceptance_kinematic_summary_replay_tolerance_units
        <= obligation.acceptance_kinematic_summary_replay_tolerance_limit_units
    )
    if (
        obligation.acceptance_replay_certificate_discharged
        != expected_acceptance_replay_discharged
    ):
        raise ValueError(
            "acceptance_replay_certificate_discharged does not match Lean replay",
        )
    expected_acceptance_certificate_discharged = (
        expected_margin >= 0
        and expected_phase_budget_discharged
        and expected_acceptance_replay_discharged
    )
    if (
        obligation.acceptance_certificate_discharged
        != expected_acceptance_certificate_discharged
    ):
        raise ValueError(
            "acceptance_certificate_discharged does not match Lean certificate",
        )
    expected_discharged = (
        expected_margin >= 0
        and expected_phase_budget_discharged
        and expected_acceptance_certificate_discharged
        and expected_continuous_discharged
        and obligation.execution_disabled
        and not obligation.actuating
    )
    if obligation.proof_obligations_discharged != expected_discharged:
        raise ValueError("proof_obligations_discharged does not match certificate math")
    expected_hash = _sha256_json(_dict_without_record_hash(obligation))
    if obligation.record_sha256 != expected_hash:
        raise ValueError("record_sha256 does not match canonical obligation payload")
    return obligation
