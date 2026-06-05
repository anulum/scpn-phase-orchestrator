# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — PHA-C acceptance accelerator validation

"""Shared validation for PHA-C acceptance accelerator contracts."""

from __future__ import annotations

from scpn_phase_orchestrator.upde.pha_c_acceptance import (
    PHACAcceptanceRecord,
    build_pha_c_acceptance_record,
    verify_pha_c_acceptance_record,
)

_NUMERIC_FIELDS = (
    "start_time",
    "end_time",
    "dt",
    "first_lock_time",
    "max_abs_doppler_term",
    "max_abs_spatial_coupling",
    "max_phase_dispersion_rad",
    "max_spatial_dispersion_m",
    "kinematic_residual_max_m",
    "max_abs_velocity_m_per_s",
    "path_length_max_m",
    "kinematic_summary_replay_tolerance",
    "min_phase_margin_rad",
    "min_spatial_margin_m",
    "min_phase_order_parameter",
    "max_distance_to_reference_m",
    "reference_phase",
    "reference_point",
    "phase_tol_rad",
    "spatial_tol_m",
    "tolerance_profile_multiplier",
)
_DISCRETE_FIELDS = (
    "sample_count",
    "step_count",
    "oscillator_count",
    "first_lock_index",
    "first_lock_observed",
    "final_lock_achieved",
    "lock_sample_count",
    "lock_loss_count",
    "reset_count",
    "max_consecutive_lock_samples",
    "final_position_equation_validated",
    "max_abs_velocity_equation_validated",
    "path_length_equation_validated",
    "kinematic_equations_validated",
    "tolerance_profile_name",
    "required_consecutive_samples",
    "moving_frame_backend_request",
    "claim_boundary",
    "evidence_kind",
    "execution_disabled",
    "actuating",
    "omega_schedule_sha256",
    "velocity_schedule_sha256",
    "phase_trajectory_sha256",
    "position_trajectory_sha256",
    "initial_spatial_coupling_sha256",
    "final_spatial_coupling_sha256",
    "doppler_trace_sha256",
    "timeline_sha256",
    "acceptance_sha256",
)


def expected_pha_c_acceptance_record(
    *args: object,
    **kwargs: object,
) -> PHACAcceptanceRecord:
    """Return the Python reference acceptance after fail-closed validation."""

    return build_pha_c_acceptance_record(*args, **kwargs)


def validate_pha_c_acceptance_record(
    got: PHACAcceptanceRecord,
    expected: PHACAcceptanceRecord,
    *,
    tolerance: float = 1.0e-12,
) -> PHACAcceptanceRecord:
    """Validate an accelerator acceptance record against the reference."""

    verify_pha_c_acceptance_record(got)
    verify_pha_c_acceptance_record(expected)
    got_dict = got.to_dict()
    expected_dict = expected.to_dict()
    for field in _NUMERIC_FIELDS:
        error = abs(float(got_dict[field]) - float(expected_dict[field]))
        if error > tolerance:
            raise ValueError(f"PHA-C acceptance field {field!r} diverged by {error}")
    for field in _DISCRETE_FIELDS:
        if got_dict[field] != expected_dict[field]:
            raise ValueError(f"PHA-C acceptance field {field!r} diverged")
    return got
