# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — PHA-C acceptance accelerator validation

"""Shared validation for PHA-C acceptance accelerator contracts."""

from __future__ import annotations

from math import isfinite
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike

from scpn_phase_orchestrator.coupling.spatial_modulator import (
    SpatialCouplingModulator,
)
from scpn_phase_orchestrator.monitor.merge_window import (
    DEFAULT_PHASE_TOL_RAD,
    DEFAULT_SPATIAL_TOL_M,
)
from scpn_phase_orchestrator.upde.pha_c_acceptance import (
    PHACAcceptanceRecord,
    build_pha_c_acceptance_record,
    verify_pha_c_acceptance_record,
)

from ._validation_common import validate_non_negative_tolerance

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
_INT_FIELDS = (
    "sample_count",
    "step_count",
    "oscillator_count",
    "first_lock_index",
    "lock_sample_count",
    "lock_loss_count",
    "reset_count",
    "max_consecutive_lock_samples",
    "required_consecutive_samples",
)
_BOOL_FIELDS = (
    "first_lock_observed",
    "final_lock_achieved",
    "final_position_equation_validated",
    "max_abs_velocity_equation_validated",
    "path_length_equation_validated",
    "kinematic_equations_validated",
    "execution_disabled",
    "actuating",
)
_STRING_FIELDS = (
    "tolerance_profile_name",
    "moving_frame_backend_request",
    "claim_boundary",
    "evidence_kind",
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
    phases_t0: ArrayLike,
    positions_t0: ArrayLike,
    omega_schedule: ArrayLike,
    knm: ArrayLike,
    velocity_schedule: ArrayLike,
    *,
    alpha: object = 0.0,
    spatial_modulator: SpatialCouplingModulator | None = None,
    doppler_strength: object = 1.0e-3,
    doppler_epsilon: object = 1.0e-9,
    zeta: object = 0.0,
    psi: object = 0.0,
    dt: object = 1.0e-3,
    method: str = "rk4",
    n_substeps: object = 1,
    atol: object = 1.0e-9,
    rtol: object = 1.0e-9,
    reference_phase: object = 0.0,
    reference_point: object = 0.0,
    phase_tol_rad: object = DEFAULT_PHASE_TOL_RAD,
    spatial_tol_m: object = DEFAULT_SPATIAL_TOL_M,
    required_consecutive_samples: object = 3,
    tolerance_profile: object | None = "baseline_1x",
    backend: object = "python",
) -> PHACAcceptanceRecord:
    """Return the Python reference acceptance after fail-closed validation."""
    return build_pha_c_acceptance_record(
        phases_t0,
        positions_t0,
        omega_schedule,
        knm,
        velocity_schedule,
        alpha=alpha,
        spatial_modulator=spatial_modulator,
        doppler_strength=doppler_strength,
        doppler_epsilon=doppler_epsilon,
        zeta=zeta,
        psi=psi,
        dt=dt,
        method=method,
        n_substeps=n_substeps,
        atol=atol,
        rtol=rtol,
        reference_phase=reference_phase,
        reference_point=reference_point,
        phase_tol_rad=phase_tol_rad,
        spatial_tol_m=spatial_tol_m,
        required_consecutive_samples=required_consecutive_samples,
        tolerance_profile=tolerance_profile,
        backend=backend,
    )


def _validate_real_record_field(record: PHACAcceptanceRecord, field: str) -> float:
    """Return a record field as a finite non-boolean real scalar."""
    value = getattr(record, field)
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (Real, np.floating, np.integer),
    ):
        raise ValueError(
            f"PHA-C acceptance field {field!r} must be a finite real scalar"
        )
    parsed = float(value)
    if not isfinite(parsed):
        raise ValueError(f"PHA-C acceptance field {field!r} must be finite")
    return parsed


def _validate_int_record_field(record: PHACAcceptanceRecord, field: str) -> int:
    """Return a record field as an integer after rejecting boolean aliases."""
    value = getattr(record, field)
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (Integral, np.integer),
    ):
        raise ValueError(f"PHA-C acceptance field {field!r} must be an integer")
    return int(value)


def _validate_bool_record_field(record: PHACAcceptanceRecord, field: str) -> bool:
    """Return a record field after confirming it is a plain ``bool``."""
    value = getattr(record, field)
    if type(value) is not bool:
        raise ValueError(f"PHA-C acceptance field {field!r} must be bool")
    return value


def _validate_string_record_field(record: PHACAcceptanceRecord, field: str) -> str:
    """Return a record field after confirming it is a string."""
    value = getattr(record, field)
    if not isinstance(value, str):
        raise ValueError(f"PHA-C acceptance field {field!r} must be a string")
    return value


def pha_c_acceptance_record_max_abs_error(
    got: PHACAcceptanceRecord,
    expected: PHACAcceptanceRecord,
) -> float:
    """Return strict maximum field error for PHA-C acceptance parity."""
    numeric_error = max(
        abs(
            _validate_real_record_field(got, field)
            - _validate_real_record_field(expected, field)
        )
        for field in _NUMERIC_FIELDS
    )
    int_error = max(
        int(
            _validate_int_record_field(got, field)
            != _validate_int_record_field(expected, field)
        )
        for field in _INT_FIELDS
    )
    bool_error = max(
        int(
            _validate_bool_record_field(got, field)
            is not _validate_bool_record_field(expected, field)
        )
        for field in _BOOL_FIELDS
    )
    string_error = max(
        int(
            _validate_string_record_field(got, field)
            != _validate_string_record_field(expected, field)
        )
        for field in _STRING_FIELDS
    )
    verify_pha_c_acceptance_record(got)
    verify_pha_c_acceptance_record(expected)
    return max(numeric_error, float(int_error), float(bool_error), float(string_error))


def validate_pha_c_acceptance_record(
    got: PHACAcceptanceRecord,
    expected: PHACAcceptanceRecord,
    *,
    tolerance: object = 1.0e-12,
) -> PHACAcceptanceRecord:
    """Validate an accelerator acceptance record against the reference."""
    tolerance_f = validate_non_negative_tolerance(tolerance)
    for field in _NUMERIC_FIELDS:
        error = abs(
            _validate_real_record_field(got, field)
            - _validate_real_record_field(expected, field)
        )
        if error > tolerance_f:
            raise ValueError(f"PHA-C acceptance field {field!r} diverged by {error}")
    for field in _INT_FIELDS:
        if _validate_int_record_field(got, field) != _validate_int_record_field(
            expected,
            field,
        ):
            raise ValueError(f"PHA-C acceptance field {field!r} diverged")
    for field in _BOOL_FIELDS:
        if _validate_bool_record_field(got, field) is not _validate_bool_record_field(
            expected,
            field,
        ):
            raise ValueError(f"PHA-C acceptance field {field!r} diverged")
    for field in _STRING_FIELDS:
        if _validate_string_record_field(got, field) != _validate_string_record_field(
            expected,
            field,
        ):
            raise ValueError(f"PHA-C acceptance field {field!r} diverged")
    verify_pha_c_acceptance_record(got)
    verify_pha_c_acceptance_record(expected)
    return got
