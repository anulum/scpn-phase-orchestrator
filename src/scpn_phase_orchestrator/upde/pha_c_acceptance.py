# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — PHA-C acceptance chain

"""End-to-end PHA-C acceptance evidence.

The PHA-C work lane has several reviewed modules: spatial coupling,
time-varying frequencies, Doppler correction, moving-frame integration,
merge-window monitoring, handoff records, and event timelines. This module
binds those surfaces into one deterministic, non-actuating acceptance record so
review lanes can prove the complete chain was exercised rather than only one
slice.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import isfinite
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from scpn_phase_orchestrator.coupling.spatial_modulator import (
    SpatialCouplingModulator,
)
from scpn_phase_orchestrator.monitor.merge_window import (
    DEFAULT_PHASE_TOL_RAD,
    DEFAULT_SPATIAL_TOL_M,
)
from scpn_phase_orchestrator.upde.doppler import doppler_term
from scpn_phase_orchestrator.upde.moving_frame import moving_frame_run
from scpn_phase_orchestrator.upde.pha_c_timeline import build_pha_c_event_timeline

FloatArray: TypeAlias = NDArray[np.float64]

PHA_C_ACCEPTANCE_CLAIM_BOUNDARY = "pha_c_end_to_end_acceptance_review_only"
PHA_C_ACCEPTANCE_EVIDENCE_KIND = "deterministic_non_actuating_acceptance"
PHA_C_KINEMATIC_RESIDUAL_TOLERANCE_M = 1.0e-9

__all__ = [
    "PHA_C_ACCEPTANCE_CLAIM_BOUNDARY",
    "PHA_C_ACCEPTANCE_EVIDENCE_KIND",
    "PHA_C_KINEMATIC_RESIDUAL_TOLERANCE_M",
    "PHACAcceptanceRecord",
    "build_pha_c_acceptance_record",
    "pha_c_acceptance_record_to_dict",
    "verify_pha_c_acceptance_record",
]


@dataclass(frozen=True, slots=True)
class PHACAcceptanceRecord:
    """Audit-ready acceptance evidence for the complete PHA-C chain."""

    sample_count: int
    step_count: int
    oscillator_count: int
    start_time: float
    end_time: float
    dt: float
    first_lock_index: int
    first_lock_time: float
    first_lock_observed: bool
    final_lock_achieved: bool
    lock_sample_count: int
    lock_loss_count: int
    reset_count: int
    max_consecutive_lock_samples: int
    max_abs_doppler_term: float
    max_abs_spatial_coupling: float
    max_phase_dispersion_rad: float
    max_spatial_dispersion_m: float
    kinematic_residual_max_m: float
    max_abs_velocity_m_per_s: float
    path_length_max_m: float
    min_phase_margin_rad: float
    min_spatial_margin_m: float
    min_phase_order_parameter: float
    max_distance_to_reference_m: float
    reference_phase: float
    reference_point: float
    phase_tol_rad: float
    spatial_tol_m: float
    tolerance_profile_name: str
    tolerance_profile_multiplier: float
    required_consecutive_samples: int
    moving_frame_backend_request: str
    claim_boundary: str
    evidence_kind: str
    execution_disabled: bool
    actuating: bool
    omega_schedule_sha256: str
    velocity_schedule_sha256: str
    phase_trajectory_sha256: str
    position_trajectory_sha256: str
    initial_spatial_coupling_sha256: str
    final_spatial_coupling_sha256: str
    doppler_trace_sha256: str
    timeline_sha256: str
    acceptance_sha256: str

    def to_dict(self) -> dict[str, float | int | bool | str]:
        """Return a JSON-safe canonical representation."""

        return pha_c_acceptance_record_to_dict(self)


def _validate_real_scalar(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _validate_positive_scalar(value: object, *, name: str) -> float:
    parsed = _validate_real_scalar(value, name=name)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _as_float_vector(values: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(values)
    if array.dtype == np.dtype("O"):
        raise ValueError(f"{name} must be a finite real-valued vector")
    if np.issubdtype(array.dtype, np.bool_) or np.issubdtype(
        array.dtype,
        np.complexfloating,
    ):
        raise ValueError(f"{name} must be real-valued, not boolean or complex")
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one oscillator")
    try:
        out = np.ascontiguousarray(array, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite values")
    return out


def _as_schedule(values: ArrayLike, *, name: str, n: int) -> FloatArray:
    array = np.asarray(values)
    if array.dtype == np.dtype("O"):
        raise ValueError(f"{name} must be a finite real-valued matrix")
    if np.issubdtype(array.dtype, np.bool_) or np.issubdtype(
        array.dtype,
        np.complexfloating,
    ):
        raise ValueError(f"{name} must be real-valued, not boolean or complex")
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if array.shape[0] < 1 or array.shape[1] != n:
        raise ValueError(f"{name} must have shape (steps, oscillator_count)")
    try:
        out = np.ascontiguousarray(array, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite values")
    return out


def _as_knm(values: ArrayLike, *, n: int) -> FloatArray:
    array = np.asarray(values)
    if array.dtype == np.dtype("O"):
        raise ValueError("knm must be a finite real square matrix")
    if np.issubdtype(array.dtype, np.bool_) or np.issubdtype(
        array.dtype,
        np.complexfloating,
    ):
        raise ValueError("knm must be real-valued, not boolean or complex")
    try:
        out = np.ascontiguousarray(array, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("knm must be numeric") from exc
    if out.shape != (n, n):
        raise ValueError("knm must have shape (oscillator_count, oscillator_count)")
    if not np.all(np.isfinite(out)):
        raise ValueError("knm must contain only finite values")
    if np.max(np.abs(np.diag(out))) > 1.0e-12:
        raise ValueError("knm diagonal must be zero")
    return out


def _validate_backend_request(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("backend must be a non-empty string")
    return value


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


_SHA256_HEX_DIGITS = frozenset("0123456789abcdef")


def _validate_sha256_hex(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in _SHA256_HEX_DIGITS for char in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _validate_record_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _validate_record_int(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return parsed


def _validate_nonnegative_record_scalar(value: object, *, name: str) -> float:
    parsed = _validate_real_scalar(value, name=name)
    if parsed < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _array_sha256(values: FloatArray) -> str:
    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    payload = {
        "dtype": "float64",
        "shape": list(contiguous.shape),
        "values_sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }
    return _sha256_json(payload)


def _record_dict_without_hash(
    *,
    sample_count: int,
    step_count: int,
    oscillator_count: int,
    start_time: float,
    end_time: float,
    dt: float,
    first_lock_index: int,
    first_lock_time: float,
    first_lock_observed: bool,
    final_lock_achieved: bool,
    lock_sample_count: int,
    lock_loss_count: int,
    reset_count: int,
    max_consecutive_lock_samples: int,
    max_abs_doppler_term: float,
    max_abs_spatial_coupling: float,
    max_phase_dispersion_rad: float,
    max_spatial_dispersion_m: float,
    kinematic_residual_max_m: float,
    max_abs_velocity_m_per_s: float,
    path_length_max_m: float,
    min_phase_margin_rad: float,
    min_spatial_margin_m: float,
    min_phase_order_parameter: float,
    max_distance_to_reference_m: float,
    reference_phase: float,
    reference_point: float,
    phase_tol_rad: float,
    spatial_tol_m: float,
    tolerance_profile_name: str,
    tolerance_profile_multiplier: float,
    required_consecutive_samples: int,
    moving_frame_backend_request: str,
    omega_schedule_sha256: str,
    velocity_schedule_sha256: str,
    phase_trajectory_sha256: str,
    position_trajectory_sha256: str,
    initial_spatial_coupling_sha256: str,
    final_spatial_coupling_sha256: str,
    doppler_trace_sha256: str,
    timeline_sha256: str,
) -> dict[str, float | int | bool | str]:
    return {
        "sample_count": int(sample_count),
        "step_count": int(step_count),
        "oscillator_count": int(oscillator_count),
        "start_time": float(start_time),
        "end_time": float(end_time),
        "dt": float(dt),
        "first_lock_index": int(first_lock_index),
        "first_lock_time": float(first_lock_time),
        "first_lock_observed": bool(first_lock_observed),
        "final_lock_achieved": bool(final_lock_achieved),
        "lock_sample_count": int(lock_sample_count),
        "lock_loss_count": int(lock_loss_count),
        "reset_count": int(reset_count),
        "max_consecutive_lock_samples": int(max_consecutive_lock_samples),
        "max_abs_doppler_term": float(max_abs_doppler_term),
        "max_abs_spatial_coupling": float(max_abs_spatial_coupling),
        "max_phase_dispersion_rad": float(max_phase_dispersion_rad),
        "max_spatial_dispersion_m": float(max_spatial_dispersion_m),
        "kinematic_residual_max_m": float(kinematic_residual_max_m),
        "max_abs_velocity_m_per_s": float(max_abs_velocity_m_per_s),
        "path_length_max_m": float(path_length_max_m),
        "min_phase_margin_rad": float(min_phase_margin_rad),
        "min_spatial_margin_m": float(min_spatial_margin_m),
        "min_phase_order_parameter": float(min_phase_order_parameter),
        "max_distance_to_reference_m": float(max_distance_to_reference_m),
        "reference_phase": float(reference_phase),
        "reference_point": float(reference_point),
        "phase_tol_rad": float(phase_tol_rad),
        "spatial_tol_m": float(spatial_tol_m),
        "tolerance_profile_name": str(tolerance_profile_name),
        "tolerance_profile_multiplier": float(tolerance_profile_multiplier),
        "required_consecutive_samples": int(required_consecutive_samples),
        "moving_frame_backend_request": str(moving_frame_backend_request),
        "claim_boundary": PHA_C_ACCEPTANCE_CLAIM_BOUNDARY,
        "evidence_kind": PHA_C_ACCEPTANCE_EVIDENCE_KIND,
        "execution_disabled": True,
        "actuating": False,
        "omega_schedule_sha256": omega_schedule_sha256,
        "velocity_schedule_sha256": velocity_schedule_sha256,
        "phase_trajectory_sha256": phase_trajectory_sha256,
        "position_trajectory_sha256": position_trajectory_sha256,
        "initial_spatial_coupling_sha256": initial_spatial_coupling_sha256,
        "final_spatial_coupling_sha256": final_spatial_coupling_sha256,
        "doppler_trace_sha256": doppler_trace_sha256,
        "timeline_sha256": timeline_sha256,
    }


def build_pha_c_acceptance_record(
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
    """Build deterministic end-to-end PHA-C acceptance evidence.

    The builder advances a moving-frame trajectory one schedule row at a time,
    recording spatially modulated coupling and Doppler traces before converting
    the trajectory into a PHA-C event timeline. The output remains review-only;
    it is evidence for downstream gates and never permits actuation.
    """

    phases = _as_float_vector(phases_t0, name="phases_t0")
    n = int(phases.size)
    positions = _as_float_vector(positions_t0, name="positions_t0")
    if positions.shape != phases.shape:
        raise ValueError("positions_t0 must have the same shape as phases_t0")
    omega = _as_schedule(omega_schedule, name="omega_schedule", n=n)
    velocities = _as_schedule(velocity_schedule, name="velocity_schedule", n=n)
    if velocities.shape[0] != omega.shape[0]:
        raise ValueError("velocity_schedule step count must match omega_schedule")
    k = _as_knm(knm, n=n)
    dt_s = _validate_positive_scalar(dt, name="dt")
    strength = _validate_real_scalar(doppler_strength, name="doppler_strength")
    doppler_eps = _validate_positive_scalar(doppler_epsilon, name="doppler_epsilon")
    zeta_f = _validate_real_scalar(zeta, name="zeta")
    psi_f = _validate_real_scalar(psi, name="psi")
    atol_f = _validate_positive_scalar(atol, name="atol")
    rtol_f = _validate_positive_scalar(rtol, name="rtol")
    backend_request = _validate_backend_request(backend)
    if isinstance(n_substeps, (bool, np.bool_)) or not isinstance(
        n_substeps,
        (int, np.integer),
    ):
        raise ValueError("n_substeps must be a positive integer")
    n_substeps_i = int(n_substeps)
    if n_substeps_i < 1:
        raise ValueError("n_substeps must be a positive integer")
    if spatial_modulator is None:
        modulator = SpatialCouplingModulator(K_base=1.0)
    elif isinstance(spatial_modulator, SpatialCouplingModulator):
        modulator = spatial_modulator
    else:
        raise ValueError("spatial_modulator must be a SpatialCouplingModulator")

    phase_rows: list[FloatArray] = [phases.copy()]
    position_rows: list[FloatArray] = [positions.copy()]
    spatial_rows: list[FloatArray] = []
    doppler_rows: list[FloatArray] = []
    current_phases = phases.copy()
    current_positions = positions.copy()
    for step in range(int(omega.shape[0])):
        spatial_coupling = modulator.modulate(k, current_positions)
        spatial_rows.append(spatial_coupling)
        correction = doppler_term(
            velocities[step],
            spatial_coupling,
            doppler_strength=strength,
            doppler_epsilon=doppler_eps,
        )
        doppler_rows.append(correction)
        state = moving_frame_run(
            current_phases,
            current_positions,
            omega[step : step + 1],
            k,
            alpha,
            velocities[step : step + 1],
            modulator,
            doppler_strength=strength,
            doppler_epsilon=doppler_eps,
            zeta=zeta_f,
            psi=psi_f,
            dt=dt_s,
            method=method,
            n_substeps=n_substeps_i,
            atol=atol_f,
            rtol=rtol_f,
            backend=backend_request,
        )
        current_phases = np.ascontiguousarray(state[:n], dtype=np.float64)
        current_positions = np.ascontiguousarray(state[n:], dtype=np.float64)
        phase_rows.append(current_phases)
        position_rows.append(current_positions)
    final_spatial_coupling = modulator.modulate(k, current_positions)

    phase_trajectory = np.vstack(phase_rows).astype(np.float64, copy=False)
    position_trajectory = np.vstack(position_rows).astype(np.float64, copy=False)
    doppler_trace = np.vstack(doppler_rows).astype(np.float64, copy=False)
    spatial_trace = np.stack(spatial_rows).astype(np.float64, copy=False)
    predicted_position_steps = position_trajectory[:-1] + velocities * dt_s
    kinematic_residual_max_m = float(
        np.max(np.abs(position_trajectory[1:] - predicted_position_steps))
    )
    max_abs_velocity_m_per_s = float(np.max(np.abs(velocities)))
    path_length_max_m = float(np.max(np.sum(np.abs(velocities * dt_s), axis=0)))
    times = np.arange(phase_trajectory.shape[0], dtype=np.float64) * dt_s
    timeline = build_pha_c_event_timeline(
        phase_trajectory,
        position_trajectory,
        times=times,
        reference_phase=reference_phase,
        reference_point=reference_point,
        phase_tol_rad=phase_tol_rad,
        spatial_tol_m=spatial_tol_m,
        required_consecutive_samples=required_consecutive_samples,
        tolerance_profile=tolerance_profile,
    )
    first_spatial_coupling = spatial_rows[0]
    payload = _record_dict_without_hash(
        sample_count=timeline.sample_count,
        step_count=int(omega.shape[0]),
        oscillator_count=n,
        start_time=float(times[0]),
        end_time=float(times[-1]),
        dt=dt_s,
        first_lock_index=timeline.first_lock_index,
        first_lock_time=timeline.first_lock_time,
        first_lock_observed=timeline.first_lock_observed,
        final_lock_achieved=timeline.final_lock_achieved,
        lock_sample_count=timeline.lock_sample_count,
        lock_loss_count=timeline.lock_loss_count,
        reset_count=timeline.reset_count,
        max_consecutive_lock_samples=timeline.max_consecutive_lock_samples,
        max_abs_doppler_term=float(np.max(np.abs(doppler_trace))),
        max_abs_spatial_coupling=float(np.max(np.abs(spatial_trace))),
        max_phase_dispersion_rad=timeline.max_phase_dispersion_rad,
        max_spatial_dispersion_m=timeline.max_spatial_dispersion_m,
        kinematic_residual_max_m=kinematic_residual_max_m,
        max_abs_velocity_m_per_s=max_abs_velocity_m_per_s,
        path_length_max_m=path_length_max_m,
        min_phase_margin_rad=timeline.min_phase_margin_rad,
        min_spatial_margin_m=timeline.min_spatial_margin_m,
        min_phase_order_parameter=timeline.min_phase_order_parameter,
        max_distance_to_reference_m=timeline.max_distance_to_reference_m,
        reference_phase=timeline.reference_phase,
        reference_point=timeline.reference_point,
        phase_tol_rad=timeline.phase_tol_rad,
        spatial_tol_m=timeline.spatial_tol_m,
        tolerance_profile_name=timeline.tolerance_profile_name,
        tolerance_profile_multiplier=timeline.tolerance_profile_multiplier,
        required_consecutive_samples=timeline.required_consecutive_samples,
        moving_frame_backend_request=backend_request,
        omega_schedule_sha256=_array_sha256(omega),
        velocity_schedule_sha256=_array_sha256(velocities),
        phase_trajectory_sha256=_array_sha256(phase_trajectory),
        position_trajectory_sha256=_array_sha256(position_trajectory),
        initial_spatial_coupling_sha256=_array_sha256(first_spatial_coupling),
        final_spatial_coupling_sha256=_array_sha256(final_spatial_coupling),
        doppler_trace_sha256=_array_sha256(doppler_trace),
        timeline_sha256=timeline.timeline_sha256,
    )
    return PHACAcceptanceRecord(
        **payload,
        acceptance_sha256=_sha256_json(payload),
    )


def pha_c_acceptance_record_to_dict(
    record: PHACAcceptanceRecord,
) -> dict[str, float | int | bool | str]:
    """Return the canonical JSON-safe PHA-C acceptance payload."""

    payload = _record_dict_without_hash(
        sample_count=record.sample_count,
        step_count=record.step_count,
        oscillator_count=record.oscillator_count,
        start_time=record.start_time,
        end_time=record.end_time,
        dt=record.dt,
        first_lock_index=record.first_lock_index,
        first_lock_time=record.first_lock_time,
        first_lock_observed=record.first_lock_observed,
        final_lock_achieved=record.final_lock_achieved,
        lock_sample_count=record.lock_sample_count,
        lock_loss_count=record.lock_loss_count,
        reset_count=record.reset_count,
        max_consecutive_lock_samples=record.max_consecutive_lock_samples,
        max_abs_doppler_term=record.max_abs_doppler_term,
        max_abs_spatial_coupling=record.max_abs_spatial_coupling,
        max_phase_dispersion_rad=record.max_phase_dispersion_rad,
        max_spatial_dispersion_m=record.max_spatial_dispersion_m,
        kinematic_residual_max_m=record.kinematic_residual_max_m,
        max_abs_velocity_m_per_s=record.max_abs_velocity_m_per_s,
        path_length_max_m=record.path_length_max_m,
        min_phase_margin_rad=record.min_phase_margin_rad,
        min_spatial_margin_m=record.min_spatial_margin_m,
        min_phase_order_parameter=record.min_phase_order_parameter,
        max_distance_to_reference_m=record.max_distance_to_reference_m,
        reference_phase=record.reference_phase,
        reference_point=record.reference_point,
        phase_tol_rad=record.phase_tol_rad,
        spatial_tol_m=record.spatial_tol_m,
        tolerance_profile_name=record.tolerance_profile_name,
        tolerance_profile_multiplier=record.tolerance_profile_multiplier,
        required_consecutive_samples=record.required_consecutive_samples,
        moving_frame_backend_request=record.moving_frame_backend_request,
        omega_schedule_sha256=record.omega_schedule_sha256,
        velocity_schedule_sha256=record.velocity_schedule_sha256,
        phase_trajectory_sha256=record.phase_trajectory_sha256,
        position_trajectory_sha256=record.position_trajectory_sha256,
        initial_spatial_coupling_sha256=record.initial_spatial_coupling_sha256,
        final_spatial_coupling_sha256=record.final_spatial_coupling_sha256,
        doppler_trace_sha256=record.doppler_trace_sha256,
        timeline_sha256=record.timeline_sha256,
    )
    payload["acceptance_sha256"] = record.acceptance_sha256
    return payload


def verify_pha_c_acceptance_record(
    record: PHACAcceptanceRecord,
) -> PHACAcceptanceRecord:
    """Replay and validate a complete PHA-C acceptance record."""

    if not isinstance(record, PHACAcceptanceRecord):
        raise ValueError("record must be a PHACAcceptanceRecord")
    for field in (
        "omega_schedule_sha256",
        "velocity_schedule_sha256",
        "phase_trajectory_sha256",
        "position_trajectory_sha256",
        "initial_spatial_coupling_sha256",
        "final_spatial_coupling_sha256",
        "doppler_trace_sha256",
        "timeline_sha256",
    ):
        _validate_sha256_hex(getattr(record, field), name=field)
    acceptance_hash = _validate_sha256_hex(
        record.acceptance_sha256,
        name="acceptance_sha256",
    )
    if record.claim_boundary != PHA_C_ACCEPTANCE_CLAIM_BOUNDARY:
        raise ValueError("claim_boundary must be the PHA-C acceptance review boundary")
    if record.evidence_kind != PHA_C_ACCEPTANCE_EVIDENCE_KIND:
        raise ValueError("evidence_kind must be deterministic acceptance evidence")
    if (
        _validate_record_bool(
            record.execution_disabled,
            name="execution_disabled",
        )
        is not True
    ):
        raise ValueError("execution_disabled must be true")
    if _validate_record_bool(record.actuating, name="actuating") is not False:
        raise ValueError("actuating must be false")
    if (
        not isinstance(record.moving_frame_backend_request, str)
        or not record.moving_frame_backend_request
    ):
        raise ValueError("moving_frame_backend_request must be a non-empty string")

    sample_count = _validate_record_int(
        record.sample_count,
        name="sample_count",
        minimum=2,
    )
    step_count = _validate_record_int(record.step_count, name="step_count", minimum=1)
    if sample_count != step_count + 1:
        raise ValueError("sample_count must equal step_count + 1")
    _validate_record_int(record.oscillator_count, name="oscillator_count", minimum=1)
    required = _validate_record_int(
        record.required_consecutive_samples,
        name="required_consecutive_samples",
        minimum=1,
    )
    count_fields = (
        "lock_sample_count",
        "lock_loss_count",
        "reset_count",
        "max_consecutive_lock_samples",
    )
    counts = {
        field: _validate_record_int(getattr(record, field), name=field, minimum=0)
        for field in count_fields
    }
    if counts["lock_sample_count"] > sample_count:
        raise ValueError("lock_sample_count cannot exceed sample_count")
    if counts["max_consecutive_lock_samples"] > sample_count:
        raise ValueError("max_consecutive_lock_samples cannot exceed sample_count")
    for field in ("lock_loss_count", "reset_count"):
        if counts[field] > step_count:
            raise ValueError(f"{field} cannot exceed step_count")
    final_lock_achieved = _validate_record_bool(
        record.final_lock_achieved,
        name="final_lock_achieved",
    )
    if counts["max_consecutive_lock_samples"] < required and final_lock_achieved:
        raise ValueError("final_lock_achieved requires the consecutive threshold")

    start_time = _validate_real_scalar(record.start_time, name="start_time")
    end_time = _validate_real_scalar(record.end_time, name="end_time")
    dt = _validate_positive_scalar(record.dt, name="dt")
    if end_time < start_time:
        raise ValueError("end_time must be greater than or equal to start_time")
    if abs(end_time - (start_time + dt * step_count)) > 1.0e-12:
        raise ValueError("end_time must equal start_time + dt * step_count")
    first_lock_index = _validate_record_int(
        record.first_lock_index,
        name="first_lock_index",
        minimum=-1,
    )
    first_lock_observed = _validate_record_bool(
        record.first_lock_observed,
        name="first_lock_observed",
    )
    first_lock_time = _validate_real_scalar(
        record.first_lock_time,
        name="first_lock_time",
    )
    if first_lock_observed:
        if first_lock_index < 0 or first_lock_index >= sample_count:
            raise ValueError("first_lock_index must refer to an observed sample")
        if first_lock_time < start_time or first_lock_time > end_time:
            raise ValueError("first_lock_time must be inside the acceptance range")
    else:
        if first_lock_index != -1:
            raise ValueError("first_lock_index must be -1 when no lock is observed")
        if first_lock_time != 0.0:
            raise ValueError("first_lock_time must be 0.0 when no lock is observed")

    for field in (
        "max_abs_doppler_term",
        "max_abs_spatial_coupling",
        "max_phase_dispersion_rad",
        "max_spatial_dispersion_m",
        "kinematic_residual_max_m",
        "max_abs_velocity_m_per_s",
        "path_length_max_m",
        "max_distance_to_reference_m",
        "phase_tol_rad",
        "spatial_tol_m",
    ):
        _validate_nonnegative_record_scalar(getattr(record, field), name=field)
    max_phase_dispersion = _validate_nonnegative_record_scalar(
        record.max_phase_dispersion_rad,
        name="max_phase_dispersion_rad",
    )
    max_spatial_dispersion = _validate_nonnegative_record_scalar(
        record.max_spatial_dispersion_m,
        name="max_spatial_dispersion_m",
    )
    kinematic_residual = _validate_nonnegative_record_scalar(
        record.kinematic_residual_max_m,
        name="kinematic_residual_max_m",
    )
    if kinematic_residual > PHA_C_KINEMATIC_RESIDUAL_TOLERANCE_M:
        raise ValueError(
            "kinematic_residual_max_m must not exceed "
            f"{PHA_C_KINEMATIC_RESIDUAL_TOLERANCE_M} m"
        )
    phase_tol = _validate_nonnegative_record_scalar(
        record.phase_tol_rad,
        name="phase_tol_rad",
    )
    spatial_tol = _validate_nonnegative_record_scalar(
        record.spatial_tol_m,
        name="spatial_tol_m",
    )
    min_phase_margin = _validate_real_scalar(
        record.min_phase_margin_rad,
        name="min_phase_margin_rad",
    )
    min_spatial_margin = _validate_real_scalar(
        record.min_spatial_margin_m,
        name="min_spatial_margin_m",
    )
    if abs(min_phase_margin - (phase_tol - max_phase_dispersion)) > 1.0e-12:
        raise ValueError(
            "min_phase_margin_rad must equal phase_tol_rad - max_phase_dispersion_rad"
        )
    if abs(min_spatial_margin - (spatial_tol - max_spatial_dispersion)) > 1.0e-12:
        raise ValueError(
            "min_spatial_margin_m must equal spatial_tol_m - max_spatial_dispersion_m"
        )
    order_parameter = _validate_nonnegative_record_scalar(
        record.min_phase_order_parameter,
        name="min_phase_order_parameter",
    )
    if order_parameter > 1.0 + 1.0e-12:
        raise ValueError("min_phase_order_parameter must be inside [0, 1]")
    multiplier = _validate_real_scalar(
        record.tolerance_profile_multiplier,
        name="tolerance_profile_multiplier",
    )
    if multiplier <= 0.0:
        raise ValueError("tolerance_profile_multiplier must be positive")
    if (
        not isinstance(record.tolerance_profile_name, str)
        or not record.tolerance_profile_name
    ):
        raise ValueError("tolerance_profile_name must be a non-empty string")
    for field in ("reference_phase", "reference_point"):
        _validate_real_scalar(getattr(record, field), name=field)

    payload = pha_c_acceptance_record_to_dict(record)
    replay_payload = dict(payload)
    replay_payload.pop("acceptance_sha256")
    if _sha256_json(replay_payload) != acceptance_hash:
        raise ValueError(
            "acceptance_sha256 does not match the canonical acceptance payload",
        )
    return record
