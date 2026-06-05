# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — PHA-C event timeline

"""Deterministic PHA-C event timelines over moving-frame trajectories.

Single-sample PHA-C handoff records are useful review atoms. This module
builds the trajectory-level event timeline that downstream replay, MIF/FRC
review, and Studio panels need: first lock, lock loss, reset counts, tolerance
profile provenance, and stable hashes over every sample record. The timeline is
review-only and never enables actuation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import isfinite
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from scpn_phase_orchestrator.monitor.merge_window import (
    DEFAULT_PHASE_TOL_RAD,
    DEFAULT_SPATIAL_TOL_M,
)
from scpn_phase_orchestrator.upde.pha_c_handoff import (
    PHACHandoffRecord,
    build_pha_c_handoff_record,
)

FloatArray: TypeAlias = NDArray[np.float64]

PHA_C_TIMELINE_CLAIM_BOUNDARY = "pha_c_event_timeline_review_only"
PHA_C_TIMELINE_EVIDENCE_KIND = "deterministic_non_actuating_timeline"
PHA_C_TIMELINE_MARGIN_REPLAY_TOLERANCE = 1.0e-12

__all__ = [
    "PHA_C_TIMELINE_CLAIM_BOUNDARY",
    "PHA_C_TIMELINE_EVIDENCE_KIND",
    "PHA_C_TIMELINE_MARGIN_REPLAY_TOLERANCE",
    "PHACTimelineRecord",
    "build_pha_c_event_timeline",
    "pha_c_event_timeline_to_dict",
    "verify_pha_c_event_timeline",
]


@dataclass(frozen=True, slots=True)
class PHACTimelineRecord:
    """Audit-ready PHA-C trajectory event timeline.

    The record stores scalar trajectory evidence plus SHA-256 digests of the
    time vector, per-sample handoff records, transition table, and final
    timeline. It is designed for review and replay lanes, not for direct
    control output.
    """

    sample_count: int
    oscillator_count: int
    start_time: float
    end_time: float
    duration_s: float
    first_lock_index: int
    first_lock_time: float
    first_lock_observed: bool
    final_lock_achieved: bool
    lock_sample_count: int
    phase_lock_sample_count: int
    spatial_lock_sample_count: int
    lock_loss_count: int
    reset_count: int
    max_consecutive_lock_samples: int
    max_phase_dispersion_rad: float
    max_spatial_dispersion_m: float
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
    claim_boundary: str
    evidence_kind: str
    execution_disabled: bool
    actuating: bool
    time_state_sha256: str
    sample_records_sha256: str
    transition_table_sha256: str
    timeline_sha256: str

    def to_dict(self) -> dict[str, float | int | bool | str]:
        """Return a JSON-safe canonical representation."""

        return pha_c_event_timeline_to_dict(self)


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


def _as_float_matrix(values: ArrayLike, *, name: str) -> FloatArray:
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
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one time sample and oscillator")
    try:
        out = np.ascontiguousarray(array, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite values")
    return out


def _as_time_vector(times: ArrayLike | None, *, sample_count: int) -> FloatArray:
    if times is None:
        return np.arange(sample_count, dtype=np.float64)
    array = np.asarray(times)
    if array.dtype == np.dtype("O"):
        raise ValueError("times must be a finite real-valued vector")
    if np.issubdtype(array.dtype, np.bool_) or np.issubdtype(
        array.dtype,
        np.complexfloating,
    ):
        raise ValueError("times must be real-valued, not boolean or complex")
    if array.ndim != 1:
        raise ValueError("times must be one-dimensional")
    if array.shape[0] != sample_count:
        raise ValueError("times must have one entry per trajectory sample")
    try:
        out = np.ascontiguousarray(array, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("times must be numeric") from exc
    if not np.all(np.isfinite(out)):
        raise ValueError("times must contain only finite values")
    if sample_count > 1 and np.any(np.diff(out) <= 0.0):
        raise ValueError("times must be strictly increasing")
    return out


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


def _vector_sha256(values: FloatArray) -> str:
    payload = {
        "dtype": "float64",
        "shape": list(values.shape),
        "values_sha256": hashlib.sha256(
            np.ascontiguousarray(values, dtype=np.float64).tobytes(),
        ).hexdigest(),
    }
    return _sha256_json(payload)


def _transition_table(records: list[PHACHandoffRecord]) -> list[dict[str, bool | int]]:
    transitions: list[dict[str, bool | int]] = []
    for index in range(1, len(records)):
        previous = records[index - 1]
        current = records[index]
        lock_lost = previous.lock_achieved and not current.lock_achieved
        reset = (
            previous.consecutive_lock_samples > 0
            and current.consecutive_lock_samples == 0
        )
        transitions.append(
            {
                "from_index": index - 1,
                "to_index": index,
                "lock_lost": bool(lock_lost),
                "reset": bool(reset),
            },
        )
    return transitions


def _timeline_dict_without_hash(
    *,
    sample_count: int,
    oscillator_count: int,
    start_time: float,
    end_time: float,
    duration_s: float,
    first_lock_index: int,
    first_lock_time: float,
    first_lock_observed: bool,
    final_lock_achieved: bool,
    lock_sample_count: int,
    phase_lock_sample_count: int,
    spatial_lock_sample_count: int,
    lock_loss_count: int,
    reset_count: int,
    max_consecutive_lock_samples: int,
    max_phase_dispersion_rad: float,
    max_spatial_dispersion_m: float,
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
    time_state_sha256: str,
    sample_records_sha256: str,
    transition_table_sha256: str,
) -> dict[str, float | int | bool | str]:
    return {
        "sample_count": int(sample_count),
        "oscillator_count": int(oscillator_count),
        "start_time": float(start_time),
        "end_time": float(end_time),
        "duration_s": float(duration_s),
        "first_lock_index": int(first_lock_index),
        "first_lock_time": float(first_lock_time),
        "first_lock_observed": bool(first_lock_observed),
        "final_lock_achieved": bool(final_lock_achieved),
        "lock_sample_count": int(lock_sample_count),
        "phase_lock_sample_count": int(phase_lock_sample_count),
        "spatial_lock_sample_count": int(spatial_lock_sample_count),
        "lock_loss_count": int(lock_loss_count),
        "reset_count": int(reset_count),
        "max_consecutive_lock_samples": int(max_consecutive_lock_samples),
        "max_phase_dispersion_rad": float(max_phase_dispersion_rad),
        "max_spatial_dispersion_m": float(max_spatial_dispersion_m),
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
        "claim_boundary": PHA_C_TIMELINE_CLAIM_BOUNDARY,
        "evidence_kind": PHA_C_TIMELINE_EVIDENCE_KIND,
        "execution_disabled": True,
        "actuating": False,
        "time_state_sha256": time_state_sha256,
        "sample_records_sha256": sample_records_sha256,
        "transition_table_sha256": transition_table_sha256,
    }


def build_pha_c_event_timeline(
    phases_by_step: ArrayLike,
    positions_by_step: ArrayLike,
    *,
    times: ArrayLike | None = None,
    reference_phase: object = 0.0,
    reference_point: object = 0.0,
    phase_tol_rad: object = DEFAULT_PHASE_TOL_RAD,
    spatial_tol_m: object = DEFAULT_SPATIAL_TOL_M,
    required_consecutive_samples: object = 3,
    tolerance_profile: object | None = None,
) -> PHACTimelineRecord:
    """Build deterministic PHA-C lock/loss timeline evidence.

    Parameters mirror :func:`build_pha_c_handoff_record`, but consume complete
    trajectory matrices with shape ``(time, oscillator)``. The consecutive-lock
    counter is threaded through every sample so downstream consumers can review
    acquisition, loss, and reset events without raw state arrays.
    """

    phase_matrix = _as_float_matrix(phases_by_step, name="phases_by_step")
    position_matrix = _as_float_matrix(positions_by_step, name="positions_by_step")
    if position_matrix.shape != phase_matrix.shape:
        raise ValueError("positions_by_step must have the same shape as phases_by_step")

    sample_count, oscillator_count = phase_matrix.shape
    time_vector = _as_time_vector(times, sample_count=sample_count)

    records: list[PHACHandoffRecord] = []
    consecutive = 0
    for index, timestamp in enumerate(time_vector):
        record = build_pha_c_handoff_record(
            phase_matrix[index],
            position_matrix[index],
            t=float(timestamp),
            reference_phase=reference_phase,
            reference_point=reference_point,
            phase_tol_rad=phase_tol_rad,
            spatial_tol_m=spatial_tol_m,
            required_consecutive_samples=required_consecutive_samples,
            prior_consecutive_lock_samples=consecutive,
            tolerance_profile=tolerance_profile,
        )
        consecutive = record.consecutive_lock_samples
        records.append(record)

    first = records[0]
    first_lock_index = next(
        (index for index, record in enumerate(records) if record.lock_achieved),
        -1,
    )
    first_lock_observed = first_lock_index >= 0
    first_lock_time = (
        float(time_vector[first_lock_index]) if first_lock_observed else 0.0
    )
    transition_rows = _transition_table(records)
    record_dicts = [record.to_dict() for record in records]

    sample_records_sha256 = _sha256_json(record_dicts)
    time_state_sha256 = _vector_sha256(time_vector)
    transition_table_sha256 = _sha256_json(transition_rows)
    timeline_payload = _timeline_dict_without_hash(
        sample_count=sample_count,
        oscillator_count=oscillator_count,
        start_time=float(time_vector[0]),
        end_time=float(time_vector[-1]),
        duration_s=float(time_vector[-1] - time_vector[0]),
        first_lock_index=first_lock_index,
        first_lock_time=first_lock_time,
        first_lock_observed=first_lock_observed,
        final_lock_achieved=records[-1].lock_achieved,
        lock_sample_count=sum(record.lock_achieved for record in records),
        phase_lock_sample_count=sum(record.phase_locked for record in records),
        spatial_lock_sample_count=sum(record.spatial_locked for record in records),
        lock_loss_count=sum(row["lock_lost"] for row in transition_rows),
        reset_count=sum(row["reset"] for row in transition_rows),
        max_consecutive_lock_samples=max(
            record.consecutive_lock_samples for record in records
        ),
        max_phase_dispersion_rad=max(record.phase_dispersion_rad for record in records),
        max_spatial_dispersion_m=max(record.spatial_dispersion_m for record in records),
        min_phase_margin_rad=min(record.phase_margin_rad for record in records),
        min_spatial_margin_m=min(record.spatial_margin_m for record in records),
        min_phase_order_parameter=min(
            record.phase_order_parameter for record in records
        ),
        max_distance_to_reference_m=max(
            record.distance_to_reference_max_m for record in records
        ),
        reference_phase=first.reference_phase,
        reference_point=first.reference_point,
        phase_tol_rad=first.phase_tol_rad,
        spatial_tol_m=first.spatial_tol_m,
        tolerance_profile_name=first.tolerance_profile_name,
        tolerance_profile_multiplier=first.tolerance_profile_multiplier,
        required_consecutive_samples=first.required_consecutive_samples,
        time_state_sha256=time_state_sha256,
        sample_records_sha256=sample_records_sha256,
        transition_table_sha256=transition_table_sha256,
    )
    return PHACTimelineRecord(
        **cast(Any, timeline_payload),
        timeline_sha256=_sha256_json(timeline_payload),
    )


def pha_c_event_timeline_to_dict(
    timeline: PHACTimelineRecord,
) -> dict[str, float | int | bool | str]:
    """Return the canonical JSON-safe PHA-C timeline payload."""

    payload = _timeline_dict_without_hash(
        sample_count=timeline.sample_count,
        oscillator_count=timeline.oscillator_count,
        start_time=timeline.start_time,
        end_time=timeline.end_time,
        duration_s=timeline.duration_s,
        first_lock_index=timeline.first_lock_index,
        first_lock_time=timeline.first_lock_time,
        first_lock_observed=timeline.first_lock_observed,
        final_lock_achieved=timeline.final_lock_achieved,
        lock_sample_count=timeline.lock_sample_count,
        phase_lock_sample_count=timeline.phase_lock_sample_count,
        spatial_lock_sample_count=timeline.spatial_lock_sample_count,
        lock_loss_count=timeline.lock_loss_count,
        reset_count=timeline.reset_count,
        max_consecutive_lock_samples=timeline.max_consecutive_lock_samples,
        max_phase_dispersion_rad=timeline.max_phase_dispersion_rad,
        max_spatial_dispersion_m=timeline.max_spatial_dispersion_m,
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
        time_state_sha256=timeline.time_state_sha256,
        sample_records_sha256=timeline.sample_records_sha256,
        transition_table_sha256=timeline.transition_table_sha256,
    )
    payload["timeline_sha256"] = timeline.timeline_sha256
    return payload


def verify_pha_c_event_timeline(
    timeline: PHACTimelineRecord,
) -> PHACTimelineRecord:
    """Replay and validate a PHA-C event timeline hash and safety boundary."""

    if not isinstance(timeline, PHACTimelineRecord):
        raise ValueError("timeline must be a PHACTimelineRecord")
    _validate_sha256_hex(timeline.time_state_sha256, name="time_state_sha256")
    _validate_sha256_hex(
        timeline.sample_records_sha256,
        name="sample_records_sha256",
    )
    _validate_sha256_hex(
        timeline.transition_table_sha256,
        name="transition_table_sha256",
    )
    timeline_hash = _validate_sha256_hex(
        timeline.timeline_sha256,
        name="timeline_sha256",
    )
    if timeline.claim_boundary != PHA_C_TIMELINE_CLAIM_BOUNDARY:
        raise ValueError("claim_boundary must be the PHA-C timeline review boundary")
    if timeline.evidence_kind != PHA_C_TIMELINE_EVIDENCE_KIND:
        raise ValueError("evidence_kind must be deterministic timeline evidence")
    if (
        _validate_record_bool(
            timeline.execution_disabled,
            name="execution_disabled",
        )
        is not True
    ):
        raise ValueError("execution_disabled must be true")
    if _validate_record_bool(timeline.actuating, name="actuating") is not False:
        raise ValueError("actuating must be false")

    sample_count = _validate_record_int(
        timeline.sample_count,
        name="sample_count",
        minimum=1,
    )
    _validate_record_int(timeline.oscillator_count, name="oscillator_count", minimum=1)
    required = _validate_record_int(
        timeline.required_consecutive_samples,
        name="required_consecutive_samples",
        minimum=1,
    )
    count_fields = (
        "lock_sample_count",
        "phase_lock_sample_count",
        "spatial_lock_sample_count",
        "lock_loss_count",
        "reset_count",
        "max_consecutive_lock_samples",
    )
    counts = {
        field: _validate_record_int(getattr(timeline, field), name=field, minimum=0)
        for field in count_fields
    }
    for field in (
        "lock_sample_count",
        "phase_lock_sample_count",
        "spatial_lock_sample_count",
        "max_consecutive_lock_samples",
    ):
        if counts[field] > sample_count:
            raise ValueError(f"{field} cannot exceed sample_count")
    for field in ("lock_loss_count", "reset_count"):
        if counts[field] > max(sample_count - 1, 0):
            raise ValueError(f"{field} cannot exceed the transition count")
    final_lock_achieved = _validate_record_bool(
        timeline.final_lock_achieved,
        name="final_lock_achieved",
    )
    if counts["max_consecutive_lock_samples"] < required and final_lock_achieved:
        raise ValueError("final_lock_achieved requires the consecutive threshold")

    start_time = _validate_real_scalar(timeline.start_time, name="start_time")
    end_time = _validate_real_scalar(timeline.end_time, name="end_time")
    duration_s = _validate_nonnegative_record_scalar(
        timeline.duration_s,
        name="duration_s",
    )
    if end_time < start_time:
        raise ValueError("end_time must be greater than or equal to start_time")
    if abs(duration_s - (end_time - start_time)) > 1.0e-12:
        raise ValueError("duration_s must equal end_time - start_time")
    first_lock_index = _validate_record_int(
        timeline.first_lock_index,
        name="first_lock_index",
        minimum=-1,
    )
    first_lock_observed = _validate_record_bool(
        timeline.first_lock_observed,
        name="first_lock_observed",
    )
    first_lock_time = _validate_real_scalar(
        timeline.first_lock_time,
        name="first_lock_time",
    )
    if first_lock_observed:
        if first_lock_index < 0 or first_lock_index >= sample_count:
            raise ValueError("first_lock_index must refer to an observed sample")
        if first_lock_time < start_time or first_lock_time > end_time:
            raise ValueError("first_lock_time must be inside the timeline range")
    else:
        if first_lock_index != -1:
            raise ValueError("first_lock_index must be -1 when no lock is observed")
        if first_lock_time != 0.0:
            raise ValueError("first_lock_time must be 0.0 when no lock is observed")

    for field in (
        "max_phase_dispersion_rad",
        "max_spatial_dispersion_m",
        "max_distance_to_reference_m",
        "phase_tol_rad",
        "spatial_tol_m",
    ):
        _validate_nonnegative_record_scalar(getattr(timeline, field), name=field)
    max_phase_dispersion = _validate_nonnegative_record_scalar(
        timeline.max_phase_dispersion_rad,
        name="max_phase_dispersion_rad",
    )
    max_spatial_dispersion = _validate_nonnegative_record_scalar(
        timeline.max_spatial_dispersion_m,
        name="max_spatial_dispersion_m",
    )
    phase_tol = _validate_nonnegative_record_scalar(
        timeline.phase_tol_rad,
        name="phase_tol_rad",
    )
    spatial_tol = _validate_nonnegative_record_scalar(
        timeline.spatial_tol_m,
        name="spatial_tol_m",
    )
    min_phase_margin = _validate_real_scalar(
        timeline.min_phase_margin_rad,
        name="min_phase_margin_rad",
    )
    min_spatial_margin = _validate_real_scalar(
        timeline.min_spatial_margin_m,
        name="min_spatial_margin_m",
    )
    if (
        abs(min_phase_margin - (phase_tol - max_phase_dispersion))
        > PHA_C_TIMELINE_MARGIN_REPLAY_TOLERANCE
    ):
        raise ValueError(
            "min_phase_margin_rad must equal phase_tol_rad - max_phase_dispersion_rad"
        )
    if (
        abs(min_spatial_margin - (spatial_tol - max_spatial_dispersion))
        > PHA_C_TIMELINE_MARGIN_REPLAY_TOLERANCE
    ):
        raise ValueError(
            "min_spatial_margin_m must equal spatial_tol_m - max_spatial_dispersion_m"
        )
    order_parameter = _validate_nonnegative_record_scalar(
        timeline.min_phase_order_parameter,
        name="min_phase_order_parameter",
    )
    if order_parameter > 1.0 + 1.0e-12:
        raise ValueError("min_phase_order_parameter must be inside [0, 1]")
    multiplier = _validate_real_scalar(
        timeline.tolerance_profile_multiplier,
        name="tolerance_profile_multiplier",
    )
    if multiplier <= 0.0:
        raise ValueError("tolerance_profile_multiplier must be positive")
    if (
        not isinstance(timeline.tolerance_profile_name, str)
        or not timeline.tolerance_profile_name
    ):
        raise ValueError("tolerance_profile_name must be a non-empty string")
    for field in ("reference_phase", "reference_point"):
        _validate_real_scalar(getattr(timeline, field), name=field)

    payload = pha_c_event_timeline_to_dict(timeline)
    replay_payload = dict(payload)
    replay_payload.pop("timeline_sha256")
    if _sha256_json(replay_payload) != timeline_hash:
        raise ValueError(
            "timeline_sha256 does not match the canonical timeline payload",
        )
    return timeline
