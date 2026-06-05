# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — PHA-C event-state handoff

"""Deterministic PHA-C event/state handoff records.

The PHA-C moving-frame lane produces phases and axial positions. The merge
window monitor decides whether that phase-space state is inside the reviewed
merge tolerance. This module binds those two surfaces into a non-actuating,
hash-stable handoff record for downstream replay, Studio review, MIF import, or
operator evidence streams.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from scpn_phase_orchestrator.monitor.merge_window import (
    DEFAULT_PHASE_TOL_RAD,
    DEFAULT_SPATIAL_TOL_M,
    MergeReport,
    evaluate_merge_window,
    resolve_merge_window_tolerance_profile,
)

FloatArray: TypeAlias = NDArray[np.float64]

PHA_C_HANDOFF_CLAIM_BOUNDARY = "pha_c_event_state_handoff_review_only"
PHA_C_HANDOFF_EVIDENCE_KIND = "deterministic_non_actuating_handoff"
PHA_C_HANDOFF_MARGIN_REPLAY_TOLERANCE = 1.0e-12
_TWO_PI = 2.0 * np.pi

__all__ = [
    "PHA_C_HANDOFF_CLAIM_BOUNDARY",
    "PHA_C_HANDOFF_EVIDENCE_KIND",
    "PHA_C_HANDOFF_MARGIN_REPLAY_TOLERANCE",
    "PHACHandoffRecord",
    "build_pha_c_handoff_record",
    "pha_c_handoff_record_to_dict",
    "verify_pha_c_handoff_record",
]


@dataclass(frozen=True, slots=True)
class PHACHandoffRecord:
    """Audit-ready downstream handoff for a PHA-C moving-frame sample.

    The record intentionally carries only scalar evidence and SHA-256 digests of
    the sampled phase/position vectors. It is suitable for replay and review
    lanes, but it never authorises actuation.
    """

    t: float
    oscillator_count: int
    phase_dispersion_rad: float
    spatial_dispersion_m: float
    phase_margin_rad: float
    spatial_margin_m: float
    phase_locked: bool
    spatial_locked: bool
    lock_achieved: bool
    consecutive_lock_samples: int
    phase_order_parameter: float
    distance_to_reference_max_m: float
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
    phase_state_sha256: str
    position_state_sha256: str
    merge_report_sha256: str
    source_chain_sha256: str
    record_sha256: str

    def to_dict(self) -> dict[str, float | int | bool | str]:
        """Return a JSON-safe canonical representation."""

        return pha_c_handoff_record_to_dict(self)


def _validate_real_scalar(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        parsed = float(value)  # type: ignore[arg-type]  # type ignore: runtime validator coerces object input
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _validate_tolerance(value: object, *, name: str) -> float:
    parsed = _validate_real_scalar(value, name=name)
    if parsed < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _validate_sample_count(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return parsed


def _as_float_vector(values: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(values)
    if array.dtype == np.dtype("O"):
        raise ValueError(f"{name} must be a finite real-valued vector")
    if np.issubdtype(array.dtype, np.bool_) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise ValueError(f"{name} must be real-valued, not boolean or complex")
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one sample")
    try:
        out = np.ascontiguousarray(array, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite values")
    return out


def _sha256_json(payload: Mapping[str, object]) -> str:
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
            np.ascontiguousarray(values, dtype=np.float64).tobytes()
        ).hexdigest(),
    }
    return _sha256_json(payload)


def _merge_report_sha256(report: MergeReport) -> str:
    return _sha256_json(report.to_dict())


def _phase_order_parameter(phases: FloatArray) -> float:
    return float(np.abs(np.mean(np.exp(1j * np.remainder(phases, _TWO_PI)))))


def _distance_to_reference_max(positions: FloatArray, reference_point: float) -> float:
    return float(np.max(np.abs(positions - reference_point)))


def _record_dict_without_hash(
    *,
    report: MergeReport,
    oscillator_count: int,
    phase_order_parameter: float,
    distance_to_reference_max_m: float,
    reference_phase: float,
    reference_point: float,
    phase_tol_rad: float,
    spatial_tol_m: float,
    tolerance_profile_name: str,
    tolerance_profile_multiplier: float,
    required_consecutive_samples: int,
    phase_state_sha256: str,
    position_state_sha256: str,
    merge_report_sha256: str,
    source_chain_sha256: str,
) -> dict[str, float | int | bool | str]:
    return {
        "t": float(report.t),
        "oscillator_count": int(oscillator_count),
        "phase_dispersion_rad": float(report.phase_dispersion_rad),
        "spatial_dispersion_m": float(report.spatial_dispersion_m),
        "phase_margin_rad": float(report.phase_margin_rad),
        "spatial_margin_m": float(report.spatial_margin_m),
        "phase_locked": bool(report.phase_locked),
        "spatial_locked": bool(report.spatial_locked),
        "lock_achieved": bool(report.lock_achieved),
        "consecutive_lock_samples": int(report.consecutive_lock_samples),
        "phase_order_parameter": float(phase_order_parameter),
        "distance_to_reference_max_m": float(distance_to_reference_max_m),
        "reference_phase": float(reference_phase),
        "reference_point": float(reference_point),
        "phase_tol_rad": float(phase_tol_rad),
        "spatial_tol_m": float(spatial_tol_m),
        "tolerance_profile_name": str(tolerance_profile_name),
        "tolerance_profile_multiplier": float(tolerance_profile_multiplier),
        "required_consecutive_samples": int(required_consecutive_samples),
        "claim_boundary": PHA_C_HANDOFF_CLAIM_BOUNDARY,
        "evidence_kind": PHA_C_HANDOFF_EVIDENCE_KIND,
        "execution_disabled": True,
        "actuating": False,
        "phase_state_sha256": phase_state_sha256,
        "position_state_sha256": position_state_sha256,
        "merge_report_sha256": merge_report_sha256,
        "source_chain_sha256": source_chain_sha256,
    }


def build_pha_c_handoff_record(
    phases: ArrayLike,
    positions: ArrayLike,
    *,
    t: object = 0.0,
    reference_phase: object = 0.0,
    reference_point: object = 0.0,
    phase_tol_rad: object = DEFAULT_PHASE_TOL_RAD,
    spatial_tol_m: object = DEFAULT_SPATIAL_TOL_M,
    required_consecutive_samples: object = 3,
    prior_consecutive_lock_samples: object = 0,
    tolerance_profile: object | None = None,
) -> PHACHandoffRecord:
    """Build a deterministic PHA-C event/state handoff record.

    The handoff consumes the same phase/position sample as
    :func:`evaluate_merge_window`, mirrors its fail-closed validation, adds
    Kuramoto order-parameter and source-chain digests, then returns a
    non-actuating record with a canonical hash.
    """

    phase_vector = _as_float_vector(phases, name="phases")
    position_vector = _as_float_vector(positions, name="positions")
    if position_vector.shape != phase_vector.shape:
        raise ValueError("positions must have the same one-dimensional shape as phases")

    timestamp = _validate_real_scalar(t, name="t")
    phase_reference = _validate_real_scalar(reference_phase, name="reference_phase")
    spatial_reference = _validate_real_scalar(reference_point, name="reference_point")
    tolerance_profile_name = "explicit"
    tolerance_profile_multiplier = 1.0
    if tolerance_profile is None:
        phase_tol = _validate_tolerance(phase_tol_rad, name="phase_tol_rad")
        spatial_tol = _validate_tolerance(spatial_tol_m, name="spatial_tol_m")
    else:
        profile = resolve_merge_window_tolerance_profile(
            tolerance_profile,
            phase_baseline_rad=phase_tol_rad,
            spatial_baseline_m=spatial_tol_m,
        )
        phase_tol = profile.phase_tol_rad
        spatial_tol = profile.spatial_tol_m
        tolerance_profile_name = profile.name
        tolerance_profile_multiplier = profile.multiplier
    required = _validate_sample_count(
        required_consecutive_samples,
        name="required_consecutive_samples",
        minimum=1,
    )
    prior = _validate_sample_count(
        prior_consecutive_lock_samples,
        name="prior_consecutive_lock_samples",
        minimum=0,
    )

    report = evaluate_merge_window(
        phase_vector,
        position_vector,
        t=timestamp,
        reference_phase=phase_reference,
        reference_point=spatial_reference,
        phase_tol_rad=phase_tol,
        spatial_tol_m=spatial_tol,
        required_consecutive_samples=required,
        prior_consecutive_lock_samples=prior,
    )
    phase_hash = _vector_sha256(phase_vector)
    position_hash = _vector_sha256(position_vector)
    merge_hash = _merge_report_sha256(report)
    source_chain_hash = _sha256_json(
        {
            "phase_state_sha256": phase_hash,
            "position_state_sha256": position_hash,
            "merge_report_sha256": merge_hash,
        }
    )
    order_parameter = _phase_order_parameter(phase_vector)
    max_distance = _distance_to_reference_max(position_vector, spatial_reference)
    record_payload = _record_dict_without_hash(
        report=report,
        oscillator_count=int(phase_vector.size),
        phase_order_parameter=order_parameter,
        distance_to_reference_max_m=max_distance,
        reference_phase=phase_reference,
        reference_point=spatial_reference,
        phase_tol_rad=phase_tol,
        spatial_tol_m=spatial_tol,
        tolerance_profile_name=tolerance_profile_name,
        tolerance_profile_multiplier=tolerance_profile_multiplier,
        required_consecutive_samples=required,
        phase_state_sha256=phase_hash,
        position_state_sha256=position_hash,
        merge_report_sha256=merge_hash,
        source_chain_sha256=source_chain_hash,
    )
    record_hash = _sha256_json(record_payload)
    return PHACHandoffRecord(
        **cast(Any, record_payload),
        record_sha256=record_hash,
    )


def pha_c_handoff_record_to_dict(
    record: PHACHandoffRecord,
) -> dict[str, float | int | bool | str]:
    """Convert a :class:`PHACHandoffRecord` into a JSON-safe dictionary."""

    return {
        "t": float(record.t),
        "oscillator_count": int(record.oscillator_count),
        "phase_dispersion_rad": float(record.phase_dispersion_rad),
        "spatial_dispersion_m": float(record.spatial_dispersion_m),
        "phase_margin_rad": float(record.phase_margin_rad),
        "spatial_margin_m": float(record.spatial_margin_m),
        "phase_locked": bool(record.phase_locked),
        "spatial_locked": bool(record.spatial_locked),
        "lock_achieved": bool(record.lock_achieved),
        "consecutive_lock_samples": int(record.consecutive_lock_samples),
        "phase_order_parameter": float(record.phase_order_parameter),
        "distance_to_reference_max_m": float(record.distance_to_reference_max_m),
        "reference_phase": float(record.reference_phase),
        "reference_point": float(record.reference_point),
        "phase_tol_rad": float(record.phase_tol_rad),
        "spatial_tol_m": float(record.spatial_tol_m),
        "tolerance_profile_name": str(record.tolerance_profile_name),
        "tolerance_profile_multiplier": float(record.tolerance_profile_multiplier),
        "required_consecutive_samples": int(record.required_consecutive_samples),
        "claim_boundary": str(record.claim_boundary),
        "evidence_kind": str(record.evidence_kind),
        "execution_disabled": bool(record.execution_disabled),
        "actuating": bool(record.actuating),
        "phase_state_sha256": str(record.phase_state_sha256),
        "position_state_sha256": str(record.position_state_sha256),
        "merge_report_sha256": str(record.merge_report_sha256),
        "source_chain_sha256": str(record.source_chain_sha256),
        "record_sha256": str(record.record_sha256),
    }


def verify_pha_c_handoff_record(record: PHACHandoffRecord) -> PHACHandoffRecord:
    """Replay and validate a PHA-C handoff record hash and safety boundary.

    The verifier is intentionally independent of the builder path: it checks the
    scalar invariants carried by an existing record, validates all SHA-256 digest
    fields, and recomputes the canonical payload hash from ``to_dict()``. This
    lets benchmark gates, replay ledgers, and downstream MIF/FRC lanes reject
    tampered evidence without access to the original phase and position vectors.
    """

    if not isinstance(record, PHACHandoffRecord):
        raise ValueError("record must be a PHACHandoffRecord")
    _validate_sha256_hex(record.phase_state_sha256, name="phase_state_sha256")
    _validate_sha256_hex(record.position_state_sha256, name="position_state_sha256")
    _validate_sha256_hex(record.merge_report_sha256, name="merge_report_sha256")
    _validate_sha256_hex(record.source_chain_sha256, name="source_chain_sha256")
    record_hash = _validate_sha256_hex(record.record_sha256, name="record_sha256")
    if record.claim_boundary != PHA_C_HANDOFF_CLAIM_BOUNDARY:
        raise ValueError("claim_boundary must be the PHA-C handoff review boundary")
    if record.evidence_kind != PHA_C_HANDOFF_EVIDENCE_KIND:
        raise ValueError("evidence_kind must be deterministic handoff evidence")
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

    _validate_record_int(record.oscillator_count, name="oscillator_count", minimum=1)
    required = _validate_record_int(
        record.required_consecutive_samples,
        name="required_consecutive_samples",
        minimum=1,
    )
    consecutive = _validate_record_int(
        record.consecutive_lock_samples,
        name="consecutive_lock_samples",
        minimum=0,
    )
    phase_locked = _validate_record_bool(record.phase_locked, name="phase_locked")
    spatial_locked = _validate_record_bool(record.spatial_locked, name="spatial_locked")
    lock_achieved = _validate_record_bool(record.lock_achieved, name="lock_achieved")
    if lock_achieved and (not phase_locked or not spatial_locked):
        raise ValueError("lock_achieved requires phase and spatial locks")
    if lock_achieved and consecutive < required:
        raise ValueError("lock_achieved requires the consecutive-sample threshold")
    if (not phase_locked or not spatial_locked) and consecutive != 0:
        raise ValueError("unlocked samples must reset consecutive_lock_samples")

    for field in (
        "phase_dispersion_rad",
        "spatial_dispersion_m",
        "distance_to_reference_max_m",
        "phase_tol_rad",
        "spatial_tol_m",
    ):
        _validate_nonnegative_record_scalar(getattr(record, field), name=field)
    phase_dispersion = _validate_nonnegative_record_scalar(
        record.phase_dispersion_rad,
        name="phase_dispersion_rad",
    )
    spatial_dispersion = _validate_nonnegative_record_scalar(
        record.spatial_dispersion_m,
        name="spatial_dispersion_m",
    )
    phase_tol = _validate_nonnegative_record_scalar(
        record.phase_tol_rad,
        name="phase_tol_rad",
    )
    spatial_tol = _validate_nonnegative_record_scalar(
        record.spatial_tol_m,
        name="spatial_tol_m",
    )
    phase_margin = _validate_real_scalar(
        record.phase_margin_rad,
        name="phase_margin_rad",
    )
    spatial_margin = _validate_real_scalar(
        record.spatial_margin_m,
        name="spatial_margin_m",
    )
    if (
        abs(phase_margin - (phase_tol - phase_dispersion))
        > PHA_C_HANDOFF_MARGIN_REPLAY_TOLERANCE
    ):
        raise ValueError(
            "phase_margin_rad must equal phase_tol_rad - phase_dispersion_rad"
        )
    if (
        abs(spatial_margin - (spatial_tol - spatial_dispersion))
        > PHA_C_HANDOFF_MARGIN_REPLAY_TOLERANCE
    ):
        raise ValueError(
            "spatial_margin_m must equal spatial_tol_m - spatial_dispersion_m"
        )
    if phase_locked and phase_margin < -PHA_C_HANDOFF_MARGIN_REPLAY_TOLERANCE:
        raise ValueError("phase_locked requires a non-negative phase_margin_rad")
    if not phase_locked and phase_margin >= 0.0:
        raise ValueError("phase-unlocked records require a negative phase_margin_rad")
    if spatial_locked and spatial_margin < -PHA_C_HANDOFF_MARGIN_REPLAY_TOLERANCE:
        raise ValueError("spatial_locked requires a non-negative spatial_margin_m")
    if not spatial_locked and spatial_margin >= 0.0:
        raise ValueError("spatial-unlocked records require a negative spatial_margin_m")
    order_parameter = _validate_nonnegative_record_scalar(
        record.phase_order_parameter,
        name="phase_order_parameter",
    )
    if order_parameter > 1.0 + 1.0e-12:
        raise ValueError("phase_order_parameter must be inside [0, 1]")
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
    for field in ("t", "reference_phase", "reference_point"):
        _validate_real_scalar(getattr(record, field), name=field)

    payload = pha_c_handoff_record_to_dict(record)
    replay_payload = dict(payload)
    replay_payload.pop("record_sha256")
    if _sha256_json(replay_payload) != record_hash:
        raise ValueError("record_sha256 does not match the canonical handoff payload")
    return record
