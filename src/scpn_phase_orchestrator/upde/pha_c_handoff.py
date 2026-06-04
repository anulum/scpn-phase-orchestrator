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
from dataclasses import dataclass
from math import isfinite
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from scpn_phase_orchestrator.monitor.merge_window import (
    MergeReport,
    evaluate_merge_window,
)

FloatArray: TypeAlias = NDArray[np.float64]

PHA_C_HANDOFF_CLAIM_BOUNDARY = "pha_c_event_state_handoff_review_only"
PHA_C_HANDOFF_EVIDENCE_KIND = "deterministic_non_actuating_handoff"
_TWO_PI = 2.0 * np.pi

__all__ = [
    "PHA_C_HANDOFF_CLAIM_BOUNDARY",
    "PHA_C_HANDOFF_EVIDENCE_KIND",
    "PHACHandoffRecord",
    "build_pha_c_handoff_record",
    "pha_c_handoff_record_to_dict",
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
        parsed = float(value)  # type: ignore[arg-type]
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


def _sha256_json(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


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
    phase_tol_rad: object = 0.01,
    spatial_tol_m: object = 0.002,
    required_consecutive_samples: object = 3,
    prior_consecutive_lock_samples: object = 0,
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
    phase_tol = _validate_tolerance(phase_tol_rad, name="phase_tol_rad")
    spatial_tol = _validate_tolerance(spatial_tol_m, name="spatial_tol_m")
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
        required_consecutive_samples=required,
        phase_state_sha256=phase_hash,
        position_state_sha256=position_hash,
        merge_report_sha256=merge_hash,
        source_chain_sha256=source_chain_hash,
    )
    record_hash = _sha256_json(record_payload)
    return PHACHandoffRecord(
        **record_payload,
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
