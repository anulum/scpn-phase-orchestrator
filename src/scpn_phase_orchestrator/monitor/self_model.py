# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Self-model error monitor

"""Deterministic self-model discrepancy monitor with auditable evidence.

Computes channel-wise and aggregate errors between observed and predicted phase
trajectories, optional order-parameter errors, deterministic breach flags, and a
stable evidence hash suitable for non-actuating industrial reporting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from numbers import Real
from typing import Final

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

__all__ = [
    "SelfModelErrorResult",
    "SelfModelErrorThresholdConfig",
    "compute_self_model_error",
]

CLAIM_BOUNDARY: Final = "self_model_error_monitor_not_live_reconfiguration"
BACKEND: Final = "numpy_self_model_error_reference"


@dataclass(frozen=True)
class SelfModelErrorThresholdConfig:
    """Thresholds and optional order-specific thresholds for monitor evaluation."""

    tolerance: float
    max_abs_tolerance: float
    order_tolerance: float | None = None
    order_max_abs_tolerance: float | None = None


@dataclass(frozen=True)
class SelfModelErrorResult:
    """Deterministic result of one self-model error monitor invocation."""

    domain: str
    scenario_id: str | None
    channel_labels: tuple[str, ...]
    channel_count: int
    sample_count: int
    overall_rmse: float
    overall_mae: float
    overall_max_abs_error: float
    channel_rmse: tuple[float, ...]
    channel_mae: tuple[float, ...]
    channel_max_abs_error: tuple[float, ...]
    channel_breaches: tuple[bool, ...]
    weighted_rmse: float | None
    weighted_mae: float | None
    weighted_max_abs_error: float | None
    channel_weights: tuple[float, ...] | None
    tolerance: float
    max_abs_tolerance: float
    breached: bool
    order_rmse: float | None
    order_mae: float | None
    order_max_abs_error: float | None
    order_breached: bool | None
    claim_boundary: str
    non_actuating: bool
    execution_disabled: bool
    backend: str
    record_hash: str

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe audit record for the computed monitor output.

        Returns
        -------
        dict[str, object]
            Return a JSON-safe audit record for the computed monitor output.
        """
        record: dict[str, object] = {
            "domain": self.domain,
            "scenario_id": self.scenario_id,
            "backend": self.backend,
            "channel_labels": list(self.channel_labels),
            "channel_count": self.channel_count,
            "sample_count": self.sample_count,
            "channel_rmse": list(self.channel_rmse),
            "channel_mae": list(self.channel_mae),
            "channel_max_abs_error": list(self.channel_max_abs_error),
            "channel_breaches": list(self.channel_breaches),
            "channel_weights": None
            if self.channel_weights is None
            else list(self.channel_weights),
            "overall_rmse": self.overall_rmse,
            "overall_mae": self.overall_mae,
            "overall_max_abs_error": self.overall_max_abs_error,
            "weighted_rmse": self.weighted_rmse,
            "weighted_mae": self.weighted_mae,
            "weighted_max_abs_error": self.weighted_max_abs_error,
            "tolerance": self.tolerance,
            "max_abs_tolerance": self.max_abs_tolerance,
            "breached": self.breached,
            "order_rmse": self.order_rmse,
            "order_mae": self.order_mae,
            "order_max_abs_error": self.order_max_abs_error,
            "order_breached": self.order_breached,
            "claim_boundary": self.claim_boundary,
            "non_actuating": self.non_actuating,
            "execution_disabled": self.execution_disabled,
        }
        record["record_hash"] = _deterministic_record_hash(record)
        return record


def compute_self_model_error(
    observed_phases: object,
    predicted_phases: object,
    *,
    observed_order: object | None = None,
    predicted_order: object | None = None,
    channel_labels: object | None = None,
    channel_weights: object | None = None,
    tolerance: float = 0.0,
    max_abs_tolerance: float = 0.0,
    domain: str = "self_model",
    scenario_id: str | None = None,
) -> SelfModelErrorResult:
    """Compute deterministic channel-wise discrepancy metrics for a self-model pair.

    Parameters
    ----------
    observed_phases : object
        Observed phase trajectories shaped ``(C, T)`` or ``(T,)``.
    predicted_phases : object
        Predicted phase trajectories with matching shape.
    observed_order : object | None
        Optional observed order signal, shape ``(C,)``.
    predicted_order : object | None
        Optional predicted order signal, shape ``(C,)``.
    channel_labels : object | None
        Optional channel names for audit output.
    channel_weights : object | None
        Optional positive weights for channels.
    tolerance : float
        Global RMSE threshold used for pass/fail decisions.
    max_abs_tolerance : float
        Global max-abs threshold used for pass/fail decisions.
    domain : str
        Logical monitor domain identifier.
    scenario_id : str | None
        Optional scenario identifier for evidence context.

    Returns
    -------
    SelfModelErrorResult
        SelfModelErrorResult with deterministic hash and audit payload.

    Raises
    ------
    ValueError
        If the observed or predicted inputs are invalid.
    """
    observed = _coerce_channel_matrix(observed_phases, name="observed_phases")
    predicted = _coerce_channel_matrix(predicted_phases, name="predicted_phases")
    if observed.shape != predicted.shape:
        raise ValueError(
            "observed_phases and predicted_phases must have matching shapes"
        )

    thresholds = SelfModelErrorThresholdConfig(
        tolerance=_require_finite_non_negative_float(tolerance, name="tolerance"),
        max_abs_tolerance=_require_finite_non_negative_float(
            max_abs_tolerance,
            name="max_abs_tolerance",
        ),
    )
    channel_count = int(observed.shape[0])
    sample_count = int(observed.shape[1])

    labels = _coerce_channel_labels(
        channel_labels,
        channel_count=channel_count,
    )
    weights = _coerce_channel_weights(
        channel_weights,
        channel_count=channel_count,
    )

    phase_errors = _wrapped_phase_errors(predicted, observed)
    channel_rmse = tuple(
        float(np.sqrt(np.mean(np.square(errors)))) for errors in phase_errors
    )
    channel_mae = tuple(float(np.mean(np.abs(errors))) for errors in phase_errors)
    channel_max_abs = tuple(float(np.max(np.abs(errors))) for errors in phase_errors)
    channel_breaches = tuple(
        rmse > thresholds.tolerance or max_abs > thresholds.max_abs_tolerance
        for rmse, max_abs in zip(channel_rmse, channel_max_abs, strict=True)
    )

    flattened = phase_errors.ravel()
    overall_rmse = float(np.sqrt(np.mean(np.square(flattened))))
    overall_mae = float(np.mean(np.abs(flattened)))
    overall_max_abs = float(np.max(np.abs(flattened)))

    weighted_rmse: float | None
    weighted_mae: float | None
    weighted_max_abs: float | None
    if weights is None:
        weighted_rmse = None
        weighted_mae = None
        weighted_max_abs = None
        weight_tuple: tuple[float, ...] | None = None
    else:
        normalized = _normalise_positive_weights(weights)
        weight_tuple = tuple(float(w) for w in weights.tolist())
        channel_rmse_array = np.asarray(channel_rmse, dtype=np.float64)
        channel_mae_array = np.asarray(channel_mae, dtype=np.float64)
        channel_max_abs_array = np.asarray(channel_max_abs, dtype=np.float64)
        normalized = normalized / np.sum(normalized)
        weighted_rmse = float(np.sqrt(np.sum(normalized * channel_rmse_array**2)))
        weighted_mae = float(np.sum(normalized * channel_mae_array))
        weighted_max_abs = float(np.max(normalized * channel_max_abs_array))

    breached = overall_rmse > thresholds.tolerance or (
        overall_max_abs > thresholds.max_abs_tolerance
    )

    order_rmse: float | None
    order_mae: float | None
    order_max_abs_error: float | None
    order_breached: bool | None
    if (observed_order is None) ^ (predicted_order is None):
        raise ValueError(
            "both observed_order and predicted_order must be provided together"
        )

    if observed_order is None:
        order_rmse = None
        order_mae = None
        order_max_abs_error = None
        order_breached = None
    else:
        obs_order = _coerce_order_vector(observed_order, name="observed_order")
        pred_order = _coerce_order_vector(predicted_order, name="predicted_order")
        if obs_order.shape != pred_order.shape:
            raise ValueError("observed_order and predicted_order shapes must match")
        if obs_order.shape[0] != channel_count:
            raise ValueError(
                "observed_order shape must match the number of observed phases channels"
            )
        order_errors = pred_order - obs_order
        order_rmse = float(np.sqrt(np.mean(np.square(order_errors))))
        order_mae = float(np.mean(np.abs(order_errors)))
        order_max_abs_error = float(np.max(np.abs(order_errors)))
        order_breached = (
            order_rmse > thresholds.tolerance
            or order_max_abs_error > thresholds.max_abs_tolerance
        )
        breached = breached or bool(order_breached)

    result_payload: dict[str, object] = {
        "domain": domain,
        "scenario_id": scenario_id,
        "backend": BACKEND,
        "channel_labels": list(labels),
        "channel_count": channel_count,
        "sample_count": sample_count,
        "channel_rmse": list(channel_rmse),
        "channel_mae": list(channel_mae),
        "channel_max_abs_error": list(channel_max_abs),
        "channel_breaches": list(channel_breaches),
        "channel_weights": None if weight_tuple is None else list(weight_tuple),
        "overall_rmse": overall_rmse,
        "overall_mae": overall_mae,
        "overall_max_abs_error": overall_max_abs,
        "weighted_rmse": weighted_rmse,
        "weighted_mae": weighted_mae,
        "weighted_max_abs_error": weighted_max_abs,
        "tolerance": thresholds.tolerance,
        "max_abs_tolerance": thresholds.max_abs_tolerance,
        "breached": breached,
        "order_rmse": order_rmse,
        "order_mae": order_mae,
        "order_max_abs_error": order_max_abs_error,
        "order_breached": order_breached,
        "claim_boundary": CLAIM_BOUNDARY,
        "non_actuating": True,
        "execution_disabled": True,
    }
    record_hash = _deterministic_record_hash(result_payload)

    return SelfModelErrorResult(
        domain=domain,
        scenario_id=scenario_id,
        channel_labels=labels,
        channel_count=channel_count,
        sample_count=sample_count,
        overall_rmse=overall_rmse,
        overall_mae=overall_mae,
        overall_max_abs_error=overall_max_abs,
        channel_rmse=tuple(float(v) for v in channel_rmse),
        channel_mae=tuple(float(v) for v in channel_mae),
        channel_max_abs_error=tuple(float(v) for v in channel_max_abs),
        channel_breaches=tuple(bool(v) for v in channel_breaches),
        weighted_rmse=weighted_rmse,
        weighted_mae=weighted_mae,
        weighted_max_abs_error=weighted_max_abs,
        channel_weights=weight_tuple,
        tolerance=thresholds.tolerance,
        max_abs_tolerance=thresholds.max_abs_tolerance,
        breached=breached,
        order_rmse=order_rmse,
        order_mae=order_mae,
        order_max_abs_error=order_max_abs_error,
        order_breached=order_breached,
        claim_boundary=CLAIM_BOUNDARY,
        non_actuating=True,
        execution_disabled=True,
        backend=BACKEND,
        record_hash=record_hash,
    )


def _require_finite_non_negative_float(value: object, *, name: str) -> float:
    """Return ``value`` as a finite non-negative float, else raise."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real value")
    float_value = float(value)
    if not np.isfinite(float_value):
        raise ValueError(f"{name} must be finite")
    if float_value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return float_value


def _coerce_channel_matrix(values: object, *, name: str) -> FloatArray:
    """Return the channel matrix as a validated 2-D finite array, else raise."""
    raw = np.asarray(values)
    if raw.dtype == np.bool_:
        raise ValueError(f"{name} must be numeric, got boolean values")

    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be convertible to a finite float array") from exc

    if array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim != 2:
        raise ValueError(f"{name} must be one-dimensional or two-dimensional")

    if array.size == 0 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one channel and one sample")
    if array.dtype == np.bool_:
        raise ValueError(f"{name} must be numeric")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _wrapped_phase_errors(predicted: FloatArray, observed: FloatArray) -> FloatArray:
    """Return the wrapped circular phase errors between channels."""
    return np.asarray(
        np.arctan2(np.sin(predicted - observed), np.cos(predicted - observed)),
        dtype=np.float64,
    )


def _coerce_order_vector(values: object, *, name: str) -> FloatArray:
    """Return the order vector as a validated finite array, else raise."""
    raw = np.asarray(values)
    if raw.dtype == np.bool_:
        raise ValueError(f"{name} must be numeric, got boolean values")

    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be convertible to a finite float vector"
        ) from exc

    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _coerce_channel_labels(
    channel_labels: object,
    *,
    channel_count: int,
) -> tuple[str, ...]:
    """Return validated non-empty channel labels, else raise."""
    if channel_labels is None:
        return tuple(f"channel_{idx}" for idx in range(channel_count))

    if not isinstance(channel_labels, (list, tuple)):
        raise ValueError("channel_labels must be a sequence of strings")

    labels = tuple(str(label) for label in channel_labels)
    if len(labels) != channel_count:
        raise ValueError(
            f"channel_labels length {len(labels)} does not match channel count "
            f"{channel_count}",
        )
    if any(len(label) == 0 for label in labels):
        raise ValueError("channel_labels must not contain empty values")
    return labels


def _coerce_channel_weights(
    channel_weights: object,
    *,
    channel_count: int,
) -> NDArray[np.float64] | None:
    """Return validated finite channel weights, else raise."""
    if channel_weights is None:
        return None

    raw = np.asarray(channel_weights)
    if raw.dtype == np.bool_:
        raise ValueError("channel_weights must be numeric, got boolean values")

    try:
        weights = np.asarray(channel_weights, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("channel_weights must be a numeric vector") from exc

    if weights.ndim != 1:
        raise ValueError("channel_weights must be a one-dimensional vector")
    if len(weights) == 0:
        raise ValueError("channel_weights must be non-empty")
    if len(weights) != channel_count:
        raise ValueError(
            "channel_weights length must match channel count",
        )
    if not np.all(np.isfinite(weights)):
        raise ValueError("channel_weights must contain finite values")
    if np.any(weights <= 0.0):
        raise ValueError("channel_weights must be strictly positive")
    return np.ascontiguousarray(weights, dtype=np.float64)


def _normalise_positive_weights(weights: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return positive weights normalised to sum to one, else raise."""
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("channel_weights sum must be finite and positive")
    return weights / total


def _deterministic_record_hash(record: dict[str, object]) -> str:
    """Return the canonical-JSON SHA-256 hash of a record."""
    payload = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()
