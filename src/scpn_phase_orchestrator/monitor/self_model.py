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
from typing import Final, cast

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

    def __post_init__(self) -> None:
        """Validate and canonicalise the frozen threshold configuration."""
        object.__setattr__(
            self,
            "tolerance",
            _require_finite_non_negative_float(self.tolerance, name="tolerance"),
        )
        object.__setattr__(
            self,
            "max_abs_tolerance",
            _require_finite_non_negative_float(
                self.max_abs_tolerance, name="max_abs_tolerance"
            ),
        )
        if self.order_tolerance is not None:
            object.__setattr__(
                self,
                "order_tolerance",
                _require_finite_non_negative_float(
                    self.order_tolerance, name="order_tolerance"
                ),
            )
        if self.order_max_abs_tolerance is not None:
            object.__setattr__(
                self,
                "order_max_abs_tolerance",
                _require_finite_non_negative_float(
                    self.order_max_abs_tolerance,
                    name="order_max_abs_tolerance",
                ),
            )


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
    order_tolerance: float
    order_max_abs_tolerance: float
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

    def __post_init__(self) -> None:
        """Replay the frozen result's structural and derived evidence."""
        _validate_self_model_error_result(self)

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe audit record for the computed monitor output.

        Returns
        -------
        dict[str, object]
            Return a JSON-safe audit record for the computed monitor output.
        """
        record = _result_payload(self)
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
    order_tolerance: float | None = None,
    order_max_abs_tolerance: float | None = None,
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
    order_tolerance : float | None
        Optional order-signal RMSE threshold; defaults to ``tolerance``.
    order_max_abs_tolerance : float | None
        Optional order-signal max-abs threshold; defaults to
        ``max_abs_tolerance``.
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
        order_tolerance=order_tolerance,
        order_max_abs_tolerance=order_max_abs_tolerance,
    )
    order_rmse_tolerance = (
        thresholds.tolerance
        if thresholds.order_tolerance is None
        else thresholds.order_tolerance
    )
    order_max_tolerance = (
        thresholds.max_abs_tolerance
        if thresholds.order_max_abs_tolerance is None
        else thresholds.order_max_abs_tolerance
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
    canonical_domain = _require_canonical_identity(domain, name="domain")
    canonical_scenario = _require_optional_canonical_identity(
        scenario_id, name="scenario_id"
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
            order_rmse > order_rmse_tolerance
            or order_max_abs_error > order_max_tolerance
        )
        breached = breached or bool(order_breached)

    result_payload: dict[str, object] = {
        "domain": canonical_domain,
        "scenario_id": canonical_scenario,
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
        "order_tolerance": order_rmse_tolerance,
        "order_max_abs_tolerance": order_max_tolerance,
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
        domain=canonical_domain,
        scenario_id=canonical_scenario,
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
        order_tolerance=order_rmse_tolerance,
        order_max_abs_tolerance=order_max_tolerance,
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
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite real value")
    try:
        float_value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real value") from exc
    if not np.isfinite(float_value):
        raise ValueError(f"{name} must be finite")
    if float_value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return float_value


def _coerce_channel_matrix(values: object, *, name: str) -> FloatArray:
    """Return the channel matrix as a validated 2-D finite array, else raise."""
    array = _coerce_real_array(
        values,
        name=name,
        expected="real float array convertible to a finite float array",
    )

    if array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim != 2:
        raise ValueError(f"{name} must be one-dimensional or two-dimensional")

    if array.size == 0 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one channel and one sample")
    return array


def _wrapped_phase_errors(predicted: FloatArray, observed: FloatArray) -> FloatArray:
    """Return the wrapped circular phase errors between channels."""
    return np.asarray(
        np.arctan2(np.sin(predicted - observed), np.cos(predicted - observed)),
        dtype=np.float64,
    )


def _coerce_order_vector(values: object, *, name: str) -> FloatArray:
    """Return the order vector as a validated finite array, else raise."""
    array = _coerce_real_array(
        values,
        name=name,
        expected="real float vector convertible to a finite float vector",
    )

    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    return array


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

    if any(type(label) is not str for label in channel_labels):
        raise ValueError("channel_labels must contain only strings")
    labels = tuple(channel_labels)
    if len(labels) != channel_count:
        raise ValueError(
            f"channel_labels length {len(labels)} does not match channel count "
            f"{channel_count}",
        )
    if any(not label.strip() for label in labels):
        raise ValueError("channel_labels must not contain empty values")
    if any(label != label.strip() for label in labels):
        raise ValueError("channel_labels must contain canonical trimmed values")
    if len(set(labels)) != len(labels):
        raise ValueError("channel_labels must contain unique values")
    return labels


def _coerce_channel_weights(
    channel_weights: object,
    *,
    channel_count: int,
) -> NDArray[np.float64] | None:
    """Return validated finite channel weights, else raise."""
    if channel_weights is None:
        return None

    weights = _coerce_real_array(
        channel_weights,
        name="channel_weights",
        expected="numeric vector; expected a real float vector",
    )

    if weights.ndim != 1:
        raise ValueError("channel_weights must be a one-dimensional vector")
    if len(weights) == 0:
        raise ValueError("channel_weights must be non-empty")
    if len(weights) != channel_count:
        raise ValueError(
            "channel_weights length must match channel count",
        )
    if np.any(weights <= 0.0):
        raise ValueError("channel_weights must be strictly positive")
    return np.ascontiguousarray(weights, dtype=np.float64)


def _coerce_real_array(
    value: object,
    *,
    name: str,
    expected: str,
) -> FloatArray:
    """Return a copied finite non-coercive real array, else raise."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError, RuntimeError) as exc:
        raise ValueError(f"{name} must be a {expected}") from exc
    if np.issubdtype(raw.dtype, np.bool_):
        raise ValueError(f"{name} must be numeric, got boolean values")
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be a {expected}")
    if raw.dtype == np.dtype("O"):
        if not all(
            isinstance(item, (Real, np.integer, np.floating))
            and not isinstance(item, (bool, np.bool_))
            for item in raw.flat
        ):
            raise ValueError(f"{name} must be a {expected}")
    elif raw.dtype.kind not in {"f", "i", "u"}:
        raise ValueError(f"{name} must be a {expected}")
    try:
        array = np.array(raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a {expected}") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _require_canonical_identity(value: object, *, name: str) -> str:
    """Return one non-empty canonical evidence identifier, else raise."""
    if type(value) is not str or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical string")
    return value


def _require_optional_canonical_identity(value: object, *, name: str) -> str | None:
    """Return an optional canonical evidence identifier, else raise."""
    if value is None:
        return None
    return _require_canonical_identity(value, name=name)


def _normalise_positive_weights(weights: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return positive weights normalised to sum to one, else raise."""
    with np.errstate(over="ignore", invalid="ignore"):
        total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("channel_weights sum must be finite and positive")
    return weights / total


def _result_payload(result: SelfModelErrorResult) -> dict[str, object]:
    """Return the canonical result payload without its derived record hash."""
    return {
        "domain": result.domain,
        "scenario_id": result.scenario_id,
        "backend": result.backend,
        "channel_labels": list(result.channel_labels),
        "channel_count": result.channel_count,
        "sample_count": result.sample_count,
        "channel_rmse": list(result.channel_rmse),
        "channel_mae": list(result.channel_mae),
        "channel_max_abs_error": list(result.channel_max_abs_error),
        "channel_breaches": list(result.channel_breaches),
        "channel_weights": None
        if result.channel_weights is None
        else list(result.channel_weights),
        "overall_rmse": result.overall_rmse,
        "overall_mae": result.overall_mae,
        "overall_max_abs_error": result.overall_max_abs_error,
        "weighted_rmse": result.weighted_rmse,
        "weighted_mae": result.weighted_mae,
        "weighted_max_abs_error": result.weighted_max_abs_error,
        "tolerance": result.tolerance,
        "max_abs_tolerance": result.max_abs_tolerance,
        "order_tolerance": result.order_tolerance,
        "order_max_abs_tolerance": result.order_max_abs_tolerance,
        "breached": result.breached,
        "order_rmse": result.order_rmse,
        "order_mae": result.order_mae,
        "order_max_abs_error": result.order_max_abs_error,
        "order_breached": result.order_breached,
        "claim_boundary": result.claim_boundary,
        "non_actuating": result.non_actuating,
        "execution_disabled": result.execution_disabled,
    }


def _require_canonical_positive_int(value: object, *, name: str) -> int:
    """Return one positive built-in integer, else raise."""
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive canonical integer")
    return value


def _require_canonical_non_negative_float(value: object, *, name: str) -> float:
    """Return one finite non-negative built-in float, else raise."""
    if type(value) is not float or not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative canonical float")
    return value


def _require_replayed_float(actual: float, expected: float, *, name: str) -> None:
    """Require one derived float to replay its canonical equation."""
    if not np.isclose(actual, expected, rtol=1.0e-12, atol=1.0e-15):
        raise ValueError(f"{name} does not replay from channel evidence")


def _validate_metric_tuple(
    values: object,
    *,
    name: str,
    channel_count: int,
) -> tuple[float, ...]:
    """Return a canonical non-negative per-channel metric tuple."""
    if type(values) is not tuple or len(values) != channel_count:
        raise ValueError(f"{name} must be a canonical channel-length tuple")
    for value in values:
        _require_canonical_non_negative_float(value, name=name)
    return values


def _validate_self_model_error_result(result: SelfModelErrorResult) -> None:
    """Validate and replay one directly constructed self-model error result."""
    _require_canonical_identity(result.domain, name="domain")
    _require_optional_canonical_identity(result.scenario_id, name="scenario_id")
    channel_count = _require_canonical_positive_int(
        result.channel_count, name="channel_count"
    )
    _require_canonical_positive_int(result.sample_count, name="sample_count")
    if type(result.channel_labels) is not tuple:
        raise ValueError("channel_labels must be a canonical tuple")
    _coerce_channel_labels(result.channel_labels, channel_count=channel_count)

    channel_rmse = _validate_metric_tuple(
        result.channel_rmse,
        name="channel_rmse",
        channel_count=channel_count,
    )
    channel_mae = _validate_metric_tuple(
        result.channel_mae,
        name="channel_mae",
        channel_count=channel_count,
    )
    channel_max = _validate_metric_tuple(
        result.channel_max_abs_error,
        name="channel_max_abs_error",
        channel_count=channel_count,
    )
    if any(
        mae > maximum or rmse > maximum
        for rmse, mae, maximum in zip(
            channel_rmse, channel_mae, channel_max, strict=True
        )
    ):
        raise ValueError("channel metrics violate max-absolute bounds")

    tolerance = _require_canonical_non_negative_float(
        result.tolerance, name="tolerance"
    )
    max_abs_tolerance = _require_canonical_non_negative_float(
        result.max_abs_tolerance, name="max_abs_tolerance"
    )
    order_tolerance = _require_canonical_non_negative_float(
        result.order_tolerance, name="order_tolerance"
    )
    order_max_abs_tolerance = _require_canonical_non_negative_float(
        result.order_max_abs_tolerance,
        name="order_max_abs_tolerance",
    )

    expected_channel_breaches = tuple(
        rmse > tolerance or maximum > max_abs_tolerance
        for rmse, maximum in zip(channel_rmse, channel_max, strict=True)
    )
    if type(result.channel_breaches) is not tuple or any(
        type(value) is not bool for value in result.channel_breaches
    ):
        raise ValueError("channel_breaches must contain canonical booleans")
    if result.channel_breaches != expected_channel_breaches:
        raise ValueError("channel_breaches do not replay threshold decisions")

    overall_rmse = _require_canonical_non_negative_float(
        result.overall_rmse, name="overall_rmse"
    )
    overall_mae = _require_canonical_non_negative_float(
        result.overall_mae, name="overall_mae"
    )
    overall_max = _require_canonical_non_negative_float(
        result.overall_max_abs_error, name="overall_max_abs_error"
    )
    _require_replayed_float(
        overall_rmse,
        float(np.sqrt(np.mean(np.square(channel_rmse)))),
        name="overall_rmse",
    )
    _require_replayed_float(
        overall_mae,
        float(np.mean(channel_mae)),
        name="overall_mae",
    )
    _require_replayed_float(overall_max, max(channel_max), name="overall_max_abs_error")

    weighted_values = (
        result.weighted_rmse,
        result.weighted_mae,
        result.weighted_max_abs_error,
    )
    if result.channel_weights is None:
        if any(value is not None for value in weighted_values):
            raise ValueError("weighted metrics require channel_weights")
    else:
        if type(result.channel_weights) is not tuple:
            raise ValueError("channel_weights must be a canonical tuple")
        weights = _validate_metric_tuple(
            result.channel_weights,
            name="channel_weights",
            channel_count=channel_count,
        )
        if any(weight <= 0.0 for weight in weights):
            raise ValueError("channel_weights must be strictly positive")
        if any(value is None for value in weighted_values):
            raise ValueError("channel_weights require all weighted metrics")
        normalized = _normalise_positive_weights(np.asarray(weights, dtype=np.float64))
        expected_weighted = (
            float(np.sqrt(np.sum(normalized * np.square(channel_rmse)))),
            float(np.sum(normalized * np.asarray(channel_mae))),
            float(np.max(normalized * np.asarray(channel_max))),
        )
        for name, actual, expected in zip(
            (
                "weighted_rmse",
                "weighted_mae",
                "weighted_max_abs_error",
            ),
            weighted_values,
            expected_weighted,
            strict=True,
        ):
            canonical_actual = _require_canonical_non_negative_float(
                cast(float, actual), name=name
            )
            _require_replayed_float(canonical_actual, expected, name=name)

    order_values = (
        result.order_rmse,
        result.order_mae,
        result.order_max_abs_error,
    )
    if all(value is None for value in order_values):
        if result.order_breached is not None:
            raise ValueError("order_breached requires order metrics")
        expected_order_breach = False
    else:
        if any(value is None for value in order_values):
            raise ValueError("order metrics must be provided together")
        order_rmse, order_mae, order_max = (
            _require_canonical_non_negative_float(value, name=name)
            for value, name in zip(
                order_values,
                ("order_rmse", "order_mae", "order_max_abs_error"),
                strict=True,
            )
        )
        if order_mae > order_max or order_rmse > order_max:
            raise ValueError("order metrics violate max-absolute bounds")
        expected_order_breach = (
            order_rmse > order_tolerance or order_max > order_max_abs_tolerance
        )
        if type(result.order_breached) is not bool:
            raise ValueError("order_breached must be a canonical boolean")
        if result.order_breached is not expected_order_breach:
            raise ValueError("order_breached does not replay threshold decisions")

    if type(result.breached) is not bool:
        raise ValueError("breached must be a canonical boolean")
    expected_breach = (
        overall_rmse > tolerance
        or overall_max > max_abs_tolerance
        or expected_order_breach
    )
    if result.breached is not expected_breach:
        raise ValueError("breached does not replay aggregate threshold decisions")
    if result.claim_boundary != CLAIM_BOUNDARY:
        raise ValueError("claim_boundary must preserve the non-actuating boundary")
    if result.non_actuating is not True or result.execution_disabled is not True:
        raise ValueError("self-model results must remain non-actuating and disabled")
    if result.backend != BACKEND:
        raise ValueError("backend must identify the canonical NumPy reference")
    if (
        type(result.record_hash) is not str
        or len(result.record_hash) != 64
        or any(character not in "0123456789abcdef" for character in result.record_hash)
    ):
        raise ValueError("record_hash must be a lowercase SHA-256 digest")
    if result.record_hash != _deterministic_record_hash(_result_payload(result)):
        raise ValueError("record_hash does not match the canonical result payload")


def _deterministic_record_hash(record: dict[str, object]) -> str:
    """Return the canonical-JSON SHA-256 hash of a record."""
    payload = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()
