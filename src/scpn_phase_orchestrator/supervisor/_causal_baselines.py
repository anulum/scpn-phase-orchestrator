# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — causal trace and baseline calculations

"""Validated causal trace estimates and conventional baseline scores."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._array_types import require_real_values

if TYPE_CHECKING:
    from scpn_phase_orchestrator.supervisor.causal import CausalGraphEstimate

FloatArray: TypeAlias = NDArray[np.float64]


def _validate_causal_trace(
    trace: dict[str, list[float]],
    lag: int,
    min_abs_weight: float,
) -> dict[str, FloatArray]:
    """Validate the causal trace, else raise."""
    require_real_values(lag, name="lag")
    require_real_values(min_abs_weight, name="min_abs_weight")
    if isinstance(lag, bool) or int(lag) != lag or lag < 1:
        raise ValueError("lag must be a positive integer")
    if not np.isfinite(min_abs_weight) or min_abs_weight < 0.0:
        raise ValueError("min_abs_weight must be finite and non-negative")
    if len(trace) < 2:
        raise ValueError("trace must contain at least two signals")
    lengths = {len(values) for values in trace.values()}
    if len(lengths) != 1:
        raise ValueError("all trace signals must have equal length")
    length = lengths.pop()
    if length <= lag:
        raise ValueError("trace length must be greater than lag")
    arrays: dict[str, FloatArray] = {}
    for name, values in trace.items():
        data = _coerce_float_array(f"trace signal {name!r}", values)
        if data.ndim != 1:
            raise ValueError(f"trace signal {name!r} must be one-dimensional")
        if not np.all(np.isfinite(data)):
            raise ValueError(f"trace signal {name!r} contains NaN/Inf")
        arrays[name] = data
    return arrays


def _baseline_score(graph: CausalGraphEstimate) -> float:
    """Return the baseline causal score."""
    if not graph.edges:
        return 0.0
    return float(max(abs(edge.weight) * edge.confidence for edge in graph.edges))


def _causal_baseline_family(
    trace: dict[str, list[float]],
    *,
    lag: int,
    min_abs_weight: float,
    graph: CausalGraphEstimate,
) -> list[dict[str, float | int | str]]:
    """Return the family of baseline causal scores."""
    arrays = _validate_causal_trace(trace, lag, min_abs_weight)
    lagged_linear_score = _baseline_score(graph)
    records: list[dict[str, float | int | str]] = [
        {
            "name": "lagged_linear_graph",
            "score": lagged_linear_score,
            "edge_count": len(graph.edges),
            "description": "max_abs_lagged_linear_edge_weight_times_confidence",
        },
        _pairwise_correlation_baseline(
            arrays,
            lag=lag,
            min_abs_weight=min_abs_weight,
            name="lagged_pearson",
            description="max_abs_corr_source_t_target_t_plus_lag",
            use_delta=False,
        ),
        _pairwise_correlation_baseline(
            arrays,
            lag=lag,
            min_abs_weight=min_abs_weight,
            name="lagged_delta_pearson",
            description="max_abs_corr_source_t_target_delta_t_plus_lag",
            use_delta=True,
        ),
        _granger_residual_improvement_baseline(
            arrays,
            lag=lag,
            min_abs_weight=min_abs_weight,
        ),
        _target_persistence_baseline(
            arrays,
            lag=lag,
            min_abs_weight=min_abs_weight,
        ),
    ]
    return sorted(records, key=lambda record: str(record["name"]))


def _pairwise_correlation_baseline(
    arrays: dict[str, FloatArray],
    *,
    lag: int,
    min_abs_weight: float,
    name: str,
    description: str,
    use_delta: bool,
) -> dict[str, float | int | str]:
    """Return the pairwise-correlation baseline score."""
    max_score = 0.0
    edge_count = 0
    for source, source_values in arrays.items():
        source_window = source_values[:-lag]
        for target, target_values in arrays.items():
            if source == target:
                continue
            target_window = (
                target_values[lag:] - target_values[:-lag]
                if use_delta
                else target_values[lag:]
            )
            score = abs(_correlation(source_window, target_window))
            max_score = max(max_score, score)
            if score >= min_abs_weight:
                edge_count += 1
    return {
        "name": name,
        "score": float(max_score),
        "edge_count": edge_count,
        "description": description,
    }


def _granger_residual_improvement_baseline(
    arrays: dict[str, FloatArray],
    *,
    lag: int,
    min_abs_weight: float,
) -> dict[str, float | int | str]:
    """Return the Granger residual-improvement baseline score."""
    max_score = 0.0
    edge_count = 0
    for source, source_values in arrays.items():
        source_window = source_values[:-lag]
        for target, target_values in arrays.items():
            if source == target:
                continue
            autoregressive_window = target_values[:-lag]
            target_future = target_values[lag:]
            restricted_sse = _linear_residual_sse(
                autoregressive_window.reshape(-1, 1),
                target_future,
            )
            full_sse = _linear_residual_sse(
                np.column_stack((autoregressive_window, source_window)),
                target_future,
            )
            if restricted_sse <= 0.0:
                score = 0.0
            else:
                improvement = (restricted_sse - full_sse) / restricted_sse
                if not np.isfinite(improvement):
                    raise ValueError("causal baseline improvement must be finite")
                score = max(0.0, improvement)
            max_score = max(max_score, score)
            if score >= min_abs_weight:
                edge_count += 1
    return {
        "name": "granger_residual_improvement",
        "score": float(max_score),
        "edge_count": edge_count,
        "description": "max_fractional_sse_reduction_source_plus_target_history",
    }


def _target_persistence_baseline(
    arrays: dict[str, FloatArray],
    *,
    lag: int,
    min_abs_weight: float,
) -> dict[str, float | int | str]:
    """Return the target-persistence baseline score."""
    max_score = 0.0
    edge_count = 0
    for values in arrays.values():
        score = abs(_correlation(values[:-lag], values[lag:]))
        max_score = max(max_score, score)
        if score >= min_abs_weight:
            edge_count += 1
    return {
        "name": "target_persistence_null",
        "score": float(max_score),
        "edge_count": edge_count,
        "description": "max_abs_corr_target_t_target_t_plus_lag",
    }


def _linear_residual_sse(design: FloatArray, target: FloatArray) -> float:
    """Return residual SSE scaled by the target's squared maximum magnitude.

    Both regressions in the Granger comparison use the same target, so this
    scaling preserves their fractional SSE reduction without overflowing on
    finite, high-magnitude traces.
    """
    if design.shape[0] != target.shape[0]:
        raise ValueError("linear baseline design and target length mismatch")
    if not np.all(np.isfinite(design)) or not np.all(np.isfinite(target)):
        raise ValueError("causal baseline samples must be finite")
    target_scale = float(np.max(np.abs(target)))
    if target_scale == 0.0:
        return 0.0
    design_scale = np.max(np.abs(design), axis=0)
    design_scale = np.where(design_scale == 0.0, 1.0, design_scale)
    augmented = np.column_stack(
        (np.ones(design.shape[0], dtype=np.float64), design / design_scale)
    )
    scaled_target = target / target_scale
    coefficients, *_ = np.linalg.lstsq(augmented, scaled_target, rcond=None)
    residual = scaled_target - augmented @ coefficients
    sse = float(np.dot(residual, residual))
    if not np.isfinite(sse):
        raise ValueError("causal baseline residual error must be finite")
    return sse


def _coerce_float_array(name: str, value: object) -> FloatArray:
    """Return ``value`` as a validated finite float array, else raise."""
    raw = np.asarray(value, dtype=object)
    if any(isinstance(item, bool | np.bool_) for item in raw.ravel()):
        raise ValueError(f"{name} must not contain boolean values")
    if any(isinstance(item, complex | np.complexfloating) for item in raw.ravel()):
        raise ValueError(f"{name} must contain real-valued samples")
    try:
        require_real_values(value, name=name, allow_object=True)
        return np.ascontiguousarray(raw.astype(np.float64), dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric") from exc


def _lagged_linear_effect(
    source: FloatArray,
    target_delta: FloatArray,
) -> tuple[float, float]:
    """Return the lagged linear effect between two series."""
    source_centered = source - float(np.mean(source))
    target_centered = target_delta - float(np.mean(target_delta))
    source_scale = float(np.max(np.abs(source_centered)))
    target_scale = float(np.max(np.abs(target_centered)))
    if not np.isfinite(source_scale) or not np.isfinite(target_scale):
        raise ValueError("causal trace centred samples must be finite")
    if source_scale == 0.0 or target_scale == 0.0:
        return 0.0, 0.0
    source_unit = source_centered / source_scale
    target_unit = target_centered / target_scale
    source_var = float(np.dot(source_unit, source_unit))
    target_var = float(np.dot(target_unit, target_unit))
    covariance = float(np.dot(source_unit, target_unit))
    weight = (target_scale / source_scale) * (covariance / source_var)
    confidence_raw = abs(covariance) / float(np.sqrt(source_var * target_var))
    if not np.isfinite(weight) or not np.isfinite(confidence_raw):
        raise ValueError("causal trace influence must be finite")
    confidence = min(1.0, confidence_raw)
    return float(weight), float(confidence)


def _correlation(a: FloatArray, b: FloatArray) -> float:
    """Return the Pearson correlation between two series."""
    a_centered = a - float(np.mean(a))
    b_centered = b - float(np.mean(b))
    a_scale = float(np.max(np.abs(a_centered)))
    b_scale = float(np.max(np.abs(b_centered)))
    if not np.isfinite(a_scale) or not np.isfinite(b_scale):
        raise ValueError("causal baseline centred samples must be finite")
    if a_scale == 0.0 or b_scale == 0.0:
        return 0.0
    a_unit = a_centered / a_scale
    b_unit = b_centered / b_scale
    correlation = float(
        np.dot(a_unit, b_unit)
        / np.sqrt(np.dot(a_unit, a_unit) * np.dot(b_unit, b_unit))
    )
    if not np.isfinite(correlation):
        raise ValueError("causal baseline correlation must be finite")
    return correlation
