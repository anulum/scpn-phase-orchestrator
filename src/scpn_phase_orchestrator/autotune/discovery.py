# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — deterministic time-series discovery evidence

"""Deterministic evidence extraction for review-only auto-binding proposals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import cast

import numpy as np

from scpn_phase_orchestrator.autotune.sindy import PhaseSINDy
from scpn_phase_orchestrator.studio.workflow import JsonValue

__all__ = [
    "TimeSeriesDiscoveryConfig",
    "TimeSeriesDiscoveryReport",
    "discover_time_series_structure",
    "infer_sample_rate_from_time_column",
]

_TIME_COLUMNS = frozenset({"time", "timestamp", "t"})
_SINDY_LIBRARY = "affine_state_derivative"
_PHASE_SINDY_LIBRARY = "kuramoto_sine_phase_differences"
_PHASE_COLUMN_MARKERS = ("phase", "theta", "angle", "phi")


@dataclass(frozen=True, slots=True)
class TimeSeriesDiscoveryConfig:
    """Configuration for deterministic review evidence extraction."""

    correlation_threshold: float = 0.75
    sindy_threshold: float = 0.05
    phase_sindy_threshold: float = 0.05
    learned_graph_threshold: float = 0.2

    def __post_init__(self) -> None:
        if not 0.0 <= self.correlation_threshold <= 1.0:
            raise ValueError("correlation_threshold must be in [0, 1]")
        if self.sindy_threshold < 0.0 or not isfinite(self.sindy_threshold):
            raise ValueError("sindy_threshold must be finite and non-negative")
        if self.phase_sindy_threshold < 0.0 or not isfinite(self.phase_sindy_threshold):
            raise ValueError("phase_sindy_threshold must be finite and non-negative")
        if self.learned_graph_threshold < 0.0 or not isfinite(
            self.learned_graph_threshold
        ):
            raise ValueError("learned_graph_threshold must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class TimeSeriesDiscoveryReport:
    """JSON-ready discovery report for an imported time-series table."""

    sample_period_s: float
    sample_count: int
    columns: tuple[str, ...]
    sindy: Mapping[str, JsonValue]
    phase_sindy: Mapping[str, JsonValue]
    sindy_model_selection: Mapping[str, JsonValue]
    learned_graph: Mapping[str, JsonValue]
    correlation_graph: Mapping[str, JsonValue]
    clustering: Mapping[str, JsonValue]

    @property
    def sindy_sparsity(self) -> float:
        """Sparse-regression support fraction reported by the SINDy evidence."""
        return cast(float, self.sindy["sparsity"])

    @property
    def correlation_graph_density(self) -> float:
        """Density of the thresholded correlation graph evidence."""
        return cast(float, self.correlation_graph["density"])

    @property
    def cluster_coverage(self) -> float:
        """Fraction of channels covered by the largest discovered cluster."""
        if not self.columns:
            return 0.0
        largest_cluster_size = cast(int, self.clustering["largest_cluster_size"])
        return float(largest_cluster_size) / float(len(self.columns))

    @property
    def confidence_evidence(self) -> dict[str, float]:
        """Confidence factors derived from fitted discovery evidence blocks."""
        factors = {
            "sindy_sparsity": self.sindy_sparsity,
            "correlation_graph_density": self.correlation_graph_density,
            "cluster_coverage": self.cluster_coverage,
        }
        if self.phase_sindy.get("status") == "fitted":
            factors["phase_sindy_sparsity"] = cast(float, self.phase_sindy["sparsity"])
        if self.learned_graph.get("status") == "fitted":
            factors["learned_graph_density"] = cast(
                float, self.learned_graph["density"]
            )
        return factors

    def to_audit_record(self) -> dict[str, JsonValue]:
        """Return the complete JSON-safe discovery evidence record."""
        return {
            "sample_period_s": self.sample_period_s,
            "sample_count": self.sample_count,
            "columns": list(self.columns),
            "sindy": dict(self.sindy),
            "phase_sindy": dict(self.phase_sindy),
            "sindy_model_selection": dict(self.sindy_model_selection),
            "learned_graph": dict(self.learned_graph),
            "correlation_graph": dict(self.correlation_graph),
            "clustering": dict(self.clustering),
        }


def infer_sample_rate_from_time_column(
    rows: Sequence[Mapping[str, str]],
    fieldnames: Sequence[str],
) -> tuple[float, str]:
    """Infer a sampling rate from a regular finite time column."""

    time_column = next(
        (field for field in fieldnames if field.strip().lower() in _TIME_COLUMNS),
        None,
    )
    if time_column is None:
        raise ValueError("sample_rate_hz is required when CSV has no time column")
    if len(rows) < 2:
        raise ValueError("sample_rate_hz requires at least two timed samples")
    times: list[float] = []
    for row_index, row in enumerate(rows):
        try:
            value = float(row[time_column])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"time column {time_column!r} has non-numeric sample at row {row_index}"
            ) from exc
        if not isfinite(value):
            raise ValueError(
                f"time column {time_column!r} contains a non-finite sample"
            )
        times.append(value)
    deltas = np.diff(np.asarray(times, dtype=np.float64))
    if np.any(deltas <= 0.0):
        raise ValueError("time column must be strictly increasing")
    sample_period_s = float(np.median(deltas))
    if not np.allclose(deltas, sample_period_s, rtol=1e-6, atol=1e-12):
        raise ValueError("time column must use a regular sampling interval")
    if sample_period_s <= 0.0 or not isfinite(sample_period_s):
        raise ValueError("sample_rate_hz could not be inferred from time column")
    return 1.0 / sample_period_s, "time_column"


def discover_time_series_structure(
    samples: np.ndarray,
    *,
    columns: Sequence[str],
    sample_period_s: float,
    config: TimeSeriesDiscoveryConfig | None = None,
) -> TimeSeriesDiscoveryReport:
    """Extract sparse-derivative, graph, and cluster evidence from a table."""

    cfg = config or TimeSeriesDiscoveryConfig()
    table = np.asarray(samples, dtype=np.float64)
    if table.ndim != 2:
        raise ValueError("samples must be a 2-D table")
    if table.shape[0] < 2:
        raise ValueError("samples must contain at least two rows")
    if table.shape[1] != len(columns):
        raise ValueError("column count must match samples width")
    if table.shape[1] < 1:
        raise ValueError("samples must contain at least one signal column")
    if sample_period_s <= 0.0 or not isfinite(sample_period_s):
        raise ValueError("sample_period_s must be positive")
    if not np.all(np.isfinite(table)):
        raise ValueError("samples must contain only finite values")
    column_names = tuple(_normalised_column_name(column) for column in columns)
    correlation_graph = _correlation_graph(
        table,
        column_names,
        threshold=cfg.correlation_threshold,
    )
    clustering = _correlation_clusters(
        column_names,
        edges=correlation_graph["edges"],
    )
    sindy = _sparse_derivative_library(
        table,
        column_names,
        sample_period_s=sample_period_s,
        threshold=cfg.sindy_threshold,
    )
    phase_sindy = _phase_sindy_library(
        table,
        column_names,
        sample_period_s=sample_period_s,
        threshold=cfg.phase_sindy_threshold,
    )
    sindy_model_selection = _sindy_model_selection(
        sindy=sindy,
        phase_sindy=phase_sindy,
    )
    learned_graph = _lagged_learned_graph(
        table,
        column_names,
        threshold=cfg.learned_graph_threshold,
    )
    return TimeSeriesDiscoveryReport(
        sample_period_s=sample_period_s,
        sample_count=int(table.shape[0]),
        columns=column_names,
        sindy=sindy,
        phase_sindy=phase_sindy,
        sindy_model_selection=sindy_model_selection,
        learned_graph=learned_graph,
        correlation_graph=correlation_graph,
        clustering=clustering,
    )


def _normalised_column_name(column: str) -> str:
    name = str(column).strip()
    if not name:
        raise ValueError("signal column names must be non-empty")
    return name


def _correlation_graph(
    table: np.ndarray,
    columns: tuple[str, ...],
    *,
    threshold: float,
) -> dict[str, JsonValue]:
    edge_values: list[JsonValue] = []
    possible_edges = len(columns) * (len(columns) - 1) // 2
    for left in range(len(columns)):
        for right in range(left + 1, len(columns)):
            coefficient = _pearson(table[:, left], table[:, right])
            magnitude = abs(coefficient)
            if magnitude >= threshold:
                edge_values.append(
                    {
                        "source": columns[left],
                        "target": columns[right],
                        "correlation": coefficient,
                        "abs_correlation": magnitude,
                    }
                )
    density = 0.0 if possible_edges == 0 else len(edge_values) / possible_edges
    return {
        "threshold": threshold,
        "edge_count": len(edge_values),
        "possible_edges": possible_edges,
        "density": float(density),
        "edges": edge_values,
    }


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - float(np.mean(left))
    right_centered = right - float(np.mean(right))
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left_centered, right_centered) / denominator)


def _correlation_clusters(
    columns: tuple[str, ...],
    *,
    edges: JsonValue,
) -> dict[str, JsonValue]:
    adjacency: dict[str, set[str]] = {column: set() for column in columns}
    if isinstance(edges, list):
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            source = edge.get("source")
            target = edge.get("target")
            if isinstance(source, str) and isinstance(target, str):
                adjacency.setdefault(source, set()).add(target)
                adjacency.setdefault(target, set()).add(source)

    seen: set[str] = set()
    clusters: list[list[str]] = []
    for column in columns:
        if column in seen:
            continue
        stack = [column]
        component: list[str] = []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.append(current)
            for neighbour in sorted(adjacency.get(current, set()), reverse=True):
                if neighbour not in seen:
                    stack.append(neighbour)
        order = {name: index for index, name in enumerate(columns)}
        clusters.append(sorted(component, key=order.__getitem__))
    largest = max((len(cluster) for cluster in clusters), default=0)
    clusters_json: list[JsonValue] = [list(cluster) for cluster in clusters]
    return {
        "method": "correlation_connected_components",
        "cluster_count": len(clusters),
        "largest_cluster_size": largest,
        "clusters": clusters_json,
    }


def _sparse_derivative_library(
    table: np.ndarray,
    columns: tuple[str, ...],
    *,
    sample_period_s: float,
    threshold: float,
) -> dict[str, JsonValue]:
    features = _standardise(table)
    library = np.column_stack([np.ones(features.shape[0]), features])
    derivatives = np.gradient(features, sample_period_s, axis=0)
    coefficients, *_ = np.linalg.lstsq(library, derivatives, rcond=None)
    predictions = library @ coefficients
    active_mask = np.abs(coefficients) >= threshold
    active_terms = int(np.count_nonzero(active_mask))
    total_terms = int(coefficients.size)
    quality = _regression_quality(
        observations=derivatives,
        predictions=predictions,
        active_terms=active_terms,
    )
    sparsity = 1.0 if total_terms == 0 else 1.0 - active_terms / total_terms
    term_names = ("1", *columns)
    equations: list[JsonValue] = [
        _equation_for_target(
            target=columns[target_index],
            coefficients=coefficients[:, target_index],
            active=active_mask[:, target_index],
            term_names=term_names,
        )
        for target_index in range(len(columns))
    ]
    return {
        "library": _SINDY_LIBRARY,
        "threshold": threshold,
        "active_terms": active_terms,
        "total_terms": total_terms,
        "residual_rmse": quality["residual_rmse"],
        "score": quality["score"],
        "sparsity": float(max(0.0, min(1.0, sparsity))),
        "equations": equations,
    }


def _phase_sindy_library(
    table: np.ndarray,
    columns: tuple[str, ...],
    *,
    sample_period_s: float,
    threshold: float,
) -> dict[str, JsonValue]:
    if table.shape[1] < 2:
        return _skipped_phase_sindy("requires_at_least_two_phase_columns", threshold)
    if table.shape[0] < 3:
        return _skipped_phase_sindy("requires_at_least_three_samples", threshold)
    if not _is_phase_like_table(table, columns):
        return _skipped_phase_sindy("skipped_non_phase_like", threshold)

    sindy = PhaseSINDy(threshold=threshold)
    coefficients = sindy.fit(
        np.ascontiguousarray(table, dtype=np.float64), sample_period_s
    )
    equations: list[JsonValue] = list(sindy.get_equations())
    total_terms = int(sum(coefficient.size for coefficient in coefficients))
    active_terms = int(
        sum(
            np.count_nonzero(np.abs(coefficient) >= threshold)
            for coefficient in coefficients
        )
    )
    predictions = _phase_sindy_predictions(
        table,
        coefficients=coefficients,
    )
    derivatives = np.diff(np.unwrap(table, axis=0), axis=0) / sample_period_s
    quality = _regression_quality(
        observations=derivatives,
        predictions=predictions,
        active_terms=active_terms,
    )
    sparsity = 1.0 if total_terms == 0 else 1.0 - active_terms / total_terms
    coupling_edges = _phase_sindy_edges(
        columns,
        coefficients=coefficients,
        threshold=threshold,
    )
    return {
        "status": "fitted",
        "library": _PHASE_SINDY_LIBRARY,
        "threshold": threshold,
        "active_terms": active_terms,
        "total_terms": total_terms,
        "residual_rmse": quality["residual_rmse"],
        "score": quality["score"],
        "sparsity": float(max(0.0, min(1.0, sparsity))),
        "coupling_edge_count": len(coupling_edges),
        "coupling_edges": coupling_edges,
        "equations": equations,
    }


def _skipped_phase_sindy(reason: str, threshold: float) -> dict[str, JsonValue]:
    return {
        "status": reason,
        "library": _PHASE_SINDY_LIBRARY,
        "threshold": threshold,
        "active_terms": 0,
        "total_terms": 0,
        "residual_rmse": None,
        "score": None,
        "sparsity": 1.0,
        "coupling_edge_count": 0,
        "coupling_edges": [],
        "equations": [],
    }


def _is_phase_like_table(table: np.ndarray, columns: tuple[str, ...]) -> bool:
    if any(
        marker in column.strip().lower()
        for column in columns
        for marker in _PHASE_COLUMN_MARKERS
    ):
        return True
    ranges = np.ptp(table, axis=0)
    return bool(np.all(ranges <= 4.0 * np.pi))


def _phase_sindy_edges(
    columns: tuple[str, ...],
    *,
    coefficients: Sequence[np.ndarray],
    threshold: float,
) -> list[JsonValue]:
    edges: list[JsonValue] = []
    for target_index, coefficient in enumerate(coefficients):
        term_index = 1
        for source_index, source_column in enumerate(columns):
            if source_index == target_index:
                continue
            strength = float(coefficient[term_index])
            term_index += 1
            if abs(strength) < threshold:
                continue
            edges.append(
                {
                    "source": source_column,
                    "target": columns[target_index],
                    "coefficient": strength,
                    "abs_coefficient": abs(strength),
                }
            )
    return sorted(
        edges,
        key=lambda edge: (
            -cast(float, cast(dict[str, JsonValue], edge)["abs_coefficient"]),
            cast(str, cast(dict[str, JsonValue], edge)["source"]),
            cast(str, cast(dict[str, JsonValue], edge)["target"]),
        ),
    )


def _phase_sindy_predictions(
    table: np.ndarray,
    *,
    coefficients: Sequence[np.ndarray],
) -> np.ndarray:
    source = table[:-1, :]
    predictions = np.zeros_like(source, dtype=np.float64)
    for target_index, coefficient in enumerate(coefficients):
        predictions[:, target_index] = float(coefficient[0])
        term_index = 1
        for source_index in range(source.shape[1]):
            if source_index == target_index:
                continue
            predictions[:, target_index] += float(coefficient[term_index]) * np.sin(
                source[:, source_index] - source[:, target_index]
            )
            term_index += 1
    return predictions


def _regression_quality(
    *,
    observations: np.ndarray,
    predictions: np.ndarray,
    active_terms: int,
) -> dict[str, float]:
    residual = np.asarray(observations, dtype=np.float64) - np.asarray(
        predictions,
        dtype=np.float64,
    )
    sample_count = max(1, int(residual.size))
    residual_sum_squares = float(np.sum(residual * residual))
    residual_mse = max(residual_sum_squares / sample_count, np.finfo(float).tiny)
    residual_rmse = float(np.sqrt(residual_mse))
    score = float(
        sample_count * np.log(residual_mse) + active_terms * np.log(sample_count)
    )
    return {
        "residual_rmse": residual_rmse,
        "score": score,
    }


def _sindy_model_selection(
    *,
    sindy: Mapping[str, JsonValue],
    phase_sindy: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    candidates: list[JsonValue] = [
        _selection_candidate(_SINDY_LIBRARY, "fitted", sindy),
        _selection_candidate(
            _PHASE_SINDY_LIBRARY,
            str(phase_sindy["status"]),
            phase_sindy,
        ),
    ]
    fitted_candidates = [
        cast(dict[str, JsonValue], candidate)
        for candidate in candidates
        if cast(dict[str, JsonValue], candidate)["status"] == "fitted"
        and cast(dict[str, JsonValue], candidate)["score"] is not None
    ]
    selected = min(
        fitted_candidates,
        key=lambda candidate: (
            cast(float, candidate["score"]),
            -cast(float, candidate["sparsity"]),
            cast(str, candidate["library"]),
        ),
    )
    return {
        "method": "residual_bic_with_sparsity_tie_break",
        "candidate_count": len(candidates),
        "selected_library": cast(str, selected["library"]),
        "selected_score": cast(float, selected["score"]),
        "candidates": candidates,
    }


def _lagged_learned_graph(
    table: np.ndarray,
    columns: tuple[str, ...],
    *,
    threshold: float,
) -> dict[str, JsonValue]:
    if table.shape[1] < 2:
        return _skipped_learned_graph("requires_at_least_two_columns", threshold)
    if table.shape[0] < 3:
        return _skipped_learned_graph("requires_at_least_three_samples", threshold)

    features = _standardise(table)
    predictors = features[:-1, :]
    targets = features[1:, :]
    library = np.column_stack([np.ones(predictors.shape[0]), predictors])
    coefficients, *_ = np.linalg.lstsq(library, targets, rcond=None)
    predictions = library @ coefficients
    active_mask = np.abs(coefficients) >= threshold
    active_terms = int(np.count_nonzero(active_mask))
    total_terms = int(coefficients.size)
    quality = _regression_quality(
        observations=targets,
        predictions=predictions,
        active_terms=active_terms,
    )
    edges = _lagged_graph_edges(
        columns,
        coefficients=coefficients,
        threshold=threshold,
    )
    possible_edges = len(columns) * (len(columns) - 1)
    density = 0.0 if possible_edges == 0 else len(edges) / possible_edges
    return {
        "status": "fitted",
        "method": "lagged_sparse_linear_prediction",
        "threshold": threshold,
        "active_terms": active_terms,
        "total_terms": total_terms,
        "residual_rmse": quality["residual_rmse"],
        "score": quality["score"],
        "edge_count": len(edges),
        "possible_edges": possible_edges,
        "density": float(density),
        "edges": edges,
    }


def _skipped_learned_graph(reason: str, threshold: float) -> dict[str, JsonValue]:
    return {
        "status": reason,
        "method": "lagged_sparse_linear_prediction",
        "threshold": threshold,
        "active_terms": 0,
        "total_terms": 0,
        "residual_rmse": None,
        "score": None,
        "edge_count": 0,
        "possible_edges": 0,
        "density": 0.0,
        "edges": [],
    }


def _lagged_graph_edges(
    columns: tuple[str, ...],
    *,
    coefficients: np.ndarray,
    threshold: float,
) -> list[JsonValue]:
    edges: list[JsonValue] = []
    for target_index, target_column in enumerate(columns):
        for source_index, source_column in enumerate(columns):
            if source_index == target_index:
                continue
            coefficient = float(coefficients[source_index + 1, target_index])
            if abs(coefficient) < threshold:
                continue
            edges.append(
                {
                    "source": source_column,
                    "target": target_column,
                    "lag": 1,
                    "coefficient": coefficient,
                    "abs_coefficient": abs(coefficient),
                }
            )
    return sorted(
        edges,
        key=lambda edge: (
            -cast(float, cast(dict[str, JsonValue], edge)["abs_coefficient"]),
            cast(str, cast(dict[str, JsonValue], edge)["source"]),
            cast(str, cast(dict[str, JsonValue], edge)["target"]),
        ),
    )


def _selection_candidate(
    library: str,
    status: str,
    payload: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    return {
        "library": library,
        "status": status,
        "active_terms": cast(int, payload["active_terms"]),
        "total_terms": cast(int, payload["total_terms"]),
        "sparsity": cast(float, payload["sparsity"]),
        "residual_rmse": cast(float | None, payload.get("residual_rmse")),
        "score": cast(float | None, payload.get("score")),
    }


def _standardise(table: np.ndarray) -> np.ndarray:
    centre = np.mean(table, axis=0)
    scale = np.std(table, axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    return cast(np.ndarray, (table - centre) / scale)


def _equation_for_target(
    *,
    target: str,
    coefficients: np.ndarray,
    active: np.ndarray,
    term_names: Sequence[str],
) -> str:
    terms: list[str] = []
    for coefficient, is_active, term_name in zip(
        coefficients,
        active,
        term_names,
        strict=True,
    ):
        if not bool(is_active):
            continue
        terms.append(f"{float(coefficient):+.6g}*{term_name}")
    rhs = " ".join(terms) if terms else "0"
    return f"d({target})/dt = {rhs}"
