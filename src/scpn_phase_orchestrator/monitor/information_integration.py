# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Integrated-information monitor

"""Approximate integrated-information monitor for phase trajectories.

The monitor reports a bounded engineering proxy over binned circular
mutual information. It is intended for regime comparison and audit
traces, not for theoretical integrated-information claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from numbers import Integral, Real
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "IntegratedInformationBenchmarkCase",
    "IntegratedInformationBenchmarkReport",
    "IntegratedInformationResult",
    "benchmark_integrated_information_approximations",
    "integrated_information",
]

FloatArray: TypeAlias = NDArray[np.float64]
Partition: TypeAlias = tuple[tuple[int, ...], tuple[int, ...]]

_DEFAULT_BINS = 16
_TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class IntegratedInformationResult:
    """Audit-ready result from the integrated-information monitor.

    Attributes:
        phi: Minimum cross-partition information in nats. This is an
            approximate Phi-style proxy, not an exact IIT quantity.
        normalised_phi: ``phi`` divided by ``log(n_bins)`` and clipped
            to ``[0, 1]`` for dashboards.
        total_integration: Mean off-diagonal pairwise mutual information
            across all oscillator trajectories.
        minimum_partition: Bipartition that minimises cross-partition
            information.
        pairwise_mi: Symmetric pairwise mutual-information matrix.
        n_bins: Number of circular histogram bins used by the estimator.
    """

    phi: float
    normalised_phi: float
    total_integration: float
    minimum_partition: Partition
    pairwise_mi: FloatArray
    n_bins: int

    def __post_init__(self) -> None:
        n_bins = _validate_bins(self.n_bins)
        pairwise_mi = _validate_pairwise_mi(self.pairwise_mi)
        partition = _validate_partition(
            self.minimum_partition,
            n_oscillators=int(pairwise_mi.shape[0]),
        )
        phi = _validate_non_negative_scalar(self.phi, name="phi")
        normalised_phi = _validate_unit_interval_scalar(
            self.normalised_phi,
            name="normalised_phi",
        )
        total_integration = _validate_non_negative_scalar(
            self.total_integration,
            name="total_integration",
        )
        if phi > total_integration + 1e-12:
            raise ValueError("phi must not exceed total_integration")
        object.__setattr__(self, "phi", phi)
        object.__setattr__(self, "normalised_phi", normalised_phi)
        object.__setattr__(self, "total_integration", total_integration)
        object.__setattr__(self, "minimum_partition", partition)
        object.__setattr__(self, "pairwise_mi", pairwise_mi)
        object.__setattr__(self, "n_bins", n_bins)

    def to_audit_record(self) -> dict[str, Any]:
        """Return a JSON-serialisable audit record."""
        left, right = self.minimum_partition
        return {
            "monitor": "integrated_information",
            "phi": self.phi,
            "normalised_phi": self.normalised_phi,
            "total_integration": self.total_integration,
            "minimum_partition": [list(left), list(right)],
            "pairwise_mi": self.pairwise_mi.tolist(),
            "n_bins": self.n_bins,
            "method": "binned_circular_pairwise_minimum_bipartition",
            "claim_boundary": "engineering_proxy_not_theoretical_iit",
        }


@dataclass(frozen=True)
class IntegratedInformationBenchmarkCase:
    """Deterministic approximation benchmark case for the Phi proxy."""

    name: str
    description: str
    result: IntegratedInformationResult

    def to_audit_record(self) -> dict[str, Any]:
        """Return a JSON-serialisable benchmark case record."""
        return {
            "name": self.name,
            "description": self.description,
            "result": self.result.to_audit_record(),
        }


@dataclass(frozen=True)
class IntegratedInformationBenchmarkReport:
    """Audit report for deterministic integrated-information approximations."""

    cases: tuple[IntegratedInformationBenchmarkCase, ...]
    expected_ordering_passed: bool
    locked_phi_margin: float
    modular_total_margin: float
    noisy_lock_phi_margin: float
    phase_lag_total_margin: float
    n_samples: int
    n_bins: int

    def to_audit_record(self) -> dict[str, Any]:
        """Return a JSON-serialisable benchmark report."""
        return {
            "monitor": "integrated_information",
            "benchmark": "deterministic_synthetic_approximation_cases",
            "n_samples": self.n_samples,
            "n_bins": self.n_bins,
            "expected_ordering_passed": self.expected_ordering_passed,
            "locked_phi_margin": self.locked_phi_margin,
            "modular_total_margin": self.modular_total_margin,
            "noisy_lock_phi_margin": self.noisy_lock_phi_margin,
            "phase_lag_total_margin": self.phase_lag_total_margin,
            "cases": [case.to_audit_record() for case in self.cases],
            "claim_boundary": "engineering_proxy_not_theoretical_iit",
        }


def integrated_information(
    phase_series: FloatArray,
    n_bins: int = _DEFAULT_BINS,
) -> IntegratedInformationResult:
    """Estimate an approximate integrated-information metric.

    Args:
        phase_series: Phase trajectory array with shape
            ``(n_oscillators, n_samples)``. Values are wrapped onto the
            circular interval before histogramming.
        n_bins: Number of circular bins for mutual-information
            estimation. Must be at least two.

    Returns:
        ``IntegratedInformationResult`` containing the minimum
        information bipartition and audit fields.

    Raises:
        ValueError: If the trajectory is not a finite two-dimensional
            array with at least two oscillators and two samples.
    """
    phases = _validate_phase_series(phase_series)
    bins = _validate_bins(n_bins)

    pairwise_mi = _pairwise_mi_matrix(phases, bins)
    total_integration = _mean_off_diagonal(pairwise_mi)
    minimum_partition, phi = _minimum_bipartition(pairwise_mi)
    normalised_phi = _normalise_phi(phi, bins)

    return IntegratedInformationResult(
        phi=phi,
        normalised_phi=normalised_phi,
        total_integration=total_integration,
        minimum_partition=minimum_partition,
        pairwise_mi=pairwise_mi,
        n_bins=bins,
    )


def benchmark_integrated_information_approximations(
    *,
    n_samples: int = 256,
    n_bins: int = 8,
) -> IntegratedInformationBenchmarkReport:
    """Run deterministic approximation checks for the Phi proxy.

    This is a numerical calibration, not a hardware performance benchmark. It
    checks five synthetic regimes: independent streams, modular streams with
    high within-module information but weak cross-module Phi, phase-lagged
    chains, noisy globally locked streams, and globally locked streams with high
    cross-partition Phi.
    """
    if n_samples < 32:
        raise ValueError("n_samples must be at least 32")
    bins = _validate_bins(n_bins)
    cases = (
        IntegratedInformationBenchmarkCase(
            name="independent",
            description="seeded independent circular phase streams",
            result=integrated_information(
                _independent_benchmark_series(n_samples), bins
            ),
        ),
        IntegratedInformationBenchmarkCase(
            name="modular",
            description=(
                "two internally locked modules with weak cross-module "
                "minimum-partition Phi"
            ),
            result=integrated_information(_modular_benchmark_series(n_samples), bins),
        ),
        IntegratedInformationBenchmarkCase(
            name="phase_lag_chain",
            description="deterministic phase-lagged chain with coherent offsets",
            result=integrated_information(
                _phase_lag_chain_benchmark_series(n_samples), bins
            ),
        ),
        IntegratedInformationBenchmarkCase(
            name="noisy_locked",
            description="globally locked streams with deterministic phase noise",
            result=integrated_information(
                _noisy_locked_benchmark_series(n_samples), bins
            ),
        ),
        IntegratedInformationBenchmarkCase(
            name="locked",
            description="globally phase-locked streams with high cross-partition Phi",
            result=integrated_information(_locked_benchmark_series(n_samples), bins),
        ),
    )
    by_name = {case.name: case.result for case in cases}
    locked_phi_margin = by_name["locked"].phi - by_name["independent"].phi
    modular_total_margin = (
        by_name["modular"].total_integration - by_name["independent"].total_integration
    )
    noisy_lock_phi_margin = by_name["noisy_locked"].phi - by_name["independent"].phi
    phase_lag_total_margin = (
        by_name["phase_lag_chain"].total_integration
        - by_name["independent"].total_integration
    )
    expected_ordering_passed = (
        locked_phi_margin > 0.0
        and modular_total_margin > 0.0
        and noisy_lock_phi_margin > 0.0
        and phase_lag_total_margin > 0.0
        and by_name["locked"].phi > by_name["modular"].phi
        and by_name["locked"].phi > by_name["noisy_locked"].phi
    )
    return IntegratedInformationBenchmarkReport(
        cases=cases,
        expected_ordering_passed=expected_ordering_passed,
        locked_phi_margin=float(locked_phi_margin),
        modular_total_margin=float(modular_total_margin),
        noisy_lock_phi_margin=float(noisy_lock_phi_margin),
        phase_lag_total_margin=float(phase_lag_total_margin),
        n_samples=n_samples,
        n_bins=bins,
    )


def _validate_phase_series(phase_series: FloatArray) -> FloatArray:
    if _contains_boolean_alias(phase_series):
        msg = "phase_series must not contain boolean values"
        raise ValueError(msg)
    raw = np.asarray(phase_series)
    phases = raw.astype(np.float64, copy=True)
    if phases.ndim != 2:
        msg = "phase_series must have shape (n_oscillators, n_samples)"
        raise ValueError(msg)
    n_oscillators, n_samples = phases.shape
    if n_oscillators < 2:
        msg = "phase_series must contain at least two oscillators"
        raise ValueError(msg)
    if n_samples < 2:
        msg = "phase_series must contain at least two samples"
        raise ValueError(msg)
    if not np.all(np.isfinite(phases)):
        msg = "phase_series must contain only finite values"
        raise ValueError(msg)
    return phases


def _validate_bins(n_bins: int) -> int:
    if isinstance(n_bins, bool) or not isinstance(n_bins, Integral):
        msg = "n_bins must be an integer"
        raise ValueError(msg)
    bins = int(n_bins)
    if bins < 2:
        msg = "n_bins must be at least 2"
        raise ValueError(msg)
    return bins


def _contains_boolean_alias(value: object) -> bool:
    raw = np.asarray(value, dtype=object)
    if raw.dtype == np.bool_:
        return True
    return any(isinstance(item, bool) for item in raw.flat)


def _validate_non_negative_scalar(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite non-negative real")
    scalar = float(value)
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return scalar


def _validate_unit_interval_scalar(value: object, *, name: str) -> float:
    scalar = _validate_non_negative_scalar(value, name=name)
    if scalar > 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return scalar


def _validate_pairwise_mi(value: object) -> FloatArray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("pairwise_mi must be a numeric matrix") from exc
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("pairwise_mi must be a square matrix")
    if matrix.shape[0] < 2:
        raise ValueError("pairwise_mi must contain at least two oscillators")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("pairwise_mi must contain only finite values")
    if np.any(matrix < -1e-12):
        raise ValueError("pairwise_mi must be non-negative")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12):
        raise ValueError("pairwise_mi must be symmetric")
    if not np.allclose(np.diag(matrix), 0.0, rtol=0.0, atol=1e-12):
        raise ValueError("pairwise_mi diagonal must be zero")
    matrix = np.maximum(matrix, 0.0)
    np.fill_diagonal(matrix, 0.0)
    return np.ascontiguousarray(matrix, dtype=np.float64)


def _validate_partition(value: object, *, n_oscillators: int) -> Partition:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("minimum_partition must contain two index groups")
    left_raw = value[0]
    right_raw = value[1]
    left = _validate_partition_side(left_raw, name="minimum_partition")
    right = _validate_partition_side(right_raw, name="minimum_partition")
    if not left or not right:
        raise ValueError("minimum_partition groups must be non-empty")
    if set(left).intersection(right):
        raise ValueError("minimum_partition groups must be disjoint")
    expected = set(range(n_oscillators))
    if set(left).union(right) != expected:
        raise ValueError("minimum_partition must cover every oscillator exactly once")
    return left, right


def _validate_partition_side(value: object, *, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} groups must contain integer indices")
    try:
        items = cast("tuple[object, ...]", tuple(value))  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{name} groups must contain integer indices") from exc
    indices: list[int] = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, Integral):
            raise ValueError(f"{name} groups must contain integer indices")
        index = int(item)
        if index < 0:
            raise ValueError(f"{name} groups must contain non-negative indices")
        indices.append(index)
    if len(set(indices)) != len(indices):
        raise ValueError(f"{name} groups must not contain duplicates")
    return tuple(indices)


def _pairwise_mi_matrix(phases: FloatArray, n_bins: int) -> FloatArray:
    n_oscillators = phases.shape[0]
    matrix = np.zeros((n_oscillators, n_oscillators), dtype=np.float64)
    for i in range(n_oscillators):
        for j in range(i + 1, n_oscillators):
            value = _mutual_information(phases[i], phases[j], n_bins)
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix


def _mutual_information(
    phases_a: FloatArray,
    phases_b: FloatArray,
    n_bins: int,
) -> float:
    wrapped_a = phases_a % _TWO_PI
    wrapped_b = phases_b % _TWO_PI
    edges = np.linspace(0.0, _TWO_PI, n_bins + 1, dtype=np.float64)
    hist_ab, _, _ = np.histogram2d(wrapped_a, wrapped_b, bins=[edges, edges])
    total = float(hist_ab.sum())
    if total == 0.0:
        return 0.0

    probabilities = hist_ab / total
    p_a = probabilities.sum(axis=1)
    p_b = probabilities.sum(axis=0)
    expected = np.outer(p_a, p_b)
    mask = (probabilities > 0.0) & (expected > 0.0)
    mi = float(
        np.sum(probabilities[mask] * np.log(probabilities[mask] / expected[mask]))
    )
    return max(0.0, mi)


def _mean_off_diagonal(matrix: FloatArray) -> float:
    n = matrix.shape[0]
    upper = matrix[np.triu_indices(n, k=1)]
    if upper.size == 0:
        return 0.0
    return float(np.mean(upper))


def _minimum_bipartition(matrix: FloatArray) -> tuple[Partition, float]:
    best_partition: Partition | None = None
    best_score = float("inf")
    for left, right in _unique_bipartitions(matrix.shape[0]):
        score = _cross_partition_mean(matrix, left, right)
        if score < best_score:
            best_partition = (left, right)
            best_score = score

    if best_partition is None:
        msg = "at least two oscillators are required for bipartition analysis"
        raise ValueError(msg)
    return best_partition, max(0.0, float(best_score))


def _unique_bipartitions(n_items: int) -> tuple[Partition, ...]:
    items = tuple(range(n_items))
    partitions: list[Partition] = []
    for size in range(1, (n_items // 2) + 1):
        for left in combinations(items, size):
            if 0 not in left:
                continue
            right = tuple(item for item in items if item not in left)
            partitions.append((left, right))
    return tuple(partitions)


def _cross_partition_mean(
    matrix: FloatArray,
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> float:
    cross = matrix[np.ix_(left, right)]
    if cross.size == 0:
        return 0.0
    return float(np.mean(cross))


def _normalise_phi(phi: float, n_bins: int) -> float:
    scale = float(np.log(n_bins))
    if scale <= 0.0:
        return 0.0
    return float(np.clip(phi / scale, 0.0, 1.0))


def _independent_benchmark_series(n_samples: int) -> FloatArray:
    rng = np.random.default_rng(137)
    series: FloatArray = rng.uniform(0.0, _TWO_PI, size=(4, n_samples)).astype(
        np.float64
    )
    return series


def _modular_benchmark_series(n_samples: int) -> FloatArray:
    base_a = np.linspace(0.0, 5.0 * _TWO_PI, n_samples, dtype=np.float64) % _TWO_PI
    base_b = np.linspace(0.0, 6.0 * _TWO_PI, n_samples, dtype=np.float64) % _TWO_PI
    series: FloatArray = np.vstack(
        [
            base_a,
            (base_a + 0.03) % _TWO_PI,
            (base_b + np.pi) % _TWO_PI,
            (base_b + np.pi + 0.03) % _TWO_PI,
        ]
    ).astype(np.float64)
    return series


def _phase_lag_chain_benchmark_series(n_samples: int) -> FloatArray:
    base = np.linspace(0.0, 6.0 * _TWO_PI, n_samples, dtype=np.float64) % _TWO_PI
    series: FloatArray = np.vstack(
        [
            base,
            (base + 0.35) % _TWO_PI,
            (base + 0.70) % _TWO_PI,
            (base + 1.05) % _TWO_PI,
        ]
    ).astype(np.float64)
    return series


def _noisy_locked_benchmark_series(n_samples: int) -> FloatArray:
    rng = np.random.default_rng(211)
    base = np.linspace(0.0, 6.0 * _TWO_PI, n_samples, dtype=np.float64)
    series: FloatArray = np.vstack(
        [
            base,
            base + rng.normal(0.0, 0.08, size=n_samples),
            base + rng.normal(0.0, 0.16, size=n_samples),
            base + rng.normal(0.0, 0.24, size=n_samples),
        ]
    ).astype(np.float64)
    return series % _TWO_PI


def _locked_benchmark_series(n_samples: int) -> FloatArray:
    base = np.linspace(0.0, 6.0 * _TWO_PI, n_samples, dtype=np.float64) % _TWO_PI
    series: FloatArray = np.vstack(
        [
            base,
            (base + 0.02) % _TWO_PI,
            (base + 0.04) % _TWO_PI,
            (base + 0.06) % _TWO_PI,
        ]
    ).astype(np.float64)
    return series
