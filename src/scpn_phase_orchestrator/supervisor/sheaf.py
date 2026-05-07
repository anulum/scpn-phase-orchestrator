# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Supervisor sheaf coherence primitive

"""Sheaf-Laplacian coherence assessment for N-channel supervisor states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "SheafCoherenceResult",
    "SheafObstructionSummary",
    "SheafCoherenceSupervisor",
    "build_sheaf_obstruction_summary",
    "sheaf_coherence",
    "sheaf_laplacian",
]

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class SheafCoherenceResult:
    """Audit-ready obstruction assessment for a cellular-sheaf state."""

    laplacian: FloatArray
    residuals: FloatArray
    obstruction_score: float
    consistency_energy: float
    kernel_dimension: int
    obstruction_dimension: int
    edge_count: int
    tolerance: float

    def to_audit_record(self) -> dict[str, object]:
        """Return a compact serialisable payload for supervisor audit logs."""
        return {
            "obstruction_score": self.obstruction_score,
            "consistency_energy": self.consistency_energy,
            "kernel_dimension": self.kernel_dimension,
            "obstruction_dimension": self.obstruction_dimension,
            "edge_count": self.edge_count,
            "laplacian_shape": list(self.laplacian.shape),
            "residual_shape": list(self.residuals.shape),
            "tolerance": self.tolerance,
            "method": "directed_cellular_sheaf_laplacian",
        }


@dataclass(frozen=True)
class SheafObstructionSummary:
    """Review summary for obstruction hardening and audit triage."""

    severity: str
    top_residual_edges: tuple[tuple[int, int, float, tuple[float, ...]], ...]
    obstruction_score: float
    warning_threshold: float
    critical_threshold: float

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-serialisable obstruction summary."""
        return {
            "severity": self.severity,
            "obstruction_score": self.obstruction_score,
            "warning_threshold": self.warning_threshold,
            "critical_threshold": self.critical_threshold,
            "top_residual_edges": [
                {
                    "target": target,
                    "source": source,
                    "norm": norm,
                    "residual": list(residual),
                }
                for target, source, norm, residual in self.top_residual_edges
            ],
        }


class SheafCoherenceSupervisor:
    """Assess whether N-channel states agree across restriction maps."""

    def __init__(self, tolerance: float = 1e-8) -> None:
        self.tolerance = _validate_tolerance(tolerance)

    def assess(
        self,
        node_states: FloatArray,
        restriction_maps: FloatArray,
    ) -> SheafCoherenceResult:
        """Return sheaf obstruction metrics for one supervisor tick."""
        return sheaf_coherence(
            node_states,
            restriction_maps,
            tolerance=self.tolerance,
        )


def build_sheaf_obstruction_summary(
    result: SheafCoherenceResult,
    *,
    warning_threshold: float = 0.05,
    critical_threshold: float = 0.25,
    top_k: int = 5,
) -> SheafObstructionSummary:
    """Build a passive triage summary from a sheaf-coherence result."""
    warn = _validate_tolerance(warning_threshold)
    critical = _validate_tolerance(critical_threshold)
    if critical < warn:
        raise ValueError("critical_threshold must be >= warning_threshold")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    severity = _obstruction_severity(result.obstruction_score, warn, critical)
    return SheafObstructionSummary(
        severity=severity,
        top_residual_edges=_top_residual_edges(result.residuals, top_k),
        obstruction_score=result.obstruction_score,
        warning_threshold=warn,
        critical_threshold=critical,
    )


def sheaf_coherence(
    node_states: FloatArray,
    restriction_maps: FloatArray,
    tolerance: float = 1e-8,
) -> SheafCoherenceResult:
    """Measure cross-channel consistency over a directed cellular sheaf.

    Args:
        node_states: N-channel node state matrix with shape
            ``(n_nodes, n_channels)``.
        restriction_maps: Directed restriction maps with shape
            ``(n_nodes, n_nodes, n_channels, n_channels)``. Entry
            ``restriction_maps[i, j]`` maps node ``j`` into node ``i``.
        tolerance: Numerical threshold used for approximate nullity and
            obstruction counts.

    Returns:
        A ``SheafCoherenceResult`` with the block sheaf Laplacian,
        directed residual tensor, obstruction score, consistency energy,
        and audit-visible approximate dimensions.
    """
    states = _validate_node_states(node_states)
    maps = _validate_restriction_maps(restriction_maps, states.shape)
    tol = _validate_tolerance(tolerance)
    residuals, edge_count = _restriction_residuals(states, maps, tol)
    laplacian = sheaf_laplacian(maps, tol)
    consistency_energy = float(np.sum(residuals * residuals))
    obstruction_dimension = int(
        np.count_nonzero(np.linalg.norm(residuals, axis=2) > tol)
    )
    obstruction_score = (
        0.0 if edge_count == 0 else float(np.sqrt(consistency_energy / edge_count))
    )
    kernel_dimension = _kernel_dimension(laplacian, tol)

    return SheafCoherenceResult(
        laplacian=laplacian,
        residuals=residuals,
        obstruction_score=obstruction_score,
        consistency_energy=consistency_energy,
        kernel_dimension=kernel_dimension,
        obstruction_dimension=obstruction_dimension,
        edge_count=edge_count,
        tolerance=tol,
    )


def sheaf_laplacian(
    restriction_maps: FloatArray,
    tolerance: float = 1e-8,
) -> FloatArray:
    """Build the block sheaf Laplacian from directed restriction maps."""
    maps = np.asarray(restriction_maps, dtype=np.float64)
    if (
        maps.ndim != 4
        or maps.shape[0] != maps.shape[1]
        or maps.shape[2] != maps.shape[3]
    ):
        raise ValueError("restriction_maps must have shape (N, N, D, D)")
    if not np.all(np.isfinite(maps)):
        raise ValueError("restriction_maps must be finite")
    tol = _validate_tolerance(tolerance)

    n_nodes, _, n_channels, _ = maps.shape
    dim = n_nodes * n_channels
    laplacian = np.zeros((dim, dim), dtype=np.float64)
    identity = np.eye(n_channels, dtype=np.float64)

    for target in range(n_nodes):
        target_slice = _block_slice(target, n_channels)
        for source in range(n_nodes):
            if target == source:
                continue
            restriction = maps[target, source]
            if not _has_edge(restriction, tol):
                continue
            source_slice = _block_slice(source, n_channels)
            laplacian[target_slice, target_slice] += identity
            laplacian[target_slice, source_slice] -= restriction
            laplacian[source_slice, target_slice] -= restriction.T
            laplacian[source_slice, source_slice] += restriction.T @ restriction

    result: FloatArray = 0.5 * (laplacian + laplacian.T)
    return result


def _restriction_residuals(
    states: FloatArray,
    maps: FloatArray,
    tolerance: float,
) -> tuple[FloatArray, int]:
    n_nodes, n_channels = states.shape
    residuals = np.zeros((n_nodes, n_nodes, n_channels), dtype=np.float64)
    edge_count = 0
    for target in range(n_nodes):
        for source in range(n_nodes):
            if target == source:
                continue
            restriction = maps[target, source]
            if not _has_edge(restriction, tolerance):
                continue
            residuals[target, source] = states[target] - restriction @ states[source]
            edge_count += 1
    return residuals, edge_count


def _validate_node_states(node_states: FloatArray) -> FloatArray:
    states = np.asarray(node_states, dtype=np.float64)
    if states.ndim != 2:
        raise ValueError("node_states must have shape (N, D)")
    if states.shape[0] < 1 or states.shape[1] < 1:
        raise ValueError("node_states must contain at least one node and one channel")
    if not np.all(np.isfinite(states)):
        raise ValueError("node_states must be finite")
    return states


def _validate_restriction_maps(
    restriction_maps: FloatArray,
    state_shape: tuple[int, int],
) -> FloatArray:
    maps = np.asarray(restriction_maps, dtype=np.float64)
    n_nodes, n_channels = state_shape
    expected = (n_nodes, n_nodes, n_channels, n_channels)
    if maps.shape != expected:
        raise ValueError(f"restriction_maps must have shape {expected}")
    if not np.all(np.isfinite(maps)):
        raise ValueError("restriction_maps must be finite")
    return maps


def _validate_tolerance(tolerance: float) -> float:
    value = float(tolerance)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    return value


def _kernel_dimension(laplacian: FloatArray, tolerance: float) -> int:
    if laplacian.size == 0:
        return 0
    eigenvalues = np.linalg.eigvalsh(laplacian)
    return int(np.count_nonzero(eigenvalues <= tolerance))


def _has_edge(restriction: FloatArray, tolerance: float) -> bool:
    return bool(np.linalg.norm(restriction) > tolerance)


def _block_slice(node_index: int, n_channels: int) -> slice:
    start = node_index * n_channels
    return slice(start, start + n_channels)


def _obstruction_severity(
    score: float,
    warning_threshold: float,
    critical_threshold: float,
) -> str:
    if score >= critical_threshold:
        return "critical"
    if score >= warning_threshold:
        return "warning"
    return "nominal"


def _top_residual_edges(
    residuals: FloatArray,
    top_k: int,
) -> tuple[tuple[int, int, float, tuple[float, ...]], ...]:
    edges: list[tuple[int, int, float, tuple[float, ...]]] = []
    for target, source in np.argwhere(np.linalg.norm(residuals, axis=2) > 0.0):
        vector = residuals[int(target), int(source)]
        norm = float(np.linalg.norm(vector))
        edges.append(
            (
                int(target),
                int(source),
                norm,
                tuple(float(value) for value in vector),
            )
        )
    edges.sort(key=lambda item: (-item[2], item[0], item[1]))
    return tuple(edges[:top_k])
