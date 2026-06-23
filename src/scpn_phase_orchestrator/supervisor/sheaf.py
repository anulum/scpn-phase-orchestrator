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
from numbers import Integral, Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "SheafCoherenceResult",
    "SheafObstructionSummary",
    "SheafControlProposal",
    "SheafCoherenceSupervisor",
    "build_sheaf_obstruction_summary",
    "propose_sheaf_obstruction_control",
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
        """Return a compact serialisable payload for supervisor audit logs.

        Returns
        -------
        dict[str, object]
            Return a compact serialisable payload for supervisor audit logs.
        """
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
        """Return a JSON-serialisable obstruction summary.

        Returns
        -------
        dict[str, object]
            Return a JSON-serialisable obstruction summary.
        """
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


@dataclass(frozen=True)
class SheafControlProposal:
    """Review-only obstruction-aware sheaf-Laplacian control proposal."""

    baseline_obstruction_score: float
    projected_obstruction_score: float
    baseline_consistency_energy: float
    projected_consistency_energy: float
    baseline_kernel_dimension: int
    projected_kernel_dimension: int
    baseline_obstruction_dimension: int
    projected_obstruction_dimension: int
    recommended_update: FloatArray
    projected_node_states: FloatArray
    update_norm: float
    step_size: float
    max_update_norm: float
    accepted_for_review: bool
    non_actuating: bool
    execution_disabled: bool
    operator_review_required: bool
    blocked_reasons: tuple[str, ...]

    def to_audit_record(self) -> dict[str, object]:
        """Return a compact serialisable payload for operator review.

        Returns
        -------
        dict[str, object]
            Return a compact serialisable payload for operator review.
        """
        return {
            "method": "sheaf_laplacian_gradient_descent_review",
            "baseline_obstruction_score": self.baseline_obstruction_score,
            "projected_obstruction_score": self.projected_obstruction_score,
            "baseline_consistency_energy": self.baseline_consistency_energy,
            "projected_consistency_energy": self.projected_consistency_energy,
            "cohomology_dimensions": {
                "baseline_kernel_dimension": self.baseline_kernel_dimension,
                "projected_kernel_dimension": self.projected_kernel_dimension,
                "baseline_obstruction_dimension": (self.baseline_obstruction_dimension),
                "projected_obstruction_dimension": (
                    self.projected_obstruction_dimension
                ),
            },
            "recommended_update_shape": list(self.recommended_update.shape),
            "projected_node_state_shape": list(self.projected_node_states.shape),
            "update_norm": self.update_norm,
            "step_size": self.step_size,
            "max_update_norm": self.max_update_norm,
            "accepted_for_review": self.accepted_for_review,
            "non_actuating": self.non_actuating,
            "execution_disabled": self.execution_disabled,
            "operator_review_required": self.operator_review_required,
            "blocked_reasons": list(self.blocked_reasons),
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
        """Return sheaf obstruction metrics for one supervisor tick.

        Parameters
        ----------
        node_states : FloatArray
            Per-node channel states, shape ``(N, C)``.
        restriction_maps : FloatArray
            Directed sheaf restriction maps.

        Returns
        -------
        SheafCoherenceResult
            The sheaf obstruction metrics for the tick.
        """
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
    """Build a passive triage summary from a sheaf-coherence result.

    Parameters
    ----------
    result : SheafCoherenceResult
        The sheaf-coherence result to summarise.
    warning_threshold : float
        Obstruction value above which a warning is raised.
    critical_threshold : float
        Coherence threshold below which a child is critical.
    top_k : int
        Number of strongest entries to retain.

    Returns
    -------
    SheafObstructionSummary
        The passive triage summary.

    Raises
    ------
    ValueError
        If the result or thresholds are invalid.
    """
    if not isinstance(result, SheafCoherenceResult):
        raise ValueError("result must be a SheafCoherenceResult")
    warn = _validate_tolerance(warning_threshold)
    critical = _validate_tolerance(critical_threshold)
    if critical < warn:
        raise ValueError("critical_threshold must be >= warning_threshold")
    top_k = _validate_top_k(top_k)
    severity = _obstruction_severity(result.obstruction_score, warn, critical)
    return SheafObstructionSummary(
        severity=severity,
        top_residual_edges=_top_residual_edges(result.residuals, top_k),
        obstruction_score=result.obstruction_score,
        warning_threshold=warn,
        critical_threshold=critical,
    )


def propose_sheaf_obstruction_control(
    node_states: FloatArray,
    restriction_maps: FloatArray,
    *,
    step_size: float = 0.25,
    max_update_norm: float = 0.25,
    tolerance: float = 1e-8,
    max_backtracking_steps: int = 12,
) -> SheafControlProposal:
    """Propose a review-only correction along the sheaf-Laplacian gradient.

    The proposal minimises the cellular-sheaf consistency energy
    ``x.T @ L @ x`` by a bounded explicit gradient step. A small deterministic
    backtracking line search is used so accepted proposals never increase the
    measured obstruction energy. The result is an audit artefact only:
    execution is disabled and any live actuation requires a separate operator
    approval path.

    Parameters
    ----------
    node_states : FloatArray
        Per-node channel states, shape ``(N, C)``.
    restriction_maps : FloatArray
        Directed sheaf restriction maps.
    step_size : float
        Gradient step size for the proposed correction.
    max_update_norm : float
        Maximum norm of the proposed correction.
    tolerance : float
        Numerical tolerance.
    max_backtracking_steps : int
        Maximum backtracking line-search steps.

    Returns
    -------
    SheafControlProposal
        The review-only sheaf correction proposal.

    Raises
    ------
    ValueError
        If the node states or step parameters are invalid.
    """
    states = _validate_node_states(node_states)
    if len(states.shape) != 2:
        raise ValueError("node_states must be a 2-D matrix")
    maps = _validate_restriction_maps(
        restriction_maps,
        (states.shape[0], states.shape[1]),
    )
    step = _validate_positive_step(step_size, "step_size")
    max_norm = _validate_update_norm(max_update_norm)
    tol = _validate_tolerance(tolerance)
    max_steps = _validate_backtracking_steps(max_backtracking_steps)

    baseline = sheaf_coherence(states, maps, tolerance=tol)
    zero_update = np.zeros_like(states, dtype=np.float64)
    if baseline.obstruction_score <= tol or baseline.consistency_energy <= tol:
        return _sheaf_control_proposal(
            baseline=baseline,
            projected=baseline,
            update=zero_update,
            projected_node_states=states,
            step_size=step,
            max_update_norm=max_norm,
            accepted=False,
            blocked_reasons=("no_obstruction_detected",),
        )

    flat_state = states.reshape(-1)
    gradient = (2.0 * baseline.laplacian @ flat_state).reshape(states.shape)
    gradient_norm = float(np.linalg.norm(gradient))
    if gradient_norm <= tol:
        return _sheaf_control_proposal(
            baseline=baseline,
            projected=baseline,
            update=zero_update,
            projected_node_states=states,
            step_size=step,
            max_update_norm=max_norm,
            accepted=False,
            blocked_reasons=("zero_sheaf_laplacian_gradient",),
        )

    candidate_projected = baseline
    candidate_update = zero_update
    scale = step
    for _ in range(max_steps):
        update = -scale * gradient
        update_norm = float(np.linalg.norm(update))
        if max_norm == 0.0:
            update = zero_update
        elif update_norm > max_norm:
            update = update * (max_norm / update_norm)
        projected_states = states + update
        projected = sheaf_coherence(projected_states, maps, tolerance=tol)
        if (
            projected.consistency_energy <= baseline.consistency_energy
            and projected.obstruction_score <= baseline.obstruction_score
            and float(np.linalg.norm(update)) > tol
        ):
            return _sheaf_control_proposal(
                baseline=baseline,
                projected=projected,
                update=update,
                projected_node_states=projected_states,
                step_size=scale,
                max_update_norm=max_norm,
                accepted=True,
                blocked_reasons=(),
            )
        candidate_projected = projected
        candidate_update = update
        scale *= 0.5

    return _sheaf_control_proposal(
        baseline=baseline,
        projected=candidate_projected,
        update=candidate_update,
        projected_node_states=states + candidate_update,
        step_size=scale,
        max_update_norm=max_norm,
        accepted=False,
        blocked_reasons=("no_monotone_sheaf_projection",),
    )


def sheaf_coherence(
    node_states: FloatArray,
    restriction_maps: FloatArray,
    tolerance: float = 1e-8,
) -> SheafCoherenceResult:
    """Measure cross-channel consistency over a directed cellular sheaf.

    Parameters
    ----------
    node_states : FloatArray
        N-channel node state matrix with shape ``(n_nodes, n_channels)``.
    restriction_maps : FloatArray
        Directed restriction maps with shape ``(n_nodes, n_nodes, n_channels,
        n_channels)``. Entry ``restriction_maps[i, j]`` maps node ``j`` into node ``i``.
    tolerance : float
        Numerical threshold used for approximate nullity and obstruction counts.

    Returns
    -------
    SheafCoherenceResult
        A ``SheafCoherenceResult`` with the block sheaf Laplacian, directed residual
        tensor, obstruction score, consistency energy, and audit-visible approximate
        dimensions.

    Raises
    ------
    ValueError
        If the node states or restriction maps are invalid.
    """
    states = _validate_node_states(node_states)
    if len(states.shape) != 2:
        raise ValueError("node_states must be a 2-D matrix")
    maps = _validate_restriction_maps(
        restriction_maps,
        (states.shape[0], states.shape[1]),
    )
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
    """Build the block sheaf Laplacian from directed restriction maps.

    Parameters
    ----------
    restriction_maps : FloatArray
        Directed sheaf restriction maps.
    tolerance : float
        Numerical tolerance.

    Returns
    -------
    FloatArray
        The block sheaf Laplacian.

    Raises
    ------
    ValueError
        If the restriction maps are invalid.
    """
    if _contains_boolean_alias(restriction_maps):
        raise ValueError("restriction_maps must not contain boolean values")
    if _contains_complex_alias(restriction_maps):
        raise ValueError("restriction_maps must not contain complex values")
    try:
        maps = np.asarray(restriction_maps, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("restriction_maps must be real-valued") from exc
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
    """Return the restriction-map residuals over the edges."""
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
    """Return the validated per-node states, else raise."""
    if _contains_boolean_alias(node_states):
        raise ValueError("node_states must not contain boolean values")
    if _contains_complex_alias(node_states):
        raise ValueError("node_states must not contain complex values")
    try:
        states = np.asarray(node_states, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("node_states must be real-valued") from exc
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
    """Return the validated restriction maps, else raise."""
    if _contains_boolean_alias(restriction_maps):
        raise ValueError("restriction_maps must not contain boolean values")
    if _contains_complex_alias(restriction_maps):
        raise ValueError("restriction_maps must not contain complex values")
    try:
        maps = np.asarray(restriction_maps, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("restriction_maps must be real-valued") from exc
    n_nodes, n_channels = state_shape
    expected = (n_nodes, n_nodes, n_channels, n_channels)
    if maps.shape != expected:
        raise ValueError(f"restriction_maps must have shape {expected}")
    if not np.all(np.isfinite(maps)):
        raise ValueError("restriction_maps must be finite")
    return maps


def _validate_tolerance(tolerance: float) -> float:
    """Return the tolerance as a non-negative finite float, else raise."""
    if isinstance(tolerance, (bool, np.bool_)) or not isinstance(tolerance, Real):
        raise ValueError("tolerance must be a finite real number")
    value = float(tolerance)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    return value


def _validate_positive_step(value: float, name: str) -> float:
    """Return the step size as a strictly positive float, else raise."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive real number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _validate_update_norm(max_update_norm: float) -> float:
    """Return the update norm as a validated value, else raise."""
    if isinstance(max_update_norm, (bool, np.bool_)) or not isinstance(
        max_update_norm, Real
    ):
        raise ValueError("max_update_norm must be a finite non-negative real number")
    result = float(max_update_norm)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError("max_update_norm must be finite and non-negative")
    return result


def _validate_backtracking_steps(max_backtracking_steps: int) -> int:
    """Return the backtracking step count as a positive integer, else raise."""
    if (
        isinstance(max_backtracking_steps, (bool, np.bool_))
        or not isinstance(max_backtracking_steps, Integral)
        or max_backtracking_steps < 1
    ):
        raise ValueError("max_backtracking_steps must be a positive integer")
    return int(max_backtracking_steps)


def _validate_top_k(top_k: int) -> int:
    """Return the top-k count as a non-negative integer, else raise."""
    if (
        isinstance(top_k, (bool, np.bool_))
        or not isinstance(top_k, Integral)
        or top_k < 0
    ):
        raise ValueError("top_k must be a non-negative integer")
    return int(top_k)


def _contains_boolean_alias(value: object) -> bool:
    """Return whether the value contains any boolean alias."""
    try:
        array = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (bool, np.bool_)) for item in array.flat)


def _contains_complex_alias(value: object) -> bool:
    """Return whether the value contains any complex-number alias."""
    try:
        array = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (complex, np.complexfloating)) for item in array.flat)


def _kernel_dimension(laplacian: FloatArray, tolerance: float) -> int:
    """Return the kernel dimension of the sheaf Laplacian."""
    if laplacian.size == 0:
        return 0
    eigenvalues = np.linalg.eigvalsh(laplacian)
    return int(np.count_nonzero(eigenvalues <= tolerance))


def _has_edge(restriction: FloatArray, tolerance: float) -> bool:
    """Return whether an edge exists between two nodes."""
    return bool(np.linalg.norm(restriction) > tolerance)


def _block_slice(node_index: int, n_channels: int) -> slice:
    """Return the block slice for a node in the stacked vector."""
    start = node_index * n_channels
    return slice(start, start + n_channels)


def _obstruction_severity(
    score: float,
    warning_threshold: float,
    critical_threshold: float,
) -> str:
    """Return the obstruction-severity label for a score."""
    if score >= critical_threshold:
        return "critical"
    if score >= warning_threshold:
        return "warning"
    return "nominal"


def _top_residual_edges(
    residuals: FloatArray,
    top_k: int,
) -> tuple[tuple[int, int, float, tuple[float, ...]], ...]:
    """Return the edges with the largest restriction residuals."""
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


def _sheaf_control_proposal(
    *,
    baseline: SheafCoherenceResult,
    projected: SheafCoherenceResult,
    update: FloatArray,
    projected_node_states: FloatArray,
    step_size: float,
    max_update_norm: float,
    accepted: bool,
    blocked_reasons: tuple[str, ...],
) -> SheafControlProposal:
    """Return the review-only sheaf control proposal."""
    return SheafControlProposal(
        baseline_obstruction_score=baseline.obstruction_score,
        projected_obstruction_score=projected.obstruction_score,
        baseline_consistency_energy=baseline.consistency_energy,
        projected_consistency_energy=projected.consistency_energy,
        baseline_kernel_dimension=baseline.kernel_dimension,
        projected_kernel_dimension=projected.kernel_dimension,
        baseline_obstruction_dimension=baseline.obstruction_dimension,
        projected_obstruction_dimension=projected.obstruction_dimension,
        recommended_update=np.asarray(update, dtype=np.float64),
        projected_node_states=np.asarray(projected_node_states, dtype=np.float64),
        update_norm=float(np.linalg.norm(update)),
        step_size=step_size,
        max_update_norm=max_update_norm,
        accepted_for_review=accepted,
        non_actuating=True,
        execution_disabled=True,
        operator_review_required=True,
        blocked_reasons=blocked_reasons,
    )
