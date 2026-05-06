# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Supervisor topology adaptation

"""Supervisor-side higher-order topology mutation utilities.

The functions here do not replace the UPDE, simplicial, or hypergraph
engines. They prepare the next-step coupling topology from live phase
evidence so an existing engine can consume pairwise ``K_nm`` and optional
triadic hyperedges.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.upde.hypergraph import Hyperedge

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = [
    "HigherOrderTopologySupervisor",
    "TopologyMutationPolicy",
    "TopologyMutationResult",
]


@dataclass(frozen=True)
class TopologyMutationPolicy:
    """Policy knobs for one topology mutation step.

    ``mutation_rate`` is the main supervisor knob: zero freezes topology;
    one applies the maximum allowed per-step pairwise and triadic changes.
    """

    mutation_rate: float = 0.1
    coherence_floor: float = 0.75
    pairwise_threshold: float = 0.85
    simplex_threshold: float = 0.9
    max_pairwise_delta: float = 0.05
    max_simplex_strength: float = 0.2
    max_new_simplices: int = 4
    prune_threshold: float = 0.2
    simplex_pairwise_support_floor: float = 0.0
    max_coupling: float = 10.0

    def __post_init__(self) -> None:
        _require_unit_interval(self.mutation_rate, "mutation_rate")
        _require_unit_interval(self.coherence_floor, "coherence_floor")
        _require_unit_interval(self.pairwise_threshold, "pairwise_threshold")
        _require_unit_interval(self.simplex_threshold, "simplex_threshold")
        _require_non_negative(self.max_pairwise_delta, "max_pairwise_delta")
        _require_non_negative(self.max_simplex_strength, "max_simplex_strength")
        _require_non_negative(self.prune_threshold, "prune_threshold")
        _require_non_negative(
            self.simplex_pairwise_support_floor, "simplex_pairwise_support_floor"
        )
        _require_non_negative(self.max_coupling, "max_coupling")
        if self.max_new_simplices < 0:
            raise ValueError("max_new_simplices must be non-negative")


@dataclass(frozen=True)
class TopologyMutationResult:
    """Result of a supervisor topology mutation step."""

    knm: FloatArray
    hyperedges: tuple[Hyperedge, ...]
    added_simplices: tuple[Hyperedge, ...]
    pruned_simplices: tuple[Hyperedge, ...]
    pairwise_delta_norm: float
    global_coherence: float

    def to_audit_record(self) -> dict[str, object]:
        """Return a serialisable audit payload for topology mutation."""
        return {
            "global_coherence": self.global_coherence,
            "pairwise_delta_norm": self.pairwise_delta_norm,
            "hyperedge_count": len(self.hyperedges),
            "added_simplices": [
                {"nodes": edge.nodes, "strength": edge.strength}
                for edge in self.added_simplices
            ],
            "pruned_simplices": [
                {"nodes": edge.nodes, "strength": edge.strength}
                for edge in self.pruned_simplices
            ],
        }


class HigherOrderTopologySupervisor:
    """Edit pairwise and triadic topology from live phase evidence."""

    def __init__(self, policy: TopologyMutationPolicy | None = None) -> None:
        self.policy = policy or TopologyMutationPolicy()

    def mutate(
        self,
        phases: FloatArray,
        knm: FloatArray,
        hyperedges: tuple[Hyperedge, ...] | None = None,
    ) -> TopologyMutationResult:
        """Return a mutated topology for the next supervisor actuation step."""
        phases_arr = _validate_phases(phases)
        knm_arr = _validate_knm(knm, phases_arr.size)
        existing = _canonical_hyperedges(tuple(hyperedges or ()))
        _validate_hyperedges(existing, phases_arr.size)

        global_coherence = _order_parameter(phases_arr)
        if self.policy.mutation_rate == 0.0:
            return TopologyMutationResult(
                knm=knm_arr.copy(),
                hyperedges=existing,
                added_simplices=(),
                pruned_simplices=(),
                pairwise_delta_norm=0.0,
                global_coherence=global_coherence,
            )

        local = _pairwise_phase_alignment(phases_arr)
        mutated_knm = _mutate_pairwise(knm_arr, local, self.policy)
        kept, pruned = _prune_simplices(existing, phases_arr, self.policy)
        added = _candidate_simplices(
            phases_arr,
            mutated_knm,
            kept,
            self.policy,
            global_coherence,
        )
        hyperedge_map = {edge.nodes: edge for edge in kept}
        for edge in added:
            hyperedge_map[edge.nodes] = edge
        hyperedge_tuple = tuple(hyperedge_map[nodes] for nodes in sorted(hyperedge_map))

        return TopologyMutationResult(
            knm=mutated_knm,
            hyperedges=hyperedge_tuple,
            added_simplices=added,
            pruned_simplices=pruned,
            pairwise_delta_norm=float(np.linalg.norm(mutated_knm - knm_arr)),
            global_coherence=global_coherence,
        )


def _validate_phases(phases: FloatArray) -> FloatArray:
    arr = np.asarray(phases, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("phases must be a one-dimensional array")
    if arr.size < 1:
        raise ValueError("phases must contain at least one oscillator")
    if not np.all(np.isfinite(arr)):
        raise ValueError("phases must be finite")
    return arr


def _validate_knm(knm: FloatArray, n: int) -> FloatArray:
    arr = np.asarray(knm, dtype=np.float64)
    if arr.shape != (n, n):
        raise ValueError(f"knm must have shape ({n}, {n})")
    if not np.all(np.isfinite(arr)):
        raise ValueError("knm must be finite")
    if np.any(arr < 0.0):
        raise ValueError("knm must be non-negative")
    return arr


def _validate_hyperedges(hyperedges: tuple[Hyperedge, ...], n: int) -> None:
    for edge in hyperedges:
        if len(edge.nodes) < 3:
            raise ValueError("higher-order hyperedges must contain at least 3 nodes")
        if len(set(edge.nodes)) != len(edge.nodes):
            raise ValueError("hyperedge nodes must be unique")
        if any(node < 0 or node >= n for node in edge.nodes):
            raise ValueError("hyperedge node index out of range")
        if not np.isfinite(edge.strength) or edge.strength < 0.0:
            raise ValueError("hyperedge strength must be finite and non-negative")


def _canonical_hyperedges(hyperedges: tuple[Hyperedge, ...]) -> tuple[Hyperedge, ...]:
    return tuple(
        Hyperedge(nodes=tuple(sorted(edge.nodes)), strength=float(edge.strength))
        for edge in hyperedges
    )


def _order_parameter(phases: FloatArray) -> float:
    return float(np.abs(np.mean(np.exp(1j * phases))))


def _pairwise_phase_alignment(phases: FloatArray) -> FloatArray:
    diffs = phases[np.newaxis, :] - phases[:, np.newaxis]
    alignment = 0.5 * (np.cos(diffs) + 1.0)
    np.fill_diagonal(alignment, 0.0)
    result: FloatArray = alignment
    return result


def _mutate_pairwise(
    knm: FloatArray,
    local_alignment: FloatArray,
    policy: TopologyMutationPolicy,
) -> FloatArray:
    step = policy.mutation_rate * policy.max_pairwise_delta
    strengthen = local_alignment >= policy.pairwise_threshold
    weaken = local_alignment < policy.prune_threshold
    delta = np.zeros_like(knm)
    delta[strengthen] = step * (local_alignment[strengthen] - policy.pairwise_threshold)
    delta[weaken] = -step * (policy.prune_threshold - local_alignment[weaken])
    mutated = np.clip(knm + delta, 0.0, policy.max_coupling)
    np.fill_diagonal(mutated, 0.0)
    result: FloatArray = mutated
    return result


def _prune_simplices(
    hyperedges: tuple[Hyperedge, ...],
    phases: FloatArray,
    policy: TopologyMutationPolicy,
) -> tuple[tuple[Hyperedge, ...], tuple[Hyperedge, ...]]:
    kept: list[Hyperedge] = []
    pruned: list[Hyperedge] = []
    for edge in hyperedges:
        coherence = _simplex_coherence(phases, edge.nodes)
        if coherence < policy.prune_threshold or edge.strength <= 0.0:
            pruned.append(edge)
        else:
            kept.append(edge)
    return tuple(kept), tuple(pruned)


def _candidate_simplices(
    phases: FloatArray,
    knm: FloatArray,
    existing: tuple[Hyperedge, ...],
    policy: TopologyMutationPolicy,
    global_coherence: float,
) -> tuple[Hyperedge, ...]:
    if phases.size < 3 or global_coherence >= policy.coherence_floor:
        return ()
    existing_nodes = {edge.nodes for edge in existing}
    candidates: list[tuple[float, tuple[int, int, int]]] = []
    for nodes in combinations(range(phases.size), 3):
        if nodes in existing_nodes:
            continue
        coherence = _simplex_coherence(phases, nodes)
        if coherence >= policy.simplex_threshold and _has_pairwise_support(
            knm, nodes, policy.simplex_pairwise_support_floor
        ):
            candidates.append((coherence, nodes))
    candidates.sort(reverse=True)
    strength = policy.mutation_rate * policy.max_simplex_strength
    return tuple(
        Hyperedge(nodes=nodes, strength=strength)
        for _, nodes in candidates[: policy.max_new_simplices]
    )


def _simplex_coherence(phases: FloatArray, nodes: tuple[int, ...]) -> float:
    local_phases = phases[np.asarray(nodes, dtype=np.int64)]
    return _order_parameter(local_phases)


def _has_pairwise_support(
    knm: FloatArray, nodes: tuple[int, ...], floor: float
) -> bool:
    if floor <= 0.0:
        return True
    return all(knm[i, j] >= floor for i, j in combinations(nodes, 2))


def _require_unit_interval(value: float, name: str) -> None:
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")


def _require_non_negative(value: float, name: str) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
