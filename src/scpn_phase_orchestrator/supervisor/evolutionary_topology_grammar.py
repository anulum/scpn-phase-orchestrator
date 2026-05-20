# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Offline evolutionary topology mutation grammar

"""Offline review-only topology mutation grammar.

This module provides deterministic candidates for topology mutation operations.
All generated candidates are review-only and include stable audit records.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Integral, Real
from typing import Any, Literal

__all__ = [
    "TopologyMutationCandidate",
    "TopologyMutationConfig",
    "TopologyMutationEdge",
    "TopologyMutationNode",
    "TopologyMutationPlan",
    "TopologyMutationReport",
    "run_offline_evolutionary_topology_mutation_search",
]

_SCHEMA_NAME = "evolutionary_topology_mutation_grammar"
_SCHEMA_VERSION = "0.1.0"
_CLAIM_BOUNDARY = "offline_topology_mutation_grammar_review_only"
_EDGE_OPERATIONS = ("edge_add", "edge_remove", "edge_reweight", "community_bridge")


@dataclass(frozen=True)
class TopologyMutationConfig:
    """Knobs used to shape deterministic grammar expansion."""

    generation_count: int = 2
    population_size: int = 8
    mutation_step: float = 0.05
    min_edge_weight: float = 0.0
    max_edge_weight: float = 10.0
    edge_add_base_weight: float = 0.4
    max_add_candidates: int = 16

    def __post_init__(self) -> None:
        _require_positive_int(self.generation_count, "generation_count")
        _require_positive_int(self.population_size, "population_size")
        _require_positive_float(self.mutation_step, "mutation_step")
        _require_non_negative_float(self.min_edge_weight, "min_edge_weight")
        _require_positive_float(self.max_edge_weight, "max_edge_weight")
        _require_non_negative_float(self.edge_add_base_weight, "edge_add_base_weight")
        if self.min_edge_weight > self.max_edge_weight:
            raise ValueError("min_edge_weight must not exceed max_edge_weight")
        if self.max_add_candidates <= 0:
            raise ValueError("max_add_candidates must be a positive integer")


@dataclass(frozen=True)
class TopologyMutationNode:
    """Normalised topology node record."""

    node_id: int
    community: str | None = None

    def to_audit_record(self) -> dict[str, object]:
        return {"node_id": self.node_id, "community": self.community}


@dataclass(frozen=True)
class TopologyMutationEdge:
    """Normalised pairwise edge record."""

    source: int
    target: int
    weight: float

    @property
    def pair(self) -> tuple[int, int]:
        return (self.source, self.target)

    def to_audit_record(self) -> dict[str, object]:
        return {
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
        }


@dataclass(frozen=True)
class TopologyMutationPlan:
    """One planned grammar mutation."""

    operation: str
    node_a: int
    node_b: int
    source_weight: float
    candidate_weight: float | None
    mutation_delta: float
    source_communities: tuple[str | None, str | None]

    def to_audit_record(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "node_a": self.node_a,
            "node_b": self.node_b,
            "source_weight": self.source_weight,
            "candidate_weight": self.candidate_weight,
            "mutation_delta": self.mutation_delta,
            "source_communities": list(self.source_communities),
        }


@dataclass(frozen=True)
class TopologyMutationCandidate:
    """One review-only topology candidate."""

    candidate_id: str
    generation: int
    mutation_index: int
    source_topology_hash: str
    plan: TopologyMutationPlan
    source_edge_count: int
    candidate_edges: tuple[TopologyMutationEdge, ...]
    blocked_reasons: tuple[str, ...]
    candidate_hash: str
    operator_review_required: bool = True
    execution_disabled: bool = True
    live_merge_permitted: bool = False
    hot_patch_permitted: bool = False
    actuation_permitted: bool = False

    @property
    def accepted(self) -> bool:
        return not self.blocked_reasons

    @property
    def status(self) -> str:
        return "accepted" if self.accepted else "rejected"

    def to_audit_record(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "generation": self.generation,
            "mutation_index": self.mutation_index,
            "source_topology_hash": self.source_topology_hash,
            "plan": self.plan.to_audit_record(),
            "source_edge_count": self.source_edge_count,
            "candidate_edges": [
                edge.to_audit_record() for edge in self.candidate_edges
            ],
            "blocked_reasons": list(self.blocked_reasons),
            "candidate_hash": self.candidate_hash,
            "operator_review_required": self.operator_review_required,
            "execution_disabled": self.execution_disabled,
            "live_merge_permitted": self.live_merge_permitted,
            "hot_patch_permitted": self.hot_patch_permitted,
            "actuation_permitted": self.actuation_permitted,
            "status": self.status,
        }


@dataclass(frozen=True)
class TopologyMutationReport:
    """Offline-only topology mutation audit report."""

    schema_name: str
    schema_version: str
    config: TopologyMutationConfig
    source_topology_hash: str
    node_records: tuple[TopologyMutationNode, ...]
    edge_records: tuple[TopologyMutationEdge, ...]
    candidate_count: int
    accepted_count: int
    rejected_count: int
    candidates: tuple[TopologyMutationCandidate, ...]
    claim_boundary: str
    operator_review_required: bool
    non_actuating: bool
    execution_disabled: bool
    hot_patch_permitted: bool
    live_merge_permitted: bool
    actuation_permitted: bool
    report_hash: str

    def to_audit_record(self) -> dict[str, object]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "generation_count": self.config.generation_count,
            "population_size": self.config.population_size,
            "mutation_step": self.config.mutation_step,
            "min_edge_weight": self.config.min_edge_weight,
            "max_edge_weight": self.config.max_edge_weight,
            "edge_add_base_weight": self.config.edge_add_base_weight,
            "max_add_candidates": self.config.max_add_candidates,
            "source_topology_hash": self.source_topology_hash,
            "source_nodes": [node.to_audit_record() for node in self.node_records],
            "source_edges": [edge.to_audit_record() for edge in self.edge_records],
            "candidate_count": self.candidate_count,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "candidates": [
                candidate.to_audit_record() for candidate in self.candidates
            ],
            "claim_boundary": self.claim_boundary,
            "operator_review_required": self.operator_review_required,
            "non_actuating": self.non_actuating,
            "execution_disabled": self.execution_disabled,
            "hot_patch_permitted": self.hot_patch_permitted,
            "live_merge_permitted": self.live_merge_permitted,
            "actuation_permitted": self.actuation_permitted,
            "report_hash": self.report_hash,
        }


def run_offline_evolutionary_topology_mutation_search(
    node_records: Sequence[Mapping[str, object]],
    edge_records: Sequence[Mapping[str, object]],
    *,
    generation_count: int = 2,
    population_size: int = 8,
    mutation_step: float = 0.05,
    min_edge_weight: float = 0.0,
    max_edge_weight: float = 10.0,
    edge_add_base_weight: float = 0.4,
    max_add_candidates: int = 16,
) -> TopologyMutationReport:
    """Generate deterministic offline topology mutation candidates."""

    config = TopologyMutationConfig(
        generation_count=generation_count,
        population_size=population_size,
        mutation_step=mutation_step,
        min_edge_weight=min_edge_weight,
        max_edge_weight=max_edge_weight,
        edge_add_base_weight=edge_add_base_weight,
        max_add_candidates=max_add_candidates,
    )

    nodes = _validate_nodes(node_records)
    edges = _validate_edges(edge_records=edge_records, known_nodes=nodes)

    source_topology_hash = _build_topology_hash(nodes, edges)
    axes = _build_mutation_axes(config=config, nodes=nodes, edges=edges)
    if not axes:
        raise ValueError(
            "topology records do not enable topology mutation axis generation"
        )

    candidates: list[TopologyMutationCandidate] = []
    axis_count = len(axes)
    axis_cursor = 0
    for generation in range(config.generation_count):
        for local_index in range(config.population_size):
            axis = axes[axis_cursor]
            axis_cursor = (axis_cursor + 1) % axis_count
            delta = _deterministic_delta(
                axis_index=axis_cursor,
                generation=generation,
                local_index=local_index,
                mutation_step=config.mutation_step,
            )
            next_edges = {edge.pair: edge.weight for edge in edges}
            blocked_reasons: list[str] = []

            if axis.operation == "edge_reweight":
                candidate_weight = axis.source_weight + delta
                if candidate_weight < config.min_edge_weight:
                    blocked_reasons.append("edge_reweight_below_min_weight")
                if candidate_weight > config.max_edge_weight:
                    blocked_reasons.append("edge_reweight_above_max_weight")
                if not blocked_reasons:
                    next_edges[axis.nodes] = candidate_weight

            elif axis.operation == "edge_remove":
                candidate_weight = 0.0
                if axis.source_weight <= 0.0:
                    blocked_reasons.append("edge_remove_from_zero_weight")
                if not blocked_reasons:
                    next_edges.pop(axis.nodes, None)

            elif axis.operation in ("edge_add", "community_bridge"):
                candidate_weight = config.edge_add_base_weight + abs(delta)
                if candidate_weight < config.min_edge_weight:
                    blocked_reasons.append("edge_add_below_min_weight")
                if candidate_weight > config.max_edge_weight:
                    blocked_reasons.append("edge_add_above_max_weight")
                if not blocked_reasons:
                    next_edges[axis.nodes] = candidate_weight

            else:
                raise ValueError(f"unsupported mutation operation: {axis.operation}")

            mutation_delta = (
                candidate_weight - axis.source_weight
                if candidate_weight is not None
                else -axis.source_weight
            )
            if axis.operation == "edge_remove":
                mutation_delta = -axis.source_weight

            plan = TopologyMutationPlan(
                operation=axis.operation,
                node_a=axis.nodes[0],
                node_b=axis.nodes[1],
                source_weight=axis.source_weight,
                candidate_weight=None
                if axis.operation == "edge_remove"
                else candidate_weight,
                mutation_delta=mutation_delta,
                source_communities=axis.communities,
            )

            mutation_index = generation * config.population_size + local_index + 1
            candidate_id = (
                f"g{generation + 1:03d}-c{local_index + 1:03d}-x{mutation_index:03d}"
            )
            candidate = TopologyMutationCandidate(
                candidate_id=candidate_id,
                generation=generation + 1,
                mutation_index=mutation_index,
                source_topology_hash=source_topology_hash,
                plan=plan,
                source_edge_count=len(edges),
                candidate_edges=_sort_edges(
                    TopologyMutationEdge(
                        source=pair[0],
                        target=pair[1],
                        weight=weight,
                    )
                    for pair, weight in next_edges.items()
                ),
                blocked_reasons=tuple(blocked_reasons),
                candidate_hash="",
            )
            candidates.append(
                replace(
                    candidate,
                    candidate_hash=_build_stable_hash(candidate.to_audit_record()),
                )
            )

    accepted = tuple(candidate for candidate in candidates if candidate.accepted)
    rejected = tuple(candidate for candidate in candidates if not candidate.accepted)

    report = TopologyMutationReport(
        schema_name=_SCHEMA_NAME,
        schema_version=_SCHEMA_VERSION,
        config=config,
        source_topology_hash=source_topology_hash,
        node_records=nodes,
        edge_records=edges,
        candidate_count=len(candidates),
        accepted_count=len(accepted),
        rejected_count=len(rejected),
        candidates=tuple(candidates),
        claim_boundary=_CLAIM_BOUNDARY,
        operator_review_required=True,
        non_actuating=True,
        execution_disabled=True,
        hot_patch_permitted=False,
        live_merge_permitted=False,
        actuation_permitted=False,
        report_hash="",
    )

    return TopologyMutationReport(
        schema_name=report.schema_name,
        schema_version=report.schema_version,
        config=report.config,
        source_topology_hash=report.source_topology_hash,
        node_records=report.node_records,
        edge_records=report.edge_records,
        candidate_count=report.candidate_count,
        accepted_count=report.accepted_count,
        rejected_count=report.rejected_count,
        candidates=report.candidates,
        claim_boundary=report.claim_boundary,
        operator_review_required=report.operator_review_required,
        non_actuating=report.non_actuating,
        execution_disabled=report.execution_disabled,
        hot_patch_permitted=report.hot_patch_permitted,
        live_merge_permitted=report.live_merge_permitted,
        actuation_permitted=report.actuation_permitted,
        report_hash=_build_stable_hash(report.to_audit_record()),
    )


@dataclass(frozen=True)
class _MutationAxis:
    operation: Literal["edge_add", "edge_remove", "edge_reweight", "community_bridge"]
    nodes: tuple[int, int]
    source_weight: float
    communities: tuple[str | None, str | None]


def _validate_nodes(
    node_records: Sequence[Mapping[str, object]],
) -> tuple[TopologyMutationNode, ...]:
    if not isinstance(node_records, Sequence) or isinstance(
        node_records, (str, bytes, bytearray)
    ):
        raise ValueError("node_records must be a non-empty sequence of mappings")
    if not node_records:
        raise ValueError("node_records must be a non-empty sequence")

    seen: set[int] = set()
    out: list[TopologyMutationNode] = []
    for index, raw in enumerate(node_records):
        if not isinstance(raw, Mapping):
            raise ValueError(f"node_records[{index}] must be a mapping")
        node_id = raw.get("node_id", raw.get("id"))
        if node_id is None:
            raise ValueError(f"node_records[{index}] requires node_id")
        normalized = _require_finite_int(node_id, f"node_records[{index}].node_id")
        if normalized in seen:
            raise ValueError(f"node_records[{index}] duplicate node_id {normalized}")
        seen.add(normalized)
        community_raw = raw.get("community")
        community: str | None = None if community_raw is None else str(community_raw)
        out.append(TopologyMutationNode(node_id=normalized, community=community))

    return tuple(sorted(out, key=lambda item: item.node_id))


def _validate_edges(
    *,
    edge_records: Sequence[Mapping[str, object]],
    known_nodes: tuple[TopologyMutationNode, ...],
) -> tuple[TopologyMutationEdge, ...]:
    if not isinstance(edge_records, Sequence) or isinstance(
        edge_records, (str, bytes, bytearray)
    ):
        raise ValueError("edge_records must be a sequence of mappings")

    node_ids = {node.node_id for node in known_nodes}
    seen: set[tuple[int, int]] = set()
    out: list[TopologyMutationEdge] = []

    for index, raw in enumerate(edge_records):
        if not isinstance(raw, Mapping):
            raise ValueError(f"edge_records[{index}] must be a mapping")

        node_pair = raw.get("nodes")
        if node_pair is None:
            node_a = raw.get("source")
            node_b = raw.get("target")
            if node_a is None or node_b is None:
                raise ValueError(
                    f"edge_records[{index}] must define nodes or source/target"
                )
        else:
            if not isinstance(node_pair, Sequence) or isinstance(
                node_pair, (str, bytes, bytearray)
            ):
                raise ValueError(f"edge_records[{index}] nodes must be a sequence")
            if len(node_pair) != 2:
                raise ValueError(
                    f"edge_records[{index}] nodes must be a two-element sequence"
                )
            node_a, node_b = node_pair

        source = _require_finite_int(node_a, f"edge_records[{index}].source")
        target = _require_finite_int(node_b, f"edge_records[{index}].target")
        if source == target:
            raise ValueError(f"edge_records[{index}] must connect two distinct nodes")
        source_n, target_n = sorted((source, target))
        pair = (source_n, target_n)
        if pair in seen:
            raise ValueError(f"edge_records[{index}] duplicate edge {pair}")
        if source_n not in node_ids or target_n not in node_ids:
            raise ValueError(f"edge_records[{index}] references unknown node")
        seen.add(pair)

        weight = _require_finite_real(
            raw.get("weight", 1.0), f"edge_records[{index}].weight"
        )
        if weight < 0.0:
            raise ValueError(f"edge_records[{index}].weight must be non-negative")

        out.append(
            TopologyMutationEdge(source=source_n, target=target_n, weight=weight)
        )

    return tuple(_sort_edges(out))


def _build_mutation_axes(
    config: TopologyMutationConfig,
    nodes: tuple[TopologyMutationNode, ...],
    edges: tuple[TopologyMutationEdge, ...],
) -> tuple[_MutationAxis, ...]:
    axes: list[_MutationAxis] = []
    add_axes: list[_MutationAxis] = []
    edge_map = {edge.pair: edge.weight for edge in edges}
    for edge in edges:
        communities = _node_communities(edge.pair, nodes)
        axes.append(
            _MutationAxis(
                operation="edge_reweight",
                nodes=edge.pair,
                source_weight=edge.weight,
                communities=communities,
            )
        )
        axes.append(
            _MutationAxis(
                operation="edge_remove",
                nodes=edge.pair,
                source_weight=edge.weight,
                communities=communities,
            )
        )

    node_pairs = [
        (left.node_id, right.node_id)
        for left_index, left in enumerate(nodes)
        for right in nodes[left_index + 1 :]
    ]
    for pair in node_pairs:
        if pair in edge_map:
            continue
        add_axes.append(
            _MutationAxis(
                operation="edge_add",
                nodes=pair,
                source_weight=0.0,
                communities=_node_communities(pair, nodes),
            )
        )
        if len(add_axes) >= config.max_add_candidates:
            break

    axes.extend(add_axes)

    axes.extend(_build_community_bridge_axes(nodes=nodes, existing_edges=edge_map))

    axis_priority = {
        operation: index for index, operation in enumerate(_EDGE_OPERATIONS)
    }
    axes.sort(
        key=lambda axis: (
            axis_priority[axis.operation],
            axis.nodes,
        )
    )
    return tuple(axes)


def _build_community_bridge_axes(
    nodes: tuple[TopologyMutationNode, ...],
    existing_edges: dict[tuple[int, int], float],
) -> tuple[_MutationAxis, ...]:
    if len(nodes) < 2:
        return ()

    communities: dict[str | None, list[int]] = {}
    for node in nodes:
        communities.setdefault(node.community, []).append(node.node_id)
    if len(communities) < 2:
        return ()

    labels = sorted(
        communities.keys(), key=lambda value: (value is not None, str(value))
    )
    axes: list[_MutationAxis] = []
    for left_index in range(len(labels) - 1):
        left_label = labels[left_index]
        right_label = labels[left_index + 1]
        axis_pair: tuple[int, int] | None = None

        for source in communities[left_label]:
            for target in communities[right_label]:
                pair = tuple(sorted((source, target)))
                if pair not in existing_edges:
                    axis_pair = pair
                    break
            if axis_pair is not None:
                break

        if axis_pair is None:
            continue

        axes.append(
            _MutationAxis(
                operation="community_bridge",
                nodes=axis_pair,
                source_weight=0.0,
                communities=(left_label, right_label),
            )
        )
    return tuple(axes)


def _deterministic_delta(
    *,
    axis_index: int,
    generation: int,
    local_index: int,
    mutation_step: float,
) -> float:
    direction = 1.0 if (generation + local_index) % 2 == 0 else -1.0
    period = 17
    position = ((axis_index - 1) % period) / period
    return direction * mutation_step * (0.5 + position)


def _sort_edges(
    edges: Sequence[TopologyMutationEdge] | tuple[TopologyMutationEdge, ...],
) -> tuple[TopologyMutationEdge, ...]:
    return tuple(sorted(edges, key=lambda edge: edge.pair))


def _node_communities(
    pair: tuple[int, int],
    nodes: tuple[TopologyMutationNode, ...],
) -> tuple[str | None, str | None]:
    mapping = {node.node_id: node.community for node in nodes}
    return (mapping[pair[0]], mapping[pair[1]])


def _build_topology_hash(
    nodes: tuple[TopologyMutationNode, ...],
    edges: tuple[TopologyMutationEdge, ...],
) -> str:
    return _build_stable_hash(
        {
            "schema": _SCHEMA_NAME,
            "nodes": [node.to_audit_record() for node in nodes],
            "edges": [edge.to_audit_record() for edge in edges],
        }
    )


def _build_stable_hash(payload: Mapping[str, object]) -> str:
    normalised = _coerce_json_safe(payload)
    encoded = json.dumps(
        normalised,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _coerce_json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _coerce_json_safe(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_coerce_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_coerce_json_safe(item) for item in value]
    return value


def _require_positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field} must be a positive integer")
    number = int(value)
    if number <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return number


def _require_positive_float(value: object, field: str) -> float:
    number = _require_finite_real(value, field)
    if number <= 0.0:
        raise ValueError(f"{field} must be a positive finite number")
    return number


def _require_non_negative_float(value: object, field: str) -> float:
    number = _require_finite_real(value, field)
    if number < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return number


def _require_finite_real(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite real number")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{field} must be a finite real number")
    return number


def _require_finite_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field} must be an integer")
    return int(value)
