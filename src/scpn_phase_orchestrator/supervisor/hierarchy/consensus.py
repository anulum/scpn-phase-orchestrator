# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hierarchy gossip consensus simulation

"""Gossip consensus rounds and state over bounded hierarchy sync envelopes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .boundary import (
    _DEFAULT_HIERARCHY_SYNC_PROTOCOL,
    ChildSupervisorSummary,
    HierarchySyncEnvelope,
    _parent_regime,
    _require_integer,
    _require_unit_interval,
    _weighted_order_parameter,
)
from .plan import HierarchicalOrchestrationPlan, build_hierarchical_orchestration_plan
from .sync import ingest_hierarchy_sync_envelopes


@dataclass(frozen=True)
class HierarchyConsensusState:
    """Reduced node state after an offline hierarchy gossip round."""

    source_node: str
    sequence: int
    summary: ChildSupervisorSummary

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe consensus node record.

        Returns
        -------
        dict[str, object]
            Return a JSON-safe consensus node record.
        """
        return {
            "source_node": self.source_node,
            "sequence": self.sequence,
            "summary": self.summary.to_audit_record(),
        }


@dataclass(frozen=True)
class HierarchyConsensusRound:
    """Deterministic non-networked gossip/local-consensus replay result."""

    round_index: int
    states: tuple[HierarchyConsensusState, ...]
    plan: HierarchicalOrchestrationPlan
    rejected: tuple[dict[str, object], ...] = ()

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe consensus-round audit record.

        Returns
        -------
        dict[str, object]
            Return a JSON-safe consensus-round audit record.
        """
        return {
            "round_index": self.round_index,
            "states": [state.to_audit_record() for state in self.states],
            "rejected": list(self.rejected),
            "plan": self.plan.to_audit_record(),
        }


def simulate_hierarchy_gossip_consensus(
    envelopes: Sequence[HierarchySyncEnvelope],
    *,
    neighbour_map: Mapping[str, Sequence[str]],
    rounds: int = 1,
    self_weight: float = 0.5,
    hierarchy: str = "offline_hierarchy_gossip_consensus",
    previous_sequences: Mapping[str, int] | None = None,
    degraded_threshold: float = 0.65,
    critical_threshold: float = 0.35,
    min_confidence: float = 0.5,
    protocol_version: str = _DEFAULT_HIERARCHY_SYNC_PROTOCOL,
) -> tuple[HierarchyConsensusRound, ...]:
    """Replay local consensus over hierarchy sync envelopes without networking.

    Each round updates every accepted node from its own reduced summary and the
    summaries of configured neighbours. The update averages confidence-weighted
    coherence and circular phase only; raw child observations never enter the
    consensus state. This is a deterministic simulation surface for testing
    distributed orchestration policies before any live gossip transport exists.

    Parameters
    ----------
    envelopes : Sequence[HierarchySyncEnvelope]
        The ordered transport envelopes.
    neighbour_map : Mapping[str, Sequence[str]]
        Per-node neighbour lists for gossip.
    rounds : int
        Number of gossip rounds.
    self_weight : float
        Self-weight in the gossip consensus update.
    hierarchy : str
        Hierarchy label.
    previous_sequences : Mapping[str, int] | None
        Accepted per-source sequence watermarks, or ``None``.
    degraded_threshold : float
        Coherence threshold below which a child is degraded.
    critical_threshold : float
        Coherence threshold below which a child is critical.
    min_confidence : float
        Minimum child summary confidence to include.
    protocol_version : str
        Hierarchy sync protocol version.

    Returns
    -------
    tuple[HierarchyConsensusRound, ...]
        The per-round gossip consensus states.
    """
    _validate_gossip_inputs(rounds=rounds, self_weight=self_weight)
    _validate_neighbour_map(neighbour_map)
    ledger = ingest_hierarchy_sync_envelopes(
        envelopes,
        previous_sequences=previous_sequences,
        hierarchy=hierarchy,
        degraded_threshold=degraded_threshold,
        critical_threshold=critical_threshold,
        min_confidence=min_confidence,
        protocol_version=protocol_version,
    )
    current = {
        envelope.source_node: HierarchyConsensusState(
            source_node=envelope.source_node,
            sequence=envelope.sequence,
            summary=envelope.summary,
        )
        for envelope in ledger.accepted
    }
    history: list[HierarchyConsensusRound] = []
    for round_index in range(1, rounds + 1):
        current = _advance_consensus_round(
            current,
            neighbour_map=neighbour_map,
            self_weight=self_weight,
            degraded_threshold=degraded_threshold,
            critical_threshold=critical_threshold,
        )
        states = tuple(current[node] for node in sorted(current))
        plan = build_hierarchical_orchestration_plan(
            [state.summary for state in states],
            hierarchy=f"{hierarchy}_round_{round_index}",
            degraded_threshold=degraded_threshold,
            critical_threshold=critical_threshold,
            min_confidence=min_confidence,
        )
        history.append(
            HierarchyConsensusRound(
                round_index=round_index,
                states=states,
                plan=plan,
                rejected=ledger.rejected if round_index == 1 else (),
            )
        )
    return tuple(history)


def _advance_consensus_round(
    states: Mapping[str, HierarchyConsensusState],
    *,
    neighbour_map: Mapping[str, Sequence[str]],
    self_weight: float,
    degraded_threshold: float,
    critical_threshold: float,
) -> dict[str, HierarchyConsensusState]:
    """Advance the gossip consensus one round."""
    next_states: dict[str, HierarchyConsensusState] = {}
    for node, state in states.items():
        neighbours = tuple(
            states[neighbour]
            for neighbour in neighbour_map.get(node, ())
            if neighbour in states
        )
        next_states[node] = _consensus_state(
            state,
            neighbours=neighbours,
            self_weight=self_weight,
            degraded_threshold=degraded_threshold,
            critical_threshold=critical_threshold,
        )
    return next_states


def _consensus_state(
    state: HierarchyConsensusState,
    *,
    neighbours: Sequence[HierarchyConsensusState],
    self_weight: float,
    degraded_threshold: float,
    critical_threshold: float,
) -> HierarchyConsensusState:
    """Return the consensus state over the sync envelopes."""
    if not neighbours:
        return state
    summaries = (state.summary, *(neighbour.summary for neighbour in neighbours))
    neighbour_weight = (1.0 - self_weight) / len(neighbours)
    weights = np.asarray(
        [self_weight, *([neighbour_weight] * len(neighbours))],
        dtype=np.float64,
    )
    weighted_r = np.asarray(
        [summary.weighted_R for summary in summaries],
        dtype=np.float64,
    )
    phases = np.asarray([summary.psi for summary in summaries], dtype=np.float64)
    consensus_weighted_r = float(np.dot(weights, weighted_r))
    consensus_confidence = float(
        np.clip(
            np.dot(weights, [summary.confidence for summary in summaries]),
            0.0,
            1.0,
        )
    )
    consensus_r = (
        0.0
        if consensus_confidence == 0.0
        else float(np.clip(consensus_weighted_r / consensus_confidence, 0.0, 1.0))
    )
    _, consensus_psi = _weighted_order_parameter(weights, phases)
    regime = _parent_regime(
        consensus_r,
        degraded_threshold=degraded_threshold,
        critical_threshold=critical_threshold,
    )
    summary = ChildSupervisorSummary(
        name=state.summary.name,
        channel=state.summary.channel,
        R=consensus_r,
        psi=consensus_psi,
        regime=regime,
        confidence=consensus_confidence,
        metadata={
            "consensus": "offline_gossip",
            "source_node": state.source_node,
            "neighbour_count": len(neighbours),
        },
    )
    return HierarchyConsensusState(
        source_node=state.source_node,
        sequence=state.sequence,
        summary=summary,
    )


def _validate_gossip_inputs(*, rounds: int, self_weight: float) -> None:
    """Validate the gossip-round inputs, else raise."""
    rounds = _require_integer(rounds, "rounds")
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    _require_unit_interval(self_weight, "self_weight")


def _validate_neighbour_map(neighbour_map: Mapping[str, Sequence[str]]) -> None:
    """Return the validated neighbour map, else raise."""
    if not isinstance(neighbour_map, Mapping):
        raise ValueError("neighbour_map must be a mapping")
    for node, neighbours in neighbour_map.items():
        if not isinstance(node, str):
            raise ValueError("neighbour_map node keys must be strings")
        if not isinstance(neighbours, Sequence) or isinstance(
            neighbours,
            str | bytes | bytearray,
        ):
            raise ValueError("neighbour_map neighbours must be a sequence")
        for neighbour in neighbours:
            if not isinstance(neighbour, str):
                raise ValueError("neighbour_map neighbour names must be strings")
