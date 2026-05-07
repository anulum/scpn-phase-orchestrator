# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hierarchical supervisor summaries

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

__all__ = [
    "ChildSupervisorSummary",
    "HierarchicalOrchestrationPlan",
    "HierarchyConsensusRound",
    "HierarchyConsensusState",
    "HierarchySyncEnvelope",
    "HierarchySyncLedger",
    "HierarchyTransportRuntime",
    "HierarchyEscalation",
    "build_hierarchical_orchestration_plan",
    "build_hierarchy_sync_envelope",
    "ingest_hierarchy_sync_envelopes",
    "load_hierarchy_sync_envelope",
    "simulate_hierarchy_gossip_consensus",
]

FloatArray: TypeAlias = NDArray[np.float64]

_REGIME_CRITICAL = "critical"
_REGIME_DEGRADED = "degraded"
_REGIME_NOMINAL = "nominal"
_AUDIT_SCOPE_REDUCED_SUMMARIES = "reduced_child_summaries_only"
_DEFAULT_HIERARCHY_SYNC_PROTOCOL = "spo-hierarchy-sync/v1"
_MAX_JSON_SAFE_INTEGER = (1 << 53) - 1
_HIERARCHY_RAW_MARKER_TERMS = frozenset(
    {
        "evidence",
        "history",
        "payload",
        "raw",
    }
)
_HIERARCHY_RAW_DATA_TERMS = frozenset(
    {
        "actuator",
        "actuators",
        "coupling",
        "couplings",
        "event",
        "events",
        "graph",
        "graphs",
        "observation",
        "observations",
        "phase",
        "phases",
        "series",
        "signal",
        "signals",
        "time",
        "timeseries",
    }
)
_HIERARCHY_SYNC_ENVELOPE_KEYS = frozenset(
    {
        "monotonic_time_s",
        "protocol_version",
        "sequence",
        "source_node",
        "summary",
    }
)
_HIERARCHY_SYNC_SUMMARY_KEYS = frozenset(
    {
        "R",
        "channel",
        "confidence",
        "metadata",
        "name",
        "psi",
        "regime",
        "weighted_R",
    }
)
_FORBIDDEN_RAW_HIERARCHY_KEYS = frozenset(
    {
        "actuator",
        "actuators",
        "actuator_handle",
        "actuator_handles",
        "actuator_target",
        "actuator_targets",
        "child_evidence",
        "child_observations",
        "coupling",
        "coupling_matrix",
        "coupling_matrices",
        "couplings",
        "event",
        "events",
        "evidence",
        "graph",
        "knm",
        "local_coupling_matrix",
        "local_coupling_matrices",
        "observation",
        "observations",
        "phase",
        "phases",
        "raw",
        "raw_coupling",
        "raw_coupling_matrix",
        "raw_event",
        "raw_events",
        "raw_graph",
        "raw_observation",
        "raw_observations",
        "raw_time_series",
        "raw_actuator",
        "raw_actuator_target",
        "raw_actuator_targets",
        "raw_actuators",
        "raw_child_observations",
        "raw_evidence",
        "raw_phases",
        "raw_signal",
        "raw_signals",
        "signal",
        "signals",
        "time_series",
    }
)


@dataclass(frozen=True)
class ChildSupervisorSummary:
    """Bounded child-supervisor evidence for parent orchestration.

    The summary intentionally carries reduced coherence evidence only. Raw
    child phases, time series, local coupling matrices, and actuator targets do
    not cross the hierarchy boundary in this foundation slice.
    """

    name: str
    channel: str
    R: float
    psi: float
    regime: str = _REGIME_NOMINAL
    confidence: float = 1.0
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_child_summary_reduced_only(self)
        object.__setattr__(
            self,
            "metadata",
            _normalise_metadata_value(self.metadata, "summary.metadata"),
        )

    @property
    def weighted_R(self) -> float:
        """Return coherence weighted by summary confidence."""
        return float(self.R * self.confidence)

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe reduced child summary."""
        return {
            "name": self.name,
            "channel": self.channel,
            "R": float(self.R),
            "psi": float(self.psi),
            "regime": self.regime,
            "confidence": float(self.confidence),
            "weighted_R": self.weighted_R,
            "metadata": _metadata_to_audit_record(self.metadata),
        }


@dataclass(frozen=True)
class HierarchyEscalation:
    """Bounded evidence escalated from a child to the parent supervisor."""

    child: str
    channel: str
    severity: str
    reason: str
    R: float
    confidence: float
    child_regime: str

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe escalation record."""
        return {
            "child": self.child,
            "channel": self.channel,
            "severity": self.severity,
            "reason": self.reason,
            "R": float(self.R),
            "confidence": float(self.confidence),
            "child_regime": self.child_regime,
        }


@dataclass(frozen=True)
class HierarchicalOrchestrationPlan:
    """Parent orchestration input built from reduced child summaries."""

    hierarchy: str
    children: tuple[ChildSupervisorSummary, ...]
    parent_state: UPDEState
    escalations: tuple[HierarchyEscalation, ...]
    parent_R: float
    parent_psi: float
    audit_scope: str = _AUDIT_SCOPE_REDUCED_SUMMARIES

    def to_audit_record(self) -> dict[str, object]:
        """Return a serialisable plan record for hierarchy audit logs."""
        return {
            "hierarchy": self.hierarchy,
            "audit_scope": self.audit_scope,
            "parent": {
                "R": float(self.parent_R),
                "psi": float(self.parent_psi),
                "stability_proxy": float(self.parent_state.stability_proxy),
                "regime_id": self.parent_state.regime_id,
                "layer_count": len(self.parent_state.layers),
            },
            "children": [child.to_audit_record() for child in self.children],
            "escalations": [
                escalation.to_audit_record() for escalation in self.escalations
            ],
        }


@dataclass(frozen=True)
class HierarchySyncEnvelope:
    """Transport-neutral hierarchy summary exchanged by edge/cloud nodes."""

    protocol_version: str
    source_node: str
    sequence: int
    summary: ChildSupervisorSummary
    monotonic_time_s: float | None = None

    def __post_init__(self) -> None:
        _validate_envelope_reduced_only(self)

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe transport envelope audit record."""
        record: dict[str, object] = {
            "protocol_version": self.protocol_version,
            "source_node": self.source_node,
            "sequence": self.sequence,
            "summary": self.summary.to_audit_record(),
        }
        if self.monotonic_time_s is not None:
            record["monotonic_time_s"] = float(self.monotonic_time_s)
        return record

    def to_json(self) -> str:
        """Serialise the envelope with deterministic key ordering."""
        return json.dumps(self.to_audit_record(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class HierarchySyncLedger:
    """Parent-side ingestion result for sync envelopes."""

    accepted: tuple[HierarchySyncEnvelope, ...]
    rejected: tuple[dict[str, object], ...]
    plan: HierarchicalOrchestrationPlan

    def to_audit_record(self) -> dict[str, object]:
        """Return a serialisable sync-ingestion audit payload."""
        return {
            "accepted": [envelope.to_audit_record() for envelope in self.accepted],
            "rejected": list(self.rejected),
            "plan": self.plan.to_audit_record(),
        }


@dataclass(frozen=True)
class HierarchyConsensusState:
    """Reduced node state after an offline hierarchy gossip round."""

    source_node: str
    sequence: int
    summary: ChildSupervisorSummary

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe consensus node record."""
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
        """Return a JSON-safe consensus-round audit record."""
        return {
            "round_index": self.round_index,
            "states": [state.to_audit_record() for state in self.states],
            "rejected": list(self.rejected),
            "plan": self.plan.to_audit_record(),
        }


class HierarchyTransportRuntime:
    """Socket-free runtime state for hierarchy transport adapters."""

    def __init__(
        self,
        *,
        previous_sequences: Mapping[str, int] | None = None,
        hierarchy: str = "edge_cloud_summary_sync",
        degraded_threshold: float = 0.65,
        critical_threshold: float = 0.35,
        min_confidence: float = 0.5,
        protocol_version: str = _DEFAULT_HIERARCHY_SYNC_PROTOCOL,
    ) -> None:
        _require_non_empty(hierarchy, "hierarchy")
        _require_non_empty(protocol_version, "protocol_version")
        _require_unit_interval(degraded_threshold, "degraded_threshold")
        _require_unit_interval(critical_threshold, "critical_threshold")
        _require_unit_interval(min_confidence, "min_confidence")
        if critical_threshold > degraded_threshold:
            raise ValueError("critical_threshold must be <= degraded_threshold")
        self._previous_sequences = _normalise_previous_sequences(previous_sequences)
        self._hierarchy = hierarchy
        self._degraded_threshold = degraded_threshold
        self._critical_threshold = critical_threshold
        self._min_confidence = min_confidence
        self._protocol_version = protocol_version

    @property
    def previous_sequences(self) -> dict[str, int]:
        """Return the accepted per-source sequence watermarks."""
        return dict(self._previous_sequences)

    def ingest(
        self,
        records: Sequence[HierarchySyncEnvelope | Mapping[str, object] | str],
    ) -> HierarchySyncLedger:
        """Parse a transport batch, ingest it, and advance accepted watermarks."""
        envelopes = tuple(
            load_hierarchy_sync_envelope(record)
            for record in records
        )
        ledger = ingest_hierarchy_sync_envelopes(
            envelopes,
            previous_sequences=self._previous_sequences,
            hierarchy=self._hierarchy,
            degraded_threshold=self._degraded_threshold,
            critical_threshold=self._critical_threshold,
            min_confidence=self._min_confidence,
            protocol_version=self._protocol_version,
        )
        for envelope in ledger.accepted:
            self._previous_sequences[envelope.source_node] = envelope.sequence
        return ledger

    def ingest_batch(
        self,
        records: Sequence[HierarchySyncEnvelope | Mapping[str, object] | str],
    ) -> HierarchySyncLedger:
        """Alias for adapter batch ingestion."""
        return self.ingest(records)

    def to_audit_record(self) -> dict[str, object]:
        """Return socket-free runtime state for audit logging."""
        return {
            "hierarchy": self._hierarchy,
            "protocol_version": self._protocol_version,
            "previous_sequences": {
                source: self._previous_sequences[source]
                for source in sorted(self._previous_sequences)
            },
            "audit_scope": _AUDIT_SCOPE_REDUCED_SUMMARIES,
        }


def build_hierarchical_orchestration_plan(
    children: Iterable[ChildSupervisorSummary],
    *,
    hierarchy: str = "child_supervisors_to_parent",
    degraded_threshold: float = 0.65,
    critical_threshold: float = 0.35,
    min_confidence: float = 0.5,
) -> HierarchicalOrchestrationPlan:
    """Build a parent UPDE state and escalation set from child summaries.

    This is a non-networked hierarchy foundation. It composes child coherence
    summaries into a parent-level ``UPDEState`` so existing regime, policy, FEP,
    causal, and audit paths can reason over nested supervisors without reading
    raw child observations.
    """
    child_tuple = tuple(children)
    _validate_plan_inputs(
        children=child_tuple,
        hierarchy=hierarchy,
        degraded_threshold=degraded_threshold,
        critical_threshold=critical_threshold,
        min_confidence=min_confidence,
    )

    weighted_r = np.asarray(
        [child.weighted_R for child in child_tuple],
        dtype=np.float64,
    )
    phases = np.asarray([child.psi for child in child_tuple], dtype=np.float64)
    parent_r, parent_psi = _weighted_order_parameter(weighted_r, phases)
    parent_regime = _parent_regime(
        parent_r,
        degraded_threshold=degraded_threshold,
        critical_threshold=critical_threshold,
    )
    parent_state = UPDEState(
        layers=[
            LayerState(R=child.weighted_R, psi=float(child.psi))
            for child in child_tuple
        ],
        cross_layer_alignment=_cross_child_alignment(phases),
        stability_proxy=float(np.mean(weighted_r)),
        regime_id=f"hierarchical_{parent_regime}",
    )
    escalations = tuple(
        escalation
        for child in child_tuple
        for escalation in _child_escalations(
            child,
            degraded_threshold=degraded_threshold,
            critical_threshold=critical_threshold,
            min_confidence=min_confidence,
        )
    )
    return HierarchicalOrchestrationPlan(
        hierarchy=hierarchy,
        children=child_tuple,
        parent_state=parent_state,
        escalations=escalations,
        parent_R=parent_r,
        parent_psi=parent_psi,
    )


def build_hierarchy_sync_envelope(
    summary: ChildSupervisorSummary,
    *,
    source_node: str,
    sequence: int,
    protocol_version: str = _DEFAULT_HIERARCHY_SYNC_PROTOCOL,
    monotonic_time_s: float | None = None,
) -> HierarchySyncEnvelope:
    """Build a deterministic edge/cloud hierarchy sync envelope.

    The envelope is transport-neutral: callers may write it to JSONL, send it
    over a message bus, or hand it to tests without this module opening sockets
    or performing live deployment work.
    """
    return HierarchySyncEnvelope(
        protocol_version=protocol_version,
        source_node=source_node,
        sequence=sequence,
        summary=summary,
        monotonic_time_s=monotonic_time_s,
    )


def load_hierarchy_sync_envelope(
    record: HierarchySyncEnvelope | Mapping[str, object] | str,
) -> HierarchySyncEnvelope:
    """Parse a JSON string or decoded mapping into a strict sync envelope."""
    if isinstance(record, HierarchySyncEnvelope):
        _validate_envelope_reduced_only(record)
        return _canonical_hierarchy_sync_envelope(record)
    payload = _load_mapping_record(record)
    _reject_unknown_keys(
        payload,
        allowed=_HIERARCHY_SYNC_ENVELOPE_KEYS,
        location="hierarchy sync envelope",
    )
    _reject_raw_hierarchy_keys(payload, "hierarchy sync envelope")
    summary_record = payload.get("summary")
    if not isinstance(summary_record, Mapping):
        raise ValueError("summary must be a decoded mapping")
    _reject_unknown_keys(
        summary_record,
        allowed=_HIERARCHY_SYNC_SUMMARY_KEYS,
        location="hierarchy sync summary",
    )
    _reject_raw_hierarchy_keys(summary_record, "hierarchy sync summary")

    sequence = _require_integer(payload.get("sequence"), "sequence")
    monotonic_time_s = payload.get("monotonic_time_s")
    if monotonic_time_s is not None:
        monotonic_time_s = _require_float(monotonic_time_s, "monotonic_time_s")
        if monotonic_time_s < 0.0:
            raise ValueError("monotonic_time_s must be finite and non-negative")

    envelope = HierarchySyncEnvelope(
        protocol_version=_require_text_field(payload, "protocol_version"),
        source_node=_require_text_field(payload, "source_node"),
        sequence=sequence,
        summary=_load_child_summary(summary_record),
        monotonic_time_s=monotonic_time_s,
    )
    _validate_envelope_reduced_only(envelope)
    return envelope


def ingest_hierarchy_sync_envelopes(
    envelopes: Sequence[HierarchySyncEnvelope],
    *,
    previous_sequences: Mapping[str, int] | None = None,
    hierarchy: str = "edge_cloud_summary_sync",
    degraded_threshold: float = 0.65,
    critical_threshold: float = 0.35,
    min_confidence: float = 0.5,
    protocol_version: str = _DEFAULT_HIERARCHY_SYNC_PROTOCOL,
) -> HierarchySyncLedger:
    """Validate envelopes and build a parent plan from accepted summaries.

    Parent nodes reject stale or duplicate sequence numbers per source node and
    reject protocol-version mismatches. Accepted envelopes are sorted by source
    node and sequence before parent-state composition, making JSONL replay and
    cloud ingestion deterministic.
    """
    canonical_envelopes = tuple(
        _canonical_hierarchy_sync_envelope(envelope)
        for envelope in envelopes
    )
    for envelope in canonical_envelopes:
        _validate_envelope_reduced_only(envelope)
    _validate_plan_inputs(
        children=[envelope.summary for envelope in canonical_envelopes]
        or [_dummy_summary()],
        hierarchy=hierarchy,
        degraded_threshold=degraded_threshold,
        critical_threshold=critical_threshold,
        min_confidence=min_confidence,
    )
    expected_sequences = _normalise_previous_sequences(previous_sequences)
    accepted: list[HierarchySyncEnvelope] = []
    rejected: list[dict[str, object]] = []
    valid_protocol: list[HierarchySyncEnvelope] = []
    for envelope in sorted(
        canonical_envelopes,
        key=lambda item: (item.source_node, item.sequence),
    ):
        if envelope.protocol_version != protocol_version:
            rejected.append(_rejection(envelope, "protocol_version_mismatch"))
            continue
        valid_protocol.append(envelope)

    duplicate_sequence_conflict_sources = _duplicate_sequence_conflict_sources(
        valid_protocol
    )
    latest_by_source: dict[str, HierarchySyncEnvelope] = {}
    for envelope in valid_protocol:
        if envelope.source_node in duplicate_sequence_conflict_sources:
            continue
        latest = latest_by_source.get(envelope.source_node)
        if latest is None or envelope.sequence > latest.sequence:
            latest_by_source[envelope.source_node] = envelope

    for envelope in valid_protocol:
        if envelope.source_node in duplicate_sequence_conflict_sources:
            rejected.append(_rejection(envelope, "duplicate_sequence_conflict"))
            continue
        latest = latest_by_source[envelope.source_node]
        previous_sequence = expected_sequences.get(envelope.source_node, -1)
        if envelope.sequence <= previous_sequence or envelope is not latest:
            rejected.append(_rejection(envelope, "stale_or_duplicate_sequence"))
            continue
        expected_sequences[envelope.source_node] = envelope.sequence
        accepted.append(envelope)

    accepted_tuple = tuple(
        sorted(accepted, key=lambda item: (item.source_node, item.sequence))
    )
    if not accepted_tuple:
        raise ValueError("at least one hierarchy sync envelope must be accepted")
    plan = build_hierarchical_orchestration_plan(
        [envelope.summary for envelope in accepted_tuple],
        hierarchy=hierarchy,
        degraded_threshold=degraded_threshold,
        critical_threshold=critical_threshold,
        min_confidence=min_confidence,
    )
    return HierarchySyncLedger(
        accepted=accepted_tuple,
        rejected=tuple(
            sorted(
                rejected,
                key=_rejection_sort_key,
            )
        ),
        plan=plan,
    )


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


def _child_escalations(
    child: ChildSupervisorSummary,
    *,
    degraded_threshold: float,
    critical_threshold: float,
    min_confidence: float,
) -> tuple[HierarchyEscalation, ...]:
    records: list[HierarchyEscalation] = []
    regime = child.regime.lower()
    if child.confidence < min_confidence:
        records.append(
            _escalation(child, "degraded", "child_summary_below_min_confidence")
        )
    if critical_threshold > child.R:
        records.append(_escalation(child, "critical", "child_coherence_below_critical"))
    elif degraded_threshold > child.R:
        records.append(_escalation(child, "degraded", "child_coherence_below_degraded"))
    if "critical" in regime and critical_threshold <= child.R:
        records.append(_escalation(child, "critical", "child_regime_escalation"))
    elif "degraded" in regime and degraded_threshold <= child.R:
        records.append(_escalation(child, "degraded", "child_regime_escalation"))
    return tuple(records)


def _advance_consensus_round(
    states: Mapping[str, HierarchyConsensusState],
    *,
    neighbour_map: Mapping[str, Sequence[str]],
    self_weight: float,
    degraded_threshold: float,
    critical_threshold: float,
) -> dict[str, HierarchyConsensusState]:
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


def _dummy_summary() -> ChildSupervisorSummary:
    return ChildSupervisorSummary("validation", "validation", 1.0, 0.0)


def _load_mapping_record(
    record: Mapping[str, object] | str,
) -> Mapping[str, object]:
    if isinstance(record, str):
        try:
            decoded = json.loads(record)
        except json.JSONDecodeError as exc:
            raise ValueError("hierarchy sync envelope must be valid JSON") from exc
        if not isinstance(decoded, Mapping):
            raise ValueError("hierarchy sync envelope JSON must decode to a mapping")
        return decoded
    if not isinstance(record, Mapping):
        raise ValueError("hierarchy sync envelope must be a mapping or JSON string")
    return record


def _load_child_summary(record: Mapping[str, object]) -> ChildSupervisorSummary:
    _reject_unknown_keys(
        record,
        allowed=_HIERARCHY_SYNC_SUMMARY_KEYS,
        location="hierarchy sync summary",
    )
    metadata = record.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a decoded mapping")
    _validate_metadata_value(metadata, "summary.metadata")
    return ChildSupervisorSummary(
        name=_require_text_field(record, "name"),
        channel=_require_text_field(record, "channel"),
        R=_require_float(record.get("R"), "R"),
        psi=_require_float(record.get("psi"), "psi"),
        regime=_require_text_field(record, "regime", default=_REGIME_NOMINAL),
        confidence=_require_float(record.get("confidence", 1.0), "confidence"),
        metadata=dict(metadata),
    )


def _validate_envelope_reduced_only(envelope: HierarchySyncEnvelope) -> None:
    if not isinstance(envelope, HierarchySyncEnvelope):
        raise ValueError("envelope must be a HierarchySyncEnvelope")
    _reject_raw_instance_attributes(envelope, "envelope")
    _require_non_empty(envelope.protocol_version, "protocol_version")
    _require_non_empty(envelope.source_node, "source_node")
    _validate_child_summary_reduced_only(envelope.summary)
    sequence = _require_integer(envelope.sequence, "sequence")
    if sequence < 0:
        raise ValueError("sequence must be >= 0")
    if envelope.monotonic_time_s is not None:
        monotonic_time_s = _require_float(
            envelope.monotonic_time_s,
            "monotonic_time_s",
        )
        if monotonic_time_s < 0.0:
            raise ValueError("monotonic_time_s must be finite and non-negative")


def _canonical_hierarchy_sync_envelope(
    envelope: HierarchySyncEnvelope,
) -> HierarchySyncEnvelope:
    _validate_envelope_reduced_only(envelope)
    return HierarchySyncEnvelope(
        protocol_version=envelope.protocol_version,
        source_node=envelope.source_node,
        sequence=envelope.sequence,
        summary=_canonical_child_summary(envelope.summary),
        monotonic_time_s=envelope.monotonic_time_s,
    )


def _canonical_child_summary(summary: ChildSupervisorSummary) -> ChildSupervisorSummary:
    _validate_child_summary_reduced_only(summary)
    return ChildSupervisorSummary(
        name=summary.name,
        channel=summary.channel,
        R=summary.R,
        psi=summary.psi,
        regime=summary.regime,
        confidence=summary.confidence,
        metadata=summary.metadata,
    )


def _validate_reduced_summary_metadata(summary: ChildSupervisorSummary) -> None:
    _validate_child_summary_reduced_only(summary)


def _validate_child_summary_reduced_only(summary: ChildSupervisorSummary) -> None:
    if not isinstance(summary, ChildSupervisorSummary):
        raise ValueError("summary must be a ChildSupervisorSummary")
    _reject_raw_instance_attributes(summary, "summary")
    _require_non_empty(summary.name, "name")
    _require_non_empty(summary.channel, "channel")
    _require_unit_interval(summary.R, "R")
    _require_finite(summary.psi, "psi")
    _require_unit_interval(summary.confidence, "confidence")
    _require_non_empty(summary.regime, "regime")
    metadata = summary.metadata
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a decoded mapping")
    _validate_metadata_value(metadata, "summary.metadata")


def _require_text_field(
    record: Mapping[str, object],
    key: str,
    *,
    default: str | None = None,
) -> str:
    value = record.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _require_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if abs(value) > _MAX_JSON_SAFE_INTEGER:
        raise ValueError(
            f"{name} must be between -{_MAX_JSON_SAFE_INTEGER} "
            f"and {_MAX_JSON_SAFE_INTEGER}"
        )
    return value


def _require_float(value: object, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    _require_finite_number(value, name)
    return float(value)


def _reject_raw_hierarchy_keys(
    record: Mapping[str, object],
    location: str,
) -> None:
    forbidden = {
        key
        for key in record
        if isinstance(key, str) and _is_forbidden_hierarchy_key(key)
    }
    if forbidden:
        keys = ", ".join(sorted(forbidden))
        raise ValueError(f"{location} contains raw child evidence: {keys}")


def _reject_unknown_keys(
    record: Mapping[str, object],
    *,
    allowed: frozenset[str],
    location: str,
) -> None:
    unknown: list[str] = []
    for key in record:
        if not isinstance(key, str):
            raise ValueError(f"{location} keys must be strings")
        if key not in allowed:
            unknown.append(key)
    if unknown:
        keys = ", ".join(sorted(unknown))
        raise ValueError(f"{location} contains unknown fields: {keys}")


def _validate_metadata_value(value: object, path: str) -> None:
    _normalise_metadata_value(value, path)


def _normalise_metadata_value(value: object, path: str) -> object:
    if isinstance(value, Mapping):
        normalised: dict[str, object] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ValueError("metadata keys must be strings")
            child_path = f"{path}.{key}"
            if _is_forbidden_hierarchy_key(key):
                raise ValueError(f"{child_path} contains raw child evidence: {key}")
            normalised[key] = _normalise_metadata_value(nested, child_path)
        return MappingProxyType(normalised)
    if isinstance(value, list | tuple):
        return tuple(
            _normalise_metadata_value(nested, f"{path}[{index}]")
            for index, nested in enumerate(value)
        )
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int):
        try:
            _require_integer(value, path)
        except ValueError as exc:
            raise ValueError(f"{path} must be JSON-safe metadata") from exc
        return value
    if isinstance(value, float):
        try:
            _require_finite_number(value, path)
        except ValueError as exc:
            raise ValueError(f"{path} must be finite JSON-safe metadata") from exc
        return value
    raise ValueError(f"{path} must be JSON-safe metadata")


def _metadata_to_audit_record(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _metadata_to_audit_record(nested) for key, nested in value.items()}
    if isinstance(value, tuple | list):
        return [_metadata_to_audit_record(nested) for nested in value]
    return value


def _reject_raw_instance_attributes(instance: object, location: str) -> None:
    try:
        names = vars(instance)
    except TypeError:
        return
    forbidden = {name for name in names if _is_forbidden_hierarchy_key(name)}
    if forbidden:
        keys = ", ".join(sorted(forbidden))
        raise ValueError(f"{location} contains raw child evidence attributes: {keys}")


def _is_forbidden_hierarchy_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    if lowered in _FORBIDDEN_RAW_HIERARCHY_KEYS:
        return True
    if lowered.startswith("raw_"):
        return True
    if lowered.startswith("raw") and any(
        term in lowered for term in _HIERARCHY_RAW_DATA_TERMS
    ):
        return True
    tokens = {token for token in lowered.split("_") if token}
    return bool(
        tokens.intersection(_HIERARCHY_RAW_MARKER_TERMS)
        and tokens.intersection(_HIERARCHY_RAW_DATA_TERMS)
    )


def _normalise_previous_sequences(
    previous_sequences: Mapping[str, int] | None,
) -> dict[str, int]:
    if previous_sequences is not None and not isinstance(previous_sequences, Mapping):
        raise ValueError("previous_sequences must be a mapping")
    normalised: dict[str, int] = {}
    for source_node, sequence in (previous_sequences or {}).items():
        _require_non_empty(source_node, "source_node")
        normalised[source_node] = _require_integer(sequence, "sequence")
        if normalised[source_node] < 0:
            raise ValueError("sequence must be >= 0")
    return normalised


def _duplicate_sequence_conflict_sources(
    envelopes: Sequence[HierarchySyncEnvelope],
) -> set[str]:
    grouped: dict[tuple[str, int], list[HierarchySyncEnvelope]] = {}
    for envelope in envelopes:
        grouped.setdefault((envelope.source_node, envelope.sequence), []).append(
            envelope
        )
    conflict_sources: set[str] = set()
    for (source_node, _sequence), records in grouped.items():
        if len(records) < 2:
            continue
        fingerprints = {
            json.dumps(
                record.to_audit_record(),
                sort_keys=True,
                separators=(",", ":"),
            )
            for record in records
        }
        if len(fingerprints) > 1:
            conflict_sources.add(source_node)
    return conflict_sources


def _rejection(
    envelope: HierarchySyncEnvelope,
    reason: str,
) -> dict[str, object]:
    return {
        "source_node": envelope.source_node,
        "sequence": envelope.sequence,
        "reason": reason,
        "protocol_version": envelope.protocol_version,
    }


def _rejection_sort_key(record: Mapping[str, object]) -> tuple[str, int, str]:
    return (
        str(record["source_node"]),
        _require_integer(record["sequence"], "sequence"),
        str(record["reason"]),
    )


def _escalation(
    child: ChildSupervisorSummary,
    severity: str,
    reason: str,
) -> HierarchyEscalation:
    return HierarchyEscalation(
        child=child.name,
        channel=child.channel,
        severity=severity,
        reason=reason,
        R=float(child.R),
        confidence=float(child.confidence),
        child_regime=child.regime,
    )


def _weighted_order_parameter(
    weights: FloatArray,
    phases: FloatArray,
) -> tuple[float, float]:
    if float(np.sum(weights)) == 0.0:
        return 0.0, 0.0
    vector = np.mean(weights * np.exp(1j * phases))
    return float(np.abs(vector)), float(np.angle(vector))


def _cross_child_alignment(phases: FloatArray) -> FloatArray:
    delta = phases[:, None] - phases[None, :]
    return np.asarray((1.0 + np.cos(delta)) / 2.0, dtype=np.float64)


def _parent_regime(
    parent_r: float,
    *,
    degraded_threshold: float,
    critical_threshold: float,
) -> str:
    if parent_r < critical_threshold:
        return _REGIME_CRITICAL
    if parent_r < degraded_threshold:
        return _REGIME_DEGRADED
    return _REGIME_NOMINAL


def _validate_plan_inputs(
    *,
    children: Sequence[ChildSupervisorSummary],
    hierarchy: str,
    degraded_threshold: float,
    critical_threshold: float,
    min_confidence: float,
) -> None:
    if not children:
        raise ValueError("children must contain at least one child summary")
    for child in children:
        _validate_reduced_summary_metadata(child)
    _require_non_empty(hierarchy, "hierarchy")
    _require_unit_interval(degraded_threshold, "degraded_threshold")
    _require_unit_interval(critical_threshold, "critical_threshold")
    _require_unit_interval(min_confidence, "min_confidence")
    if critical_threshold > degraded_threshold:
        raise ValueError("critical_threshold must be <= degraded_threshold")


def _validate_gossip_inputs(*, rounds: int, self_weight: float) -> None:
    rounds = _require_integer(rounds, "rounds")
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    _require_unit_interval(self_weight, "self_weight")


def _validate_neighbour_map(neighbour_map: Mapping[str, Sequence[str]]) -> None:
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


def _require_non_empty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_finite_number(value: float, name: str) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")


def _require_finite(value: float, name: str) -> None:
    _require_finite_number(value, name)


def _require_unit_interval(value: float, name: str) -> None:
    _require_finite_number(value, name)
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
