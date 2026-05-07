# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hierarchical supervisor summaries

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

__all__ = [
    "ChildSupervisorSummary",
    "HierarchicalOrchestrationPlan",
    "HierarchySyncEnvelope",
    "HierarchySyncLedger",
    "HierarchyEscalation",
    "build_hierarchical_orchestration_plan",
    "build_hierarchy_sync_envelope",
    "ingest_hierarchy_sync_envelopes",
]

FloatArray: TypeAlias = NDArray[np.float64]

_REGIME_CRITICAL = "critical"
_REGIME_DEGRADED = "degraded"
_REGIME_NOMINAL = "nominal"


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
        _require_non_empty(self.name, "name")
        _require_non_empty(self.channel, "channel")
        _require_unit_interval(self.R, "R")
        _require_finite(self.psi, "psi")
        _require_unit_interval(self.confidence, "confidence")
        _require_non_empty(self.regime, "regime")

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
            "metadata": dict(self.metadata),
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
    audit_scope: str = "reduced_child_summaries_only"

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
        _require_non_empty(self.protocol_version, "protocol_version")
        _require_non_empty(self.source_node, "source_node")
        if self.sequence < 0:
            raise ValueError("sequence must be >= 0")
        if self.monotonic_time_s is not None and (
            not np.isfinite(self.monotonic_time_s) or self.monotonic_time_s < 0.0
        ):
            raise ValueError("monotonic_time_s must be finite and non-negative")

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


def build_hierarchical_orchestration_plan(
    children: Sequence[ChildSupervisorSummary],
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
    _validate_plan_inputs(
        children=children,
        hierarchy=hierarchy,
        degraded_threshold=degraded_threshold,
        critical_threshold=critical_threshold,
        min_confidence=min_confidence,
    )

    child_tuple = tuple(children)
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
    protocol_version: str = "spo-hierarchy-sync/v1",
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


def ingest_hierarchy_sync_envelopes(
    envelopes: Sequence[HierarchySyncEnvelope],
    *,
    previous_sequences: Mapping[str, int] | None = None,
    hierarchy: str = "edge_cloud_summary_sync",
    degraded_threshold: float = 0.65,
    critical_threshold: float = 0.35,
    min_confidence: float = 0.5,
    protocol_version: str = "spo-hierarchy-sync/v1",
) -> HierarchySyncLedger:
    """Validate envelopes and build a parent plan from accepted summaries.

    Parent nodes reject stale or duplicate sequence numbers per source node and
    reject protocol-version mismatches. Accepted envelopes are sorted by source
    node and sequence before parent-state composition, making JSONL replay and
    cloud ingestion deterministic.
    """
    _validate_plan_inputs(
        children=[envelope.summary for envelope in envelopes] or [_dummy_summary()],
        hierarchy=hierarchy,
        degraded_threshold=degraded_threshold,
        critical_threshold=critical_threshold,
        min_confidence=min_confidence,
    )
    expected_sequences = dict(previous_sequences or {})
    accepted: list[HierarchySyncEnvelope] = []
    rejected: list[dict[str, object]] = []
    for envelope in envelopes:
        if envelope.protocol_version != protocol_version:
            rejected.append(_rejection(envelope, "protocol_version_mismatch"))
            continue
        previous_sequence = expected_sequences.get(envelope.source_node, -1)
        if envelope.sequence <= previous_sequence:
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
        rejected=tuple(rejected),
        plan=plan,
    )


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


def _dummy_summary() -> ChildSupervisorSummary:
    return ChildSupervisorSummary("validation", "validation", 1.0, 0.0)


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
    _require_non_empty(hierarchy, "hierarchy")
    _require_unit_interval(degraded_threshold, "degraded_threshold")
    _require_unit_interval(critical_threshold, "critical_threshold")
    _require_unit_interval(min_confidence, "min_confidence")
    if critical_threshold > degraded_threshold:
        raise ValueError("critical_threshold must be <= degraded_threshold")


def _require_non_empty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_finite(value: float, name: str) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _require_unit_interval(value: float, name: str) -> None:
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
