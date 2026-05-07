# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hierarchical supervisor summaries

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

__all__ = [
    "ChildSupervisorSummary",
    "HierarchicalOrchestrationPlan",
    "HierarchyEscalation",
    "build_hierarchical_orchestration_plan",
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
