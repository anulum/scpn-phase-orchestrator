# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hierarchical orchestration plan builder

"""Parent orchestration plan assembly from bounded child supervisor summaries."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

from .boundary import (
    _AUDIT_SCOPE_REDUCED_SUMMARIES,
    ChildSupervisorSummary,
    HierarchyEscalation,
    _child_escalations,
    _cross_child_alignment,
    _parent_regime,
    _validate_plan_inputs,
    _weighted_order_parameter,
)


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
        """Return a serialisable plan record for hierarchy audit logs.

        Returns
        -------
        dict[str, object]
            Return a serialisable plan record for hierarchy audit logs.
        """
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

    Parameters
    ----------
    children : Iterable[ChildSupervisorSummary]
        Child supervisor summaries.
    hierarchy : str
        Hierarchy label.
    degraded_threshold : float
        Coherence threshold below which a child is degraded.
    critical_threshold : float
        Coherence threshold below which a child is critical.
    min_confidence : float
        Minimum child summary confidence to include.

    Returns
    -------
    HierarchicalOrchestrationPlan
        The parent plan and escalation set.
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
