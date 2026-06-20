# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — STL closed-loop synthesis plan

"""Closed-loop synthesis plan combining automaton, controller, projection, gate."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .actuation_gate import STLRuntimeActuationGate, validate_stl_runtime_actuation_gate
from .automaton import STLMonitoringAutomaton, _validate_horizon_steps
from .controller import STLControllerSynthesis, synthesise_stl_controller_candidates
from .monitor import _validate_trace
from .projection import (
    STLActionProjectionTemplate,
    STLProjectedActionPlan,
    project_stl_controller_candidates,
)


@dataclass(frozen=True)
class STLClosedLoopSynthesisPlan:
    """Offline closed-loop STL controller plan for operator review.

    The plan binds the current monitor state, signal feedback surface, projected
    action proposals, and next review horizon. It is intentionally non-actuating:
    callers must still pass approved actions through runtime policy, safety, and
    actuation gates before any live controller can use them.
    """

    spec: str
    trace_length: int
    horizon_steps: int
    next_review_start_index: int
    next_review_end_index: int
    feedback_signals: tuple[str, ...]
    satisfied: bool
    actuating: bool
    synthesis: STLControllerSynthesis
    projected_plan: STLProjectedActionPlan
    runtime_gate: STLRuntimeActuationGate
    blocked_reasons: tuple[str, ...]

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-serialisable closed-loop synthesis plan.

        Returns
        -------
        dict[str, object]
            Return a JSON-serialisable closed-loop synthesis plan.
        """
        return {
            "spec": self.spec,
            "trace_length": self.trace_length,
            "horizon_steps": self.horizon_steps,
            "next_review_start_index": self.next_review_start_index,
            "next_review_end_index": self.next_review_end_index,
            "feedback_signals": list(self.feedback_signals),
            "satisfied": self.satisfied,
            "actuating": self.actuating,
            "controller_synthesis": self.synthesis.to_audit_record(),
            "projected_action_plan": self.projected_plan.to_audit_record(),
            "runtime_actuation_gate": self.runtime_gate.to_audit_record(),
            "blocked_reasons": list(self.blocked_reasons),
        }


def synthesise_stl_closed_loop_plan(
    automaton: STLMonitoringAutomaton,
    trace: dict[str, list[float]],
    templates: Sequence[STLActionProjectionTemplate],
    *,
    horizon_steps: int = 1,
    action_map: dict[str, str] | None = None,
) -> STLClosedLoopSynthesisPlan:
    """Build an offline closed-loop STL controller plan.

    The function synthesizes candidates from the current STL automaton, projects
    them through explicit policy templates, and records the future feedback
    review window. It does not mutate runtime state or permit actuation.

    Parameters
    ----------
    automaton : STLMonitoringAutomaton
        The STL monitoring automaton.
    trace : dict[str, list[float]]
        Signal trace keyed by variable name, each a list of floats.
    templates : Sequence[STLActionProjectionTemplate]
        STL action-projection templates.
    horizon_steps : int
        Closed-loop planning horizon in steps.
    action_map : dict[str, str] | None
        Mapping of automaton state to action name, or ``None``.

    Returns
    -------
    STLClosedLoopSynthesisPlan
        The offline closed-loop STL controller plan.
    """
    _validate_horizon_steps(horizon_steps)
    _validate_trace(trace)
    trace_length = len(next(iter(trace.values())))
    synthesis = synthesise_stl_controller_candidates(
        automaton,
        trace,
        action_map=action_map,
    )
    projected_plan = project_stl_controller_candidates(synthesis, templates)
    runtime_gate = validate_stl_runtime_actuation_gate(projected_plan, templates)
    blocked_reasons: list[str] = []
    if synthesis.satisfied:
        blocked_reasons.append("stl_satisfied_no_control_needed")
    if synthesis.candidates and not projected_plan.approved_actions:
        blocked_reasons.append("no_projected_actions")
    if projected_plan.rejected_candidates:
        blocked_reasons.append("unprojected_candidates")
    return STLClosedLoopSynthesisPlan(
        spec=automaton.spec,
        trace_length=trace_length,
        horizon_steps=horizon_steps,
        next_review_start_index=trace_length,
        next_review_end_index=trace_length + horizon_steps - 1,
        feedback_signals=automaton.signals,
        satisfied=synthesis.satisfied,
        actuating=False,
        synthesis=synthesis,
        projected_plan=projected_plan,
        runtime_gate=runtime_gate,
        blocked_reasons=tuple(blocked_reasons),
    )


synthesize_stl_closed_loop_plan = synthesise_stl_closed_loop_plan
