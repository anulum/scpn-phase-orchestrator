# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — STL automata tests

from __future__ import annotations

import pytest

import scpn_phase_orchestrator.monitor.stl as stl_module
from scpn_phase_orchestrator.monitor.stl import (
    STLActionProjectionTemplate,
    STLAutomatonState,
    STLAutomatonTransition,
    STLClosedLoopSynthesisPlan,
    STLControllerCandidate,
    STLControllerSynthesis,
    STLMonitoringAutomaton,
    STLProjectedActionPlan,
    project_stl_controller_candidates,
    synthesise_stl_closed_loop_plan,
    synthesise_stl_controller_candidates,
    synthesise_stl_monitoring_automaton,
    synthesize_stl_closed_loop_plan,
    synthesize_stl_controller_candidates,
    synthesize_stl_monitoring_automaton,
)
from scpn_phase_orchestrator.supervisor.policy_rules import (
    PolicySTLAutomaton,
    PolicySTLSpec,
    synthesise_policy_stl_automata,
)


def test_always_automaton_records_first_violation_and_worst_margin():
    automaton = synthesise_stl_monitoring_automaton(
        "always (R >= 0.3 and amplitude_spread < 0.2)",
        {"R": [0.8, 0.2, 0.5], "amplitude_spread": [0.1, 0.1, 0.35]},
    )

    assert isinstance(automaton, STLMonitoringAutomaton)
    assert automaton.temporal_op == "always"
    assert automaton.signals == ("R", "amplitude_spread")
    assert automaton.satisfied is False
    assert automaton.robustness == pytest.approx(-0.15)

    violated = _state_by_name(automaton.states, "violated")
    assert violated == STLAutomatonState(
        name="violated",
        accepting=False,
        violation=True,
        first_hit_index=1,
    )
    assert automaton.transitions == (
        STLAutomatonTransition(
            source="holding",
            target="holding",
            time_index=0,
            guard="R >= 0.3 and amplitude_spread < 0.2",
            robustness=pytest.approx(0.1),
        ),
        STLAutomatonTransition(
            source="holding",
            target="violated",
            time_index=1,
            guard="R >= 0.3 and amplitude_spread < 0.2",
            robustness=pytest.approx(-0.1),
        ),
        STLAutomatonTransition(
            source="violated",
            target="violated",
            time_index=2,
            guard="R >= 0.3 and amplitude_spread < 0.2",
            robustness=pytest.approx(-0.15),
        ),
    )

    record = automaton.to_audit_record()
    assert record["backend"] == "builtin"
    assert record["states"][1]["first_hit_index"] == 1
    assert record["transitions"][2]["target"] == "violated"


def test_eventually_automaton_records_first_satisfaction_and_best_margin():
    automaton = synthesize_stl_monitoring_automaton(
        "eventually (R >= 0.8)",
        {"R": [0.1, 0.75, 0.85, 0.9]},
    )

    assert automaton.temporal_op == "eventually"
    assert automaton.satisfied is True
    assert automaton.robustness == pytest.approx(0.1)
    assert _state_by_name(automaton.states, "satisfied").first_hit_index == 2
    assert [transition.target for transition in automaton.transitions] == [
        "pending",
        "pending",
        "satisfied",
        "satisfied",
    ]


def test_automaton_synthesis_rejects_unsupported_or_incomplete_traces():
    with pytest.raises(ValueError, match="simple STL syntax"):
        synthesise_stl_monitoring_automaton("R >= 0.3", {"R": [0.5]})

    with pytest.raises(ValueError, match="trace missing signal"):
        synthesise_stl_monitoring_automaton(
            "always (R >= 0.3 and amplitude_spread < 0.2)",
            {"R": [0.5, 0.6]},
        )

    with pytest.raises(ValueError, match="equal length"):
        synthesise_stl_monitoring_automaton(
            "always (R >= 0.3)",
            {"R": [0.5, 0.6], "unused": [1.0]},
        )


def test_policy_stl_automata_preserve_policy_identity_and_audit_severity():
    specs = [
        PolicySTLSpec(
            name="hard_floor",
            spec="always (R >= 0.3)",
            severity="hard",
        ),
        PolicySTLSpec(
            name="eventual_recovery",
            spec="eventually (R >= 0.8)",
        ),
    ]

    automata = synthesise_policy_stl_automata(specs, {"R": [0.2, 0.4, 0.9]})
    audit_records = [automaton.to_audit_record() for automaton in automata]

    assert all(isinstance(automaton, PolicySTLAutomaton) for automaton in automata)
    assert audit_records[0]["name"] == "hard_floor"
    assert audit_records[0]["severity"] == "hard"
    assert audit_records[0]["satisfied"] is False
    assert audit_records[0]["states"][1]["first_hit_index"] == 0
    assert audit_records[1]["name"] == "eventual_recovery"
    assert audit_records[1]["severity"] == "soft"
    assert audit_records[1]["satisfied"] is True
    assert audit_records[1]["states"][1]["first_hit_index"] == 2


def test_stl_controller_synthesis_proposes_non_actuating_violation_actions():
    trace = {"R": [0.8, 0.2, 0.5], "amplitude_spread": [0.1, 0.1, 0.35]}
    automaton = synthesise_stl_monitoring_automaton(
        "always (R >= 0.3 and amplitude_spread < 0.2)",
        trace,
    )

    synthesis = synthesise_stl_controller_candidates(
        automaton,
        trace,
        action_map={"R": "raise_coupling", "amplitude_spread": "dampen_amplitude"},
    )

    assert isinstance(synthesis, STLControllerSynthesis)
    assert synthesis.satisfied is False
    assert synthesis.actuating is False
    assert synthesis.source_backend == "builtin"
    assert synthesis.candidates == (
        STLControllerCandidate(
            signal="amplitude_spread",
            action="dampen_amplitude",
            direction="decrease",
            time_index=2,
            robustness=pytest.approx(-0.15),
            rationale="amplitude_spread < 0.2 violated at t=2 with robustness -0.15",
        ),
    )
    assert synthesis.to_audit_record() == {
        "spec": "always (R >= 0.3 and amplitude_spread < 0.2)",
        "satisfied": False,
        "actuating": False,
        "source_backend": "builtin",
        "candidates": [
            {
                "signal": "amplitude_spread",
                "action": "dampen_amplitude",
                "direction": "decrease",
                "time_index": 2,
                "robustness": pytest.approx(-0.15),
                "rationale": (
                    "amplitude_spread < 0.2 violated at t=2 with robustness -0.15"
                ),
            }
        ],
    }


def test_stl_controller_synthesis_handles_unsatisfied_eventually_specs():
    trace = {"R": [0.1, 0.2, 0.75]}
    automaton = synthesize_stl_monitoring_automaton(
        "eventually (R >= 0.8)",
        trace,
    )

    synthesis = synthesize_stl_controller_candidates(automaton, trace)

    assert synthesis.satisfied is False
    assert synthesis.actuating is False
    assert synthesis.candidates == (
        STLControllerCandidate(
            signal="R",
            action="increase_R",
            direction="increase",
            time_index=2,
            robustness=pytest.approx(-0.05),
            rationale="R >= 0.8 violated at t=2 with robustness -0.05",
        ),
    )


def test_stl_controller_synthesis_returns_no_actions_for_satisfied_specs():
    trace = {"K": [1.0, 2.0, 3.0]}
    automaton = synthesise_stl_monitoring_automaton(
        "always (K <= 10.0)",
        trace,
    )

    synthesis = synthesise_stl_controller_candidates(automaton, trace)

    assert synthesis.satisfied is True
    assert synthesis.actuating is False
    assert synthesis.candidates == ()


def test_stl_controller_synthesis_rejects_mismatched_automata():
    trace = {"R": [0.1, 0.2, 0.4]}
    automaton = synthesise_stl_monitoring_automaton(
        "always (R >= 0.3)",
        trace,
    )
    mismatched = STLMonitoringAutomaton(
        spec=automaton.spec,
        temporal_op="eventually",
        signals=automaton.signals,
        states=automaton.states,
        transitions=automaton.transitions,
        robustness=automaton.robustness,
        satisfied=automaton.satisfied,
    )

    with pytest.raises(ValueError, match="temporal operator"):
        synthesise_stl_controller_candidates(mismatched, trace)


def test_stl_candidate_projection_uses_policy_templates_and_projector():
    trace = {"R": [0.1, 0.2, 0.75]}
    automaton = synthesise_stl_monitoring_automaton(
        "eventually (R >= 0.8)",
        trace,
    )
    synthesis = synthesise_stl_controller_candidates(
        automaton,
        trace,
        action_map={"R": "raise_coupling"},
    )

    plan = project_stl_controller_candidates(
        synthesis,
        (
            STLActionProjectionTemplate(
                action="raise_coupling",
                knob="K",
                scope="global",
                base_value=0.9,
                step=10.0,
                ttl_s=0.5,
                previous_value=0.9,
                value_bounds=(0.0, 1.0),
                rate_limit=0.05,
            ),
        ),
    )

    assert isinstance(plan, STLProjectedActionPlan)
    assert plan.actuating is False
    assert plan.rejected_candidates == ()
    assert len(plan.approved_actions) == 1
    action = plan.approved_actions[0]
    assert action.knob == "K"
    assert action.scope == "global"
    assert action.value == pytest.approx(0.95)
    assert action.ttl_s == pytest.approx(0.5)
    assert "STL candidate raise_coupling" in action.justification
    assert plan.to_audit_record()["approved_actions"] == [
        {
            "knob": "K",
            "scope": "global",
            "value": pytest.approx(0.95),
            "ttl_s": 0.5,
            "justification": action.justification,
        }
    ]


def test_stl_candidate_projection_rejects_unmapped_candidates_without_actuation():
    trace = {"R": [0.1, 0.2, 0.75]}
    automaton = synthesise_stl_monitoring_automaton(
        "eventually (R >= 0.8)",
        trace,
    )
    synthesis = synthesise_stl_controller_candidates(automaton, trace)

    plan = project_stl_controller_candidates(synthesis, ())

    assert plan.actuating is False
    assert plan.approved_actions == ()
    assert plan.rejected_candidates == (
        {
            "action": "increase_R",
            "signal": "R",
            "reason": "projection_template_missing",
        },
    )


def test_stl_projection_template_validation_raises_for_invalid_inputs():
    with pytest.raises(ValueError, match="projection value_bounds must be ordered"):
        STLActionProjectionTemplate(
            action="a",
            knob="K",
            scope="global",
            base_value=0.0,
            step=1.0,
            ttl_s=0.0,
            previous_value=0.0,
            value_bounds=(1.0, 0.0),
        )


def test_stl_closed_loop_plan_binds_projection_and_review_horizon():
    trace = {"R": [0.1, 0.2, 0.75]}
    automaton = synthesise_stl_monitoring_automaton(
        "eventually (R >= 0.8)",
        trace,
    )

    plan = synthesise_stl_closed_loop_plan(
        automaton,
        trace,
        (
            STLActionProjectionTemplate(
                action="raise_coupling",
                knob="K",
                scope="global",
                base_value=0.9,
                step=10.0,
                ttl_s=0.5,
                previous_value=0.9,
                value_bounds=(0.0, 1.0),
                rate_limit=0.05,
            ),
        ),
        horizon_steps=4,
        action_map={"R": "raise_coupling"},
    )

    assert isinstance(plan, STLClosedLoopSynthesisPlan)
    assert plan.spec == "eventually (R >= 0.8)"
    assert plan.trace_length == 3
    assert plan.horizon_steps == 4
    assert plan.next_review_start_index == 3
    assert plan.next_review_end_index == 6
    assert plan.feedback_signals == ("R",)
    assert plan.satisfied is False
    assert plan.actuating is False
    assert plan.blocked_reasons == ()
    assert len(plan.synthesis.candidates) == 1
    assert len(plan.projected_plan.approved_actions) == 1
    assert plan.projected_plan.approved_actions[0].value == pytest.approx(0.95)
    assert plan.runtime_gate == stl_module.STLRuntimeActuationGate(
        spec="eventually (R >= 0.8)",
        non_actuating=True,
        execution_disabled=True,
        accepted=True,
        action_count=1,
        mapper_valid_action_count=1,
        mapped_command_count=1,
        commands=(
            {
                "actuator": "stl_runtime_K_global",
                "value": pytest.approx(0.95),
                "knob": "K",
                "scope": "global",
                "ttl_s": 0.5,
            },
        ),
        blocked_reasons=(),
    )
    assert plan.to_audit_record()["projected_action_plan"]["approved_actions"] == [
        {
            "knob": "K",
            "scope": "global",
            "value": pytest.approx(0.95),
            "ttl_s": 0.5,
            "justification": (
                "STL candidate raise_coupling: "
                "R >= 0.8 violated at t=2 with robustness -0.05"
            ),
        }
    ]
    assert plan.to_audit_record()["runtime_actuation_gate"] == {
        "spec": "eventually (R >= 0.8)",
        "non_actuating": True,
        "execution_disabled": True,
        "accepted": True,
        "action_count": 1,
        "mapper_valid_action_count": 1,
        "mapped_command_count": 1,
        "commands": [
            {
                "actuator": "stl_runtime_K_global",
                "knob": "K",
                "scope": "global",
                "value": pytest.approx(0.95),
                "ttl_s": 0.5,
            }
        ],
        "blocked_reasons": [],
    }


def test_stl_closed_loop_plan_records_fail_closed_blockers():
    trace = {"R": [0.1, 0.2, 0.75]}
    automaton = synthesize_stl_monitoring_automaton(
        "eventually (R >= 0.8)",
        trace,
    )

    plan = synthesize_stl_closed_loop_plan(automaton, trace, (), horizon_steps=1)

    assert plan.actuating is False
    assert plan.projected_plan.approved_actions == ()
    assert plan.blocked_reasons == ("no_projected_actions", "unprojected_candidates")
    assert plan.runtime_gate.accepted is False
    assert plan.runtime_gate.action_count == 0
    assert plan.runtime_gate.mapped_command_count == 0
    assert plan.runtime_gate.blocked_reasons == ("no_runtime_actions",)
    assert plan.to_audit_record()["blocked_reasons"] == [
        "no_projected_actions",
        "unprojected_candidates",
    ]


def test_stl_closed_loop_plan_satisfied_monitor_never_projects_actions():
    trace = {"R": [0.8, 0.9]}
    automaton = synthesise_stl_monitoring_automaton(
        "always (R >= 0.3)",
        trace,
    )

    plan = synthesise_stl_closed_loop_plan(automaton, trace, (), horizon_steps=2)

    assert plan.satisfied is True
    assert plan.actuating is False
    assert plan.synthesis.candidates == ()
    assert plan.projected_plan.approved_actions == ()
    assert plan.blocked_reasons == ("stl_satisfied_no_control_needed",)
    assert plan.runtime_gate.accepted is False
    assert plan.runtime_gate.action_count == 0
    assert plan.runtime_gate.blocked_reasons == ("no_runtime_actions",)


def test_stl_runtime_actuation_gate_fails_closed_for_invalid_runtime_knob():
    synthesis = STLControllerSynthesis(
        spec="always (theta >= 0.0)",
        satisfied=False,
        actuating=False,
        source_backend="builtin",
        candidates=(
            STLControllerCandidate(
                signal="theta",
                action="raise_invalid_runtime_knob",
                direction="increase",
                time_index=1,
                robustness=-0.2,
                rationale="theta below runtime threshold",
            ),
        ),
    )
    template = STLActionProjectionTemplate(
        action="raise_invalid_runtime_knob",
        knob="theta_gain",
        scope="global",
        base_value=0.4,
        step=0.5,
        ttl_s=0.25,
        previous_value=0.4,
        value_bounds=(0.0, 1.0),
        rate_limit=0.25,
    )
    projection = project_stl_controller_candidates(synthesis, (template,))

    gate = stl_module.validate_stl_runtime_actuation_gate(projection, (template,))

    assert gate.accepted is False
    assert gate.non_actuating is True
    assert gate.execution_disabled is True
    assert gate.action_count == 1
    assert gate.mapper_valid_action_count == 0
    assert gate.mapped_command_count == 0
    assert gate.blocked_reasons == ("actuation_mapper_rejected_template",)


@pytest.mark.parametrize("bad_horizon", [0, -1, True, 1.5])
def test_stl_closed_loop_plan_rejects_invalid_horizon(bad_horizon):
    trace = {"R": [0.1, 0.2, 0.75]}
    automaton = synthesise_stl_monitoring_automaton(
        "eventually (R >= 0.8)",
        trace,
    )

    with pytest.raises(ValueError, match="horizon_steps"):
        synthesise_stl_closed_loop_plan(
            automaton,
            trace,
            (),
            horizon_steps=bad_horizon,
        )


def _state_by_name(
    states: tuple[STLAutomatonState, ...],
    name: str,
) -> STLAutomatonState:
    return next(state for state in states if state.name == name)
