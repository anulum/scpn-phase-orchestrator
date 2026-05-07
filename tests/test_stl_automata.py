# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — STL automata tests

from __future__ import annotations

import pytest

from scpn_phase_orchestrator.monitor.stl import (
    STLAutomatonState,
    STLAutomatonTransition,
    STLControllerCandidate,
    STLControllerSynthesis,
    STLMonitoringAutomaton,
    synthesise_stl_controller_candidates,
    synthesise_stl_monitoring_automaton,
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


def _state_by_name(
    states: tuple[STLAutomatonState, ...],
    name: str,
) -> STLAutomatonState:
    return next(state for state in states if state.name == name)
