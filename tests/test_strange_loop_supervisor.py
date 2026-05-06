# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Strange-loop supervisor tests

from __future__ import annotations

from typing import get_type_hints

import numpy as np
import pytest

from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
from scpn_phase_orchestrator.supervisor import (
    RegimeManager,
    StrangeLoopAssessment,
    StrangeLoopSupervisor,
    SupervisorPolicy,
)
from scpn_phase_orchestrator.supervisor import strange_loop as strange_loop_module
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState


def _action(knob: str, value: float) -> ControlAction:
    return ControlAction(
        knob=knob,
        scope="global",
        value=value,
        ttl_s=1.0,
        justification="test action",
    )


def _state(r_value: float) -> UPDEState:
    return UPDEState(
        layers=[LayerState(R=r_value, psi=0.0)],
        cross_layer_alignment=np.eye(1),
        stability_proxy=r_value,
        regime_id="nominal",
    )


class TestStrangeLoopValidation:
    def test_rejects_invalid_constructor_values(self) -> None:
        with pytest.raises(ValueError, match="history_size"):
            StrangeLoopSupervisor(history_size=1)
        with pytest.raises(ValueError, match="damping_gain"):
            StrangeLoopSupervisor(damping_gain=0.0)
        with pytest.raises(ValueError, match="ttl_s"):
            StrangeLoopSupervisor(ttl_s=float("inf"))

    def test_rejects_non_finite_action_values(self) -> None:
        supervisor = StrangeLoopSupervisor()
        with pytest.raises(ValueError, match="finite"):
            supervisor.observe([_action("K", float("nan"))])


class TestStrangeLoopAssessment:
    def test_internal_array_helpers_have_parameterised_signatures(self) -> None:
        vector_hints = get_type_hints(strange_loop_module._actions_to_vector)
        coherence_hints = get_type_hints(strange_loop_module._control_coherence)
        drift_hints = get_type_hints(strange_loop_module._drift_score)
        oscillation_hints = get_type_hints(strange_loop_module._oscillation_score)

        assert "numpy.ndarray" in str(vector_hints["return"])
        assert "float64" in str(vector_hints["return"])
        for hints in (coherence_hints, drift_hints, oscillation_hints):
            assert "numpy.ndarray" in str(hints["matrix"])
            assert "float64" in str(hints["matrix"])

    def test_actions_to_vector_returns_float64_vector(self) -> None:
        vector = strange_loop_module._actions_to_vector(
            [_action("K", 0.2), _action("zeta", -0.1)]
        )

        assert vector.dtype == np.float64
        assert vector.tolist() == pytest.approx([0.2, 0.0, -0.1, 0.0])

    def test_quiet_supervisor_needs_no_damping(self) -> None:
        supervisor = StrangeLoopSupervisor()

        assessment = supervisor.observe([])

        assert isinstance(assessment, StrangeLoopAssessment)
        assert assessment.control_coherence == pytest.approx(1.0)
        assert assessment.drift_score == pytest.approx(0.0)
        assert assessment.oscillation_score == pytest.approx(0.0)
        assert assessment.recommended_actions == ()
        assert supervisor.last_assessment is assessment

    def test_overcontrol_emits_damping_actions(self) -> None:
        supervisor = StrangeLoopSupervisor(overcontrol_threshold=0.05)

        assessment = supervisor.observe([_action("K", 0.2)])

        assert assessment.overcontrol_score > 0.05
        assert [action.knob for action in assessment.recommended_actions] == [
            "zeta",
            "K",
        ]
        assert "over-control" in assessment.recommended_actions[0].justification
        assert assessment.recommended_actions[1].value < 0.0

    def test_oscillating_action_history_is_detected(self) -> None:
        supervisor = StrangeLoopSupervisor(
            oscillation_threshold=0.2,
            overcontrol_threshold=10.0,
            drift_threshold=10.0,
        )

        supervisor.observe([_action("K", 0.1)])
        supervisor.observe([_action("K", -0.1)])
        assessment = supervisor.observe([_action("K", 0.1)])

        assert assessment.oscillation_score > 0.2
        assert assessment.recommended_actions
        assert (
            "control-loop oscillation"
            in assessment.recommended_actions[0].justification
        )

    def test_drift_history_is_serialisable(self) -> None:
        supervisor = StrangeLoopSupervisor(drift_threshold=0.01)

        supervisor.observe([_action("K", 0.01)])
        assessment = supervisor.observe([_action("Psi", 0.4)])
        record = assessment.to_audit_record()

        assert record["drift_score"] == assessment.drift_score
        assert record["recommended_actions"]
        assert set(record) == {
            "control_phase",
            "control_coherence",
            "drift_score",
            "oscillation_score",
            "overcontrol_score",
            "recommended_actions",
        }

    def test_reset_clears_history_and_last_assessment(self) -> None:
        supervisor = StrangeLoopSupervisor()
        supervisor.observe([_action("K", 0.1)])

        supervisor.reset()

        assert supervisor.last_assessment is None
        assert supervisor.observe([]).drift_score == pytest.approx(0.0)


class TestStrangeLoopPipelineWiring:
    def test_supervisor_policy_actions_feed_strange_loop_guard(self) -> None:
        policy = SupervisorPolicy(RegimeManager(cooldown_steps=0))
        strange_loop = StrangeLoopSupervisor(overcontrol_threshold=0.04)

        actions = policy.decide(_state(0.4), BoundaryState())
        assessment = strange_loop.observe(actions)

        assert actions
        assert assessment.recommended_actions
        assert all(
            action.scope == "global" for action in assessment.recommended_actions
        )

    def test_exported_from_supervisor_package(self) -> None:
        import scpn_phase_orchestrator.supervisor as supervisor

        assert supervisor.StrangeLoopSupervisor is StrangeLoopSupervisor
        assert supervisor.StrangeLoopAssessment is StrangeLoopAssessment
