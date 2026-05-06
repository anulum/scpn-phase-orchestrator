# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — FEP predictive supervisor tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
from scpn_phase_orchestrator.supervisor import (
    FEPHierarchyAssessment,
    FEPHierarchyChildAssessment,
    FEPPredictionAssessment,
    FEPPredictiveSupervisor,
    assess_fep_hierarchy,
)
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState


def _state(r_value: float) -> UPDEState:
    return UPDEState(
        layers=[LayerState(R=r_value, psi=0.0)],
        cross_layer_alignment=np.eye(1),
        stability_proxy=r_value,
        regime_id="nominal",
    )


class TestFEPPredictiveSupervisorValidation:
    def test_rejects_invalid_constructor_values(self) -> None:
        with pytest.raises(ValueError, match="n_oscillators"):
            FEPPredictiveSupervisor(0, dt=0.01)
        with pytest.raises(ValueError, match="dt"):
            FEPPredictiveSupervisor(4, dt=0.0)
        with pytest.raises(ValueError, match="target_R"):
            FEPPredictiveSupervisor(4, dt=0.01, target_R=1.5)

    def test_rejects_bad_phase_shapes(self) -> None:
        supervisor = FEPPredictiveSupervisor(4, dt=0.01)
        with pytest.raises(ValueError, match="phases"):
            supervisor.assess(np.zeros(3), np.ones(4))
        with pytest.raises(ValueError, match="omegas"):
            supervisor.assess(np.zeros(4), np.ones(3))


class TestFEPPredictionAssessment:
    def test_assess_returns_audit_ready_metrics(self) -> None:
        supervisor = FEPPredictiveSupervisor(4, dt=0.01, target_R=0.8)

        assessment = supervisor.assess(np.zeros(4), np.ones(4))

        assert isinstance(assessment, FEPPredictionAssessment)
        assert np.isfinite(assessment.free_energy)
        assert np.isfinite(assessment.mean_abs_error)
        assert 0.0 <= assessment.observed_R <= 1.0
        assert 0.0 <= assessment.predicted_R <= 1.0
        assert assessment.to_audit_record()["target_R"] == 0.8
        assert supervisor.last_assessment is assessment

    def test_reset_clears_last_assessment(self) -> None:
        supervisor = FEPPredictiveSupervisor(3, dt=0.01)
        supervisor.assess(np.zeros(3), np.ones(3))
        assert supervisor.last_assessment is not None

        supervisor.reset()

        assert supervisor.last_assessment is None


class TestFEPPredictiveDecisions:
    def test_hard_boundary_violation_fails_closed(self) -> None:
        supervisor = FEPPredictiveSupervisor(4, dt=0.01, drive_gain=0.2)

        actions = supervisor.decide(
            np.zeros(4),
            np.ones(4),
            _state(0.9),
            BoundaryState(hard_violations=["R_floor"]),
        )

        assert len(actions) == 1
        assert actions[0].knob == "zeta"
        assert actions[0].value == pytest.approx(0.2)

    def test_low_coherence_high_surprise_drives_observed_phase(self) -> None:
        phases = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
        supervisor = FEPPredictiveSupervisor(
            8,
            dt=0.01,
            target_R=0.8,
            free_energy_threshold=0.0,
            drive_gain=0.15,
        )

        actions = supervisor.decide(phases, np.ones(8), _state(0.2), BoundaryState())

        assert [action.knob for action in actions] == ["zeta", "Psi"]
        assert actions[0].value == pytest.approx(0.15)
        assert "FEP-MPC" in actions[0].justification
        assert 0.0 <= actions[1].value < 2.0 * np.pi

    def test_above_target_uses_antiphase_target(self) -> None:
        phases = np.zeros(6)
        supervisor = FEPPredictiveSupervisor(
            6,
            dt=0.01,
            target_R=0.4,
            free_energy_threshold=0.0,
        )

        actions = supervisor.decide(phases, np.ones(6), _state(1.0), BoundaryState())

        assert actions[1].knob == "Psi"
        assert actions[1].value == pytest.approx(np.pi)

    def test_no_action_when_thresholds_not_crossed_and_state_stable(self) -> None:
        supervisor = FEPPredictiveSupervisor(
            4,
            dt=0.01,
            target_R=0.5,
            free_energy_threshold=1e9,
            error_threshold=1e9,
        )

        actions = supervisor.decide(
            np.zeros(4), np.ones(4), _state(0.9), BoundaryState()
        )

        assert actions == []


class TestFEPPipelineWiring:
    def test_engine_phases_feed_fep_supervisor_actions(self) -> None:
        n = 6
        engine = UPDEEngine(n, dt=0.01)
        phases = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        omegas = np.linspace(0.8, 1.3, n)
        knm = 0.05 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))

        for _ in range(5):
            phases = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)

        supervisor = FEPPredictiveSupervisor(
            n,
            dt=0.01,
            target_R=0.9,
            free_energy_threshold=0.0,
        )
        actions = supervisor.decide(phases, omegas, _state(0.3), BoundaryState())

        assert isinstance(supervisor.last_assessment, FEPPredictionAssessment)
        assert len(actions) == 2
        assert {action.knob for action in actions} == {"zeta", "Psi"}


class TestFEPHierarchyAssessment:
    def test_assess_fep_hierarchy_emits_child_and_parent_audit_records(self) -> None:
        children = {
            "coherent_child": (
                np.array([0.0, 0.02, 0.04], dtype=np.float64),
                np.array([1.0, 1.0, 1.0], dtype=np.float64),
            ),
            "dispersed_child": (
                np.array([0.0, 2.1, 4.2], dtype=np.float64),
                np.array([0.8, 1.1, 1.4], dtype=np.float64),
            ),
        }

        hierarchy = assess_fep_hierarchy(
            children,
            dt=0.01,
            parent_dt=0.05,
            child_target_R=0.75,
            parent_target_R=0.7,
            free_energy_threshold=0.0,
            hierarchy="unit_test_hierarchy",
        )
        record = hierarchy.to_audit_record()

        assert isinstance(hierarchy, FEPHierarchyAssessment)
        assert all(
            isinstance(child, FEPHierarchyChildAssessment)
            for child in hierarchy.children
        )
        assert record["hierarchy"] == "unit_test_hierarchy"
        assert [child["name"] for child in record["children"]] == [
            "coherent_child",
            "dispersed_child",
        ]
        assert len(record["child_R_values"]) == 2
        assert len(record["parent_phase_encoding"]) == 2
        assert record["parent"]["actions"]
        assert 0.0 <= record["parent"]["assessment"]["observed_R"] <= 1.0

    def test_assess_fep_hierarchy_rejects_invalid_child_inputs(self) -> None:
        with pytest.raises(ValueError, match="at least one child"):
            assess_fep_hierarchy({}, dt=0.01)

        with pytest.raises(ValueError, match="non-empty strings"):
            assess_fep_hierarchy(
                {"": (np.zeros(2), np.ones(2))},
                dt=0.01,
            )

        with pytest.raises(ValueError, match="omegas must match"):
            assess_fep_hierarchy(
                {"bad_shape": (np.zeros(3), np.ones(2))},
                dt=0.01,
            )

    def test_assess_fep_hierarchy_is_exported_from_supervisor(self) -> None:
        import scpn_phase_orchestrator.supervisor as supervisor

        assert supervisor.assess_fep_hierarchy is assess_fep_hierarchy
