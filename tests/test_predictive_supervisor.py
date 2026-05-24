# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Predictive supervisor tests

from __future__ import annotations

import numpy as np
import pytest

import scpn_phase_orchestrator.supervisor.predictive as predictive
from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
from scpn_phase_orchestrator.supervisor.predictive import (
    FEPHierarchyAssessment,
    FEPHierarchyChildAssessment,
    FEPPredictionAssessment,
    FEPPredictiveSupervisor,
    Prediction,
    PredictiveSupervisor,
    assess_fep_hierarchy,
)
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState


def _make_state(R: float) -> UPDEState:
    return UPDEState(
        layers=[LayerState(R=R, psi=0.0)],
        cross_layer_alignment=np.eye(1),
        stability_proxy=R,
        regime_id="nominal",
    )


class TestPrediction:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"n_oscillators": 0, "dt": 0.01}, "n_oscillators"),
            ({"n_oscillators": True, "dt": 0.01}, "n_oscillators"),
            ({"n_oscillators": 4, "dt": 0.0}, "dt"),
            ({"n_oscillators": 4, "dt": np.nan}, "dt"),
            ({"n_oscillators": 4, "dt": 0.01, "horizon": 0}, "horizon"),
            ({"n_oscillators": 4, "dt": 0.01, "horizon": 1.5}, "horizon"),
            (
                {
                    "n_oscillators": 4,
                    "dt": 0.01,
                    "divergence_threshold": -0.1,
                },
                "divergence_threshold",
            ),
            (
                {
                    "n_oscillators": 4,
                    "dt": 0.01,
                    "divergence_threshold": np.inf,
                },
                "divergence_threshold",
            ),
        ],
    )
    def test_constructor_rejects_invalid_bounds(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            PredictiveSupervisor(**kwargs)

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("phases", np.zeros(3), "phases"),
            ("omegas", np.zeros(3), "omegas"),
            ("knm", np.zeros((4, 3)), "knm"),
            ("alpha", np.zeros((3, 4)), "alpha"),
            ("phases", np.array([0.0, np.nan, 0.0, 0.0]), "phases"),
            ("omegas", np.array([1.0, np.inf, 1.0, 1.0]), "omegas"),
            ("knm", np.full((4, 4), np.nan), "knm"),
            ("alpha", np.full((4, 4), np.inf), "alpha"),
        ],
    )
    def test_predict_rejects_invalid_inputs(self, field, value, match):
        values = {
            "phases": np.zeros(4),
            "omegas": np.ones(4),
            "knm": np.zeros((4, 4)),
            "alpha": np.zeros((4, 4)),
        }
        values[field] = value
        supervisor = PredictiveSupervisor(4, dt=0.01)
        with pytest.raises(ValueError, match=match):
            supervisor.predict(
                values["phases"],
                values["omegas"],
                values["knm"],
                values["alpha"],
            )

    def test_returns_prediction(self):
        ps = PredictiveSupervisor(8, dt=0.01, horizon=10)
        phases = np.zeros(8)
        omegas = np.ones(8)
        knm = np.full((8, 8), 0.5)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((8, 8))
        pred = ps.predict(phases, omegas, knm, alpha)
        assert isinstance(pred, Prediction)
        assert len(pred.R_predicted) == 11

    def test_synced_no_degradation(self):
        ps = PredictiveSupervisor(8, dt=0.01, horizon=10)
        phases = np.zeros(8)
        omegas = np.ones(8)
        knm = np.full((8, 8), 2.0)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((8, 8))
        pred = ps.predict(phases, omegas, knm, alpha)
        assert not pred.will_critical

    def test_divergence_fallback_writes_constant_trajectory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class ExplodingReduction:
            def __init__(
                self, omega_0: float, delta: float, k_eff: float, dt: float
            ) -> None:
                self._counter = 0

            def step(self, z: complex) -> complex:
                self._counter += 1
                return z + 0.5

        monkeypatch.setattr(predictive, "OttAntonsenReduction", ExplodingReduction)

        ps = PredictiveSupervisor(4, dt=0.01, horizon=4, divergence_threshold=0.0)
        phases = np.zeros(4)
        omegas = np.ones(4)
        knm = np.zeros((4, 4))
        alpha = np.zeros((4, 4))

        pred = ps.predict(phases, omegas, knm, alpha)

        assert pred.R_predicted == [1.0] * 5
        assert pred.will_degrade is False
        assert pred.will_critical is False
        assert pred.steps_to_degradation == 4

    def test_weak_coupling_predicts_degradation(self):
        ps = PredictiveSupervisor(8, dt=0.01, horizon=50)
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, 8)
        omegas = rng.uniform(0.5, 1.5, 8)
        knm = np.full((8, 8), 0.01)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((8, 8))
        pred = ps.predict(phases, omegas, knm, alpha)
        assert pred.steps_to_degradation <= 50


class TestDecide:
    def test_no_action_when_stable(self):
        ps = PredictiveSupervisor(8, dt=0.01)
        phases = np.zeros(8)
        omegas = np.ones(8)
        knm = np.full((8, 8), 2.0)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((8, 8))
        actions = ps.decide(
            phases, omegas, knm, alpha, _make_state(0.9), BoundaryState()
        )
        assert len(actions) == 0

    def test_hard_violation_overrides(self):
        ps = PredictiveSupervisor(4, dt=0.01)
        phases = np.zeros(4)
        omegas = np.ones(4)
        knm = np.eye(4)
        alpha = np.zeros((4, 4))
        bs = BoundaryState(hard_violations=["test"])
        actions = ps.decide(phases, omegas, knm, alpha, _make_state(0.9), bs)
        assert len(actions) == 1
        assert actions[0].knob == "zeta"

    @pytest.mark.parametrize(
        ("upde_state", "boundary_state", "match"),
        [
            (object(), BoundaryState(), "upde_state"),
            (_make_state(0.9), object(), "boundary_state"),
            (_make_state(0.9), True, "boundary_state"),
        ],
    )
    def test_decide_rejects_malformed_state_objects(
        self,
        upde_state: object,
        boundary_state: object,
        match: str,
    ):
        ps = PredictiveSupervisor(4, dt=0.01)

        with pytest.raises(ValueError, match=match):
            ps.decide(
                np.zeros(4),
                np.ones(4),
                np.eye(4),
                np.zeros((4, 4)),
                upde_state,  # type: ignore[arg-type]
                boundary_state,  # type: ignore[arg-type]
            )

    def test_preemptive_action_on_predicted_degradation(self):
        ps = PredictiveSupervisor(8, dt=0.01, horizon=50)
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, 8)
        omegas = rng.uniform(0.5, 2.0, 8)
        knm = np.full((8, 8), 0.01)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((8, 8))
        actions = ps.decide(
            phases, omegas, knm, alpha, _make_state(0.5), BoundaryState()
        )
        if actions:
            assert "MPC" in actions[0].justification

    def test_prediction_fields(self):
        pred = Prediction(
            R_predicted=[0.9, 0.85, 0.8],
            will_degrade=False,
            will_critical=False,
            steps_to_degradation=3,
        )
        assert len(pred.R_predicted) == 3
        assert pred.steps_to_degradation == 3

    def test_divergence_fallback(self):
        """OA diverges → trajectory falls back to constant R_current."""
        ps = PredictiveSupervisor(8, dt=0.01, horizon=5, divergence_threshold=0.01)
        rng = np.random.default_rng(99)
        phases = rng.uniform(0, 2 * np.pi, 8)
        omegas = rng.uniform(-10, 10, 8)
        knm = np.full((8, 8), 5.0)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((8, 8))
        pred = ps.predict(phases, omegas, knm, alpha)
        assert len(pred.R_predicted) == 6

    def test_will_critical_action(self):
        """Predicted R < 0.3 → K boost action."""
        ps = PredictiveSupervisor(8, dt=0.01, horizon=10)
        # Nearly desynchronised phases → low R → OA predicts critical
        phases = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        omegas = np.linspace(-5, 5, 8)
        knm = np.full((8, 8), 0.001)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((8, 8))
        actions = ps.decide(
            phases, omegas, knm, alpha, _make_state(0.15), BoundaryState()
        )
        assert len(actions) == 1
        assert actions[0].knob == "K"
        assert actions[0].value == pytest.approx(0.2)
        assert "CRITICAL" in actions[0].justification

    def test_zero_step_critical_prediction_triggers_max_boost(self):
        ps = PredictiveSupervisor(2, dt=0.01, horizon=6, divergence_threshold=0.0)
        phases = np.array([0.0, np.pi], dtype=np.float64)
        omegas = np.ones(2, dtype=np.float64)
        knm = np.zeros((2, 2), dtype=np.float64)
        alpha = np.zeros((2, 2), dtype=np.float64)

        pred = ps.predict(phases, omegas, knm, alpha)
        actions = ps.decide(
            phases, omegas, knm, alpha, _make_state(0.5), BoundaryState()
        )

        assert pred.steps_to_degradation == 0
        assert pred.will_critical
        assert len(actions) == 1
        assert actions[0].knob == "K"
        assert actions[0].value == pytest.approx(0.2)
        assert "in 0 steps" in actions[0].justification

    def test_hard_violation_action(self):
        """Hard boundary violation → zeta=0.1 override."""
        ps = PredictiveSupervisor(4, dt=0.01)
        phases = np.zeros(4)
        omegas = np.zeros(4)
        knm = np.eye(4)
        alpha = np.zeros((4, 4))
        bstate = BoundaryState(hard_violations=["test"])
        actions = ps.decide(phases, omegas, knm, alpha, _make_state(0.5), bstate)
        assert len(actions) == 1
        assert actions[0].knob == "zeta"


class TestPredictiveSupervisorPipelineWiring:
    """Pipeline: engine → phases → predictive supervisor → MPC actions."""

    def test_engine_phases_to_predictive_actions(self):
        """UPDEEngine → phases → PredictiveSupervisor.decide →
        actions based on OA prediction. Full MPC feedback loop."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import (
            compute_order_parameter,
        )

        n = 8
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = rng.normal(1.0, 0.5, n)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))

        for _ in range(100):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        r, _ = compute_order_parameter(phases)

        ps = PredictiveSupervisor(n, dt=0.01, horizon=5)
        actions = ps.decide(
            phases,
            omegas,
            knm,
            alpha,
            _make_state(r),
            BoundaryState(),
        )
        assert isinstance(actions, list)
        assert 0.0 <= r <= 1.0


class TestFEPPredictiveSupervisorInPredictiveCoverage:
    def test_fep_assessment_updates_audit_record_and_reset_state(self):
        supervisor = FEPPredictiveSupervisor(4, dt=0.01, target_R=0.75)
        phases = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float64)
        omegas = np.array([0.8, 0.9, 1.1, 1.2], dtype=np.float64)

        assessment = supervisor.assess(phases, omegas)
        record = assessment.to_audit_record()

        assert supervisor.target_R == pytest.approx(0.75)
        assert supervisor.last_assessment is assessment
        assert isinstance(assessment, FEPPredictionAssessment)
        assert set(record) == {
            "free_energy",
            "complexity",
            "mean_abs_error",
            "precision_mean",
            "precision_spread",
            "observed_R",
            "observed_psi",
            "predicted_R",
            "target_R",
            "surprise",
        }
        assert 0.0 <= record["observed_R"] <= 1.0
        assert 0.0 <= record["predicted_R"] <= 1.0

        supervisor.reset()

        assert supervisor.last_assessment is None

    def test_fep_hard_boundary_violation_fails_closed_before_assessment(self):
        supervisor = FEPPredictiveSupervisor(4, dt=0.01, drive_gain=0.12)

        actions = supervisor.decide(
            np.zeros(4),
            np.ones(4),
            _make_state(0.9),
            BoundaryState(hard_violations=["R_floor"]),
        )

        assert len(actions) == 1
        assert actions[0].knob == "zeta"
        assert actions[0].value == pytest.approx(0.12)
        assert actions[0].justification == "FEP-MPC: hard boundary violation"
        assert supervisor.last_assessment is None

    def test_fep_error_threshold_ranks_zeta_before_antiphase_target(self):
        supervisor = FEPPredictiveSupervisor(
            4,
            dt=0.01,
            target_R=0.5,
            free_energy_threshold=1e9,
            error_threshold=0.0,
            drive_gain=0.2,
        )

        actions = supervisor.decide(
            np.zeros(4),
            np.ones(4),
            _make_state(1.0),
            BoundaryState(),
        )

        assert [action.knob for action in actions] == ["zeta", "Psi"]
        assert actions[0].value == pytest.approx(0.2)
        assert "free energy" in actions[0].justification
        assert actions[1].value == pytest.approx(np.pi)

    def test_fep_stability_risk_branch_acts_without_threshold_breach(self):
        supervisor = FEPPredictiveSupervisor(
            6,
            dt=0.01,
            target_R=0.95,
            free_energy_threshold=1e9,
            error_threshold=1e9,
            drive_gain=0.07,
        )
        phases = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)

        actions = supervisor.decide(
            phases,
            np.ones(6),
            _make_state(0.2),
            BoundaryState(),
        )

        assert [action.knob for action in actions] == ["zeta", "Psi"]
        assert actions[0].value == pytest.approx(0.07)
        assert 0.0 <= actions[1].value < 2.0 * np.pi

    def test_fep_stable_state_does_not_emit_actions(self):
        supervisor = FEPPredictiveSupervisor(
            4,
            dt=0.01,
            target_R=0.5,
            free_energy_threshold=1e9,
            error_threshold=1e9,
        )

        actions = supervisor.decide(
            np.zeros(4),
            np.ones(4),
            _make_state(0.9),
            BoundaryState(),
        )

        assert actions == []

    @pytest.mark.parametrize(
        ("upde_state", "boundary_state", "match"),
        [
            (object(), BoundaryState(), "upde_state"),
            (_make_state(0.9), object(), "boundary_state"),
            (_make_state(0.9), True, "boundary_state"),
        ],
    )
    def test_fep_decide_rejects_malformed_state_objects(
        self,
        upde_state: object,
        boundary_state: object,
        match: str,
    ):
        supervisor = FEPPredictiveSupervisor(4, dt=0.01)

        with pytest.raises(ValueError, match=match):
            supervisor.decide(
                np.zeros(4),
                np.ones(4),
                upde_state,  # type: ignore[arg-type]
                boundary_state,  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"n_oscillators": 0, "dt": 0.01}, "n_oscillators"),
            ({"n_oscillators": True, "dt": 0.01}, "n_oscillators"),
            ({"n_oscillators": 4, "dt": 0.0}, "dt"),
            ({"n_oscillators": 4, "dt": True}, "dt"),
            ({"n_oscillators": 4, "dt": 0.01, "target_R": 1.1}, "target_R"),
            ({"n_oscillators": 4, "dt": 0.01, "target_R": True}, "target_R"),
            (
                {"n_oscillators": 4, "dt": 0.01, "free_energy_threshold": -0.1},
                "free_energy_threshold",
            ),
            (
                {"n_oscillators": 4, "dt": 0.01, "free_energy_threshold": "0.1"},
                "free_energy_threshold",
            ),
            (
                {"n_oscillators": 4, "dt": 0.01, "error_threshold": np.inf},
                "error_threshold",
            ),
            ({"n_oscillators": 4, "dt": 0.01, "drive_gain": -0.1}, "drive_gain"),
            ({"n_oscillators": 4, "dt": 0.01, "drive_gain": True}, "drive_gain"),
            (
                {"n_oscillators": 4, "dt": 0.01, "learning_rate": np.nan},
                "learning_rate",
            ),
            (
                {"n_oscillators": 4, "dt": 0.01, "prior_precision": -1.0},
                "prior_precision",
            ),
        ],
    )
    def test_fep_constructor_rejects_invalid_bounds(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            FEPPredictiveSupervisor(**kwargs)

    def test_fep_assess_rejects_invalid_phase_and_omega_vectors(self):
        supervisor = FEPPredictiveSupervisor(4, dt=0.01)

        with pytest.raises(ValueError, match="phases must have shape"):
            supervisor.assess(np.zeros(3), np.ones(4))
        with pytest.raises(ValueError, match="omegas must have shape"):
            supervisor.assess(np.zeros(4), np.ones(3))
        with pytest.raises(ValueError, match="phases must be finite"):
            supervisor.assess(np.array([0.0, np.nan, 0.2, 0.3]), np.ones(4))
        with pytest.raises(ValueError, match="omegas must be finite"):
            supervisor.assess(np.zeros(4), np.array([1.0, 1.1, np.inf, 1.3]))


class TestFEPHierarchyInPredictiveCoverage:
    def test_fep_hierarchy_serialises_child_parent_actions_and_phase_encoding(self):
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
            parent_dt=0.02,
            child_target_R=0.75,
            parent_target_R=0.7,
            free_energy_threshold=0.0,
            child_drive_gain=0.08,
            parent_drive_gain=0.05,
            hierarchy="predictive_coverage_hierarchy",
        )
        record = hierarchy.to_audit_record()

        assert isinstance(hierarchy, FEPHierarchyAssessment)
        assert all(
            isinstance(child, FEPHierarchyChildAssessment)
            for child in hierarchy.children
        )
        assert record["hierarchy"] == "predictive_coverage_hierarchy"
        assert [child["name"] for child in record["children"]] == [
            "coherent_child",
            "dispersed_child",
        ]
        assert len(record["child_R_values"]) == 2
        assert len(record["parent_phase_encoding"]) == 2
        assert all(0.0 <= value <= np.pi for value in record["parent_phase_encoding"])
        assert record["parent"]["actions"]
        assert all(child["actions"] for child in record["children"])
        assert record["parent"]["assessment"]["target_R"] == pytest.approx(0.7)

    def test_fep_hierarchy_assessment_is_deterministic_for_identical_inputs(self):
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

        first = assess_fep_hierarchy(
            children,
            dt=0.01,
            parent_dt=0.02,
            child_target_R=0.75,
            parent_target_R=0.7,
            free_energy_threshold=0.0,
            child_drive_gain=0.08,
            parent_drive_gain=0.05,
            hierarchy="predictive_coverage_hierarchy",
        )
        second = assess_fep_hierarchy(
            children,
            dt=0.01,
            parent_dt=0.02,
            child_target_R=0.75,
            parent_target_R=0.7,
            free_energy_threshold=0.0,
            child_drive_gain=0.08,
            parent_drive_gain=0.05,
            hierarchy="predictive_coverage_hierarchy",
        )

        assert first.to_audit_record() == second.to_audit_record()

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"children": {}, "dt": 0.01}, "at least one child"),
            (
                {"children": {"a": (np.zeros(2), np.ones(2))}, "dt": 0.0},
                "dt",
            ),
            (
                {"children": {"a": (np.zeros(2), np.ones(2))}, "dt": True},
                "dt",
            ),
            (
                {"children": {"a": (np.zeros(2), np.ones(2))}, "dt": "0.01"},
                "dt",
            ),
            (
                {
                    "children": {"a": (np.zeros(2), np.ones(2))},
                    "dt": 0.01,
                    "parent_dt": np.inf,
                },
                "parent_dt",
            ),
            (
                {
                    "children": {"a": (np.zeros(2), np.ones(2))},
                    "dt": 0.01,
                    "parent_dt": True,
                },
                "parent_dt",
            ),
            (
                {
                    "children": {"a": (np.zeros(2), np.ones(2))},
                    "dt": 0.01,
                    "parent_dt": "0.01",
                },
                "parent_dt",
            ),
            (
                {
                    "children": {"a": (np.zeros(2), np.ones(2))},
                    "dt": 0.01,
                    "child_target_R": -0.1,
                },
                "child_target_R",
            ),
            (
                {
                    "children": {"a": (np.zeros(2), np.ones(2))},
                    "dt": 0.01,
                    "parent_target_R": 1.1,
                },
                "parent_target_R",
            ),
            (
                {
                    "children": {"a": (np.zeros(2), np.ones(2))},
                    "dt": 0.01,
                    "free_energy_threshold": -0.1,
                },
                "free_energy_threshold",
            ),
            (
                {
                    "children": {"a": (np.zeros(2), np.ones(2))},
                    "dt": 0.01,
                    "child_drive_gain": np.nan,
                },
                "child_drive_gain",
            ),
            (
                {
                    "children": {"a": (np.zeros(2), np.ones(2))},
                    "dt": 0.01,
                    "parent_drive_gain": -0.1,
                },
                "parent_drive_gain",
            ),
        ],
    )
    def test_fep_hierarchy_rejects_invalid_global_bounds(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            assess_fep_hierarchy(**kwargs)

    @pytest.mark.parametrize(
        ("children", "match"),
        [
            ({"": (np.zeros(2), np.ones(2))}, "non-empty strings"),
            ({"empty": (np.empty((0,), dtype=np.float64), np.ones(0))}, "phases"),
            ({"rank": (np.zeros((1, 2)), np.ones((1, 2)))}, "phases"),
            ({"shape": (np.zeros(3), np.ones(2))}, "omegas must match"),
            (
                {"phase_nan": (np.array([0.0, np.nan]), np.ones(2))},
                "phases must be finite",
            ),
            (
                {"omega_nan": (np.zeros(2), np.array([1.0, np.nan]))},
                "omegas must be finite",
            ),
        ],
    )
    def test_fep_hierarchy_rejects_invalid_child_observations(self, children, match):
        with pytest.raises(ValueError, match=match):
            assess_fep_hierarchy(children, dt=0.01)

    def test_fep_hierarchy_rejects_non_mapping_children(self):
        with pytest.raises(ValueError, match="children must be a mapping"):
            assess_fep_hierarchy("bad-children", dt=0.01)

    def test_fep_hierarchy_parent_phase_encoding_extremes(self):
        children = {
            "coherent_child": (
                np.array([0.0, 0.0], dtype=np.float64),
                np.array([1.0, 1.0], dtype=np.float64),
            ),
            "antiphase_child": (
                np.array([0.0, np.pi], dtype=np.float64),
                np.array([1.0, 1.0], dtype=np.float64),
            ),
        }

        hierarchy = assess_fep_hierarchy(children, dt=0.01)

        assert hierarchy.parent_phase_encoding == (
            pytest.approx(0.0),
            pytest.approx(np.pi),
        )


# Salvaged module-specific behavioural contracts from deleted broad tests.
class TestPredictiveSupervisorBehavioural:
    """Verify that predictive supervisor makes future predictions
    with valid fields and sensible structure."""

    def _make_supervisor(self, n=8):
        from scpn_phase_orchestrator.supervisor.predictive import PredictiveSupervisor

        return PredictiveSupervisor(
            n_oscillators=n,
            dt=0.01,
            horizon=5,
            divergence_threshold=0.01,
        )

    def test_prediction_has_required_fields(self):
        ps = self._make_supervisor()
        phases = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        omegas = np.random.default_rng(0).normal(0, 5, 8)
        knm = np.ones((8, 8)) * 0.01
        np.fill_diagonal(knm, 0)
        alpha = np.zeros((8, 8))
        pred = ps.predict(phases, omegas, knm, alpha)
        assert hasattr(pred, "will_degrade")
        assert hasattr(pred, "will_critical")
        assert isinstance(pred.will_degrade, bool)
        assert isinstance(pred.will_critical, bool)

    def test_stable_system_not_predicted_critical(self):
        """Strong coupling + low omegas → should not predict critical."""
        ps = self._make_supervisor()
        phases = np.zeros(8)  # synchronised
        omegas = np.zeros(8)  # no drift
        knm = np.ones((8, 8)) * 2.0  # strong coupling
        np.fill_diagonal(knm, 0)
        alpha = np.zeros((8, 8))
        pred = ps.predict(phases, omegas, knm, alpha)
        # Synchronised with strong coupling should not predict degradation
        assert not pred.will_critical
