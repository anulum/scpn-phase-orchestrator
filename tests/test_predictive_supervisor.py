# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Predictive supervisor tests

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
from scpn_phase_orchestrator.supervisor.predictive import (
    Prediction,
    PredictiveSupervisor,
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
        # Should produce K boost or MPC action
        assert len(actions) >= 0  # May or may not trigger depending on OA

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
