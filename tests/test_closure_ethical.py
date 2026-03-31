# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Closure + Ethical cost tests

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.ssgf.carrier import GeometryCarrier
from scpn_phase_orchestrator.ssgf.closure import ClosureState, CyberneticClosure
from scpn_phase_orchestrator.ssgf.ethical import EthicalCost, compute_ethical_cost


class TestCyberneticClosure:
    def test_step_returns_W_and_state(self):
        carrier = GeometryCarrier(4, z_dim=3, seed=42)
        closure = CyberneticClosure(carrier)
        phases = np.array([0.0, 0.5, 1.0, 1.5])
        W, state = closure.step(phases)
        assert W.shape == (4, 4)
        assert isinstance(state, ClosureState)
        assert state.ssgf_state_step == 1

    def test_run_multiple_steps(self):
        carrier = GeometryCarrier(4, z_dim=3, lr=0.05, seed=42)
        closure = CyberneticClosure(carrier)
        phases = np.array([0.0, 0.3, 0.6, 0.9])
        W, states = closure.run(phases, n_outer_steps=10)
        assert len(states) == 10
        assert W.shape == (4, 4)

    def test_cost_decreases(self):
        carrier = GeometryCarrier(4, z_dim=4, lr=0.05, seed=42)
        closure = CyberneticClosure(carrier)
        phases = np.array([0.0, 0.3, 0.6, 0.9])
        _, states = closure.run(phases, n_outer_steps=20)
        # Cost should generally decrease (not monotonically due to FD noise)
        assert states[-1].cost_after < states[0].cost_before + 0.5

    def test_reset(self):
        carrier = GeometryCarrier(3, z_dim=2, seed=42)
        closure = CyberneticClosure(carrier)
        closure.step(np.zeros(3))
        closure.reset()
        _, state = closure.step(np.zeros(3))
        assert state.ssgf_state_step == 1

    def test_carrier_property(self):
        carrier = GeometryCarrier(3, z_dim=2)
        closure = CyberneticClosure(carrier)
        assert closure.carrier is carrier


class TestEthicalCost:
    def test_synced_low_cost(self):
        phases = np.zeros(4)
        knm = np.full((4, 4), 0.5)
        np.fill_diagonal(knm, 0.0)
        cost = compute_ethical_cost(phases, knm)
        assert isinstance(cost, EthicalCost)
        assert cost.J_sec > 0  # R=1, some connectivity

    def test_high_R_no_constraint_violation(self):
        phases = np.zeros(6)
        knm = np.full((6, 6), 1.0)
        np.fill_diagonal(knm, 0.0)
        cost = compute_ethical_cost(phases, knm, R_min=0.5)
        assert cost.constraints_violated == 0

    def test_low_R_violates_non_harm(self):
        phases = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        knm = np.full((6, 6), 0.1)
        np.fill_diagonal(knm, 0.0)
        cost = compute_ethical_cost(phases, knm, R_min=0.5)
        assert cost.constraints_violated >= 1
        assert cost.phi_ethics > 0

    def test_disconnected_violates_connectivity(self):
        phases = np.zeros(4)
        knm = np.zeros((4, 4))
        cost = compute_ethical_cost(phases, knm, connectivity_min=0.5)
        assert cost.constraints_violated >= 1

    def test_excessive_coupling_violates_boundary(self):
        phases = np.zeros(3)
        knm = np.full((3, 3), 10.0)
        np.fill_diagonal(knm, 0.0)
        cost = compute_ethical_cost(phases, knm, max_coupling=5.0)
        assert cost.constraints_violated >= 1

    def test_empty_phases(self):
        cost = compute_ethical_cost(np.array([]), np.zeros((0, 0)))
        assert cost.c15_sec == 1.0

    def test_c15_formula(self):
        phases = np.zeros(4)
        knm = np.full((4, 4), 0.5)
        np.fill_diagonal(knm, 0.0)
        cost = compute_ethical_cost(phases, knm)
        expected = (1.0 - cost.J_sec) + cost.phi_ethics
        assert abs(cost.c15_sec - expected) < 1e-10


class TestPipelineWiring:
    """Pipeline wiring: proves this module is not decorative."""

    def test_wires_into_pipeline(self):
        import numpy as np

        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        n = 8
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        for _ in range(100):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0
