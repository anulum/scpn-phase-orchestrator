# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Closure + Ethical cost tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.ssgf import ethical as ethical_module
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

    def test_single_phase_exact_decomposition(self):
        """Single-node input has no possible coupling but still scores coherence."""
        cost = compute_ethical_cost(
            np.array([0.25]),
            np.zeros((1, 1)),
            alpha_R=0.4,
            beta_K=0.3,
            gamma_Q=0.2,
            nu_S=0.1,
            kappa=2.0,
            R_min=0.2,
            connectivity_min=0.1,
        )

        assert cost.J_sec == pytest.approx(0.4)
        assert cost.phi_ethics == pytest.approx(2.0 * 0.1**2)
        assert cost.c15_sec == pytest.approx((1.0 - 0.4) + 2.0 * 0.1**2)
        assert cost.constraints_violated == 1

    def test_cbf_penalty_counts_all_independent_violations(self):
        """Low coherence, no connectivity, and excessive coupling add separately."""
        phases = np.array([0.0, np.pi])
        knm = np.array([[0.0, 7.0], [7.0, 0.0]])

        cost = compute_ethical_cost(
            phases,
            knm,
            kappa=0.5,
            R_min=0.5,
            connectivity_min=20.0,
            max_coupling=5.0,
        )

        assert cost.constraints_violated == 3
        assert cost.phi_ethics == pytest.approx(
            0.5 * (0.5**2 + (20.0 - 14.0) ** 2 + 2.0**2)
        )
        assert cost.c15_sec == pytest.approx((1.0 - cost.J_sec) + cost.phi_ethics)

    def test_optional_rust_path_preserves_return_contract(self, monkeypatch):
        """Rust acceleration path must flatten K_nm and preserve field ordering."""
        calls = []

        def fake_rust_ethical_cost(
            phases,
            flattened_knm,
            n,
            alpha_R,
            beta_K,
            gamma_Q,
            nu_S,
            kappa,
            R_min,
            connectivity_min,
            max_coupling,
        ):
            calls.append(
                (
                    phases.copy(),
                    flattened_knm.copy(),
                    n,
                    alpha_R,
                    beta_K,
                    gamma_Q,
                    nu_S,
                    kappa,
                    R_min,
                    connectivity_min,
                    max_coupling,
                )
            )
            return 0.1, 0.2, 1.1, 2

        monkeypatch.setattr(ethical_module, "_HAS_RUST", True)
        monkeypatch.setattr(
            ethical_module,
            "_rust_ethical_cost",
            fake_rust_ethical_cost,
            raising=False,
        )

        phases = np.array([0.0, 0.5], dtype=np.float64)
        knm = np.array([[0.0, 0.25], [0.75, 0.0]], dtype=np.float64)
        cost = compute_ethical_cost(
            phases,
            knm,
            alpha_R=0.11,
            beta_K=0.22,
            gamma_Q=0.33,
            nu_S=0.44,
            kappa=0.55,
            R_min=0.66,
            connectivity_min=0.77,
            max_coupling=0.88,
        )

        assert cost == EthicalCost(
            J_sec=0.1,
            phi_ethics=0.2,
            c15_sec=1.1,
            constraints_violated=2,
        )
        assert len(calls) == 1
        call = calls[0]
        np.testing.assert_array_equal(call[0], phases)
        np.testing.assert_array_equal(call[1], np.array([0.0, 0.25, 0.75, 0.0]))
        assert call[2:] == (2, 0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88)

    def test_python_fallback_single_phase_exact(self, monkeypatch):
        """Force the Python reference path and verify the exact SEC/CBF maths
        regardless of whether the Rust kernel is present on the host.

        n=1 gives R=1, lambda2=0, Q=0 (no possible edges) and S_dev=0, so the
        SEC score reduces to alpha_R and only the connectivity barrier fires.
        """
        monkeypatch.setattr(ethical_module, "_HAS_RUST", False)
        cost = compute_ethical_cost(
            np.array([0.25]),
            np.zeros((1, 1)),
            alpha_R=0.4,
            beta_K=0.3,
            gamma_Q=0.2,
            nu_S=0.1,
            kappa=2.0,
            R_min=0.2,
            connectivity_min=0.1,
        )
        assert cost.J_sec == pytest.approx(0.4)
        assert cost.phi_ethics == pytest.approx(2.0 * 0.1**2)
        assert cost.c15_sec == pytest.approx((1.0 - 0.4) + 2.0 * 0.1**2)
        assert cost.constraints_violated == 1

    def test_python_fallback_counts_independent_violations(self, monkeypatch):
        """Python path: low coherence, no connectivity, and excess coupling each
        contribute a separate squared CBF penalty."""
        monkeypatch.setattr(ethical_module, "_HAS_RUST", False)
        phases = np.array([0.0, np.pi])
        knm = np.array([[0.0, 7.0], [7.0, 0.0]])
        cost = compute_ethical_cost(
            phases,
            knm,
            kappa=0.5,
            R_min=0.5,
            connectivity_min=20.0,
            max_coupling=5.0,
        )
        assert cost.constraints_violated == 3
        assert cost.phi_ethics == pytest.approx(
            0.5 * (0.5**2 + (20.0 - 14.0) ** 2 + 2.0**2)
        )
        assert cost.c15_sec == pytest.approx((1.0 - cost.J_sec) + cost.phi_ethics)

    @pytest.mark.parametrize(
        ("phases", "knm", "expect_violations"),
        [
            # Synchronised, well-connected, bounded coupling -> no barrier fires.
            (np.zeros(6), np.full((6, 6), 1.0) - np.eye(6), 0),
            # Splayed, weakly coupled -> low coherence barrier fires.
            (
                np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False),
                (np.full((6, 6), 0.05) - np.eye(6) * 0.05),
                None,
            ),
        ],
    )
    def test_rust_python_parity(self, monkeypatch, phases, knm, expect_violations):
        """Rust and Python paths must agree field-for-field on the same input."""
        if not ethical_module._HAS_RUST:
            pytest.skip("Rust backend not built")

        knm = np.ascontiguousarray(knm, dtype=np.float64)
        rust_cost = compute_ethical_cost(phases, knm)
        monkeypatch.setattr(ethical_module, "_HAS_RUST", False)
        py_cost = compute_ethical_cost(phases, knm)

        assert py_cost.J_sec == pytest.approx(rust_cost.J_sec, abs=1e-9)
        assert py_cost.phi_ethics == pytest.approx(rust_cost.phi_ethics, abs=1e-9)
        assert py_cost.c15_sec == pytest.approx(rust_cost.c15_sec, abs=1e-9)
        assert py_cost.constraints_violated == rust_cost.constraints_violated
        if expect_violations is not None:
            assert py_cost.constraints_violated == expect_violations


class TestClosureEthicalPipelineWiring:
    """Pipeline: SSGF closure → W → engine → ethical cost."""

    def test_closure_w_to_engine_to_ethical(self):
        """CyberneticClosure.step → W → engine → phases → ethical cost.
        Full SSGF-ethic loop."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import (
            compute_order_parameter,
        )

        n = 4
        carrier = GeometryCarrier(n, z_dim=3, seed=42)
        closure = CyberneticClosure(carrier)
        phases = np.array([0.0, 0.5, 1.0, 1.5])
        W, _ = closure.step(phases)

        eng = UPDEEngine(n, dt=0.01)
        omegas = np.ones(n)
        for _ in range(100):
            phases = eng.step(phases, omegas, W, 0.0, 0.0, np.zeros((n, n)))
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0

        cost = compute_ethical_cost(phases, W)
        assert isinstance(cost, EthicalCost)
        assert np.isfinite(cost.c15_sec)
