# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Closure + Ethical cost tests

"""Exercise geometry closure and actual ethical-cost numerical consumers."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.ethical_cost_reference import reference_cost
from scpn_phase_orchestrator.ssgf.carrier import GeometryCarrier
from scpn_phase_orchestrator.ssgf.closure import ClosureState, CyberneticClosure
from scpn_phase_orchestrator.ssgf.ethical import (
    EthicalCost,
    FloatArray,
    compute_ethical_cost,
)
from tests.test_ethical_cost_real_runtime import installed_cost


class TestCyberneticClosure:
    """Exercise closure updates, history and reset through public operations."""

    def test_step_returns_W_and_state(self) -> None:
        """A closure step produces usable coupling and advances carrier state."""
        carrier = GeometryCarrier(4, z_dim=3, seed=42)
        closure = CyberneticClosure(carrier)
        phases = np.array([0.0, 0.5, 1.0, 1.5])
        W, state = closure.step(phases)
        assert W.shape == (4, 4)
        assert isinstance(state, ClosureState)
        assert state.ssgf_state_step == 1

    def test_run_multiple_steps(self) -> None:
        """The requested outer steps produce one recorded closure state each."""
        carrier = GeometryCarrier(4, z_dim=3, lr=0.05, seed=42)
        closure = CyberneticClosure(carrier)
        phases = np.array([0.0, 0.3, 0.6, 0.9])
        W, states = closure.run(phases, n_outer_steps=10)
        assert len(states) == 10
        assert W.shape == (4, 4)

    def test_cost_decreases(self) -> None:
        """Public updates reduce the initial cost within the derivative allowance."""
        carrier = GeometryCarrier(4, z_dim=4, lr=0.05, seed=42)
        closure = CyberneticClosure(carrier)
        phases = np.array([0.0, 0.3, 0.6, 0.9])
        _, states = closure.run(phases, n_outer_steps=20)
        # Cost should generally decrease (not monotonically due to FD noise)
        assert states[-1].cost_after < states[0].cost_before + 0.5

    def test_reset(self) -> None:
        """Resetting a used closure restarts the public carrier step counter."""
        carrier = GeometryCarrier(3, z_dim=2, seed=42)
        closure = CyberneticClosure(carrier)
        closure.step(np.zeros(3))
        closure.reset()
        _, state = closure.step(np.zeros(3))
        assert state.ssgf_state_step == 1

    def test_carrier_property(self) -> None:
        """The closure exposes the same carrier that owns its generated geometry."""
        carrier = GeometryCarrier(3, z_dim=2)
        closure = CyberneticClosure(carrier)
        assert closure.carrier is carrier


class TestEthicalCost:
    """Check numerical scores and genuine installed Python/Rust owner parity."""

    def test_synced_low_cost(self) -> None:
        """A synchronized weighted complete graph has a positive diagnostic score."""
        phases = np.zeros(4)
        knm = np.full((4, 4), 0.5)
        np.fill_diagonal(knm, 0.0)
        cost = compute_ethical_cost(phases, knm)
        assert isinstance(cost, EthicalCost)
        assert cost.J_sec > 0  # R=1, some connectivity

    def test_high_R_no_constraint_violation(self) -> None:
        """A synchronized bounded complete graph satisfies all default residuals."""
        phases = np.zeros(6)
        knm = np.full((6, 6), 1.0)
        np.fill_diagonal(knm, 0.0)
        cost = compute_ethical_cost(phases, knm, R_min=0.5)
        assert cost.constraints_violated == 0

    def test_low_R_violates_non_harm(self) -> None:
        """Splayed phases violate the requested minimum coherence."""
        phases = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        knm = np.full((6, 6), 0.1)
        np.fill_diagonal(knm, 0.0)
        cost = compute_ethical_cost(phases, knm, R_min=0.5)
        assert cost.constraints_violated >= 1
        assert cost.phi_ethics > 0

    def test_disconnected_violates_connectivity(self) -> None:
        """An uncoupled graph violates a positive connectivity threshold."""
        phases = np.zeros(4)
        knm = np.zeros((4, 4))
        cost = compute_ethical_cost(phases, knm, connectivity_min=0.5)
        assert cost.constraints_violated >= 1

    def test_excessive_coupling_violates_boundary(self) -> None:
        """Positive coupling above the threshold contributes a violation."""
        phases = np.zeros(3)
        knm = np.full((3, 3), 10.0)
        np.fill_diagonal(knm, 0.0)
        cost = compute_ethical_cost(phases, knm, max_coupling=5.0)
        assert cost.constraints_violated >= 1

    def test_empty_phases(self) -> None:
        """An empty matching graph returns the declared neutral score and unit cost."""
        cost = compute_ethical_cost(np.array([]), np.zeros((0, 0)))
        assert cost.c15_sec == 1.0

    def test_c15_formula(self) -> None:
        """The actual total equals one minus score plus the already weighted penalty."""
        phases = np.zeros(4)
        knm = np.full((4, 4), 0.5)
        np.fill_diagonal(knm, 0.0)
        cost = compute_ethical_cost(phases, knm)
        expected = (1.0 - cost.J_sec) + cost.phi_ethics
        assert abs(cost.c15_sec - expected) < 1e-10

    def test_single_phase_exact_decomposition(self) -> None:
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

    def test_cbf_penalty_counts_all_independent_violations(self) -> None:
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

    @pytest.mark.native_runtime
    def test_optional_rust_path_preserves_return_contract(self) -> None:
        """Actual installed native owner preserves asymmetric inputs and field order."""
        phases = np.array([0.0, 0.5])
        knm = np.array([[0.0, 0.25], [0.75, 0.0]])
        parameters = {
            "alpha_R": 0.11,
            "beta_K": 0.22,
            "gamma_Q": 0.33,
            "nu_S": 0.44,
            "kappa": 0.55,
            "R_min": 0.66,
            "connectivity_min": 0.77,
            "max_coupling": 0.88,
        }
        actual = installed_cost("rust", phases, knm, **parameters)
        expected = reference_cost(phases, knm, **parameters)
        np.testing.assert_allclose(actual[:3], expected[:3], rtol=1e-12, atol=1e-12)
        assert actual[3] == expected[3]

    @pytest.mark.native_runtime
    def test_python_fallback_single_phase_exact(self) -> None:
        """A genuinely kernel-absent installation scores the one-node identity."""
        cost = EthicalCost(
            *installed_cost(
                "python",
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
        )
        assert cost.J_sec == pytest.approx(0.4)
        assert cost.phi_ethics == pytest.approx(2.0 * 0.1**2)
        assert cost.c15_sec == pytest.approx((1.0 - 0.4) + 2.0 * 0.1**2)
        assert cost.constraints_violated == 1

    @pytest.mark.native_runtime
    def test_python_fallback_counts_independent_violations(self) -> None:
        """Kernel absence retains the three independently weighted residuals."""
        phases = np.array([0.0, np.pi])
        knm = np.array([[0.0, 7.0], [7.0, 0.0]])
        cost = EthicalCost(
            *installed_cost(
                "python",
                phases,
                knm,
                kappa=0.5,
                R_min=0.5,
                connectivity_min=20.0,
                max_coupling=5.0,
            )
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
    @pytest.mark.native_runtime
    def test_rust_python_parity(
        self, phases: FloatArray, knm: FloatArray, expect_violations: int | None
    ) -> None:
        """Two genuinely separate installed owners match an independent oracle."""
        rust = installed_cost("rust", phases, knm)
        python = installed_cost("python", phases, knm)
        expected = reference_cost(phases, knm)
        for actual in (rust, python):
            np.testing.assert_allclose(actual[:3], expected[:3], rtol=1e-10, atol=1e-10)
            assert actual[3] == expected[3]
        if expect_violations is not None:
            assert python[3] == expect_violations


class TestClosureEthicalPipelineWiring:
    """Pipeline: SSGF closure → W → engine → ethical cost."""

    def test_closure_w_to_engine_to_ethical(self) -> None:
        """Evaluate closure-produced coupling through the engine and cost oracle."""
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
        expected = reference_cost(phases, W)
        np.testing.assert_allclose(
            [cost.J_sec, cost.phi_ethics, cost.c15_sec],
            expected[:3],
            rtol=1e-10,
            atol=1e-10,
        )
        assert cost.constraints_violated == expected[3]
