# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Order parameter tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.order_params import (
    compute_layer_coherence,
    compute_order_parameter,
    compute_plv,
)

TWO_PI = 2.0 * np.pi


def test_identical_phases_R_one():
    phases = np.full(100, 1.5)
    R, _ = compute_order_parameter(phases)
    np.testing.assert_allclose(R, 1.0, atol=1e-12)


def test_uniform_phases_R_near_zero():
    phases = np.linspace(0, TWO_PI, 1000, endpoint=False)
    R, _ = compute_order_parameter(phases)
    assert R < 0.02


def test_plv_identical_series():
    phases = np.ones(200) * 2.0
    plv = compute_plv(phases, phases)
    np.testing.assert_allclose(plv, 1.0, atol=1e-12)


def test_plv_random_uncorrelated():
    rng = np.random.default_rng(55)
    a = rng.uniform(0, TWO_PI, size=5000)
    b = rng.uniform(0, TWO_PI, size=5000)
    plv = compute_plv(a, b)
    assert plv < 0.1


def test_layer_coherence_full_mask():
    phases = np.array([0.0, 0.0, 0.0, 0.0])
    mask = np.ones(4, dtype=bool)
    R_layer = compute_layer_coherence(phases, mask)
    R_global, _ = compute_order_parameter(phases)
    np.testing.assert_allclose(R_layer, R_global, atol=1e-12)


def test_layer_coherence_partial_mask():
    phases = np.array([0.0, 0.0, np.pi, np.pi])
    mask = np.array([True, True, False, False])
    R_sub = compute_layer_coherence(phases, mask)
    np.testing.assert_allclose(R_sub, 1.0, atol=1e-12)


def test_layer_coherence_empty_mask():
    phases = np.array([1.0, 2.0, 3.0])
    mask = np.zeros(3, dtype=bool)
    assert compute_layer_coherence(phases, mask) == 0.0


def test_plv_length_mismatch_raises():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="equal-length"):
        compute_plv(a, b)


class TestOrderParameterAlgebraicInvariants:
    """Information-theoretic and algebraic bounds on R and ψ."""

    def test_R_bounded_zero_one_for_random_phases(self):
        """R ∈ [0, 1] for 100 random draws."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            phases = rng.uniform(0, TWO_PI, 64)
            r, psi = compute_order_parameter(phases)
            assert 0.0 <= r <= 1.0 + 1e-12
            assert 0.0 <= psi < TWO_PI

    def test_psi_is_mean_phase_for_identical(self):
        """All phases = θ₀ → ψ = θ₀."""
        theta_0 = 2.7
        phases = np.full(50, theta_0)
        r, psi = compute_order_parameter(phases)
        assert abs(r - 1.0) < 1e-12
        assert abs(psi - theta_0) < 1e-10

    def test_R_decreases_with_spread(self):
        """As phase spread increases from 0 → π, R monotonically decreases."""
        n = 64
        base = 1.0
        prev_r = 1.0
        for spread in np.linspace(0, np.pi, 20):
            phases = np.linspace(base - spread / 2, base + spread / 2, n)
            r, _ = compute_order_parameter(phases)
            assert r <= prev_r + 1e-10
            prev_r = r

    def test_single_oscillator_R_one(self):
        r, _ = compute_order_parameter(np.array([4.2]))
        assert abs(r - 1.0) < 1e-12

    def test_two_antipodal_R_zero(self):
        """θ₁ = 0, θ₂ = π → R = 0."""
        r, _ = compute_order_parameter(np.array([0.0, np.pi]))
        assert r < 1e-12


class TestPLVStatisticalProperties:
    """PLV: deeper statistical and edge-case testing."""

    def test_plv_symmetric(self):
        """PLV(a, b) = PLV(b, a)."""
        rng = np.random.default_rng(66)
        a = rng.uniform(0, TWO_PI, 200)
        b = rng.uniform(0, TWO_PI, 200)
        assert abs(compute_plv(a, b) - compute_plv(b, a)) < 1e-12

    def test_plv_constant_phase_diff_one(self):
        """a = b + δ (constant offset) → PLV = 1."""
        rng = np.random.default_rng(77)
        a = rng.uniform(0, TWO_PI, 500)
        b = (a + 1.3) % TWO_PI
        plv = compute_plv(a, b)
        assert plv > 0.99

    def test_plv_converges_to_zero_for_independent(self):
        """As N grows, PLV of independent signals → 0."""
        rng = np.random.default_rng(88)
        plvs = []
        for n in [100, 500, 2000, 10000]:
            a = rng.uniform(0, TWO_PI, n)
            b = rng.uniform(0, TWO_PI, n)
            plvs.append(compute_plv(a, b))
        # PLV should decrease with sample size for independent signals
        assert plvs[-1] < plvs[0]


class TestOrderParamsPipelineEndToEnd:
    """Full pipeline: CouplingBuilder → Engine → R/PLV/coherence → Regime."""

    def test_coupling_engine_R_psi_regime(self):
        from scpn_phase_orchestrator.coupling.knm import CouplingBuilder
        from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
        from scpn_phase_orchestrator.supervisor.regimes import RegimeManager
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

        n = 16
        cb = CouplingBuilder()
        cs = cb.build(n_layers=n, base_strength=0.5, decay_alpha=0.2)
        eng = UPDEEngine(n, dt=0.01, method="rk4")
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = np.ones(n)
        # Collect trajectory for PLV
        history = [[] for _ in range(n)]
        for _ in range(300):
            phases = eng.step(phases, omegas, cs.knm, 0.0, 0.0, cs.alpha)
            for i in range(n):
                history[i].append(phases[i])
        # Order parameter
        r, psi = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0
        # PLV between first two oscillators
        plv_01 = compute_plv(np.array(history[0]), np.array(history[1]))
        assert 0.0 <= plv_01 <= 1.0
        # Layer coherence on first half
        mask = np.zeros(n, dtype=bool)
        mask[: n // 2] = True
        r_layer = compute_layer_coherence(phases, mask)
        assert 0.0 <= r_layer <= 1.0 + 1e-12
        # Feed R into RegimeManager
        layer = LayerState(R=r, psi=psi)
        state = UPDEState(
            layers=[layer],
            cross_layer_alignment=np.array([r]),
            stability_proxy=r,
            regime_id="nominal",
        )
        rm = RegimeManager(hysteresis=0.05)
        regime = rm.evaluate(state, BoundaryState())
        assert regime.name in {"NOMINAL", "DEGRADED", "CRITICAL", "RECOVERY"}

    def test_sync_trajectory_high_R_and_plv(self):
        """Strong coupling + identical ω → high R and high PLV."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        n = 8
        eng = UPDEEngine(n, dt=0.01, method="rk4")
        rng = np.random.default_rng(11)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = np.ones(n)
        knm = 3.0 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        h0, h1 = [], []
        for _ in range(500):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
            h0.append(phases[0])
            h1.append(phases[1])
        r, _ = compute_order_parameter(phases)
        plv = compute_plv(np.array(h0), np.array(h1))
        assert r > 0.9, f"Expected high R for strong coupling, got {r}"
        assert plv > 0.8, f"Expected high PLV for synced pair, got {plv}"

    def test_performance_R_256_under_100us(self):
        """compute_order_parameter(256) < 100μs budget."""
        import time

        rng = np.random.default_rng(0)
        phases = rng.uniform(0, TWO_PI, 256)
        compute_order_parameter(phases)
        t0 = time.perf_counter()
        for _ in range(1000):
            compute_order_parameter(phases)
        elapsed = (time.perf_counter() - t0) / 1000
        assert elapsed < 1e-4, f"R(256) took {elapsed * 1e6:.1f}μs"

    def test_performance_plv_1000_under_500us(self):
        """compute_plv(1000 samples) < 500μs budget."""
        import time

        rng = np.random.default_rng(0)
        a = rng.uniform(0, TWO_PI, 1000)
        b = rng.uniform(0, TWO_PI, 1000)
        compute_plv(a, b)
        t0 = time.perf_counter()
        for _ in range(1000):
            compute_plv(a, b)
        elapsed = (time.perf_counter() - t0) / 1000
        assert elapsed < 5e-4, f"PLV(1000) took {elapsed * 1e6:.1f}μs"


# Pipeline wiring: order_params tests exercise compute_order_parameter,
# compute_plv, compute_layer_coherence via CouplingBuilder → UPDEEngine(RK4)
# → RegimeManager. Algebraic: R∈[0,1], PLV symmetry, convergence, monotonicity.
# Performance: R(256)<100μs, PLV(1000)<500μs.
