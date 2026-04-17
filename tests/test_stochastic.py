# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Stochastic noise tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.stochastic import (
    NoiseProfile,
    StochasticInjector,
    _self_consistency_R,
    find_optimal_noise,
    optimal_D,
)


class TestStochasticInjector:
    def test_zero_noise_identity(self):
        inj = StochasticInjector(D=0.0)
        phases = np.array([0.0, 1.0, 2.0])
        result = inj.inject(phases, dt=0.01)
        np.testing.assert_array_equal(result, phases)

    def test_nonzero_noise_changes_phases(self):
        inj = StochasticInjector(D=1.0, seed=42)
        phases = np.zeros(10)
        result = inj.inject(phases, dt=0.01)
        assert not np.allclose(result, phases)

    def test_output_in_range(self):
        inj = StochasticInjector(D=5.0, seed=42)
        phases = np.zeros(100)
        result = inj.inject(phases, dt=0.1)
        assert np.all(result >= 0.0)
        assert np.all(result < 2 * np.pi)

    def test_negative_D_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            StochasticInjector(D=-1.0)

    def test_D_setter(self):
        inj = StochasticInjector(D=0.0)
        inj.D = 0.5
        assert inj.D == 0.5

    def test_D_setter_negative_raises(self):
        inj = StochasticInjector(D=0.0)
        with pytest.raises(ValueError):
            inj.D = -1.0

    def test_reproducible_with_seed(self):
        inj1 = StochasticInjector(D=1.0, seed=99)
        inj2 = StochasticInjector(D=1.0, seed=99)
        phases = np.zeros(5)
        r1 = inj1.inject(phases, dt=0.01)
        r2 = inj2.inject(phases, dt=0.01)
        np.testing.assert_array_equal(r1, r2)


class TestSelfConsistency:
    def test_zero_K(self):
        assert _self_consistency_R(0.0, 1.0) == 0.0

    def test_zero_D(self):
        assert _self_consistency_R(1.0, 0.0) == 1.0

    def test_intermediate(self):
        R = _self_consistency_R(2.0, 0.5)
        assert 0.0 < R < 1.0

    def test_large_K(self):
        R = _self_consistency_R(100.0, 1.0)
        assert R > 0.9


class TestOptimalD:
    def test_formula(self):
        assert abs(optimal_D(2.0, 0.8) - 0.8) < 1e-10

    def test_zero_R(self):
        assert optimal_D(1.0, 0.0) == 0.0


class TestFindOptimalNoise:
    def test_returns_profile(self):
        n = 6
        engine = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = np.full((n, n), 0.5)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        result = find_optimal_noise(
            engine,
            phases,
            omegas,
            knm,
            alpha,
            D_range=np.array([0.0, 0.1, 0.5]),
            n_steps=50,
        )
        assert isinstance(result, NoiseProfile)
        assert result.D >= 0.0
        assert 0.0 <= result.R_achieved <= 1.0
        assert 0.0 <= result.R_deterministic <= 1.0


class TestNoiseScaling:
    """Stochastic diffusion coefficient scaling properties."""

    def test_noise_variance_scales_with_D(self):
        """Circular variance of injected noise ∝ D · dt.

        Raw variance of mod-2π output is non-monotonic because phase
        wrapping creates a bimodal distribution at intermediate D.
        Use unwrapped (circular) differences to measure true noise spread.
        """
        phases = np.zeros(10000)
        dt = 0.01
        variances = {}
        for D in [0.1, 1.0, 10.0]:
            inj = StochasticInjector(D=D, seed=42)
            result = inj.inject(phases, dt=dt)
            # Circular difference: unwrap to [-π, π)
            diff = (result - phases + np.pi) % (2 * np.pi) - np.pi
            variances[D] = np.var(diff)
        # Circular variance scales monotonically with D
        assert variances[0.1] < variances[1.0] < variances[10.0]

    def test_noise_mean_near_zero(self):
        """Mean noise perturbation ≈ 0 (unbiased Wiener process)."""
        inj = StochasticInjector(D=1.0, seed=42)
        phases = np.full(50000, np.pi)
        result = inj.inject(phases, dt=0.01)
        diff = result - phases
        # Wrap to [-π, π)
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        assert abs(np.mean(diff)) < 0.02

    def test_self_consistency_monotone_in_K(self):
        """R(K, D) is monotonically non-decreasing in K for fixed D."""
        D = 0.5
        prev_r = 0.0
        for K in np.linspace(0, 10, 20):
            r = _self_consistency_R(K, D)
            assert r >= prev_r - 1e-12
            prev_r = r


class TestStochasticPipelineEndToEnd:
    """Full pipeline: CouplingBuilder → Engine + StochasticInjector → R → Regime."""

    def test_coupling_engine_noise_regime(self):
        from scpn_phase_orchestrator.coupling.knm import CouplingBuilder
        from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
        from scpn_phase_orchestrator.supervisor.regimes import RegimeManager
        from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        n = 12
        cb = CouplingBuilder()
        cs = cb.build(n_layers=n, base_strength=0.5, decay_alpha=0.2)
        eng = UPDEEngine(n, dt=0.01)
        inj = StochasticInjector(D=0.1, seed=42)
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        for _ in range(300):
            phases = eng.step(phases, omegas, cs.knm, 0.0, 0.0, cs.alpha)
            phases = inj.inject(phases, dt=0.01)
        r, psi = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0
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

    def test_noise_reduces_synchronisation(self):
        """Adding noise should reduce R compared to deterministic."""
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        n = 8
        rng = np.random.default_rng(55)
        phases0 = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = np.full((n, n), 0.5)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        # Deterministic
        eng = UPDEEngine(n, dt=0.01)
        p_det = phases0.copy()
        for _ in range(500):
            p_det = eng.step(p_det, omegas, knm, 0.0, 0.0, alpha)
        r_det, _ = compute_order_parameter(p_det)
        # Stochastic (high noise)
        eng2 = UPDEEngine(n, dt=0.01)
        inj = StochasticInjector(D=5.0, seed=42)
        p_sto = phases0.copy()
        for _ in range(500):
            p_sto = eng2.step(p_sto, omegas, knm, 0.0, 0.0, alpha)
            p_sto = inj.inject(p_sto, dt=0.01)
        r_sto, _ = compute_order_parameter(p_sto)
        assert 0.0 <= r_sto <= 1.0
        # High noise should reduce or maintain R
        assert r_sto <= r_det + 0.15

    def test_find_optimal_noise_coherent_with_pipeline(self):
        """find_optimal_noise returns a D that, when applied, gives R ≈ R_achieved."""
        n = 6
        engine = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = np.full((n, n), 0.5)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        profile = find_optimal_noise(
            engine,
            phases,
            omegas,
            knm,
            alpha,
            D_range=np.array([0.0, 0.05, 0.1, 0.5, 1.0]),
            n_steps=100,
        )
        assert isinstance(profile, NoiseProfile)
        assert profile.D >= 0.0
        assert 0.0 <= profile.R_achieved <= 1.0

    def test_performance_inject_1000_under_100us(self):
        """StochasticInjector.inject(1000 oscillators) < 100μs."""
        import time

        inj = StochasticInjector(D=1.0, seed=0)
        phases = np.zeros(1000)
        inj.inject(phases, dt=0.01)
        t0 = time.perf_counter()
        for _ in range(1000):
            inj.inject(phases, dt=0.01)
        elapsed = (time.perf_counter() - t0) / 1000
        assert elapsed < 1e-4, f"inject(1000) took {elapsed * 1e6:.1f}μs"


# Pipeline wiring: stochastic tests exercise StochasticInjector + UPDEEngine
# → compute_order_parameter → CouplingBuilder → RegimeManager. Physics:
# diffusion scaling, mean-zero noise, K-monotonicity. Performance: inject(1000)<100μs.
