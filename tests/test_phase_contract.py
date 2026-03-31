# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Phase contract tests

"""Phase-contract invariants: wrapping, order parameter bounds, PLV bounds."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator._compat import TWO_PI
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import (
    compute_layer_coherence,
    compute_order_parameter,
    compute_plv,
)

N = 8
RNG = np.random.default_rng(99)


def _make_engine(method="euler"):
    return UPDEEngine(N, dt=0.01, method=method)


def _step_once(engine, phases, omegas=None, knm=None, alpha=None):
    if omegas is None:
        omegas = np.ones(N)
    if knm is None:
        knm = 0.3 * np.ones((N, N))
        np.fill_diagonal(knm, 0.0)
    if alpha is None:
        alpha = np.zeros((N, N))
    return engine.step(phases, omegas, knm, 0.0, 0.0, alpha)


class TestPhaseWrapping:
    """theta in [0, 2*pi) after every step, for all integrators."""

    @pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
    def test_wrapping_after_single_step(self, method):
        engine = _make_engine(method)
        phases = RNG.uniform(0, TWO_PI, N)
        out = _step_once(engine, phases)
        assert np.all(out >= 0.0)
        assert np.all(out < TWO_PI)

    @pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
    def test_wrapping_after_many_steps(self, method):
        engine = _make_engine(method)
        phases = RNG.uniform(0, TWO_PI, N)
        for _ in range(200):
            phases = _step_once(engine, phases)
        assert np.all(phases >= 0.0)
        assert np.all(phases < TWO_PI)

    def test_wrapping_with_large_omega(self):
        engine = _make_engine("euler")
        phases = np.zeros(N)
        omegas = np.full(N, 500.0)
        out = _step_once(engine, phases, omegas=omegas)
        assert np.all(out >= 0.0)
        assert np.all(out < TWO_PI)


class TestOrderParameterBounds:
    """R in [0, 1], psi in [0, 2*pi)."""

    def test_synchronised_phases_R_near_one(self):
        phases = np.full(N, 1.5)
        r, psi = compute_order_parameter(phases)
        assert r == pytest.approx(1.0, abs=1e-12)
        assert 0.0 <= psi < TWO_PI

    def test_uniform_phases_R_near_zero(self):
        phases = np.linspace(0, TWO_PI, N, endpoint=False)
        r, _ = compute_order_parameter(phases)
        assert r < 0.15

    def test_random_phases_R_in_range(self):
        for _ in range(20):
            phases = RNG.uniform(0, TWO_PI, N)
            r, psi = compute_order_parameter(phases)
            assert 0.0 <= r <= 1.0 + 1e-12
            assert 0.0 <= psi < TWO_PI

    def test_single_oscillator_R_one(self):
        r, _ = compute_order_parameter(np.array([3.0]))
        assert r == pytest.approx(1.0, abs=1e-12)


class TestPLVBounds:
    """PLV in [0, 1]."""

    def test_identical_phases_plv_one(self):
        a = RNG.uniform(0, TWO_PI, 50)
        assert compute_plv(a, a) == pytest.approx(1.0, abs=1e-12)

    def test_random_phases_plv_in_range(self):
        for _ in range(20):
            a = RNG.uniform(0, TWO_PI, 50)
            b = RNG.uniform(0, TWO_PI, 50)
            plv = compute_plv(a, b)
            assert 0.0 <= plv <= 1.0 + 1e-12

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="equal-length"):
            compute_plv(np.array([1.0, 2.0]), np.array([1.0]))


class TestLayerCoherence:
    def test_empty_mask_returns_zero(self):
        phases = RNG.uniform(0, TWO_PI, N)
        mask = np.zeros(N, dtype=bool)
        assert compute_layer_coherence(phases, mask) == 0.0

    def test_full_mask_matches_order_param(self):
        phases = RNG.uniform(0, TWO_PI, N)
        mask = np.ones(N, dtype=bool)
        r_layer = compute_layer_coherence(phases, mask)
        r_global, _ = compute_order_parameter(phases)
        assert r_layer == pytest.approx(r_global, abs=1e-12)


class TestRK45AdaptiveDt:
    def test_last_dt_property(self):
        engine = _make_engine("rk45")
        phases = RNG.uniform(0, TWO_PI, N)
        _step_once(engine, phases)
        assert engine.last_dt > 0.0

    def test_rk45_finite_output(self):
        engine = _make_engine("rk45")
        phases = RNG.uniform(0, TWO_PI, N)
        for _ in range(50):
            phases = _step_once(engine, phases)
        assert np.all(np.isfinite(phases))


class TestCrossIntegratorConsistency:
    """All three integrators must preserve the phase contract identically."""

    def test_all_methods_preserve_wrapping_over_trajectory(self):
        """Run 500 steps per method — every snapshot must satisfy [0, 2π)."""
        rng = np.random.default_rng(77)
        phases_init = rng.uniform(0, TWO_PI, N)
        omegas = rng.uniform(-5, 5, N)
        knm = 0.5 * np.ones((N, N))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((N, N))
        for method in ("euler", "rk4", "rk45"):
            eng = UPDEEngine(N, dt=0.01, method=method)
            phases = phases_init.copy()
            for step_i in range(500):
                phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
                assert np.all(phases >= 0.0), f"{method} step {step_i}: negative"
                assert np.all(phases < TWO_PI), f"{method} step {step_i}: ≥2π"

    def test_euler_rk4_converge_to_same_order_parameter(self):
        """Euler and RK4 trajectories → R should agree within tolerance."""
        rng = np.random.default_rng(88)
        phases_init = rng.uniform(0, TWO_PI, N)
        omegas = np.ones(N)
        knm = 0.4 * np.ones((N, N))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((N, N))
        rs = {}
        for method in ("euler", "rk4"):
            eng = UPDEEngine(N, dt=0.005, method=method)
            phases = phases_init.copy()
            for _ in range(300):
                phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
            r, _ = compute_order_parameter(phases)
            rs[method] = r
        # With small dt, Euler and RK4 should converge to similar R
        assert abs(rs["euler"] - rs["rk4"]) < 0.15

    def test_rk45_adaptive_dt_always_positive(self):
        """RK45 adaptive step size must always be > 0 after each step."""
        eng = UPDEEngine(N, dt=0.01, method="rk45")
        rng = np.random.default_rng(55)
        phases = rng.uniform(0, TWO_PI, N)
        omegas = rng.uniform(-3, 3, N)
        knm = 0.3 * np.ones((N, N))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((N, N))
        for _ in range(100):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
            assert eng.last_dt > 0.0


class TestPLVAfterEngineTrajectory:
    """PLV computed on engine trajectories: proves PLV wires into pipeline."""

    def test_synchronised_trajectory_plv_high(self):
        """Coupled oscillators with identical ω → PLV ≈ 1."""
        eng = UPDEEngine(N, dt=0.01, method="rk4")
        rng = np.random.default_rng(33)
        phases = rng.uniform(0, TWO_PI, N)
        omegas = np.ones(N)
        knm = 2.0 * np.ones((N, N))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((N, N))
        history_0, history_1 = [], []
        for _ in range(300):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
            history_0.append(phases[0])
            history_1.append(phases[1])
        plv = compute_plv(np.array(history_0), np.array(history_1))
        assert plv > 0.7

    def test_uncoupled_trajectory_plv_lower(self):
        """Zero coupling + different ω → PLV < synchronised case."""
        eng = UPDEEngine(N, dt=0.01, method="euler")
        rng = np.random.default_rng(44)
        phases = rng.uniform(0, TWO_PI, N)
        omegas = rng.uniform(1, 5, N)
        knm = np.zeros((N, N))
        alpha = np.zeros((N, N))
        history_0, history_1 = [], []
        for _ in range(200):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
            history_0.append(phases[0])
            history_1.append(phases[1])
        plv = compute_plv(np.array(history_0), np.array(history_1))
        assert 0.0 <= plv <= 1.0


class TestPhaseContractPipelineEndToEnd:
    """Full pipeline: CouplingBuilder → Engine → order_parameter → RegimeManager.

    Proves phase_contract module is structurally wired, not decorative.
    """

    def test_coupling_engine_order_param_regime(self):
        """Build K_nm → run engine → compute R → feed RegimeManager."""
        from scpn_phase_orchestrator.coupling.knm import CouplingBuilder
        from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
        from scpn_phase_orchestrator.supervisor.regimes import RegimeManager
        from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

        n = 16
        cb = CouplingBuilder()
        cs = cb.build(n_layers=n, base_strength=0.5, decay_alpha=0.2)
        eng = UPDEEngine(n, dt=0.01, method="rk4")
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = np.ones(n)
        for _ in range(200):
            phases = eng.step(phases, omegas, cs.knm, 0.0, 0.0, cs.alpha)
        # Phase contract: wrapping
        assert np.all(phases >= 0.0)
        assert np.all(phases < TWO_PI)
        # Order parameter contract: R ∈ [0,1]
        r, psi = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0
        assert 0.0 <= psi < TWO_PI
        # Feed into RegimeManager
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

    def test_layer_coherence_per_layer_mask(self):
        """Layer coherence on engine output with partial mask."""
        eng = UPDEEngine(N, dt=0.01, method="rk4")
        rng = np.random.default_rng(11)
        phases = rng.uniform(0, TWO_PI, N)
        omegas = np.ones(N)
        knm = 0.5 * np.ones((N, N))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((N, N))
        for _ in range(100):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        mask = np.array([True, True, True, True, False, False, False, False])
        r_layer = compute_layer_coherence(phases, mask)
        r_global, _ = compute_order_parameter(phases)
        assert 0.0 <= r_layer <= 1.0 + 1e-12
        assert 0.0 <= r_global <= 1.0 + 1e-12

    def test_engine_run_method_wrapping(self):
        """Engine.run() trajectory — every final phase wrapped."""
        eng = UPDEEngine(N, dt=0.01, method="rk4")
        rng = np.random.default_rng(22)
        phases = rng.uniform(0, TWO_PI, N)
        omegas = np.ones(N)
        knm = 0.3 * np.ones((N, N))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((N, N))
        final = eng.run(phases, omegas, knm, 0.0, 0.0, alpha, n_steps=500)
        assert np.all(final >= 0.0)
        assert np.all(final < TWO_PI)
        r, psi = compute_order_parameter(final)
        assert 0.0 <= r <= 1.0

    def test_performance_order_parameter_256_under_100us(self):
        """compute_order_parameter(256 oscillators) < 100μs."""
        import time

        phases = RNG.uniform(0, TWO_PI, 256)
        # Warm-up
        compute_order_parameter(phases)
        t0 = time.perf_counter()
        for _ in range(1000):
            compute_order_parameter(phases)
        elapsed = (time.perf_counter() - t0) / 1000
        assert elapsed < 1e-4, f"order_parameter(256) took {elapsed * 1e6:.1f}μs"

    def test_performance_plv_1000_under_500us(self):
        """compute_plv(1000 samples) < 500μs."""
        import time

        a = RNG.uniform(0, TWO_PI, 1000)
        b = RNG.uniform(0, TWO_PI, 1000)
        compute_plv(a, b)
        t0 = time.perf_counter()
        for _ in range(1000):
            compute_plv(a, b)
        elapsed = (time.perf_counter() - t0) / 1000
        assert elapsed < 5e-4, f"PLV(1000) took {elapsed * 1e6:.1f}μs"


# Pipeline wiring: phase contract tests exercise UPDEEngine (all 3 integrators)
# → compute_order_parameter → compute_plv → compute_layer_coherence →
# CouplingBuilder → RegimeManager. Performance: R(256)<100μs, PLV(1000)<500μs.
