# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Cross-engine equivalence + analytical proofs

"""Cross-engine parity matrix: all engines that should agree on a given
scenario produce the same result. Plus analytical validation tests
that compare simulation to closed-form solutions.

These tests are the computational backbone of correctness claims
in any paper or production deployment.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.coupling.spectral import critical_coupling
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.geometric import TorusEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter
from scpn_phase_orchestrator.upde.simplicial import SimplicialEngine
from scpn_phase_orchestrator.upde.splitting import SplittingEngine

TWO_PI = 2.0 * np.pi


def _setup(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0, TWO_PI, n)
    omegas = rng.uniform(-1, 1, n)
    raw = rng.uniform(0.3, 1.0, (n, n))
    knm = 0.5 * (raw + raw.T)
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))
    return phases, omegas, knm, alpha


# ── Cross-engine parity: identical input → same output ───────────────────


class TestUPDEvsTorusEngine:
    """UPDE Euler and TorusEngine should agree at small dt."""

    @given(
        n=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=30, deadline=None)
    def test_single_step_close(self, n: int, seed: int) -> None:
        phases, omegas, knm, alpha = _setup(n, seed)
        dt = 0.001  # small dt for agreement
        upde = UPDEEngine(n, dt=dt, method="euler")
        torus = TorusEngine(n, dt=dt)
        out_upde = upde.step(phases, omegas, knm, 0.0, 0.0, alpha)
        out_torus = torus.step(phases, omegas, knm, 0.0, 0.0, alpha)
        np.testing.assert_allclose(out_upde % TWO_PI, out_torus, atol=1e-4)

    def test_100_steps_converge(self) -> None:
        """Over 100 steps both engines reach same sync state."""
        n = 6
        phases, omegas, knm, alpha = _setup(n, 0)
        omegas = np.zeros(n)  # identical → must sync
        dt = 0.01
        upde = UPDEEngine(n, dt=dt, method="euler")
        torus = TorusEngine(n, dt=dt)
        p_u, p_t = phases.copy(), phases.copy()
        for _ in range(100):
            p_u = upde.step(p_u, omegas, knm, 0.0, 0.0, alpha)
            p_t = torus.step(p_t, omegas, knm, 0.0, 0.0, alpha)
        R_u, _ = compute_order_parameter(p_u)
        R_t, _ = compute_order_parameter(p_t)
        assert abs(R_u - R_t) < 0.05


class TestUPDEvsSplittingEngine:
    """SplittingEngine (Strang) should agree with UPDE Euler at small dt."""

    @given(
        n=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=30, deadline=None)
    def test_single_step_close(self, n: int, seed: int) -> None:
        phases, omegas, knm, alpha = _setup(n, seed)
        dt = 0.001
        upde = UPDEEngine(n, dt=dt, method="euler")
        split = SplittingEngine(n, dt=dt)
        out_upde = upde.step(phases, omegas, knm, 0.0, 0.0, alpha)
        out_split = split.step(phases, omegas, knm, 0.0, 0.0, alpha)
        np.testing.assert_allclose(out_upde % TWO_PI, out_split, atol=1e-3)


class TestUPDEEulerVsRK4:
    """Euler and RK4 converge to same state for long runs."""

    @pytest.mark.parametrize("n", [4, 8])
    def test_converged_r_agrees(self, n: int) -> None:
        phases, _, knm, alpha = _setup(n, 42)
        omegas = np.zeros(n)
        p_euler = phases.copy()
        p_rk4 = phases.copy()
        euler = UPDEEngine(n, dt=0.01, method="euler")
        rk4 = UPDEEngine(n, dt=0.01, method="rk4")
        for _ in range(500):
            p_euler = euler.step(p_euler, omegas, knm, 0.0, 0.0, alpha)
            p_rk4 = rk4.step(p_rk4, omegas, knm, 0.0, 0.0, alpha)
        R_e, _ = compute_order_parameter(p_euler)
        R_r, _ = compute_order_parameter(p_rk4)
        assert abs(R_e - R_r) < 0.05


class TestSimplicialParity:
    """σ₂=0 reduces to UPDE; σ₂=0 reduces to TorusEngine."""

    @given(
        n=st.integers(min_value=3, max_value=8),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(
        max_examples=30,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    def test_sigma2_zero_vs_upde(self, n: int, seed: int) -> None:
        phases, omegas, knm, alpha = _setup(n, seed)
        upde = UPDEEngine(n, dt=0.01, method="euler")
        simp = SimplicialEngine(n, dt=0.01, sigma2=0.0)
        out_u = upde.step(phases, omegas, knm, 0.0, 0.0, alpha)
        out_s = simp.step(phases, omegas, knm, 0.0, 0.0, alpha)
        np.testing.assert_allclose(out_s, out_u, atol=1e-10)


# ── Analytical validation ────────────────────────────────────────────────


class TestSpectralKcVsBifurcation:
    """Dörfler-Bullo K_c predicts actual sync onset in simulation."""

    def test_above_kc_syncs(self) -> None:
        n = 8
        rng = np.random.default_rng(42)
        omegas = rng.uniform(-1, 1, n)
        knm_template = np.ones((n, n)) / n
        np.fill_diagonal(knm_template, 0.0)
        kc = critical_coupling(omegas, knm_template)

        # K = 2 * K_c → should synchronise
        K = max(kc * 2, 1.0)
        knm = knm_template * K
        eng = UPDEEngine(n, dt=0.01)
        alpha = np.zeros((n, n))
        phases = rng.uniform(0, TWO_PI, n)
        for _ in range(2000):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        R, _ = compute_order_parameter(phases)
        assert R > 0.5

    def test_below_kc_no_sync(self) -> None:
        n = 8
        rng = np.random.default_rng(42)
        omegas = rng.uniform(-2, 2, n)
        knm_template = np.ones((n, n)) / n
        np.fill_diagonal(knm_template, 0.0)
        kc = critical_coupling(omegas, knm_template)

        if kc == float("inf"):
            return
        # K = K_c / 10 → should not synchronise
        K = kc * 0.1
        knm = knm_template * K
        eng = UPDEEngine(n, dt=0.01)
        alpha = np.zeros((n, n))
        phases = rng.uniform(0, TWO_PI, n)
        for _ in range(1000):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        R, _ = compute_order_parameter(phases)
        assert R < 0.7


class TestStuartLandauAmplitudeProperty:
    """Stuart-Landau: r → √μ is a property, not just parametrised."""

    @given(mu=st.floats(min_value=0.1, max_value=5.0))
    @settings(max_examples=20, deadline=None)
    def test_amplitude_converges_to_sqrt_mu(self, mu: float) -> None:
        from scpn_phase_orchestrator.upde.stuart_landau import (
            StuartLandauEngine,
        )

        n = 2
        eng = StuartLandauEngine(n, dt=0.001)
        omegas = np.array([1.0, 1.5])
        mu_arr = np.full(n, mu)
        knm = np.zeros((n, n))
        knm_r = np.zeros((n, n))
        alpha = np.zeros((n, n))
        state = np.array([0.0, 0.5, 0.5, 0.5])  # [θ₀, θ₁, r₀, r₁]
        for _ in range(5000):
            state = eng.step(state, omegas, mu_arr, knm, knm_r, 0.0, 0.0, alpha)
        r_mean = float(np.mean(state[n:]))
        r_theory = np.sqrt(mu)
        assert abs(r_mean - r_theory) < 0.05


class TestFreeRotationExact:
    """K=0: all engines must produce exact free rotation θ(t) = θ₀ + ωt."""

    @pytest.mark.parametrize(
        "engine_cls,kwargs",
        [
            (UPDEEngine, {"method": "euler"}),
            (TorusEngine, {}),
            (SplittingEngine, {}),
        ],
    )
    def test_free_rotation(self, engine_cls, kwargs) -> None:
        n = 4
        dt = 0.01
        n_steps = 100
        eng = engine_cls(n, dt=dt, **kwargs)
        omegas = np.array([1.0, 2.0, -1.5, 3.0])
        phases = np.zeros(n)
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))
        for _ in range(n_steps):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        expected = (omegas * dt * n_steps) % TWO_PI
        np.testing.assert_allclose(phases, expected, atol=1e-6)
