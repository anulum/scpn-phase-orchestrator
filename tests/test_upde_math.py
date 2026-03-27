# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for geometric, numerics, OA reduction

from __future__ import annotations

import math

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.geometric import TorusEngine
from scpn_phase_orchestrator.upde.numerics import IntegrationConfig, check_stability
from scpn_phase_orchestrator.upde.reduction import OAState, OttAntonsenReduction

TWO_PI = 2.0 * np.pi


def _connected_knm(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.3, 1.0, (n, n))
    knm = 0.5 * (raw + raw.T)
    np.fill_diagonal(knm, 0.0)
    return knm


# ── TorusEngine ──────────────────────────────────────────────────────────


class TestTorusEngine:
    def test_output_in_0_2pi(self) -> None:
        n = 5
        eng = TorusEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(-2, 2, n)
        knm = _connected_knm(n)
        alpha = np.zeros((n, n))
        out = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        assert np.all(out >= 0)
        assert np.all(out < TWO_PI + 1e-10)

    def test_output_length(self) -> None:
        n = 4
        eng = TorusEngine(n, dt=0.01)
        phases = np.zeros(n)
        omegas = np.ones(n)
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))
        out = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        assert len(out) == n

    def test_finite_output(self) -> None:
        n = 6
        eng = TorusEngine(n, dt=0.01)
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(-5, 5, n)
        knm = _connected_knm(n)
        alpha = np.zeros((n, n))
        out = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        assert np.all(np.isfinite(out))

    def test_run_n_steps(self) -> None:
        n = 3
        eng = TorusEngine(n, dt=0.01)
        phases = np.zeros(n)
        omegas = np.array([1.0, 2.0, 3.0])
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))
        out = eng.run(phases, omegas, knm, 0.0, 0.0, alpha, n_steps=100)
        assert len(out) == n
        assert not np.allclose(out, phases)

    def test_zeta_driving(self) -> None:
        n = 3
        eng = TorusEngine(n, dt=0.01)
        phases = np.zeros(n)
        omegas = np.zeros(n)
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))
        out_no_drive = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        out_drive = eng.step(phases, omegas, knm, 1.0, 1.0, alpha)
        assert not np.allclose(out_no_drive, out_drive)

    def test_zero_coupling_free_rotation(self) -> None:
        """K=0 → each oscillator rotates at its own ω."""
        n = 2
        eng = TorusEngine(n, dt=0.01)
        phases = np.array([0.0, 0.0])
        omegas = np.array([1.0, -1.0])
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))
        out = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        assert out[0] > 0  # positive ω → advances
        assert out[1] > np.pi  # negative ω → wraps backward


# ── Numerics ─────────────────────────────────────────────────────────────


class TestIntegrationConfig:
    def test_defaults(self) -> None:
        cfg = IntegrationConfig(dt=0.01)
        assert cfg.substeps == 1
        assert cfg.method == "euler"
        assert cfg.max_dt == 0.01

    def test_custom(self) -> None:
        cfg = IntegrationConfig(dt=0.005, substeps=4, method="rk4")
        assert cfg.substeps == 4
        assert cfg.method == "rk4"

    def test_frozen(self) -> None:
        cfg = IntegrationConfig(dt=0.01)
        with pytest.raises(AttributeError):
            cfg.dt = 0.02  # type: ignore[misc]


class TestCheckStability:
    def test_stable(self) -> None:
        assert check_stability(0.01, max_omega=10.0, max_coupling=5.0) is True

    def test_unstable(self) -> None:
        assert check_stability(1.0, max_omega=10.0, max_coupling=5.0) is False

    def test_zero_deriv(self) -> None:
        assert check_stability(100.0, max_omega=0.0, max_coupling=0.0) is True

    def test_boundary(self) -> None:
        # dt * (omega + coupling) = pi → False (not strictly less)
        dt = math.pi / 15.0
        assert check_stability(dt, 10.0, 5.0) is False

    @pytest.mark.parametrize("dt", [0.001, 0.005, 0.01])
    def test_small_dt_stable(self, dt: float) -> None:
        assert check_stability(dt, max_omega=5.0, max_coupling=5.0) is True


# ── OttAntonsenReduction ─────────────────────────────────────────────────


class TestOttAntonsenReduction:
    def test_k_critical(self) -> None:
        oa = OttAntonsenReduction(omega_0=0.0, delta=0.5, K=3.0)
        assert oa.K_c == 1.0

    def test_steady_state_above_kc(self) -> None:
        """K > K_c → R_ss > 0."""
        oa = OttAntonsenReduction(omega_0=0.0, delta=0.5, K=4.0)
        R_ss = oa.steady_state_R()
        assert 0.0 < R_ss < 1.0
        expected = np.sqrt(1.0 - 2 * 0.5 / 4.0)
        assert abs(R_ss - expected) < 1e-12

    def test_steady_state_below_kc(self) -> None:
        """K < K_c → R_ss = 0."""
        oa = OttAntonsenReduction(omega_0=0.0, delta=1.0, K=1.0)
        assert oa.steady_state_R() == 0.0

    def test_negative_delta_raises(self) -> None:
        with pytest.raises(ValueError):
            OttAntonsenReduction(omega_0=0.0, delta=-1.0, K=1.0)

    def test_step_preserves_boundedness(self) -> None:
        oa = OttAntonsenReduction(omega_0=0.0, delta=0.5, K=3.0)
        z = complex(0.5, 0.0)
        for _ in range(100):
            z = oa.step(z)
        assert abs(z) <= 1.0 + 1e-6

    def test_run_returns_oa_state(self) -> None:
        oa = OttAntonsenReduction(omega_0=0.0, delta=0.5, K=3.0)
        state = oa.run(complex(0.1, 0.0), n_steps=100)
        assert isinstance(state, OAState)
        assert 0.0 <= state.R <= 1.0 + 1e-6
        assert state.K_c == 1.0

    def test_run_converges_above_kc(self) -> None:
        """K > K_c → R converges to R_ss."""
        oa = OttAntonsenReduction(omega_0=0.0, delta=0.5, K=4.0, dt=0.01)
        state = oa.run(complex(0.01, 0.0), n_steps=5000)
        R_ss = oa.steady_state_R()
        assert abs(state.R - R_ss) < 0.05

    def test_predict_from_oscillators(self) -> None:
        oa = OttAntonsenReduction(omega_0=0.0, delta=1.0, K=5.0)
        omegas = np.random.default_rng(0).standard_cauchy(50) * 0.5
        omegas = np.clip(omegas, -5, 5)
        state = oa.predict_from_oscillators(omegas, K=5.0)
        assert isinstance(state, OAState)
        assert 0.0 <= state.R <= 1.0 + 1e-6
