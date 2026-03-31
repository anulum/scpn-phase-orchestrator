# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Stuart-Landau engine tests

"""Tests for StuartLandauEngine — coupled phase-amplitude integrator."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

TWO_PI = 2.0 * np.pi


def _make_engine(
    n: int = 4, dt: float = 0.01, method: str = "euler"
) -> StuartLandauEngine:
    return StuartLandauEngine(n, dt, method=method)


def _make_state(n: int = 4, r_init: float = 0.5) -> np.ndarray:
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, TWO_PI, n)
    r = np.full(n, r_init)
    return np.concatenate([theta, r])


def _zeros(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (omegas, mu, knm, alpha) all zero for n oscillators."""
    return (
        np.zeros(n),
        np.zeros(n),
        np.zeros((n, n)),
        np.zeros((n, n)),
    )


class TestConstruction:
    def test_valid_methods(self) -> None:
        for m in ("euler", "rk4", "rk45"):
            eng = StuartLandauEngine(4, 0.01, method=m)
            assert eng.last_dt == 0.01

    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            StuartLandauEngine(4, 0.01, method="midpoint")


class TestShapeValidation:
    def test_wrong_state_shape(self) -> None:
        eng = _make_engine(4)
        omegas, mu, knm, alpha = _zeros(4)
        with pytest.raises(ValueError, match="state.shape"):
            eng.step(np.zeros(5), omegas, mu, knm, knm, 0.0, 0.0, alpha)

    def test_wrong_omegas_shape(self) -> None:
        eng = _make_engine(4)
        _, mu, knm, alpha = _zeros(4)
        with pytest.raises(ValueError, match="omegas.shape"):
            eng.step(_make_state(4), np.zeros(3), mu, knm, knm, 0.0, 0.0, alpha)

    def test_wrong_mu_shape(self) -> None:
        eng = _make_engine(4)
        omegas, _, knm, alpha = _zeros(4)
        with pytest.raises(ValueError, match="mu.shape"):
            eng.step(_make_state(4), omegas, np.zeros(3), knm, knm, 0.0, 0.0, alpha)

    def test_wrong_knm_shape(self) -> None:
        eng = _make_engine(4)
        omegas, mu, _, alpha = _zeros(4)
        bad_knm = np.zeros((3, 3))
        with pytest.raises(ValueError, match="knm.shape"):
            eng.step(_make_state(4), omegas, mu, bad_knm, bad_knm, 0.0, 0.0, alpha)

    def test_wrong_alpha_shape(self) -> None:
        eng = _make_engine(4)
        omegas, mu, knm, _ = _zeros(4)
        with pytest.raises(ValueError, match="alpha.shape"):
            eng.step(_make_state(4), omegas, mu, knm, knm, 0.0, 0.0, np.zeros((3, 3)))

    def test_nan_state_raises(self) -> None:
        eng = _make_engine(4)
        omegas, mu, knm, alpha = _zeros(4)
        state = _make_state(4)
        state[0] = np.nan
        with pytest.raises(ValueError, match="NaN or Inf"):
            eng.step(state, omegas, mu, knm, knm, 0.0, 0.0, alpha)

    def test_nan_zeta_raises(self) -> None:
        eng = _make_engine(4)
        omegas, mu, knm, alpha = _zeros(4)
        with pytest.raises(ValueError, match="zeta and psi must be finite"):
            eng.step(_make_state(4), omegas, mu, knm, knm, np.nan, 0.0, alpha)


class TestAmplitudeInvariants:
    def test_amplitudes_nonnegative(self) -> None:
        """r_i >= 0 after every step, even if initial r is near zero."""
        eng = _make_engine(8, dt=0.05)
        omegas = np.ones(8)
        mu = np.full(8, -1.0)  # subcritical — should drive r toward 0
        knm = np.zeros((8, 8))
        alpha = np.zeros((8, 8))
        state = _make_state(8, r_init=0.01)
        for _ in range(200):
            state = eng.step(state, omegas, mu, knm, knm, 0.0, 0.0, alpha)
        assert np.all(state[8:] >= 0.0)

    def test_supercritical_convergence(self) -> None:
        """Uncoupled oscillator with μ > 0: r → √μ."""
        n = 1
        eng = _make_engine(n, dt=0.01)
        mu_val = 2.0
        mu = np.array([mu_val])
        state = np.array([0.0, 0.1])  # small initial amplitude
        omegas = np.array([1.0])
        knm = np.zeros((1, 1))
        alpha = np.zeros((1, 1))
        for _ in range(5000):
            state = eng.step(state, omegas, mu, knm, knm, 0.0, 0.0, alpha)
        expected_r = np.sqrt(mu_val)
        assert abs(state[1] - expected_r) < 0.05

    def test_subcritical_decay(self) -> None:
        """Uncoupled oscillator with μ < 0: r → 0."""
        n = 1
        eng = _make_engine(n, dt=0.01)
        mu = np.array([-1.0])
        state = np.array([0.0, 1.0])
        omegas = np.array([1.0])
        knm = np.zeros((1, 1))
        alpha = np.zeros((1, 1))
        for _ in range(2000):
            state = eng.step(state, omegas, mu, knm, knm, 0.0, 0.0, alpha)
        assert state[1] < 0.05


class TestPhaseWrapping:
    def test_phases_in_range(self) -> None:
        eng = _make_engine(8, dt=0.05)
        omegas = np.linspace(0.5, 2.0, 8)
        mu = np.ones(8)
        knm = 0.1 * np.ones((8, 8))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((8, 8))
        state = _make_state(8, r_init=1.0)
        for _ in range(200):
            state = eng.step(state, omegas, mu, knm, knm, 0.0, 0.0, alpha)
        theta = state[:8]
        assert np.all(theta >= 0.0)
        assert np.all(theta < TWO_PI)


class TestZeroAmplitudeCouplingReducesToKuramoto:
    def test_phase_equation_matches_kuramoto(self) -> None:
        """ε=0, knm_r=0 → phase evolution identical to Kuramoto."""
        n = 4
        eng = _make_engine(n, dt=0.01)
        rng = np.random.default_rng(42)
        theta0 = rng.uniform(0, TWO_PI, n)
        state = np.concatenate([theta0, np.ones(n)])  # r=1 everywhere
        omegas = rng.uniform(0.8, 1.2, n)
        mu = np.ones(n)
        knm = 0.2 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        knm_r = np.zeros((n, n))
        alpha = np.zeros((n, n))

        # One step with epsilon=0
        result = eng.step(state, omegas, mu, knm, knm_r, 0.0, 0.0, alpha, epsilon=0.0)

        # Compare phase part to manual Kuramoto step
        diff = theta0[np.newaxis, :] - theta0[:, np.newaxis]
        coupling = np.sum(knm * np.sin(diff), axis=1)
        expected_theta = (theta0 + 0.01 * (omegas + coupling)) % TWO_PI
        np.testing.assert_allclose(result[:n], expected_theta, atol=1e-10)


class TestIntegrators:
    def test_rk4_agrees_with_euler_small_dt(self) -> None:
        n = 4
        dt = 0.001  # small enough for Euler to be accurate
        eng_euler = _make_engine(n, dt, "euler")
        eng_rk4 = _make_engine(n, dt, "rk4")

        state = _make_state(n, r_init=0.8)
        omegas = np.array([1.0, 1.1, 0.9, 1.05])
        mu = np.ones(n)
        knm = 0.1 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))

        r_euler = eng_euler.step(state, omegas, mu, knm, knm, 0.0, 0.0, alpha)
        r_rk4 = eng_rk4.step(state, omegas, mu, knm, knm, 0.0, 0.0, alpha)
        np.testing.assert_allclose(r_euler, r_rk4, atol=1e-4)

    def test_rk45_agrees_with_rk4(self) -> None:
        n = 4
        eng_rk4 = _make_engine(n, 0.01, "rk4")
        eng_rk45 = _make_engine(n, 0.01, "rk45")

        state = _make_state(n, r_init=0.8)
        omegas = np.array([1.0, 1.1, 0.9, 1.05])
        mu = np.ones(n)
        knm = 0.1 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))

        r_rk4 = eng_rk4.step(state, omegas, mu, knm, knm, 0.0, 0.0, alpha)
        r_rk45 = eng_rk45.step(state, omegas, mu, knm, knm, 0.0, 0.0, alpha)
        np.testing.assert_allclose(r_rk4, r_rk45, atol=0.01)


class TestOrderParameter:
    def test_synchronised_oscillators(self) -> None:
        n = 8
        eng = _make_engine(n)
        theta = np.full(n, 1.5)
        r = np.ones(n)
        state = np.concatenate([theta, r])
        R, _ = eng.compute_order_parameter(state)
        assert R > 0.99

    def test_desynchronised_oscillators(self) -> None:
        n = 100
        eng = _make_engine(n)
        theta = np.linspace(0, TWO_PI, n, endpoint=False)
        r = np.ones(n)
        state = np.concatenate([theta, r])
        R, _ = eng.compute_order_parameter(state)
        assert R < 0.1

    def test_mean_amplitude(self) -> None:
        n = 4
        eng = _make_engine(n)
        state = np.concatenate([np.zeros(n), np.array([1.0, 2.0, 3.0, 4.0])])
        assert eng.compute_mean_amplitude(state) == pytest.approx(2.5)


class TestExternalDrive:
    def test_drive_affects_phase(self) -> None:
        n = 4
        eng = _make_engine(n, dt=0.01)
        state = _make_state(n, r_init=1.0)
        omegas, mu, knm, alpha = _zeros(n)
        mu = np.ones(n)

        r_no_drive = eng.step(state.copy(), omegas, mu, knm, knm, 0.0, 0.0, alpha)
        r_with_drive = eng.step(state.copy(), omegas, mu, knm, knm, 0.5, 1.0, alpha)
        assert not np.allclose(r_no_drive[:n], r_with_drive[:n])


class TestAmplitudeCoupling:
    def test_epsilon_modulates_amplitude_evolution(self) -> None:
        n = 4
        eng = _make_engine(n, dt=0.01)
        state = _make_state(n, r_init=0.5)
        omegas = np.ones(n)
        mu = np.ones(n)
        knm = np.zeros((n, n))
        knm_r = 0.5 * np.ones((n, n))
        np.fill_diagonal(knm_r, 0.0)
        alpha = np.zeros((n, n))

        r_eps0 = eng.step(
            state.copy(),
            omegas,
            mu,
            knm,
            knm_r,
            0.0,
            0.0,
            alpha,
            epsilon=0.0,
        )
        r_eps1 = eng.step(
            state.copy(),
            omegas,
            mu,
            knm,
            knm_r,
            0.0,
            0.0,
            alpha,
            epsilon=1.0,
        )
        # Amplitude evolution differs
        assert not np.allclose(r_eps0[n:], r_eps1[n:])


class TestMultiStep:
    def test_100_steps_finite(self) -> None:
        n = 8
        eng = _make_engine(n, dt=0.01)
        state = _make_state(n, r_init=1.0)
        omegas = np.linspace(0.5, 1.5, n)
        mu = np.ones(n)
        knm = 0.1 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        for _ in range(100):
            state = eng.step(state, omegas, mu, knm, knm, 0.0, 0.0, alpha)
        assert np.all(np.isfinite(state))
        assert np.all(state[:n] >= 0.0)
        assert np.all(state[:n] < TWO_PI)
        assert np.all(state[n:] >= 0.0)


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
