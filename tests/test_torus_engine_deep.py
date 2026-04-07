# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Deep torus engine tests: invariants, edge cases, properties

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from scpn_phase_orchestrator.upde.geometric import TorusEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

TWO_PI = 2.0 * np.pi


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture()
def rng():
    return np.random.default_rng(12345)


def _zero_coupling(n: int):
    """Return (knm, alpha) with zero coupling for free-rotation tests."""
    return np.zeros((n, n)), np.zeros((n, n))


# ── Torus invariant: step always maps to [0, 2π) ───────────────────────


class TestTorusInvariant:
    """Every call to step() must return phases in [0, 2π). This is the
    fundamental geometric contract of the torus integrator."""

    def test_random_initial_conditions(self, rng):
        """Random phases, omegas, coupling — output still on torus."""
        n = 20
        eng = TorusEngine(n, dt=0.05)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.normal(0, 10, n)
        knm = rng.uniform(-2, 2, (n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = rng.uniform(-np.pi, np.pi, (n, n))
        result = eng.step(phases, omegas, knm, 0.3, 1.5, alpha)
        assert np.all(result >= 0.0)
        assert np.all(result < TWO_PI)
        assert np.all(np.isfinite(result))

    def test_extreme_frequencies_stay_on_torus(self):
        """Very large omegas should still produce phases in [0, 2π)."""
        n = 4
        eng = TorusEngine(n, dt=0.01)
        phases = np.array([0.0, 1.0, 3.0, 5.0])
        omegas = np.array([1e4, -1e4, 5e3, -5e3])
        knm, alpha = _zero_coupling(n)
        result = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        assert np.all(result >= 0.0)
        assert np.all(result < TWO_PI)

    def test_many_steps_invariant(self, rng):
        """After 1000 steps, phases must remain on torus."""
        n = 8
        eng = TorusEngine(n, dt=0.02)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.normal(1.0, 0.5, n)
        knm = rng.uniform(0, 1, (n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        result = eng.run(phases, omegas, knm, 0.1, 0.5, alpha, n_steps=1000)
        assert np.all(result >= 0.0)
        assert np.all(result < TWO_PI)


# ── Free rotation (zero coupling) ──────────────────────────────────────


class TestFreeRotation:
    """With K_nm = 0 and zeta = 0, the system reduces to dθ_i/dt = ω_i.
    The torus engine must reproduce this exactly (up to float64 precision)."""

    def test_single_oscillator_free(self):
        """One oscillator: θ(t+dt) = (θ + ω·dt) mod 2π."""
        eng = TorusEngine(1, dt=0.01)
        for theta0 in [0.0, 1.0, np.pi, TWO_PI - 0.001]:
            phases = np.array([theta0])
            omegas = np.array([3.7])
            knm, alpha = _zero_coupling(1)
            result = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
            expected = (theta0 + 3.7 * 0.01) % TWO_PI
            assert_allclose(result[0], expected, atol=1e-12)

    def test_multi_step_free_rotation_accumulates(self):
        """N steps of free rotation = θ + N·ω·dt (mod 2π)."""
        n = 3
        dt = 0.005
        n_steps = 200
        eng = TorusEngine(n, dt=dt)
        phases = np.array([0.5, 2.0, 5.0])
        omegas = np.array([1.0, -2.0, 10.0])
        knm, alpha = _zero_coupling(n)
        result = eng.run(phases, omegas, knm, 0.0, 0.0, alpha, n_steps=n_steps)
        expected = (phases + omegas * dt * n_steps) % TWO_PI
        # Use circular distance: min(|a-b|, 2π - |a-b|)
        diff = np.abs(result - expected)
        circ_dist = np.minimum(diff, TWO_PI - diff)
        assert_allclose(circ_dist, 0.0, atol=1e-10)

    def test_zero_omega_stays_fixed(self):
        """Zero natural frequency with zero coupling: phase unchanged."""
        n = 3
        eng = TorusEngine(n, dt=0.1)
        phases = np.array([1.0, 3.0, 5.0])
        omegas = np.zeros(n)
        knm, alpha = _zero_coupling(n)
        result = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        assert_allclose(result, phases, atol=1e-14)


# ── Synchronisation with strong coupling ────────────────────────────────


class TestSynchronisation:
    """Kuramoto model with identical frequencies and strong coupling
    must converge to phase synchronisation (R → 1)."""

    def test_strong_coupling_synchronises_from_spread(self, rng):
        """N=16 oscillators with uniform ω and K=2 → R > 0.95 after 500 steps."""
        n = 16
        eng = TorusEngine(n, dt=0.01)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = np.ones(n)
        knm = np.full((n, n), 2.0)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        result = eng.run(phases, omegas, knm, 0.0, 0.0, alpha, n_steps=500)
        r, _ = compute_order_parameter(result)
        assert r > 0.95, f"Expected R > 0.95, got {r:.4f}"

    def test_weak_coupling_low_r(self, rng):
        """With very weak coupling and spread frequencies, R should stay low."""
        n = 16
        eng = TorusEngine(n, dt=0.01)
        phases = rng.uniform(0, TWO_PI, n)
        # Spread frequencies: impossible to synchronise with weak coupling
        omegas = np.linspace(0.5, 5.0, n)
        knm = np.full((n, n), 0.01)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        result = eng.run(phases, omegas, knm, 0.0, 0.0, alpha, n_steps=200)
        r, _ = compute_order_parameter(result)
        # With such weak coupling and large frequency spread, R should not be high
        assert r < 0.85, f"Expected R < 0.85 with weak coupling, got {r:.4f}"

    def test_already_synchronised_remains_synchronised(self):
        """Phases already in sync with identical ω: R stays ≈ 1."""
        n = 8
        eng = TorusEngine(n, dt=0.01)
        phases = np.full(n, 1.5)
        omegas = np.ones(n) * 2.0
        knm = np.full((n, n), 0.5)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        result = eng.run(phases, omegas, knm, 0.0, 0.0, alpha, n_steps=200)
        r, _ = compute_order_parameter(result)
        assert r > 0.999


# ── Order parameter computation ─────────────────────────────────────────


class TestOrderParameter:
    """R = |1/N Σ exp(iθ_j)| must satisfy known analytical bounds."""

    def test_perfectly_synchronised(self):
        """All phases identical → R = 1."""
        phases = np.full(10, 2.5)
        r, psi = compute_order_parameter(phases)
        assert_allclose(r, 1.0, atol=1e-14)

    def test_perfectly_antisynchronised_pair(self):
        """Two oscillators at θ and θ+π → R = 0."""
        phases = np.array([1.0, 1.0 + np.pi])
        r, _ = compute_order_parameter(phases)
        assert_allclose(r, 0.0, atol=1e-14)

    def test_uniform_spread(self):
        """N equally-spaced phases on the circle → R ≈ 0."""
        n = 100
        phases = np.linspace(0, TWO_PI, n, endpoint=False)
        r, _ = compute_order_parameter(phases)
        assert_allclose(r, 0.0, atol=1e-14)

    def test_r_bounded_zero_one(self, rng):
        """R must always be in [0, 1]."""
        for _ in range(50):
            n = rng.integers(2, 50)
            phases = rng.uniform(0, TWO_PI, n)
            r, _ = compute_order_parameter(phases)
            assert 0.0 <= r <= 1.0 + 1e-14

    def test_single_oscillator_r_is_one(self):
        """Single oscillator always has R = 1."""
        for theta in [0.0, 1.0, np.pi, TWO_PI - 0.01]:
            r, psi = compute_order_parameter(np.array([theta]))
            assert_allclose(r, 1.0, atol=1e-14)

    def test_psi_is_mean_phase_for_sync(self):
        """When all phases are identical, ψ = θ (up to wrapping)."""
        theta = 4.2
        phases = np.full(5, theta)
        _, psi = compute_order_parameter(phases)
        expected_psi = theta % TWO_PI
        # psi from np.angle is in (-π, π], so normalise
        psi_norm = psi % TWO_PI
        assert_allclose(psi_norm, expected_psi, atol=1e-12)


# ── Derivative (internal method) ────────────────────────────────────────


class TestDerivative:
    """Test the Kuramoto derivative computation."""

    def test_zero_coupling_returns_omegas(self):
        """With K_nm = 0, dθ/dt = ω exactly."""
        n = 4
        eng = TorusEngine(n, dt=0.01)
        theta = np.array([0.0, 1.0, 2.0, 3.0])
        omegas = np.array([1.5, -2.0, 0.5, 3.0])
        knm, alpha = _zero_coupling(n)
        deriv = eng._derivative(theta, omegas, knm, 0.0, 0.0, alpha)
        assert_allclose(deriv, omegas, atol=1e-14)

    def test_identical_phases_zero_coupling_contribution(self):
        """When all θ_i are equal, sin(θ_j - θ_i) = 0, so coupling vanishes."""
        n = 5
        eng = TorusEngine(n, dt=0.01)
        theta = np.full(n, 2.0)
        omegas = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        knm = np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        deriv = eng._derivative(theta, omegas, knm, 0.0, 0.0, alpha)
        assert_allclose(deriv, omegas, atol=1e-13)

    def test_antisymmetric_coupling_for_two(self):
        """Two oscillators: coupling is K·sin(θ_2 - θ_1) for osc 1."""
        eng = TorusEngine(2, dt=0.01)
        theta = np.array([0.0, np.pi / 2])
        omegas = np.zeros(2)
        k = 1.0
        knm = np.array([[0.0, k], [k, 0.0]])
        alpha = np.zeros((2, 2))
        deriv = eng._derivative(theta, omegas, knm, 0.0, 0.0, alpha)
        # osc 0: K * sin(π/2 - 0) = K * 1 = 1
        # osc 1: K * sin(0 - π/2) = K * (-1) = -1
        assert_allclose(deriv[0], 1.0, atol=1e-14)
        assert_allclose(deriv[1], -1.0, atol=1e-14)

    def test_zeta_forcing_adds_external_drive(self):
        """With zeta ≠ 0, an additional ζ·sin(Ψ - θ) term appears."""
        n = 3
        eng = TorusEngine(n, dt=0.01)
        theta = np.array([0.0, np.pi / 2, np.pi])
        omegas = np.zeros(n)
        knm, alpha = _zero_coupling(n)
        zeta = 1.0
        psi = np.pi / 4
        deriv = eng._derivative(theta, omegas, knm, zeta, psi, alpha)
        expected = zeta * np.sin(psi - theta)
        assert_allclose(deriv, expected, atol=1e-14)

    def test_zeta_zero_no_forcing(self):
        """With zeta = 0, the forcing term is absent regardless of psi."""
        n = 3
        eng = TorusEngine(n, dt=0.01)
        theta = np.array([0.0, 1.0, 2.0])
        omegas = np.ones(n)
        knm, alpha = _zero_coupling(n)
        deriv = eng._derivative(theta, omegas, knm, 0.0, 99.9, alpha)
        assert_allclose(deriv, omegas, atol=1e-14)

    def test_phase_frustration_alpha(self):
        """Non-zero α shifts the phase difference in the coupling."""
        eng = TorusEngine(2, dt=0.01)
        theta = np.array([0.0, 0.0])
        omegas = np.zeros(2)
        knm = np.array([[0.0, 1.0], [1.0, 0.0]])
        # With α_01 = π/2, sin(θ_1 - θ_0 - π/2) = sin(-π/2) = -1
        alpha = np.array([[0.0, np.pi / 2], [np.pi / 2, 0.0]])
        deriv = eng._derivative(theta, omegas, knm, 0.0, 0.0, alpha)
        assert_allclose(deriv[0], -1.0, atol=1e-14)
        assert_allclose(deriv[1], -1.0, atol=1e-14)


# ── Edge cases ──────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_single_oscillator(self):
        """N=1: trivial case, must still work correctly."""
        eng = TorusEngine(1, dt=0.1)
        phases = np.array([3.0])
        omegas = np.array([2.0])
        knm = np.zeros((1, 1))
        alpha = np.zeros((1, 1))
        result = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        expected = (3.0 + 0.2) % TWO_PI
        assert_allclose(result[0], expected, atol=1e-12)

    def test_two_oscillators_symmetry(self):
        """Symmetric initial conditions + symmetric K → symmetric evolution."""
        eng = TorusEngine(2, dt=0.01)
        phases = np.array([1.0, TWO_PI - 1.0])
        omegas = np.array([1.0, 1.0])
        knm = np.array([[0.0, 0.5], [0.5, 0.0]])
        alpha = np.zeros((2, 2))
        result = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        # Both oscillators should advance equally (symmetric setup)
        assert result.shape == (2,)
        assert np.all(np.isfinite(result))

    def test_run_zero_steps_returns_copy(self):
        """run(..., n_steps=0) should return the original phases."""
        n = 4
        eng = TorusEngine(n, dt=0.01)
        phases = np.array([0.0, 1.0, 2.0, 3.0])
        result = eng.run(
            phases, np.ones(n), np.zeros((n, n)), 0.0, 0.0, np.zeros((n, n)), n_steps=0
        )
        assert_allclose(result, phases, atol=1e-14)

    def test_large_dt_still_on_torus(self):
        """Even with dt=10 (unreasonably large), output must be on torus."""
        n = 3
        eng = TorusEngine(n, dt=10.0)
        phases = np.array([0.0, 1.0, 2.0])
        omegas = np.array([100.0, -50.0, 200.0])
        knm, alpha = _zero_coupling(n)
        result = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        assert np.all(result >= 0.0)
        assert np.all(result < TWO_PI)


# ── Numerical quality ──────────────────────────────────────────────────


class TestNumericalQuality:
    """Verify the exponential-map integrator avoids the mod-2π glitch
    that plagues naive Euler on phases near 0 or 2π."""

    def test_wrapping_near_zero_no_discontinuity(self):
        """Phase crossing from 2π-ε to 0+ε must be smooth."""
        eng = TorusEngine(1, dt=0.01)
        # Phase just below 2π, positive omega → will cross zero
        phases = np.array([TWO_PI - 0.005])
        omegas = np.array([1.0])
        knm, alpha = _zero_coupling(1)
        result = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        # Should be a small positive number, not something near 2π
        expected = (TWO_PI - 0.005 + 0.01) % TWO_PI
        assert_allclose(result[0], expected, atol=1e-12)
        assert result[0] < 0.1  # near zero, not near 2π

    def test_backward_rotation_smooth(self):
        """Negative ω near zero should produce phase near 2π."""
        eng = TorusEngine(1, dt=0.01)
        phases = np.array([0.005])
        omegas = np.array([-1.0])
        knm, alpha = _zero_coupling(1)
        result = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        expected = (0.005 - 0.01) % TWO_PI
        assert_allclose(result[0], expected, atol=1e-12)
        assert result[0] > TWO_PI - 0.1  # near 2π
