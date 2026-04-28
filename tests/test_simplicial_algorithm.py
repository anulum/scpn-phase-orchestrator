# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Algorithmic tests for simplicial Kuramoto

"""Algorithmic properties of :class:`SimplicialEngine`.

Covered: constructor validation; phase wrap in ``[0, 2π)``;
``σ₂ = 0`` limit reduces to standard Kuramoto; fully-synchronised
state is an exact fixed point at any σ₂; explicit cross-check of
the ``Σ sin(θ_j + θ_k − 2θ_i) = 2 S_i C_i`` identity on a
hand-computed 3-oscillator example; ``sigma2`` setter validation;
external-drive ζ forcing; Hypothesis invariants.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.upde import simplicial as s_mod
from scpn_phase_orchestrator.upde.simplicial import SimplicialEngine

TWO_PI = 2.0 * math.pi


def _python(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        prev = s_mod.ACTIVE_BACKEND
        s_mod.ACTIVE_BACKEND = "python"
        try:
            return func(*args, **kwargs)
        finally:
            s_mod.ACTIVE_BACKEND = prev

    return wrapper


def _problem(seed: int, n: int = 6):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, TWO_PI, n)
    omegas = rng.normal(1.0, 0.2, n)
    knm = rng.uniform(0, 0.3, (n, n))
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))
    return theta, omegas, knm, alpha


class TestConstructor:
    def test_rejects_zero_n(self):
        with pytest.raises(ValueError, match="n_oscillators"):
            SimplicialEngine(0, 0.01)

    def test_rejects_non_positive_dt(self):
        with pytest.raises(ValueError, match="dt"):
            SimplicialEngine(4, 0.0)

    def test_rejects_negative_sigma2(self):
        with pytest.raises(ValueError, match="sigma2"):
            SimplicialEngine(4, 0.01, sigma2=-0.1)

    def test_sigma2_setter_rejects_negative(self):
        eng = SimplicialEngine(4, 0.01, sigma2=0.0)
        with pytest.raises(ValueError, match="sigma2"):
            eng.sigma2 = -0.5
        eng.sigma2 = 0.3
        assert eng.sigma2 == 0.3


class TestStep:
    @_python
    def test_phases_wrap_in_two_pi(self):
        theta, omegas, knm, alpha = _problem(0)
        eng = SimplicialEngine(6, 0.1, sigma2=0.5)
        new_ph = eng.step(theta, omegas, knm, 0.0, 0.0, alpha)
        assert np.all(new_ph >= 0.0)
        assert np.all(new_ph < TWO_PI + 1e-12)

    @_python
    def test_sigma2_zero_recovers_standard_kuramoto(self):
        """With σ₂ = 0 the 3-body contribution vanishes and the
        stepper must reproduce the standard Kuramoto Euler step."""
        n = 5
        rng = np.random.default_rng(2)
        theta = rng.uniform(0, TWO_PI, n)
        omegas = rng.normal(1.0, 0.2, n)
        knm = rng.uniform(0, 0.3, (n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        dt = 0.01
        eng = SimplicialEngine(n, dt, sigma2=0.0)
        got = eng.step(theta, omegas, knm, 0.0, 0.0, alpha)
        diff = theta[np.newaxis, :] - theta[:, np.newaxis]
        coupling = np.sum(knm * np.sin(diff), axis=1)
        expected = (theta + dt * (omegas + coupling)) % TWO_PI
        np.testing.assert_allclose(got, expected, atol=1e-12)


class TestThreeBodyIdentity:
    @_python
    def test_two_sc_identity_matches_direct_sum(self):
        """Brute-force Σ_{j,k} sin(θ_j + θ_k − 2θ_i) on a tiny
        N=4 example must equal 2·S_i·C_i where S_i, C_i are the
        sin/cos sums of phase differences used by the stepper."""
        n = 4
        theta = np.array([0.3, 1.2, 2.0, 3.5])
        omegas = np.zeros(n)
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))
        sigma2 = 1.0
        dt = 0.01
        eng = SimplicialEngine(n, dt, sigma2=sigma2)
        got = eng.step(theta, omegas, knm, 0.0, 0.0, alpha)
        # Direct O(N³) reference.
        direct = np.zeros(n)
        for i in range(n):
            acc = 0.0
            for j in range(n):
                for k in range(n):
                    acc += np.sin(theta[j] + theta[k] - 2 * theta[i])
            direct[i] = acc
        expected = (theta + dt * (omegas + sigma2 / (n * n) * direct)) % TWO_PI
        np.testing.assert_allclose(got, expected, atol=1e-12)


class TestSyncFixedPoint:
    @_python
    def test_fully_synced_is_fixed_point(self):
        """θ_i = θ_j = θ₀ → sin(θ_j − θ_i) = 0 and
        sin(θ_j + θ_k − 2θ_i) = 0 identically, so the
        derivative is exactly ω. With ω = 0 the synced state
        is preserved (mod 2π)."""
        n = 7
        theta = np.full(n, 1.3)
        omegas = np.zeros(n)
        knm = np.ones((n, n)) / n
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        eng = SimplicialEngine(n, 0.01, sigma2=0.8)
        fin = eng.run(theta, omegas, knm, 0.0, 0.0, alpha, n_steps=100)
        np.testing.assert_allclose(fin, theta, atol=1e-12)


class TestExternalDrive:
    @_python
    def test_zeta_shifts_trajectory(self):
        n = 3
        theta = np.zeros(n)
        omegas = np.zeros(n)
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))
        dt = 0.01
        eng = SimplicialEngine(n, dt, sigma2=0.0)
        no_drive = eng.step(theta, omegas, knm, 0.0, 0.0, alpha)
        drive = eng.step(theta, omegas, knm, 1.0, np.pi / 2, alpha)
        # With ψ = π/2 and θ = 0, forcing = sin(π/2) = 1 → step of dt.
        np.testing.assert_allclose(no_drive, theta, atol=1e-12)
        np.testing.assert_allclose(drive, theta + dt * 1.0, atol=1e-12)


class TestHypothesis:
    @_python
    @given(
        n=st.integers(min_value=3, max_value=6),
        sigma2=st.floats(min_value=0.0, max_value=2.0),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=8,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_finite_output(self, n, sigma2, seed):
        theta, omegas, knm, alpha = _problem(seed, n)
        eng = SimplicialEngine(n, 0.01, sigma2=sigma2)
        fin = eng.run(theta, omegas, knm, 0.0, 0.0, alpha, n_steps=30)
        assert np.all(np.isfinite(fin))
        assert np.all(fin >= 0.0)
        assert np.all(fin < TWO_PI + 1e-12)


class TestDispatcherSurface:
    def test_available_non_empty(self):
        assert s_mod.AVAILABLE_BACKENDS
        assert "python" in s_mod.AVAILABLE_BACKENDS

    def test_active_is_first(self):
        assert s_mod.AVAILABLE_BACKENDS[0] == s_mod.ACTIVE_BACKEND


class TestOrderParameter:
    @_python
    def test_locked(self):
        eng = SimplicialEngine(10, 0.01, sigma2=0.0)
        assert eng.order_parameter(np.full(10, 1.3)) == pytest.approx(
            1.0,
            abs=1e-12,
        )

    @_python
    def test_uniform_near_zero(self):
        eng = SimplicialEngine(1000, 0.01, sigma2=0.0)
        assert (
            eng.order_parameter(
                np.linspace(0, TWO_PI, 1000, endpoint=False),
            )
            < 1e-10
        )
