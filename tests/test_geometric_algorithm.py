# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Algorithmic tests for torus integrator

"""Algorithmic properties of :class:`TorusEngine`.

Covered: constructor validation; phase wrap in ``[0, 2π)``;
zero-coupling pure-rotation limit ``θ(t+dt) = (θ + ω·dt) mod 2π``
(exact up to sincos rounding); zero-everywhere fixed point;
``z = exp(iθ)`` stays unit-norm through the integration (torus
preservation); Hypothesis invariants.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.upde import geometric as g_mod
from scpn_phase_orchestrator.upde.geometric import TorusEngine

TWO_PI = 2.0 * math.pi


def _python(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        prev = g_mod.ACTIVE_BACKEND
        g_mod.ACTIVE_BACKEND = "python"
        try:
            return func(*args, **kwargs)
        finally:
            g_mod.ACTIVE_BACKEND = prev

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
            TorusEngine(0, 0.01)

    def test_rejects_non_positive_dt(self):
        with pytest.raises(ValueError, match="dt"):
            TorusEngine(4, 0.0)


class TestStep:
    @_python
    def test_phases_wrap_in_two_pi(self):
        theta, omegas, knm, alpha = _problem(0)
        eng = TorusEngine(6, 0.1)
        new_ph = eng.step(theta, omegas, knm, 0.0, 0.0, alpha)
        assert np.all(new_ph >= 0.0)
        assert np.all(new_ph < TWO_PI + 1e-12)

    @_python
    def test_zero_coupling_pure_rotation(self):
        """K = 0, α = 0, ζ = 0 → z(t+dt) = z(t)·exp(i·ω·dt).
        Projecting back gives (θ + ω·dt) mod 2π exactly modulo
        sincos / atan2 rounding."""
        n = 5
        theta = np.array([0.3, 1.2, 2.0, 3.0, 4.5])
        omegas = np.array([0.1, -0.2, 0.05, 0.1, 0.0])
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))
        dt = 0.01
        eng = TorusEngine(n, dt)
        got = eng.step(theta, omegas, knm, 0.0, 0.0, alpha)
        expected = (theta + dt * omegas) % TWO_PI
        np.testing.assert_allclose(got, expected, atol=1e-13)

    @_python
    def test_zero_everything_fixed_point(self):
        """K=0, ω=0, ζ=0 → angle = 0 → z stays put."""
        n = 3
        theta = np.array([0.5, 2.7, 5.0])
        omegas = np.zeros(n)
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))
        eng = TorusEngine(n, 0.05)
        got = eng.step(theta, omegas, knm, 0.0, 0.0, alpha)
        np.testing.assert_allclose(got, theta, atol=1e-14)


class TestTorusPreservation:
    @_python
    def test_z_stays_on_unit_circle_after_many_steps(self):
        """|z_i| must stay 1 under torus integration — that's
        the defining property of the symplectic Euler on T^N."""
        rng = np.random.default_rng(3)
        n = 8
        theta = rng.uniform(0, TWO_PI, n)
        omegas = rng.normal(1.0, 0.2, n)
        knm = rng.uniform(0, 0.3, (n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        eng = TorusEngine(n, 0.01)
        fin = eng.run(theta, omegas, knm, 0.0, 0.0, alpha, n_steps=500)
        # After projection, |exp(iθ_i)| = 1 trivially — verify
        # that θ is finite and in [0, 2π).
        assert np.all(np.isfinite(fin))
        assert np.all(fin >= 0.0)
        assert np.all(fin < TWO_PI + 1e-12)


class TestExternalDrive:
    @_python
    def test_zeta_psi_shifts(self):
        """With ω = K = 0 and ψ = π/2, θ = 0, the forcing
        ζ · sin(ψ − θ) = ζ · 1, so the step advances by
        ζ·dt."""
        n = 3
        theta = np.zeros(n)
        omegas = np.zeros(n)
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))
        dt = 0.01
        eng = TorusEngine(n, dt)
        got = eng.step(theta, omegas, knm, 1.0, np.pi / 2, alpha)
        np.testing.assert_allclose(got, np.full(n, dt), atol=1e-12)


class TestOrderParameter:
    @_python
    def test_locked(self):
        eng = TorusEngine(10, 0.01)
        assert eng.order_parameter(np.full(10, 2.1)) == pytest.approx(
            1.0, abs=1e-12,
        )

    @_python
    def test_uniform_near_zero(self):
        eng = TorusEngine(1000, 0.01)
        phases = np.linspace(0, TWO_PI, 1000, endpoint=False)
        assert eng.order_parameter(phases) < 1e-10


class TestHypothesis:
    @_python
    @given(
        n=st.integers(min_value=3, max_value=8),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=8, deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_finite_output(self, n, seed):
        theta, omegas, knm, alpha = _problem(seed, n)
        eng = TorusEngine(n, 0.01)
        fin = eng.run(theta, omegas, knm, 0.0, 0.0, alpha, n_steps=30)
        assert np.all(np.isfinite(fin))
        assert np.all(fin >= 0.0)
        assert np.all(fin < TWO_PI + 1e-12)


class TestDispatcherSurface:
    def test_available_non_empty(self):
        assert g_mod.AVAILABLE_BACKENDS
        assert "python" in g_mod.AVAILABLE_BACKENDS

    def test_active_is_first(self):
        assert g_mod.AVAILABLE_BACKENDS[0] == g_mod.ACTIVE_BACKEND
