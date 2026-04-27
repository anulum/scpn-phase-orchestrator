# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Algorithmic tests for inertial Kuramoto

"""Algorithmic properties of :class:`InertialKuramotoEngine`.

Covered: constructor validation; output shapes; phase wrap in
``[0, 2π)``; zero-coupling + zero-damping limit recovers free
rotation at constant ``ω = P/M``; zero-coupling + damping limit
pulls ``ω → 0`` over long integration; coherence / frequency
deviation helpers; Hypothesis-driven finite-output invariant.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.upde import inertial as i_mod
from scpn_phase_orchestrator.upde.inertial import InertialKuramotoEngine

TWO_PI = 2.0 * math.pi


def _python(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        prev = i_mod.ACTIVE_BACKEND
        i_mod.ACTIVE_BACKEND = "python"
        try:
            return func(*args, **kwargs)
        finally:
            i_mod.ACTIVE_BACKEND = prev

    return wrapper


def _problem(seed: int, n: int = 8):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, TWO_PI, n)
    omega_dot = rng.normal(0, 0.1, n)
    power = rng.normal(0, 0.5, n)
    knm = rng.uniform(0, 0.5, (n, n))
    np.fill_diagonal(knm, 0.0)
    inertia = np.ones(n)
    damping = np.ones(n) * 0.1
    return theta, omega_dot, power, knm, inertia, damping


class TestConstructor:
    def test_rejects_zero_n(self):
        with pytest.raises(ValueError, match="n"):
            InertialKuramotoEngine(n=0, dt=0.01)

    def test_rejects_non_positive_dt(self):
        with pytest.raises(ValueError, match="dt"):
            InertialKuramotoEngine(n=4, dt=0.0)


class TestStep:
    @_python
    def test_output_shapes(self):
        theta, od, p, k, m, d = _problem(0)
        eng = InertialKuramotoEngine(8, 0.01)
        new_th, new_od = eng.step(theta, od, p, k, m, d)
        assert new_th.shape == (8,)
        assert new_od.shape == (8,)

    @_python
    def test_phases_wrap_in_two_pi(self):
        theta, od, p, k, m, d = _problem(1)
        eng = InertialKuramotoEngine(8, 0.01)
        new_th, _ = eng.step(theta, od, p, k, m, d)
        assert np.all(new_th >= 0.0)
        assert np.all(new_th < TWO_PI + 1e-12)

    @_python
    def test_zero_coupling_no_damping_recovers_free_rotation(self):
        """With K=0, D=0, inertia=1, ``dθ/dt = ω`` and
        ``dω/dt = P`` — constant acceleration. After one RK4 step
        of duration ``dt``, exactness requires polynomial-in-dt
        truncation error = 0 for polynomials up to degree 4.
        Here θ(t) = θ₀ + ω₀ t + ½ P t², ω(t) = ω₀ + P t,
        both polynomial of degree 2, so RK4 is exact modulo
        floating-point."""
        n = 4
        theta = np.array([0.3, 1.1, 2.0, 3.0])
        omega = np.array([0.1, -0.2, 0.05, 0.0])
        power = np.array([0.1, 0.1, -0.05, 0.2])
        knm = np.zeros((n, n))
        inertia = np.ones(n)
        damping = np.zeros(n)
        dt = 0.01
        eng = InertialKuramotoEngine(n, dt)
        new_th, new_od = eng.step(theta, omega, power, knm, inertia, damping)
        expected_th = (theta + omega * dt + 0.5 * power * dt**2) % TWO_PI
        expected_od = omega + power * dt
        np.testing.assert_allclose(new_th, expected_th, atol=1e-14)
        np.testing.assert_allclose(new_od, expected_od, atol=1e-14)

    @_python
    def test_damping_decays_omega(self):
        """With K=0, P=0, finite D/M: ω(t) = ω₀ · exp(−(D/M)·t).
        RK4 gives a truncated expansion; after 50 steps at dt=0.01
        and D/M=2, ω should decay substantially."""
        n = 3
        theta = np.zeros(n)
        omega_init = np.ones(n) * 1.0
        power = np.zeros(n)
        knm = np.zeros((n, n))
        inertia = np.ones(n)
        damping = np.ones(n) * 2.0
        eng = InertialKuramotoEngine(n, 0.01)
        _, _, _, omega_traj = eng.run(
            theta,
            omega_init,
            power,
            knm,
            inertia,
            damping,
            n_steps=50,
        )
        # After t=0.5, analytical ω = exp(-1.0) ≈ 0.368
        assert np.all(np.abs(omega_traj[-1]) < 0.5)


class TestRun:
    @_python
    def test_trajectory_shapes(self):
        theta, od, p, k, m, d = _problem(3)
        eng = InertialKuramotoEngine(8, 0.01)
        final_th, final_od, th_traj, od_traj = eng.run(
            theta,
            od,
            p,
            k,
            m,
            d,
            n_steps=5,
        )
        assert final_th.shape == (8,)
        assert final_od.shape == (8,)
        assert th_traj.shape == (5, 8)
        assert od_traj.shape == (5, 8)


class TestHelpers:
    @_python
    def test_coherence_perfectly_locked(self):
        eng = InertialKuramotoEngine(10, 0.01)
        phases = np.full(10, 1.3)
        assert eng.coherence(phases) == pytest.approx(1.0, abs=1e-12)

    @_python
    def test_coherence_uniform_approaches_zero(self):
        eng = InertialKuramotoEngine(1000, 0.01)
        phases = np.linspace(0, TWO_PI, 1000, endpoint=False)
        assert eng.coherence(phases) < 1e-10

    @_python
    def test_frequency_deviation(self):
        eng = InertialKuramotoEngine(3, 0.01)
        omega = np.array([0.1, -0.5, 0.2])
        assert eng.frequency_deviation(omega) == pytest.approx(
            0.5 / TWO_PI,
            abs=1e-12,
        )


class TestHypothesis:
    @_python
    @given(
        n=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_finite_output(self, n, seed):
        theta, od, p, k, m, d = _problem(seed, n)
        eng = InertialKuramotoEngine(n, 0.01)
        new_th, new_od = eng.step(theta, od, p, k, m, d)
        assert np.all(np.isfinite(new_th))
        assert np.all(np.isfinite(new_od))


class TestDispatcherSurface:
    def test_available_non_empty(self):
        assert i_mod.AVAILABLE_BACKENDS
        assert "python" in i_mod.AVAILABLE_BACKENDS

    def test_active_is_first(self):
        assert i_mod.AVAILABLE_BACKENDS[0] == i_mod.ACTIVE_BACKEND
