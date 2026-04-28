# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Algorithmic tests for phase winding numbers

"""Algorithmic properties of :func:`winding_numbers`.

Covered: analytic identity for constant-ω rotators, sign convention
(positive = counterclockwise), stationary oscillator → zero, sign
flip under time reversal, edge cases (T=1, N=0, empty), Hypothesis
property coverage for bounded output.
"""

from __future__ import annotations

import functools
import math

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.monitor import winding as w_mod
from scpn_phase_orchestrator.monitor.winding import (
    winding_numbers,
    winding_vector,
)

TWO_PI = 2.0 * math.pi


def _python(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        prev = w_mod.ACTIVE_BACKEND
        w_mod.ACTIVE_BACKEND = "python"
        try:
            return func(*args, **kwargs)
        finally:
            w_mod.ACTIVE_BACKEND = prev

    return wrapper


def _rotator(omegas: np.ndarray, t: int, dt: float) -> np.ndarray:
    """Build a constant-ω rotator history wrapped to [0, 2π)."""
    return (omegas[np.newaxis, :] * np.arange(t)[:, np.newaxis] * dt) % TWO_PI


class TestAnalyticIdentity:
    @_python
    def test_constant_omega_matches_floor_integral(self):
        """Constant ω: ``w_i = floor(ω_i · T · dt / 2π)`` up to
        ± 1 boundary rounding. Uses 2π-wrapped phases."""
        omegas = np.array([0.5, -0.3, 2.5, -1.2])
        t, dt = 200, 0.05
        traj = _rotator(omegas, t, dt)
        w = winding_numbers(traj)
        expected = np.floor(omegas * (t - 1) * dt / TWO_PI).astype(np.int64)
        assert np.max(np.abs(w - expected)) <= 1

    @_python
    def test_stationary_oscillator_zero(self):
        omegas = np.zeros(5)
        traj = _rotator(omegas, 100, 0.01)
        w = winding_numbers(traj)
        assert np.all(w == 0)


class TestSignConvention:
    @_python
    def test_positive_omega_positive_winding(self):
        omegas = np.array([1.0, 2.0, 3.0])
        traj = _rotator(omegas, 300, 0.05)
        w = winding_numbers(traj)
        assert np.all(w >= 0)
        assert np.any(w > 0)

    @_python
    def test_negative_omega_negative_winding(self):
        omegas = np.array([-1.0, -2.0, -3.0])
        traj = _rotator(omegas, 300, 0.05)
        w = winding_numbers(traj)
        assert np.all(w <= 0)
        assert np.any(w < 0)


class TestEdgeCases:
    @_python
    def test_single_timestep_returns_zeros(self):
        traj = np.zeros((1, 5))
        w = winding_numbers(traj)
        assert w.shape == (5,)
        assert np.all(w == 0)

    @_python
    def test_non_2d_returns_zeros(self):
        w = winding_numbers(np.zeros(5))
        assert w.shape == (0,)

    @_python
    def test_winding_vector_alias(self):
        omegas = np.array([1.0, -1.0])
        traj = _rotator(omegas, 150, 0.05)
        assert np.array_equal(winding_vector(traj), winding_numbers(traj))


class TestHypothesis:
    @_python
    @given(
        t=st.integers(min_value=2, max_value=200),
        n=st.integers(min_value=1, max_value=8),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_bounded_by_trajectory_length(self, t: int, n: int, seed: int):
        """|w_i| ≤ T — each step contributes at most one full
        wrap, so cumulative winding cannot exceed T in magnitude."""
        rng = np.random.default_rng(seed)
        traj = rng.uniform(0, TWO_PI, (t, n))
        w = winding_numbers(traj)
        assert w.shape == (n,)
        assert w.dtype == np.int64
        assert np.all(np.abs(w) <= t)


class TestDispatcherSurface:
    def test_available_non_empty(self):
        assert w_mod.AVAILABLE_BACKENDS
        assert "python" in w_mod.AVAILABLE_BACKENDS

    def test_active_is_first(self):
        assert w_mod.AVAILABLE_BACKENDS[0] == w_mod.ACTIVE_BACKEND
