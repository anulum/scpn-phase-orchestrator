# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Stability invariants for torus integrator

"""Long-run stability invariants for :class:`TorusEngine`.

The symplectic-Euler-on-T^N has no ``mod 2π`` discontinuity, so
the pure-ω trajectory is reproduced to sincos rounding over any
number of steps.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest

from scpn_phase_orchestrator.upde import geometric as g_mod
from scpn_phase_orchestrator.upde.geometric import TorusEngine

TWO_PI = 2.0 * math.pi

pytestmark = pytest.mark.slow


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


class TestExactRotation:
    @_python
    def test_pure_omega_matches_analytical_over_1000_steps(self):
        """K = 0, α = 0, ζ = 0 → θ(t + N·dt) = θ + N·dt·ω (mod 2π).
        Symplectic Euler on T^N uses the exponential map, so the
        trajectory has no ``mod 2π`` discontinuity artefact.
        Agreement should stay below ``O(n·eps)``."""
        n = 5
        rng = np.random.default_rng(0)
        theta = rng.uniform(0, TWO_PI, n)
        omegas = rng.normal(1.0, 0.3, n)
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))
        dt = 0.01
        n_steps = 1000
        eng = TorusEngine(n, dt)
        fin = eng.run(theta, omegas, knm, 0.0, 0.0, alpha, n_steps=n_steps)
        expected = (theta + n_steps * dt * omegas) % TWO_PI
        # Wrap-aware distance.
        diff = np.minimum(
            np.abs(fin - expected),
            TWO_PI - np.abs(fin - expected),
        )
        assert np.max(diff) < 1e-10


class TestLongRunBounds:
    @_python
    def test_phases_stay_wrapped(self):
        rng = np.random.default_rng(1)
        n = 8
        theta = rng.uniform(0, TWO_PI, n)
        omegas = rng.normal(1.0, 0.3, n)
        knm = rng.uniform(0, 0.3, (n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        eng = TorusEngine(n, 0.01)
        fin = eng.run(theta, omegas, knm, 0.5, 1.0, alpha, n_steps=500)
        assert np.all(fin >= 0.0)
        assert np.all(fin < TWO_PI + 1e-12)

    @_python
    def test_outputs_remain_finite(self):
        rng = np.random.default_rng(2)
        n = 10
        theta = rng.uniform(0, TWO_PI, n)
        omegas = rng.normal(0.5, 0.3, n)
        knm = rng.uniform(0, 0.2, (n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = rng.uniform(-0.3, 0.3, (n, n))
        np.fill_diagonal(alpha, 0.0)
        eng = TorusEngine(n, 0.01)
        fin = eng.run(theta, omegas, knm, 0.0, 0.0, alpha, n_steps=400)
        assert np.all(np.isfinite(fin))


class TestSyncFixedPoint:
    @_python
    def test_fully_synced_zero_omega(self):
        """All equal phases + ω = 0 → coupling = 0 → fixed
        point."""
        n = 6
        theta = np.full(n, 2.5)
        omegas = np.zeros(n)
        knm = np.ones((n, n)) / n
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        eng = TorusEngine(n, 0.01)
        fin = eng.run(theta, omegas, knm, 0.0, 0.0, alpha, n_steps=500)
        np.testing.assert_allclose(fin, theta, atol=1e-12)
