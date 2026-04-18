# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Stability invariants for Strang splitting

"""Long-run stability for :class:`SplittingEngine`.

The key promise of Strang splitting is that the ``ω``-rotation
direction carries **no** truncation error — so a pure-ω
trajectory over many steps equals ``θ₀ + ω·T mod 2π``
with machine precision. We test that invariant directly along
with the usual long-run bounds.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest

from scpn_phase_orchestrator.upde import splitting as sp_mod
from scpn_phase_orchestrator.upde.splitting import SplittingEngine

TWO_PI = 2.0 * math.pi

pytestmark = pytest.mark.slow


def _python(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        prev = sp_mod.ACTIVE_BACKEND
        sp_mod.ACTIVE_BACKEND = "python"
        try:
            return func(*args, **kwargs)
        finally:
            sp_mod.ACTIVE_BACKEND = prev

    return wrapper


class TestExactRotationProperty:
    @_python
    def test_pure_omega_flow_has_no_truncation_error(self):
        """K = 0, α = 0, ζ = 0 → Strang collapses to exact
        rotation. Over 1000 steps at dt = 0.01 the angular
        distance from θ₀ + 1000·dt·ω should stay within
        O(n_steps · eps) — easily below 1e-10."""
        n = 5
        rng = np.random.default_rng(0)
        theta = rng.uniform(0, TWO_PI, n)
        omegas = rng.normal(1.0, 0.3, n)
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))
        dt = 0.01
        n_steps = 1000
        eng = SplittingEngine(n, dt)
        fin = eng.run(theta, omegas, knm, 0.0, 0.0, alpha, n_steps=n_steps)
        expected = (theta + n_steps * dt * omegas) % TWO_PI
        # Wrap-aware circular distance.
        diff = np.minimum(
            np.abs(fin - expected),
            TWO_PI - np.abs(fin - expected),
        )
        assert np.max(diff) < 1e-10


class TestLongRunBounds:
    @_python
    def test_phases_stay_wrapped_over_500_steps(self):
        rng = np.random.default_rng(1)
        n = 8
        theta = rng.uniform(0, TWO_PI, n)
        omegas = rng.normal(1.0, 0.2, n)
        knm = rng.uniform(0, 0.3, (n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        eng = SplittingEngine(n, 0.01)
        fin = eng.run(theta, omegas, knm, 0.3, 1.0, alpha, n_steps=500)
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
        eng = SplittingEngine(n, 0.01)
        fin = eng.run(theta, omegas, knm, 0.0, 0.0, alpha, n_steps=400)
        assert np.all(np.isfinite(fin))


class TestSyncFixedPoint:
    @_python
    def test_fully_synced_with_zero_omega(self):
        """θ_i = θ_j and ω = 0 → coupling = 0, rotation = 0 →
        fixed point."""
        n = 6
        theta = np.full(n, 2.5)
        omegas = np.zeros(n)
        knm = np.ones((n, n)) / n
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        eng = SplittingEngine(n, 0.01)
        fin = eng.run(theta, omegas, knm, 0.0, 0.0, alpha, n_steps=500)
        np.testing.assert_allclose(fin, theta, atol=1e-12)
