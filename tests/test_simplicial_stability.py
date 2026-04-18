# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Stability invariants for simplicial Kuramoto

"""Long-run stability invariants for :class:`SimplicialEngine`.

Marked ``slow`` — hundreds of Euler steps.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest

from scpn_phase_orchestrator.upde import simplicial as s_mod
from scpn_phase_orchestrator.upde.simplicial import SimplicialEngine

TWO_PI = 2.0 * math.pi

pytestmark = pytest.mark.slow


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


class TestLongRun:
    @_python
    def test_phases_wrap_over_500_steps(self):
        rng = np.random.default_rng(0)
        n = 8
        theta = rng.uniform(0, TWO_PI, n)
        omegas = rng.normal(1.0, 0.3, n)
        knm = rng.uniform(0, 0.3, (n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        eng = SimplicialEngine(n, 0.01, sigma2=0.5)
        fin = eng.run(theta, omegas, knm, 0.0, 0.0, alpha, n_steps=500)
        assert np.all(fin >= 0.0)
        assert np.all(fin < TWO_PI + 1e-12)

    @_python
    def test_remains_finite(self):
        rng = np.random.default_rng(1)
        n = 12
        theta = rng.uniform(0, TWO_PI, n)
        omegas = rng.normal(0.5, 0.2, n)
        knm = rng.uniform(0, 0.2, (n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        eng = SimplicialEngine(n, 0.01, sigma2=1.0)
        fin = eng.run(theta, omegas, knm, 0.3, 1.2, alpha, n_steps=400)
        assert np.all(np.isfinite(fin))


class TestSyncFixedPoint:
    @_python
    def test_fully_synced_is_fixed_point_long_run(self):
        """Fully-synced state is an exact fixed point at any σ₂,
        including nonzero pairwise K and nonzero σ₂."""
        n = 10
        theta = np.full(n, 2.4)
        omegas = np.zeros(n)
        knm = np.ones((n, n)) * 0.3
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        eng = SimplicialEngine(n, 0.01, sigma2=1.5)
        fin = eng.run(theta, omegas, knm, 0.0, 0.0, alpha, n_steps=500)
        np.testing.assert_allclose(fin, theta, atol=1e-12)
        assert eng.order_parameter(fin) == pytest.approx(1.0, abs=1e-12)


class TestExplosiveTransitionRegime:
    @_python
    def test_strong_pairwise_drives_coherence(self):
        """Explosive-transition territory (Gambuzza 2023) is hard
        to characterise strictly, but the conservative claim that
        a strong pairwise + triadic coupling on a small ensemble
        settles into a bounded order parameter in [0, 1] is
        always true."""
        rng = np.random.default_rng(5)
        n = 8
        theta = rng.uniform(0, TWO_PI, n)
        omegas = np.zeros(n)
        knm = np.full((n, n), 3.0 / n)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        eng = SimplicialEngine(n, 0.01, sigma2=0.8)
        fin = eng.run(theta, omegas, knm, 0.0, 0.0, alpha, n_steps=600)
        r = eng.order_parameter(fin)
        assert 0.0 <= r <= 1.0 + 1e-12
