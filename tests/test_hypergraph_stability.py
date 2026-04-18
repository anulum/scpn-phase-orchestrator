# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Stability invariants for hypergraph Kuramoto

"""Long-run stability invariants for :class:`HypergraphEngine`.

Marked ``slow`` — hundreds of Euler steps with mixed pairwise +
k-body coupling.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest

from scpn_phase_orchestrator.upde import hypergraph as h_mod
from scpn_phase_orchestrator.upde.hypergraph import HypergraphEngine

TWO_PI = 2.0 * math.pi

pytestmark = pytest.mark.slow


def _python(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        prev = h_mod.ACTIVE_BACKEND
        h_mod.ACTIVE_BACKEND = "python"
        try:
            return func(*args, **kwargs)
        finally:
            h_mod.ACTIVE_BACKEND = prev

    return wrapper


class TestLongRunInvariants:
    @_python
    def test_phases_stay_wrapped_over_400_steps(self):
        rng = np.random.default_rng(0)
        n = 8
        theta = rng.uniform(0, TWO_PI, n)
        omega = rng.normal(1.0, 0.3, n)
        knm = rng.uniform(0, 0.3, (n, n))
        np.fill_diagonal(knm, 0.0)
        eng = HypergraphEngine(n, 0.01)
        eng.add_edge((0, 1, 2), strength=0.4)
        eng.add_edge((3, 4, 5, 6), strength=0.2)
        fin = eng.run(theta, omega, n_steps=400, pairwise_knm=knm)
        assert np.all(fin >= 0.0)
        assert np.all(fin < TWO_PI + 1e-12)

    @_python
    def test_outputs_remain_finite(self):
        rng = np.random.default_rng(1)
        n = 10
        theta = rng.uniform(0, TWO_PI, n)
        omega = rng.normal(0.0, 0.1, n)
        eng = HypergraphEngine(n, 0.01)
        eng.add_all_to_all(order=3, strength=0.1)
        fin = eng.run(theta, omega, n_steps=300)
        assert np.all(np.isfinite(fin))

    @_python
    def test_zero_coupling_recovers_pure_rotation(self):
        """No edges, no pairwise, no drive → exact linear
        rotation over 300 Euler steps (modular arithmetic)."""
        n = 4
        theta = np.array([0.1, 1.2, 2.5, 3.7])
        omega = np.array([0.2, -0.15, 0.05, -0.08])
        dt = 0.01
        n_steps = 300
        eng = HypergraphEngine(n, dt)
        fin = eng.run(theta, omega, n_steps=n_steps)
        expected = (theta + n_steps * dt * omega) % TWO_PI
        diff = np.minimum(
            np.abs(fin - expected),
            TWO_PI - np.abs(fin - expected),
        )
        assert np.max(diff) < 1e-12

    @_python
    def test_triadic_fully_synced_is_fixed_point(self):
        """The fully-synchronised state is an exact fixed point
        of the triadic dynamics for any σ: ``phase_sum − k·θ_m
        = 0`` at every node, so the derivative vanishes and the
        Euler update is the identity. Long-run invariant."""
        n = 12
        theta = np.full(n, 2.1)
        omega = np.zeros(n)
        eng = HypergraphEngine(n, 0.01)
        eng.add_all_to_all(order=3, strength=0.5)
        fin = eng.run(theta, omega, n_steps=500)
        np.testing.assert_allclose(fin, theta, atol=1e-12)
        assert eng.order_parameter(fin) == pytest.approx(1.0, abs=1e-12)


class TestMixedOrderLongRun:
    @_python
    def test_pairwise_plus_triadic_bounded_coherence(self):
        """Pairwise K + triadic hyperedges must keep coherence
        in ``[0, 1]`` over a long run."""
        rng = np.random.default_rng(5)
        n = 6
        theta = rng.uniform(0, TWO_PI, n)
        omega = rng.normal(1.0, 0.2, n)
        knm = rng.uniform(0, 0.2, (n, n))
        np.fill_diagonal(knm, 0.0)
        eng = HypergraphEngine(n, 0.01)
        eng.add_edge((0, 1, 2), strength=0.3)
        eng.add_edge((3, 4, 5), strength=0.3)
        fin = eng.run(theta, omega, n_steps=500, pairwise_knm=knm)
        r = eng.order_parameter(fin)
        assert 0.0 <= r <= 1.0 + 1e-12
