# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Algorithmic tests for swarmalator stepper

"""Algorithmic properties of :class:`SwarmalatorEngine`.

Covered: constructor validation; step output shape; phase wrap
within ``[0, 2π)``; zero-coupling limit reduces to pure-ω rotation;
empty-/single-agent edge cases; run trajectory shapes;
order-parameter helper; Hypothesis invariants.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.upde import swarmalator as sw_mod
from scpn_phase_orchestrator.upde.swarmalator import SwarmalatorEngine

TWO_PI = 2.0 * math.pi


def _python(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        prev = sw_mod.ACTIVE_BACKEND
        sw_mod.ACTIVE_BACKEND = "python"
        try:
            return func(*args, **kwargs)
        finally:
            sw_mod.ACTIVE_BACKEND = prev

    return wrapper


def _problem(seed: int, n: int = 16, dim: int = 2):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1, 1, (n, dim))
    phases = rng.uniform(0, TWO_PI, n)
    omegas = rng.normal(0.5, 0.2, n)
    return pos, phases, omegas


class TestConstructor:
    def test_rejects_zero_agents(self):
        with pytest.raises(ValueError, match="n_agents"):
            SwarmalatorEngine(n_agents=0, dim=2, dt=0.01)

    def test_rejects_zero_dim(self):
        with pytest.raises(ValueError, match="dim"):
            SwarmalatorEngine(n_agents=4, dim=0, dt=0.01)

    def test_rejects_non_positive_dt(self):
        with pytest.raises(ValueError, match="dt"):
            SwarmalatorEngine(n_agents=4, dim=2, dt=0.0)


class TestStep:
    @_python
    def test_output_shape(self):
        pos, phases, omegas = _problem(0)
        eng = SwarmalatorEngine(16, 2, 0.01)
        p, ph = eng.step(pos, phases, omegas, 1.0, 1.0, 0.8, 1.2)
        assert p.shape == (16, 2)
        assert ph.shape == (16,)

    @_python
    def test_phases_wrap_in_two_pi(self):
        pos, phases, omegas = _problem(1)
        eng = SwarmalatorEngine(16, 2, 0.1)
        _, ph = eng.step(pos, phases, omegas, 1.0, 1.0, 1.0, 1.0)
        assert np.all(ph >= 0.0)
        assert np.all(ph < TWO_PI + 1e-12)

    @_python
    def test_zero_couplings_pure_rotation(self):
        """``a = b = j = k = 0`` → position frozen, phases evolve
        purely by ω."""
        pos, phases, omegas = _problem(2)
        eng = SwarmalatorEngine(16, 2, 0.01)
        new_pos, new_ph = eng.step(pos, phases, omegas, 0.0, 0.0, 0.0, 0.0)
        np.testing.assert_allclose(new_pos, pos, atol=1e-12)
        expected = (phases + 0.01 * omegas) % TWO_PI
        np.testing.assert_allclose(new_ph, expected, atol=1e-12)


class TestRun:
    @_python
    def test_trajectory_shapes(self):
        pos, phases, omegas = _problem(3)
        eng = SwarmalatorEngine(16, 2, 0.01)
        final_pos, final_ph, pos_traj, phase_traj = eng.run(
            pos, phases, omegas, n_steps=5,
        )
        assert final_pos.shape == (16, 2)
        assert final_ph.shape == (16,)
        assert pos_traj.shape == (5, 16, 2)
        assert phase_traj.shape == (5, 16)


class TestOrderParameter:
    @_python
    def test_perfectly_locked(self):
        phases = np.full(10, 0.5)
        eng = SwarmalatorEngine(10, 2, 0.01)
        assert eng.order_parameter(phases) == pytest.approx(1.0, abs=1e-12)

    @_python
    def test_uniform_approaches_zero(self):
        phases = np.linspace(0, TWO_PI, 1000, endpoint=False)
        eng = SwarmalatorEngine(1000, 2, 0.01)
        r = eng.order_parameter(phases)
        assert r < 1e-10


class TestHypothesis:
    @_python
    @given(
        n=st.integers(min_value=2, max_value=16),
        dim=st.integers(min_value=1, max_value=3),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_finite_output(self, n: int, dim: int, seed: int):
        pos, phases, omegas = _problem(seed, n, dim)
        eng = SwarmalatorEngine(n, dim, 0.01)
        p, ph = eng.step(pos, phases, omegas)
        assert np.all(np.isfinite(p))
        assert np.all(np.isfinite(ph))


class TestDispatcherSurface:
    def test_available_non_empty(self):
        assert sw_mod.AVAILABLE_BACKENDS
        assert "python" in sw_mod.AVAILABLE_BACKENDS

    def test_active_is_first(self):
        assert sw_mod.AVAILABLE_BACKENDS[0] == sw_mod.ACTIVE_BACKEND
