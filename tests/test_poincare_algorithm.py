# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Algorithmic tests for Poincaré sections

"""Algorithmic properties of :func:`poincare_section`,
:func:`phase_poincare`, :func:`return_times`.

Covered: hyperplane crossings detected at correct times, direction
filtering, interpolation accuracy on a sinusoid, phase-oscillator
2π wraparound detection, constant-frequency return-time = 2π/ω,
empty-trajectory safety, invalid direction rejection.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.monitor import poincare as p_mod
from scpn_phase_orchestrator.monitor.poincare import (
    PoincareResult,
    phase_poincare,
    poincare_section,
    return_times,
)


def _python(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        prev = p_mod.ACTIVE_BACKEND
        p_mod.ACTIVE_BACKEND = "python"
        try:
            return func(*args, **kwargs)
        finally:
            p_mod.ACTIVE_BACKEND = prev

    return wrapper


class TestPoincareSection:
    @_python
    def test_sinusoid_crossings_positive_slope(self):
        """``y = sin(t)`` has positive-slope zero crossings at
        ``t = 2π, 4π, …``. Over ``[0, 6π]`` discretised into 600
        points we detect 2 such crossings — the one at ``t = 6π``
        falls exactly on the last sample and ``sin(6π)`` is a
        negative IEEE-754 value (≈ ``−2.4e-16``), so the
        ``d0 < 0, d1 >= 0`` test does not fire there."""
        t = np.linspace(0, 6 * math.pi, 600)
        traj = np.column_stack([t, np.sin(t)])
        normal = np.array([0.0, 1.0])
        res = poincare_section(traj, normal, direction="positive")
        assert len(res.crossings) == 2
        # Crossings at sample index ≈ 200 and 400.
        assert res.crossing_times[1] - res.crossing_times[0] == pytest.approx(
            200, abs=2.0
        )

    @_python
    def test_direction_both_counts_all(self):
        """``direction='both'`` counts positive- and negative-going."""
        t = np.linspace(0, 4 * math.pi, 400)
        traj = np.column_stack([t, np.sin(t)])
        normal = np.array([0.0, 1.0])
        res = poincare_section(traj, normal, direction="both")
        assert len(res.crossings) >= 3

    @_python
    def test_no_crossing_returns_empty(self):
        traj = np.zeros((50, 2))
        normal = np.array([0.0, 1.0])
        res = poincare_section(traj, normal, offset=10.0)
        assert len(res.crossings) == 0
        assert res.mean_return_time == 0.0
        assert res.std_return_time == 0.0

    @_python
    def test_invalid_direction_raises(self):
        traj = np.zeros((10, 2))
        normal = np.array([1.0, 0.0])
        with pytest.raises(ValueError, match="direction"):
            poincare_section(traj, normal, direction="sideways")

    @_python
    def test_return_times_shortcut(self):
        t = np.linspace(0, 8 * math.pi, 800)
        traj = np.column_stack([t, np.sin(t)])
        normal = np.array([0.0, 1.0])
        rt = return_times(traj, normal)
        # 4 periods over [0, 8π] → 3 positive-slope crossings → 2
        # return times (the t=0 and t=8π endpoints fall through the
        # d0<0, d1>=0 test because ``sin(0)=0`` and ``sin(8π)``
        # returns a negative IEEE-754 rounding value).
        assert len(rt) == 2
        for v in rt:
            assert v == pytest.approx(200, abs=2.0)


class TestPhasePoincare:
    @_python
    def test_constant_frequency_return_time(self):
        """A constant-frequency oscillator has return time
        ``2π / (ω · dt⁻¹) ≈ 62.83`` samples. The last return time
        is slightly smaller (~62) because the very last crossing
        falls one sample short of a full period before the trace
        ends, so the interpolated return time gets truncated by
        one-sample-worth; tolerance here reflects that."""
        T, N = 400, 3
        omega, dt = 0.1, 1.0
        phases = np.zeros((T, N))
        for i in range(1, T):
            phases[i] = phases[i - 1] + omega * dt
        res = phase_poincare(phases, oscillator_idx=0, section_phase=0.0)
        assert len(res.crossings) >= 1
        expected_period = 2 * math.pi / omega  # ≈ 62.83 samples
        if len(res.return_times) > 0:
            np.testing.assert_allclose(
                res.return_times,
                expected_period,
                atol=1.5,
            )

    @_python
    def test_no_rotation_no_crossings(self):
        phases = np.full((50, 4), 0.5)
        res = phase_poincare(phases, oscillator_idx=0, section_phase=0.0)
        assert len(res.crossings) == 0


class TestHypothesis:
    @_python
    @given(
        t=st.integers(min_value=20, max_value=300),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_section_output_shapes(self, t: int, seed: int):
        rng = np.random.default_rng(seed)
        traj = rng.normal(0, 1, (t, 3))
        normal = rng.normal(0, 1, 3)
        res = poincare_section(traj, normal)
        assert isinstance(res, PoincareResult)
        n_cr = len(res.crossings)
        assert res.crossings.shape == (n_cr, 3)
        assert res.crossing_times.shape == (n_cr,)
        assert res.return_times.shape == (max(0, n_cr - 1),)


class TestDispatcherSurface:
    def test_available_non_empty(self):
        assert p_mod.AVAILABLE_BACKENDS
        assert "python" in p_mod.AVAILABLE_BACKENDS

    def test_active_is_first(self):
        assert p_mod.AVAILABLE_BACKENDS[0] == p_mod.ACTIVE_BACKEND
