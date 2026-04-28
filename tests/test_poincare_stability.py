# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Stability tests for Poincaré sections

"""Long-run invariants for Poincaré-section kernels.

* Large-T stress for ``poincare_section``.
* Periodic trajectory has constant return time.
* Chaotic / noisy trajectory has non-zero return-time variance.

Marked ``@pytest.mark.slow``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor import poincare as p_mod
from scpn_phase_orchestrator.monitor.poincare import (
    phase_poincare,
    poincare_section,
)

pytestmark = pytest.mark.slow


def _python(func):
    def wrapper(*args, **kwargs):
        prev = p_mod.ACTIVE_BACKEND
        p_mod.ACTIVE_BACKEND = "python"
        try:
            return func(*args, **kwargs)
        finally:
            p_mod.ACTIVE_BACKEND = prev

    wrapper.__name__ = func.__name__
    return wrapper


@_python
def test_large_T_stress():
    rng = np.random.default_rng(0)
    t = 5000
    traj = rng.normal(0, 1, (t, 3))
    normal = np.array([1.0, 0.0, 0.0])
    res = poincare_section(traj, normal)
    assert res.crossings.shape[1] == 3
    # Random walk crosses the plane repeatedly.
    assert len(res.crossings) > 50
    assert np.all(np.isfinite(res.crossing_times))


@_python
def test_periodic_return_time_constant():
    """Constant-frequency oscillator → std(return_times) ≈ 0."""
    T, N = 1000, 2
    omega = 0.05
    phases = np.zeros((T, N))
    for i in range(1, T):
        phases[i] = phases[i - 1] + omega
    res = phase_poincare(phases, 0, 0.0)
    if len(res.return_times) > 1:
        assert res.std_return_time < 1.0


@_python
def test_chaotic_return_time_variance():
    """A random-walk phase has non-constant return intervals."""
    rng = np.random.default_rng(7)
    T, N = 2000, 2
    phases = np.zeros((T, N))
    phases[0, 0] = 0.0
    for i in range(1, T):
        phases[i, 0] = phases[i - 1, 0] + rng.normal(0.1, 0.05)
        phases[i, 1] = phases[i - 1, 1]
    res = phase_poincare(phases, 0, 0.0)
    if len(res.return_times) > 2:
        # Std should be bounded away from zero for a noisy walk.
        assert res.std_return_time > 0.1


@_python
def test_sinusoid_section_across_periods():
    """Three periods of ``sin(t)`` on the normal ``[1, 0]`` yield 2–3
    positive-slope crossings depending on whether ``sin(6π)``'s IEEE
    rounding value lands ``>= 0`` (detected) or ``< 0`` (missed)."""
    t = np.linspace(0, 6 * math.pi, 3000)
    traj = np.column_stack([np.sin(t), np.cos(t)])
    normal = np.array([1.0, 0.0])
    res = poincare_section(traj, normal, direction="positive")
    assert 2 <= len(res.crossings) <= 3
