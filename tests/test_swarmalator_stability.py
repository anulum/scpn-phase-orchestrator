# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Stability invariants for swarmalator

"""Long-run stability invariants for ``SwarmalatorEngine``.

Marked ``slow`` — integrates for hundreds of steps. Verifies that
the stepper keeps (a) phases inside ``[0, 2π)``; (b) positions
bounded; (c) phase order parameter in ``[0, 1]``; (d) the
``j = 0, k = 0`` limit conserves purely ω-driven rotation over a
long trajectory.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest

from scpn_phase_orchestrator.upde import swarmalator as sw_mod
from scpn_phase_orchestrator.upde.swarmalator import SwarmalatorEngine

TWO_PI = 2.0 * math.pi

pytestmark = pytest.mark.slow


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


def _problem(seed: int, n: int, dim: int):
    rng = np.random.default_rng(seed)
    return (
        rng.uniform(-1, 1, (n, dim)),
        rng.uniform(0, TWO_PI, n),
        rng.normal(0.5, 0.2, n),
    )


class TestLongRunInvariants:
    @_python
    def test_phases_stay_wrapped_over_500_steps(self):
        pos, ph, om = _problem(0, 12, 2)
        eng = SwarmalatorEngine(12, 2, 0.01)
        _, _, _, phase_traj = eng.run(pos, ph, om, n_steps=500)
        assert np.all(phase_traj >= 0.0)
        assert np.all(phase_traj < TWO_PI + 1e-12)

    @_python
    def test_positions_remain_finite(self):
        pos, ph, om = _problem(1, 10, 2)
        eng = SwarmalatorEngine(10, 2, 0.01)
        _, _, pos_traj, _ = eng.run(pos, ph, om, n_steps=400)
        assert np.all(np.isfinite(pos_traj))

    @_python
    def test_order_parameter_in_unit_interval(self):
        pos, ph, om = _problem(2, 16, 2)
        eng = SwarmalatorEngine(16, 2, 0.01)
        _, _, _, phase_traj = eng.run(pos, ph, om, n_steps=300)
        rs = np.array([eng.order_parameter(phase_traj[i]) for i in range(0, 300, 10)])
        assert np.all(rs >= 0.0)
        assert np.all(rs <= 1.0 + 1e-12)

    @_python
    def test_zero_coupling_recovers_rotation_over_long_run(self):
        """With ``a = b = j = k = 0`` the positions must be frozen
        exactly and phases must evolve as a pure linear rotation
        ``θ(t) = θ₀ + ω · t`` modulo ``2π`` — even after hundreds
        of Euler steps the drift from this analytical answer should
        stay within ``O(n_steps · eps)``."""
        pos, ph, om = _problem(3, 8, 2)
        dt = 0.01
        n_steps = 500
        eng = SwarmalatorEngine(8, 2, dt)
        fin_pos, fin_ph, _, _ = eng.run(
            pos,
            ph,
            om,
            a=0.0,
            b=0.0,
            j=0.0,
            k=0.0,
            n_steps=n_steps,
        )
        np.testing.assert_allclose(fin_pos, pos, atol=1e-12)
        expected = (ph + n_steps * dt * om) % TWO_PI
        # Wrap-aware comparison: angular distance on the circle.
        diff = np.minimum(
            np.abs(fin_ph - expected),
            TWO_PI - np.abs(fin_ph - expected),
        )
        assert np.max(diff) < 1e-10
