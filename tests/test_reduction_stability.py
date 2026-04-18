# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Stability invariants for OA reduction

"""Long-run stability invariants for :class:`OttAntonsenReduction`.

The OA manifold is **compact**: ``|z| ≤ 1`` is invariant. We test
(a) long-run ``R`` bounds, (b) incoherent fixed point is attracting
below criticality, (c) the analytical steady state is approached
from multiple initial seeds above criticality.
"""

from __future__ import annotations

import functools
import math

import pytest

from scpn_phase_orchestrator.upde import reduction as r_mod
from scpn_phase_orchestrator.upde.reduction import OttAntonsenReduction

pytestmark = pytest.mark.slow


def _python(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        prev = r_mod.ACTIVE_BACKEND
        r_mod.ACTIVE_BACKEND = "python"
        try:
            return func(*args, **kwargs)
        finally:
            r_mod.ACTIVE_BACKEND = prev

    return wrapper


class TestOAManifoldInvariant:
    @_python
    def test_R_remains_in_unit_disk_long_run(self):
        """|z| ≤ 1 is invariant for OA. Integrate from a
        generic seed for 20 000 steps and check ``R`` never
        leaves the unit disc."""
        red = OttAntonsenReduction(
            omega_0=0.5, delta=0.2, K=1.0, dt=0.01,
        )
        r_max = 0.0
        z = complex(0.3, 0.2)
        for _ in range(100):
            state = red.run(z, n_steps=200)
            z = state.z
            r_max = max(r_max, state.R)
        assert r_max <= 1.0 + 1e-10


class TestSubcriticalAttraction:
    @_python
    def test_multiple_seeds_decay_below_critical(self):
        """Below K_c all seeds must decay towards z = 0."""
        red = OttAntonsenReduction(
            omega_0=0.0, delta=1.0, K=1.5, dt=0.01,
        )
        for seed in (complex(0.5, 0.0), complex(0.0, 0.5),
                     complex(0.3, 0.3), complex(0.6, -0.2)):
            state = red.run(seed, n_steps=5000)
            assert state.R < 0.05


class TestSupercriticalConvergence:
    @_python
    def test_multiple_seeds_approach_analytical_ss(self):
        """Above K_c the trajectory converges to the
        analytical R_ss independent of (real, non-degenerate)
        initial seed."""
        delta = 0.1
        K = 1.0
        red = OttAntonsenReduction(
            omega_0=0.0, delta=delta, K=K, dt=0.01,
        )
        R_ss = red.steady_state_R()
        for seed in (complex(0.05, 0.0), complex(0.0, 0.05),
                     complex(0.2, 0.1), complex(-0.1, 0.1)):
            state = red.run(seed, n_steps=8000)
            assert pytest.approx(R_ss, abs=5e-3) == state.R


class TestRelaxationMonotonicity:
    @_python
    def test_radius_monotone_below_critical(self):
        """For K < K_c the radius |z(t)| decays monotonically
        for a real-axis seed with ω_0 = 0 (trajectory is
        confined to the real axis; dR/dt = −Δ·R below K_c)."""
        red = OttAntonsenReduction(
            omega_0=0.0, delta=1.0, K=1.5, dt=0.01,
        )
        r_prev = math.inf
        z = complex(0.4, 0.0)
        for _ in range(30):
            state = red.run(z, n_steps=100)
            assert r_prev + 1e-12 > state.R
            r_prev = state.R
            z = state.z
