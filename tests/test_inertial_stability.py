# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Stability invariants for inertial Kuramoto

"""Long-run stability invariants for
:class:`InertialKuramotoEngine`.

Marked ``slow`` — integrates hundreds of RK4 steps. Covered:
(a) phases stay wrapped in ``[0, 2π)``; (b) frequency deviations
remain finite and bounded; (c) with ``P = 0`` and finite damping
the ``ω`` decays towards zero (dissipative attractor); (d) with
zero coupling, zero power and zero damping the swing equation
integrates a constant-velocity rotation exactly (within ``O(n·eps)``).
"""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest

from scpn_phase_orchestrator.upde import inertial as i_mod
from scpn_phase_orchestrator.upde.inertial import InertialKuramotoEngine

TWO_PI = 2.0 * math.pi

pytestmark = pytest.mark.slow


def _python(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        prev = i_mod.ACTIVE_BACKEND
        i_mod.ACTIVE_BACKEND = "python"
        try:
            return func(*args, **kwargs)
        finally:
            i_mod.ACTIVE_BACKEND = prev

    return wrapper


class TestLongRunInvariants:
    @_python
    def test_phases_stay_wrapped_over_400_steps(self):
        rng = np.random.default_rng(0)
        n = 8
        theta = rng.uniform(0, TWO_PI, n)
        omega = rng.normal(0, 0.1, n)
        power = rng.normal(0, 0.3, n)
        knm = rng.uniform(0, 0.4, (n, n))
        np.fill_diagonal(knm, 0.0)
        inertia = np.ones(n)
        damping = np.ones(n) * 0.1
        eng = InertialKuramotoEngine(n, 0.01)
        _, _, theta_traj, _ = eng.run(
            theta,
            omega,
            power,
            knm,
            inertia,
            damping,
            n_steps=400,
        )
        assert np.all(theta_traj >= 0.0)
        assert np.all(theta_traj < TWO_PI + 1e-12)

    @_python
    def test_omega_remains_finite(self):
        rng = np.random.default_rng(1)
        n = 6
        theta = rng.uniform(0, TWO_PI, n)
        omega = rng.normal(0, 0.2, n)
        power = np.zeros(n)
        knm = rng.uniform(0, 0.3, (n, n))
        np.fill_diagonal(knm, 0.0)
        inertia = np.ones(n)
        damping = np.ones(n) * 0.5
        eng = InertialKuramotoEngine(n, 0.01)
        _, _, _, omega_traj = eng.run(
            theta,
            omega,
            power,
            knm,
            inertia,
            damping,
            n_steps=300,
        )
        assert np.all(np.isfinite(omega_traj))

    @_python
    def test_dissipative_attractor_pulls_omega_to_zero(self):
        """With ``P = 0``, ``K = 0`` and ``D/M = 3.0`` the analytic
        solution is ``ω(t) = ω₀ · exp(−3 t)``. After 300 steps at
        dt=0.01 the envelope is ``exp(−9) ≈ 1.2e-4``. RK4 tracks
        this with large margin; we assert a much looser bound."""
        n = 4
        theta = np.zeros(n)
        omega = np.ones(n) * 2.0
        power = np.zeros(n)
        knm = np.zeros((n, n))
        inertia = np.ones(n)
        damping = np.ones(n) * 3.0
        eng = InertialKuramotoEngine(n, 0.01)
        _, final_omega, _, _ = eng.run(
            theta,
            omega,
            power,
            knm,
            inertia,
            damping,
            n_steps=300,
        )
        assert np.all(np.abs(final_omega) < 1e-2)

    @_python
    def test_undamped_unforced_conserves_angular_velocity(self):
        """``K = 0``, ``D = 0``, ``P = 0`` → ``ω'' = 0`` →
        ``θ(t) = θ₀ + ω₀·t``. RK4 integrates linear ODEs exactly
        (modulo floating-point), so after 200 steps the drift
        from ``ω₀`` must stay tiny."""
        n = 3
        theta = np.array([0.1, 1.2, 2.5])
        omega = np.array([0.2, -0.15, 0.05])
        power = np.zeros(n)
        knm = np.zeros((n, n))
        inertia = np.ones(n)
        damping = np.zeros(n)
        eng = InertialKuramotoEngine(n, 0.01)
        _, final_omega, _, _ = eng.run(
            theta,
            omega,
            power,
            knm,
            inertia,
            damping,
            n_steps=200,
        )
        np.testing.assert_allclose(final_omega, omega, atol=1e-12)


class TestHelpersLongRun:
    @_python
    def test_coherence_monotone_in_sync_regime(self):
        """High coupling should drive an initially random ensemble
        towards a synchronous state; the final coherence must
        be noticeably higher than the initial value."""
        rng = np.random.default_rng(5)
        n = 12
        theta = rng.uniform(0, TWO_PI, n)
        omega = np.zeros(n)
        power = np.zeros(n)
        knm = np.full((n, n), 3.0 / n)
        np.fill_diagonal(knm, 0.0)
        inertia = np.ones(n)
        damping = np.ones(n) * 0.3
        eng = InertialKuramotoEngine(n, 0.01)
        r0 = eng.coherence(theta)
        final_theta, _, _, _ = eng.run(
            theta,
            omega,
            power,
            knm,
            inertia,
            damping,
            n_steps=500,
        )
        r_final = eng.coherence(final_theta)
        assert r_final > r0
