# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Algorithmic tests for Strang splitting

"""Algorithmic properties of :class:`SplittingEngine`.

Covered: constructor validation; phase wrap; ``K = 0`` pure-A
exact-rotation limit (Strang collapses to ``θ += ω·dt``);
zero-coupling-plus-zero-drive fixed point; ζ-forcing shifts
phases; order-parameter helper; Hypothesis invariants.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.upde import splitting as sp_mod
from scpn_phase_orchestrator.upde.splitting import SplittingEngine

TWO_PI = 2.0 * math.pi


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


def _problem(seed: int, n: int = 6):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, TWO_PI, n)
    omegas = rng.normal(1.0, 0.2, n)
    knm = rng.uniform(0, 0.3, (n, n))
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))
    return theta, omegas, knm, alpha


class TestConstructor:
    def test_rejects_zero_n(self):
        with pytest.raises(ValueError, match="n_oscillators"):
            SplittingEngine(0, 0.01)

    def test_rejects_zero_dt(self):
        with pytest.raises(ValueError, match="dt"):
            SplittingEngine(4, 0.0)

    def test_negative_dt_accepted_for_reversibility(self):
        """Negative dt is supported because Strang splitting is
        time-reversible: stepping forward at +dt then backward
        at −dt returns to the starting state (symplectic property).
        """
        SplittingEngine(4, -0.01)  # no exception


class TestStep:
    @_python
    def test_phases_wrap_in_two_pi(self):
        theta, omegas, knm, alpha = _problem(0)
        eng = SplittingEngine(6, 0.1)
        new_ph = eng.step(theta, omegas, knm, 0.0, 0.0, alpha)
        assert np.all(new_ph >= 0.0)
        assert np.all(new_ph < TWO_PI + 1e-12)

    @_python
    def test_zero_coupling_recovers_exact_rotation(self):
        """K = 0, ζ = 0 → the B-stage is the identity, so Strang
        collapses to ``θ ← (θ + dt·ω) mod 2π``. The splitting
        preserves this exactly (A is exact rotation)."""
        n = 5
        theta = np.array([0.1, 1.0, 2.0, 3.0, 4.5])
        omegas = np.array([0.2, -0.3, 0.05, 0.1, 0.0])
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))
        dt = 0.01
        eng = SplittingEngine(n, dt)
        got = eng.step(theta, omegas, knm, 0.0, 0.0, alpha)
        expected = (theta + dt * omegas) % TWO_PI
        np.testing.assert_allclose(got, expected, atol=1e-14)

    @_python
    def test_zero_coupling_zero_drive_fixed_point(self):
        """K = 0, ω = 0, ζ = 0 → derivative is zero; Strang step
        returns the input unchanged (mod 2π)."""
        n = 4
        theta = np.array([0.5, 1.7, 2.3, 5.0])
        omegas = np.zeros(n)
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))
        eng = SplittingEngine(n, 0.05)
        got = eng.step(theta, omegas, knm, 0.0, 0.0, alpha)
        np.testing.assert_allclose(got, theta, atol=1e-14)


class TestExternalDrive:
    @_python
    def test_zeta_psi_shifts_phases(self):
        """With ω = 0, K = 0, the B-stage derivative becomes
        ``ζ·sin(ψ − θ)`` in RK4. For θ = 0, ψ = π/2 and ζ = 1
        the first-step RK4 effectively integrates
        ``sin(π/2 − θ) ≈ 1`` over dt, so the final phase is
        close to dt."""
        n = 3
        theta = np.zeros(n)
        omegas = np.zeros(n)
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))
        dt = 0.01
        eng = SplittingEngine(n, dt)
        got = eng.step(theta, omegas, knm, 1.0, np.pi / 2, alpha)
        np.testing.assert_allclose(got, np.full(n, dt), atol=1e-6)


class TestRun:
    @_python
    def test_run_matches_repeated_step(self):
        theta, omegas, knm, alpha = _problem(2)
        eng = SplittingEngine(6, 0.01)
        chunk = eng.run(theta, omegas, knm, 0.0, 0.0, alpha, n_steps=5)
        # Compose via single-step
        p = theta.copy()
        for _ in range(5):
            p = eng.step(p, omegas, knm, 0.0, 0.0, alpha)
        np.testing.assert_allclose(chunk, p, atol=1e-12)


class TestOrderParameter:
    @_python
    def test_perfectly_locked(self):
        eng = SplittingEngine(10, 0.01)
        assert eng.order_parameter(np.full(10, 2.1)) == pytest.approx(
            1.0,
            abs=1e-12,
        )

    @_python
    def test_uniform_near_zero(self):
        eng = SplittingEngine(1000, 0.01)
        phases = np.linspace(0, TWO_PI, 1000, endpoint=False)
        assert eng.order_parameter(phases) < 1e-10


class TestHypothesis:
    @_python
    @given(
        n=st.integers(min_value=3, max_value=8),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=8,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_finite_output(self, n, seed):
        theta, omegas, knm, alpha = _problem(seed, n)
        eng = SplittingEngine(n, 0.01)
        fin = eng.run(theta, omegas, knm, 0.0, 0.0, alpha, n_steps=20)
        assert np.all(np.isfinite(fin))
        assert np.all(fin >= 0.0)
        assert np.all(fin < TWO_PI + 1e-12)


class TestDispatcherSurface:
    def test_available_non_empty(self):
        assert sp_mod.AVAILABLE_BACKENDS
        assert "python" in sp_mod.AVAILABLE_BACKENDS

    def test_active_is_first(self):
        assert sp_mod.AVAILABLE_BACKENDS[0] == sp_mod.ACTIVE_BACKEND
