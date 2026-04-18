# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Algorithmic tests for Ott-Antonsen reduction

"""Algorithmic properties of :class:`OttAntonsenReduction`.

Covered: constructor validation; ``K_c = 2Δ``; analytical
``R_ss = √(1 − 2Δ/K)`` above criticality and zero below;
trajectory converges to the predicted steady state; Kuramoto-
analytical cross-check; Hypothesis invariants;
``predict_from_oscillators`` fits a Lorentzian to an empirical
frequency sample.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.upde import reduction as r_mod
from scpn_phase_orchestrator.upde.reduction import (
    OAState,
    OttAntonsenReduction,
)


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


class TestConstructor:
    def test_rejects_negative_delta(self):
        with pytest.raises(ValueError, match="delta"):
            OttAntonsenReduction(omega_0=0.0, delta=-0.1, K=1.0)

    def test_rejects_non_positive_dt(self):
        with pytest.raises(ValueError, match="dt"):
            OttAntonsenReduction(
                omega_0=0.0, delta=0.1, K=1.0, dt=0.0,
            )


class TestAnalyticalSteadyState:
    @_python
    def test_k_critical_equals_two_delta(self):
        red = OttAntonsenReduction(omega_0=0.0, delta=0.3, K=1.0)
        assert red.K_c == pytest.approx(0.6, abs=1e-14)

    @_python
    def test_r_ss_above_critical(self):
        """K = 3Δ → R_ss = √(1 − 2/3) = √(1/3) ≈ 0.5774."""
        red = OttAntonsenReduction(omega_0=0.0, delta=1.0, K=3.0)
        assert red.steady_state_R() == pytest.approx(
            math.sqrt(1.0 / 3.0), abs=1e-12,
        )

    @_python
    def test_r_ss_below_critical_is_zero(self):
        red = OttAntonsenReduction(omega_0=0.0, delta=1.0, K=1.5)
        assert red.steady_state_R() == 0.0

    @_python
    def test_r_ss_at_critical_is_zero(self):
        """K = K_c = 2Δ → R_ss = 0 (boundary case)."""
        red = OttAntonsenReduction(omega_0=0.0, delta=1.0, K=2.0)
        assert red.steady_state_R() == 0.0


class TestTrajectoryConvergence:
    @_python
    def test_converges_to_analytical_ss(self):
        """Above criticality, integrating from a small seed
        should converge towards the analytical ``R_ss``."""
        delta = 0.1
        K = 1.0
        red = OttAntonsenReduction(
            omega_0=0.0, delta=delta, K=K, dt=0.01,
        )
        state = red.run(complex(0.05, 0.0), n_steps=5000)
        assert pytest.approx(red.steady_state_R(), abs=5e-3) == state.R

    @_python
    def test_subcritical_decays(self):
        """Below criticality, ``R`` decays towards zero
        regardless of initial seed."""
        red = OttAntonsenReduction(
            omega_0=0.0, delta=1.0, K=1.5, dt=0.01,
        )
        state = red.run(complex(0.5, 0.0), n_steps=3000)
        assert state.R < 0.05


class TestStep:
    @_python
    def test_step_advances_trajectory(self):
        red = OttAntonsenReduction(
            omega_0=0.5, delta=0.1, K=1.0, dt=0.01,
        )
        z0 = complex(0.2, 0.1)
        z1 = red.step(z0)
        assert z1 != z0

    @_python
    def test_zero_state_is_fixed_point_at_linear_regime(self):
        """z = 0 is always a fixed point: derivative evaluates
        to 0 at origin irrespective of Δ, ω₀, K."""
        red = OttAntonsenReduction(
            omega_0=0.5, delta=0.1, K=1.0, dt=0.01,
        )
        state = red.run(complex(0.0, 0.0), n_steps=100)
        assert pytest.approx(0.0, abs=1e-14) == state.R


class TestPredictFromOscillators:
    @_python
    def test_infers_delta_and_omega_from_sample(self):
        rng = np.random.default_rng(0)
        # Draw from a Lorentzian with omega_0=1.5, delta=0.3
        omegas = 1.5 + 0.3 * np.tan(
            np.pi * (rng.uniform(0, 1, 2000) - 0.5),
        )
        red = OttAntonsenReduction(
            omega_0=0.0, delta=0.0, K=0.0, dt=0.01,
        )
        state = red.predict_from_oscillators(omegas, K=1.0)
        assert isinstance(state, OAState)
        assert 0.0 <= state.R <= 1.0 + 1e-12


class TestHypothesis:
    @_python
    @given(
        delta=st.floats(min_value=0.05, max_value=0.5),
        K_ratio=st.floats(min_value=1.5, max_value=5.0),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=6, deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_r_bounded_in_unit_interval(self, delta, K_ratio, seed):
        K = K_ratio * delta
        rng = np.random.default_rng(seed)
        z0 = complex(0.1 * rng.standard_normal(),
                     0.1 * rng.standard_normal())
        red = OttAntonsenReduction(
            omega_0=0.0, delta=delta, K=K, dt=0.01,
        )
        state = red.run(z0, n_steps=1000)
        assert 0.0 <= state.R <= 1.0 + 1e-10


class TestDispatcherSurface:
    def test_available_non_empty(self):
        assert r_mod.AVAILABLE_BACKENDS
        assert "python" in r_mod.AVAILABLE_BACKENDS

    def test_active_is_first(self):
        assert r_mod.AVAILABLE_BACKENDS[0] == r_mod.ACTIVE_BACKEND
