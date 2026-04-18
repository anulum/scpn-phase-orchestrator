# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Algorithmic tests for basin stability

"""Algorithmic properties of ``upde.basin_stability``.

Covered: physics limits (high K → high S_B, zero K → low S_B,
lower R_threshold yields higher S_B); bounds 0 ≤ S_B ≤ 1 and
0 ≤ R_final ≤ 1; threshold classification consistency;
``multi_basin_stability`` keys and monotonicity; deterministic
result for identical seed; order-parameter helper via
``steady_state_r`` with locked initial phases;
Hypothesis-driven invariants.
"""

from __future__ import annotations

import functools
import math

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.upde import basin_stability as b_mod
from scpn_phase_orchestrator.upde.basin_stability import (
    BasinStabilityResult,
    basin_stability,
    multi_basin_stability,
    steady_state_r,
)

TWO_PI = 2.0 * math.pi


def _python(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        prev = b_mod.ACTIVE_BACKEND
        b_mod.ACTIVE_BACKEND = "python"
        try:
            return func(*args, **kwargs)
        finally:
            b_mod.ACTIVE_BACKEND = prev

    return wrapper


def _all_to_all(n: int, strength: float = 1.0) -> np.ndarray:
    k = np.ones((n, n)) * strength / n
    np.fill_diagonal(k, 0.0)
    return k


class TestSteadyStateRKernel:
    @_python
    def test_locked_initial_gives_r_near_one_at_strong_coupling(self):
        n = 8
        omegas = np.ones(n) * 0.5
        knm = _all_to_all(n, strength=10.0)
        phases = np.full(n, 1.3)
        r = steady_state_r(
            phases, omegas, knm, dt=0.01,
            n_transient=200, n_measure=100,
        )
        assert r > 0.99

    @_python
    def test_zero_coupling_desynchronises(self):
        n = 6
        omegas = np.array([1.0, 1.7, 2.3, 3.1, 4.2, 5.5])
        knm = np.zeros((n, n))
        phases = np.linspace(0.0, TWO_PI, n, endpoint=False)
        r = steady_state_r(
            phases, omegas, knm, dt=0.01,
            n_transient=500, n_measure=500,
        )
        assert r < 0.9

    @_python
    def test_zero_measure_returns_zero(self):
        n = 3
        omegas = np.ones(n)
        knm = _all_to_all(n)
        phases = np.zeros(n)
        r = steady_state_r(
            phases, omegas, knm, dt=0.01,
            n_transient=10, n_measure=0,
        )
        assert r == 0.0

    @_python
    def test_r_in_bounds(self):
        n = 5
        omegas = np.linspace(0.5, 2.0, n)
        knm = _all_to_all(n, strength=2.0)
        rng = np.random.default_rng(1)
        phases = rng.uniform(0, TWO_PI, n)
        r = steady_state_r(
            phases, omegas, knm, dt=0.01,
            n_transient=200, n_measure=100,
        )
        assert 0.0 <= r <= 1.0 + 1e-12


class TestBasinStability:
    @_python
    def test_strong_coupling_high_sb(self):
        n = 5
        omegas = np.ones(n)
        knm = _all_to_all(n, strength=10.0)
        result = basin_stability(
            omegas, knm,
            dt=0.01, n_transient=300, n_measure=100,
            n_samples=20, R_threshold=0.8, seed=42,
        )
        assert isinstance(result, BasinStabilityResult)
        assert result.S_B > 0.5
        assert result.n_samples == 20
        assert 0 <= result.n_converged <= 20

    @_python
    def test_zero_coupling_low_sb(self):
        n = 6
        omegas = np.linspace(0.5, 3.5, n)
        knm = np.zeros((n, n))
        result = basin_stability(
            omegas, knm,
            dt=0.01, n_transient=300, n_measure=100,
            n_samples=15, R_threshold=0.9, seed=7,
        )
        assert result.S_B <= 0.2

    @_python
    def test_bounds(self):
        n = 4
        omegas = np.ones(n)
        knm = _all_to_all(n, strength=2.0)
        result = basin_stability(
            omegas, knm,
            dt=0.01, n_transient=200, n_measure=100,
            n_samples=10, R_threshold=0.5, seed=3,
        )
        assert 0.0 <= result.S_B <= 1.0
        assert result.R_final.shape == (10,)
        assert np.all(result.R_final >= 0.0)
        assert np.all(result.R_final <= 1.0 + 1e-12)

    @_python
    def test_deterministic_same_seed(self):
        n = 4
        omegas = np.array([1.0, 1.5, 2.0, 2.5])
        knm = _all_to_all(n, strength=3.0)
        r1 = basin_stability(
            omegas, knm, dt=0.01, n_transient=150, n_measure=80,
            n_samples=10, R_threshold=0.5, seed=123,
        )
        r2 = basin_stability(
            omegas, knm, dt=0.01, n_transient=150, n_measure=80,
            n_samples=10, R_threshold=0.5, seed=123,
        )
        assert r1.S_B == r2.S_B
        assert r1.n_converged == r2.n_converged
        np.testing.assert_array_equal(r1.R_final, r2.R_final)

    @_python
    def test_lower_threshold_higher_sb(self):
        n = 4
        omegas = np.array([1.0, 2.0, 3.0, 4.0])
        knm = _all_to_all(n, strength=3.0)
        r_low = basin_stability(
            omegas, knm, dt=0.01, n_transient=200, n_measure=100,
            n_samples=15, R_threshold=0.3, seed=42,
        )
        r_high = basin_stability(
            omegas, knm, dt=0.01, n_transient=200, n_measure=100,
            n_samples=15, R_threshold=0.9, seed=42,
        )
        assert r_low.S_B >= r_high.S_B


class TestMultiBasinStability:
    @_python
    def test_keys_and_monotonicity(self):
        n = 4
        omegas = np.ones(n)
        knm = _all_to_all(n, strength=4.0)
        results = multi_basin_stability(
            omegas, knm, dt=0.01, n_transient=150, n_measure=80,
            n_samples=12, R_thresholds=(0.3, 0.6, 0.9), seed=5,
        )
        assert set(results.keys()) == {"R>=0.30", "R>=0.60", "R>=0.90"}
        sb_vals = [results[k].S_B for k in
                   ("R>=0.30", "R>=0.60", "R>=0.90")]
        assert sb_vals[0] >= sb_vals[1] >= sb_vals[2]

    @_python
    def test_shared_r_finals(self):
        """All threshold entries must share the same R_final array."""
        n = 4
        omegas = np.ones(n)
        knm = _all_to_all(n, strength=3.0)
        results = multi_basin_stability(
            omegas, knm, dt=0.01, n_transient=120, n_measure=60,
            n_samples=8, R_thresholds=(0.3, 0.8), seed=11,
        )
        r1 = results["R>=0.30"].R_final
        r2 = results["R>=0.80"].R_final
        np.testing.assert_array_equal(r1, r2)


class TestHypothesis:
    @_python
    @given(
        n=st.integers(min_value=2, max_value=6),
        strength=st.floats(min_value=0.5, max_value=5.0),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=8, deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_sb_in_unit_interval(self, n, strength, seed):
        omegas = np.ones(n)
        knm = _all_to_all(n, strength=strength)
        result = basin_stability(
            omegas, knm, dt=0.01, n_transient=80, n_measure=40,
            n_samples=6, R_threshold=0.5, seed=seed,
        )
        assert 0.0 <= result.S_B <= 1.0
        assert np.all(np.isfinite(result.R_final))


class TestDispatcherSurface:
    def test_available_non_empty(self):
        assert b_mod.AVAILABLE_BACKENDS
        assert "python" in b_mod.AVAILABLE_BACKENDS

    def test_active_is_first(self):
        assert b_mod.AVAILABLE_BACKENDS[0] == b_mod.ACTIVE_BACKEND


class TestInputShapes:
    @_python
    def test_default_alpha_zeros(self):
        n = 3
        omegas = np.ones(n)
        knm = _all_to_all(n)
        r_no_alpha = basin_stability(
            omegas, knm, dt=0.01, n_transient=60, n_measure=30,
            n_samples=4, R_threshold=0.5, seed=1,
        )
        r_zero_alpha = basin_stability(
            omegas, knm, alpha=np.zeros((n, n)),
            dt=0.01, n_transient=60, n_measure=30,
            n_samples=4, R_threshold=0.5, seed=1,
        )
        np.testing.assert_allclose(
            r_no_alpha.R_final, r_zero_alpha.R_final, atol=1e-15,
        )

    @_python
    def test_zero_samples_returns_empty(self):
        n = 3
        omegas = np.ones(n)
        knm = _all_to_all(n)
        result = basin_stability(
            omegas, knm, dt=0.01, n_transient=20, n_measure=10,
            n_samples=0, R_threshold=0.5, seed=1,
        )
        assert result.S_B == 0.0
        assert result.n_samples == 0
        assert result.n_converged == 0
        assert result.R_final.shape == (0,)
