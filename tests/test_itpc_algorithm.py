# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Algorithmic tests for ITPC

"""Algorithmic properties of :func:`compute_itpc` /
:func:`itpc_persistence`.

Covers: output shape, value bounds ``[0, 1]``, perfect-sync limit,
uniform-random limit, monotonicity under added noise, empty inputs,
and the persistence mask behaviour.
"""

from __future__ import annotations

import functools
import math

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.monitor import itpc as it_mod
from scpn_phase_orchestrator.monitor.itpc import (
    compute_itpc,
    itpc_persistence,
)

TWO_PI = 2.0 * math.pi


def _python(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        prev = it_mod.ACTIVE_BACKEND
        it_mod.ACTIVE_BACKEND = "python"
        try:
            return func(*args, **kwargs)
        finally:
            it_mod.ACTIVE_BACKEND = prev

    return wrapper


class TestShape:
    @_python
    def test_output_shape_matches_timepoints(self):
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, TWO_PI, (30, 120))
        out = compute_itpc(phases)
        assert out.shape == (120,)

    @_python
    def test_single_trial_constant(self):
        """1-D input is treated as a single trial."""
        phases = np.linspace(0, TWO_PI, 50)
        out = compute_itpc(phases)
        assert out.shape == (1,)
        assert abs(out[0] - 1.0) < 1e-12

    @_python
    def test_empty_trials_returns_empty(self):
        phases = np.zeros((0, 10))
        out = compute_itpc(phases)
        assert out.size == 0


class TestValueBounds:
    @_python
    def test_itpc_in_unit_interval(self):
        rng = np.random.default_rng(11)
        phases = rng.uniform(0, TWO_PI, (40, 200))
        out = compute_itpc(phases)
        assert np.all(out >= 0.0 - 1e-12)
        assert np.all(out <= 1.0 + 1e-12)


class TestAnalyticLimits:
    @_python
    def test_perfect_sync_gives_R_equals_one(self):
        phases = np.full((20, 50), 0.7)
        out = compute_itpc(phases)
        assert np.all(np.abs(out - 1.0) < 1e-12)

    @_python
    def test_uniform_distribution_approaches_zero(self):
        """A large number of uniformly random phases → ITPC ≈ 0
        (concentration parameter of a uniform ≡ 0)."""
        rng = np.random.default_rng(42)
        n_trials = 5000
        phases = rng.uniform(0, TWO_PI, (n_trials, 1))
        out = compute_itpc(phases)
        assert out[0] < 0.05  # within sqrt(1/N) noise floor

    @_python
    def test_opposite_phases_cancel(self):
        """Pairs of antiphase oscillators cancel: ITPC → 0."""
        phases = np.array([[0.0] * 5 + [math.pi] * 5] * 2).reshape(4, 5)
        # Construction: 2 trials at 0, 2 trials at π → mean exp = 0.
        phases = np.array([[0.0] * 5, [math.pi] * 5, [0.0] * 5, [math.pi] * 5])
        out = compute_itpc(phases)
        assert np.all(out < 1e-12)


class TestPersistence:
    @_python
    def test_empty_pause_returns_zero(self):
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, TWO_PI, (10, 20))
        assert itpc_persistence(phases, []) == 0.0

    @_python
    def test_persistence_on_perfect_sync_is_one(self):
        phases = np.full((20, 30), 0.4)
        out = itpc_persistence(phases, [5, 10, 15])
        assert abs(out - 1.0) < 1e-12

    @_python
    def test_out_of_range_indices_ignored(self):
        phases = np.full((8, 10), 0.2)
        out = itpc_persistence(phases, [-5, 3, 100])
        # Only index 3 is valid; ITPC at every point = 1 on perfect sync.
        assert abs(out - 1.0) < 1e-12

    @_python
    def test_persistence_equals_mean_on_uniform_itpc(self):
        """If ITPC is constant across time, persistence == that value."""
        phases = np.full((10, 15), 1.23)
        idx = np.array([0, 4, 9, 14])
        assert abs(itpc_persistence(phases, idx) - 1.0) < 1e-12


class TestMonotonicityUnderNoise:
    @_python
    def test_more_noise_lowers_itpc(self):
        rng = np.random.default_rng(7)
        base = np.full((200, 20), 0.5)
        low = base + rng.normal(0.0, 0.05, base.shape)
        high = base + rng.normal(0.0, 0.8, base.shape)
        itpc_low = float(np.mean(compute_itpc(low)))
        itpc_high = float(np.mean(compute_itpc(high)))
        assert itpc_low > itpc_high


class TestHypothesisProperty:
    @_python
    @given(
        n_trials=st.integers(min_value=2, max_value=50),
        n_tp=st.integers(min_value=1, max_value=80),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_random_input_bounded(self, n_trials: int, n_tp: int, seed: int):
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, (n_trials, n_tp))
        out = compute_itpc(phases)
        assert out.shape == (n_tp,)
        assert np.all(np.isfinite(out))
        assert np.all(out >= -1e-12)
        assert np.all(out <= 1.0 + 1e-12)


class TestDispatcherSurface:
    def test_available_backends_non_empty(self):
        assert len(it_mod.AVAILABLE_BACKENDS) >= 1

    def test_python_always_available(self):
        assert "python" in it_mod.AVAILABLE_BACKENDS

    def test_active_backend_is_first_available(self):
        assert it_mod.AVAILABLE_BACKENDS[0] == it_mod.ACTIVE_BACKEND


class TestInputValidation:
    @_python
    def test_zero_trials_empty_output(self):
        assert compute_itpc(np.zeros((0, 50))).size == 0

    @_python
    def test_persistence_with_no_trials_handles_gracefully(self):
        # Empty pause indices short-circuits the backend call.
        assert itpc_persistence(np.zeros((0, 10)), []) == 0.0
