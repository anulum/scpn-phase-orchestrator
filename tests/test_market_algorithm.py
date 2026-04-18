# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Algorithmic tests for market PLV / R(t)

"""Algorithmic properties of ``upde.market``.

Covered: locked ensemble → ``R ≈ 1``; uniform ensemble → ``R ≈ 0``;
PLV matrix diagonal is 1 (identity coupling); PLV is bounded in
``[0, 1]``; ``detect_regimes`` classifier + ``sync_warning``
crossing detector; ``extract_phase`` Hilbert transform shape
invariant; Hypothesis property that output shapes match the spec.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.upde import market as m_mod
from scpn_phase_orchestrator.upde.market import (
    detect_regimes,
    extract_phase,
    market_order_parameter,
    market_plv,
    sync_warning,
)


def _python(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        prev = m_mod.ACTIVE_BACKEND
        m_mod.ACTIVE_BACKEND = "python"
        try:
            return func(*args, **kwargs)
        finally:
            m_mod.ACTIVE_BACKEND = prev

    return wrapper


class TestOrderParameter:
    @_python
    def test_locked_ensemble_gives_R_one(self):
        T, N = 10, 5
        phases = np.full((T, N), 1.3)
        R = market_order_parameter(phases)
        assert R.shape == (T,)
        np.testing.assert_allclose(R, 1.0, atol=1e-12)

    @_python
    def test_uniform_ensemble_gives_R_zero(self):
        T, N = 5, 1000
        phases = np.tile(
            np.linspace(0, 2 * np.pi, N, endpoint=False), (T, 1),
        )
        R = market_order_parameter(phases)
        assert R.shape == (T,)
        assert np.all(R < 1e-10)

    @_python
    def test_R_bounded_in_unit_interval(self):
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, (20, 8))
        R = market_order_parameter(phases)
        assert np.all(R >= 0.0)
        assert np.all(R <= 1.0 + 1e-12)

    @_python
    def test_rejects_non_2d(self):
        with pytest.raises(ValueError, match="phases must be"):
            market_order_parameter(np.array([1.0, 2.0, 3.0]))


class TestPLV:
    @_python
    def test_diagonal_is_one(self):
        """PLV_ii = |<exp(i·(θ_i − θ_i))>| = 1."""
        rng = np.random.default_rng(1)
        T, N, W = 30, 4, 10
        phases = rng.uniform(0, 2 * np.pi, (T, N))
        plv = market_plv(phases, window=W)
        assert plv.shape == (T - W + 1, N, N)
        for w in range(T - W + 1):
            np.testing.assert_allclose(
                np.diag(plv[w]), 1.0, atol=1e-12,
            )

    @_python
    def test_bounded_in_unit_interval(self):
        rng = np.random.default_rng(2)
        T, N, W = 40, 5, 10
        phases = rng.uniform(0, 2 * np.pi, (T, N))
        plv = market_plv(phases, window=W)
        assert np.all(plv >= 0.0)
        assert np.all(plv <= 1.0 + 1e-12)

    @_python
    def test_locked_pair_gives_plv_one(self):
        """Two assets with identical phases across the window
        give PLV_ij = 1."""
        T, N, W = 20, 3, 10
        phases = np.zeros((T, N))
        # All three assets locked to the same trajectory.
        for t in range(T):
            phases[t, :] = 0.1 * t
        plv = market_plv(phases, window=W)
        np.testing.assert_allclose(plv, 1.0, atol=1e-12)

    @_python
    def test_window_larger_than_t_returns_empty(self):
        T, N = 5, 3
        phases = np.zeros((T, N))
        plv = market_plv(phases, window=10)
        assert plv.shape == (0, N, N)

    @_python
    def test_rejects_non_2d(self):
        with pytest.raises(ValueError, match="phases must be"):
            market_plv(np.array([1.0, 2.0, 3.0]))


class TestDetectRegimes:
    def test_sync_and_desync_classes(self):
        R = np.array([0.1, 0.5, 0.9, 0.25, 0.8])
        regimes = detect_regimes(R, sync_threshold=0.7, desync_threshold=0.3)
        # 0=desync, 1=transition, 2=synchronised
        np.testing.assert_array_equal(regimes, [0, 1, 2, 0, 2])

    def test_output_dtype_is_int32(self):
        R = np.array([0.5])
        regimes = detect_regimes(R)
        assert regimes.dtype == np.int32


class TestSyncWarning:
    def test_crossing_detected(self):
        # Cross threshold around index 6
        R = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 0.8])
        w = sync_warning(R, threshold=0.7, lookback=1)
        assert np.any(w)
        # First crossing from below at index 6
        assert w[6]

    def test_no_crossing_below_threshold(self):
        R = np.full(10, 0.3)
        w = sync_warning(R, threshold=0.7, lookback=1)
        assert not np.any(w)


class TestExtractPhase:
    def test_hilbert_output_shape_1d(self):
        x = np.sin(np.linspace(0, 4 * np.pi, 100))
        phase = extract_phase(x)
        assert phase.shape == (100,)
        assert np.all(phase >= 0.0)
        assert np.all(phase < 2 * np.pi + 1e-12)

    def test_hilbert_output_shape_2d(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal((200, 4))
        phase = extract_phase(x)
        assert phase.shape == (200, 4)
        assert np.all(phase >= 0.0)
        assert np.all(phase < 2 * np.pi + 1e-12)


class TestHypothesis:
    @_python
    @given(
        t=st.integers(min_value=2, max_value=20),
        n=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=8, deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_order_parameter_shape_and_bounds(self, t, n, seed):
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, 2 * np.pi, (t, n))
        R = market_order_parameter(phases)
        assert R.shape == (t,)
        assert np.all(R >= 0.0)
        assert np.all(R <= 1.0 + 1e-12)


class TestDispatcherSurface:
    def test_available_non_empty(self):
        assert m_mod.AVAILABLE_BACKENDS
        assert "python" in m_mod.AVAILABLE_BACKENDS

    def test_active_is_first(self):
        assert m_mod.AVAILABLE_BACKENDS[0] == m_mod.ACTIVE_BACKEND
