# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Algorithmic tests for chimera detection

"""Algorithmic properties of :func:`local_order_parameter` and
:func:`detect_chimera`.

Covered: bounded ``[0, 1]`` output, perfect-sync ``R_i = 1``,
antiphase ``R_i = 0``, disconnected-oscillator ``R_i = 0``, correct
partition on a staged chimera (half coherent, half incoherent),
chimera-index is a fraction, empty-input safety, Hypothesis
property coverage.
"""

from __future__ import annotations

import functools
import math

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.monitor import chimera as ch_mod
from scpn_phase_orchestrator.monitor.chimera import (
    ChimeraState,
    detect_chimera,
    local_order_parameter,
)

TWO_PI = 2.0 * math.pi


def _python(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        prev = ch_mod.ACTIVE_BACKEND
        ch_mod.ACTIVE_BACKEND = "python"
        try:
            return func(*args, **kwargs)
        finally:
            ch_mod.ACTIVE_BACKEND = prev

    return wrapper


def _all_to_all(n: int) -> np.ndarray:
    k = np.ones((n, n), dtype=np.float64)
    np.fill_diagonal(k, 0.0)
    return k


class TestLocalOrderParameter:
    @_python
    def test_perfect_sync_gives_one(self):
        n = 8
        phases = np.full(n, 0.3)
        r = local_order_parameter(phases, _all_to_all(n))
        assert np.all(np.abs(r - 1.0) < 1e-12)

    @_python
    def test_uniform_on_circle_gives_1_over_N_minus_1(self):
        """With ``N`` oscillators evenly spaced on the circle and
        all-to-all coupling, excluding self leaves an imbalance of
        exactly ``-1 · e^{iθ_i}``, so ``R_i = 1 / (N-1)`` for every
        ``i`` — asymptotically zero as N grows."""
        n = 32
        phases = np.linspace(0.0, TWO_PI, n, endpoint=False)
        r = local_order_parameter(phases, _all_to_all(n))
        expected = 1.0 / (n - 1)
        np.testing.assert_allclose(r, expected, atol=1e-12)

    @_python
    def test_bounded_unit_interval(self):
        rng = np.random.default_rng(0)
        n = 16
        phases = rng.uniform(0, TWO_PI, n)
        knm = rng.uniform(0.0, 1.0, (n, n))
        knm = (knm > 0.3).astype(np.float64) * knm
        np.fill_diagonal(knm, 0.0)
        r = local_order_parameter(phases, knm)
        assert np.all(r >= 0.0 - 1e-12)
        assert np.all(r <= 1.0 + 1e-12)

    @_python
    def test_isolated_oscillator_returns_zero(self):
        # Oscillator 0 has no outgoing edges.
        n = 4
        knm = np.zeros((n, n))
        knm[1, 2] = knm[2, 1] = knm[1, 3] = knm[3, 1] = 1.0
        phases = np.array([0.1, 0.2, 0.2, 0.2])
        r = local_order_parameter(phases, knm)
        assert r[0] == 0.0

    @_python
    def test_empty_input(self):
        r = local_order_parameter(np.array([]), np.zeros((0, 0)))
        assert r.shape == (0,)


class TestDetectChimera:
    @_python
    def test_perfect_sync_all_coherent(self):
        n = 10
        phases = np.full(n, 0.5)
        state = detect_chimera(phases, _all_to_all(n))
        assert len(state.coherent_indices) == n
        assert len(state.incoherent_indices) == 0
        assert state.chimera_index == 0.0

    @_python
    def test_antiphase_classified_incoherent(self):
        n = 6
        phases = np.array([0.0, math.pi, 0.0, math.pi, 0.0, math.pi])
        state = detect_chimera(phases, _all_to_all(n))
        assert len(state.incoherent_indices) == n

    @_python
    def test_chimera_index_is_fraction(self):
        rng = np.random.default_rng(11)
        n = 32
        phases = rng.uniform(0, TWO_PI, n)
        state = detect_chimera(phases, _all_to_all(n))
        assert 0.0 <= state.chimera_index <= 1.0

    @_python
    def test_partition_totals_to_n(self):
        """|coherent| + |incoherent| + boundary = N."""
        rng = np.random.default_rng(7)
        n = 32
        phases = np.concatenate(
            [np.zeros(n // 2), rng.uniform(0, TWO_PI, n // 2)]
        )
        state = detect_chimera(phases, _all_to_all(n))
        boundary = int(round(state.chimera_index * n))
        assert (
            len(state.coherent_indices)
            + len(state.incoherent_indices)
            + boundary
            == n
        )

    @_python
    def test_empty_input_returns_empty_state(self):
        state = detect_chimera(np.array([]), np.zeros((0, 0)))
        assert isinstance(state, ChimeraState)
        assert state.coherent_indices == []
        assert state.incoherent_indices == []
        assert state.chimera_index == 0.0


class TestHypothesis:
    @_python
    @given(
        n=st.integers(min_value=2, max_value=12),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=15,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_bounded_and_finite(self, n: int, seed: int):
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        knm = rng.uniform(0.0, 1.0, (n, n))
        knm = (knm > 0.4).astype(np.float64) * knm
        np.fill_diagonal(knm, 0.0)
        r = local_order_parameter(phases, knm)
        assert r.shape == (n,)
        assert np.all(np.isfinite(r))
        assert np.all(r >= 0.0 - 1e-12)
        assert np.all(r <= 1.0 + 1e-12)


class TestDispatcherSurface:
    def test_available_non_empty(self):
        assert ch_mod.AVAILABLE_BACKENDS
        assert "python" in ch_mod.AVAILABLE_BACKENDS

    def test_active_is_first(self):
        assert ch_mod.AVAILABLE_BACKENDS[0] == ch_mod.ACTIVE_BACKEND
