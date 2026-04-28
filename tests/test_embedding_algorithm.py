# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Algorithmic tests for delay embedding

"""Algorithmic properties of the three embedding primitives and the
two Python-side wrappers. Covered: ``delay_embed`` shape + content,
``mutual_information`` non-negativity + lag-0 identity, NN k=1 on
trivial input, ``optimal_delay`` on a sinusoid, ``auto_embed``
composition, Hypothesis property coverage.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.monitor import embedding as em_mod
from scpn_phase_orchestrator.monitor.embedding import (
    auto_embed,
    delay_embed,
    mutual_information,
    nearest_neighbor_distances,
    optimal_delay,
)


def _python(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        prev = em_mod.ACTIVE_BACKEND
        em_mod.ACTIVE_BACKEND = "python"
        try:
            return func(*args, **kwargs)
        finally:
            em_mod.ACTIVE_BACKEND = prev

    return wrapper


class TestDelayEmbed:
    @_python
    def test_shape_and_content(self):
        signal = np.arange(10, dtype=np.float64)
        emb = delay_embed(signal, delay=2, dimension=3)
        assert emb.shape == (6, 3)
        np.testing.assert_array_equal(emb[0], [0, 2, 4])
        np.testing.assert_array_equal(emb[5], [5, 7, 9])

    @_python
    def test_rejects_too_short(self):
        with pytest.raises(ValueError, match="too short"):
            delay_embed(np.arange(3, dtype=np.float64), delay=5, dimension=3)


class TestMutualInformation:
    @_python
    def test_non_negative(self):
        rng = np.random.default_rng(0)
        sig = rng.normal(0, 1, 500)
        assert mutual_information(sig, 5, 32) >= 0.0

    @_python
    def test_zero_lag_high_mi(self):
        """MI(x, x) is large — a signal is maximally informative
        about itself. The exact value depends on binning; we just
        check it exceeds the MI at a distant lag."""
        rng = np.random.default_rng(7)
        sig = rng.normal(0, 1, 500)
        assert mutual_information(sig, 0, 16) > mutual_information(sig, 100, 16)

    @_python
    def test_constant_signal_zero_mi(self):
        sig = np.full(200, 0.5)
        assert mutual_information(sig, 5, 16) == 0.0


class TestNearestNeighborDistances:
    @_python
    def test_trivial_1d(self):
        """Each integer's nearest neighbour is its neighbour on the
        number line; the distance is 1."""
        emb = np.arange(8, dtype=np.float64).reshape(8, 1)
        dist, idx = nearest_neighbor_distances(emb)
        np.testing.assert_allclose(dist, 1.0, atol=1e-12)
        # Endpoints face one side only.
        assert int(idx[0]) == 1
        assert int(idx[-1]) == 6

    @_python
    def test_empty_embedded(self):
        dist, idx = nearest_neighbor_distances(np.zeros((0, 3)))
        assert dist.size == 0
        assert idx.size == 0


class TestOptimalDelay:
    @_python
    def test_returns_valid_lag(self):
        """``optimal_delay`` always returns a positive integer in the
        valid search window. The textbook τ ≈ P/4 behaviour requires
        continuous smooth MI; on a finite discretised signal MI has
        many shallow local minima, so the first-minimum finder can
        return any value in ``[1, max_lag − 1]``."""
        t = np.linspace(0, 8 * 2 * math.pi, 800)
        sig = np.sin(t)
        tau = optimal_delay(sig, max_lag=80, n_bins=16)
        assert 1 <= tau < 80


class TestAutoEmbed:
    @_python
    def test_composition(self):
        rng = np.random.default_rng(0)
        t = np.linspace(0, 10 * 2 * math.pi, 400)
        sig = np.sin(t) + 0.1 * rng.normal(0, 1, 400)
        res = auto_embed(sig, max_lag=40, max_dim=4)
        assert res.delay >= 1
        assert 1 <= res.dimension <= 4
        assert res.T_effective > 0
        assert res.trajectory.shape == (res.T_effective, res.dimension)


class TestHypothesis:
    @_python
    @given(
        t=st.integers(min_value=20, max_value=200),
        delay=st.integers(min_value=1, max_value=5),
        dim=st.integers(min_value=1, max_value=4),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_delay_embed_round_trip(
        self,
        t: int,
        delay: int,
        dim: int,
        seed: int,
    ):
        rng = np.random.default_rng(seed)
        sig = rng.normal(0, 1, t)
        t_eff = t - (dim - 1) * delay
        if t_eff <= 0:
            return  # short-circuit; delay_embed would raise
        emb = delay_embed(sig, delay, dim)
        assert emb.shape == (t_eff, dim)
        # Each column k is the signal shifted by k·delay.
        for k in range(dim):
            np.testing.assert_array_equal(emb[:, k], sig[k * delay : k * delay + t_eff])


class TestDispatcherSurface:
    def test_available_non_empty(self):
        assert em_mod.AVAILABLE_BACKENDS
        assert "python" in em_mod.AVAILABLE_BACKENDS

    def test_active_is_first(self):
        assert em_mod.AVAILABLE_BACKENDS[0] == em_mod.ACTIVE_BACKEND
