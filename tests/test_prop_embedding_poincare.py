# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Property-based embedding & optimal parameter proofs

"""Hypothesis-driven invariant proofs for delay embedding, optimal delay
(mutual information), and optimal dimension (false nearest neighbours).

Takens 1981, Fraser & Swinney 1986, Kennel et al. 1992.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.monitor.embedding import (
    auto_embed,
    delay_embed,
    optimal_delay,
    optimal_dimension,
)

# ── 1. delay_embed shape invariant ──────────────────────────────────────


class TestDelayEmbedInvariants:
    @given(
        delay=st.integers(min_value=1, max_value=5),
        dim=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=50, deadline=None)
    def test_shape(self, delay: int, dim: int, seed: int) -> None:
        """Output shape = (T - (m-1)*τ, m)."""
        T = 200
        rng = np.random.default_rng(seed)
        signal = rng.standard_normal(T)
        emb = delay_embed(signal, delay=delay, dimension=dim)
        expected_rows = T - (dim - 1) * delay
        assert emb.shape == (expected_rows, dim)

    @given(
        delay=st.integers(min_value=1, max_value=3),
        dim=st.integers(min_value=1, max_value=4),
    )
    @settings(max_examples=30, deadline=None)
    def test_first_column_is_signal_tail(self, delay: int, dim: int) -> None:
        """First column of embedding = signal[0:T_eff]."""
        T = 100
        signal = np.arange(T, dtype=float)
        emb = delay_embed(signal, delay=delay, dimension=dim)
        T_eff = T - (dim - 1) * delay
        np.testing.assert_array_equal(emb[:, 0], signal[:T_eff])

    def test_dim1_delay1_is_signal(self) -> None:
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        emb = delay_embed(signal, delay=1, dimension=1)
        np.testing.assert_array_equal(emb.ravel(), signal)

    def test_too_short_signal_raises(self) -> None:
        with pytest.raises(ValueError):
            delay_embed(np.array([1.0, 2.0]), delay=5, dimension=3)

    @given(seed=st.integers(min_value=0, max_value=200))
    @settings(max_examples=30, deadline=None)
    def test_all_finite(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        signal = rng.standard_normal(100)
        emb = delay_embed(signal, delay=2, dimension=3)
        assert np.all(np.isfinite(emb))


# ── 2. optimal_delay ────────────────────────────────────────────────────


class TestOptimalDelay:
    @given(seed=st.integers(min_value=0, max_value=200))
    @settings(max_examples=30, deadline=None)
    def test_at_least_one(self, seed: int) -> None:
        """Delay ≥ 1."""
        rng = np.random.default_rng(seed)
        signal = rng.standard_normal(200)
        tau = optimal_delay(signal, max_lag=50)
        assert tau >= 1

    @given(seed=st.integers(min_value=0, max_value=200))
    @settings(max_examples=30, deadline=None)
    def test_integer(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        signal = rng.standard_normal(200)
        tau = optimal_delay(signal, max_lag=50)
        assert isinstance(tau, (int, np.integer))

    def test_periodic_signal_finds_quarter_period(self) -> None:
        """Sinusoidal → optimal delay ≈ T/4 (MI first minimum)."""
        T = 500
        period = 40
        signal = np.sin(2 * np.pi * np.arange(T) / period)
        tau = optimal_delay(signal, max_lag=period)
        # Should be near period/4 ≈ 10, within reasonable range
        assert 3 <= tau <= period // 2

    def test_constant_signal_returns_one(self) -> None:
        signal = np.ones(100)
        tau = optimal_delay(signal, max_lag=20)
        assert tau == 1


# ── 3. optimal_dimension ────────────────────────────────────────────────


class TestOptimalDimension:
    @given(seed=st.integers(min_value=0, max_value=100))
    @settings(
        max_examples=20, suppress_health_check=[HealthCheck.too_slow], deadline=None
    )
    def test_at_least_one(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        signal = rng.standard_normal(200)
        m = optimal_dimension(signal, delay=2, max_dim=5)
        assert m >= 1

    @given(seed=st.integers(min_value=0, max_value=100))
    @settings(
        max_examples=20, suppress_health_check=[HealthCheck.too_slow], deadline=None
    )
    def test_at_most_max_dim(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        signal = rng.standard_normal(200)
        max_dim = 5
        m = optimal_dimension(signal, delay=2, max_dim=max_dim)
        assert m <= max_dim

    def test_constant_signal_dim_one(self) -> None:
        signal = np.ones(100)
        m = optimal_dimension(signal, delay=1, max_dim=5)
        assert m == 1

    def test_sine_wave_low_dimension(self) -> None:
        """Sine is 2D attractor → optimal m should be 2 or 3."""
        T = 500
        signal = np.sin(np.linspace(0, 20 * np.pi, T))
        m = optimal_dimension(signal, delay=5, max_dim=8)
        assert 1 <= m <= 4


# ── 4. auto_embed ───────────────────────────────────────────────────────


class TestAutoEmbed:
    @given(seed=st.integers(min_value=0, max_value=100))
    @settings(
        max_examples=15, suppress_health_check=[HealthCheck.too_slow], deadline=None
    )
    def test_result_fields(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        signal = rng.standard_normal(300)
        result = auto_embed(signal, max_lag=30, max_dim=5)
        assert result.delay >= 1
        assert result.dimension >= 1
        T_eff = len(signal) - (result.dimension - 1) * result.delay
        assert result.T_effective == T_eff
        assert result.trajectory.shape == (T_eff, result.dimension)

    @given(seed=st.integers(min_value=0, max_value=100))
    @settings(
        max_examples=15, suppress_health_check=[HealthCheck.too_slow], deadline=None
    )
    def test_all_finite(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        signal = rng.standard_normal(300)
        result = auto_embed(signal, max_lag=30, max_dim=5)
        assert np.all(np.isfinite(result.trajectory))


# Pipeline wiring: hypothesis-driven invariant proofs exercise the pipeline.
