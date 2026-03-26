# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Delay embedding tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor.embedding import (
    EmbeddingResult,
    auto_embed,
    delay_embed,
    optimal_delay,
    optimal_dimension,
)


class TestDelayEmbed:
    def test_shape(self):
        s = np.arange(100, dtype=float)
        emb = delay_embed(s, delay=3, dimension=4)
        # T' = 100 - (4-1)*3 = 91
        assert emb.shape == (91, 4)

    def test_values(self):
        """Check that columns are correct time-shifted copies."""
        s = np.arange(20, dtype=float)
        emb = delay_embed(s, delay=2, dimension=3)
        # First row: [0, 2, 4]
        np.testing.assert_array_equal(emb[0], [0, 2, 4])
        # Last row: [14, 16, 18]
        np.testing.assert_array_equal(emb[-1], [14, 16, 18])

    def test_too_short_raises(self):
        s = np.arange(5, dtype=float)
        with pytest.raises(ValueError, match="Signal too short"):
            delay_embed(s, delay=3, dimension=5)

    def test_1d_input(self):
        s = np.sin(np.linspace(0, 4 * np.pi, 200))
        emb = delay_embed(s, delay=5, dimension=3)
        assert emb.shape[1] == 3
        assert emb.shape[0] == 200 - 2 * 5


class TestOptimalDelay:
    def test_sine_wave(self):
        """Sine wave: MI minimum should be near quarter-period."""
        t = np.linspace(0, 20 * np.pi, 2000)
        s = np.sin(t)
        tau = optimal_delay(s, max_lag=80)
        # Quarter period = 2000/(20*2) * 0.5 = 50 samples (approx)
        # MI minimum should be in a reasonable range
        assert 5 < tau < 100

    def test_constant_signal(self):
        """Constant signal has no structure → returns 1."""
        s = np.ones(500)
        tau = optimal_delay(s, max_lag=50)
        assert tau == 1

    def test_short_signal(self):
        """Short signal still returns valid delay."""
        s = np.sin(np.linspace(0, 2 * np.pi, 30))
        tau = optimal_delay(s, max_lag=10)
        assert tau >= 1


class TestOptimalDimension:
    def test_sine_2d(self):
        """Sine wave lives on a 2D manifold (circle)."""
        t = np.linspace(0, 40 * np.pi, 4000)
        s = np.sin(t)
        tau = optimal_delay(s, max_lag=50)
        m = optimal_dimension(s, delay=tau, max_dim=6)
        assert 2 <= m <= 4

    def test_constant_returns_1(self):
        s = np.ones(500)
        m = optimal_dimension(s, delay=1, max_dim=5)
        assert m == 1


class TestAutoEmbed:
    def test_returns_result(self):
        t = np.linspace(0, 20 * np.pi, 2000)
        s = np.sin(t) + 0.1 * np.sin(3 * t)
        result = auto_embed(s)
        assert isinstance(result, EmbeddingResult)
        assert result.trajectory.ndim == 2
        assert result.delay >= 1
        assert result.dimension >= 1
        assert result.T_effective == result.trajectory.shape[0]

    def test_embedding_preserves_structure(self):
        """Embedded sine should form a roughly circular trajectory."""
        t = np.linspace(0, 40 * np.pi, 4000)
        s = np.sin(t)
        emb = delay_embed(s, delay=25, dimension=2)
        # Check it spans both axes (not degenerate)
        assert np.std(emb[:, 0]) > 0.3
        assert np.std(emb[:, 1]) > 0.3
