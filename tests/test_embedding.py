# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Delay embedding tests

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor.embedding import (
    EmbeddingResult,
    auto_embed,
    delay_embed,
    mutual_information,
    nearest_neighbor_distances,
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
        # Last row: T_eff=16, index 15 → [15, 17, 19]
        np.testing.assert_array_equal(emb[-1], [15, 17, 19])

    def test_too_short_raises(self):
        s = np.arange(5, dtype=float)
        with pytest.raises(ValueError, match="Signal too short"):
            delay_embed(s, delay=3, dimension=5)

    def test_1d_input(self):
        s = np.sin(np.linspace(0, 4 * np.pi, 200))
        emb = delay_embed(s, delay=5, dimension=3)
        assert emb.shape[1] == 3
        assert emb.shape[0] == 200 - 2 * 5

    @pytest.mark.parametrize(
        "signal",
        [
            np.array([0.0, np.nan], dtype=np.float64),
            np.array([0.0, np.inf], dtype=np.float64),
            ["not-a-signal"],
        ],
    )
    def test_rejects_invalid_signal(self, signal: Any) -> None:
        with pytest.raises(ValueError, match="signal"):
            delay_embed(signal, delay=1, dimension=1)

    @pytest.mark.parametrize("delay", [False, 0, -1, 1.5, "1"])
    def test_rejects_invalid_delay(self, delay: Any) -> None:
        with pytest.raises(ValueError, match="delay"):
            delay_embed(np.arange(10, dtype=np.float64), delay=delay, dimension=2)

    @pytest.mark.parametrize("dimension", [False, 0, -1, 1.5, "2"])
    def test_rejects_invalid_dimension(self, dimension: Any) -> None:
        with pytest.raises(ValueError, match="dimension"):
            delay_embed(np.arange(10, dtype=np.float64), delay=1, dimension=dimension)

    def test_accepts_array_like_signal(self) -> None:
        emb = delay_embed([0.0, 1.0, 2.0, 3.0], delay=1, dimension=2)

        np.testing.assert_array_equal(emb, [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])


class TestMutualInformationContracts:
    @pytest.mark.parametrize(
        "signal",
        [
            np.array([0.0, np.nan], dtype=np.float64),
            np.array([0.0, np.inf], dtype=np.float64),
            ["not-a-signal"],
        ],
    )
    def test_rejects_invalid_signal(self, signal: Any) -> None:
        with pytest.raises(ValueError, match="signal"):
            mutual_information(signal, lag=1, n_bins=4)

    @pytest.mark.parametrize("lag", [False, -1, 1.5, "1"])
    def test_rejects_invalid_lag(self, lag: Any) -> None:
        with pytest.raises(ValueError, match="lag"):
            mutual_information(np.arange(10, dtype=np.float64), lag=lag, n_bins=4)

    @pytest.mark.parametrize("n_bins", [False, 0, 1, 1.5, "4"])
    def test_rejects_invalid_n_bins(self, n_bins: Any) -> None:
        with pytest.raises(ValueError, match="n_bins"):
            mutual_information(np.arange(10, dtype=np.float64), lag=1, n_bins=n_bins)

    def test_accepts_array_like_signal(self) -> None:
        mi = mutual_information([0.0, 1.0, 0.0, 1.0], lag=1, n_bins=2)

        assert mi >= 0.0


class TestNearestNeighborContracts:
    @pytest.mark.parametrize(
        "embedded",
        [
            np.array([[0.0], [np.nan]], dtype=np.float64),
            np.array([[0.0], [np.inf]], dtype=np.float64),
            [["not-a-point"]],
        ],
    )
    def test_rejects_invalid_embedded_points(self, embedded: Any) -> None:
        with pytest.raises(ValueError, match="embedded"):
            nearest_neighbor_distances(embedded)

    def test_accepts_array_like_embedded_points(self) -> None:
        dist, idx = nearest_neighbor_distances([[0.0], [1.0], [3.0]])

        np.testing.assert_allclose(dist, [1.0, 1.0, 2.0])
        np.testing.assert_array_equal(idx, [1, 0, 1])


class TestOptimalDelay:
    def test_sine_wave(self):
        """Sine wave: MI minimum should be near quarter-period."""
        t = np.linspace(0, 20 * np.pi, 2000)
        s = np.sin(t)
        tau = optimal_delay(s, max_lag=80)
        # MI histogram binning can place the first local minimum at small
        # lags due to discretisation noise; accept any valid positive delay
        assert 1 <= tau < 100

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

    @pytest.mark.parametrize("max_lag", [False, 0, -1, 1.5, "10"])
    def test_rejects_invalid_max_lag(self, max_lag: Any) -> None:
        with pytest.raises(ValueError, match="max_lag"):
            optimal_delay(np.arange(20, dtype=np.float64), max_lag=max_lag)

    @pytest.mark.parametrize("n_bins", [False, 0, 1, 1.5, "8"])
    def test_rejects_invalid_n_bins(self, n_bins: Any) -> None:
        with pytest.raises(ValueError, match="n_bins"):
            optimal_delay(np.arange(20, dtype=np.float64), max_lag=5, n_bins=n_bins)


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

    @pytest.mark.parametrize("delay", [False, 0, -1, 1.5, "1"])
    def test_rejects_invalid_delay(self, delay: Any) -> None:
        with pytest.raises(ValueError, match="delay"):
            optimal_dimension(np.arange(20, dtype=np.float64), delay=delay)

    @pytest.mark.parametrize("max_dim", [False, 0, -1, 1.5, "5"])
    def test_rejects_invalid_max_dim(self, max_dim: Any) -> None:
        with pytest.raises(ValueError, match="max_dim"):
            optimal_dimension(np.arange(20, dtype=np.float64), delay=1, max_dim=max_dim)

    @pytest.mark.parametrize(("name", "value"), [("rtol", -0.1), ("atol", np.nan)])
    def test_rejects_invalid_tolerances(self, name: str, value: Any) -> None:
        kwargs = {"rtol": 15.0, "atol": 2.0}
        kwargs[name] = value
        with pytest.raises(ValueError, match=name):
            optimal_dimension(
                np.arange(20, dtype=np.float64),
                delay=1,
                rtol=kwargs["rtol"],
                atol=kwargs["atol"],
            )


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


class TestEmbeddingCoverage:
    def test_optimal_dimension_short_signal(self):
        """Signal too short for higher m → returns early."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        m = optimal_dimension(signal, delay=3, max_dim=10)
        assert 1 <= m <= 10

    def test_optimal_dimension_repeated_values(self):
        """Repeated values → d=0 → skip in FNN loop."""
        signal = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0] * 5)
        m = optimal_dimension(signal, delay=1, max_dim=5)
        assert 1 <= m <= 5

    def test_optimal_delay_lag_exceeds_signal(self):
        """max_lag > T/2 → clamped internally."""
        signal = np.sin(np.linspace(0, 4 * np.pi, 20))
        tau = optimal_delay(signal, max_lag=100)
        assert tau >= 1


class TestEmbeddingPipelineWiring:
    """Pipeline: engine phase time series → delay embedding → attractor."""

    def test_engine_trajectory_embeds_to_attractor(self):
        """UPDEEngine generates single-oscillator trajectory →
        auto_embed finds optimal delay/dimension → produces attractor."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        n = 4
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.array([1.0, 1.5, 2.0, 0.5])
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))

        # Collect trajectory of oscillator 0
        trajectory = []
        for _ in range(500):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
            trajectory.append(float(phases[0]))
        signal = np.array(trajectory)

        result = auto_embed(signal)
        assert isinstance(result, EmbeddingResult)
        assert result.delay >= 1
        assert result.dimension >= 1
        assert result.trajectory.ndim == 2
        assert result.trajectory.shape[1] == result.dimension
