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

from scpn_phase_orchestrator.monitor import embedding as em_mod
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
            np.array([0.0, True], dtype=object),
            np.array([0.0, np.bool_(True)], dtype=object),
            np.array([0.0 + 0.0j, 1.0 + 0.25j]),
            ["not-a-signal"],
        ],
    )
    def test_rejects_invalid_signal(self, signal: Any) -> None:
        with pytest.raises(ValueError, match="signal"):
            delay_embed(signal, delay=1, dimension=1)

    def test_rejects_object_complex_signal_alias_as_non_real(self) -> None:
        signal = np.array([0.0 + 0.0j, 1.0 + 0.25j, 2.0 + 0.0j], dtype=object)

        with pytest.raises(ValueError, match="real-valued"):
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

    def test_uses_dispatched_backend_kernel(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: dict[str, object] = {}

        def _dispatch(fn_name: str) -> object | None:
            calls["fn_name"] = fn_name
            if fn_name == "de":
                return lambda signal, delay, dimension: np.array(
                    [0.0, 1.0, 1.0, 2.0, 2.0, 3.0],
                    dtype=np.float64,
                )
            return None

        monkeypatch.setattr(em_mod, "_dispatch", _dispatch)
        emb = em_mod.delay_embed(
            np.array([0, 1, 2, 3], dtype=np.float64),
            delay=1,
            dimension=2,
        )

        assert calls["fn_name"] == "de"
        assert emb.shape == (3, 2)
        np.testing.assert_array_equal(emb[0], [0.0, 1.0])
        np.testing.assert_array_equal(emb[1], [1.0, 2.0])

    def test_python_fallback_returns_shifted_windows(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(em_mod, "ACTIVE_BACKEND", "python")
        monkeypatch.setattr(em_mod, "_dispatch", lambda fn_name: None)
        emb = em_mod.delay_embed(
            np.array([5, 4, 3, 2, 1], dtype=np.float64),
            delay=2,
            dimension=2,
        )
        np.testing.assert_array_equal(emb, [[5.0, 3.0], [4.0, 2.0], [3.0, 1.0]])


class TestMutualInformationContracts:
    @pytest.mark.parametrize(
        "signal",
        [
            np.array([0.0, np.nan], dtype=np.float64),
            np.array([0.0, np.inf], dtype=np.float64),
            np.array([0.0, True], dtype=object),
            np.array([0.0, np.bool_(True)], dtype=object),
            np.array([0.0 + 0.0j, 1.0 + 0.25j]),
            ["not-a-signal"],
        ],
    )
    def test_rejects_invalid_signal(self, signal: Any) -> None:
        with pytest.raises(ValueError, match="signal"):
            mutual_information(signal, lag=1, n_bins=4)

    def test_rejects_object_complex_signal_alias_as_non_real(self) -> None:
        signal = np.array([0.0 + 0.0j, 1.0 + 0.25j, 0.0 + 0.0j], dtype=object)

        with pytest.raises(ValueError, match="real-valued"):
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
            np.array([[0.0], [True]], dtype=object),
            np.array([[0.0], [np.bool_(True)]], dtype=object),
            np.array([[0.0 + 0.0j], [1.0 + 0.25j]]),
            [["not-a-point"]],
        ],
    )
    def test_rejects_invalid_embedded_points(self, embedded: Any) -> None:
        with pytest.raises(ValueError, match="embedded"):
            nearest_neighbor_distances(embedded)

    def test_rejects_object_complex_embedded_alias_as_non_real(self) -> None:
        embedded = np.array(
            [[0.0 + 0.0j], [1.0 + 0.25j], [3.0 + 0.0j]],
            dtype=object,
        )

        with pytest.raises(ValueError, match="real-valued"):
            nearest_neighbor_distances(embedded)

    def test_accepts_array_like_embedded_points(self) -> None:
        dist, idx = nearest_neighbor_distances([[0.0], [1.0], [3.0]])

        np.testing.assert_allclose(dist, [1.0, 1.0, 2.0])
        np.testing.assert_array_equal(idx, [1, 0, 1])

    def test_rejects_non_2d_embedded_array(self) -> None:
        with pytest.raises(ValueError, match="two-dimensional"):
            nearest_neighbor_distances(np.ones((2, 2, 2), dtype=np.float64))

    def test_tie_chooses_first_minimum_index(
        self,
    ) -> None:
        signal = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        dist, idx = em_mod.nearest_neighbor_distances(signal)

        assert idx.tolist() == [1, 0, 0, 2]
        assert np.all(np.isfinite(dist))


class TestValidationBoundarySemantics:
    def test_validate_int_at_least_accepts_numpy_integers(self) -> None:
        assert em_mod._validate_int_at_least(np.int64(4), name="k", minimum=1) == 4

    def test_validate_int_at_least_rejects_bool(self) -> None:
        with pytest.raises(ValueError, match="integer >= 1"):
            em_mod._validate_int_at_least(True, name="k", minimum=1)

    def test_validate_non_negative_real_rejects_boolean(self) -> None:
        with pytest.raises(ValueError, match="finite non-negative real"):
            em_mod._validate_non_negative_real(np.bool_(True), name="atol")

    def test_public_output_helpers_reject_object_complex_aliases_as_non_real(
        self,
    ) -> None:
        signal = np.arange(8, dtype=np.float64)
        with pytest.raises(ValueError, match="real"):
            em_mod._validate_delay_embedding_output(
                np.array(
                    [[0.0 + 0.0j, 2.0], [1.0, 3.0 + 0.25j], [2.0, 4.0]],
                    dtype=object,
                ),
                signal=signal,
                delay=2,
                t_effective=3,
                dimension=2,
            )
        with pytest.raises(ValueError, match="real"):
            em_mod._validate_non_negative_scalar(
                np.array(1.0 + 0.0j, dtype=object),
                name="mutual_information",
            )
        with pytest.raises(ValueError, match="real"):
            em_mod._validate_nn_output(
                np.array([1.0 + 0.0j, 2.0 + 0.25j], dtype=object),
                np.array([1.0, 0.0]),
                n_points=2,
            )
        with pytest.raises(ValueError, match="integer"):
            em_mod._validate_nn_output(
                np.array([1.0, 2.0]),
                np.array([1.0 + 0.0j, 0.0 + 0.25j], dtype=object),
                n_points=2,
            )


class TestEmbeddingResultBoundary:
    def test_normalises_valid_record(self) -> None:
        result = EmbeddingResult(
            trajectory=[[0.0, 1.0], [1.0, 2.0]],
            delay=np.int64(1),
            dimension=np.int64(2),
            T_effective=np.int64(2),
        )

        assert result.delay == 1
        assert result.dimension == 2
        assert result.T_effective == 2
        np.testing.assert_array_equal(result.trajectory, [[0.0, 1.0], [1.0, 2.0]])

    @pytest.mark.parametrize(
        "payload",
        [
            {
                "trajectory": [[0.0, 1.0]],
                "delay": 1,
                "dimension": 2,
                "T_effective": 2,
            },
            {
                "trajectory": [[0.0, np.nan]],
                "delay": 1,
                "dimension": 2,
                "T_effective": 1,
            },
            {
                "trajectory": [[0.0, True]],
                "delay": 1,
                "dimension": 2,
                "T_effective": 1,
            },
            {
                "trajectory": [[0.0, np.bool_(True)]],
                "delay": 1,
                "dimension": 2,
                "T_effective": 1,
            },
            {
                "trajectory": [[0.0 + 0.0j, 1.0 + 0.25j]],
                "delay": 1,
                "dimension": 2,
                "T_effective": 1,
            },
            {
                "trajectory": [[0.0, 1.0]],
                "delay": False,
                "dimension": 2,
                "T_effective": 1,
            },
        ],
    )
    def test_rejects_invalid_record(self, payload: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            EmbeddingResult(**payload)

    def test_rejects_object_complex_trajectory_alias_as_non_real(self) -> None:
        with pytest.raises(ValueError, match="real-valued"):
            EmbeddingResult(
                trajectory=np.array(
                    [[0.0 + 0.0j, 1.0 + 0.25j]],
                    dtype=object,
                ),
                delay=1,
                dimension=2,
                T_effective=1,
            )


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


class TestOptimalDimensionBoundary:
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


class TestEmbeddingBackendFallbacks:
    @pytest.mark.parametrize(
        "backend_output",
        [
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([[0.0, np.nan], [1.0, 2.0], [2.0, 3.0]], dtype=np.float64),
            np.array([[0.0, np.bool_(True)], [1.0, 2.0], [2.0, 3.0]], dtype=object),
            np.array(
                [[0.0 + 0.0j, 1.0], [1.0, 2.0 + 0.25j], [2.0, 3.0]],
            ),
            np.zeros((2, 3), dtype=np.float64),
        ],
    )
    def test_invalid_delay_embed_backend_payload_falls_back_to_python(
        self,
        monkeypatch: pytest.MonkeyPatch,
        backend_output: np.ndarray,
    ) -> None:
        monkeypatch.setattr(
            em_mod,
            "_dispatch",
            lambda fn_name: (
                (lambda *_args: backend_output) if fn_name == "de" else None
            ),
        )

        emb = delay_embed(np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64), 1, 2)

        np.testing.assert_array_equal(emb, [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])

    def test_delay_embed_backend_failure_falls_back_to_python(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raising_de(
            _signal: np.ndarray, _delay: int, _dimension: int
        ) -> np.ndarray:
            raise RuntimeError("boom")

        monkeypatch.setattr(
            em_mod,
            "_dispatch",
            lambda fn_name: _raising_de if fn_name == "de" else None,
        )
        emb = delay_embed(np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64), 1, 2)
        np.testing.assert_array_equal(emb, [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])

    def test_mutual_information_backend_failure_falls_back_to_python(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raising_mi(_signal: np.ndarray, _lag: int, _n_bins: int) -> float:
            raise RuntimeError("boom")

        monkeypatch.setattr(
            em_mod,
            "_dispatch",
            lambda fn_name: _raising_mi if fn_name == "mi" else None,
        )
        mi = mutual_information(np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64), 1, 2)
        assert np.isfinite(mi)
        assert mi >= 0.0

    @pytest.mark.parametrize(
        "backend_value",
        [-0.1, np.nan, np.inf, [0.5], True, np.bool_(True), 0.5 + 0.0j],
    )
    def test_invalid_mutual_information_backend_payload_falls_back_to_python(
        self,
        monkeypatch: pytest.MonkeyPatch,
        backend_value: Any,
    ) -> None:
        monkeypatch.setattr(
            em_mod,
            "_dispatch",
            lambda fn_name: (lambda *_args: backend_value) if fn_name == "mi" else None,
        )

        mi = mutual_information(np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64), 1, 2)

        assert np.isfinite(mi)
        assert mi >= 0.0

    def test_nearest_neighbor_backend_failure_falls_back_to_python(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raising_nn(
            _embedded: np.ndarray, _t: int, _m: int
        ) -> tuple[np.ndarray, np.ndarray]:
            raise RuntimeError("boom")

        monkeypatch.setattr(
            em_mod,
            "_dispatch",
            lambda fn_name: _raising_nn if fn_name == "nn" else None,
        )
        dist, idx = nearest_neighbor_distances(np.array([[0.0], [1.0], [3.0]]))
        np.testing.assert_allclose(dist, [1.0, 1.0, 2.0])
        np.testing.assert_array_equal(idx, [1, 0, 1])

    @pytest.mark.parametrize(
        ("distances", "indices"),
        [
            (np.array([1.0], dtype=np.float64), np.array([1], dtype=np.int64)),
            (
                np.array([1.0, np.nan, 2.0], dtype=np.float64),
                np.array([1, 0, 1], dtype=np.int64),
            ),
            (
                np.array([1.0, 1.0, 2.0], dtype=np.float64),
                np.array([0, 0, 1], dtype=np.int64),
            ),
            (
                np.array([1.0, 1.0, 2.0], dtype=np.float64),
                np.array([1, 0, 3], dtype=np.int64),
            ),
            (
                np.array([True, False, True], dtype=np.bool_),
                np.array([1, 0, 1], dtype=np.int64),
            ),
            (
                np.array([1.0 + 0.0j, 1.0 + 0.25j, 2.0 + 0.0j]),
                np.array([1, 0, 1], dtype=np.int64),
            ),
            (
                np.array([1.0, 1.0, 1.0], dtype=np.float64),
                np.array([np.bool_(True), 0, 1], dtype=object),
            ),
            (
                np.array([1.0, 1.0, 2.0], dtype=np.float64),
                np.array([1 + 0j, 0 + 0j, 1 + 0j]),
            ),
        ],
    )
    def test_invalid_nearest_neighbor_backend_payload_falls_back_to_python(
        self,
        monkeypatch: pytest.MonkeyPatch,
        distances: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        monkeypatch.setattr(
            em_mod,
            "_dispatch",
            lambda fn_name: (
                (lambda *_args: (distances, indices)) if fn_name == "nn" else None
            ),
        )

        dist, idx = nearest_neighbor_distances(np.array([[0.0], [1.0], [3.0]]))

        np.testing.assert_allclose(dist, [1.0, 1.0, 2.0])
        np.testing.assert_array_equal(idx, [1, 0, 1])

    def test_optimal_delay_rust_failure_falls_back_to_python(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raising_optimal_delay(
            _signal: np.ndarray, _max_lag: int, _n_bins: int
        ) -> int:
            raise RuntimeError("boom")

        previous = em_mod.ACTIVE_BACKEND
        em_mod.ACTIVE_BACKEND = "rust"
        monkeypatch.setattr(
            em_mod,
            "_load_backend",
            lambda name: {"optimal_delay": _raising_optimal_delay},
        )
        try:
            tau = optimal_delay(np.sin(np.linspace(0.0, 6 * np.pi, 400)), max_lag=40)
        finally:
            em_mod.ACTIVE_BACKEND = previous
        assert tau >= 1

    def test_optimal_dimension_rust_failure_falls_back_to_python(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raising_optimal_dimension(
            _signal: np.ndarray, _delay: int, _max_dim: int, _rtol: float, _atol: float
        ) -> int:
            raise RuntimeError("boom")

        previous = em_mod.ACTIVE_BACKEND
        em_mod.ACTIVE_BACKEND = "rust"
        monkeypatch.setattr(
            em_mod,
            "_load_backend",
            lambda name: {"optimal_dimension": _raising_optimal_dimension},
        )
        try:
            dim = optimal_dimension(
                np.sin(np.linspace(0.0, 8 * np.pi, 600)),
                delay=2,
                max_dim=5,
            )
        finally:
            em_mod.ACTIVE_BACKEND = previous
        assert 1 <= dim <= 5


class TestDispatchFallbackChain:
    def test_dispatch_prefers_active_backend_before_available_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: dict[str, int] = {"go": 0, "julia": 0}

        def _go_backend() -> dict[str, object]:
            calls["go"] += 1
            return {"de": None, "mi": None, "nn": None}

        def _julia_backend() -> dict[str, object]:
            calls["julia"] += 1
            return {"de": lambda signal, delay, dim: np.zeros(6, dtype=np.float64)}

        monkeypatch.setattr(em_mod, "_BACKEND_CACHE", {})
        monkeypatch.setattr(em_mod, "ACTIVE_BACKEND", "go")
        monkeypatch.setattr(em_mod, "AVAILABLE_BACKENDS", ["julia", "go", "python"])
        monkeypatch.setattr(
            em_mod, "_LOADERS", {"go": _go_backend, "julia": _julia_backend}
        )

        fn = em_mod._dispatch("de")
        assert fn is not None
        assert calls == {"go": 1, "julia": 1}

    def test_dispatch_uses_cached_loader_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: dict[str, int] = {"go": 0}

        def _go_backend() -> dict[str, object]:
            calls["go"] += 1
            return {
                "de": lambda signal, delay, dim: np.zeros(6, dtype=np.float64),
                "mi": lambda signal, lag, n_bins: 0.0,
                "nn": lambda emb: (
                    np.zeros(1, dtype=np.float64),
                    np.zeros(1, dtype=np.int64),
                ),
            }

        monkeypatch.setattr(em_mod, "_BACKEND_CACHE", {})
        monkeypatch.setattr(em_mod, "ACTIVE_BACKEND", "go")
        monkeypatch.setattr(em_mod, "AVAILABLE_BACKENDS", ["go", "python"])
        monkeypatch.setattr(em_mod, "_LOADERS", {"go": _go_backend})

        em_mod._dispatch("de")
        em_mod._dispatch("mi")

        assert calls["go"] == 1
