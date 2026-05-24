# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Fractal dimension tests

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor import dimension as dim_mod
from scpn_phase_orchestrator.monitor.dimension import (
    CorrelationDimensionResult,
    correlation_dimension,
    correlation_integral,
    kaplan_yorke_dimension,
)


class TestCorrelationIntegral:
    def test_increases_with_epsilon(self):
        """C(ε) is monotonically non-decreasing."""
        t = np.linspace(0, 10 * np.pi, 500)
        traj = np.column_stack([np.sin(t), np.cos(t)])
        epsilons = np.logspace(-1, 0.5, 15)
        C = correlation_integral(traj, epsilons)
        for i in range(len(C) - 1):
            assert C[i] <= C[i + 1] + 1e-10

    def test_all_same_point(self):
        """All points identical → C(ε)=1 for any ε>0."""
        traj = np.ones((50, 2))
        C = correlation_integral(traj, np.array([0.01, 0.1, 1.0]))
        np.testing.assert_array_equal(C, [1.0, 1.0, 1.0])

    def test_subsampling(self):
        """Large trajectory triggers pair subsampling."""
        rng = np.random.default_rng(42)
        traj = rng.normal(0, 1, (1000, 3))
        epsilons = np.logspace(-1, 1, 10)
        C = correlation_integral(traj, epsilons, max_pairs=5000)
        assert C[-1] > 0

    def test_empty_subsample_returns_zero_for_all_thresholds(self, monkeypatch):
        """If deterministic subsampling yields no distinct pairs, C(eps)=0."""
        previous_backend = dim_mod.ACTIVE_BACKEND
        dim_mod.ACTIVE_BACKEND = "python"
        monkeypatch.setattr(
            dim_mod,
            "_prepare_pair_indices",
            lambda _t, _max_pairs, _seed: (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
            ),
        )
        traj = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
        epsilons = np.array([0.1, 1.0, 10.0], dtype=np.float64)
        try:
            C = correlation_integral(traj, epsilons, max_pairs=2, seed=4)
        finally:
            dim_mod.ACTIVE_BACKEND = previous_backend

        np.testing.assert_array_equal(C, np.zeros_like(epsilons))

    def test_one_dimensional_trajectory_is_time_axis(self):
        """A 1D signal is interpreted as T scalar samples, not one T-D point."""
        C = correlation_integral([0.0, 1.0, 2.0], np.array([0.5, 1.5]))

        np.testing.assert_allclose(C, [0.0, 2.0 / 3.0])

    def test_prepare_pair_indices_filters_equal_indices(self):
        """Subsampling never allows i == j diagonal pairs."""
        idx_i, idx_j = dim_mod._prepare_pair_indices(total_t=20, max_pairs=10, seed=1)
        assert len(idx_i) == len(idx_j)
        assert np.all(idx_i != idx_j)
        assert idx_i.dtype == np.int64
        assert idx_j.dtype == np.int64

    @pytest.mark.parametrize(
        "trajectory",
        [
            np.array([0.0, np.nan], dtype=np.float64),
            np.array([[0.0], [np.inf]], dtype=np.float64),
            np.zeros((2, 3, 4), dtype=np.float64),
            np.array([0.0, True], dtype=object),
            [["not-a-point"]],
        ],
    )
    def test_rejects_invalid_trajectory(self, trajectory: Any) -> None:
        with pytest.raises(ValueError, match="trajectory"):
            correlation_integral(trajectory, np.array([1.0]))

    @pytest.mark.parametrize(
        "epsilons",
        [
            np.array([np.nan], dtype=np.float64),
            np.array([np.inf], dtype=np.float64),
            np.array([-0.1], dtype=np.float64),
            np.zeros((1, 1), dtype=np.float64),
            np.array([0.1, True], dtype=object),
            ["not-epsilon"],
        ],
    )
    def test_rejects_invalid_epsilons(self, epsilons: Any) -> None:
        with pytest.raises(ValueError, match="epsilons"):
            correlation_integral(np.zeros((3, 1)), epsilons)

    @pytest.mark.parametrize("max_pairs", [False, 0, -1, 1.5, "10"])
    def test_rejects_invalid_max_pairs(self, max_pairs: Any) -> None:
        with pytest.raises(ValueError, match="max_pairs"):
            correlation_integral(np.zeros((3, 1)), np.array([1.0]), max_pairs=max_pairs)

    @pytest.mark.parametrize("seed", [False, -1, 1.5, "42"])
    def test_rejects_invalid_seed(self, seed: Any) -> None:
        with pytest.raises(ValueError, match="seed"):
            correlation_integral(np.zeros((3, 1)), np.array([1.0]), seed=seed)


class TestCorrelationDimension:
    def test_circle_dimension(self):
        """Circle (1D manifold) → D2 ≈ 1."""
        t = np.linspace(0, 20 * np.pi, 2000)
        traj = np.column_stack([np.sin(t), np.cos(t)])
        result = correlation_dimension(traj)
        assert isinstance(result, CorrelationDimensionResult)
        # D2 for a circle should be ~1.0
        assert 0.5 < result.D2 < 1.8

    def test_plane_filling(self):
        """2D Gaussian cloud → D2 ≈ 2."""
        rng = np.random.default_rng(42)
        traj = rng.normal(0, 1, (2000, 2))
        result = correlation_dimension(traj)
        assert 1.5 < result.D2 < 2.5

    def test_constant_trajectory(self):
        """Single point → D2 = 0."""
        traj = np.ones((100, 3))
        result = correlation_dimension(traj)
        assert result.D2 == 0.0

    def test_result_fields(self):
        t = np.linspace(0, 10 * np.pi, 500)
        traj = np.column_stack([np.sin(t), np.cos(t)])
        result = correlation_dimension(traj, n_epsilons=20)
        assert len(result.epsilons) == 20
        assert len(result.C_eps) == 20
        assert result.scaling_range[0] <= result.scaling_range[1]

    @pytest.mark.parametrize("n_epsilons", [False, 1, 0, -1, 2.5, "10"])
    def test_rejects_invalid_n_epsilons(self, n_epsilons: Any) -> None:
        with pytest.raises(ValueError, match="n_epsilons"):
            correlation_dimension(np.zeros((4, 1)), n_epsilons=n_epsilons)


class TestKaplanYorkeDimension:
    def test_stable_fixed_point(self):
        """All negative exponents → D_KY = 0."""
        le = np.array([-0.5, -1.0, -2.0])
        assert kaplan_yorke_dimension(le) == 0.0

    def test_limit_cycle(self):
        """One zero + one negative → D_KY = 1."""
        le = np.array([0.0, -1.0])
        d = kaplan_yorke_dimension(le)
        assert abs(d - 1.0) < 1e-10

    def test_lorenz_like(self):
        """Lorenz-like spectrum [+0.9, 0, -14.6] → D_KY ≈ 2.06."""
        le = np.array([0.9, 0.0, -14.6])
        d = kaplan_yorke_dimension(le)
        assert 2.0 < d < 2.1

    def test_hyperchaos(self):
        """Two positive exponents."""
        le = np.array([0.5, 0.2, -0.1, -1.5])
        d = kaplan_yorke_dimension(le)
        # Sum of first 3: 0.5+0.2-0.1 = 0.6 > 0
        # j=2, D_KY = 3 + 0.6/1.5 = 3.4
        assert abs(d - 3.4) < 1e-10

    def test_all_positive(self):
        """All positive → D_KY = N (fills all dimensions)."""
        le = np.array([0.5, 0.3, 0.1])
        d = kaplan_yorke_dimension(le)
        assert d == 3.0

    def test_unsorted_input(self):
        """Should handle unsorted input."""
        le = np.array([-1.0, 0.5, -0.3])
        d = kaplan_yorke_dimension(le)
        # Sorted: [0.5, -0.3, -1.0] → cumsum=[0.5, 0.2, -0.8] → j=1
        # D_KY = 2 + 0.2/1.0 = 2.2
        assert abs(d - 2.2) < 1e-10

    def test_zero_denominator(self):
        """λ_{j+1} = 0 → D_KY = j + 1."""
        le = np.array([1.0, 0.0, -1.0])
        d = kaplan_yorke_dimension(le)
        # cumsum = [1.0, 1.0, 0.0] → j=2, but j+1=3 >= len → returns 3
        # Actually: j=2 (0-indexed cumsum[2]=0.0 ≥ 0), j+1=3 ≥ 3 → returns 3.0
        assert d == 3.0

    @pytest.mark.parametrize(
        "lyapunov_exponents",
        [
            np.array([0.1, np.nan], dtype=np.float64),
            np.array([0.1, np.inf], dtype=np.float64),
            np.zeros((2, 2), dtype=np.float64),
            np.array([0.1, True], dtype=object),
            ["not-an-exponent"],
        ],
    )
    def test_rejects_invalid_spectrum(self, lyapunov_exponents: Any) -> None:
        with pytest.raises(ValueError, match="lyapunov_exponents"):
            kaplan_yorke_dimension(lyapunov_exponents)


class TestCorrelationDimensionResult:
    def test_normalises_valid_record(self) -> None:
        result = CorrelationDimensionResult(
            D2=np.float64(1.25),
            epsilons=[0.1, 1.0],
            C_eps=[0.25, 0.75],
            slope=[1.25],
            scaling_range=(0.1, 1.0),
        )

        assert result.D2 == 1.25
        assert isinstance(result.epsilons, np.ndarray)
        assert isinstance(result.C_eps, np.ndarray)
        assert isinstance(result.slope, np.ndarray)
        assert result.scaling_range == (0.1, 1.0)

    @pytest.mark.parametrize("D2", [np.nan, np.inf, -0.1, True, "1.0"])
    def test_rejects_invalid_dimension(self, D2: Any) -> None:
        with pytest.raises(ValueError, match="D2"):
            CorrelationDimensionResult(
                D2=D2,
                epsilons=[0.1, 1.0],
                C_eps=[0.25, 0.75],
                slope=[1.25],
                scaling_range=(0.1, 1.0),
            )

    @pytest.mark.parametrize(
        "epsilons",
        [
            [0.1, True],
            [np.nan, 1.0],
            [[0.1, 1.0]],
        ],
    )
    def test_rejects_invalid_epsilons(self, epsilons: Any) -> None:
        with pytest.raises(ValueError, match="epsilons"):
            CorrelationDimensionResult(
                D2=1.0,
                epsilons=epsilons,
                C_eps=[0.25, 0.75],
                slope=[1.25],
                scaling_range=(0.1, 1.0),
            )

    @pytest.mark.parametrize(
        "C_eps",
        [
            [0.25],
            [0.25, np.nan],
            [0.25, 1.1],
            [0.75, 0.25],
        ],
    )
    def test_rejects_invalid_correlation_integral(self, C_eps: Any) -> None:
        with pytest.raises(ValueError, match="C_eps"):
            CorrelationDimensionResult(
                D2=1.0,
                epsilons=[0.1, 1.0],
                C_eps=C_eps,
                slope=[1.25],
                scaling_range=(0.1, 1.0),
            )

    @pytest.mark.parametrize("slope", [[1.0, 2.0, 3.0], [np.nan], [[1.0]]])
    def test_rejects_invalid_slope(self, slope: Any) -> None:
        with pytest.raises(ValueError, match="slope"):
            CorrelationDimensionResult(
                D2=1.0,
                epsilons=[0.1, 1.0],
                C_eps=[0.25, 0.75],
                slope=slope,
                scaling_range=(0.1, 1.0),
            )

    @pytest.mark.parametrize(
        "scaling_range",
        [
            (1.0, 0.1),
            (0.1, np.nan),
            (-0.1, 1.0),
            (0.1,),
            [0.1, 1.0],
        ],
    )
    def test_rejects_invalid_scaling_range(self, scaling_range: Any) -> None:
        with pytest.raises(ValueError, match="scaling_range"):
            CorrelationDimensionResult(
                D2=1.0,
                epsilons=[0.1, 1.0],
                C_eps=[0.25, 0.75],
                slope=[1.25],
                scaling_range=scaling_range,
            )


class TestBackendDispatch:
    def test_dispatch_returns_none_when_active_loader_fails(self, monkeypatch):
        previous_backend = dim_mod.ACTIVE_BACKEND
        previous_available = list(dim_mod.AVAILABLE_BACKENDS)
        previous_loader = dim_mod._LOADERS["go"]
        dim_mod.ACTIVE_BACKEND = "go"
        dim_mod.AVAILABLE_BACKENDS = ["go", "python"]
        dim_mod._BACKEND_FN_CACHE.clear()
        monkeypatch.setitem(
            dim_mod._LOADERS,
            "go",
            lambda: (_ for _ in ()).throw(ImportError("go backend unavailable")),
        )
        try:
            backend_fn = dim_mod._dispatch("ci")
        finally:
            dim_mod.ACTIVE_BACKEND = previous_backend
            dim_mod.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(dim_mod._LOADERS, "go", previous_loader)
            dim_mod._BACKEND_FN_CACHE.clear()

        assert backend_fn is None

    def test_dispatch_returns_none_for_missing_backend_function(self, monkeypatch):
        previous_backend = dim_mod.ACTIVE_BACKEND
        previous_available = list(dim_mod.AVAILABLE_BACKENDS)
        previous_loader = dim_mod._LOADERS["go"]
        dim_mod.ACTIVE_BACKEND = "go"
        dim_mod.AVAILABLE_BACKENDS = ["go", "python"]
        dim_mod._BACKEND_FN_CACHE.clear()
        monkeypatch.setitem(dim_mod._LOADERS, "go", lambda: {"ky": lambda _x: 1.0})
        try:
            backend_fn = dim_mod._dispatch("ci")
        finally:
            dim_mod.ACTIVE_BACKEND = previous_backend
            dim_mod.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(dim_mod._LOADERS, "go", previous_loader)
            dim_mod._BACKEND_FN_CACHE.clear()

        assert backend_fn is None

    @pytest.mark.parametrize(
        "backend_output",
        [
            np.array([0.5], dtype=np.float64),
            np.array([0.5, np.nan], dtype=np.float64),
            np.array([0.5, 1.1], dtype=np.float64),
            np.array([0.5, 0.4], dtype=np.float64),
        ],
    )
    def test_invalid_correlation_integral_backend_payload_falls_back(
        self,
        monkeypatch,
        backend_output: np.ndarray,
    ) -> None:
        previous_backend = dim_mod.ACTIVE_BACKEND
        previous_available = list(dim_mod.AVAILABLE_BACKENDS)
        previous_loader = dim_mod._LOADERS["go"]

        def fake_ci(*_args: object, **_kwargs: object) -> np.ndarray:
            return backend_output

        dim_mod.ACTIVE_BACKEND = "go"
        dim_mod.AVAILABLE_BACKENDS = ["go", "python"]
        dim_mod._BACKEND_FN_CACHE.clear()
        monkeypatch.setitem(dim_mod._LOADERS, "go", lambda: {"ci": fake_ci})
        try:
            C = correlation_integral(
                np.array([[0.0], [1.0], [2.0]], dtype=np.float64),
                np.array([0.5, 1.5], dtype=np.float64),
                max_pairs=10,
            )
        finally:
            dim_mod.ACTIVE_BACKEND = previous_backend
            dim_mod.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(dim_mod._LOADERS, "go", previous_loader)
            dim_mod._BACKEND_FN_CACHE.clear()

        np.testing.assert_allclose(C, [0.0, 2.0 / 3.0])

    @pytest.mark.parametrize("backend_value", [-0.1, np.nan, np.inf, 4.0])
    def test_invalid_kaplan_yorke_backend_payload_falls_back(
        self,
        monkeypatch,
        backend_value: float,
    ) -> None:
        previous_backend = dim_mod.ACTIVE_BACKEND
        previous_available = list(dim_mod.AVAILABLE_BACKENDS)
        previous_loader = dim_mod._LOADERS["go"]
        le = np.array([0.5, -1.0, -2.0], dtype=np.float64)

        def fake_ky(*_args: object, **_kwargs: object) -> float:
            return float(backend_value)

        dim_mod.ACTIVE_BACKEND = "python"
        expected = kaplan_yorke_dimension(le)
        dim_mod.ACTIVE_BACKEND = "go"
        dim_mod.AVAILABLE_BACKENDS = ["go", "python"]
        dim_mod._BACKEND_FN_CACHE.clear()
        monkeypatch.setitem(dim_mod._LOADERS, "go", lambda: {"ky": fake_ky})
        try:
            got = kaplan_yorke_dimension(le)
        finally:
            dim_mod.ACTIVE_BACKEND = previous_backend
            dim_mod.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(dim_mod._LOADERS, "go", previous_loader)
            dim_mod._BACKEND_FN_CACHE.clear()

        assert got == expected

    def test_dispatch_falls_through_to_next_available_backend(self, monkeypatch):
        previous_backend = dim_mod.ACTIVE_BACKEND
        previous_available = list(dim_mod.AVAILABLE_BACKENDS)
        previous_go = dim_mod._LOADERS["go"]
        previous_rust = dim_mod._LOADERS["rust"]
        dim_mod.ACTIVE_BACKEND = "go"
        dim_mod.AVAILABLE_BACKENDS = ["go", "rust", "python"]
        dim_mod._BACKEND_FN_CACHE.clear()

        def fake_ci(*_args):
            return np.array([1.0], dtype=np.float64)

        monkeypatch.setitem(
            dim_mod._LOADERS,
            "go",
            lambda: (_ for _ in ()).throw(ImportError("go backend unavailable")),
        )
        monkeypatch.setitem(dim_mod._LOADERS, "rust", lambda: {"ci": fake_ci})
        try:
            backend_fn = dim_mod._dispatch("ci")
        finally:
            dim_mod.ACTIVE_BACKEND = previous_backend
            dim_mod.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(dim_mod._LOADERS, "go", previous_go)
            monkeypatch.setitem(dim_mod._LOADERS, "rust", previous_rust)
            dim_mod._BACKEND_FN_CACHE.clear()

        assert backend_fn is fake_ci

    def test_rust_loader_exposes_ci_and_ky_functions(self, monkeypatch):
        def fake_ci(*_args):
            return np.array([0.0], dtype=np.float64)

        def fake_ky(_exponents):
            return 0.0

        monkeypatch.setitem(
            sys.modules,
            "spo_kernel",
            types.SimpleNamespace(
                correlation_integral_rust=fake_ci,
                kaplan_yorke_dimension_rust=fake_ky,
            ),
        )

        loaded = dim_mod._load_rust_fns()

        assert loaded == {"ci": fake_ci, "ky": fake_ky}

    def test_rust_correlation_dispatch_sorts_epsilons_and_uses_contiguous_buffers(
        self,
        monkeypatch,
    ):
        calls: list[tuple[int, int, int, int]] = []

        def fake_ci(traj_flat, t, d, eps_sorted, max_pairs, seed):
            calls.append((int(t), int(d), int(max_pairs), int(seed)))
            assert traj_flat.flags.c_contiguous
            assert traj_flat.dtype == np.float64
            np.testing.assert_array_equal(eps_sorted, np.array([0.1, 1.0, 2.0]))
            return np.array([0.25, 0.5, 1.0], dtype=np.float64)

        previous_backend = dim_mod.ACTIVE_BACKEND
        previous_loader = dim_mod._LOADERS["rust"]
        dim_mod.ACTIVE_BACKEND = "rust"
        monkeypatch.setitem(dim_mod._LOADERS, "rust", lambda: {"ci": fake_ci})
        try:
            traj = np.array([[0.0, 1.0], [1.0, 1.5], [2.0, 3.0]])
            eps = np.array([2.0, 0.1, 1.0])
            C = correlation_integral(traj, eps, max_pairs=17, seed=9)
        finally:
            dim_mod.ACTIVE_BACKEND = previous_backend
            monkeypatch.setitem(dim_mod._LOADERS, "rust", previous_loader)
            dim_mod._BACKEND_FN_CACHE.clear()

        np.testing.assert_array_equal(C, np.array([0.25, 0.5, 1.0]))
        assert calls == [(3, 2, 17, 9)]

    def test_non_rust_dispatch_uses_prepared_indices(
        self,
        monkeypatch,
    ) -> None:
        calls: list[tuple[int, int, np.ndarray, np.ndarray]] = []
        idx_i = np.array([0, 2], dtype=np.int64)
        idx_j = np.array([1, 3], dtype=np.int64)

        def fake_ci(traj_flat, t, d, i, j, eps_sorted):
            calls.append((int(t), int(d), np.array(i), np.array(j)))
            assert traj_flat.dtype == np.float64
            assert traj_flat.flags.c_contiguous
            np.testing.assert_array_equal(eps_sorted, np.array([0.1, 0.5, 1.0]))
            return np.array([0.0, 0.5, 1.0], dtype=np.float64)

        previous_backend = dim_mod.ACTIVE_BACKEND
        previous_loader = dim_mod._LOADERS["go"]
        previous_prepare = dim_mod._prepare_pair_indices
        dim_mod.ACTIVE_BACKEND = "go"
        monkeypatch.setitem(dim_mod._LOADERS, "go", lambda: {"ci": fake_ci})
        monkeypatch.setattr(
            dim_mod,
            "_prepare_pair_indices",
            lambda _t, _max_pairs, _seed: (idx_i, idx_j),
        )
        try:
            traj = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
            eps = np.array([1.0, 0.1, 0.5])
            C = correlation_integral(traj, eps, max_pairs=7, seed=99)
        finally:
            dim_mod.ACTIVE_BACKEND = previous_backend
            monkeypatch.setitem(dim_mod._LOADERS, "go", previous_loader)
            monkeypatch.setattr(dim_mod, "_prepare_pair_indices", previous_prepare)
            dim_mod._BACKEND_FN_CACHE.clear()

        np.testing.assert_array_equal(C, np.array([0.0, 0.5, 1.0]))
        assert len(calls) == 1
        assert calls[0][0] == 4
        assert calls[0][1] == 2
        np.testing.assert_array_equal(calls[0][2], idx_i)
        np.testing.assert_array_equal(calls[0][3], idx_j)


class TestCorrelationDimensionEdgeCases:
    def test_few_valid_c_eps(self):
        """Trajectory with very few valid C(ε) → D2=0."""
        # Two points very close together → only small ε gives C>0
        traj = np.array([[0.0, 0.0], [1e-15, 1e-15], [0.0, 0.0]])
        result = correlation_dimension(traj, n_epsilons=5)
        assert result.D2 >= 0.0

    def test_single_point_trajectory(self):
        """T=1 → diameter=0 → D2=0."""
        traj = np.array([[1.0, 2.0]])
        result = correlation_dimension(traj)
        assert result.D2 == 0.0

    def test_short_slope_window(self):
        """Very few epsilons → window < 2 branch."""
        traj = np.array([[0.0], [1.0], [2.0], [3.0]])
        result = correlation_dimension(traj, n_epsilons=3)
        assert isinstance(result.D2, float)


class TestDimensionPipelineWiring:
    """Pipeline: engine trajectory → embed → correlation dimension."""

    def test_engine_trajectory_to_correlation_dimension(self):
        """UPDEEngine → trajectory → delay_embed → D2."""
        from scpn_phase_orchestrator.monitor.embedding import delay_embed
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        n = 4
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.array([1.0, 1.5, 2.0, 0.5])
        knm = 0.5 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))

        trajectory = []
        for _ in range(500):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
            trajectory.append(float(phases[0]))

        emb = delay_embed(np.array(trajectory), delay=5, dimension=3)
        result = correlation_dimension(emb)
        assert isinstance(result, CorrelationDimensionResult)
        assert result.D2 >= 0.0
