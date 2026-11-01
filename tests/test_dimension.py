# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Fractal dimension tests

from __future__ import annotations

import inspect
import math
import sys
from importlib import import_module
from pathlib import Path
from types import FrameType
from typing import NoReturn, TypeAlias, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor import dimension as dim_mod
from scpn_phase_orchestrator.monitor.dimension import (
    CorrelationDimensionResult,
    correlation_dimension,
    correlation_integral,
    kaplan_yorke_dimension,
)
from tests.test_dimension_julia_absence import run_source_profile


def _call(fn: object, *args: object, **kwargs: object) -> object:
    """Call ``fn`` with arguments mypy cannot check: invalid or dynamic kwargs."""
    assert callable(fn)
    return fn(*args, **kwargs)


FloatArray: TypeAlias = NDArray[np.float64]


class _ArrayCoercionFailure:
    """Array-like test double that refuses NumPy coercion."""

    def __array__(
        self, dtype: object | None = None, copy: object | None = None
    ) -> NoReturn:
        raise TypeError("array coercion disabled")


class TestCorrelationIntegral:
    def test_increases_with_epsilon(self) -> None:
        """C(ε) is monotonically non-decreasing."""
        t = np.linspace(0, 10 * np.pi, 500)
        traj = np.column_stack([np.sin(t), np.cos(t)])
        epsilons = np.logspace(-1, 0.5, 15)
        C = correlation_integral(traj, epsilons)
        for i in range(len(C) - 1):
            assert C[i] <= C[i + 1] + 1e-10

    def test_all_same_point(self) -> None:
        """All points identical → C(ε)=1 for any ε>0."""
        traj = np.ones((50, 2))
        C = correlation_integral(traj, np.array([0.01, 0.1, 1.0]))
        np.testing.assert_array_equal(C, [1.0, 1.0, 1.0])

    def test_subsampling(self) -> None:
        """Large trajectory triggers pair subsampling."""
        rng = np.random.default_rng(42)
        traj = rng.normal(0, 1, (1000, 3))
        epsilons = np.logspace(-1, 1, 10)
        C = correlation_integral(traj, epsilons, max_pairs=5000)
        assert C[-1] > 0

    def test_empty_subsample_returns_zero_for_all_thresholds(self) -> None:
        """If deterministic subsampling yields no distinct pairs, C(eps)=0."""
        # Three potential pairs exceed the quota; the draw (1, 1) is discarded.
        rng = np.random.default_rng(1)
        assert rng.integers(0, 3, 1)[0] == rng.integers(0, 3, 1)[0]
        traj = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
        epsilons = np.array([0.1, 1.0, 10.0], dtype=np.float64)
        C = correlation_integral(traj, epsilons, max_pairs=1, seed=1, backend="python")

        np.testing.assert_array_equal(C, np.zeros_like(epsilons))

    def test_one_dimensional_trajectory_is_time_axis(self) -> None:
        """A 1D signal is interpreted as T scalar samples, not one T-D point."""
        C = correlation_integral([0.0, 1.0, 2.0], np.array([0.5, 1.5]))

        np.testing.assert_allclose(C, [0.0, 2.0 / 3.0])

    def test_single_sample_trajectory_returns_zero_for_all_thresholds(self) -> None:
        """A single sample has no distinct pairs, so C(eps)=0."""
        C = correlation_integral(np.array([[1.0, 2.0]]), np.array([0.1, 1.0]))

        np.testing.assert_array_equal(C, np.array([0.0, 0.0]))

    def test_prepare_pair_indices_filters_equal_indices(self) -> None:
        """Subsampling never allows i == j diagonal pairs."""
        prepared = dim_mod._prepare_pair_indices(total_t=20, max_pairs=10, seed=1)
        assert prepared is not None
        idx_i, idx_j = prepared
        assert len(idx_i) == len(idx_j)
        assert np.all(idx_i != idx_j)
        assert idx_i.dtype == np.int64
        assert idx_j.dtype == np.int64

    def test_prepare_pair_indices_returns_none_for_single_sample(self) -> None:
        """One sample cannot form off-diagonal pairs."""
        assert dim_mod._prepare_pair_indices(total_t=1, max_pairs=10, seed=1) is None

    @pytest.mark.parametrize(
        "trajectory",
        [
            np.array([0.0, np.nan], dtype=np.float64),
            np.array([[0.0], [np.inf]], dtype=np.float64),
            np.zeros((2, 3, 4), dtype=np.float64),
            np.array([0.0, True], dtype=object),
            np.array([0.0 + 1.0j, 1.0 + 0.0j], dtype=np.complex128),
            np.array([0.0 + 1.0j, 1.0], dtype=object),
            np.array(["0.0", "1.0", "2.0"]),
            np.array([0.0, "1.0", 2.0], dtype=object),
            [["not-a-point"]],
        ],
    )
    def test_rejects_invalid_trajectory(self, trajectory: object) -> None:
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
            np.array([0.1 + 1.0j], dtype=np.complex128),
            np.array([0.1 + 1.0j], dtype=object),
            np.array(["0.1", "1.0"]),
            np.array([0.1, "1.0"], dtype=object),
            ["not-epsilon"],
        ],
    )
    def test_rejects_invalid_epsilons(self, epsilons: object) -> None:
        with pytest.raises(ValueError, match="epsilons"):
            correlation_integral(np.zeros((3, 1)), epsilons)

    @pytest.mark.parametrize("max_pairs", [False, 0, -1, 1.5, "10"])
    def test_rejects_invalid_max_pairs(self, max_pairs: object) -> None:
        with pytest.raises(ValueError, match="max_pairs"):
            correlation_integral(np.zeros((3, 1)), np.array([1.0]), max_pairs=max_pairs)

    @pytest.mark.parametrize("seed", [False, -1, 1.5, "42"])
    def test_rejects_invalid_seed(self, seed: object) -> None:
        with pytest.raises(ValueError, match="seed"):
            correlation_integral(np.zeros((3, 1)), np.array([1.0]), seed=seed)


class TestCorrelationDimension:
    def test_circle_dimension(self) -> None:
        """Circle (1D manifold) → D2 ≈ 1."""
        t = np.linspace(0, 20 * np.pi, 2000)
        traj = np.column_stack([np.sin(t), np.cos(t)])
        result = correlation_dimension(traj)
        assert isinstance(result, CorrelationDimensionResult)
        # D2 for a circle should be ~1.0
        assert 0.5 < result.D2 < 1.8

    def test_plane_filling(self) -> None:
        """2D Gaussian cloud → D2 ≈ 2."""
        rng = np.random.default_rng(42)
        traj = rng.normal(0, 1, (2000, 2))
        result = correlation_dimension(traj)
        assert 1.5 < result.D2 < 2.5

    def test_constant_trajectory(self) -> None:
        """Single point → D2 = 0."""
        traj = np.ones((100, 3))
        result = correlation_dimension(traj)
        assert result.D2 == 0.0

    def test_result_fields(self) -> None:
        t = np.linspace(0, 10 * np.pi, 500)
        traj = np.column_stack([np.sin(t), np.cos(t)])
        result = correlation_dimension(traj, n_epsilons=20)
        assert len(result.epsilons) == 20
        assert len(result.C_eps) == 20
        assert len(result.slope) == len(result.epsilons) - 1
        assert result.scaling_range[0] <= result.scaling_range[1]

    def test_sparse_positive_correlation_grid_preserves_slope_contract(self) -> None:
        """Sparse early ε values keep the public result arrays aligned."""
        rng = np.random.default_rng(97)
        traj = rng.standard_normal((100, 2))
        result = correlation_dimension(traj)

        assert result.D2 >= 0.0
        assert len(result.slope) == len(result.epsilons) - 1
        assert np.all(np.isfinite(result.slope))

    @pytest.mark.parametrize("n_epsilons", [False, 1, 0, -1, 2.5, "10"])
    def test_rejects_invalid_n_epsilons(self, n_epsilons: object) -> None:
        with pytest.raises(ValueError, match="n_epsilons"):
            correlation_dimension(np.zeros((4, 1)), n_epsilons=n_epsilons)


class TestKaplanYorkeDimension:
    def test_stable_fixed_point(self) -> None:
        """All negative exponents → D_KY = 0."""
        le = np.array([-0.5, -1.0, -2.0])
        assert kaplan_yorke_dimension(le) == 0.0

    def test_limit_cycle(self) -> None:
        """One zero + one negative → D_KY = 1."""
        le = np.array([0.0, -1.0])
        d = kaplan_yorke_dimension(le)
        assert abs(d - 1.0) < 1e-10

    def test_lorenz_like(self) -> None:
        """Lorenz-like spectrum [+0.9, 0, -14.6] → D_KY ≈ 2.06."""
        le = np.array([0.9, 0.0, -14.6])
        d = kaplan_yorke_dimension(le)
        assert 2.0 < d < 2.1

    def test_hyperchaos(self) -> None:
        """Two positive exponents."""
        le = np.array([0.5, 0.2, -0.1, -1.5])
        d = kaplan_yorke_dimension(le)
        # Sum of first 3: 0.5+0.2-0.1 = 0.6 > 0
        # j=2, D_KY = 3 + 0.6/1.5 = 3.4
        assert abs(d - 3.4) < 1e-10

    def test_all_positive(self) -> None:
        """All positive → D_KY = N (fills all dimensions)."""
        le = np.array([0.5, 0.3, 0.1])
        d = kaplan_yorke_dimension(le)
        assert d == 3.0

    def test_unsorted_input(self) -> None:
        """Should handle unsorted input."""
        le = np.array([-1.0, 0.5, -0.3])
        d = kaplan_yorke_dimension(le)
        # Sorted: [0.5, -0.3, -1.0] → cumsum=[0.5, 0.2, -0.8] → j=1
        # D_KY = 2 + 0.2/1.0 = 2.2
        assert abs(d - 2.2) < 1e-10

    def test_zero_denominator(self) -> None:
        """λ_{j+1} = 0 → D_KY = j + 1."""
        le = np.array([1.0, 0.0, -1.0])
        d = kaplan_yorke_dimension(le)
        # cumsum = [1.0, 1.0, 0.0] → j=2, but j+1=3 >= len → returns 3
        # Actually: j=2 (0-indexed cumsum[2]=0.0 ≥ 0), j+1=3 ≥ 3 → returns 3.0
        assert d == 3.0

    def test_empty_spectrum_returns_zero_dimension(self) -> None:
        """An empty Lyapunov spectrum has zero Kaplan-Yorke dimension."""
        assert kaplan_yorke_dimension(np.array([], dtype=np.float64)) == 0.0
        assert dim_mod._kaplan_yorke_exact_reference(
            np.array([], dtype=np.float64)
        ) == pytest.approx(0.0)

    @pytest.mark.parametrize(
        "lyapunov_exponents",
        [
            np.array([0.1, np.nan], dtype=np.float64),
            np.array([0.1, np.inf], dtype=np.float64),
            np.zeros((2, 2), dtype=np.float64),
            np.array([0.1, True], dtype=object),
            np.array([0.1 + 1.0j, -0.2 + 0.0j], dtype=np.complex128),
            np.array([0.1 + 1.0j, -0.2], dtype=object),
            np.array(["0.1", "-0.2"]),
            np.array([0.1, "-0.2"], dtype=object),
            ["not-an-exponent"],
        ],
    )
    def test_rejects_invalid_spectrum(self, lyapunov_exponents: object) -> None:
        with pytest.raises(ValueError, match="lyapunov_exponents"):
            kaplan_yorke_dimension(lyapunov_exponents)


class TestCorrelationDimensionResult:
    def test_normalises_valid_record(self) -> None:
        result = _call(
            CorrelationDimensionResult,
            D2=np.float64(1.25),
            epsilons=[0.1, 1.0],
            C_eps=[0.25, 0.75],
            slope=[1.25],
            scaling_range=(0.1, 1.0),
        )

        assert isinstance(result, CorrelationDimensionResult)
        assert result.D2 == 1.25
        assert isinstance(result.epsilons, np.ndarray)
        assert isinstance(result.C_eps, np.ndarray)
        assert isinstance(result.slope, np.ndarray)
        assert result.scaling_range == (0.1, 1.0)

    @pytest.mark.parametrize("D2", [np.nan, np.inf, -0.1, True, "1.0"])
    def test_rejects_invalid_dimension(self, D2: object) -> None:
        with pytest.raises(ValueError, match="D2"):
            _call(
                CorrelationDimensionResult,
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
            ["0.1", "1.0"],
            np.array([0.1, "1.0"], dtype=object),
            [np.nan, 1.0],
            [0.1 + 1.0j, 1.0 + 0.0j],
            np.array([0.1 + 1.0j, 1.0], dtype=object),
            [[0.1, 1.0]],
        ],
    )
    def test_rejects_invalid_epsilons(self, epsilons: object) -> None:
        with pytest.raises(ValueError, match="epsilons"):
            _call(
                CorrelationDimensionResult,
                D2=1.0,
                epsilons=epsilons,
                C_eps=[0.25, 0.75],
                slope=[1.25],
                scaling_range=(0.1, 1.0),
            )

    @pytest.mark.parametrize("epsilons", [[0.0, 1.0], [0.1, 0.1]])
    def test_rejects_non_positive_or_duplicate_log_scale_epsilons(
        self,
        epsilons: list[float],
    ) -> None:
        with pytest.raises(ValueError, match="epsilons"):
            _call(
                CorrelationDimensionResult,
                D2=1.0,
                epsilons=cast(FloatArray, epsilons),
                C_eps=[0.25, 0.75],
                slope=[1.25],
                scaling_range=(0.1, 1.0),
            )

    @pytest.mark.parametrize(
        "C_eps",
        [
            [0.25],
            [0.25, True],
            ["not-ci", "still-not-ci"],
            ["0.25", "0.75"],
            np.array([0.25, "0.75"], dtype=object),
            [0.25 + 1.0j, 0.75 + 0.0j],
            np.array([0.25 + 1.0j, 0.75], dtype=object),
            [0.25, np.nan],
            [0.25, 1.1],
            [0.75, 0.25],
        ],
    )
    def test_rejects_invalid_correlation_integral(self, C_eps: object) -> None:
        with pytest.raises(ValueError, match="C_eps"):
            _call(
                CorrelationDimensionResult,
                D2=1.0,
                epsilons=[0.1, 1.0],
                C_eps=C_eps,
                slope=[1.25],
                scaling_range=(0.1, 1.0),
            )

    @pytest.mark.parametrize(
        "slope",
        [
            [1.0, 2.0, 3.0],
            [1.0, True],
            ["not-a-slope"],
            ["1.25"],
            np.array([1.25, "1.5"], dtype=object),
            [np.nan],
            [[1.0]],
            np.array([1.0 + 0.1j], dtype=object),
        ],
    )
    def test_rejects_invalid_slope(self, slope: object) -> None:
        with pytest.raises(ValueError, match="slope"):
            _call(
                CorrelationDimensionResult,
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
    def test_rejects_invalid_scaling_range(self, scaling_range: object) -> None:
        with pytest.raises(ValueError, match="scaling_range"):
            _call(
                CorrelationDimensionResult,
                D2=1.0,
                epsilons=[0.1, 1.0],
                C_eps=[0.25, 0.75],
                slope=[1.25],
                scaling_range=scaling_range,
            )


class TestBackendDispatch:
    def test_alias_helpers_treat_uncoercible_arrays_as_absent(self) -> None:
        """Alias scanners fail closed when NumPy cannot inspect a payload."""
        broken = _ArrayCoercionFailure()

        assert dim_mod._contains_boolean_alias(broken) is False
        assert dim_mod._contains_complex_alias(broken) is False
        assert dim_mod._contains_numeric_string_alias(broken) is False
        assert dim_mod._has_complex_payload(broken) is False

    @pytest.mark.native_runtime
    def test_broken_asset_uses_public_python_fallback(self, tmp_path: Path) -> None:
        """A real Julia parse refusal preserves public Python results."""
        run_source_profile(tmp_path, "broken")

    @pytest.mark.native_runtime
    def test_missing_assets_use_public_python_fallback(self, tmp_path: Path) -> None:
        """Actual absent native assets leave an exercised Python-only profile."""
        run_source_profile(tmp_path, "missing")

    @pytest.mark.native_runtime
    def test_missing_entry_points_use_public_python_fallback(
        self, tmp_path: Path
    ) -> None:
        """A real side-file without numerical exports never admits Julia."""
        run_source_profile(tmp_path, "missing-entry")

    @pytest.mark.parametrize(
        "backend_output",
        [
            np.array([0.5], dtype=np.float64),
            np.array([0.5, np.nan], dtype=np.float64),
            np.array([0.5, 1.1], dtype=np.float64),
            np.array([0.5, 0.4], dtype=np.float64),
            np.array([True, False], dtype=np.bool_),
            np.array([0.5, np.bool_(True)], dtype=object),
            np.array(["0.0", "0.5"], dtype=object),
            np.array([0.5 + 1.0j, 0.75 + 0.0j], dtype=np.complex128),
        ],
    )
    def test_invalid_correlation_integral_backend_payload_fails_closed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        backend_output: object,
    ) -> None:
        previous_backend = dim_mod.ACTIVE_BACKEND
        previous_available = list(dim_mod.AVAILABLE_BACKENDS)
        previous_loader = dim_mod._LOADERS["go"]

        def fake_ci(*_args: object, **_kwargs: object) -> object:
            return backend_output

        dim_mod.ACTIVE_BACKEND = "go"
        dim_mod.AVAILABLE_BACKENDS = ["go", "python"]
        dim_mod._BACKEND_FN_CACHE.clear()
        monkeypatch.setitem(dim_mod._LOADERS, "go", lambda: {"ci": fake_ci})
        try:
            with pytest.raises(ValueError):
                correlation_integral(
                    np.array([[0.0], [1.0], [2.0]], dtype=np.float64),
                    np.array([0.5, 1.5], dtype=np.float64),
                    max_pairs=10,
                )
        finally:
            dim_mod.ACTIVE_BACKEND = previous_backend
            dim_mod.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(dim_mod._LOADERS, "go", previous_loader)
            dim_mod._BACKEND_FN_CACHE.clear()

    @pytest.mark.parametrize(
        "backend_value",
        [-0.1, np.nan, np.inf, 4.0, True, np.bool_(True), 1.0 + 0.1j],
    )
    def test_invalid_kaplan_yorke_backend_payload_fails_closed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        backend_value: object,
    ) -> None:
        previous_backend = dim_mod.ACTIVE_BACKEND
        previous_available = list(dim_mod.AVAILABLE_BACKENDS)
        previous_loader = dim_mod._LOADERS["go"]
        le = np.array([0.5, -1.0, -2.0], dtype=np.float64)

        def fake_ky(*_args: object, **_kwargs: object) -> object:
            return backend_value

        dim_mod.ACTIVE_BACKEND = "go"
        dim_mod.AVAILABLE_BACKENDS = ["go", "python"]
        dim_mod._BACKEND_FN_CACHE.clear()
        monkeypatch.setitem(dim_mod._LOADERS, "go", lambda: {"ky": fake_ky})
        try:
            with pytest.raises(ValueError):
                kaplan_yorke_dimension(le)
        finally:
            dim_mod.ACTIVE_BACKEND = previous_backend
            dim_mod.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(dim_mod._LOADERS, "go", previous_loader)
            dim_mod._BACKEND_FN_CACHE.clear()

    @pytest.mark.native_runtime
    def test_next_actual_runtime_is_selected_when_faster_owners_are_absent(
        self, tmp_path: Path
    ) -> None:
        """Actual Rust/Mojo absence selects numerical Julia before Python."""
        run_source_profile(tmp_path, "present")

    @pytest.mark.native_runtime
    def test_julia_loader_exposes_ci_and_ky_functions(self) -> None:
        """Explicit public Julia selection executes both actual numerical exports."""
        np.testing.assert_allclose(
            correlation_integral([0.0, 1.0, 3.0], [1.0, 2.0, 4.0], backend="julia"),
            [0.0, 1.0 / 3.0, 1.0],
            rtol=0.0,
            atol=1e-15,
        )
        assert kaplan_yorke_dimension([1.0, -2.0], backend="julia") == 1.5

    @pytest.mark.native_runtime
    def test_rust_loader_exposes_ci_and_ky_functions(self) -> None:
        """Observe compiled CI/KY calls through public selection and exact oracles."""
        kernel = import_module("spo_kernel")
        ci: object = kernel.correlation_integral_rust
        ky: object = kernel.kaplan_yorke_dimension_rust
        assert inspect.isbuiltin(ci) and inspect.isbuiltin(ky)
        events: list[tuple[str, str]] = []

        def observe(_frame: FrameType, event: str, arg: object) -> None:
            """Record successful native entries/returns without replacing a solver."""
            if event in {"c_call", "c_return", "c_exception"}:
                if arg is ci:
                    events.append(("ci", event))
                elif arg is ky:
                    events.append(("ky", event))

        previous = sys.getprofile()
        sys.setprofile(observe)
        try:
            actual = correlation_integral(
                [0.0, 1.0, 3.0], [1.0, 2.0, 4.0], backend="rust"
            )
            dimension = kaplan_yorke_dimension([1.0, -2.0], backend="rust")
        finally:
            sys.setprofile(previous)
        np.testing.assert_allclose(actual, [0.0, 1.0 / 3.0, 1.0], rtol=0, atol=1e-15)
        assert dimension == 1.5
        assert events == [
            ("ci", "c_call"),
            ("ci", "c_return"),
            ("ky", "c_call"),
            ("ky", "c_return"),
        ]

    @pytest.mark.native_runtime
    def test_rust_correlation_dispatch_sorts_epsilons_and_uses_contiguous_buffers(
        self,
    ) -> None:
        """Actual Rust accepts a noncontiguous public trajectory without mutation."""
        traj = np.array([[0.0, 1.0], [1.0, 1.5], [2.0, 3.0]], order="F")
        assert not traj.flags.c_contiguous
        original = traj.copy()
        eps = np.array([2.0, 0.1, 1.0])
        C = correlation_integral(traj, eps, max_pairs=17, seed=9, backend="rust")
        np.testing.assert_array_equal(C, np.array([0.0, 0.0, 2.0 / 3.0]))
        np.testing.assert_array_equal(traj, original)
        np.testing.assert_array_equal(eps, [2.0, 0.1, 1.0])

    @pytest.mark.native_runtime
    def test_non_rust_dispatch_uses_prepared_indices(
        self,
    ) -> None:
        """Actual Go follows seeded distinct-pair sampling and its normalization."""
        traj = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        eps = np.array([4.0, 2.0, 1.0])
        rng = np.random.default_rng(99)
        first = rng.integers(0, len(traj), 3)
        second = rng.integers(0, len(traj), 3)
        distances = [
            math.dist(traj[i], traj[j])
            for i, j in zip(first, second, strict=True)
            if i != j
        ]
        assert distances
        expected = [
            sum(d < radius for d in distances) / len(distances)
            for radius in sorted(eps)
        ]
        C = correlation_integral(traj, eps, max_pairs=3, seed=99, backend="go")
        np.testing.assert_array_equal(C, expected)

    @pytest.mark.parametrize("backend", ["rust", "go"])
    def test_correlation_integral_backend_runtime_failure_falls_back_to_python(
        self,
        monkeypatch: pytest.MonkeyPatch,
        backend: str,
    ) -> None:
        """Runtime backend failures preserve the exact Python result."""
        previous_backend = dim_mod.ACTIVE_BACKEND
        previous_loader = dim_mod._LOADERS[backend]

        def failing_ci(*_args: object) -> FloatArray:
            raise RuntimeError(f"{backend} backend unavailable")

        dim_mod.ACTIVE_BACKEND = backend
        dim_mod._BACKEND_FN_CACHE.clear()
        monkeypatch.setitem(dim_mod._LOADERS, backend, lambda: {"ci": failing_ci})
        try:
            C = correlation_integral(
                np.array([[0.0], [1.0], [2.0]], dtype=np.float64),
                np.array([0.5, 1.5], dtype=np.float64),
                max_pairs=10,
            )
        finally:
            dim_mod.ACTIVE_BACKEND = previous_backend
            monkeypatch.setitem(dim_mod._LOADERS, backend, previous_loader)
            dim_mod._BACKEND_FN_CACHE.clear()

        np.testing.assert_array_equal(C, np.array([0.0, 2.0 / 3.0]))

    def test_kaplan_yorke_backend_runtime_failure_falls_back_to_python(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Runtime backend failures preserve the exact Python KY result."""
        previous_backend = dim_mod.ACTIVE_BACKEND
        previous_loader = dim_mod._LOADERS["go"]

        def failing_ky(*_args: object) -> float:
            raise RuntimeError("go backend unavailable")

        dim_mod.ACTIVE_BACKEND = "go"
        dim_mod._BACKEND_FN_CACHE.clear()
        monkeypatch.setitem(dim_mod._LOADERS, "go", lambda: {"ky": failing_ky})
        try:
            result = kaplan_yorke_dimension(np.array([0.5, -1.0]))
        finally:
            dim_mod.ACTIVE_BACKEND = previous_backend
            monkeypatch.setitem(dim_mod._LOADERS, "go", previous_loader)
            dim_mod._BACKEND_FN_CACHE.clear()

        assert result == pytest.approx(1.5)


class TestCorrelationDimensionEdgeCases:
    def test_few_valid_c_eps(self) -> None:
        """Trajectory with very few valid C(ε) → D2=0."""
        # Two points very close together → only small ε gives C>0
        traj = np.array([[0.0, 0.0], [1e-15, 1e-15], [0.0, 0.0]])
        result = correlation_dimension(traj, n_epsilons=5)
        assert result.D2 >= 0.0

    def test_single_point_trajectory(self) -> None:
        """T=1 → diameter=0 → D2=0."""
        traj = np.array([[1.0, 2.0]])
        result = correlation_dimension(traj)
        assert result.D2 == 0.0

    def test_short_slope_window(self) -> None:
        """Very few epsilons → window < 2 branch."""
        traj = np.array([[0.0], [1.0], [2.0], [3.0]])
        result = correlation_dimension(traj, n_epsilons=3)
        assert isinstance(result.D2, float)


class TestDimensionPipelineWiring:
    """Pipeline: engine trajectory → embed → correlation dimension."""

    def test_engine_trajectory_to_correlation_dimension(self) -> None:
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


@pytest.mark.parametrize("required", [False, True])
def test_public_dispatch_refuses_missing_callable_contract(
    monkeypatch: pytest.MonkeyPatch,
    required: bool,
) -> None:
    """Malformed backend registration cannot report a successful native result."""
    monkeypatch.setattr(dim_mod, "ACTIVE_BACKEND", "rust")
    monkeypatch.setattr(dim_mod, "AVAILABLE_BACKENDS", ["rust"])
    monkeypatch.setattr(dim_mod, "_BACKEND_FN_CACHE", {"rust": {}})
    if required:
        with pytest.raises(RuntimeError, match="lacks ci"):
            correlation_integral([[0.0], [1.0]], [2.0], backend="rust")
    else:
        np.testing.assert_array_equal(
            correlation_integral([[0.0], [1.0]], [2.0]), [1.0]
        )


def test_public_optional_loader_refusal_preserves_python_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Automatic ownership falls back after an explicit loader refusal."""

    def refuse() -> dict[str, object]:
        raise ImportError("deliberate loader refusal")

    monkeypatch.setattr(dim_mod, "ACTIVE_BACKEND", "rust")
    monkeypatch.setattr(dim_mod, "AVAILABLE_BACKENDS", ["rust", "python"])
    monkeypatch.setattr(dim_mod, "_BACKEND_FN_CACHE", {})
    monkeypatch.setitem(dim_mod._LOADERS, "rust", refuse)
    np.testing.assert_array_equal(correlation_integral([[0.0], [1.0]], [2.0]), [1.0])


def test_required_rust_execution_failure_never_silently_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure-only Rust callback verifies explicit-owner error propagation."""

    def refuse(*_args: object) -> NoReturn:
        raise RuntimeError("deliberate native execution refusal")

    monkeypatch.setattr(dim_mod, "_BACKEND_FN_CACHE", {"rust": {"ci": refuse}})
    with pytest.raises(RuntimeError, match="backend 'rust' failed"):
        correlation_integral([[0.0], [1.0]], [2.0], backend="rust")
