# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Algorithmic tests for coupling.spectral

"""Algorithmic properties of ``coupling.spectral``.

Covered: Laplacian structure (row-sums = 0); the smallest
eigenvalue of ``L`` is exactly zero for any connected weighted
graph; ``λ₂ = 0`` iff the graph is disconnected; Dörfler-Bullo
``K_c`` for the complete graph; ``fiedler_partition`` returns
disjoint non-empty partitions; ``spectral_gap`` ≥ 0; single-
oscillator and empty edge cases.
"""

from __future__ import annotations

import cProfile
import functools
import math
from collections.abc import Callable
from typing import NoReturn, ParamSpec, TypeVar

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.coupling import spectral as s_mod
from scpn_phase_orchestrator.coupling.spectral import (
    critical_coupling,
    fiedler_partition,
    fiedler_value,
    fiedler_vector,
    graph_laplacian,
    spectral_eig,
    spectral_gap,
    sync_convergence_rate,
)
from tests._runtime_owners import NEEDS_KERNEL

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _python(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """Run ``func`` with the pure-Python backend forced."""

    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        prev = s_mod.ACTIVE_BACKEND
        s_mod.ACTIVE_BACKEND = "python"
        s_mod._PRIM_CACHE = None
        try:
            return func(*args, **kwargs)
        finally:
            s_mod.ACTIVE_BACKEND = prev
            s_mod._PRIM_CACHE = None

    return wrapper


def _path_graph(n: int, weight: float) -> np.ndarray:
    """Return the weighted path graph ``0 – 1 – … – (n−1)``."""
    knm = np.zeros((n, n))
    for i in range(n - 1):
        knm[i, i + 1] = knm[i + 1, i] = weight
    return knm


_CallR = TypeVar("_CallR")
_Primitive = Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray]]


def _call(fn: Callable[..., _CallR], *args: object, **kwargs: object) -> _CallR:
    """Call ``fn`` with arguments that are deliberately outside its types."""
    return fn(*args, **kwargs)


def _path_spectrum(n: int, weight: float) -> list[float]:
    """Return the path-graph Laplacian eigenvalues ``2w(1 − cos(πk/n))``."""
    return [2 * weight * (1 - math.cos(math.pi * k / n)) for k in range(n)]


def _path_fiedler(n: int) -> np.ndarray:
    """Return the unit Fiedler vector ``cos(π(j + ½)/n)`` of the path graph."""
    vector = np.array([math.cos(math.pi * (j + 0.5) / n) for j in range(n)])
    unit: np.ndarray = vector / np.linalg.norm(vector)
    return unit


def _count_calls(code_name: str, action: Callable[[], object]) -> int:
    """Return how often a Python function named ``code_name`` ran in ``action``."""
    profile = cProfile.Profile()
    profile.enable()
    try:
        action()
    finally:
        profile.disable()
    return sum(
        entry.callcount
        for entry in profile.getstats()
        if not isinstance(entry.code, str) and entry.code.co_name == code_name
    )


def _builtin_calls(profile: cProfile.Profile, name: str) -> int:
    """Return how often a compiled function named ``name`` ran under ``profile``."""
    return sum(
        entry.callcount
        for entry in profile.getstats()
        if isinstance(entry.code, str) and f"{name}>" in entry.code
    )


class TestGraphLaplacian:
    def test_row_sums_zero(self) -> None:
        W = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
        L = graph_laplacian(W)
        np.testing.assert_allclose(L.sum(axis=1), 0.0, atol=1e-14)

    def test_negative_offdiag(self) -> None:
        """Off-diagonal entries of L are ``−|W_ij|``."""
        W = np.array([[0.0, 2.5], [2.5, 0.0]])
        L = graph_laplacian(W)
        assert L[0, 1] == -2.5
        assert L[1, 0] == -2.5

    def test_abs_on_input(self) -> None:
        """Negative weights are treated as magnitudes."""
        W = np.array([[0.0, -1.0], [-1.0, 0.0]])
        L = graph_laplacian(W)
        assert L[0, 0] == 1.0  # |−1| contributes to degree

    @_python
    def test_asymmetric_coupling_preserves_undirected_spectrum(self) -> None:
        """The spectral primitive operates on reciprocal undirected
        magnitude weights, so reversing the measurement direction does
        not change the Laplacian spectrum."""
        W = np.array(
            [
                [0.0, 2.0, 0.0],
                [6.0, 0.0, 4.0],
                [8.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        eigvals, _ = spectral_eig(W)
        reversed_eigvals, _ = spectral_eig(W.T)

        np.testing.assert_allclose(eigvals, [0.0, 8.0, 12.0], atol=1e-10)
        np.testing.assert_allclose(reversed_eigvals, eigvals, atol=1e-12)
        assert fiedler_value(W) == pytest.approx(fiedler_value(W.T), abs=1e-12)

    @pytest.mark.parametrize(
        ("knm", "match"),
        [
            (np.array([[True, False], [False, True]]), "knm"),
            (np.array([[0.0, np.nan], [1.0, 0.0]]), "knm"),
            (np.zeros((2, 3)), "knm"),
        ],
    )
    def test_rejects_invalid_coupling_matrix(self, knm: np.ndarray, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            graph_laplacian(knm)

    def test_rejects_mixed_boolean_coupling_alias(self) -> None:
        with pytest.raises(ValueError, match="knm must not contain boolean"):
            _call(graph_laplacian, [[0.0, True], [1.0, 0.0]])

    def test_rejects_numpy_boolean_coupling_alias(self) -> None:
        knm = np.array(
            [[0.0, np.bool_(True)], [1.0, 0.0]],
            dtype=object,
        )

        with pytest.raises(ValueError, match="knm must not contain boolean"):
            graph_laplacian(knm)


class TestFiedlerValue:
    @_python
    def test_connected_graph_has_positive_lambda2(self) -> None:
        n = 5
        W = np.ones((n, n))
        np.fill_diagonal(W, 0.0)
        assert fiedler_value(W) > 0.0

    @_python
    def test_smallest_eigvalue_is_zero(self) -> None:
        """λ₁ = 0 for any weighted graph (``1 ∈ ker L``)."""
        rng = np.random.default_rng(0)
        n = 6
        W = rng.uniform(0, 1, (n, n))
        W = (W + W.T) / 2
        np.fill_diagonal(W, 0.0)
        eigvals, _ = spectral_eig(W)
        assert abs(eigvals[0]) < 1e-12

    @_python
    def test_complete_graph_lambda2_equals_n(self) -> None:
        """For the complete graph with unit weights,
        ``λ₂ = λ₃ = … = λ_N = N`` (closed-form)."""
        n = 5
        W = np.ones((n, n))
        np.fill_diagonal(W, 0.0)
        assert fiedler_value(W) == pytest.approx(n, abs=1e-10)


class TestFiedlerVectorOrthogonality:
    @_python
    def test_vector_orthogonal_to_ones(self) -> None:
        """All non-zero eigenvectors are orthogonal to
        ``1``, which spans ``ker L``."""
        rng = np.random.default_rng(1)
        n = 6
        W = rng.uniform(0, 1, (n, n))
        W = (W + W.T) / 2
        np.fill_diagonal(W, 0.0)
        v2 = fiedler_vector(W)
        assert abs(float(np.sum(v2))) < 1e-10

    @_python
    def test_vector_satisfies_eigenequation(self) -> None:
        """``L v₂ = λ₂ v₂`` within numerical tolerance."""
        rng = np.random.default_rng(2)
        n = 6
        W = rng.uniform(0, 1, (n, n))
        W = (W + W.T) / 2
        np.fill_diagonal(W, 0.0)
        L = graph_laplacian(W)
        lam2 = fiedler_value(W)
        v2 = fiedler_vector(W)
        residual = L @ v2 - lam2 * v2
        assert np.max(np.abs(residual)) < 1e-10


class TestCriticalCoupling:
    @_python
    def test_critical_coupling_complete_graph(self) -> None:
        """``K_c = Δω / λ₂ = Δω / N`` for a unit-weight complete
        graph of ``N`` oscillators."""
        n = 4
        omegas = np.array([0.0, 1.0, 2.0, 3.0])
        W = np.ones((n, n))
        np.fill_diagonal(W, 0.0)
        kc = critical_coupling(omegas, W)
        assert kc == pytest.approx(3.0 / n, abs=1e-10)

    @_python
    def test_disconnected_graph_returns_infinity(self) -> None:
        """A graph with two disconnected components has
        ``λ₂ = 0`` → ``K_c = +∞``."""
        n = 4
        W = np.zeros((n, n))
        W[0, 1] = W[1, 0] = 1.0  # component {0, 1}
        W[2, 3] = W[3, 2] = 1.0  # component {2, 3}
        omegas = np.array([0.0, 1.0, 2.0, 3.0])
        kc = critical_coupling(omegas, W)
        assert kc == float("inf")

    @_python
    @pytest.mark.parametrize(
        ("omegas", "knm", "match"),
        [
            (np.array([True, False]), np.ones((2, 2)), "omegas"),
            (np.array([0.0, np.inf]), np.ones((2, 2)), "omegas"),
            (np.array([]), np.ones((2, 2)), "omegas"),
            (np.zeros(2), np.array([[True, False], [False, True]]), "knm"),
        ],
    )
    def test_rejects_invalid_frequency_or_coupling_inputs(
        self, omegas: np.ndarray, knm: np.ndarray, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            critical_coupling(omegas, knm)

    @_python
    def test_rejects_frequency_vector_shorter_than_layer_matrix(self) -> None:
        W = np.ones((4, 4), dtype=np.float64)
        np.fill_diagonal(W, 0.0)

        with pytest.raises(ValueError, match="omegas"):
            critical_coupling(np.array([5.0, 10.0]), W)

    @_python
    def test_rejects_mixed_boolean_frequency_alias(self) -> None:
        W = np.ones((2, 2), dtype=np.float64)
        np.fill_diagonal(W, 0.0)

        with pytest.raises(ValueError, match="omegas must not contain boolean"):
            _call(critical_coupling, [True, 1.0], W)

    @_python
    def test_rejects_complex_frequency_alias(self) -> None:
        W = np.ones((2, 2), dtype=np.float64)
        np.fill_diagonal(W, 0.0)

        with pytest.raises(ValueError, match="omegas must be a finite real-valued"):
            critical_coupling(np.array([0.0, 1.0 + 2.0j]), W)


class TestFiedlerPartition:
    @_python
    def test_partitions_are_disjoint(self) -> None:
        """``pos`` and ``neg`` must be disjoint and cover all
        ``N`` nodes."""
        rng = np.random.default_rng(3)
        n = 6
        W = rng.uniform(0, 1, (n, n))
        W = (W + W.T) / 2
        np.fill_diagonal(W, 0.0)
        pos, neg = fiedler_partition(W)
        assert len(pos) + len(neg) == n
        assert set(pos).isdisjoint(set(neg))


class TestSpectralGap:
    @_python
    def test_non_negative(self) -> None:
        rng = np.random.default_rng(4)
        n = 6
        W = rng.uniform(0, 1, (n, n))
        W = (W + W.T) / 2
        np.fill_diagonal(W, 0.0)
        assert spectral_gap(W) >= -1e-14

    @_python
    def test_complete_graph_gap_is_zero(self) -> None:
        """``K_N`` has λ₂ = … = λ_N = N, so ``λ₃ − λ₂ = 0``."""
        n = 5
        W = np.ones((n, n))
        np.fill_diagonal(W, 0.0)
        assert abs(spectral_gap(W)) < 1e-10


class TestSyncConvergenceRate:
    @_python
    def test_returns_zero_for_empty(self) -> None:
        assert (
            sync_convergence_rate(
                np.zeros((0, 0)),
                np.zeros(0),
            )
            == 0.0
        )

    @_python
    def test_positive_for_connected(self) -> None:
        n = 4
        W = np.ones((n, n))
        np.fill_diagonal(W, 0.0)
        omegas = np.arange(n, dtype=np.float64)
        mu = sync_convergence_rate(W, omegas, gamma_max=0.0)
        assert mu > 0.0

    @_python
    @pytest.mark.parametrize(
        ("knm", "omegas", "gamma_max", "match"),
        [
            (np.array([[True, False], [False, True]]), np.zeros(2), 0.0, "knm"),
            (np.ones((2, 2)), np.array([True, False]), 0.0, "omegas"),
            (np.ones((2, 2)), np.zeros(3), 0.0, "omegas"),
            (np.ones((2, 2)), np.zeros(2), True, "gamma_max"),
            (np.ones((2, 2)), np.zeros(2), "0.1", "numeric-string"),
            (np.ones((2, 2)), np.zeros(2), np.nan, "gamma_max"),
        ],
    )
    def test_rejects_invalid_convergence_inputs(
        self, knm: np.ndarray, omegas: np.ndarray, gamma_max: object, match: str
    ) -> None:
        with pytest.raises((TypeError, ValueError), match=match):
            _call(sync_convergence_rate, knm, omegas, gamma_max=gamma_max)


class TestHypothesis:
    @_python
    @given(
        n=st.integers(min_value=3, max_value=8),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=6,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_eigvals_ascending_and_nonneg(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        W = rng.uniform(0, 1, (n, n))
        W = (W + W.T) / 2
        np.fill_diagonal(W, 0.0)
        eigvals, _ = spectral_eig(W)
        # Ascending order.


class TestPrimitiveDispatchFallback:
    def test_primitive_falls_back_to_next_backend_when_active_loader_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _fail_go() -> NoReturn:
            raise ImportError("go unavailable")

        def _ok_julia() -> _Primitive:
            return lambda flat, n: (
                np.zeros(n, dtype=np.float64),
                np.zeros(n, dtype=np.float64),
            )

        monkeypatch.setattr(s_mod, "_PRIM_CACHE", {})
        monkeypatch.setattr(s_mod, "ACTIVE_BACKEND", "go")
        monkeypatch.setattr(s_mod, "AVAILABLE_BACKENDS", ["go", "julia", "python"])
        monkeypatch.setattr(
            s_mod, "_LOADERS", {"go": _fail_go, "julia": _ok_julia, "rust": lambda: {}}
        )

        primitive = s_mod._primitive()
        eigvals, fiedler = primitive(np.zeros(4, dtype=np.float64), 2)
        np.testing.assert_allclose(eigvals, np.zeros(2, dtype=np.float64))
        np.testing.assert_allclose(fiedler, np.zeros(2, dtype=np.float64))

    @pytest.mark.native_runtime
    @pytest.mark.parametrize("owner", ["mojo", "julia", "go"])
    def test_primitive_uses_cached_loader_once(
        self, owner: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On a cold cache two loads of an installed primitive run its loader once."""
        if owner not in s_mod.AVAILABLE_BACKENDS:
            pytest.skip(f"{owner} unavailable")
        monkeypatch.setattr(s_mod, "_PRIM_CACHE", {})
        loads = _count_calls(
            f"_load_{owner}_primitive",
            lambda: [s_mod._load_primitive_backend(owner) for _ in range(2)],
        )
        assert loads == 1
        cache = s_mod._PRIM_CACHE
        assert cache is not None
        assert s_mod._load_primitive_backend(owner) is cache[owner]


class TestDispatcherSurface:
    def test_resolve_backends_ignores_optional_loader_failures(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unavailable optional backends must be skipped while preserving
        import-path order and a deterministic python fallback."""
        calls: list[str] = []

        def fail() -> NoReturn:
            calls.append("fail")
            raise RuntimeError("backend not available")

        def ok() -> _Primitive:
            calls.append("ok")

            def _primitive(flat: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
                return np.arange(n, dtype=np.float64), np.ones(n, dtype=np.float64)

            return _primitive

        monkeypatch.setitem(s_mod._LOADERS, "rust", fail)
        monkeypatch.setitem(s_mod._LOADERS, "mojo", fail)
        monkeypatch.setitem(s_mod._LOADERS, "julia", fail)
        monkeypatch.setitem(s_mod._LOADERS, "go", ok)

        active_backend, available_backends = s_mod._resolve_backends()

        assert active_backend == "go"
        assert available_backends == ["go", "python"]
        assert calls == ["fail", "fail", "fail", "ok"]

    def test_resolve_backends_falls_back_to_python_when_all_fail(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If all optional loaders fail, python remains the runtime default."""

        def fail() -> NoReturn:
            raise OSError("toolchain not usable")

        monkeypatch.setitem(s_mod._LOADERS, "rust", fail)
        monkeypatch.setitem(s_mod._LOADERS, "mojo", fail)
        monkeypatch.setitem(s_mod._LOADERS, "julia", fail)
        monkeypatch.setitem(s_mod._LOADERS, "go", fail)

        active_backend, available_backends = s_mod._resolve_backends()

        assert active_backend == "python"
        assert available_backends == ["python"]

    def test_available_non_empty(self) -> None:
        assert s_mod.AVAILABLE_BACKENDS
        assert "python" in s_mod.AVAILABLE_BACKENDS

    def test_active_is_first(self) -> None:
        assert s_mod.AVAILABLE_BACKENDS[0] == s_mod.ACTIVE_BACKEND

    @pytest.mark.native_runtime
    @pytest.mark.parametrize("owner", ["mojo", "julia", "go"])
    def test_toolchain_loaders_return_verified_backend_callables(
        self, owner: str
    ) -> None:
        """Each real primitive loader returns the analytic path-graph spectrum."""
        if owner not in s_mod.AVAILABLE_BACKENDS:
            pytest.skip(f"{owner} unavailable")
        loader = {
            "mojo": s_mod._load_mojo_primitive,
            "julia": s_mod._load_julia_primitive,
            "go": s_mod._load_go_primitive,
        }[owner]
        eigvals, fiedler = loader()(_path_graph(7, 0.8).ravel(), 7)
        atol = 1e-9 if owner == "mojo" else 1e-12
        np.testing.assert_allclose(eigvals, _path_spectrum(7, 0.8), atol=atol)
        np.testing.assert_allclose(
            abs(float(np.dot(fiedler, _path_fiedler(7)))), 1.0, atol=atol
        )

    @pytest.mark.native_runtime
    @pytest.mark.parametrize("owner", ["mojo", "julia", "go"])
    def test_non_rust_non_python_primitive_is_cached(
        self, owner: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A dropped primitive cache is rebuilt by one real load and then reused."""
        if owner not in s_mod.AVAILABLE_BACKENDS:
            pytest.skip(f"{owner} unavailable")
        monkeypatch.setattr(s_mod, "_PRIM_CACHE", None)
        first: list[object] = []
        loads = _count_calls(
            f"_load_{owner}_primitive",
            lambda: first.extend(
                s_mod._load_primitive_backend(owner) for _ in range(2)
            ),
        )
        assert loads == 1
        assert first[0] is first[1]
        assert isinstance(s_mod._PRIM_CACHE, dict)


class TestValidationBoundaries:
    def test_graph_laplacian_rejects_complex_weights(self) -> None:
        knm = np.array([[0.0, 1.0 + 1.0j], [1.0 + 1.0j, 0.0]])
        with pytest.raises(ValueError, match="knm must be a finite square matrix"):
            graph_laplacian(knm)

    def test_defensive_boolean_dtype_matrix_guard(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(s_mod, "_contains_boolean_alias", lambda _value: False)

        with pytest.raises(ValueError, match="knm must not contain boolean"):
            graph_laplacian(np.array([[True, False], [False, True]]))

    def test_defensive_boolean_dtype_omega_guard(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(s_mod, "_contains_boolean_alias", lambda _value: False)

        with pytest.raises(ValueError, match="omegas must not contain boolean"):
            sync_convergence_rate(
                np.array([[0.0, 1.0], [1.0, 0.0]]),
                np.array([True, False]),
            )

    def test_optional_backend_boolean_output_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        knm = np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )

        def invalid_primitive(
            flat: np.ndarray, n: int
        ) -> tuple[np.ndarray, np.ndarray]:
            assert n == 3
            assert flat.shape == (9,)
            return (
                np.array([False, True, True], dtype=np.bool_),
                np.array([True, False, True], dtype=np.bool_),
            )

        monkeypatch.setattr(s_mod, "_primitive", lambda: invalid_primitive)

        with pytest.raises(ValueError):
            spectral_eig(knm)

    def test_rust_fast_path_rejects_boolean_return_aliases(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        knm = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)

        def rust_bundle() -> dict[str, Callable[[np.ndarray, int], object]]:
            return {
                "fv": lambda flat, n: np.bool_(True),
                "fvec": lambda flat, n: np.array([np.bool_(True), 0.0], dtype=object),
            }

        monkeypatch.setattr(s_mod, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(s_mod, "_RUST_CACHE", None)
        monkeypatch.setattr(s_mod, "_load_rust_bundle", rust_bundle)

        with pytest.raises(ValueError, match="Fiedler value"):
            fiedler_value(knm)
        with pytest.raises(ValueError, match="Fiedler vector"):
            fiedler_vector(knm)

    def test_rust_fast_path_rejects_numeric_string_scalar(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        knm = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)

        def rust_bundle() -> dict[str, Callable[[np.ndarray, int], object]]:
            return {
                "fv": lambda flat, n: "1.0",
                "fvec": lambda flat, n: np.array([1.0, -1.0], dtype=np.float64),
            }

        monkeypatch.setattr(s_mod, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(s_mod, "_RUST_CACHE", None)
        monkeypatch.setattr(s_mod, "_load_rust_bundle", rust_bundle)

        with pytest.raises(ValueError, match="numeric-string"):
            fiedler_value(knm)

    def test_checked_primitive_reraises_python_floor_loader_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def failing_python_floor(flat: np.ndarray, n: int) -> NoReturn:
            raise ImportError("python floor unavailable")

        monkeypatch.setattr(s_mod, "_python_spectral_eig", failing_python_floor)
        monkeypatch.setattr(s_mod, "_primitive", lambda: failing_python_floor)

        with pytest.raises(ImportError, match="python floor unavailable"):
            s_mod._spectral_eig_checked(np.zeros(4, dtype=np.float64), 2)

    def test_checked_primitive_falls_back_after_optional_loader_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def failing_optional(_flat: np.ndarray, _n: int) -> NoReturn:
            raise ImportError("optional backend unavailable")

        monkeypatch.setattr(s_mod, "_primitive", lambda: failing_optional)

        eigvals, fiedler = s_mod._spectral_eig_checked(
            np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64),
            2,
        )

        np.testing.assert_allclose(eigvals, [0.0, 2.0], atol=1e-12)
        assert fiedler.shape == (2,)

    def test_primitive_uses_python_floor_when_no_named_backend_loads(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(s_mod, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(s_mod, "AVAILABLE_BACKENDS", ["rust"])

        assert s_mod._primitive() is s_mod._python_spectral_eig

    @pytest.mark.parametrize(
        ("knm", "omegas", "match"),
        [
            (
                np.array([[0.0, 1.0], [1.0, np.inf]]),
                np.array([0.0, 1.0]),
                "finite",
            ),
            (
                np.array([[0.0, 1.0], [1.0, 0.0]]),
                np.array([], dtype=np.float64),
                "at least one frequency",
            ),
        ],
    )
    def test_convergence_rate_rejects_invalid_inputs(
        self, knm: np.ndarray, omegas: np.ndarray, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            sync_convergence_rate(knm, omegas)

    @pytest.mark.parametrize(
        ("knm", "omegas"),
        [
            (
                np.array([[0.0, 1.0], [1.0, 0.0]]),
                np.array([True, False]),
            ),
            (
                np.array([[0.0, 1.0], [1.0, 0.0]], dtype=bool),
                np.array([0.0, 1.0]),
            ),
        ],
    )
    def test_critical_coupling_rejects_invalid_omega_or_matrix(
        self, knm: np.ndarray, omegas: np.ndarray
    ) -> None:
        with pytest.raises(ValueError, match="knm|omegas"):
            critical_coupling(omegas, knm)

    def test_rust_primitive_uses_python_eigendecomposition_without_fast_path_roundtrip(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        knm = np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        monkeypatch.setattr(s_mod, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(s_mod, "_PRIM_CACHE", None)

        eigvals, fiedler = s_mod._primitive()(knm.ravel(), 3)

        np.testing.assert_allclose(eigvals, [0.0, 1.0, 3.0], atol=1e-12)
        assert fiedler.shape == (3,)

    @NEEDS_KERNEL
    def test_rust_bundle_loader_exposes_direct_fast_path_functions(self) -> None:
        """The compiled Rust bundle returns the analytic path-graph metrics."""
        if "rust" not in s_mod.AVAILABLE_BACKENDS:
            pytest.skip("rust unavailable")
        bundle = s_mod._load_rust_bundle()
        flat, spectrum = _path_graph(7, 0.8).ravel(), _path_spectrum(7, 0.8)
        omegas = np.array([0.3, -0.2, 1.1, 0.0, 0.5, 0.9, -0.4])
        assert bundle["fv"](flat, 7) == pytest.approx(spectrum[1], abs=1e-12)
        fvec = np.asarray(bundle["fvec"](flat, 7))
        assert abs(float(np.dot(fvec, _path_fiedler(7)))) == pytest.approx(
            1.0, abs=1e-12
        )
        assert bundle["sg"](flat, 7) == pytest.approx(
            spectrum[2] - spectrum[1], abs=1e-12
        )
        assert bundle["kc"](omegas, flat, 7) == pytest.approx(
            1.5 / spectrum[1], rel=1e-12
        )
        assert bundle["scr"](flat, omegas, 7, 0.3) == pytest.approx(
            0.8 * spectrum[1] * math.cos(0.3) / 7, rel=1e-12
        )

    @NEEDS_KERNEL
    def test_rust_active_backend_routes_derived_metrics_to_fast_paths(self) -> None:
        """With Rust active, each derived metric calls its compiled fast path once."""
        if s_mod.ACTIVE_BACKEND != "rust":
            pytest.skip("Rust is not the active spectral backend")
        knm = _path_graph(6, 1.2)
        omegas = np.linspace(-1.0, 1.0, 6)
        spectrum = _path_spectrum(6, 1.2)
        profile = cProfile.Profile()
        profile.enable()
        try:
            values = (
                fiedler_value(knm),
                fiedler_vector(knm),
                spectral_gap(knm),
                critical_coupling(omegas, knm),
                sync_convergence_rate(knm, omegas, gamma_max=0.2),
            )
        finally:
            profile.disable()
        names = [
            "fiedler_value_rust",
            "fiedler_vector_rust",
            "spectral_gap_rust",
            "critical_coupling_rust",
            "sync_convergence_rate_rust",
        ]
        for name in names:
            assert _builtin_calls(profile, name) == 1, name
        assert values[0] == pytest.approx(spectrum[1], abs=1e-12)
        assert abs(float(np.dot(values[1], _path_fiedler(6)))) == pytest.approx(1.0)
        assert values[2] == pytest.approx(spectrum[2] - spectrum[1], abs=1e-12)
        assert values[3] == pytest.approx(2.0 / spectrum[1], rel=1e-12)
        assert values[4] == pytest.approx(
            1.2 * spectrum[1] * math.cos(0.2) / 6, rel=1e-12
        )

    @_python
    def test_critical_coupling_empty_graph_rejects_frequency_mismatch(self) -> None:
        with pytest.raises(ValueError, match="omegas"):
            critical_coupling(np.array([0.0, 1.0]), np.zeros((0, 0)))

    def test_spectral_eig_accepts_integer_matrices_as_floats(self) -> None:
        knm = np.array(
            [
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ],
            dtype=np.int64,
        )
        eigvals, fiedler = _call(spectral_eig, knm)

        assert eigvals.dtype.kind == "f"
        assert fiedler.dtype.kind == "f"
        np.testing.assert_allclose(
            np.sort(eigvals),
            np.array([0.0, 1.0, 3.0]),
            atol=1e-12,
        )
        assert fiedler.shape == (3,)
