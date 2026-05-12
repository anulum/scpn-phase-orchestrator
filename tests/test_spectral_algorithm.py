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

import functools
import sys
import types

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


def _python(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        prev = s_mod.ACTIVE_BACKEND
        s_mod.ACTIVE_BACKEND = "python"
        s_mod._PRIM_CACHE = None
        try:
            return func(*args, **kwargs)
        finally:
            s_mod.ACTIVE_BACKEND = prev
            s_mod._PRIM_CACHE = None

    return wrapper


class TestGraphLaplacian:
    def test_row_sums_zero(self):
        W = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
        L = graph_laplacian(W)
        np.testing.assert_allclose(L.sum(axis=1), 0.0, atol=1e-14)

    def test_negative_offdiag(self):
        """Off-diagonal entries of L are ``−|W_ij|``."""
        W = np.array([[0.0, 2.5], [2.5, 0.0]])
        L = graph_laplacian(W)
        assert L[0, 1] == -2.5
        assert L[1, 0] == -2.5

    def test_abs_on_input(self):
        """Negative weights are treated as magnitudes."""
        W = np.array([[0.0, -1.0], [-1.0, 0.0]])
        L = graph_laplacian(W)
        assert L[0, 0] == 1.0  # |−1| contributes to degree


class TestFiedlerValue:
    @_python
    def test_connected_graph_has_positive_lambda2(self):
        n = 5
        W = np.ones((n, n))
        np.fill_diagonal(W, 0.0)
        assert fiedler_value(W) > 0.0

    @_python
    def test_smallest_eigvalue_is_zero(self):
        """λ₁ = 0 for any weighted graph (``1 ∈ ker L``)."""
        rng = np.random.default_rng(0)
        n = 6
        W = rng.uniform(0, 1, (n, n))
        W = (W + W.T) / 2
        np.fill_diagonal(W, 0.0)
        eigvals, _ = spectral_eig(W)
        assert abs(eigvals[0]) < 1e-12

    @_python
    def test_complete_graph_lambda2_equals_n(self):
        """For the complete graph with unit weights,
        ``λ₂ = λ₃ = … = λ_N = N`` (closed-form)."""
        n = 5
        W = np.ones((n, n))
        np.fill_diagonal(W, 0.0)
        assert fiedler_value(W) == pytest.approx(n, abs=1e-10)


class TestFiedlerVectorOrthogonality:
    @_python
    def test_vector_orthogonal_to_ones(self):
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
    def test_vector_satisfies_eigenequation(self):
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
    def test_critical_coupling_complete_graph(self):
        """``K_c = Δω / λ₂ = Δω / N`` for a unit-weight complete
        graph of ``N`` oscillators."""
        n = 4
        omegas = np.array([0.0, 1.0, 2.0, 3.0])
        W = np.ones((n, n))
        np.fill_diagonal(W, 0.0)
        kc = critical_coupling(omegas, W)
        assert kc == pytest.approx(3.0 / n, abs=1e-10)

    @_python
    def test_disconnected_graph_returns_infinity(self):
        """A graph with two disconnected components has
        ``λ₂ = 0`` → ``K_c = +∞``."""
        n = 4
        W = np.zeros((n, n))
        W[0, 1] = W[1, 0] = 1.0  # component {0, 1}
        W[2, 3] = W[3, 2] = 1.0  # component {2, 3}
        omegas = np.array([0.0, 1.0, 2.0, 3.0])
        kc = critical_coupling(omegas, W)
        assert kc == float("inf")


class TestFiedlerPartition:
    @_python
    def test_partitions_are_disjoint(self):
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
    def test_non_negative(self):
        rng = np.random.default_rng(4)
        n = 6
        W = rng.uniform(0, 1, (n, n))
        W = (W + W.T) / 2
        np.fill_diagonal(W, 0.0)
        assert spectral_gap(W) >= -1e-14

    @_python
    def test_complete_graph_gap_is_zero(self):
        """``K_N`` has λ₂ = … = λ_N = N, so ``λ₃ − λ₂ = 0``."""
        n = 5
        W = np.ones((n, n))
        np.fill_diagonal(W, 0.0)
        assert abs(spectral_gap(W)) < 1e-10


class TestSyncConvergenceRate:
    @_python
    def test_returns_zero_for_empty(self):
        assert (
            sync_convergence_rate(
                np.zeros((0, 0)),
                np.zeros(0),
            )
            == 0.0
        )

    @_python
    def test_positive_for_connected(self):
        n = 4
        W = np.ones((n, n))
        np.fill_diagonal(W, 0.0)
        omegas = np.arange(n, dtype=np.float64)
        mu = sync_convergence_rate(W, omegas, gamma_max=0.0)
        assert mu > 0.0


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
    def test_eigvals_ascending_and_nonneg(self, n, seed):
        rng = np.random.default_rng(seed)
        W = rng.uniform(0, 1, (n, n))
        W = (W + W.T) / 2
        np.fill_diagonal(W, 0.0)
        eigvals, _ = spectral_eig(W)
        # Ascending order.
        assert np.all(np.diff(eigvals) >= -1e-10)
        # L is positive semi-definite.
        assert np.all(eigvals >= -1e-10)


class TestDispatcherSurface:
    def test_available_non_empty(self):
        assert s_mod.AVAILABLE_BACKENDS
        assert "python" in s_mod.AVAILABLE_BACKENDS

    def test_active_is_first(self):
        assert s_mod.AVAILABLE_BACKENDS[0] == s_mod.ACTIVE_BACKEND

    def test_toolchain_loaders_return_verified_backend_callables(self, monkeypatch):
        def spectral_eig_backend(flat, n):
            return np.arange(n, dtype=np.float64), np.ones(n, dtype=np.float64)

        mojo = types.ModuleType("scpn_phase_orchestrator.coupling._spectral_mojo")
        mojo.spectral_eig_mojo = spectral_eig_backend
        mojo.ensure_called = False

        def ensure_exe():
            mojo.ensure_called = True

        mojo._ensure_exe = ensure_exe
        julia = types.ModuleType("scpn_phase_orchestrator.coupling._spectral_julia")
        julia.spectral_eig_julia = spectral_eig_backend
        go = types.ModuleType("scpn_phase_orchestrator.coupling._spectral_go")
        go.spectral_eig_go = spectral_eig_backend
        go.load_called = False

        def load_lib():
            go.load_called = True

        go._load_lib = load_lib

        monkeypatch.setitem(sys.modules, "juliacall", types.ModuleType("juliacall"))
        monkeypatch.setitem(
            sys.modules,
            "scpn_phase_orchestrator.coupling._spectral_mojo",
            mojo,
        )
        monkeypatch.setitem(
            sys.modules,
            "scpn_phase_orchestrator.coupling._spectral_julia",
            julia,
        )
        monkeypatch.setitem(
            sys.modules,
            "scpn_phase_orchestrator.coupling._spectral_go",
            go,
        )

        assert s_mod._load_mojo_primitive() is spectral_eig_backend
        assert mojo.ensure_called is True
        assert s_mod._load_julia_primitive() is spectral_eig_backend
        assert s_mod._load_go_primitive() is spectral_eig_backend
        assert go.load_called is True

    def test_non_rust_non_python_primitive_is_cached(self, monkeypatch):
        calls = 0

        def backend(flat, n):
            return np.arange(n, dtype=np.float64), np.ones(n, dtype=np.float64)

        def loader():
            nonlocal calls
            calls += 1
            return backend

        monkeypatch.setitem(s_mod._LOADERS, "go", loader)
        monkeypatch.setattr(s_mod, "ACTIVE_BACKEND", "go")
        monkeypatch.setattr(s_mod, "_PRIM_CACHE", None)

        first = s_mod._primitive()
        second = s_mod._primitive()

        assert first is backend
        assert second is backend
        assert calls == 1

    def test_rust_primitive_uses_python_eigendecomposition_without_fast_path_roundtrip(
        self, monkeypatch
    ):
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

    def test_rust_bundle_loader_exposes_direct_fast_path_functions(self, monkeypatch):
        spo_kernel = types.ModuleType("spo_kernel")
        spo_kernel.fiedler_value_rust = lambda flat, n: 2.0
        spo_kernel.fiedler_vector_rust = lambda flat, n: np.arange(n, dtype=np.float64)
        spo_kernel.spectral_gap_rust = lambda flat, n: 3.0
        spo_kernel.critical_coupling_rust = lambda omegas, flat, n: 4.0
        spo_kernel.sync_convergence_rate_rust = lambda flat, omegas, n, gamma_max: (
            5.0 + gamma_max
        )
        monkeypatch.setitem(sys.modules, "spo_kernel", spo_kernel)

        bundle = s_mod._load_rust_bundle()

        assert bundle["fv"](np.zeros(4), 2) == 2.0
        np.testing.assert_allclose(bundle["fvec"](np.zeros(4), 2), [0.0, 1.0])
        assert bundle["sg"](np.zeros(4), 2) == 3.0
        assert bundle["kc"](np.zeros(2), np.zeros(4), 2) == 4.0
        assert bundle["scr"](np.zeros(4), np.zeros(2), 2, 0.25) == 5.25

    def test_rust_active_backend_routes_derived_metrics_to_fast_paths(
        self, monkeypatch
    ):
        calls = []
        knm = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        knm_gap = np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 2.0],
                [0.0, 2.0, 0.0],
            ],
            dtype=np.float64,
        )
        omegas = np.array([0.0, 2.0], dtype=np.float64)

        def rust_bundle():
            return {
                "fv": lambda flat, n: calls.append(("fv", flat.copy(), n)) or 2.0,
                "fvec": lambda flat, n: (
                    calls.append(("fvec", flat.copy(), n)) or np.array([1.0, -1.0])
                ),
                "sg": lambda flat, n: calls.append(("sg", flat.copy(), n)) or 0.0,
                "kc": lambda o, flat, n: (
                    calls.append(("kc", o.copy(), flat.copy(), n)) or 1.0
                ),
                "scr": lambda flat, o, n, gamma_max: (
                    calls.append(("scr", flat.copy(), o.copy(), n, gamma_max)) or 0.5
                ),
            }

        monkeypatch.setattr(s_mod, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(s_mod, "_RUST_CACHE", None)
        monkeypatch.setattr(s_mod, "_load_rust_bundle", rust_bundle)

        assert fiedler_value(knm) == 2.0
        np.testing.assert_allclose(fiedler_vector(knm), [1.0, -1.0])
        assert spectral_gap(knm_gap) == 0.0
        assert critical_coupling(omegas, knm) == 1.0
        assert sync_convergence_rate(knm, omegas, gamma_max=0.25) == 0.5
        assert [call[0] for call in calls] == ["fv", "fvec", "sg", "kc", "scr"]
