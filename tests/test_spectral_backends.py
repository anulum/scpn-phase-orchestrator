# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Cross-backend parity for coupling.spectral

"""Cross-backend parity for ``coupling.spectral``.

Tolerance notes
---------------
LAPACK-based backends (Rust, Julia, Mojo, Python) agree to
``~1e-15`` on eigenvalues. Go uses ``gonum.EigenSym`` — a
pure-Go symmetric solver — which drifts by up to ``~1e-13``
on well-conditioned Laplacians.

Eigenvectors carry a ``±sign`` ambiguity inherent to
eigendecomposition. Parity is checked on ``|v|`` element-wise
or via the eigen-equation residual ``||L·v − λ·v||`` rather
than on raw values.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.coupling import spectral as s_mod
from scpn_phase_orchestrator.coupling.spectral import (
    fiedler_value,
    fiedler_vector,
    graph_laplacian,
    spectral_eig,
    spectral_gap,
)

TOL_LAPACK = 1e-12
TOL_GONUM = 1e-11  # gonum EigenSym vs LAPACK
TOL_MOJO = 1e-10  # subprocess text round-trip on top of LAPACK


@contextlib.contextmanager
def _force(name: str):
    prev = s_mod.ACTIVE_BACKEND
    s_mod.ACTIVE_BACKEND = name
    s_mod._PRIM_CACHE = None
    s_mod._RUST_CACHE = None
    try:
        yield
    finally:
        s_mod.ACTIVE_BACKEND = prev
        s_mod._PRIM_CACHE = None
        s_mod._RUST_CACHE = None


def _problem(seed: int, n: int = 6):
    rng = np.random.default_rng(seed)
    W = rng.uniform(0, 1, (n, n))
    W = (W + W.T) / 2
    np.fill_diagonal(W, 0.0)
    return W


def _run_backend(backend: str, seed: int, n: int = 6):
    if backend not in s_mod.AVAILABLE_BACKENDS:
        pytest.skip(f"backend {backend!r} unavailable")
    W = _problem(seed, n)
    with _force(backend):
        lam2 = fiedler_value(W)
        v2 = fiedler_vector(W)
        gap = spectral_gap(W)
        eigvals, _ = spectral_eig(W)
    return lam2, v2, gap, eigvals


class TestEigenvalueParity:
    def test_rust(self):
        _, _, _, ref = _run_backend("python", 0)
        _, _, _, got = _run_backend("rust", 0)
        assert np.max(np.abs(got - ref)) < TOL_LAPACK

    def test_julia(self):
        _, _, _, ref = _run_backend("python", 1)
        _, _, _, got = _run_backend("julia", 1)
        assert np.max(np.abs(got - ref)) < TOL_LAPACK

    def test_go(self):
        _, _, _, ref = _run_backend("python", 2)
        _, _, _, got = _run_backend("go", 2)
        assert np.max(np.abs(got - ref)) < TOL_GONUM

    def test_mojo(self):
        _, _, _, ref = _run_backend("python", 3, n=5)
        _, _, _, got = _run_backend("mojo", 3, n=5)
        assert np.max(np.abs(got - ref)) < TOL_MOJO


class TestFiedlerValueParity:
    """``λ₂`` is a specific eigenvalue — sign-stable, so
    absolute difference works directly."""

    def test_rust(self):
        ref, *_ = _run_backend("python", 4)
        got, *_ = _run_backend("rust", 4)
        assert abs(got - ref) < TOL_LAPACK

    def test_julia(self):
        ref, *_ = _run_backend("python", 5)
        got, *_ = _run_backend("julia", 5)
        assert abs(got - ref) < TOL_LAPACK

    def test_go(self):
        ref, *_ = _run_backend("python", 6)
        got, *_ = _run_backend("go", 6)
        assert abs(got - ref) < TOL_GONUM

    def test_mojo(self):
        ref, *_ = _run_backend("python", 7, n=5)
        got, *_ = _run_backend("mojo", 7, n=5)
        assert abs(got - ref) < TOL_MOJO


class TestFiedlerVectorResidual:
    """Verify the eigen-equation ``L·v₂ = λ₂·v₂`` rather than
    comparing raw eigenvector values — the ``±sign`` ambiguity
    makes element-wise comparison unreliable across backends."""

    def _check(self, backend: str, seed: int, n: int, tol: float):
        if backend not in s_mod.AVAILABLE_BACKENDS:
            pytest.skip(f"backend {backend!r} unavailable")
        W = _problem(seed, n)
        L = graph_laplacian(W)
        with _force(backend):
            lam2 = fiedler_value(W)
            v2 = fiedler_vector(W)
        residual = L @ v2 - lam2 * v2
        assert np.max(np.abs(residual)) < tol

    def test_rust(self):
        self._check("rust", 10, 6, TOL_LAPACK)

    def test_julia(self):
        self._check("julia", 11, 6, TOL_LAPACK)

    def test_go(self):
        self._check("go", 12, 6, TOL_GONUM)

    def test_mojo(self):
        self._check("mojo", 13, 5, TOL_MOJO)


class TestHypothesisParity:
    @given(
        n=st.integers(min_value=3, max_value=8),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=6,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_rust_hypothesis(self, n, seed):
        if "rust" not in s_mod.AVAILABLE_BACKENDS:
            pytest.skip("rust unavailable")
        _, _, _, ref = _run_backend("python", seed, n=n)
        _, _, _, got = _run_backend("rust", seed, n=n)
        assert np.max(np.abs(got - ref)) < TOL_LAPACK

    @given(
        n=st.integers(min_value=3, max_value=8),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=6,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_go_hypothesis(self, n, seed):
        if "go" not in s_mod.AVAILABLE_BACKENDS:
            pytest.skip("go unavailable")
        _, _, _, ref = _run_backend("python", seed, n=n)
        _, _, _, got = _run_backend("go", seed, n=n)
        assert np.max(np.abs(got - ref)) < TOL_GONUM
