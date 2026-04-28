# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Cross-backend parity for simplicial Kuramoto

"""Cross-backend parity for the ``simplicial_run`` kernel."""

from __future__ import annotations

import contextlib
import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.upde import simplicial as s_mod
from scpn_phase_orchestrator.upde.simplicial import SimplicialEngine

TWO_PI = 2.0 * math.pi
TOL = 1e-12


@contextlib.contextmanager
def _force_backend(name: str):
    prev = s_mod.ACTIVE_BACKEND
    s_mod.ACTIVE_BACKEND = name
    try:
        yield
    finally:
        s_mod.ACTIVE_BACKEND = prev


def _problem(seed: int, n: int = 6, alpha_nonzero: bool = False):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, TWO_PI, n)
    omegas = rng.normal(1.0, 0.2, n)
    knm = rng.uniform(0, 0.3, (n, n))
    np.fill_diagonal(knm, 0.0)
    if alpha_nonzero:
        alpha = rng.uniform(-0.3, 0.3, (n, n))
        np.fill_diagonal(alpha, 0.0)
    else:
        alpha = np.zeros((n, n))
    return theta, omegas, knm, alpha


def _run_backend(
    backend: str,
    seed: int,
    n: int = 6,
    n_steps: int = 20,
    sigma2: float = 0.5,
    zeta: float = 0.0,
    psi: float = 0.0,
    alpha_nonzero: bool = False,
):
    if backend not in s_mod.AVAILABLE_BACKENDS:
        pytest.skip(f"backend {backend!r} unavailable")
    theta, omegas, knm, alpha = _problem(seed, n, alpha_nonzero)
    eng = SimplicialEngine(n, 0.01, sigma2=sigma2)
    with _force_backend(backend):
        return eng.run(theta, omegas, knm, zeta, psi, alpha, n_steps=n_steps)


class TestParityAlphaZero:
    def test_rust(self):
        ref = _run_backend("python", 0)
        got = _run_backend("rust", 0)
        assert np.max(np.abs(got - ref)) < TOL

    def test_julia(self):
        ref = _run_backend("python", 1)
        got = _run_backend("julia", 1)
        assert np.max(np.abs(got - ref)) < TOL

    def test_go(self):
        ref = _run_backend("python", 2)
        got = _run_backend("go", 2)
        assert np.max(np.abs(got - ref)) < TOL

    def test_mojo(self):
        ref = _run_backend("python", 3, n=5)
        got = _run_backend("mojo", 3, n=5)
        assert np.max(np.abs(got - ref)) < 1e-10


class TestParityAlphaNonZero:
    def test_rust(self):
        ref = _run_backend("python", 4, alpha_nonzero=True, zeta=0.3, psi=1.1)
        got = _run_backend("rust", 4, alpha_nonzero=True, zeta=0.3, psi=1.1)
        assert np.max(np.abs(got - ref)) < TOL

    def test_julia(self):
        ref = _run_backend("python", 5, alpha_nonzero=True, zeta=0.3, psi=1.1)
        got = _run_backend("julia", 5, alpha_nonzero=True, zeta=0.3, psi=1.1)
        assert np.max(np.abs(got - ref)) < TOL

    def test_go(self):
        ref = _run_backend("python", 6, alpha_nonzero=True, zeta=0.3, psi=1.1)
        got = _run_backend("go", 6, alpha_nonzero=True, zeta=0.3, psi=1.1)
        assert np.max(np.abs(got - ref)) < TOL


class TestSigma2Zero:
    """σ₂ = 0 must take the pure-pairwise branch unchanged."""

    def test_rust_sigma2_zero(self):
        ref = _run_backend("python", 10, sigma2=0.0)
        got = _run_backend("rust", 10, sigma2=0.0)
        assert np.max(np.abs(got - ref)) < TOL

    def test_go_sigma2_zero(self):
        ref = _run_backend("python", 11, sigma2=0.0)
        got = _run_backend("go", 11, sigma2=0.0)
        assert np.max(np.abs(got - ref)) < TOL


class TestHypothesisParity:
    @given(
        n=st.integers(min_value=3, max_value=8),
        sigma2=st.floats(min_value=0.0, max_value=1.5),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=6,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_rust_hypothesis(self, n, sigma2, seed):
        if "rust" not in s_mod.AVAILABLE_BACKENDS:
            pytest.skip("rust unavailable")
        ref = _run_backend("python", seed, n=n, sigma2=sigma2)
        got = _run_backend("rust", seed, n=n, sigma2=sigma2)
        assert np.max(np.abs(got - ref)) < TOL

    @given(
        n=st.integers(min_value=3, max_value=8),
        sigma2=st.floats(min_value=0.0, max_value=1.5),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=6,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_go_hypothesis(self, n, sigma2, seed):
        if "go" not in s_mod.AVAILABLE_BACKENDS:
            pytest.skip("go unavailable")
        ref = _run_backend("python", seed, n=n, sigma2=sigma2)
        got = _run_backend("go", seed, n=n, sigma2=sigma2)
        assert np.max(np.abs(got - ref)) < TOL
