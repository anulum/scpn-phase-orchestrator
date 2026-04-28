# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Cross-backend parity for hypergraph Kuramoto

"""Cross-backend parity for ``HypergraphEngine.run``.

The five backends share the same sincos-expansion semantics on
the alpha-zero branch and the same direct-``sin(diff)`` form on
the alpha-nonzero branch, so bit-exact parity (``0.0``) is
expected across Rust / Julia / Go / Python. Mojo drifts only by
its subprocess text-round-trip epsilon.
"""

from __future__ import annotations

import contextlib
import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.upde import hypergraph as h_mod
from scpn_phase_orchestrator.upde.hypergraph import HypergraphEngine

TWO_PI = 2.0 * math.pi
TOL = 1e-12


@contextlib.contextmanager
def _force_backend(name: str):
    prev = h_mod.ACTIVE_BACKEND
    h_mod.ACTIVE_BACKEND = name
    try:
        yield
    finally:
        h_mod.ACTIVE_BACKEND = prev


def _problem(seed: int, n: int = 6):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, TWO_PI, n)
    omega = rng.normal(0.5, 0.2, n)
    knm = rng.uniform(0, 0.3, (n, n))
    np.fill_diagonal(knm, 0.0)
    return theta, omega, knm


def _run_backend(
    backend: str,
    seed: int,
    n: int = 6,
    n_steps: int = 20,
    *,
    alpha=None,
    zeta: float = 0.0,
    psi: float = 0.0,
):
    if backend not in h_mod.AVAILABLE_BACKENDS:
        pytest.skip(f"backend {backend!r} unavailable")
    theta, omega, knm = _problem(seed, n)
    eng = HypergraphEngine(n, 0.01)
    eng.add_edge((0, 1, 2), strength=0.4)
    eng.add_edge((1, 3, 4, 5), strength=0.25)
    with _force_backend(backend):
        return eng.run(
            theta,
            omega,
            n_steps=n_steps,
            pairwise_knm=knm,
            alpha=alpha,
            zeta=zeta,
            psi=psi,
        )


class TestBackendParity:
    def test_rust_matches_python(self):
        ref = _run_backend("python", 0)
        got = _run_backend("rust", 0)
        assert np.max(np.abs(got - ref)) < TOL

    def test_julia_matches_python(self):
        ref = _run_backend("python", 1)
        got = _run_backend("julia", 1)
        assert np.max(np.abs(got - ref)) < TOL

    def test_go_matches_python(self):
        ref = _run_backend("python", 2)
        got = _run_backend("go", 2)
        assert np.max(np.abs(got - ref)) < TOL

    def test_mojo_matches_python(self):
        ref = _run_backend("python", 3, n=6)
        got = _run_backend("mojo", 3, n=6)
        assert np.max(np.abs(got - ref)) < 1e-10


class TestAlphaNonZero:
    """The alpha-nonzero branch uses the direct ``sin(diff)`` form."""

    def _run(self, backend: str, seed: int):
        n = 6  # ``_run_backend`` seeds edges up to index 5
        rng = np.random.default_rng(seed + 100)
        theta, omega, knm = _problem(seed, n)
        alpha = rng.uniform(-0.3, 0.3, (n, n))
        np.fill_diagonal(alpha, 0.0)
        return _run_backend(
            backend,
            seed,
            n=n,
            alpha=alpha,
            zeta=0.5,
            psi=1.2,
        )

    def test_rust_alpha(self):
        ref = self._run("python", 5)
        got = self._run("rust", 5)
        assert np.max(np.abs(got - ref)) < TOL

    def test_go_alpha(self):
        ref = self._run("python", 6)
        got = self._run("go", 6)
        assert np.max(np.abs(got - ref)) < TOL

    def test_julia_alpha(self):
        ref = self._run("python", 7)
        got = self._run("julia", 7)
        assert np.max(np.abs(got - ref)) < TOL


class TestNoPairwise:
    """Hypergraph-only coupling (no pairwise K) must still agree."""

    def _run(self, backend: str, seed: int):
        if backend not in h_mod.AVAILABLE_BACKENDS:
            pytest.skip(f"backend {backend!r} unavailable")
        n = 5
        rng = np.random.default_rng(seed)
        theta = rng.uniform(0, TWO_PI, n)
        omega = rng.normal(0.5, 0.2, n)
        eng = HypergraphEngine(n, 0.01)
        eng.add_edge((0, 1, 2), strength=0.5)
        eng.add_edge((2, 3, 4), strength=0.3)
        with _force_backend(backend):
            return eng.run(theta, omega, n_steps=30)

    def test_rust_vs_python(self):
        ref = self._run("python", 9)
        got = self._run("rust", 9)
        assert np.max(np.abs(got - ref)) < TOL

    def test_go_vs_python(self):
        ref = self._run("python", 10)
        got = self._run("go", 10)
        assert np.max(np.abs(got - ref)) < TOL


class TestHypothesisParity:
    @given(
        n=st.integers(min_value=6, max_value=10),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=6,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_rust_hypothesis(self, n, seed):
        if "rust" not in h_mod.AVAILABLE_BACKENDS:
            pytest.skip("rust unavailable")
        ref = _run_backend("python", seed, n=n)
        got = _run_backend("rust", seed, n=n)
        assert np.max(np.abs(got - ref)) < TOL

    @given(
        n=st.integers(min_value=6, max_value=10),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=6,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_go_hypothesis(self, n, seed):
        if "go" not in h_mod.AVAILABLE_BACKENDS:
            pytest.skip("go unavailable")
        ref = _run_backend("python", seed, n=n)
        got = _run_backend("go", seed, n=n)
        assert np.max(np.abs(got - ref)) < TOL
