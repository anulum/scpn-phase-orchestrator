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
import sys
import types

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


class TestOptionalLoaderSuccessPaths:
    def test_rust_loader_wraps_spo_kernel_function(self, monkeypatch):
        calls = []

        def simplicial_run_rust(*args):
            calls.append(args)
            return [0.1, 0.2, 0.3]

        fake_spo = types.ModuleType("spo_kernel")
        fake_spo.simplicial_run_rust = simplicial_run_rust
        monkeypatch.setitem(sys.modules, "spo_kernel", fake_spo)

        run = s_mod._load_rust_fn()
        out = run(
            np.array([0.0, 0.1, 0.2], dtype=np.float64),
            np.ones(3, dtype=np.float64),
            np.zeros(9, dtype=np.float64),
            np.zeros(9, dtype=np.float64),
            3,
            0.2,
            0.3,
            0.4,
            0.01,
            2,
        )
        np.testing.assert_allclose(out, [0.1, 0.2, 0.3], atol=1e-12)
        assert calls
        assert calls[0][4] == 3
        assert calls[0][9] == 2

    def test_mojo_loader_runs_availability_probe(self, monkeypatch):
        fake_mojo = types.ModuleType("scpn_phase_orchestrator.upde._simplicial_mojo")
        probe_calls = []

        def _ensure_exe():
            probe_calls.append(True)

        def simplicial_run_mojo(*args):
            return np.array([0.4, 0.5], dtype=np.float64)

        fake_mojo._ensure_exe = _ensure_exe
        fake_mojo.simplicial_run_mojo = simplicial_run_mojo
        monkeypatch.setitem(
            sys.modules,
            "scpn_phase_orchestrator.upde._simplicial_mojo",
            fake_mojo,
        )

        run = s_mod._load_mojo_fn()
        np.testing.assert_allclose(run(), [0.4, 0.5], atol=1e-12)
        assert probe_calls == [True]

    def test_julia_loader_requires_juliacall_and_returns_runner(self, monkeypatch):
        fake_juliacall = types.ModuleType("juliacall")
        fake_julia = types.ModuleType("scpn_phase_orchestrator.upde._simplicial_julia")

        def simplicial_run_julia(*args):
            return np.array([0.6, 0.7], dtype=np.float64)

        fake_julia.simplicial_run_julia = simplicial_run_julia
        monkeypatch.setitem(sys.modules, "juliacall", fake_juliacall)
        monkeypatch.setitem(
            sys.modules,
            "scpn_phase_orchestrator.upde._simplicial_julia",
            fake_julia,
        )

        run = s_mod._load_julia_fn()
        np.testing.assert_allclose(run(), [0.6, 0.7], atol=1e-12)

    def test_go_loader_runs_shared_library_probe(self, monkeypatch):
        fake_go = types.ModuleType("scpn_phase_orchestrator.upde._simplicial_go")
        probe_calls = []

        def _load_lib():
            probe_calls.append(True)

        def simplicial_run_go(*args):
            return np.array([0.8, 0.9], dtype=np.float64)

        fake_go._load_lib = _load_lib
        fake_go.simplicial_run_go = simplicial_run_go
        monkeypatch.setitem(
            sys.modules,
            "scpn_phase_orchestrator.upde._simplicial_go",
            fake_go,
        )

        run = s_mod._load_go_fn()
        np.testing.assert_allclose(run(), [0.8, 0.9], atol=1e-12)
        assert probe_calls == [True]
