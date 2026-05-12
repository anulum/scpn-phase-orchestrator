# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Cross-backend parity for torus integrator

"""Cross-backend parity for ``TorusEngine.run``.

Tolerances
----------
The alpha-zero branch uses the sincos expansion and agrees
bit-exactly on Rust / Julia / Go / Python. The alpha-nonzero
branch routes through ``atan2`` per step, which introduces
identity-wise ``~1e-15`` drift across backends — still tight
enough for sub-``1e-12`` agreement after 50 steps.
"""

from __future__ import annotations

import contextlib
import math
import sys
import types

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.upde import geometric as g_mod
from scpn_phase_orchestrator.upde.geometric import TorusEngine

TWO_PI = 2.0 * math.pi
TOL_EXPANSION = 1e-12
TOL_ATAN2 = 1e-10


@contextlib.contextmanager
def _force_backend(name: str):
    prev = g_mod.ACTIVE_BACKEND
    g_mod.ACTIVE_BACKEND = name
    try:
        yield
    finally:
        g_mod.ACTIVE_BACKEND = prev


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
    n_steps: int = 50,
    zeta: float = 0.0,
    psi: float = 0.0,
    alpha_nonzero: bool = False,
):
    if backend not in g_mod.AVAILABLE_BACKENDS:
        pytest.skip(f"backend {backend!r} unavailable")
    theta, omegas, knm, alpha = _problem(seed, n, alpha_nonzero)
    eng = TorusEngine(n, 0.01)
    with _force_backend(backend):
        return eng.run(theta, omegas, knm, zeta, psi, alpha, n_steps=n_steps)


class TestAlphaZero:
    def test_rust(self):
        ref = _run_backend("python", 0)
        got = _run_backend("rust", 0)
        assert np.max(np.abs(got - ref)) < TOL_EXPANSION

    def test_julia(self):
        ref = _run_backend("python", 1)
        got = _run_backend("julia", 1)
        assert np.max(np.abs(got - ref)) < TOL_EXPANSION

    def test_go(self):
        ref = _run_backend("python", 2)
        got = _run_backend("go", 2)
        assert np.max(np.abs(got - ref)) < TOL_EXPANSION

    def test_mojo(self):
        ref = _run_backend("python", 3, n=5)
        got = _run_backend("mojo", 3, n=5)
        assert np.max(np.abs(got - ref)) < TOL_ATAN2


class TestAlphaNonZero:
    def test_rust(self):
        ref = _run_backend("python", 4, alpha_nonzero=True, zeta=0.5, psi=1.2)
        got = _run_backend("rust", 4, alpha_nonzero=True, zeta=0.5, psi=1.2)
        assert np.max(np.abs(got - ref)) < TOL_ATAN2

    def test_julia(self):
        ref = _run_backend("python", 5, alpha_nonzero=True, zeta=0.5, psi=1.2)
        got = _run_backend("julia", 5, alpha_nonzero=True, zeta=0.5, psi=1.2)
        assert np.max(np.abs(got - ref)) < TOL_ATAN2

    def test_go(self):
        ref = _run_backend("python", 6, alpha_nonzero=True, zeta=0.5, psi=1.2)
        got = _run_backend("go", 6, alpha_nonzero=True, zeta=0.5, psi=1.2)
        assert np.max(np.abs(got - ref)) < TOL_ATAN2


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
        if "rust" not in g_mod.AVAILABLE_BACKENDS:
            pytest.skip("rust unavailable")
        ref = _run_backend("python", seed, n=n)
        got = _run_backend("rust", seed, n=n)
        assert np.max(np.abs(got - ref)) < TOL_EXPANSION

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
        if "go" not in g_mod.AVAILABLE_BACKENDS:
            pytest.skip("go unavailable")
        ref = _run_backend("python", seed, n=n)
        got = _run_backend("go", seed, n=n)
        assert np.max(np.abs(got - ref)) < TOL_EXPANSION


class TestBackendLoaderContracts:
    def test_rust_loader_returns_float64_torus_phases(self, monkeypatch):
        calls = {}

        def torus_run_rust(
            phases, omegas, knm_flat, alpha_flat, n, zeta, psi, dt, n_steps
        ):
            calls["params"] = (n, zeta, psi, dt, n_steps)
            calls["contiguous"] = (
                phases.flags.c_contiguous,
                omegas.flags.c_contiguous,
                knm_flat.flags.c_contiguous,
                alpha_flat.flags.c_contiguous,
            )
            return phases + dt * n_steps * omegas + zeta * np.sin(psi - phases)

        fake_spo = types.ModuleType("spo_kernel")
        fake_spo.torus_run_rust = torus_run_rust
        monkeypatch.setitem(sys.modules, "spo_kernel", fake_spo)

        fn = g_mod._load_rust_fn()
        phases = np.array([0.1, 0.3], dtype=np.float64)
        omegas = np.array([1.0, -0.5], dtype=np.float64)
        knm_flat = np.zeros(4, dtype=np.float64)
        alpha_flat = np.zeros(4, dtype=np.float64)

        got = fn(phases, omegas, knm_flat, alpha_flat, 2, 0.2, 1.1, 0.01, 5)

        np.testing.assert_allclose(
            got,
            phases + 0.05 * omegas + 0.2 * np.sin(1.1 - phases),
        )
        assert got.dtype == np.float64
        assert calls == {
            "params": (2, 0.2, 1.1, 0.01, 5),
            "contiguous": (True, True, True, True),
        }

    def test_optional_backend_loaders_return_callable_numeric_kernels(
        self, monkeypatch
    ):
        def install_backend(
            module_name: str, function_name: str, offset: float
        ) -> None:
            module = types.ModuleType(module_name)
            module.loaded = False

            def _ensure_exe() -> None:
                module.loaded = True

            def _load_lib() -> None:
                module.loaded = True

            def kernel(phases, omegas, knm_flat, alpha_flat, n, zeta, psi, dt, n_steps):
                return (phases + offset + dt * n_steps * omegas) % TWO_PI

            module._ensure_exe = _ensure_exe
            module._load_lib = _load_lib
            setattr(module, function_name, kernel)
            monkeypatch.setitem(sys.modules, module_name, module)

        monkeypatch.setitem(sys.modules, "juliacall", types.ModuleType("juliacall"))
        install_backend(
            "scpn_phase_orchestrator.upde._geometric_mojo",
            "torus_run_mojo",
            0.10,
        )
        install_backend(
            "scpn_phase_orchestrator.upde._geometric_julia",
            "torus_run_julia",
            0.20,
        )
        install_backend(
            "scpn_phase_orchestrator.upde._geometric_go",
            "torus_run_go",
            0.30,
        )

        phases = np.array([0.2, 0.8], dtype=np.float64)
        omegas = np.array([1.0, -0.25], dtype=np.float64)
        args = (phases, omegas, np.zeros(4), np.zeros(4), 2, 0.0, 0.0, 0.01, 3)

        for loader, offset in (
            (g_mod._load_mojo_fn, 0.10),
            (g_mod._load_julia_fn, 0.20),
            (g_mod._load_go_fn, 0.30),
        ):
            got = loader()(*args)
            np.testing.assert_allclose(got, (phases + offset + 0.03 * omegas) % TWO_PI)
