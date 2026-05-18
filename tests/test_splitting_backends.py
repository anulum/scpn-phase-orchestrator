# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Cross-backend parity for Strang splitting

"""Cross-backend parity for ``SplittingEngine.run``."""

from __future__ import annotations

import contextlib
import math
import sys
import types
from typing import get_type_hints

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _splitting_go,
    _splitting_julia,
    _splitting_mojo,
)
from scpn_phase_orchestrator.upde import splitting as sp_mod
from scpn_phase_orchestrator.upde.splitting import SplittingEngine

splitting_run_go = _splitting_go.splitting_run_go
splitting_run_julia = _splitting_julia.splitting_run_julia
splitting_run_mojo = _splitting_mojo.splitting_run_mojo

TWO_PI = 2.0 * math.pi
TOL = 1e-12


@contextlib.contextmanager
def _force_backend(name: str):
    prev = sp_mod.ACTIVE_BACKEND
    sp_mod.ACTIVE_BACKEND = name
    try:
        yield
    finally:
        sp_mod.ACTIVE_BACKEND = prev


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
    zeta: float = 0.0,
    psi: float = 0.0,
    alpha_nonzero: bool = False,
):
    if backend not in sp_mod.AVAILABLE_BACKENDS:
        pytest.skip(f"backend {backend!r} unavailable")
    theta, omegas, knm, alpha = _problem(seed, n, alpha_nonzero)
    eng = SplittingEngine(n, 0.01)
    with _force_backend(backend):
        return eng.run(theta, omegas, knm, zeta, psi, alpha, n_steps=n_steps)


class TestAlphaZero:
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


class TestAlphaNonZero:
    def test_rust(self):
        ref = _run_backend("python", 4, alpha_nonzero=True, zeta=0.5, psi=1.1)
        got = _run_backend("rust", 4, alpha_nonzero=True, zeta=0.5, psi=1.1)
        assert np.max(np.abs(got - ref)) < TOL

    def test_julia(self):
        ref = _run_backend("python", 5, alpha_nonzero=True, zeta=0.5, psi=1.1)
        got = _run_backend("julia", 5, alpha_nonzero=True, zeta=0.5, psi=1.1)
        assert np.max(np.abs(got - ref)) < TOL

    def test_go(self):
        ref = _run_backend("python", 6, alpha_nonzero=True, zeta=0.5, psi=1.1)
        got = _run_backend("go", 6, alpha_nonzero=True, zeta=0.5, psi=1.1)
        assert np.max(np.abs(got - ref)) < TOL


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
        if "rust" not in sp_mod.AVAILABLE_BACKENDS:
            pytest.skip("rust unavailable")
        ref = _run_backend("python", seed, n=n)
        got = _run_backend("rust", seed, n=n)
        assert np.max(np.abs(got - ref)) < TOL

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
        if "go" not in sp_mod.AVAILABLE_BACKENDS:
            pytest.skip("go unavailable")
        ref = _run_backend("python", seed, n=n)
        got = _run_backend("go", seed, n=n)
        assert np.max(np.abs(got - ref)) < TOL


class TestBackendTypingContracts:
    @pytest.mark.parametrize(
        ("fn", "label"),
        [
            (splitting_run_go, "go"),
            (splitting_run_julia, "julia"),
            (splitting_run_mojo, "mojo"),
        ],
    )
    def test_backend_annotations_use_float64_ndarray(self, fn, label: str) -> None:
        hints = get_type_hints(fn)
        for name in ("phases", "omegas", "knm_flat", "alpha_flat", "return"):
            text = str(hints[name])
            assert "numpy.ndarray" in text, f"{label}:{name} missing ndarray annotation"
            assert "numpy.float64" in text, f"{label}:{name} missing float64 annotation"


class TestBackendLoaderDispatch:
    def test_rust_loader_preserves_contiguous_float64_contract(self, monkeypatch):
        calls: list[tuple[np.dtype, bool, int, float]] = []

        def fake_rust(
            phases,
            omegas,
            knm_flat,
            alpha_flat,
            n,
            zeta,
            psi,
            dt,
            n_steps,
        ):
            calls.append((phases.dtype, phases.flags.c_contiguous, int(n), float(dt)))
            assert omegas.flags.c_contiguous
            assert knm_flat.flags.c_contiguous
            assert alpha_flat.flags.c_contiguous
            assert n_steps == 3
            assert zeta == 0.25
            assert psi == 0.75
            return phases + dt * omegas

        monkeypatch.setitem(
            sys.modules,
            "spo_kernel",
            types.SimpleNamespace(splitting_run_rust=fake_rust),
        )
        run = sp_mod._load_rust_fn()
        phases = np.arange(6.0, dtype=np.float64)[::2]
        omegas = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        knm = np.eye(3, dtype=np.float64).ravel()
        alpha = np.zeros(9, dtype=np.float64)

        got = run(phases, omegas, knm, alpha, 3, 0.25, 0.75, 0.5, 3)

        np.testing.assert_allclose(got, np.array([0.05, 2.1, 4.15]))
        assert calls == [(np.dtype("float64"), True, 3, 0.5)]

    def test_toolchain_loaders_return_callable_with_dependencies_stubbed(
        self,
        monkeypatch,
    ):
        monkeypatch.setitem(sys.modules, "juliacall", types.SimpleNamespace())
        monkeypatch.setattr(sp_mod, "_BACKEND_CACHE", {})
        monkeypatch.setattr(
            (
                "scpn_phase_orchestrator.experimental.accelerators.upde"
                "._splitting_mojo._ensure_exe"
            ),
            lambda: object(),
        )
        monkeypatch.setattr(
            (
                "scpn_phase_orchestrator.experimental.accelerators.upde"
                "._splitting_go._load_lib"
            ),
            lambda: object(),
        )

        assert callable(sp_mod._load_mojo_fn())
        assert callable(sp_mod._load_julia_fn())
        assert callable(sp_mod._load_go_fn())

    def test_probe_marks_backend_failure_as_infinite(self, monkeypatch):
        def failing_backend():
            raise RuntimeError("backend unavailable")

        monkeypatch.setitem(sp_mod._LOADERS, "rust", failing_backend)
        monkeypatch.setattr(sp_mod, "_BACKEND_CACHE", {})

        assert sp_mod._splitting_probe_seconds("rust") == float("inf")
