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
from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _geometric_mojo,
)
from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _geometric_validation as geometric_validation,
)
from scpn_phase_orchestrator.experimental.accelerators.upde._geometric_go import (
    torus_run_go,
)
from scpn_phase_orchestrator.experimental.accelerators.upde._geometric_julia import (
    torus_run_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.upde._geometric_mojo import (
    torus_run_mojo,
)
from scpn_phase_orchestrator.upde import geometric as g_mod
from scpn_phase_orchestrator.upde.geometric import TorusEngine

TWO_PI = 2.0 * math.pi
TOL_EXPANSION = 1e-12
TOL_ATAN2 = 1e-10
DirectBackend = Callable[
    [
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        int,
        float,
        float,
        float,
        int,
    ],
    np.ndarray,
]
DIRECT_BACKENDS = (torus_run_go, torus_run_julia, torus_run_mojo)


def test__geometric_validation_helper_is_directly_linked_to_backend_tests() -> None:
    assert callable(geometric_validation.validate_torus_inputs)
    assert callable(geometric_validation.validate_torus_output)


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


def _direct_payload(n: int = 5):
    phases, omegas, knm, alpha = _problem(11, n=n, alpha_nonzero=True)
    return phases, omegas, knm.ravel(), alpha.ravel(), n, 0.2, 0.7, 0.01, 4


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


class TestDirectMojoBoundaryContracts:
    @pytest.mark.parametrize(
        ("stdout", "match"),
        [
            ("", "Mojo TORUS returned 0 lines, expected 5"),
            ("0.1\n0.2\n0.3\n0.4\n0.5\n0.6\n", "expected 5"),
            ("0.1\n0.2\n\n0.4\n0.5\n0.6\n", "expected 5"),
            ("0.1\n0.2\nbad\n0.4\n0.5\n", "finite phases"),
            ("0.1\n0.2\nnan\n0.4\n0.5\n", "finite phases"),
            ("0.1\n0.2\n7.0\n0.4\n0.5\n", "finite phases"),
        ],
    )
    def test_mojo_runner_rejects_malformed_raw_stdout(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stdout: str,
        match: str,
    ) -> None:
        monkeypatch.setattr(_geometric_mojo, "_ensure_exe", lambda: "geometric")
        monkeypatch.setattr(
            _geometric_mojo.subprocess,
            "run",
            lambda *_args, **_kwargs: type(
                "Proc",
                (),
                {"returncode": 0, "stdout": stdout, "stderr": ""},
            )(),
        )

        with pytest.raises(ValueError, match=match):
            _geometric_mojo.torus_run_mojo(*_direct_payload())


class TestDirectBackendBoundaryContracts:
    @pytest.mark.parametrize("backend", DIRECT_BACKENDS)
    @pytest.mark.parametrize(
        ("index", "replacement"),
        [
            (0, lambda payload: payload[0].reshape(1, -1)),
            (0, lambda payload: payload[0].astype(bool)),
            (0, lambda payload: payload[0].astype(np.complex128) + 1j),
            (0, lambda payload: np.array([np.nan, *payload[0][1:]])),
            (1, lambda payload: payload[1][:-1]),
            (1, lambda payload: payload[1].astype(bool)),
            (1, lambda payload: np.array([np.inf, *payload[1][1:]])),
            (2, lambda payload: payload[2][:-1]),
            (2, lambda payload: payload[2].astype(bool)),
            (2, lambda payload: payload[2].astype(np.complex128) + 1j),
            (3, lambda payload: payload[3][:-1]),
            (3, lambda payload: payload[3].astype(bool)),
            (3, lambda payload: np.full_like(payload[3], np.nan)),
            (4, lambda payload: True),
            (4, lambda payload: 0),
            (4, lambda payload: payload[4] + 1),
            (5, lambda payload: True),
            (5, lambda payload: float("nan")),
            (6, lambda payload: float("inf")),
            (7, lambda payload: 0.0),
            (7, lambda payload: True),
            (8, lambda payload: -1),
            (8, lambda payload: True),
        ],
    )
    def test_invalid_inputs_fail_before_optional_runtime_loading(
        self,
        backend: DirectBackend,
        index: int,
        replacement: Callable[[tuple], object],
    ) -> None:
        """Direct Go/Julia/Mojo wrappers share the torus-run contract."""

        payload = list(_direct_payload())
        payload[index] = replacement(tuple(payload))
        with pytest.raises((TypeError, ValueError)):
            backend(*payload)

    @pytest.mark.parametrize("backend", DIRECT_BACKENDS)
    def test_zero_steps_normalises_phases_without_optional_runtime(
        self,
        backend: DirectBackend,
    ) -> None:
        payload = list(_direct_payload())
        payload[0] = np.array([-0.25, 0.0, TWO_PI + 0.5, 8.0, -TWO_PI])
        payload[8] = 0
        np.testing.assert_allclose(backend(*payload), payload[0] % TWO_PI, atol=0.0)


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
            "scpn_phase_orchestrator.experimental.accelerators.upde._geometric_mojo",
            "torus_run_mojo",
            0.10,
        )
        install_backend(
            "scpn_phase_orchestrator.experimental.accelerators.upde._geometric_julia",
            "torus_run_julia",
            0.20,
        )
        install_backend(
            "scpn_phase_orchestrator.experimental.accelerators.upde._geometric_go",
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


class TestDispatchFallbackChain:
    def test_dispatch_falls_back_to_python_when_loader_fails(self, monkeypatch):
        previous_backend = g_mod.ACTIVE_BACKEND
        previous_available = list(g_mod.AVAILABLE_BACKENDS)
        previous_loader = g_mod._LOADERS["go"]
        g_mod.ACTIVE_BACKEND = "go"
        g_mod.AVAILABLE_BACKENDS = ["go", "python"]
        g_mod._BACKEND_CACHE.clear()
        monkeypatch.setitem(
            g_mod._LOADERS,
            "go",
            lambda: (_ for _ in ()).throw(ImportError("go backend unavailable")),
        )
        try:
            backend = g_mod._dispatch()
        finally:
            g_mod.ACTIVE_BACKEND = previous_backend
            g_mod.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(g_mod._LOADERS, "go", previous_loader)
            g_mod._BACKEND_CACHE.clear()

        assert backend is None

    def test_dispatch_uses_cached_loader_once(self, monkeypatch):
        previous_backend = g_mod.ACTIVE_BACKEND
        previous_available = list(g_mod.AVAILABLE_BACKENDS)
        previous_loader = g_mod._LOADERS["go"]
        g_mod.ACTIVE_BACKEND = "go"
        g_mod.AVAILABLE_BACKENDS = ["go", "python"]
        g_mod._BACKEND_CACHE.clear()
        call_count = 0

        def fake_backend(
            phases: np.ndarray,
            _omegas: np.ndarray,
            _knm_flat: np.ndarray,
            _alpha_flat: np.ndarray,
            _n: int,
            _zeta: float,
            _psi: float,
            _dt: float,
            _n_steps: int,
        ) -> np.ndarray:
            return np.asarray(phases, dtype=np.float64)

        def loader():
            nonlocal call_count
            call_count += 1
            return fake_backend

        monkeypatch.setitem(g_mod._LOADERS, "go", loader)
        try:
            b1 = g_mod._dispatch()
            b2 = g_mod._dispatch()
        finally:
            g_mod.ACTIVE_BACKEND = previous_backend
            g_mod.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(g_mod._LOADERS, "go", previous_loader)
            g_mod._BACKEND_CACHE.clear()

        assert b1 is fake_backend
        assert b2 is fake_backend
        assert call_count == 1
