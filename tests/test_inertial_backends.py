# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Cross-backend parity for inertial Kuramoto

"""Cross-backend parity for the ``inertial_step`` kernel.

All five backends share the same ``sin(θ_j − θ_i) = s_j·c_i −
c_j·s_i`` derivative expansion and the same RK4 combining rule,
so bit-exact agreement is expected on Rust / Julia / Go / Python
and Mojo drifts only by the text-round-trip epsilon.
"""

from __future__ import annotations

import contextlib
import sys
import types

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _inertial_validation as inertial_validation,
)
from scpn_phase_orchestrator.experimental.accelerators.upde._inertial_go import (
    inertial_step_go,
)
from scpn_phase_orchestrator.experimental.accelerators.upde._inertial_julia import (
    inertial_step_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.upde._inertial_mojo import (
    inertial_step_mojo,
)
from scpn_phase_orchestrator.upde import inertial as i_mod
from scpn_phase_orchestrator.upde.inertial import InertialKuramotoEngine

TOL = 1e-12
DIRECT_BACKENDS = (inertial_step_go, inertial_step_julia, inertial_step_mojo)


def test__inertial_validation_helper_is_directly_linked_to_backend_tests() -> None:
    assert callable(inertial_validation.validate_inertial_inputs)
    assert callable(inertial_validation.validate_inertial_output)


@contextlib.contextmanager
def _force_backend(name: str):
    prev = i_mod.ACTIVE_BACKEND
    i_mod.ACTIVE_BACKEND = name
    try:
        yield
    finally:
        i_mod.ACTIVE_BACKEND = prev


def _problem(n: int, seed: int):
    rng = np.random.default_rng(seed)
    return (
        rng.uniform(0, 2 * np.pi, n),
        rng.normal(0, 0.1, n),
        rng.normal(0, 0.5, n),
        (lambda k: (np.fill_diagonal(k, 0), k)[1])(rng.uniform(0, 0.5, (n, n))),
        np.ones(n),
        np.ones(n) * 0.1,
    )


def _run_backend(backend: str, n: int, seed: int):
    if backend not in i_mod.AVAILABLE_BACKENDS:
        pytest.skip(f"backend {backend!r} unavailable")
    theta, od, p, k, m, d = _problem(n, seed)
    eng = InertialKuramotoEngine(n, 0.01)
    with _force_backend(backend):
        return eng.step(theta, od, p, k, m, d)


class TestBackendParity:
    def test_rust_matches_python(self):
        ref_th, ref_od = _run_backend("python", 8, 0)
        got_th, got_od = _run_backend("rust", 8, 0)
        assert np.max(np.abs(got_th - ref_th)) < TOL
        assert np.max(np.abs(got_od - ref_od)) < TOL

    def test_julia_matches_python(self):
        ref_th, ref_od = _run_backend("python", 8, 1)
        got_th, got_od = _run_backend("julia", 8, 1)
        assert np.max(np.abs(got_th - ref_th)) < TOL
        assert np.max(np.abs(got_od - ref_od)) < TOL

    def test_go_matches_python(self):
        ref_th, ref_od = _run_backend("python", 8, 2)
        got_th, got_od = _run_backend("go", 8, 2)
        assert np.max(np.abs(got_th - ref_th)) < TOL
        assert np.max(np.abs(got_od - ref_od)) < TOL

    def test_mojo_matches_python(self):
        ref_th, ref_od = _run_backend("python", 6, 3)
        got_th, got_od = _run_backend("mojo", 6, 3)
        # Text round-trip on a single RK4 step ≤ 1e-14.
        assert np.max(np.abs(got_th - ref_th)) < 1e-12
        assert np.max(np.abs(got_od - ref_od)) < 1e-12


class TestMultiStepParity:
    """After many RK4 steps the backends must still agree tightly."""

    def _ref_run(self, backend: str, n: int, seed: int, steps: int):
        if backend not in i_mod.AVAILABLE_BACKENDS:
            pytest.skip(f"backend {backend!r} unavailable")
        theta, od, p, k, m, d = _problem(n, seed)
        eng = InertialKuramotoEngine(n, 0.01)
        with _force_backend(backend):
            fin_th, fin_od, _, _ = eng.run(
                theta,
                od,
                p,
                k,
                m,
                d,
                n_steps=steps,
            )
        return fin_th, fin_od

    def test_rust_vs_python_50_steps(self):
        ref_th, ref_od = self._ref_run("python", 8, 4, 50)
        got_th, got_od = self._ref_run("rust", 8, 4, 50)
        assert np.max(np.abs(got_th - ref_th)) < 1e-10
        assert np.max(np.abs(got_od - ref_od)) < 1e-10

    def test_go_vs_python_50_steps(self):
        ref_th, ref_od = self._ref_run("python", 8, 5, 50)
        got_th, got_od = self._ref_run("go", 8, 5, 50)
        assert np.max(np.abs(got_th - ref_th)) < 1e-10
        assert np.max(np.abs(got_od - ref_od)) < 1e-10


class TestHypothesisParity:
    @given(
        n=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=6,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_rust_hypothesis(self, n, seed):
        if "rust" not in i_mod.AVAILABLE_BACKENDS:
            pytest.skip("rust unavailable")
        ref_th, ref_od = _run_backend("python", n, seed)
        got_th, got_od = _run_backend("rust", n, seed)
        assert np.max(np.abs(got_th - ref_th)) < TOL
        assert np.max(np.abs(got_od - ref_od)) < TOL

    @given(
        n=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=6,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_go_hypothesis(self, n, seed):
        if "go" not in i_mod.AVAILABLE_BACKENDS:
            pytest.skip("go unavailable")
        ref_th, ref_od = _run_backend("python", n, seed)
        got_th, got_od = _run_backend("go", n, seed)
        assert np.max(np.abs(got_th - ref_th)) < TOL
        assert np.max(np.abs(got_od - ref_od)) < TOL


class TestBackendLoaderContracts:
    def test_rust_loader_adapts_kernel_output_to_float64_arrays(self, monkeypatch):
        calls = {}

        def inertial_step_rust(
            theta, omega_dot, power, knm_flat, inertia, damping, n, dt
        ):
            calls["n"] = n
            calls["dt"] = dt
            calls["contiguous"] = (
                theta.flags.c_contiguous and knm_flat.flags.c_contiguous
            )
            return theta + dt * omega_dot, omega_dot + dt * (power - damping)

        fake_spo = types.ModuleType("spo_kernel")
        fake_spo.inertial_step_rust = inertial_step_rust
        monkeypatch.setitem(sys.modules, "spo_kernel", fake_spo)

        fn = i_mod._load_rust_fn()
        theta = np.array([0.1, 0.3], dtype=np.float64)
        omega = np.array([0.2, -0.1], dtype=np.float64)
        power = np.array([1.0, -0.5], dtype=np.float64)
        knm = np.array([[0.0, 0.2], [0.1, 0.0]], dtype=np.float64)
        inertia = np.ones(2, dtype=np.float64)
        damping = np.array([0.3, 0.4], dtype=np.float64)

        got_theta, got_omega = fn(
            theta, omega, power, knm.ravel(), inertia, damping, 2, 0.05
        )

        np.testing.assert_allclose(got_theta, theta + 0.05 * omega)
        np.testing.assert_allclose(got_omega, omega + 0.05 * (power - damping))
        assert got_theta.dtype == np.float64
        assert got_omega.dtype == np.float64
        assert calls == {"n": 2, "dt": 0.05, "contiguous": True}

    def test_optional_backend_loaders_return_callable_numeric_kernels(
        self, monkeypatch
    ):
        def install_backend(
            module_name: str, function_name: str, offset: float
        ) -> None:
            module = types.ModuleType(module_name)
            module.loaded = False

            def _ensure_exe():
                module.loaded = True

            def _load_lib():
                module.loaded = True

            def kernel(theta, omega_dot, power, knm_flat, inertia, damping, n, dt):
                return theta + offset + dt, omega_dot - offset + dt

            module._ensure_exe = _ensure_exe
            module._load_lib = _load_lib
            setattr(module, function_name, kernel)
            monkeypatch.setitem(sys.modules, module_name, module)

        fake_juliacall = types.ModuleType("juliacall")
        monkeypatch.setitem(sys.modules, "juliacall", fake_juliacall)
        install_backend(
            "scpn_phase_orchestrator.experimental.accelerators.upde._inertial_mojo",
            "inertial_step_mojo",
            0.10,
        )
        install_backend(
            "scpn_phase_orchestrator.experimental.accelerators.upde._inertial_julia",
            "inertial_step_julia",
            0.20,
        )
        install_backend(
            "scpn_phase_orchestrator.experimental.accelerators.upde._inertial_go",
            "inertial_step_go",
            0.30,
        )

        theta = np.array([0.0, 1.0], dtype=np.float64)
        omega = np.array([0.5, -0.5], dtype=np.float64)
        args = (
            theta,
            omega,
            np.ones(2),
            np.zeros(4),
            np.ones(2),
            np.ones(2),
            2,
            0.01,
        )

        for loader, offset in (
            (i_mod._load_mojo_fn, 0.10),
            (i_mod._load_julia_fn, 0.20),
            (i_mod._load_go_fn, 0.30),
        ):
            got_theta, got_omega = loader()(*args)
            np.testing.assert_allclose(got_theta, theta + offset + 0.01)
            np.testing.assert_allclose(got_omega, omega - offset + 0.01)


def _direct_payload(n: int = 4):
    theta = np.linspace(0.1, 1.0, n, dtype=np.float64)
    omega_dot = np.linspace(-0.2, 0.2, n, dtype=np.float64)
    power = np.linspace(-0.4, 0.4, n, dtype=np.float64)
    knm = np.full((n, n), 0.03, dtype=np.float64)
    np.fill_diagonal(knm, 0.0)
    inertia = np.full(n, 1.5, dtype=np.float64)
    damping = np.full(n, 0.25, dtype=np.float64)
    return theta, omega_dot, power, knm.ravel(), inertia, damping, n, 0.01


class TestDirectBackendBoundaryContracts:
    """Direct Go/Julia/Mojo bridges reject invalid inertial states early."""

    @pytest.mark.parametrize("backend", DIRECT_BACKENDS)
    @pytest.mark.parametrize(
        "index,replacement",
        [
            (0, np.array([0.0, np.nan], dtype=np.float64)),
            (1, np.array([0.0, 1.0 + 0.1j], dtype=np.complex128)),
            (2, np.array([True, False], dtype=np.bool_)),
            (3, np.ones(3, dtype=np.float64)),
            (4, np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float64)),
            (5, np.array([1.0, 1.0, -0.1, 1.0], dtype=np.float64)),
            (6, True),
            (7, 0.0),
        ],
    )
    def test_validation_precedes_runtime_load(self, backend, index, replacement):
        args = list(_direct_payload())
        args[index] = replacement
        with pytest.raises(ValueError):
            backend(*args)

    @pytest.mark.parametrize("backend", DIRECT_BACKENDS)
    def test_rejects_self_coupling_diagonal(self, backend):
        args = list(_direct_payload())
        knm = np.full((4, 4), 0.03, dtype=np.float64)
        np.fill_diagonal(knm, 0.2)
        args[3] = knm.ravel()

        with pytest.raises(ValueError, match="diagonal"):
            backend(*args)


class TestDispatchFallbackChain:
    def test_dispatch_falls_back_to_python_when_loader_fails(self, monkeypatch):
        previous_backend = i_mod.ACTIVE_BACKEND
        previous_available = list(i_mod.AVAILABLE_BACKENDS)
        previous_loader = i_mod._LOADERS["go"]
        i_mod.ACTIVE_BACKEND = "go"
        i_mod.AVAILABLE_BACKENDS = ["go", "python"]
        i_mod._BACKEND_CACHE.clear()
        monkeypatch.setitem(
            i_mod._LOADERS,
            "go",
            lambda: (_ for _ in ()).throw(ImportError("go backend unavailable")),
        )
        try:
            backend = i_mod._dispatch()
        finally:
            i_mod.ACTIVE_BACKEND = previous_backend
            i_mod.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(i_mod._LOADERS, "go", previous_loader)
            i_mod._BACKEND_CACHE.clear()

        assert backend is None

    def test_dispatch_uses_cached_loader_once(self, monkeypatch):
        previous_backend = i_mod.ACTIVE_BACKEND
        previous_available = list(i_mod.AVAILABLE_BACKENDS)
        previous_loader = i_mod._LOADERS["go"]
        i_mod.ACTIVE_BACKEND = "go"
        i_mod.AVAILABLE_BACKENDS = ["go", "python"]
        i_mod._BACKEND_CACHE.clear()
        call_count = 0

        def fake_backend(
            theta: np.ndarray,
            omega_dot: np.ndarray,
            _power: np.ndarray,
            _knm_flat: np.ndarray,
            _inertia: np.ndarray,
            _damping: np.ndarray,
            _n: int,
            _dt: float,
        ) -> tuple[np.ndarray, np.ndarray]:
            return theta.copy(), omega_dot.copy()

        def loader():
            nonlocal call_count
            call_count += 1
            return fake_backend

        monkeypatch.setitem(i_mod._LOADERS, "go", loader)
        try:
            b1 = i_mod._dispatch()
            b2 = i_mod._dispatch()
        finally:
            i_mod.ACTIVE_BACKEND = previous_backend
            i_mod.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(i_mod._LOADERS, "go", previous_loader)
            i_mod._BACKEND_CACHE.clear()

        assert b1 is fake_backend
        assert b2 is fake_backend
        assert call_count == 1
