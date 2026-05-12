# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Cross-backend parity for Ott-Antonsen reduction

"""Cross-backend parity for ``OttAntonsenReduction.run``.

The OA kernel is a scalar complex-ODE RK4 loop: five backends
should agree bit-for-bit modulo the subprocess text-round-trip
epsilon on Mojo.
"""

from __future__ import annotations

import contextlib
import importlib
import sys
import types

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.upde import reduction as r_mod
from scpn_phase_orchestrator.upde.reduction import OttAntonsenReduction

TOL = 1e-12


@contextlib.contextmanager
def _force_backend(name: str):
    prev = r_mod.ACTIVE_BACKEND
    r_mod.ACTIVE_BACKEND = name
    try:
        yield
    finally:
        r_mod.ACTIVE_BACKEND = prev


def _run(
    backend: str,
    *,
    z_re: float = 0.2,
    z_im: float = 0.1,
    omega_0: float = 0.5,
    delta: float = 0.1,
    K: float = 1.0,
    dt: float = 0.01,
    n_steps: int = 500,
):
    if backend not in r_mod.AVAILABLE_BACKENDS:
        pytest.skip(f"backend {backend!r} unavailable")
    red = OttAntonsenReduction(omega_0=omega_0, delta=delta, K=K, dt=dt)
    with _force_backend(backend):
        return red.run(complex(z_re, z_im), n_steps=n_steps)


class TestBackendParity:
    def test_rust_matches_python(self):
        ref = _run("python")
        got = _run("rust")
        assert abs(got.z.real - ref.z.real) < TOL
        assert abs(got.z.imag - ref.z.imag) < TOL
        assert abs(got.R - ref.R) < TOL
        assert abs(got.psi - ref.psi) < TOL

    def test_julia_matches_python(self):
        ref = _run("python")
        got = _run("julia")
        assert abs(got.z.real - ref.z.real) < TOL
        assert abs(got.z.imag - ref.z.imag) < TOL

    def test_go_matches_python(self):
        ref = _run("python")
        got = _run("go")
        assert abs(got.z.real - ref.z.real) < TOL
        assert abs(got.z.imag - ref.z.imag) < TOL

    def test_mojo_matches_python(self):
        ref = _run("python")
        got = _run("mojo")
        # Text round-trip tolerance on scalar RK4 over 500 steps.
        assert abs(got.z.real - ref.z.real) < 1e-10
        assert abs(got.z.imag - ref.z.imag) < 1e-10


class TestSubcriticalParity:
    def test_rust_subcritical(self):
        ref = _run("python", delta=1.0, K=1.5, n_steps=1000)
        got = _run("rust", delta=1.0, K=1.5, n_steps=1000)
        assert abs(got.R - ref.R) < TOL

    def test_go_subcritical(self):
        ref = _run("python", delta=1.0, K=1.5, n_steps=1000)
        got = _run("go", delta=1.0, K=1.5, n_steps=1000)
        assert abs(got.R - ref.R) < TOL


class TestHypothesisParity:
    @given(
        delta=st.floats(min_value=0.05, max_value=0.5),
        K_ratio=st.floats(min_value=0.5, max_value=4.0),
        n_steps=st.integers(min_value=10, max_value=500),
    )
    @settings(
        max_examples=6,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_rust_hypothesis(self, delta, K_ratio, n_steps):
        if "rust" not in r_mod.AVAILABLE_BACKENDS:
            pytest.skip("rust unavailable")
        K = K_ratio * delta
        ref = _run("python", delta=delta, K=K, n_steps=n_steps)
        got = _run("rust", delta=delta, K=K, n_steps=n_steps)
        assert abs(got.z.real - ref.z.real) < TOL
        assert abs(got.z.imag - ref.z.imag) < TOL

    @given(
        delta=st.floats(min_value=0.05, max_value=0.5),
        K_ratio=st.floats(min_value=0.5, max_value=4.0),
        n_steps=st.integers(min_value=10, max_value=500),
    )
    @settings(
        max_examples=6,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_go_hypothesis(self, delta, K_ratio, n_steps):
        if "go" not in r_mod.AVAILABLE_BACKENDS:
            pytest.skip("go unavailable")
        K = K_ratio * delta
        ref = _run("python", delta=delta, K=K, n_steps=n_steps)
        got = _run("go", delta=delta, K=K, n_steps=n_steps)
        assert abs(got.z.real - ref.z.real) < TOL
        assert abs(got.z.imag - ref.z.imag) < TOL


class TestOptionalLoaderSuccessPaths:
    def test_rust_loader_wraps_spo_kernel_function(self, monkeypatch):
        calls = []

        def oa_run_rust(*args):
            calls.append(args)
            return 0.1, 0.2, 0.3, 0.4

        fake_spo = types.ModuleType("spo_kernel")
        fake_spo.oa_run_rust = oa_run_rust
        monkeypatch.setitem(sys.modules, "spo_kernel", fake_spo)

        run = r_mod._load_rust_fn()
        assert run(0.01, 0.02, 0.5, 0.1, 1.0, 0.01, 3) == (
            0.1,
            0.2,
            0.3,
            0.4,
        )
        assert calls == [(0.01, 0.02, 0.5, 0.1, 1.0, 0.01, 3)]

    def test_mojo_loader_runs_availability_probe(self, monkeypatch):
        fake_mojo = types.ModuleType("scpn_phase_orchestrator.upde._reduction_mojo")
        probe_calls = []

        def _ensure_exe():
            probe_calls.append(True)

        def oa_run_mojo(*args):
            return 0.2, 0.3, 0.4, 0.5

        fake_mojo._ensure_exe = _ensure_exe
        fake_mojo.oa_run_mojo = oa_run_mojo
        monkeypatch.setitem(
            sys.modules,
            "scpn_phase_orchestrator.upde._reduction_mojo",
            fake_mojo,
        )

        run = r_mod._load_mojo_fn()
        assert run() == (0.2, 0.3, 0.4, 0.5)
        assert probe_calls == [True]

    def test_julia_loader_requires_juliacall_and_returns_runner(self, monkeypatch):
        fake_juliacall = types.ModuleType("juliacall")
        fake_julia = types.ModuleType("scpn_phase_orchestrator.upde._reduction_julia")

        def oa_run_julia(*args):
            return 0.3, 0.4, 0.5, 0.6

        fake_julia.oa_run_julia = oa_run_julia
        monkeypatch.setitem(sys.modules, "juliacall", fake_juliacall)
        monkeypatch.setitem(
            sys.modules,
            "scpn_phase_orchestrator.upde._reduction_julia",
            fake_julia,
        )

        run = r_mod._load_julia_fn()
        assert run() == (0.3, 0.4, 0.5, 0.6)

    def test_go_loader_runs_shared_library_probe(self, monkeypatch):
        fake_go = types.ModuleType("scpn_phase_orchestrator.upde._reduction_go")
        probe_calls = []

        def _load_lib():
            probe_calls.append(True)

        def oa_run_go(*args):
            return 0.4, 0.5, 0.6, 0.7

        fake_go._load_lib = _load_lib
        fake_go.oa_run_go = oa_run_go
        monkeypatch.setitem(
            sys.modules,
            "scpn_phase_orchestrator.upde._reduction_go",
            fake_go,
        )

        run = r_mod._load_go_fn()
        assert run() == (0.4, 0.5, 0.6, 0.7)
        assert probe_calls == [True]

    def test_scalar_rust_helpers_feed_steady_state_and_fit(self, monkeypatch):
        def steady_state(delta, coupling):
            assert (delta, coupling) == (0.2, 1.0)
            return 0.75

        def fit_lorentzian(omegas):
            assert omegas.flags.c_contiguous
            return 1.25, 0.35

        def fake_run(z_re, z_im, omega_0, delta, k_coupling, dt, n_steps):
            assert (omega_0, delta, k_coupling, dt, n_steps) == (
                1.25,
                0.35,
                1.4,
                0.01,
                1000,
            )
            return z_re, z_im, 0.5, 0.0

        monkeypatch.setattr(r_mod, "_HAS_RUST_SCALAR", True)
        monkeypatch.setattr(r_mod, "_rust_steady_state_r", steady_state, raising=False)
        monkeypatch.setattr(
            r_mod,
            "_rust_fit_lorentzian",
            fit_lorentzian,
            raising=False,
        )
        monkeypatch.setattr(r_mod, "_python_oa_run", fake_run)
        monkeypatch.setattr(r_mod, "ACTIVE_BACKEND", "python")

        red = OttAntonsenReduction(omega_0=0.0, delta=0.2, K=1.0, dt=0.01)
        assert red.steady_state_R() == 0.75

        state = red.predict_from_oscillators(
            np.array([1.0, 1.5, 2.0]),
            K=1.4,
        )
        assert state.R == 0.5
        assert state.K_c == 0.7

    def test_module_import_detects_scalar_rust_helpers(self, monkeypatch):
        """Reload-time scalar helper detection should wire the optional Rust
        steady-state and Lorentzian-fit helpers when spo_kernel provides them."""

        def fit_lorentzian_rust(_omegas):
            return 0.0, 0.1

        def steady_state_r_oa_rust(_delta, _coupling):
            return 0.2

        fake_spo = types.ModuleType("spo_kernel")
        fake_spo.fit_lorentzian_rust = fit_lorentzian_rust
        fake_spo.steady_state_r_oa_rust = steady_state_r_oa_rust
        monkeypatch.setitem(sys.modules, "spo_kernel", fake_spo)

        reloaded = importlib.reload(r_mod)
        assert reloaded._HAS_RUST_SCALAR is True
        assert reloaded._rust_fit_lorentzian is fit_lorentzian_rust
        assert reloaded._rust_steady_state_r is steady_state_r_oa_rust

        monkeypatch.setitem(sys.modules, "spo_kernel", None)
        importlib.reload(r_mod)
