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


def _run(backend: str, *, z_re: float = 0.2, z_im: float = 0.1,
         omega_0: float = 0.5, delta: float = 0.1, K: float = 1.0,
         dt: float = 0.01, n_steps: int = 500):
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
        max_examples=6, deadline=None,
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
        max_examples=6, deadline=None,
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
