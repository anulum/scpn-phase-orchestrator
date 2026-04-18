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

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.upde import inertial as i_mod
from scpn_phase_orchestrator.upde.inertial import InertialKuramotoEngine

TOL = 1e-12


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
        (lambda k: (np.fill_diagonal(k, 0), k)[1])(
            rng.uniform(0, 0.5, (n, n))
        ),
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
                theta, od, p, k, m, d, n_steps=steps,
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
        max_examples=6, deadline=None,
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
        max_examples=6, deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_go_hypothesis(self, n, seed):
        if "go" not in i_mod.AVAILABLE_BACKENDS:
            pytest.skip("go unavailable")
        ref_th, ref_od = _run_backend("python", n, seed)
        got_th, got_od = _run_backend("go", n, seed)
        assert np.max(np.abs(got_th - ref_th)) < TOL
        assert np.max(np.abs(got_od - ref_od)) < TOL
