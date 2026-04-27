# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Cross-backend parity for basin stability

"""Cross-backend parity of the ``steady_state_r`` trial kernel.

All five backends (Rust / Mojo / Julia / Go / Python) integrate the
Kuramoto ODE via explicit Euler with full-snapshot step semantics
and must produce bit-exact R values for identical inputs. This file
pins the dispatcher to each backend in turn, runs the same problem,
and cross-checks against the Python reference with a tight tolerance.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.upde import basin_stability as b_mod
from scpn_phase_orchestrator.upde.basin_stability import (
    basin_stability,
    steady_state_r,
)

TOL = 1e-12


@contextlib.contextmanager
def _force_backend(name: str):
    prev = b_mod.ACTIVE_BACKEND
    b_mod.ACTIVE_BACKEND = name
    try:
        yield
    finally:
        b_mod.ACTIVE_BACKEND = prev


def _all_to_all(n: int, strength: float = 1.0) -> np.ndarray:
    k = np.ones((n, n)) * strength / n
    np.fill_diagonal(k, 0.0)
    return k


def _reference_R(n: int, strength: float, seed: int) -> float:
    omegas = np.ones(n)
    knm = _all_to_all(n, strength=strength)
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0, 2 * np.pi, n)
    with _force_backend("python"):
        return steady_state_r(
            phases,
            omegas,
            knm,
            dt=0.01,
            n_transient=200,
            n_measure=100,
        )


def _backend_R(backend: str, n: int, strength: float, seed: int) -> float:
    if backend not in b_mod.AVAILABLE_BACKENDS:
        pytest.skip(f"backend {backend!r} unavailable")
    omegas = np.ones(n)
    knm = _all_to_all(n, strength=strength)
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0, 2 * np.pi, n)
    with _force_backend(backend):
        return steady_state_r(
            phases,
            omegas,
            knm,
            dt=0.01,
            n_transient=200,
            n_measure=100,
        )


class TestSteadyStateRParity:
    def test_rust_matches_python(self):
        ref = _reference_R(6, strength=3.0, seed=0)
        got = _backend_R("rust", 6, strength=3.0, seed=0)
        assert abs(got - ref) < TOL

    def test_julia_matches_python(self):
        ref = _reference_R(6, strength=3.0, seed=1)
        got = _backend_R("julia", 6, strength=3.0, seed=1)
        assert abs(got - ref) < TOL

    def test_go_matches_python(self):
        ref = _reference_R(6, strength=3.0, seed=2)
        got = _backend_R("go", 6, strength=3.0, seed=2)
        assert abs(got - ref) < TOL

    def test_mojo_matches_python(self):
        ref = _reference_R(5, strength=2.5, seed=3)
        got = _backend_R("mojo", 5, strength=2.5, seed=3)
        # Mojo text round-trip introduces ≤ 1e-14 drift over ~300 steps.
        assert abs(got - ref) < 1e-10


class TestBasinStabilityParity:
    """S_B must agree across backends for identical RNG seed."""

    def _compare(self, backend: str):
        if backend not in b_mod.AVAILABLE_BACKENDS:
            pytest.skip(f"backend {backend!r} unavailable")
        n = 5
        omegas = np.ones(n)
        knm = _all_to_all(n, strength=2.5)
        with _force_backend("python"):
            ref = basin_stability(
                omegas,
                knm,
                dt=0.01,
                n_transient=100,
                n_measure=50,
                n_samples=6,
                R_threshold=0.5,
                seed=42,
            )
        with _force_backend(backend):
            got = basin_stability(
                omegas,
                knm,
                dt=0.01,
                n_transient=100,
                n_measure=50,
                n_samples=6,
                R_threshold=0.5,
                seed=42,
            )
        np.testing.assert_allclose(got.R_final, ref.R_final, atol=1e-10)
        assert got.S_B == ref.S_B
        assert got.n_converged == ref.n_converged

    def test_rust(self):
        self._compare("rust")

    def test_julia(self):
        self._compare("julia")

    def test_go(self):
        self._compare("go")

    def test_mojo(self):
        self._compare("mojo")


class TestHypothesisParity:
    @given(
        n=st.integers(min_value=2, max_value=6),
        strength=st.floats(min_value=0.5, max_value=4.0),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=6,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_rust_hypothesis(self, n, strength, seed):
        if "rust" not in b_mod.AVAILABLE_BACKENDS:
            pytest.skip("rust unavailable")
        ref = _reference_R(n, strength, seed)
        got = _backend_R("rust", n, strength, seed)
        assert abs(got - ref) < TOL

    @given(
        n=st.integers(min_value=2, max_value=6),
        strength=st.floats(min_value=0.5, max_value=4.0),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=6,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_go_hypothesis(self, n, strength, seed):
        if "go" not in b_mod.AVAILABLE_BACKENDS:
            pytest.skip("go unavailable")
        ref = _reference_R(n, strength, seed)
        got = _backend_R("go", n, strength, seed)
        assert abs(got - ref) < TOL
