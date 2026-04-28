# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Cross-backend parity for market kernels

"""Cross-backend parity for ``market_order_parameter`` and
``market_plv``.

The native backends all use the sincos expansion
``sin(θ_j − θ_i) = s_j·c_i − c_j·s_i`` inside the inner loop,
while the Python reference builds the complex order parameter
directly via ``np.mean(np.exp(1j·θ))``. The two forms are
mathematically identical but accumulate different floating-point
rounding, so parity is tight (~1e-15) but not always 0.0.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.upde import market as m_mod
from scpn_phase_orchestrator.upde.market import (
    market_order_parameter,
    market_plv,
)

TOL = 1e-12


@contextlib.contextmanager
def _force_backend(name: str):
    prev = m_mod.ACTIVE_BACKEND
    m_mod.ACTIVE_BACKEND = name
    try:
        yield
    finally:
        m_mod.ACTIVE_BACKEND = prev


def _problem(seed: int, T: int = 40, N: int = 5):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 2 * np.pi, (T, N))


def _op_backend(backend: str, seed: int, T: int = 40, N: int = 5):
    if backend not in m_mod.AVAILABLE_BACKENDS:
        pytest.skip(f"backend {backend!r} unavailable")
    phases = _problem(seed, T, N)
    with _force_backend(backend):
        return market_order_parameter(phases)


def _plv_backend(backend: str, seed: int, T: int = 40, N: int = 5, W: int = 10):
    if backend not in m_mod.AVAILABLE_BACKENDS:
        pytest.skip(f"backend {backend!r} unavailable")
    phases = _problem(seed, T, N)
    with _force_backend(backend):
        return market_plv(phases, window=W)


class TestOrderParameterParity:
    def test_rust(self):
        ref = _op_backend("python", 0)
        got = _op_backend("rust", 0)
        assert np.max(np.abs(got - ref)) < TOL

    def test_julia(self):
        ref = _op_backend("python", 1)
        got = _op_backend("julia", 1)
        assert np.max(np.abs(got - ref)) < TOL

    def test_go(self):
        ref = _op_backend("python", 2)
        got = _op_backend("go", 2)
        assert np.max(np.abs(got - ref)) < TOL

    def test_mojo(self):
        ref = _op_backend("python", 3, T=20)
        got = _op_backend("mojo", 3, T=20)
        assert np.max(np.abs(got - ref)) < 1e-10


class TestPLVParity:
    def test_rust(self):
        ref = _plv_backend("python", 4)
        got = _plv_backend("rust", 4)
        assert np.max(np.abs(got - ref)) < TOL

    def test_julia(self):
        ref = _plv_backend("python", 5)
        got = _plv_backend("julia", 5)
        assert np.max(np.abs(got - ref)) < TOL

    def test_go(self):
        ref = _plv_backend("python", 6)
        got = _plv_backend("go", 6)
        assert np.max(np.abs(got - ref)) < TOL

    def test_mojo(self):
        ref = _plv_backend("python", 7, T=20, N=3, W=5)
        got = _plv_backend("mojo", 7, T=20, N=3, W=5)
        assert np.max(np.abs(got - ref)) < 1e-10


class TestHypothesisParity:
    @given(
        T=st.integers(min_value=10, max_value=40),
        N=st.integers(min_value=2, max_value=6),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=6,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_op_rust_hypothesis(self, T, N, seed):
        if "rust" not in m_mod.AVAILABLE_BACKENDS:
            pytest.skip("rust unavailable")
        ref = _op_backend("python", seed, T=T, N=N)
        got = _op_backend("rust", seed, T=T, N=N)
        assert np.max(np.abs(got - ref)) < TOL

    @given(
        T=st.integers(min_value=10, max_value=30),
        N=st.integers(min_value=2, max_value=5),
        W=st.integers(min_value=3, max_value=8),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=5,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_plv_go_hypothesis(self, T, N, W, seed):
        if "go" not in m_mod.AVAILABLE_BACKENDS or W >= T:
            pytest.skip("go unavailable or window too wide")
        ref = _plv_backend("python", seed, T=T, N=N, W=W)
        got = _plv_backend("go", seed, T=T, N=N, W=W)
        assert np.max(np.abs(got - ref)) < TOL
