# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Per-backend parity tests for order parameters

"""Per-backend parity tests for ``upde/order_params.py``.

Exercises every non-Python backend individually against the NumPy
reference for all three compute kernels.

Tolerance budgets match the AttnRes reference:

* Rust / Julia / Go — bit-exact (≤ 1e-12)
* Mojo — ≤ 1e-13 (text round-trip)
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.upde import order_params as op_mod
from scpn_phase_orchestrator.upde.order_params import (
    AVAILABLE_BACKENDS,
    compute_layer_coherence,
    compute_order_parameter,
    compute_plv,
)

TWO_PI = 2.0 * np.pi


def _force(backend: str) -> str:
    """Monkey-patch ``ACTIVE_BACKEND`` and return the previous value."""
    prev = op_mod.ACTIVE_BACKEND
    op_mod.ACTIVE_BACKEND = backend
    return prev


def _reset(prev: str) -> None:
    op_mod.ACTIVE_BACKEND = prev


def _reference(
    phases_a: np.ndarray,
    phases_b: np.ndarray,
    indices: np.ndarray,
) -> tuple[tuple[float, float], float, float]:
    prev = _force("python")
    try:
        r_psi = compute_order_parameter(phases_a)
        plv_val = compute_plv(phases_a, phases_b)
        lc = compute_layer_coherence(phases_a, indices)
    finally:
        _reset(prev)
    return r_psi, plv_val, lc


# ---------------------------------------------------------------------
# Rust parity
# ---------------------------------------------------------------------


class TestRustParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "rust" not in AVAILABLE_BACKENDS:
            pytest.skip("Rust backend not built on this host")

    @given(
        n=st.integers(min_value=2, max_value=256),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=20, deadline=None)
    def test_bit_exact_order_parameter(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0.0, TWO_PI, size=n)
        ref_op = _reference(phases, phases, np.arange(n, dtype=np.int64))[0]
        prev = _force("rust")
        try:
            r, psi = compute_order_parameter(phases)
        finally:
            _reset(prev)
        assert abs(r - ref_op[0]) < 1e-12
        assert abs(psi - ref_op[1]) < 1e-12

    @given(
        n=st.integers(min_value=2, max_value=256),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=20, deadline=None)
    def test_bit_exact_plv(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        a = rng.uniform(0.0, TWO_PI, size=n)
        b = rng.uniform(0.0, TWO_PI, size=n)
        ref = _reference(a, b, np.arange(n, dtype=np.int64))[1]
        prev = _force("rust")
        try:
            result = compute_plv(a, b)
        finally:
            _reset(prev)
        assert abs(result - ref) < 1e-12

    def test_layer_coherence_parity(self) -> None:
        rng = np.random.default_rng(0)
        phases = rng.uniform(0.0, TWO_PI, size=20)
        indices = np.array([0, 3, 5, 7, 11, 13], dtype=np.int64)
        ref = _reference(phases, phases, indices)[2]
        prev = _force("rust")
        try:
            result = compute_layer_coherence(phases, indices)
        finally:
            _reset(prev)
        assert abs(result - ref) < 1e-12


# ---------------------------------------------------------------------
# Julia parity
# ---------------------------------------------------------------------


class TestJuliaParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "julia" not in AVAILABLE_BACKENDS:
            pytest.skip("Julia backend not available")

    @pytest.mark.parametrize("n", [8, 32, 128])
    def test_order_parameter(self, n: int) -> None:
        rng = np.random.default_rng(7 + n)
        phases = rng.uniform(0.0, TWO_PI, size=n)
        ref = _reference(phases, phases, np.arange(n, dtype=np.int64))[0]
        prev = _force("julia")
        try:
            r, psi = compute_order_parameter(phases)
        finally:
            _reset(prev)
        assert abs(r - ref[0]) < 1e-12
        assert abs(psi - ref[1]) < 1e-12

    def test_plv_and_layer_coherence(self) -> None:
        rng = np.random.default_rng(13)
        a = rng.uniform(0.0, TWO_PI, size=60)
        b = rng.uniform(0.0, TWO_PI, size=60)
        indices = np.array([2, 5, 10, 20, 30], dtype=np.int64)
        ref_plv = _reference(a, b, indices)[1]
        ref_lc = _reference(a, b, indices)[2]
        prev = _force("julia")
        try:
            plv_val = compute_plv(a, b)
            lc = compute_layer_coherence(a, indices)
        finally:
            _reset(prev)
        assert abs(plv_val - ref_plv) < 1e-12
        assert abs(lc - ref_lc) < 1e-12


# ---------------------------------------------------------------------
# Go parity
# ---------------------------------------------------------------------


class TestGoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "go" not in AVAILABLE_BACKENDS:
            pytest.skip("Go backend not built")

    @given(
        n=st.integers(min_value=2, max_value=128),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=15,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_order_parameter(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0.0, TWO_PI, size=n)
        ref = _reference(phases, phases, np.arange(n, dtype=np.int64))[0]
        prev = _force("go")
        try:
            r, psi = compute_order_parameter(phases)
        finally:
            _reset(prev)
        assert abs(r - ref[0]) < 1e-12
        assert abs(psi - ref[1]) < 1e-12


# ---------------------------------------------------------------------
# Mojo parity
# ---------------------------------------------------------------------


class TestMojoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "mojo" not in AVAILABLE_BACKENDS:
            pytest.skip("Mojo backend not built")

    @pytest.mark.parametrize("n", [8, 24, 64])
    def test_order_parameter(self, n: int) -> None:
        rng = np.random.default_rng(17 + n)
        phases = rng.uniform(0.0, TWO_PI, size=n)
        ref = _reference(phases, phases, np.arange(n, dtype=np.int64))[0]
        prev = _force("mojo")
        try:
            r, psi = compute_order_parameter(phases)
        finally:
            _reset(prev)
        assert abs(r - ref[0]) < 1e-13
        assert abs(psi - ref[1]) < 1e-13

    def test_plv_and_layer_coherence(self) -> None:
        rng = np.random.default_rng(23)
        a = rng.uniform(0.0, TWO_PI, size=40)
        b = rng.uniform(0.0, TWO_PI, size=40)
        indices = np.array([3, 6, 9, 12], dtype=np.int64)
        ref_plv = _reference(a, b, indices)[1]
        ref_lc = _reference(a, b, indices)[2]
        prev = _force("mojo")
        try:
            plv_val = compute_plv(a, b)
            lc = compute_layer_coherence(a, indices)
        finally:
            _reset(prev)
        assert abs(plv_val - ref_plv) < 1e-13
        assert abs(lc - ref_lc) < 1e-13


# ---------------------------------------------------------------------
# Cross-backend consistency
# ---------------------------------------------------------------------


class TestCrossBackendConsistency:
    @pytest.mark.skipif(
        len(AVAILABLE_BACKENDS) < 2,
        reason="Only the Python fallback is available",
    )
    def test_all_backends_agree(self) -> None:
        rng = np.random.default_rng(2026)
        n = 32
        phases_a = rng.uniform(0.0, TWO_PI, size=n)
        phases_b = rng.uniform(0.0, TWO_PI, size=n)
        indices = np.arange(0, n, 3, dtype=np.int64)
        ref_op, ref_plv, ref_lc = _reference(phases_a, phases_b, indices)

        tolerances = {
            "rust": 1e-12,
            "julia": 1e-12,
            "go": 1e-12,
            "mojo": 1e-13,
            "python": 0.0,
        }

        for backend in AVAILABLE_BACKENDS:
            atol = tolerances[backend]
            prev = _force(backend)
            try:
                r, psi = compute_order_parameter(phases_a)
                plv_val = compute_plv(phases_a, phases_b)
                lc = compute_layer_coherence(phases_a, indices)
            finally:
                _reset(prev)
            assert abs(r - ref_op[0]) <= atol, (
                f"{backend} R diff {abs(r - ref_op[0]):.2e} exceeds {atol}"
            )
            assert abs(psi - ref_op[1]) <= atol
            assert abs(plv_val - ref_plv) <= atol
            assert abs(lc - ref_lc) <= atol
