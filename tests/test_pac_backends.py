# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Per-backend parity tests for PAC

"""Per-backend parity for ``upde/pac.py``.

Exercises Rust, Julia, Go, Mojo against the NumPy reference for
both ``modulation_index`` and ``pac_matrix``. Tolerance budgets
follow the AttnRes reference.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.upde import pac as pac_mod
from scpn_phase_orchestrator.upde.pac import (
    AVAILABLE_BACKENDS,
    modulation_index,
    pac_matrix,
)

TWO_PI = 2.0 * np.pi


def _force(backend: str) -> str:
    prev = pac_mod.ACTIVE_BACKEND
    pac_mod.ACTIVE_BACKEND = backend
    return prev


def _reset(prev: str) -> None:
    pac_mod.ACTIVE_BACKEND = prev


def _reference_mi(
    theta: np.ndarray, amp: np.ndarray, n_bins: int
) -> float:
    prev = _force("python")
    try:
        return modulation_index(theta, amp, n_bins)
    finally:
        _reset(prev)


def _reference_matrix(
    phases: np.ndarray, amps: np.ndarray, n_bins: int
) -> np.ndarray:
    prev = _force("python")
    try:
        return pac_matrix(phases, amps, n_bins)
    finally:
        _reset(prev)


# ---------------------------------------------------------------------
# Rust parity
# ---------------------------------------------------------------------


class TestRustParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "rust" not in AVAILABLE_BACKENDS:
            pytest.skip("Rust backend not built")

    @given(
        n=st.integers(min_value=8, max_value=512),
        n_bins=st.sampled_from([6, 12, 18, 24]),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=15, deadline=None)
    def test_modulation_index_bit_exact(
        self, n: int, n_bins: int, seed: int
    ) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(0.0, TWO_PI, size=n)
        amp = 1.0 + 0.5 * np.cos(theta) + 0.1 * rng.standard_normal(n)
        ref = _reference_mi(theta, amp, n_bins)
        prev = _force("rust")
        try:
            result = modulation_index(theta, amp, n_bins)
        finally:
            _reset(prev)
        assert abs(result - ref) < 1e-12

    def test_pac_matrix_parity(self) -> None:
        rng = np.random.default_rng(0)
        t, n = 120, 5
        phases = rng.uniform(0.0, TWO_PI, size=(t, n))
        amps = 1.0 + 0.3 * np.cos(phases) + 0.1 * rng.standard_normal((t, n))
        ref = _reference_matrix(phases, amps, 12)
        prev = _force("rust")
        try:
            result = pac_matrix(phases, amps, 12)
        finally:
            _reset(prev)
        np.testing.assert_allclose(result, ref, atol=1e-12)


# ---------------------------------------------------------------------
# Julia parity
# ---------------------------------------------------------------------


class TestJuliaParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "julia" not in AVAILABLE_BACKENDS:
            pytest.skip("Julia backend not available")

    @pytest.mark.parametrize("n", [64, 256])
    def test_modulation_index(self, n: int) -> None:
        rng = np.random.default_rng(7 + n)
        theta = rng.uniform(0.0, TWO_PI, size=n)
        amp = 1.0 + 0.4 * np.cos(theta)
        ref = _reference_mi(theta, amp, 18)
        prev = _force("julia")
        try:
            result = modulation_index(theta, amp, 18)
        finally:
            _reset(prev)
        assert abs(result - ref) < 1e-12

    def test_pac_matrix(self) -> None:
        rng = np.random.default_rng(3)
        t, n = 80, 4
        phases = rng.uniform(0.0, TWO_PI, size=(t, n))
        amps = 1.0 + 0.5 * np.cos(phases)
        ref = _reference_matrix(phases, amps, 12)
        prev = _force("julia")
        try:
            result = pac_matrix(phases, amps, 12)
        finally:
            _reset(prev)
        np.testing.assert_allclose(result, ref, atol=1e-12)


# ---------------------------------------------------------------------
# Go parity
# ---------------------------------------------------------------------


class TestGoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "go" not in AVAILABLE_BACKENDS:
            pytest.skip("Go backend not built")

    @given(
        n=st.integers(min_value=8, max_value=256),
        n_bins=st.sampled_from([6, 12, 18]),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=12,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_modulation_index(
        self, n: int, n_bins: int, seed: int
    ) -> None:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(0.0, TWO_PI, size=n)
        amp = 1.0 + 0.5 * np.cos(theta)
        ref = _reference_mi(theta, amp, n_bins)
        prev = _force("go")
        try:
            result = modulation_index(theta, amp, n_bins)
        finally:
            _reset(prev)
        assert abs(result - ref) < 1e-12


# ---------------------------------------------------------------------
# Mojo parity
# ---------------------------------------------------------------------


class TestMojoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "mojo" not in AVAILABLE_BACKENDS:
            pytest.skip("Mojo backend not built")

    @pytest.mark.parametrize("n", [32, 128])
    def test_modulation_index(self, n: int) -> None:
        rng = np.random.default_rng(17 + n)
        theta = rng.uniform(0.0, TWO_PI, size=n)
        amp = 1.0 + 0.4 * np.cos(theta)
        ref = _reference_mi(theta, amp, 18)
        prev = _force("mojo")
        try:
            result = modulation_index(theta, amp, 18)
        finally:
            _reset(prev)
        # text protocol rounding budget (log-summing amplifies the
        # 17-digit round-trip to ~1e-10 at N ≥ 64).
        assert abs(result - ref) < 1e-10

    def test_pac_matrix_small(self) -> None:
        rng = np.random.default_rng(5)
        t, n = 50, 3
        phases = rng.uniform(0.0, TWO_PI, size=(t, n))
        amps = 1.0 + 0.5 * np.cos(phases)
        ref = _reference_matrix(phases, amps, 12)
        prev = _force("mojo")
        try:
            result = pac_matrix(phases, amps, 12)
        finally:
            _reset(prev)
        np.testing.assert_allclose(result, ref, atol=1e-10)


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
        n = 150
        theta = rng.uniform(0.0, TWO_PI, size=n)
        amp = 1.0 + 0.4 * np.cos(theta) + 0.1 * rng.standard_normal(n)
        ref = _reference_mi(theta, amp, 18)

        tolerances = {
            "rust": 1e-12,
            "julia": 1e-12,
            "go": 1e-12,
            "mojo": 1e-10,
            "python": 0.0,
        }

        for backend in AVAILABLE_BACKENDS:
            atol = tolerances[backend]
            prev = _force(backend)
            try:
                result = modulation_index(theta, amp, 18)
            finally:
                _reset(prev)
            assert abs(result - ref) <= atol
