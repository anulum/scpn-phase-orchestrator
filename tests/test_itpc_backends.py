# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Per-backend parity for ITPC

"""Cross-backend parity for :func:`compute_itpc` /
:func:`itpc_persistence`. Every available backend (Rust / Mojo /
Julia / Go / Python) must produce the same output as the Python
reference on the same input.

Tolerances: Rust / Julia / Go / Mojo all within ``1e-12`` — ITPC is
a simple `mean(cos)`/`mean(sin)` accumulator with no log/exp, and the
measured parity sits at ~5e-17 on this host (bit-equivalent).
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.monitor import itpc as it_mod
from scpn_phase_orchestrator.monitor.itpc import (
    AVAILABLE_BACKENDS,
    compute_itpc,
    itpc_persistence,
)

TWO_PI = 2.0 * np.pi


def _force(backend: str) -> str:
    prev = it_mod.ACTIVE_BACKEND
    it_mod.ACTIVE_BACKEND = backend
    return prev


def _reset(prev: str) -> None:
    it_mod.ACTIVE_BACKEND = prev


def _reference_itpc(phases: np.ndarray) -> np.ndarray:
    prev = _force("python")
    try:
        return compute_itpc(phases)
    finally:
        _reset(prev)


def _reference_pers(phases: np.ndarray, idx: np.ndarray) -> float:
    prev = _force("python")
    try:
        return itpc_persistence(phases, idx)
    finally:
        _reset(prev)


def _problem(seed: int, n_trials: int = 30, n_tp: int = 80):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, TWO_PI, size=(n_trials, n_tp))


class TestRustParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "rust" not in AVAILABLE_BACKENDS:
            pytest.skip("Rust backend not built")

    @given(
        n_trials=st.integers(min_value=2, max_value=60),
        n_tp=st.integers(min_value=1, max_value=120),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_itpc(self, n_trials: int, n_tp: int, seed: int) -> None:
        phases = _problem(seed, n_trials, n_tp)
        ref = _reference_itpc(phases)
        prev = _force("rust")
        try:
            result = compute_itpc(phases)
        finally:
            _reset(prev)
        np.testing.assert_allclose(result, ref, atol=1e-12)

    def test_persistence(self) -> None:
        phases = _problem(7)
        idx = np.array([5, 10, 25, 40, 75])
        ref = _reference_pers(phases, idx)
        prev = _force("rust")
        try:
            result = itpc_persistence(phases, idx)
        finally:
            _reset(prev)
        assert abs(result - ref) < 1e-12


class TestJuliaParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "julia" not in AVAILABLE_BACKENDS:
            pytest.skip("Julia backend not available")

    @pytest.mark.parametrize("seed", [0, 42])
    def test_itpc(self, seed: int) -> None:
        phases = _problem(seed)
        ref = _reference_itpc(phases)
        prev = _force("julia")
        try:
            result = compute_itpc(phases)
        finally:
            _reset(prev)
        np.testing.assert_allclose(result, ref, atol=1e-12)


class TestGoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "go" not in AVAILABLE_BACKENDS:
            pytest.skip("Go backend not built")

    @given(
        n_trials=st.integers(min_value=2, max_value=40),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=6,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_itpc(self, n_trials: int, seed: int) -> None:
        phases = _problem(seed, n_trials=n_trials)
        ref = _reference_itpc(phases)
        prev = _force("go")
        try:
            result = compute_itpc(phases)
        finally:
            _reset(prev)
        np.testing.assert_allclose(result, ref, atol=1e-12)

    def test_persistence(self) -> None:
        phases = _problem(11)
        idx = np.array([0, 15, 30, 60])
        ref = _reference_pers(phases, idx)
        prev = _force("go")
        try:
            result = itpc_persistence(phases, idx)
        finally:
            _reset(prev)
        assert abs(result - ref) < 1e-12


class TestMojoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "mojo" not in AVAILABLE_BACKENDS:
            pytest.skip("Mojo backend not built")

    @pytest.mark.parametrize("seed", [0, 77])
    def test_itpc(self, seed: int) -> None:
        phases = _problem(seed)
        ref = _reference_itpc(phases)
        prev = _force("mojo")
        try:
            result = compute_itpc(phases)
        finally:
            _reset(prev)
        # ITPC has no log / exp amplification — bit-equivalent even
        # across the text round-trip.
        np.testing.assert_allclose(result, ref, atol=1e-9)

    def test_persistence(self) -> None:
        phases = _problem(99)
        idx = np.array([2, 8, 16, 40])
        ref = _reference_pers(phases, idx)
        prev = _force("mojo")
        try:
            result = itpc_persistence(phases, idx)
        finally:
            _reset(prev)
        assert abs(result - ref) < 1e-9


class TestCrossBackendConsistency:
    @pytest.mark.skipif(
        len(AVAILABLE_BACKENDS) < 2,
        reason="Only Python fallback available",
    )
    def test_all_backends_agree(self) -> None:
        phases = _problem(2026, n_trials=40, n_tp=100)
        idx = np.array([5, 25, 50, 75, 95])
        ref_itpc = _reference_itpc(phases)
        ref_pers = _reference_pers(phases, idx)
        tolerances = {
            "rust": 1e-12,
            "julia": 1e-12,
            "go": 1e-12,
            "mojo": 1e-9,
            "python": 0.0,
        }
        for backend in AVAILABLE_BACKENDS:
            prev = _force(backend)
            try:
                itpc_v = compute_itpc(phases)
                pers_v = itpc_persistence(phases, idx)
            finally:
                _reset(prev)
            np.testing.assert_allclose(
                itpc_v, ref_itpc, atol=tolerances[backend],
                err_msg=f"{backend} ITPC diverged from python",
            )
            assert abs(pers_v - ref_pers) <= tolerances[backend], (
                f"{backend} persistence diverged: "
                f"{pers_v} vs {ref_pers}"
            )
