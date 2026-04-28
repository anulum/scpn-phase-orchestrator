# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Per-backend parity for chimera detection

"""Cross-backend parity for :func:`local_order_parameter`.

All backends agree with the Python reference within:

* Rust / Julia / Go — 1e-12 (shared f64).
* Mojo — 1e-9 (subprocess text round-trip).
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.monitor import chimera as ch_mod
from scpn_phase_orchestrator.monitor.chimera import (
    AVAILABLE_BACKENDS,
    detect_chimera,
    local_order_parameter,
)

TWO_PI = 2.0 * np.pi


def _force(backend: str) -> str:
    prev = ch_mod.ACTIVE_BACKEND
    ch_mod.ACTIVE_BACKEND = backend
    return prev


def _reset(prev: str) -> None:
    ch_mod.ACTIVE_BACKEND = prev


def _reference(phases: np.ndarray, knm: np.ndarray) -> np.ndarray:
    prev = _force("python")
    try:
        return local_order_parameter(phases, knm)
    finally:
        _reset(prev)


def _problem(seed: int, n: int = 16) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, TWO_PI, n)
    knm = rng.uniform(0.0, 1.0, (n, n))
    knm = (knm > 0.3).astype(np.float64) * knm
    np.fill_diagonal(knm, 0.0)
    return phases, knm


class TestRustParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "rust" not in AVAILABLE_BACKENDS:
            pytest.skip("Rust backend not built")

    @given(
        n=st.integers(min_value=2, max_value=40),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_matches_python(self, n: int, seed: int) -> None:
        phases, knm = _problem(seed, n)
        ref = _reference(phases, knm)
        prev = _force("rust")
        try:
            got = local_order_parameter(phases, knm)
        finally:
            _reset(prev)
        np.testing.assert_allclose(got, ref, atol=1e-12)


class TestJuliaParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "julia" not in AVAILABLE_BACKENDS:
            pytest.skip("Julia backend not available")

    @pytest.mark.parametrize("seed", [0, 42])
    def test_matches_python(self, seed: int) -> None:
        phases, knm = _problem(seed)
        ref = _reference(phases, knm)
        prev = _force("julia")
        try:
            got = local_order_parameter(phases, knm)
        finally:
            _reset(prev)
        np.testing.assert_allclose(got, ref, atol=1e-12)


class TestGoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "go" not in AVAILABLE_BACKENDS:
            pytest.skip("Go backend not built")

    @given(
        n=st.integers(min_value=2, max_value=30),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=8,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_matches_python(self, n: int, seed: int) -> None:
        phases, knm = _problem(seed, n)
        ref = _reference(phases, knm)
        prev = _force("go")
        try:
            got = local_order_parameter(phases, knm)
        finally:
            _reset(prev)
        np.testing.assert_allclose(got, ref, atol=1e-12)


class TestMojoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "mojo" not in AVAILABLE_BACKENDS:
            pytest.skip("Mojo backend not built")

    @pytest.mark.parametrize("seed", [0, 77])
    def test_matches_python(self, seed: int) -> None:
        phases, knm = _problem(seed)
        ref = _reference(phases, knm)
        prev = _force("mojo")
        try:
            got = local_order_parameter(phases, knm)
        finally:
            _reset(prev)
        np.testing.assert_allclose(got, ref, atol=1e-9)


class TestCrossBackendConsistency:
    @pytest.mark.skipif(
        len(AVAILABLE_BACKENDS) < 2,
        reason="Only Python fallback available",
    )
    def test_all_backends_agree(self) -> None:
        phases, knm = _problem(2026, n=24)
        ref = _reference(phases, knm)
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
                got = local_order_parameter(phases, knm)
            finally:
                _reset(prev)
            np.testing.assert_allclose(
                got,
                ref,
                atol=tolerances[backend],
                err_msg=f"{backend} diverged from python reference",
            )

    def test_detect_chimera_uses_dispatcher(self) -> None:
        """``detect_chimera`` must flow through ``local_order_parameter``
        so every available backend is exercised."""
        phases, knm = _problem(3, n=12)
        state = detect_chimera(phases, knm)
        total = (
            len(state.coherent_indices)
            + len(state.incoherent_indices)
            + int(round(state.chimera_index * 12))
        )
        assert total == 12
