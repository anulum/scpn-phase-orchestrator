# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Per-backend parity for NPE

"""Per-backend parity tests for ``monitor/npe.py``."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.monitor import npe as npe_mod
from scpn_phase_orchestrator.monitor.npe import (
    AVAILABLE_BACKENDS,
    compute_npe,
    phase_distance_matrix,
)

TWO_PI = 2.0 * np.pi


def _force(backend: str) -> str:
    prev = npe_mod.ACTIVE_BACKEND
    npe_mod.ACTIVE_BACKEND = backend
    return prev


def _reset(prev: str) -> None:
    npe_mod.ACTIVE_BACKEND = prev


def _reference(
    phases: np.ndarray,
) -> tuple[np.ndarray, float]:
    prev = _force("python")
    try:
        pdm = phase_distance_matrix(phases)
        npe = compute_npe(phases)
    finally:
        _reset(prev)
    return pdm, npe


class TestRustParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "rust" not in AVAILABLE_BACKENDS:
            pytest.skip("Rust backend not built")

    @given(
        n=st.integers(min_value=4, max_value=64),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=12, deadline=None)
    def test_parity(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0.0, TWO_PI, size=n)
        ref_pdm, ref_npe = _reference(phases)
        prev = _force("rust")
        try:
            pdm = phase_distance_matrix(phases)
            npe = compute_npe(phases)
        finally:
            _reset(prev)
        np.testing.assert_allclose(pdm, ref_pdm, atol=1e-12)
        assert abs(npe - ref_npe) < 1e-12


class TestJuliaParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "julia" not in AVAILABLE_BACKENDS:
            pytest.skip("Julia backend not available")

    @pytest.mark.parametrize("n", [8, 24, 64])
    def test_parity(self, n: int) -> None:
        rng = np.random.default_rng(7 + n)
        phases = rng.uniform(0.0, TWO_PI, size=n)
        ref_pdm, ref_npe = _reference(phases)
        prev = _force("julia")
        try:
            pdm = phase_distance_matrix(phases)
            npe = compute_npe(phases)
        finally:
            _reset(prev)
        np.testing.assert_allclose(pdm, ref_pdm, atol=1e-12)
        assert abs(npe - ref_npe) < 1e-12


class TestGoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "go" not in AVAILABLE_BACKENDS:
            pytest.skip("Go backend not built")

    @given(
        n=st.integers(min_value=4, max_value=48),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_parity(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0.0, TWO_PI, size=n)
        ref_pdm, ref_npe = _reference(phases)
        prev = _force("go")
        try:
            pdm = phase_distance_matrix(phases)
            npe = compute_npe(phases)
        finally:
            _reset(prev)
        np.testing.assert_allclose(pdm, ref_pdm, atol=1e-12)
        assert abs(npe - ref_npe) < 1e-12


class TestMojoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "mojo" not in AVAILABLE_BACKENDS:
            pytest.skip("Mojo backend not built")

    @pytest.mark.parametrize("n", [6, 12, 24])
    def test_parity(self, n: int) -> None:
        rng = np.random.default_rng(17 + n)
        phases = rng.uniform(0.0, TWO_PI, size=n)
        ref_pdm, ref_npe = _reference(phases)
        prev = _force("mojo")
        try:
            pdm = phase_distance_matrix(phases)
            npe = compute_npe(phases)
        finally:
            _reset(prev)
        np.testing.assert_allclose(pdm, ref_pdm, atol=1e-12)
        # text-protocol budget amplifies in log() across N lifetimes
        assert abs(npe - ref_npe) < 1e-9


class TestCrossBackendConsistency:
    @pytest.mark.skipif(
        len(AVAILABLE_BACKENDS) < 2,
        reason="Only Python fallback available",
    )
    def test_all_backends_agree(self) -> None:
        rng = np.random.default_rng(2026)
        n = 20
        phases = rng.uniform(0.0, TWO_PI, size=n)
        ref_pdm, ref_npe = _reference(phases)

        tolerances = {
            "rust": 1e-12,
            "julia": 1e-12,
            "go": 1e-12,
            "mojo": 1e-9,
            "python": 0.0,
        }
        for backend in AVAILABLE_BACKENDS:
            atol = tolerances[backend]
            prev = _force(backend)
            try:
                pdm = phase_distance_matrix(phases)
                npe = compute_npe(phases)
            finally:
                _reset(prev)
            np.testing.assert_allclose(pdm, ref_pdm, atol=atol)
            assert abs(npe - ref_npe) <= atol
