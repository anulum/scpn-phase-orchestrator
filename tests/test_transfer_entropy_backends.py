# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Per-backend parity for transfer entropy

"""Per-backend parity tests for ``monitor/transfer_entropy.py``."""

from __future__ import annotations

from typing import get_type_hints

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.monitor import transfer_entropy as te_mod
from scpn_phase_orchestrator.monitor._te_go import phase_te_go, te_matrix_go
from scpn_phase_orchestrator.monitor._te_julia import (
    phase_te_julia,
    te_matrix_julia,
)
from scpn_phase_orchestrator.monitor._te_mojo import phase_te_mojo, te_matrix_mojo
from scpn_phase_orchestrator.monitor.transfer_entropy import (
    AVAILABLE_BACKENDS,
    phase_transfer_entropy,
    transfer_entropy_matrix,
)

TWO_PI = 2.0 * np.pi


def _force(backend: str) -> str:
    prev = te_mod.ACTIVE_BACKEND
    te_mod.ACTIVE_BACKEND = backend
    return prev


def _reset(prev: str) -> None:
    te_mod.ACTIVE_BACKEND = prev


def _reference_te(src: np.ndarray, tgt: np.ndarray, n_bins: int) -> float:
    prev = _force("python")
    try:
        return phase_transfer_entropy(src, tgt, n_bins)
    finally:
        _reset(prev)


def _reference_matrix(series: np.ndarray, n_bins: int) -> np.ndarray:
    prev = _force("python")
    try:
        return transfer_entropy_matrix(series, n_bins)
    finally:
        _reset(prev)


def test_backend_array_contracts_are_parameterised() -> None:
    functions = (
        phase_te_go,
        te_matrix_go,
        phase_te_julia,
        te_matrix_julia,
        phase_te_mojo,
        te_matrix_mojo,
    )
    for fn in functions:
        hints = get_type_hints(fn)
        checked_hints = [
            value
            for key, value in hints.items()
            if key in {"source", "target", "phase_series"}
        ]
        if fn.__name__.startswith("te_matrix"):
            checked_hints.append(hints["return"])
        for hint in checked_hints:
            assert "numpy.ndarray" in str(hint)
            assert "float64" in str(hint)


class TestRustParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "rust" not in AVAILABLE_BACKENDS:
            pytest.skip("Rust backend not built")

    @given(
        n=st.integers(min_value=50, max_value=400),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_phase_te(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        src = rng.uniform(0.0, TWO_PI, size=n)
        tgt = 0.5 * np.roll(src, -1) + 0.5 * rng.uniform(0.0, TWO_PI, size=n)
        ref = _reference_te(src, tgt, 16)
        prev = _force("rust")
        try:
            result = phase_transfer_entropy(src, tgt, 16)
        finally:
            _reset(prev)
        assert abs(result - ref) < 1e-12

    def test_matrix(self) -> None:
        rng = np.random.default_rng(0)
        n_osc, n_time = 6, 200
        series = rng.uniform(0.0, TWO_PI, size=(n_osc, n_time))
        ref = _reference_matrix(series, 8)
        prev = _force("rust")
        try:
            result = transfer_entropy_matrix(series, 8)
        finally:
            _reset(prev)
        np.testing.assert_allclose(result, ref, atol=1e-12)


class TestJuliaParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "julia" not in AVAILABLE_BACKENDS:
            pytest.skip("Julia backend not available")

    @pytest.mark.parametrize("n", [100, 400])
    def test_phase_te(self, n: int) -> None:
        rng = np.random.default_rng(7 + n)
        src = rng.uniform(0.0, TWO_PI, size=n)
        tgt = rng.uniform(0.0, TWO_PI, size=n)
        ref = _reference_te(src, tgt, 12)
        prev = _force("julia")
        try:
            result = phase_transfer_entropy(src, tgt, 12)
        finally:
            _reset(prev)
        assert abs(result - ref) < 1e-12


class TestGoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "go" not in AVAILABLE_BACKENDS:
            pytest.skip("Go backend not built")

    @given(
        n=st.integers(min_value=50, max_value=300),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=8,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_phase_te(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        src = rng.uniform(0.0, TWO_PI, size=n)
        tgt = rng.uniform(0.0, TWO_PI, size=n)
        ref = _reference_te(src, tgt, 16)
        prev = _force("go")
        try:
            result = phase_transfer_entropy(src, tgt, 16)
        finally:
            _reset(prev)
        assert abs(result - ref) < 1e-12


class TestMojoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "mojo" not in AVAILABLE_BACKENDS:
            pytest.skip("Mojo backend not built")

    @pytest.mark.parametrize("n", [80, 200])
    def test_phase_te(self, n: int) -> None:
        rng = np.random.default_rng(17 + n)
        src = rng.uniform(0.0, TWO_PI, size=n)
        tgt = rng.uniform(0.0, TWO_PI, size=n)
        ref = _reference_te(src, tgt, 12)
        prev = _force("mojo")
        try:
            result = phase_transfer_entropy(src, tgt, 12)
        finally:
            _reset(prev)
        assert abs(result - ref) < 1e-9


class TestCrossBackendConsistency:
    @pytest.mark.skipif(
        len(AVAILABLE_BACKENDS) < 2,
        reason="Only Python fallback available",
    )
    def test_all_backends_agree(self) -> None:
        rng = np.random.default_rng(2026)
        n = 250
        src = rng.uniform(0.0, TWO_PI, size=n)
        tgt = rng.uniform(0.0, TWO_PI, size=n)
        ref = _reference_te(src, tgt, 16)
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
                result = phase_transfer_entropy(src, tgt, 16)
            finally:
                _reset(prev)
            assert abs(result - ref) <= tolerances[backend]
