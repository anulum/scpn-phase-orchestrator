# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Per-backend parity for transfer entropy

"""Per-backend parity tests for ``monitor/transfer_entropy.py``."""

from __future__ import annotations

import sys
import types
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


def test_backend_resolution_records_first_available_backend(monkeypatch) -> None:
    calls: list[str] = []

    def unavailable(name: str):
        def _loader() -> dict[str, object]:
            calls.append(name)
            raise ImportError(name)

        return _loader

    def rust_loader() -> dict[str, object]:
        calls.append("rust")
        return {
            "phase_te": lambda _src, _tgt, _bins: 0.125,
            "te_matrix": lambda _flat, n_osc, _n_time, _bins: np.eye(n_osc).ravel(),
        }

    monkeypatch.setitem(te_mod._LOADERS, "rust", rust_loader)
    monkeypatch.setitem(te_mod._LOADERS, "mojo", unavailable("mojo"))
    monkeypatch.setitem(te_mod._LOADERS, "julia", unavailable("julia"))
    monkeypatch.setitem(te_mod._LOADERS, "go", unavailable("go"))

    active, available = te_mod._resolve_backends()

    assert active == "rust"
    assert available == ["rust", "python"]
    assert calls == ["rust", "mojo", "julia", "go"]


def test_rust_loader_exposes_phase_and_matrix_kernels(monkeypatch) -> None:
    fake_spo = types.ModuleType("spo_kernel")
    fake_spo.phase_transfer_entropy_rust = lambda _src, _tgt, _bins: 0.25
    fake_spo.transfer_entropy_matrix_rust = lambda _flat, n_osc, _n_time, _bins: (
        np.zeros(n_osc * n_osc)
    )
    monkeypatch.setitem(sys.modules, "spo_kernel", fake_spo)

    kernels = te_mod._load_rust_fns()

    assert kernels["phase_te"] is fake_spo.phase_transfer_entropy_rust
    assert kernels["te_matrix"] is fake_spo.transfer_entropy_matrix_rust


def test_dispatch_calls_active_backend_with_contiguous_arrays(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def phase_te(src: np.ndarray, tgt: np.ndarray, n_bins: int) -> float:
        captured["phase_src_contiguous"] = src.flags.c_contiguous
        captured["phase_tgt_contiguous"] = tgt.flags.c_contiguous
        captured["phase_bins"] = n_bins
        return 0.375

    def te_matrix(flat: np.ndarray, n_osc: int, n_time: int, n_bins: int) -> np.ndarray:
        captured["matrix_flat_contiguous"] = flat.flags.c_contiguous
        captured["matrix_shape"] = (n_osc, n_time)
        captured["matrix_bins"] = n_bins
        return np.arange(n_osc * n_osc, dtype=np.float64)

    monkeypatch.setitem(
        te_mod._LOADERS,
        "rust",
        lambda: {"phase_te": phase_te, "te_matrix": te_matrix},
    )
    previous = _force("rust")
    try:
        src = np.arange(12, dtype=np.float64)[::2]
        tgt = np.arange(12, dtype=np.float64)[1::2]
        assert phase_transfer_entropy(src, tgt, n_bins=7) == 0.375

        series = np.arange(30, dtype=np.float64).reshape(3, 10)[:, ::2]
        matrix = transfer_entropy_matrix(series, n_bins=5)
    finally:
        _reset(previous)

    assert captured == {
        "phase_src_contiguous": True,
        "phase_tgt_contiguous": True,
        "phase_bins": 7,
        "matrix_flat_contiguous": True,
        "matrix_shape": (3, 5),
        "matrix_bins": 5,
    }
    np.testing.assert_array_equal(matrix, np.arange(9, dtype=np.float64).reshape(3, 3))


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
