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

from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _pac_go as pac_go_mod,
)
from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _pac_julia as pac_julia_mod,
)
from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _pac_mojo as pac_mojo_mod,
)
from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _pac_validation as pac_validation,
)
from scpn_phase_orchestrator.upde import pac as pac_mod
from scpn_phase_orchestrator.upde.pac import (
    AVAILABLE_BACKENDS,
    modulation_index,
    pac_matrix,
)

TWO_PI = 2.0 * np.pi
MIBackend = Callable[..., float]
MatrixBackend = Callable[..., np.ndarray]
DIRECT_MI_BACKENDS: tuple[MIBackend, ...] = (
    pac_go_mod.modulation_index_go,
    pac_julia_mod.modulation_index_julia,
    pac_mojo_mod.modulation_index_mojo,
)
DIRECT_MATRIX_BACKENDS: tuple[MatrixBackend, ...] = (
    pac_go_mod.pac_matrix_go,
    pac_julia_mod.pac_matrix_julia,
    pac_mojo_mod.pac_matrix_mojo,
)


def test_pac_validation_helper_is_linked_to_backend_tests() -> None:
    assert callable(pac_validation.validate_modulation_index_inputs)
    assert callable(pac_validation.validate_modulation_index_output)
    assert callable(pac_validation.validate_pac_matrix_inputs)
    assert callable(pac_validation.validate_pac_matrix_output)


def _force(backend: str) -> str:
    prev = pac_mod.ACTIVE_BACKEND
    pac_mod.ACTIVE_BACKEND = backend
    return prev


def _reset(prev: str) -> None:
    pac_mod.ACTIVE_BACKEND = prev


def _reference_mi(theta: np.ndarray, amp: np.ndarray, n_bins: int) -> float:
    prev = _force("python")
    try:
        return modulation_index(theta, amp, n_bins)
    finally:
        _reset(prev)


def _reference_matrix(phases: np.ndarray, amps: np.ndarray, n_bins: int) -> np.ndarray:
    prev = _force("python")
    try:
        return pac_matrix(phases, amps, n_bins)
    finally:
        _reset(prev)


def _mi_payload(samples: int = 16) -> tuple[np.ndarray, np.ndarray, int]:
    theta = np.linspace(0.0, TWO_PI, samples, endpoint=False, dtype=np.float64)
    amp = 1.0 + 0.25 * np.cos(theta)
    return theta, amp, 12


def _matrix_payload(
    t: int = 8,
    n: int = 3,
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    theta = np.linspace(0.0, TWO_PI, t, endpoint=False, dtype=np.float64)
    phases = np.column_stack([theta + 0.1 * idx for idx in range(n)])
    amps = 1.0 + 0.2 * np.cos(phases)
    return phases.ravel(), amps.ravel(), t, n, 12


def _forbid_runtime_load(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail_loader() -> object:
        raise AssertionError("optional backend loaded before validation")

    monkeypatch.setattr(pac_go_mod, "_load_lib", _fail_loader)
    monkeypatch.setattr(pac_julia_mod, "_ensure_julia_loaded", _fail_loader)
    monkeypatch.setattr(pac_mojo_mod, "_ensure_exe", _fail_loader)


class TestDirectBackendBoundaryContracts:
    @pytest.mark.parametrize("backend", DIRECT_MI_BACKENDS)
    @pytest.mark.parametrize(
        ("theta", "amp", "n_bins"),
        [
            (_mi_payload()[0].reshape(2, -1), _mi_payload()[1], 12),
            (_mi_payload()[0].astype(bool), _mi_payload()[1], 12),
            (_mi_payload()[0].astype(np.complex128) + 1j, _mi_payload()[1], 12),
            (np.array([np.nan, *_mi_payload()[0][1:]]), _mi_payload()[1], 12),
            (_mi_payload()[0], _mi_payload()[1].reshape(2, -1), 12),
            (_mi_payload()[0], _mi_payload()[1].astype(bool), 12),
            (_mi_payload()[0], _mi_payload()[1].astype(np.complex128) + 1j, 12),
            (_mi_payload()[0], np.array([np.inf, *_mi_payload()[1][1:]]), 12),
            (_mi_payload()[0], np.array([-0.1, *_mi_payload()[1][1:]]), 12),
            (_mi_payload()[0], _mi_payload()[1], True),
            (_mi_payload()[0], _mi_payload()[1], 1),
        ],
    )
    def test_modulation_index_rejects_invalid_inputs_before_runtime_loading(
        self,
        monkeypatch: pytest.MonkeyPatch,
        backend: MIBackend,
        theta: object,
        amp: object,
        n_bins: object,
    ) -> None:
        _forbid_runtime_load(monkeypatch)

        with pytest.raises((TypeError, ValueError)):
            backend(theta, amp, n_bins)

    @pytest.mark.parametrize("backend", DIRECT_MI_BACKENDS)
    def test_modulation_index_empty_common_window_returns_zero_without_runtime(
        self, monkeypatch: pytest.MonkeyPatch, backend: MIBackend
    ) -> None:
        _forbid_runtime_load(monkeypatch)

        assert backend(np.array([], dtype=np.float64), np.ones(3), 12) == 0.0

    @pytest.mark.parametrize("backend", DIRECT_MATRIX_BACKENDS)
    @pytest.mark.parametrize(
        ("phases", "amps", "t", "n", "n_bins"),
        [
            (_matrix_payload()[0].reshape(2, -1), _matrix_payload()[1], 8, 3, 12),
            (_matrix_payload()[0].astype(bool), _matrix_payload()[1], 8, 3, 12),
            (
                _matrix_payload()[0].astype(np.complex128) + 1j,
                _matrix_payload()[1],
                8,
                3,
                12,
            ),
            (
                np.array([np.nan, *_matrix_payload()[0][1:]]),
                _matrix_payload()[1],
                8,
                3,
                12,
            ),
            (_matrix_payload()[0][:-1], _matrix_payload()[1], 8, 3, 12),
            (_matrix_payload()[0], _matrix_payload()[1].reshape(2, -1), 8, 3, 12),
            (_matrix_payload()[0], _matrix_payload()[1].astype(bool), 8, 3, 12),
            (
                _matrix_payload()[0],
                _matrix_payload()[1].astype(np.complex128) + 1j,
                8,
                3,
                12,
            ),
            (
                _matrix_payload()[0],
                np.array([np.inf, *_matrix_payload()[1][1:]]),
                8,
                3,
                12,
            ),
            (
                _matrix_payload()[0],
                np.array([-0.1, *_matrix_payload()[1][1:]]),
                8,
                3,
                12,
            ),
            (_matrix_payload()[0], _matrix_payload()[1], True, 3, 12),
            (_matrix_payload()[0], _matrix_payload()[1], 0, 3, 12),
            (_matrix_payload()[0], _matrix_payload()[1], 8, True, 12),
            (_matrix_payload()[0], _matrix_payload()[1], 8, 0, 12),
            (_matrix_payload()[0], _matrix_payload()[1], 8, 3, True),
            (_matrix_payload()[0], _matrix_payload()[1], 8, 3, 1),
        ],
    )
    def test_pac_matrix_rejects_invalid_inputs_before_runtime_loading(
        self,
        monkeypatch: pytest.MonkeyPatch,
        backend: MatrixBackend,
        phases: object,
        amps: object,
        t: object,
        n: object,
        n_bins: object,
    ) -> None:
        _forbid_runtime_load(monkeypatch)

        with pytest.raises((TypeError, ValueError)):
            backend(phases, amps, t, n, n_bins)

    @pytest.mark.parametrize(
        "output",
        [np.nan, -0.1, 1.2, True, np.array([0.1, 0.2]), 0.1 + 0.0j],
    )
    def test_modulation_index_output_rejects_non_physical_payloads(
        self, output: object
    ) -> None:
        with pytest.raises((TypeError, ValueError)):
            pac_validation.validate_modulation_index_output(output)

    @pytest.mark.parametrize(
        "output",
        [
            np.array([0.1, np.nan, 0.2, 0.3]),
            np.array([0.1, -0.1, 0.2, 0.3]),
            np.array([0.1, 1.2, 0.2, 0.3]),
            np.array([0.1, 0.2, 0.3]),
            np.array([True, False, True, False]),
            np.array([0.1 + 0.0j, 0.2 + 0.0j, 0.3 + 0.0j, 0.4 + 0.0j]),
        ],
    )
    def test_pac_matrix_output_rejects_non_physical_payloads(
        self, output: np.ndarray
    ) -> None:
        with pytest.raises((TypeError, ValueError)):
            pac_validation.validate_pac_matrix_output(output, n=2)


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
    def test_modulation_index_bit_exact(self, n: int, n_bins: int, seed: int) -> None:
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
    def test_modulation_index(self, n: int, n_bins: int, seed: int) -> None:
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


class TestDirectMojoBoundaryContracts:
    @pytest.mark.parametrize(
        ("stdout", "expected_count", "label", "match"),
        [
            ("", 1, "MI", "Mojo PAC MI returned 0 lines, expected 1"),
            ("0.1\n0.2\n", 1, "MI", "Mojo PAC MI returned 2 lines, expected 1"),
            (
                "0.1\n\n0.2\n",
                2,
                "matrix",
                "Mojo PAC matrix returned 3 lines, expected 2",
            ),
            ("0.1\nnot-a-number\n", 2, "matrix", "finite real values"),
            ("0.1\nnan\n", 2, "matrix", "finite real values"),
        ],
    )
    def test_mojo_runner_rejects_malformed_raw_stdout(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stdout: str,
        expected_count: int,
        label: str,
        match: str,
    ) -> None:
        monkeypatch.setattr(pac_mojo_mod, "_ensure_exe", lambda: "pac_mojo")
        monkeypatch.setattr(
            pac_mojo_mod.subprocess,
            "run",
            lambda *_args, **_kwargs: SimpleNamespace(
                returncode=0,
                stdout=stdout,
                stderr="",
            ),
        )

        with pytest.raises(ValueError, match=match):
            pac_mojo_mod._run("MI\n", expected_count=expected_count, label=label)


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
