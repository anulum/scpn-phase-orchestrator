# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Per-backend parity for Hodge decomposition

"""Cross-backend parity for :func:`hodge_decomposition`.

All four non-Python backends must match the reference within 1e-12
on gradient + curl + harmonic; Mojo at 1e-9 due to subprocess text
round-trip.
"""

from __future__ import annotations

import sys
import types
from typing import get_type_hints

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.coupling import hodge as h_mod
from scpn_phase_orchestrator.coupling.hodge import (
    AVAILABLE_BACKENDS,
    hodge_decomposition,
)
from scpn_phase_orchestrator.experimental.accelerators.coupling._hodge_go import (
    hodge_decomposition_go,
)
from scpn_phase_orchestrator.experimental.accelerators.coupling._hodge_julia import (
    hodge_decomposition_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.coupling._hodge_mojo import (
    hodge_decomposition_mojo,
)

TWO_PI = 2.0 * np.pi


def _force(backend: str) -> str:
    prev = h_mod.ACTIVE_BACKEND
    h_mod.ACTIVE_BACKEND = backend
    return prev


def _reset(prev: str) -> None:
    h_mod.ACTIVE_BACKEND = prev


def _reference(knm, phases):
    prev = _force("python")
    try:
        return hodge_decomposition(knm, phases)
    finally:
        _reset(prev)


def _problem(seed: int, n: int = 16):
    rng = np.random.default_rng(seed)
    k = rng.uniform(-1, 1, (n, n))
    np.fill_diagonal(k, 0.0)
    phases = rng.uniform(0, TWO_PI, n)
    return k, phases


class TestRustParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "rust" not in AVAILABLE_BACKENDS:
            pytest.skip("Rust backend not built")

    @given(
        n=st.integers(min_value=2, max_value=24),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_matches_python(self, n: int, seed: int) -> None:
        knm, phases = _problem(seed, n)
        ref = _reference(knm, phases)
        prev = _force("rust")
        try:
            got = hodge_decomposition(knm, phases)
        finally:
            _reset(prev)
        np.testing.assert_allclose(got.gradient, ref.gradient, atol=1e-12)
        np.testing.assert_allclose(got.curl, ref.curl, atol=1e-12)
        np.testing.assert_allclose(got.harmonic, ref.harmonic, atol=1e-12)


class TestJuliaParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "julia" not in AVAILABLE_BACKENDS:
            pytest.skip("Julia backend not available")

    @pytest.mark.parametrize("seed", [0, 42])
    def test_matches_python(self, seed: int) -> None:
        knm, phases = _problem(seed)
        ref = _reference(knm, phases)
        prev = _force("julia")
        try:
            got = hodge_decomposition(knm, phases)
        finally:
            _reset(prev)
        np.testing.assert_allclose(got.gradient, ref.gradient, atol=1e-12)
        np.testing.assert_allclose(got.curl, ref.curl, atol=1e-12)
        np.testing.assert_allclose(got.harmonic, ref.harmonic, atol=1e-12)


class TestGoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "go" not in AVAILABLE_BACKENDS:
            pytest.skip("Go backend not built")

    @given(
        n=st.integers(min_value=2, max_value=24),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=8,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_matches_python(self, n: int, seed: int) -> None:
        knm, phases = _problem(seed, n)
        ref = _reference(knm, phases)
        prev = _force("go")
        try:
            got = hodge_decomposition(knm, phases)
        finally:
            _reset(prev)
        np.testing.assert_allclose(got.gradient, ref.gradient, atol=1e-12)
        np.testing.assert_allclose(got.curl, ref.curl, atol=1e-12)
        np.testing.assert_allclose(got.harmonic, ref.harmonic, atol=1e-12)


class TestMojoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "mojo" not in AVAILABLE_BACKENDS:
            pytest.skip("Mojo backend not built")

    @pytest.mark.parametrize("seed", [0, 77])
    def test_matches_python(self, seed: int) -> None:
        knm, phases = _problem(seed)
        ref = _reference(knm, phases)
        prev = _force("mojo")
        try:
            got = hodge_decomposition(knm, phases)
        finally:
            _reset(prev)
        np.testing.assert_allclose(got.gradient, ref.gradient, atol=1e-9)
        np.testing.assert_allclose(got.curl, ref.curl, atol=1e-9)
        np.testing.assert_allclose(got.harmonic, ref.harmonic, atol=1e-9)


class TestCrossBackendConsistency:
    @pytest.mark.skipif(
        len(AVAILABLE_BACKENDS) < 2,
        reason="Only Python fallback available",
    )
    def test_all_backends_agree(self) -> None:
        knm, phases = _problem(2026, n=20)
        ref = _reference(knm, phases)
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
                got = hodge_decomposition(knm, phases)
            finally:
                _reset(prev)
            atol = tolerances[backend]
            np.testing.assert_allclose(got.gradient, ref.gradient, atol=atol)
            np.testing.assert_allclose(got.curl, ref.curl, atol=atol)
            np.testing.assert_allclose(got.harmonic, ref.harmonic, atol=atol)


class TestBackendTypingContracts:
    @pytest.mark.parametrize(
        ("fn", "label"),
        [
            (hodge_decomposition_go, "go"),
            (hodge_decomposition_julia, "julia"),
            (hodge_decomposition_mojo, "mojo"),
        ],
    )
    def test_backend_annotations_use_float64_ndarray(self, fn, label: str) -> None:
        hints = get_type_hints(fn)
        for name in ("knm_flat", "phases", "return"):
            text = str(hints[name])
            assert "numpy.ndarray" in text, f"{label}:{name} missing ndarray annotation"
            assert "numpy.float64" in text, f"{label}:{name} missing float64 annotation"


class TestBackendLoaderDispatch:
    def test_rust_loader_wraps_spo_kernel_flattened_arrays(self, monkeypatch) -> None:
        calls: list[tuple[np.ndarray, np.ndarray, int]] = []

        def fake_rust(knm_flat, phases, n):
            calls.append((knm_flat, phases, n))
            return (
                np.full(n, 1.0, dtype=np.float64),
                np.full(n, 2.0, dtype=np.float64),
                np.full(n, 3.0, dtype=np.float64),
            )

        fake_module = types.ModuleType("spo_kernel")
        fake_module.hodge_decomposition_rust = fake_rust
        monkeypatch.setitem(sys.modules, "spo_kernel", fake_module)

        backend = h_mod._load_rust_fn()
        knm = np.array([[0.0, 0.5], [-0.25, 0.0]], dtype=np.float64)
        phases = np.array([0.1, 0.4], dtype=np.float64)

        gradient, curl, harmonic = backend(knm, phases, 2)

        np.testing.assert_array_equal(gradient, [1.0, 1.0])
        np.testing.assert_array_equal(curl, [2.0, 2.0])
        np.testing.assert_array_equal(harmonic, [3.0, 3.0])
        assert calls[0][2] == 2
        assert calls[0][0].flags.c_contiguous
        np.testing.assert_array_equal(calls[0][0], knm.ravel())

    def test_julia_loader_requires_juliacall_then_returns_backend(
        self,
        monkeypatch,
    ) -> None:
        sentinel = object()
        fake_juliacall = types.ModuleType("juliacall")
        fake_backend = types.ModuleType(
            "scpn_phase_orchestrator.experimental.accelerators.coupling._hodge_julia"
        )
        fake_backend.hodge_decomposition_julia = sentinel
        monkeypatch.setitem(sys.modules, "juliacall", fake_juliacall)
        monkeypatch.setitem(
            sys.modules,
            "scpn_phase_orchestrator.experimental.accelerators.coupling._hodge_julia",
            fake_backend,
        )

        assert h_mod._load_julia_fn() is sentinel
