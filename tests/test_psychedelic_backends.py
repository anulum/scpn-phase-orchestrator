# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Per-backend parity for entropy_from_phases

"""Cross-backend parity for :func:`entropy_from_phases`.

Tolerances: Rust / Julia / Go within 1e-12; Mojo within 1e-9 due to
the subprocess text round-trip on the bin-edge float comparisons.
"""

from __future__ import annotations

from typing import get_type_hints

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.monitor import psychedelic as py_mod
from scpn_phase_orchestrator.monitor._psychedelic_go import entropy_from_phases_go
from scpn_phase_orchestrator.monitor._psychedelic_julia import entropy_from_phases_julia
from scpn_phase_orchestrator.monitor._psychedelic_mojo import entropy_from_phases_mojo
from scpn_phase_orchestrator.monitor.psychedelic import (
    AVAILABLE_BACKENDS,
    entropy_from_phases,
)

TWO_PI = 2.0 * np.pi


def _force(backend: str) -> str:
    prev = py_mod.ACTIVE_BACKEND
    py_mod.ACTIVE_BACKEND = backend
    return prev


def _reset(prev: str) -> None:
    py_mod.ACTIVE_BACKEND = prev


def _reference(phases: np.ndarray, n_bins: int) -> float:
    prev = _force("python")
    try:
        return entropy_from_phases(phases, n_bins)
    finally:
        _reset(prev)


def _phases(seed: int, n: int = 500) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0, TWO_PI, n)


def test_backend_array_contracts_are_parameterised() -> None:
    functions = (
        entropy_from_phases_go,
        entropy_from_phases_julia,
        entropy_from_phases_mojo,
    )
    for fn in functions:
        hints = get_type_hints(fn)
        assert "numpy.ndarray" in str(hints["phases"])
        assert "float64" in str(hints["phases"])


class TestRustParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "rust" not in AVAILABLE_BACKENDS:
            pytest.skip("Rust backend not built")

    @given(
        n=st.integers(min_value=10, max_value=2000),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_matches_python(self, n: int, seed: int) -> None:
        phases = _phases(seed, n)
        ref = _reference(phases, 36)
        prev = _force("rust")
        try:
            got = entropy_from_phases(phases, 36)
        finally:
            _reset(prev)
        assert abs(got - ref) < 1e-12


class TestJuliaParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "julia" not in AVAILABLE_BACKENDS:
            pytest.skip("Julia backend not available")

    @pytest.mark.parametrize("seed", [0, 42])
    def test_matches_python(self, seed: int) -> None:
        phases = _phases(seed)
        ref = _reference(phases, 36)
        prev = _force("julia")
        try:
            got = entropy_from_phases(phases, 36)
        finally:
            _reset(prev)
        assert abs(got - ref) < 1e-12


class TestGoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "go" not in AVAILABLE_BACKENDS:
            pytest.skip("Go backend not built")

    @given(
        n=st.integers(min_value=10, max_value=2000),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=8,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_matches_python(self, n: int, seed: int) -> None:
        phases = _phases(seed, n)
        ref = _reference(phases, 36)
        prev = _force("go")
        try:
            got = entropy_from_phases(phases, 36)
        finally:
            _reset(prev)
        assert abs(got - ref) < 1e-12


class TestMojoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "mojo" not in AVAILABLE_BACKENDS:
            pytest.skip("Mojo backend not built")

    @pytest.mark.parametrize("seed", [0, 77])
    def test_matches_python(self, seed: int) -> None:
        phases = _phases(seed)
        ref = _reference(phases, 36)
        prev = _force("mojo")
        try:
            got = entropy_from_phases(phases, 36)
        finally:
            _reset(prev)
        assert abs(got - ref) < 1e-9


class TestCrossBackendConsistency:
    @pytest.mark.skipif(
        len(AVAILABLE_BACKENDS) < 2,
        reason="Only Python fallback available",
    )
    def test_all_backends_agree(self) -> None:
        phases = _phases(2026, n=1000)
        ref = _reference(phases, 36)
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
                got = entropy_from_phases(phases, 36)
            finally:
                _reset(prev)
            assert abs(got - ref) <= tolerances[backend], (
                f"{backend} diverged: {got} vs {ref}"
            )
