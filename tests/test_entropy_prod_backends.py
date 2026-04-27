# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Per-backend parity for entropy production rate

"""Cross-backend parity for :func:`entropy_production_rate`.

All backends must agree with the Python reference on the same input.
Tolerances:

* Rust / Julia / Go — 1e-12 (shared f64).
* Mojo — 1e-9 (subprocess text round-trip; empirically 1e-19).
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.monitor import entropy_prod as ep_mod
from scpn_phase_orchestrator.monitor.entropy_prod import (
    AVAILABLE_BACKENDS,
    entropy_production_rate,
)

TWO_PI = 2.0 * np.pi


def _force(backend: str) -> str:
    prev = ep_mod.ACTIVE_BACKEND
    ep_mod.ACTIVE_BACKEND = backend
    return prev


def _reset(prev: str) -> None:
    ep_mod.ACTIVE_BACKEND = prev


def _reference(phases, omegas, knm, alpha, dt):
    prev = _force("python")
    try:
        return entropy_production_rate(phases, omegas, knm, alpha, dt)
    finally:
        _reset(prev)


def _problem(seed: int, n: int = 6):
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, TWO_PI, size=n)
    omegas = rng.normal(0.0, 0.2, size=n)
    knm = rng.uniform(0.3, 0.9, size=(n, n))
    np.fill_diagonal(knm, 0.0)
    return phases, omegas, knm


class TestRustParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "rust" not in AVAILABLE_BACKENDS:
            pytest.skip("Rust backend not built")

    @given(
        n=st.integers(min_value=2, max_value=16),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=12,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_matches_python(self, n: int, seed: int) -> None:
        phases, omegas, knm = _problem(seed, n)
        ref = _reference(phases, omegas, knm, 0.6, 0.01)
        prev = _force("rust")
        try:
            got = entropy_production_rate(phases, omegas, knm, 0.6, 0.01)
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
        phases, omegas, knm = _problem(seed)
        ref = _reference(phases, omegas, knm, 0.5, 0.01)
        prev = _force("julia")
        try:
            got = entropy_production_rate(phases, omegas, knm, 0.5, 0.01)
        finally:
            _reset(prev)
        assert abs(got - ref) < 1e-12


class TestGoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "go" not in AVAILABLE_BACKENDS:
            pytest.skip("Go backend not built")

    @given(
        n=st.integers(min_value=2, max_value=16),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=8,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_matches_python(self, n: int, seed: int) -> None:
        phases, omegas, knm = _problem(seed, n)
        ref = _reference(phases, omegas, knm, 0.4, 0.02)
        prev = _force("go")
        try:
            got = entropy_production_rate(phases, omegas, knm, 0.4, 0.02)
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
        phases, omegas, knm = _problem(seed)
        ref = _reference(phases, omegas, knm, 0.7, 0.01)
        prev = _force("mojo")
        try:
            got = entropy_production_rate(phases, omegas, knm, 0.7, 0.01)
        finally:
            _reset(prev)
        assert abs(got - ref) < 1e-9


class TestCrossBackendConsistency:
    @pytest.mark.skipif(
        len(AVAILABLE_BACKENDS) < 2,
        reason="Only Python fallback available",
    )
    def test_all_backends_agree(self) -> None:
        phases, omegas, knm = _problem(2026, n=10)
        ref = _reference(phases, omegas, knm, 0.5, 0.01)
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
                got = entropy_production_rate(phases, omegas, knm, 0.5, 0.01)
            finally:
                _reset(prev)
            assert abs(got - ref) <= tolerances[backend], (
                f"{backend} diverged: {got} vs {ref}"
            )
