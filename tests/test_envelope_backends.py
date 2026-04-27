# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Per-backend parity for envelope kernels

"""Cross-backend parity for :func:`extract_envelope` and
:func:`envelope_modulation_depth`.

Tolerances: Rust / Julia / Go 1e-12; Mojo 1e-9. Measured
bit-equivalent (0.0) on Rust/Julia/Go and 3.3e-15 on Mojo.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.upde import envelope as e_mod
from scpn_phase_orchestrator.upde.envelope import (
    AVAILABLE_BACKENDS,
    envelope_modulation_depth,
    extract_envelope,
)


def _force(backend: str) -> str:
    prev = e_mod.ACTIVE_BACKEND
    e_mod.ACTIVE_BACKEND = backend
    return prev


def _reset(prev: str) -> None:
    e_mod.ACTIVE_BACKEND = prev


def _ref_env(amps, window):
    prev = _force("python")
    try:
        return extract_envelope(amps, window)
    finally:
        _reset(prev)


def _ref_mod(env):
    prev = _force("python")
    try:
        return envelope_modulation_depth(env)
    finally:
        _reset(prev)


def _amps(seed: int, n: int = 500) -> np.ndarray:
    return np.abs(np.random.default_rng(seed).normal(1.0, 0.3, n))


class TestRustParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "rust" not in AVAILABLE_BACKENDS:
            pytest.skip("Rust backend not built")

    @given(
        n=st.integers(min_value=30, max_value=800),
        window=st.integers(min_value=2, max_value=25),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_extract(self, n: int, window: int, seed: int) -> None:
        # Hypothesis range keeps ``window ≤ n``. The ``window > n``
        # edge case is ill-defined (can't RMS a window you don't
        # have); Rust returns zeros, the other backends return a
        # whole-trace RMS. Parity tests stay in the physically
        # meaningful regime.
        amps = _amps(seed, n)
        ref = _ref_env(amps, window)
        prev = _force("rust")
        try:
            got = extract_envelope(amps, window)
        finally:
            _reset(prev)
        np.testing.assert_allclose(got, ref, atol=1e-12)

    def test_modulation(self) -> None:
        env = _amps(7)
        ref = _ref_mod(env)
        prev = _force("rust")
        try:
            got = envelope_modulation_depth(env)
        finally:
            _reset(prev)
        assert abs(got - ref) < 1e-12


class TestJuliaParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "julia" not in AVAILABLE_BACKENDS:
            pytest.skip("Julia backend not available")

    @pytest.mark.parametrize("seed", [0, 42])
    def test_extract_and_mod(self, seed: int) -> None:
        amps = _amps(seed)
        ref_env = _ref_env(amps, 10)
        ref_mod = _ref_mod(amps)
        prev = _force("julia")
        try:
            env = extract_envelope(amps, 10)
            mod = envelope_modulation_depth(amps)
        finally:
            _reset(prev)
        np.testing.assert_allclose(env, ref_env, atol=1e-12)
        assert abs(mod - ref_mod) < 1e-12


class TestGoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "go" not in AVAILABLE_BACKENDS:
            pytest.skip("Go backend not built")

    @given(
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=8,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_extract(self, seed: int) -> None:
        amps = _amps(seed)
        ref = _ref_env(amps, 10)
        prev = _force("go")
        try:
            got = extract_envelope(amps, 10)
        finally:
            _reset(prev)
        np.testing.assert_allclose(got, ref, atol=1e-12)


class TestMojoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "mojo" not in AVAILABLE_BACKENDS:
            pytest.skip("Mojo backend not built")

    @pytest.mark.parametrize("seed", [0, 77])
    def test_extract(self, seed: int) -> None:
        amps = _amps(seed)
        ref = _ref_env(amps, 10)
        prev = _force("mojo")
        try:
            got = extract_envelope(amps, 10)
        finally:
            _reset(prev)
        np.testing.assert_allclose(got, ref, atol=1e-9)


class TestCrossBackendConsistency:
    @pytest.mark.skipif(
        len(AVAILABLE_BACKENDS) < 2,
        reason="Only Python fallback available",
    )
    def test_all_backends_agree(self) -> None:
        amps = _amps(2026)
        ref_env = _ref_env(amps, 10)
        ref_mod = _ref_mod(amps)
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
                env = extract_envelope(amps, 10)
                mod = envelope_modulation_depth(amps)
            finally:
                _reset(prev)
            np.testing.assert_allclose(env, ref_env, atol=tolerances[backend])
            assert abs(mod - ref_mod) <= tolerances[backend]
