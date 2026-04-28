# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Per-backend parity for winding numbers

"""Cross-backend parity for :func:`winding_numbers`.

All backends must produce the same integer output (tolerance 0).
Because the final ``floor`` truncates to ``int64``, any float noise
at the ``1e-12`` level vanishes; the measured cross-backend
disagreement is exactly ``0`` across seeds and sizes.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.monitor import winding as w_mod
from scpn_phase_orchestrator.monitor.winding import (
    AVAILABLE_BACKENDS,
    winding_numbers,
)

TWO_PI = 2.0 * np.pi


def _force(backend: str) -> str:
    prev = w_mod.ACTIVE_BACKEND
    w_mod.ACTIVE_BACKEND = backend
    return prev


def _reset(prev: str) -> None:
    w_mod.ACTIVE_BACKEND = prev


def _reference(traj: np.ndarray) -> np.ndarray:
    prev = _force("python")
    try:
        return winding_numbers(traj)
    finally:
        _reset(prev)


def _problem(seed: int, t: int = 300, n: int = 6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    omegas = rng.normal(0, 0.5, n)
    dt = 0.05
    hist = np.zeros((t, n))
    hist[0] = rng.uniform(0, TWO_PI, n)
    for i in range(1, t):
        hist[i] = (hist[i - 1] + omegas * dt) % TWO_PI
    return hist


class TestRustParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "rust" not in AVAILABLE_BACKENDS:
            pytest.skip("Rust backend not built")

    @given(
        n=st.integers(min_value=1, max_value=16),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_matches_python(self, n: int, seed: int) -> None:
        traj = _problem(seed, n=n)
        ref = _reference(traj)
        prev = _force("rust")
        try:
            got = winding_numbers(traj)
        finally:
            _reset(prev)
        np.testing.assert_array_equal(got, ref)


class TestJuliaParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "julia" not in AVAILABLE_BACKENDS:
            pytest.skip("Julia backend not available")

    @pytest.mark.parametrize("seed", [0, 42])
    def test_matches_python(self, seed: int) -> None:
        traj = _problem(seed)
        ref = _reference(traj)
        prev = _force("julia")
        try:
            got = winding_numbers(traj)
        finally:
            _reset(prev)
        np.testing.assert_array_equal(got, ref)


class TestGoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "go" not in AVAILABLE_BACKENDS:
            pytest.skip("Go backend not built")

    @given(
        n=st.integers(min_value=1, max_value=16),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=8,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_matches_python(self, n: int, seed: int) -> None:
        traj = _problem(seed, n=n)
        ref = _reference(traj)
        prev = _force("go")
        try:
            got = winding_numbers(traj)
        finally:
            _reset(prev)
        np.testing.assert_array_equal(got, ref)


class TestMojoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "mojo" not in AVAILABLE_BACKENDS:
            pytest.skip("Mojo backend not built")

    @pytest.mark.parametrize("seed", [0, 77])
    def test_matches_python(self, seed: int) -> None:
        traj = _problem(seed)
        ref = _reference(traj)
        prev = _force("mojo")
        try:
            got = winding_numbers(traj)
        finally:
            _reset(prev)
        np.testing.assert_array_equal(got, ref)


class TestCrossBackendConsistency:
    @pytest.mark.skipif(
        len(AVAILABLE_BACKENDS) < 2,
        reason="Only Python fallback available",
    )
    def test_all_backends_agree(self) -> None:
        traj = _problem(2026, t=500, n=8)
        ref = _reference(traj)
        for backend in AVAILABLE_BACKENDS:
            prev = _force(backend)
            try:
                got = winding_numbers(traj)
            finally:
                _reset(prev)
            np.testing.assert_array_equal(
                got,
                ref,
                err_msg=f"{backend} diverged from python reference",
            )
