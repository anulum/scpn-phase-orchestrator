# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Per-backend parity for recurrence kernels

"""Cross-backend parity for :func:`recurrence_matrix` and
:func:`cross_recurrence_matrix`.

Outputs are booleans — tolerance is exact array equality. Both
``euclidean`` and ``angular`` metric branches are exercised.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.monitor import recurrence as r_mod
from scpn_phase_orchestrator.monitor.recurrence import (
    AVAILABLE_BACKENDS,
    cross_recurrence_matrix,
    recurrence_matrix,
)


def _force(backend: str) -> str:
    prev = r_mod.ACTIVE_BACKEND
    r_mod.ACTIVE_BACKEND = backend
    return prev


def _reset(prev: str) -> None:
    r_mod.ACTIVE_BACKEND = prev


def _reference_rm(
    traj: np.ndarray,
    epsilon: float,
    metric: str = "euclidean",
) -> np.ndarray:
    prev = _force("python")
    try:
        return recurrence_matrix(traj, epsilon, metric)
    finally:
        _reset(prev)


def _reference_cross(
    a: np.ndarray,
    b: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    prev = _force("python")
    try:
        return cross_recurrence_matrix(a, b, epsilon)
    finally:
        _reset(prev)


def _trajectory(seed: int, t: int = 30, d: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 1, (t, d))


class TestRustParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "rust" not in AVAILABLE_BACKENDS:
            pytest.skip("Rust backend not built")

    @given(
        t=st.integers(min_value=5, max_value=40),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=8,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_rm_euclidean(self, t: int, seed: int) -> None:
        traj = _trajectory(seed, t=t)
        ref = _reference_rm(traj, 0.8)
        prev = _force("rust")
        try:
            got = recurrence_matrix(traj, 0.8)
        finally:
            _reset(prev)
        np.testing.assert_array_equal(got, ref)

    def test_rm_angular(self) -> None:
        traj = _trajectory(3, t=20, d=2)
        ref = _reference_rm(traj, 0.5, metric="angular")
        prev = _force("rust")
        try:
            got = recurrence_matrix(traj, 0.5, metric="angular")
        finally:
            _reset(prev)
        np.testing.assert_array_equal(got, ref)

    def test_cross(self) -> None:
        a = _trajectory(7, t=25)
        b = _trajectory(11, t=25)
        ref = _reference_cross(a, b, 1.0)
        prev = _force("rust")
        try:
            got = cross_recurrence_matrix(a, b, 1.0)
        finally:
            _reset(prev)
        np.testing.assert_array_equal(got, ref)


class TestJuliaParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "julia" not in AVAILABLE_BACKENDS:
            pytest.skip("Julia backend not available")

    @pytest.mark.parametrize("seed", [0, 42])
    def test_rm(self, seed: int) -> None:
        traj = _trajectory(seed)
        ref = _reference_rm(traj, 0.8)
        prev = _force("julia")
        try:
            got = recurrence_matrix(traj, 0.8)
        finally:
            _reset(prev)
        np.testing.assert_array_equal(got, ref)

    def test_angular(self) -> None:
        traj = _trajectory(3, t=20, d=2)
        ref = _reference_rm(traj, 0.5, metric="angular")
        prev = _force("julia")
        try:
            got = recurrence_matrix(traj, 0.5, metric="angular")
        finally:
            _reset(prev)
        np.testing.assert_array_equal(got, ref)


class TestGoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "go" not in AVAILABLE_BACKENDS:
            pytest.skip("Go backend not built")

    @given(
        t=st.integers(min_value=5, max_value=35),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=6,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_rm_euclidean(self, t: int, seed: int) -> None:
        traj = _trajectory(seed, t=t)
        ref = _reference_rm(traj, 0.8)
        prev = _force("go")
        try:
            got = recurrence_matrix(traj, 0.8)
        finally:
            _reset(prev)
        np.testing.assert_array_equal(got, ref)

    def test_cross(self) -> None:
        a = _trajectory(2, t=20)
        b = _trajectory(3, t=20)
        ref = _reference_cross(a, b, 1.0)
        prev = _force("go")
        try:
            got = cross_recurrence_matrix(a, b, 1.0)
        finally:
            _reset(prev)
        np.testing.assert_array_equal(got, ref)


class TestMojoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "mojo" not in AVAILABLE_BACKENDS:
            pytest.skip("Mojo backend not built")

    @pytest.mark.parametrize("seed", [0, 77])
    def test_rm(self, seed: int) -> None:
        traj = _trajectory(seed, t=15)
        ref = _reference_rm(traj, 0.8)
        prev = _force("mojo")
        try:
            got = recurrence_matrix(traj, 0.8)
        finally:
            _reset(prev)
        np.testing.assert_array_equal(got, ref)


class TestCrossBackendConsistency:
    @pytest.mark.skipif(
        len(AVAILABLE_BACKENDS) < 2,
        reason="Only Python fallback available",
    )
    def test_all_backends_agree_rm(self) -> None:
        traj = _trajectory(2026, t=30)
        ref = _reference_rm(traj, 1.0)
        for backend in AVAILABLE_BACKENDS:
            prev = _force(backend)
            try:
                got = recurrence_matrix(traj, 1.0)
            finally:
                _reset(prev)
            np.testing.assert_array_equal(
                got,
                ref,
                err_msg=f"{backend} RM diverged from python",
            )

    @pytest.mark.skipif(
        len(AVAILABLE_BACKENDS) < 2,
        reason="Only Python fallback available",
    )
    def test_all_backends_agree_cross(self) -> None:
        a = _trajectory(2026, t=24)
        b = _trajectory(1337, t=24)
        ref = _reference_cross(a, b, 1.0)
        for backend in AVAILABLE_BACKENDS:
            prev = _force(backend)
            try:
                got = cross_recurrence_matrix(a, b, 1.0)
            finally:
                _reset(prev)
            np.testing.assert_array_equal(
                got,
                ref,
                err_msg=f"{backend} CROSS diverged from python",
            )
