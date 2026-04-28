# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Per-backend parity for fractal-dimension kernels

"""Cross-backend parity for :func:`correlation_integral` and
:func:`kaplan_yorke_dimension`.

Parity tests run only against the **full-pairs** branch of
``correlation_integral`` — the subsampled branch is RNG-driven and
Rust keeps its own in-kernel RNG for API stability, so the only
universally identical output comes from the deterministic
triu-indices pair list. Tolerances:

* Rust / Julia / Go — 1e-12 (shared f64 on integer pair lists).
* Mojo — 1e-9 (subprocess text round-trip).
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.monitor import dimension as dim_mod
from scpn_phase_orchestrator.monitor.dimension import (
    AVAILABLE_BACKENDS,
    correlation_integral,
    kaplan_yorke_dimension,
)


def _force(backend: str) -> str:
    prev = dim_mod.ACTIVE_BACKEND
    dim_mod.ACTIVE_BACKEND = backend
    return prev


def _reset(prev: str) -> None:
    dim_mod.ACTIVE_BACKEND = prev


def _reference_ci(
    traj: np.ndarray, eps: np.ndarray, max_pairs: int = 10_000
) -> np.ndarray:
    prev = _force("python")
    try:
        return correlation_integral(traj, eps, max_pairs=max_pairs)
    finally:
        _reset(prev)


def _reference_ky(le: np.ndarray) -> float:
    prev = _force("python")
    try:
        return kaplan_yorke_dimension(le)
    finally:
        _reset(prev)


def _trajectory(seed: int, t: int = 40, d: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 1.0, (t, d))


def _eps(n_k: int = 10) -> np.ndarray:
    return np.logspace(-1, 0.5, n_k)


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
    def test_ci_full_pairs(self, t: int, seed: int) -> None:
        traj = _trajectory(seed, t=t)
        eps = _eps()
        ref = _reference_ci(traj, eps)
        prev = _force("rust")
        try:
            got = correlation_integral(traj, eps, max_pairs=10_000)
        finally:
            _reset(prev)
        np.testing.assert_allclose(got, ref, atol=1e-12)

    def test_ky(self) -> None:
        le = np.array([0.5, 0.1, -0.2, -0.5, -0.9])
        ref = _reference_ky(le)
        prev = _force("rust")
        try:
            got = kaplan_yorke_dimension(le)
        finally:
            _reset(prev)
        assert abs(got - ref) < 1e-12


class TestJuliaParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "julia" not in AVAILABLE_BACKENDS:
            pytest.skip("Julia backend not available")

    @pytest.mark.parametrize("seed", [0, 42])
    def test_ci_full_pairs(self, seed: int) -> None:
        traj = _trajectory(seed)
        eps = _eps()
        ref = _reference_ci(traj, eps)
        prev = _force("julia")
        try:
            got = correlation_integral(traj, eps, max_pairs=10_000)
        finally:
            _reset(prev)
        np.testing.assert_allclose(got, ref, atol=1e-12)

    def test_ky(self) -> None:
        le = np.array([0.3, 0.0, -0.1, -0.8])
        ref = _reference_ky(le)
        prev = _force("julia")
        try:
            got = kaplan_yorke_dimension(le)
        finally:
            _reset(prev)
        assert abs(got - ref) < 1e-12


class TestGoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "go" not in AVAILABLE_BACKENDS:
            pytest.skip("Go backend not built")

    @given(
        t=st.integers(min_value=5, max_value=40),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=6,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_ci_full_pairs(self, t: int, seed: int) -> None:
        traj = _trajectory(seed, t=t)
        eps = _eps()
        ref = _reference_ci(traj, eps)
        prev = _force("go")
        try:
            got = correlation_integral(traj, eps, max_pairs=10_000)
        finally:
            _reset(prev)
        np.testing.assert_allclose(got, ref, atol=1e-12)

    def test_ky(self) -> None:
        le = np.array([0.7, 0.0, -1.2])
        ref = _reference_ky(le)
        prev = _force("go")
        try:
            got = kaplan_yorke_dimension(le)
        finally:
            _reset(prev)
        assert abs(got - ref) < 1e-12


class TestMojoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "mojo" not in AVAILABLE_BACKENDS:
            pytest.skip("Mojo backend not built")

    @pytest.mark.parametrize("seed", [0, 77])
    def test_ci_full_pairs(self, seed: int) -> None:
        traj = _trajectory(seed, t=20)
        eps = _eps(n_k=6)
        ref = _reference_ci(traj, eps)
        prev = _force("mojo")
        try:
            got = correlation_integral(traj, eps, max_pairs=10_000)
        finally:
            _reset(prev)
        np.testing.assert_allclose(got, ref, atol=1e-9)

    def test_ky(self) -> None:
        le = np.array([0.25, 0.05, -0.4])
        ref = _reference_ky(le)
        prev = _force("mojo")
        try:
            got = kaplan_yorke_dimension(le)
        finally:
            _reset(prev)
        assert abs(got - ref) < 1e-9


class TestCrossBackendConsistency:
    @pytest.mark.skipif(
        len(AVAILABLE_BACKENDS) < 2,
        reason="Only Python fallback available",
    )
    def test_all_backends_agree(self) -> None:
        traj = _trajectory(2026, t=30)
        eps = _eps()
        le = np.array([0.4, 0.1, -0.2, -0.5])
        ref_ci = _reference_ci(traj, eps)
        ref_ky = _reference_ky(le)
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
                got_ci = correlation_integral(traj, eps, max_pairs=10_000)
                got_ky = kaplan_yorke_dimension(le)
            finally:
                _reset(prev)
            np.testing.assert_allclose(
                got_ci,
                ref_ci,
                atol=tolerances[backend],
                err_msg=f"{backend} CI diverged from python",
            )
            assert abs(got_ky - ref_ky) <= tolerances[backend], (
                f"{backend} KY diverged: {got_ky} vs {ref_ky}"
            )
