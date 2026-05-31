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

from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _winding_julia as winding_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _winding_mojo as winding_mojo,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _winding_validation as winding_validation,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor._winding_go import (
    winding_numbers_go,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor._winding_julia import (
    winding_numbers_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor._winding_mojo import (
    winding_numbers_mojo,
)
from scpn_phase_orchestrator.monitor import winding as w_mod
from scpn_phase_orchestrator.monitor.winding import (
    AVAILABLE_BACKENDS,
    winding_numbers,
)

TWO_PI = 2.0 * np.pi


def test__winding_validation_helper_is_directly_linked_to_backend_tests() -> None:
    assert callable(winding_validation.validate_winding_backend_inputs)
    assert callable(winding_validation.validate_winding_backend_output)


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


class TestDirectBackendBoundaryContracts:
    """Direct optional winding backends validate before runtime loading."""

    @pytest.mark.parametrize(
        "backend",
        [winding_numbers_go, winding_numbers_julia, winding_numbers_mojo],
    )
    @pytest.mark.parametrize(
        ("phases_flat", "t", "n", "match"),
        [
            (np.array([True, False]), 2, 1, "phases_flat"),
            (np.array([0.0, np.nan]), 2, 1, "phases_flat"),
            (np.array([0.0, 1.0], dtype=np.complex128), 2, 1, "real-valued"),
            (np.array([[0.0], [1.0]]), 2, 1, "one-dimensional"),
            (np.array([0.0, 1.0]), True, 1, "t"),
            (np.array([0.0, 1.0]), 1, 1, "t"),
            (np.array([0.0, 1.0]), 2, 0, "n"),
            (np.array([0.0, 1.0]), 2, True, "n"),
            (np.array([0.0, 1.0]), 3, 1, "t\\*n"),
        ],
    )
    def test_validation_precedes_runtime_load(
        self,
        backend,
        phases_flat: np.ndarray,
        t: object,
        n: object,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            backend(phases_flat, t, n)

    @pytest.mark.parametrize(
        ("value", "match"),
        [
            (np.array([np.bool_(True)], dtype=object), "boolean"),
            (np.array([np.inf]), "finite"),
            (np.array([0.5]), "integer"),
            (np.array([3]), "wrapped-increment"),
            (np.array([0, 1]), "shape"),
        ],
    )
    def test_output_validation_rejects_nonphysical_winding_values(
        self, value: np.ndarray, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            winding_validation.validate_winding_backend_output(value, t=4, n=1)

    def test_julia_backend_rejects_fractional_winding_output(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _FakeJulia:
            @staticmethod
            def winding_numbers(phases: np.ndarray, t: int, n: int) -> np.ndarray:
                return np.array([0.5])

        monkeypatch.setattr(winding_julia, "_ensure", lambda: _FakeJulia())
        traj = _problem(44, t=4, n=1)

        with pytest.raises(ValueError, match="integer"):
            winding_numbers_julia(traj.ravel(), 4, 1)

    def test_mojo_backend_rejects_unbounded_winding_output(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(winding_mojo, "_run", lambda payload: [3])
        traj = _problem(45, t=4, n=1)

        with pytest.raises(ValueError, match="wrapped-increment"):
            winding_numbers_mojo(traj.ravel(), 4, 1)


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
