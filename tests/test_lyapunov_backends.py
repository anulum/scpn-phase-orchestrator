# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Per-backend parity for the Lyapunov spectrum

"""Cross-backend parity tests for :func:`lyapunov_spectrum`.

Every available backend (Rust, Mojo, Julia, Go, Python) must produce
the same spectrum on the same input up to floating-point rounding.
Backends that are not built / toolchain-missing are skipped with a
reason so CI on minimal environments stays green.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor import lyapunov as ly_mod
from scpn_phase_orchestrator.monitor.lyapunov import (
    AVAILABLE_BACKENDS,
    lyapunov_spectrum,
)

TWO_PI = 2.0 * np.pi


def _force(backend: str) -> str:
    prev = ly_mod.ACTIVE_BACKEND
    ly_mod.ACTIVE_BACKEND = backend
    return prev


def _reset(prev: str) -> None:
    ly_mod.ACTIVE_BACKEND = prev


def _reference(phases, omegas, knm, alpha, **kwargs):
    """Python reference spectrum — always the single source of truth."""
    prev = _force("python")
    try:
        return lyapunov_spectrum(phases, omegas, knm, alpha, **kwargs)
    finally:
        _reset(prev)


def _problem(seed: int, n: int = 5, zeta: float = 0.0, psi: float = 0.0):
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, TWO_PI, size=n)
    omegas = rng.normal(0.0, 0.1, size=n)
    knm = rng.uniform(0.5, 1.5, size=(n, n))
    np.fill_diagonal(knm, 0.0)
    alpha = rng.uniform(-0.1, 0.1, size=(n, n))
    np.fill_diagonal(alpha, 0.0)
    kwargs = {
        "dt": 0.01,
        "n_steps": 200,
        "qr_interval": 10,
        "zeta": zeta,
        "psi": psi,
    }
    return phases, omegas, knm, alpha, kwargs


class TestRustParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "rust" not in AVAILABLE_BACKENDS:
            pytest.skip("Rust backend not built")

    @pytest.mark.parametrize("seed", [0, 7, 2026])
    def test_matches_python(self, seed: int) -> None:
        phases, omegas, knm, alpha, kwargs = _problem(seed)
        ref = _reference(phases, omegas, knm, alpha, **kwargs)
        prev = _force("rust")
        try:
            result = lyapunov_spectrum(phases, omegas, knm, alpha, **kwargs)
        finally:
            _reset(prev)
        np.testing.assert_allclose(result, ref, atol=1e-12)

    def test_driver_path(self) -> None:
        phases, omegas, knm, alpha, kwargs = _problem(3, zeta=1.7, psi=0.6)
        ref = _reference(phases, omegas, knm, alpha, **kwargs)
        prev = _force("rust")
        try:
            result = lyapunov_spectrum(phases, omegas, knm, alpha, **kwargs)
        finally:
            _reset(prev)
        np.testing.assert_allclose(result, ref, atol=1e-12)


class TestJuliaParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "julia" not in AVAILABLE_BACKENDS:
            pytest.skip("Julia backend not available")

    @pytest.mark.parametrize("seed", [0, 42])
    def test_matches_python(self, seed: int) -> None:
        phases, omegas, knm, alpha, kwargs = _problem(seed)
        ref = _reference(phases, omegas, knm, alpha, **kwargs)
        prev = _force("julia")
        try:
            result = lyapunov_spectrum(phases, omegas, knm, alpha, **kwargs)
        finally:
            _reset(prev)
        np.testing.assert_allclose(result, ref, atol=1e-12)


class TestGoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "go" not in AVAILABLE_BACKENDS:
            pytest.skip("Go backend not built")

    @pytest.mark.parametrize("seed", [0, 99, 2026])
    def test_matches_python(self, seed: int) -> None:
        phases, omegas, knm, alpha, kwargs = _problem(seed)
        ref = _reference(phases, omegas, knm, alpha, **kwargs)
        prev = _force("go")
        try:
            result = lyapunov_spectrum(phases, omegas, knm, alpha, **kwargs)
        finally:
            _reset(prev)
        np.testing.assert_allclose(result, ref, atol=1e-12)

    def test_driver_and_phase_lag(self) -> None:
        phases, omegas, knm, alpha, kwargs = _problem(5, zeta=1.1, psi=-0.4)
        ref = _reference(phases, omegas, knm, alpha, **kwargs)
        prev = _force("go")
        try:
            result = lyapunov_spectrum(phases, omegas, knm, alpha, **kwargs)
        finally:
            _reset(prev)
        np.testing.assert_allclose(result, ref, atol=1e-12)


class TestMojoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "mojo" not in AVAILABLE_BACKENDS:
            pytest.skip("Mojo backend not built")

    @pytest.mark.parametrize("seed", [0, 42])
    def test_matches_python(self, seed: int) -> None:
        phases, omegas, knm, alpha, kwargs = _problem(seed)
        ref = _reference(phases, omegas, knm, alpha, **kwargs)
        prev = _force("mojo")
        try:
            result = lyapunov_spectrum(phases, omegas, knm, alpha, **kwargs)
        finally:
            _reset(prev)
        # Mojo atof round-trips via the subprocess text stream.
        np.testing.assert_allclose(result, ref, atol=1e-6)


class TestCrossBackendConsistency:
    @pytest.mark.skipif(
        len(AVAILABLE_BACKENDS) < 2,
        reason="Only Python fallback available",
    )
    def test_all_backends_agree(self) -> None:
        phases, omegas, knm, alpha, kwargs = _problem(2026, zeta=0.5, psi=0.2)
        ref = _reference(phases, omegas, knm, alpha, **kwargs)
        tolerances = {
            "rust": 1e-12,
            "julia": 1e-12,
            "go": 1e-12,
            "mojo": 1e-6,
            "python": 0.0,
        }
        for backend in AVAILABLE_BACKENDS:
            prev = _force(backend)
            try:
                result = lyapunov_spectrum(phases, omegas, knm, alpha, **kwargs)
            finally:
                _reset(prev)
            np.testing.assert_allclose(
                result,
                ref,
                atol=tolerances[backend],
                err_msg=f"{backend} diverged from python reference",
            )


class TestDispatcherResolution:
    def test_active_backend_is_first_available(self) -> None:
        assert AVAILABLE_BACKENDS[0] == ly_mod.ACTIVE_BACKEND

    def test_python_always_available(self) -> None:
        assert "python" in AVAILABLE_BACKENDS
