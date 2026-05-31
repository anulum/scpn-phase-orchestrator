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

from typing import get_type_hints

import numpy as np
import pytest

from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _lyapunov_validation as lyapunov_validation,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor._lyapunov_go import (
    lyapunov_spectrum_go,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor._lyapunov_julia import (
    lyapunov_spectrum_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor._lyapunov_mojo import (
    lyapunov_spectrum_mojo,
)
from scpn_phase_orchestrator.monitor import lyapunov as ly_mod
from scpn_phase_orchestrator.monitor.lyapunov import (
    AVAILABLE_BACKENDS,
    lyapunov_spectrum,
)
from tests.typing_contracts import assert_precise_ndarray_hint

TWO_PI = 2.0 * np.pi


def test__lyapunov_validation_helper_is_directly_linked_to_backend_tests() -> None:
    assert callable(lyapunov_validation.validate_lyapunov_backend_inputs)


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


class TestBackendTypingContracts:
    @pytest.mark.parametrize(
        ("fn", "label"),
        [
            (lyapunov_spectrum_go, "go"),
            (lyapunov_spectrum_julia, "julia"),
            (lyapunov_spectrum_mojo, "mojo"),
        ],
    )
    def test_backend_annotations_use_float64_ndarray(self, fn, label: str) -> None:
        hints = get_type_hints(fn)
        for name in ("phases_init", "omegas", "knm", "alpha", "return"):
            text = str(hints[name])
            assert_precise_ndarray_hint(
                hints[name],
                context=f"{label}:{name}",
            )
            assert "numpy.float64" in text, f"{label}:{name} missing float64 annotation"


class TestDirectBackendBoundaryContracts:
    def test_go_bridge_rejects_self_coupling_before_library_load(self) -> None:
        with pytest.raises(ValueError, match="knm diagonal"):
            lyapunov_spectrum_go(
                np.array([0.0, 0.1], dtype=np.float64),
                np.array([1.0, 1.0], dtype=np.float64),
                np.array([[0.2, 0.1], [0.1, 0.0]], dtype=np.float64),
                np.zeros((2, 2), dtype=np.float64),
                0.01,
                10,
                2,
                0.0,
                0.0,
            )

    def test_julia_bridge_rejects_self_coupling_before_runtime_load(self) -> None:
        with pytest.raises(ValueError, match="knm diagonal"):
            lyapunov_spectrum_julia(
                np.array([0.0, 0.1], dtype=np.float64),
                np.array([1.0, 1.0], dtype=np.float64),
                np.array([[0.2, 0.1], [0.1, 0.0]], dtype=np.float64),
                np.zeros((2, 2), dtype=np.float64),
                0.01,
                10,
                2,
                0.0,
                0.0,
            )

    def test_mojo_bridge_rejects_self_coupling_before_executable_load(self) -> None:
        with pytest.raises(ValueError, match="knm diagonal"):
            lyapunov_spectrum_mojo(
                np.array([0.0, 0.1], dtype=np.float64),
                np.array([1.0, 1.0], dtype=np.float64),
                np.array([[0.2, 0.1], [0.1, 0.0]], dtype=np.float64),
                np.zeros((2, 2), dtype=np.float64),
                0.01,
                10,
                2,
                0.0,
                0.0,
            )

    @pytest.mark.parametrize(
        ("fn", "label"),
        [
            (lyapunov_spectrum_go, "go"),
            (lyapunov_spectrum_julia, "julia"),
            (lyapunov_spectrum_mojo, "mojo"),
        ],
    )
    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            (
                "phases_init",
                np.array([0.0, np.bool_(True)], dtype=object),
                "phases_init",
            ),
            ("omegas", np.array([1.0, np.nan], dtype=np.float64), "omegas"),
            ("knm", np.array([[0.0, 0.1], [0.1, np.inf]], dtype=np.float64), "knm"),
            ("alpha", np.array([[0.0, 0.0], [0.0, 0.0j]], dtype=complex), "alpha"),
            ("dt", np.bool_(True), "dt"),
            ("n_steps", np.bool_(True), "n_steps"),
            ("qr_interval", np.bool_(True), "qr_interval"),
            ("zeta", -0.1, "zeta"),
            ("psi", np.nan, "psi"),
        ],
    )
    def test_direct_backend_rejects_invalid_inputs_before_runtime_load(
        self,
        fn,
        label: str,
        field: str,
        value: object,
        match: str,
    ) -> None:
        kwargs: dict[str, object] = {
            "phases_init": np.array([0.0, 0.1], dtype=np.float64),
            "omegas": np.array([1.0, 1.0], dtype=np.float64),
            "knm": np.array([[0.0, 0.1], [0.1, 0.0]], dtype=np.float64),
            "alpha": np.zeros((2, 2), dtype=np.float64),
            "dt": 0.01,
            "n_steps": 10,
            "qr_interval": 2,
            "zeta": 0.0,
            "psi": 0.0,
        }
        kwargs[field] = value

        with pytest.raises(ValueError, match=match):
            fn(**kwargs)
