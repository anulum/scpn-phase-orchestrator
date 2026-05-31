# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Per-backend parity for upde_run

"""Cross-backend parity tests for :func:`upde_run`.

All backends (Rust, Mojo, Julia, Go, Python) must produce the same
final phase vector for the same input across Euler, RK4, and RK45
integrators. Tolerances:

* Rust / Julia / Go: ``atol = 1e-12`` (shared ``f64`` arithmetic).
* Mojo: ``atol = 1e-6`` (subprocess text round-trip).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import get_type_hints

import numpy as np
import pytest

from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _engine_validation as engine_validation,
)
from scpn_phase_orchestrator.experimental.accelerators.upde._engine_go import (
    upde_run_go,
)
from scpn_phase_orchestrator.experimental.accelerators.upde._engine_julia import (
    upde_run_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.upde._engine_mojo import (
    upde_run_mojo,
)
from scpn_phase_orchestrator.upde import engine as eng_mod
from scpn_phase_orchestrator.upde.engine import (
    AVAILABLE_BACKENDS,
    upde_run,
)
from tests.typing_contracts import assert_precise_ndarray_hint

TWO_PI = 2.0 * np.pi
DirectBackend = Callable[
    [
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
        float,
        float,
        int,
        str,
        int,
        float,
        float,
    ],
    np.ndarray,
]


def test__engine_validation_helper_is_directly_linked_to_backend_tests() -> None:
    assert callable(engine_validation.validate_upde_backend_inputs)
    assert callable(engine_validation.validate_upde_backend_output)


def _force(backend: str) -> str:
    prev = eng_mod.ACTIVE_BACKEND
    eng_mod.ACTIVE_BACKEND = backend
    return prev


def _reset(prev: str) -> None:
    eng_mod.ACTIVE_BACKEND = prev


def _reference(phases, omegas, knm, alpha, **kwargs):
    prev = _force("python")
    try:
        return upde_run(phases, omegas, knm, alpha, **kwargs)
    finally:
        _reset(prev)


def _problem(seed: int, n: int = 5, zeta: float = 0.0, psi: float = 0.0):
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, TWO_PI, size=n)
    omegas = rng.normal(0.0, 0.2, size=n)
    knm = rng.uniform(0.3, 1.0, size=(n, n))
    np.fill_diagonal(knm, 0.0)
    alpha = rng.uniform(-0.15, 0.15, size=(n, n))
    np.fill_diagonal(alpha, 0.0)
    kwargs = {
        "zeta": zeta,
        "psi": psi,
        "dt": 0.01,
        "n_steps": 80,
        "method": "rk4",
        "n_substeps": 1,
        "atol": 1e-6,
        "rtol": 1e-3,
    }
    return phases, omegas, knm, alpha, kwargs


def _direct_payload(seed: int = 13):
    phases, omegas, knm, alpha, kwargs = _problem(seed)
    return (
        phases,
        omegas,
        knm,
        alpha,
        kwargs["zeta"],
        kwargs["psi"],
        kwargs["dt"],
        kwargs["n_steps"],
        kwargs["method"],
        kwargs["n_substeps"],
        kwargs["atol"],
        kwargs["rtol"],
    )


DIRECT_BACKENDS = (upde_run_go, upde_run_julia, upde_run_mojo)


class TestDirectBackendBoundaryContracts:
    @pytest.mark.parametrize("backend", DIRECT_BACKENDS)
    @pytest.mark.parametrize(
        ("index", "replacement"),
        [
            (0, lambda payload: payload[0].reshape(1, -1)),
            (0, lambda payload: payload[0].astype(bool)),
            (0, lambda payload: payload[0].astype(np.complex128) + 1j),
            (0, lambda payload: np.array([np.nan, *payload[0][1:]])),
            (1, lambda payload: payload[1][:-1]),
            (1, lambda payload: payload[1].astype(bool)),
            (1, lambda payload: np.array([np.inf, *payload[1][1:]])),
            (2, lambda payload: payload[2][:-1, :]),
            (2, lambda payload: payload[2].astype(bool)),
            (2, lambda payload: payload[2].astype(np.complex128) + 1j),
            (2, lambda payload: payload[2] + np.eye(payload[2].shape[0])),
            (3, lambda payload: payload[3][:-1, :]),
            (3, lambda payload: payload[3].astype(np.complex128) + 1j),
            (3, lambda payload: np.full_like(payload[3], np.nan)),
            (4, lambda payload: True),
            (4, lambda payload: float("inf")),
            (5, lambda payload: float("nan")),
            (6, lambda payload: 0.0),
            (6, lambda payload: float("nan")),
            (7, lambda payload: True),
            (7, lambda payload: -1),
            (8, lambda payload: 1),
            (8, lambda payload: "bogus"),
            (9, lambda payload: 0),
            (9, lambda payload: True),
            (10, lambda payload: 0.0),
            (10, lambda payload: float("nan")),
            (11, lambda payload: 0.0),
            (11, lambda payload: float("nan")),
        ],
    )
    def test_invalid_inputs_fail_before_optional_runtime_loading(
        self,
        backend: DirectBackend,
        index: int,
        replacement: Callable[[tuple], object],
    ) -> None:
        """Direct Go/Julia/Mojo wrappers enforce the same scientific boundary."""

        payload = list(_direct_payload())
        payload[index] = replacement(tuple(payload))
        with pytest.raises((TypeError, ValueError)):
            backend(*payload)

    @pytest.mark.parametrize("backend", DIRECT_BACKENDS)
    def test_zero_steps_returns_initial_phase_without_optional_runtime(
        self,
        backend: DirectBackend,
    ) -> None:
        payload = list(_direct_payload())
        payload[7] = 0
        result = backend(*payload)
        np.testing.assert_allclose(result, payload[0], atol=0.0)
        assert result is not payload[0]


class TestRustParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "rust" not in AVAILABLE_BACKENDS:
            pytest.skip("Rust backend not built")

    @pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
    @pytest.mark.parametrize("seed", [0, 7])
    def test_matches_python(self, method: str, seed: int) -> None:
        phases, omegas, knm, alpha, kwargs = _problem(seed)
        kwargs["method"] = method
        ref = _reference(phases, omegas, knm, alpha, **kwargs)
        prev = _force("rust")
        try:
            result = upde_run(phases, omegas, knm, alpha, **kwargs)
        finally:
            _reset(prev)
        np.testing.assert_allclose(result, ref, atol=1e-12)

    def test_driver_path(self) -> None:
        phases, omegas, knm, alpha, kwargs = _problem(3, zeta=0.6, psi=0.4)
        ref = _reference(phases, omegas, knm, alpha, **kwargs)
        prev = _force("rust")
        try:
            result = upde_run(phases, omegas, knm, alpha, **kwargs)
        finally:
            _reset(prev)
        np.testing.assert_allclose(result, ref, atol=1e-12)


class TestJuliaParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "julia" not in AVAILABLE_BACKENDS:
            pytest.skip("Julia backend not available")

    @pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
    def test_matches_python(self, method: str) -> None:
        phases, omegas, knm, alpha, kwargs = _problem(0)
        kwargs["method"] = method
        ref = _reference(phases, omegas, knm, alpha, **kwargs)
        prev = _force("julia")
        try:
            result = upde_run(phases, omegas, knm, alpha, **kwargs)
        finally:
            _reset(prev)
        np.testing.assert_allclose(result, ref, atol=1e-12)


class TestGoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "go" not in AVAILABLE_BACKENDS:
            pytest.skip("Go backend not built")

    @pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
    @pytest.mark.parametrize("seed", [0, 42])
    def test_matches_python(self, method: str, seed: int) -> None:
        phases, omegas, knm, alpha, kwargs = _problem(seed)
        kwargs["method"] = method
        ref = _reference(phases, omegas, knm, alpha, **kwargs)
        prev = _force("go")
        try:
            result = upde_run(phases, omegas, knm, alpha, **kwargs)
        finally:
            _reset(prev)
        np.testing.assert_allclose(result, ref, atol=1e-12)

    def test_n_substeps_equivalence(self) -> None:
        phases, omegas, knm, alpha, kwargs = _problem(0)
        kwargs["n_substeps"] = 4
        kwargs["method"] = "rk4"
        ref = _reference(phases, omegas, knm, alpha, **kwargs)
        prev = _force("go")
        try:
            result = upde_run(phases, omegas, knm, alpha, **kwargs)
        finally:
            _reset(prev)
        np.testing.assert_allclose(result, ref, atol=1e-12)


class TestMojoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "mojo" not in AVAILABLE_BACKENDS:
            pytest.skip("Mojo backend not built")

    @pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
    def test_matches_python(self, method: str) -> None:
        phases, omegas, knm, alpha, kwargs = _problem(0)
        kwargs["method"] = method
        ref = _reference(phases, omegas, knm, alpha, **kwargs)
        prev = _force("mojo")
        try:
            result = upde_run(phases, omegas, knm, alpha, **kwargs)
        finally:
            _reset(prev)
        np.testing.assert_allclose(result, ref, atol=1e-6)


class TestCrossBackendConsistency:
    @pytest.mark.skipif(
        len(AVAILABLE_BACKENDS) < 2,
        reason="Only Python fallback available",
    )
    @pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
    def test_all_backends_agree(self, method: str) -> None:
        phases, omegas, knm, alpha, kwargs = _problem(2026, zeta=0.3, psi=0.2)
        kwargs["method"] = method
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
                result = upde_run(phases, omegas, knm, alpha, **kwargs)
            finally:
                _reset(prev)
            np.testing.assert_allclose(
                result,
                ref,
                atol=tolerances[backend],
                err_msg=f"{backend} / {method} diverged from python",
            )


class TestDispatcherResolution:
    def test_active_backend_is_first_available(self) -> None:
        assert AVAILABLE_BACKENDS[0] == eng_mod.ACTIVE_BACKEND

    def test_python_always_available(self) -> None:
        assert "python" in AVAILABLE_BACKENDS

    def test_engine_class_uses_dispatcher(self) -> None:
        """``UPDEEngine.run`` now routes through the module-level
        ``upde_run`` so it benefits from every available backend."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        eng = UPDEEngine(n_oscillators=3, dt=0.01, method="rk4")
        phases, omegas, knm, alpha, _ = _problem(0, n=3)
        out = eng.run(phases, omegas, knm, 0.0, 0.0, alpha, n_steps=10)
        assert out.shape == (3,)
        assert np.all(np.isfinite(out))


class TestBackendTypingContracts:
    @pytest.mark.parametrize(
        ("fn", "label"),
        [
            (upde_run_go, "go"),
            (upde_run_julia, "julia"),
            (upde_run_mojo, "mojo"),
        ],
    )
    def test_backend_annotations_use_float64_ndarray(self, fn, label: str) -> None:
        hints = get_type_hints(fn)
        for name in ("phases", "omegas", "knm", "alpha", "return"):
            text = str(hints[name])
            assert_precise_ndarray_hint(
                hints[name],
                context=f"{label}:{name}",
            )
            assert "numpy.float64" in text, f"{label}:{name} missing float64 annotation"
