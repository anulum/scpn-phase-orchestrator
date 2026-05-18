# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Per-backend parity for swarmalator stepper

"""Cross-backend parity for :class:`SwarmalatorEngine`.step.

All four non-Python backends must match the Python reference on
the same input within 1e-12 (Rust / Julia / Go) or 1e-9 (Mojo).
The Python reference was aligned to Rust's canonical
``b / (dist · d² + eps)`` repulse formula during migration (the
pre-migration Python used ``dist³`` which drifted by O(1e-4)).
"""

from __future__ import annotations

import sys
import types
from typing import get_type_hints

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _swarmalator_go,
    _swarmalator_julia,
    _swarmalator_mojo,
)
from scpn_phase_orchestrator.upde import swarmalator as sw_mod
from scpn_phase_orchestrator.upde.swarmalator import (
    AVAILABLE_BACKENDS,
    SwarmalatorEngine,
)
from tests.typing_contracts import assert_precise_ndarray_hint

swarmalator_step_go = _swarmalator_go.swarmalator_step_go
swarmalator_step_julia = _swarmalator_julia.swarmalator_step_julia
swarmalator_step_mojo = _swarmalator_mojo.swarmalator_step_mojo

TWO_PI = 2.0 * np.pi


def _force(backend: str) -> str:
    prev = sw_mod.ACTIVE_BACKEND
    sw_mod.ACTIVE_BACKEND = backend
    return prev


def _reset(prev: str) -> None:
    sw_mod.ACTIVE_BACKEND = prev


def _problem(seed: int, n: int = 16, dim: int = 2):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1, 1, (n, dim))
    phases = rng.uniform(0, TWO_PI, n)
    omegas = rng.normal(0.5, 0.2, n)
    return pos, phases, omegas


def _reference_step(pos, phases, omegas, n: int, dim: int):
    prev = _force("python")
    try:
        eng = SwarmalatorEngine(n, dim, 0.01)
        return eng.step(pos, phases, omegas, 1.0, 1.0, 0.8, 1.2)
    finally:
        _reset(prev)


class TestRustParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "rust" not in AVAILABLE_BACKENDS:
            pytest.skip("Rust backend not built")

    @given(
        n=st.integers(min_value=2, max_value=24),
        dim=st.integers(min_value=1, max_value=3),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=8,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_step(self, n: int, dim: int, seed: int) -> None:
        pos, phases, omegas = _problem(seed, n, dim)
        ref_p, ref_ph = _reference_step(pos, phases, omegas, n, dim)
        prev = _force("rust")
        try:
            eng = SwarmalatorEngine(n, dim, 0.01)
            p, ph = eng.step(pos, phases, omegas, 1.0, 1.0, 0.8, 1.2)
        finally:
            _reset(prev)
        np.testing.assert_allclose(p, ref_p, atol=1e-12)
        np.testing.assert_allclose(ph, ref_ph, atol=1e-12)


class TestJuliaParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "julia" not in AVAILABLE_BACKENDS:
            pytest.skip("Julia backend not available")

    @pytest.mark.parametrize("seed", [0, 42])
    def test_step(self, seed: int) -> None:
        pos, phases, omegas = _problem(seed)
        ref_p, ref_ph = _reference_step(pos, phases, omegas, 16, 2)
        prev = _force("julia")
        try:
            eng = SwarmalatorEngine(16, 2, 0.01)
            p, ph = eng.step(pos, phases, omegas, 1.0, 1.0, 0.8, 1.2)
        finally:
            _reset(prev)
        np.testing.assert_allclose(p, ref_p, atol=1e-12)
        np.testing.assert_allclose(ph, ref_ph, atol=1e-12)


class TestGoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "go" not in AVAILABLE_BACKENDS:
            pytest.skip("Go backend not built")

    @given(
        n=st.integers(min_value=2, max_value=24),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=6,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_step(self, n: int, seed: int) -> None:
        pos, phases, omegas = _problem(seed, n, 2)
        ref_p, ref_ph = _reference_step(pos, phases, omegas, n, 2)
        prev = _force("go")
        try:
            eng = SwarmalatorEngine(n, 2, 0.01)
            p, ph = eng.step(pos, phases, omegas, 1.0, 1.0, 0.8, 1.2)
        finally:
            _reset(prev)
        np.testing.assert_allclose(p, ref_p, atol=1e-12)
        np.testing.assert_allclose(ph, ref_ph, atol=1e-12)


class TestMojoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "mojo" not in AVAILABLE_BACKENDS:
            pytest.skip("Mojo backend not built")

    @pytest.mark.parametrize("seed", [0, 77])
    def test_step(self, seed: int) -> None:
        pos, phases, omegas = _problem(seed)
        ref_p, ref_ph = _reference_step(pos, phases, omegas, 16, 2)
        prev = _force("mojo")
        try:
            eng = SwarmalatorEngine(16, 2, 0.01)
            p, ph = eng.step(pos, phases, omegas, 1.0, 1.0, 0.8, 1.2)
        finally:
            _reset(prev)
        np.testing.assert_allclose(p, ref_p, atol=1e-9)
        np.testing.assert_allclose(ph, ref_ph, atol=1e-9)


class TestCrossBackendConsistency:
    @pytest.mark.skipif(
        len(AVAILABLE_BACKENDS) < 2,
        reason="Only Python fallback available",
    )
    def test_all_backends_agree(self) -> None:
        pos, phases, omegas = _problem(2026, 20, 2)
        ref_p, ref_ph = _reference_step(pos, phases, omegas, 20, 2)
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
                eng = SwarmalatorEngine(20, 2, 0.01)
                p, ph = eng.step(pos, phases, omegas, 1.0, 1.0, 0.8, 1.2)
            finally:
                _reset(prev)
            atol = tolerances[backend]
            np.testing.assert_allclose(p, ref_p, atol=atol)
            np.testing.assert_allclose(ph, ref_ph, atol=atol)


class TestBackendTypingContracts:
    @pytest.mark.parametrize(
        ("fn", "label"),
        [
            (swarmalator_step_go, "go"),
            (swarmalator_step_julia, "julia"),
            (swarmalator_step_mojo, "mojo"),
        ],
    )
    def test_backend_annotations_use_float64_ndarray(self, fn, label: str) -> None:
        hints = get_type_hints(fn)
        for name in ("pos", "phases", "omegas", "return"):
            text = str(hints[name])
            assert_precise_ndarray_hint(
                hints[name],
                context=f"{label}:{name}",
            )
            assert "numpy.float64" in text, f"{label}:{name} missing float64 annotation"


class TestBackendLoaderContracts:
    def test_rust_loader_reshapes_flat_kernel_positions(self, monkeypatch) -> None:
        calls = {}

        class PySwarmalatorStepper:
            def __init__(self, n: int, dim: int, dt: float) -> None:
                calls["init"] = (n, dim, dt)

            def step(self, pos, phases, omegas, a, b, j, k):
                calls["contiguous"] = (
                    pos.flags.c_contiguous,
                    phases.flags.c_contiguous,
                    omegas.flags.c_contiguous,
                )
                return pos + a - b + j - k, phases + omegas

        fake_spo = types.ModuleType("spo_kernel")
        fake_spo.PySwarmalatorStepper = PySwarmalatorStepper
        monkeypatch.setitem(sys.modules, "spo_kernel", fake_spo)

        fn = sw_mod._load_rust_fn()
        pos = np.array([[0.0, 0.5], [1.0, -0.5]], dtype=np.float64)
        phases = np.array([0.2, 0.4], dtype=np.float64)
        omegas = np.array([0.1, -0.2], dtype=np.float64)

        got_pos, got_phases = fn(pos, phases, omegas, 2, 2, 1.0, 0.5, 0.25, 0.75, 0.01)

        np.testing.assert_allclose(got_pos, pos.ravel().reshape(2, 2))
        np.testing.assert_allclose(got_phases, phases + omegas)
        assert got_pos.dtype == np.float64
        assert got_phases.dtype == np.float64
        assert calls == {"init": (2, 2, 0.01), "contiguous": (True, True, True)}

    def test_optional_backend_loaders_return_callable_numeric_kernels(
        self, monkeypatch
    ) -> None:
        def install_backend(
            module_name: str, function_name: str, offset: float
        ) -> None:
            module = types.ModuleType(module_name)

            def _ensure_exe() -> None:
                module.loaded = True

            def _load_lib() -> None:
                module.loaded = True

            def kernel(pos, phases, omegas, n, dim, a, b, j, k, dt):
                return pos + offset + dt, (phases + omegas * dt + offset) % TWO_PI

            module.loaded = False
            module._ensure_exe = _ensure_exe
            module._load_lib = _load_lib
            setattr(module, function_name, kernel)
            monkeypatch.setitem(sys.modules, module_name, module)

        monkeypatch.setitem(sys.modules, "juliacall", types.ModuleType("juliacall"))
        install_backend(
            "scpn_phase_orchestrator.experimental.accelerators.upde._swarmalator_mojo",
            "swarmalator_step_mojo",
            0.10,
        )
        install_backend(
            "scpn_phase_orchestrator.experimental.accelerators.upde._swarmalator_julia",
            "swarmalator_step_julia",
            0.20,
        )
        install_backend(
            "scpn_phase_orchestrator.experimental.accelerators.upde._swarmalator_go",
            "swarmalator_step_go",
            0.30,
        )

        pos = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
        phases = np.array([0.5, 1.5], dtype=np.float64)
        omegas = np.array([0.2, -0.1], dtype=np.float64)
        args = (pos, phases, omegas, 2, 2, 1.0, 1.0, 0.8, 1.2, 0.01)

        for loader, offset in (
            (sw_mod._load_mojo_fn, 0.10),
            (sw_mod._load_julia_fn, 0.20),
            (sw_mod._load_go_fn, 0.30),
        ):
            got_pos, got_phases = loader()(*args)
            np.testing.assert_allclose(got_pos, pos + offset + 0.01)
            np.testing.assert_allclose(
                got_phases,
                (phases + omegas * 0.01 + offset) % TWO_PI,
            )
