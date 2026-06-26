# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — UPDE optional backend loader contracts

"""Optional backend loader contracts for the UPDE run dispatcher."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _engine_go as engine_go_mod,
)
from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _engine_mojo as engine_mojo_mod,
)
from scpn_phase_orchestrator.upde import _run as run_mod

FloatArray = NDArray[np.float64]


def _arrays() -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    phases = np.array([0.1, 0.2], dtype=np.float64)
    omegas = np.array([1.0, 1.2], dtype=np.float64)
    knm = np.array([[0.0, 0.3], [0.3, 0.0]], dtype=np.float64)
    alpha = np.zeros((2, 2), dtype=np.float64)
    return phases, omegas, knm, alpha


def _schedule() -> FloatArray:
    return np.array([[1.0, 1.2], [1.1, 1.3]], dtype=np.float64)


def _backend_identity(phases: FloatArray, *_args: object) -> FloatArray:
    return np.asarray(phases, dtype=np.float64) + 0.25


def _install_fake_spo_kernel(
    monkeypatch: pytest.MonkeyPatch,
    *,
    with_schedule: bool,
) -> None:
    fake_module = ModuleType("spo_kernel")

    class FakeStepper:
        def __init__(
            self,
            n_oscillators: int,
            dt: float,
            method: str,
            *,
            n_substeps: int,
            atol: float,
            rtol: float,
        ) -> None:
            self.n_oscillators = n_oscillators
            self.dt = dt
            self.method = method
            self.n_substeps = n_substeps
            self.atol = atol
            self.rtol = rtol

        def run(
            self,
            phases: FloatArray,
            omegas: FloatArray,
            knm: FloatArray,
            zeta: float,
            psi: float,
            alpha: FloatArray,
            n_steps: int,
        ) -> FloatArray:
            assert self.n_oscillators == phases.size
            assert omegas.shape == phases.shape
            assert knm.shape == (phases.size * phases.size,)
            assert alpha.shape == (phases.size * phases.size,)
            assert n_steps == 3
            return np.asarray(phases, dtype=np.float64) + zeta + psi

    stepper_type: type[object]
    if with_schedule:

        class FakeStepperWithSchedule(FakeStepper):
            def run_omega_schedule(
                self,
                phases: FloatArray,
                omega_schedule: FloatArray,
                knm: FloatArray,
                zeta: float,
                psi: float,
                alpha: FloatArray,
                n_steps: int,
            ) -> FloatArray:
                assert self.n_oscillators == phases.size
                assert omega_schedule.shape == (n_steps * phases.size,)
                assert knm.shape == (phases.size * phases.size,)
                assert alpha.shape == (phases.size * phases.size,)
                return np.asarray(phases, dtype=np.float64) + zeta - psi

        stepper_type = FakeStepperWithSchedule
    else:
        stepper_type = FakeStepper

    fake_module.__dict__["PyUPDEStepper"] = stepper_type
    monkeypatch.setitem(sys.modules, "spo_kernel", fake_module)


class TestOptionalBackendLoaders:
    def test_rust_loader_wraps_py_stepper_run(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_fake_spo_kernel(monkeypatch, with_schedule=True)
        phases, omegas, knm, alpha = _arrays()

        backend = run_mod._load_rust_fn()
        result = backend(
            phases,
            omegas,
            knm,
            alpha,
            0.5,
            0.125,
            0.01,
            3,
            "rk4",
            2,
            1e-8,
            1e-5,
        )

        np.testing.assert_allclose(result, phases + 0.625)

    def test_rust_schedule_loader_wraps_py_stepper_schedule(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_fake_spo_kernel(monkeypatch, with_schedule=True)
        phases, _, knm, alpha = _arrays()

        backend = run_mod._load_rust_schedule_fn()
        result = backend(
            phases,
            _schedule(),
            knm,
            alpha,
            0.5,
            0.125,
            0.01,
            "rk4",
            2,
            1e-8,
            1e-5,
        )

        np.testing.assert_allclose(result, phases + 0.375)

    def test_rust_schedule_loader_fails_when_symbol_is_absent(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_fake_spo_kernel(monkeypatch, with_schedule=False)
        phases, _, knm, alpha = _arrays()

        backend = run_mod._load_rust_schedule_fn()
        with pytest.raises(ImportError, match="run_omega_schedule"):
            backend(
                phases,
                _schedule(),
                knm,
                alpha,
                0.5,
                0.125,
                0.01,
                "rk4",
                2,
                1e-8,
                1e-5,
            )

    def test_mojo_schedule_loader_runs_availability_probe(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: dict[str, int] = {"ensure": 0}

        def _ensure_exe() -> str:
            calls["ensure"] += 1
            return "upde_engine_mojo"

        monkeypatch.setattr(engine_mojo_mod, "_ensure_exe", _ensure_exe)
        monkeypatch.setattr(
            engine_mojo_mod,
            "upde_run_omega_schedule_mojo",
            _backend_identity,
        )

        assert run_mod._load_mojo_schedule_fn() is _backend_identity
        assert calls["ensure"] == 1

    def test_julia_runtime_requires_main_symbol(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            run_mod.importlib,
            "import_module",
            lambda _name: SimpleNamespace(),
        )

        with pytest.raises(ImportError, match="juliacall.Main unavailable"):
            run_mod._require_juliacall_runtime()

    def test_julia_loaders_return_backend_functions(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        real_import = importlib.import_module

        def _import_module(name: str) -> object:
            if name == "juliacall":
                return SimpleNamespace(Main=object())
            return real_import(name)

        monkeypatch.setattr(run_mod.importlib, "import_module", _import_module)

        assert callable(run_mod._load_julia_fn())
        assert callable(run_mod._load_julia_schedule_fn())

    def test_go_schedule_loader_runs_shared_library_probe(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: dict[str, int] = {"load": 0}

        def _load_lib() -> object:
            calls["load"] += 1
            return object()

        monkeypatch.setattr(engine_go_mod, "_load_lib", _load_lib)
        monkeypatch.setattr(
            engine_go_mod,
            "upde_run_omega_schedule_go",
            _backend_identity,
        )

        assert run_mod._load_go_schedule_fn() is _backend_identity
        assert calls["load"] == 1
