# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — UPDE optional backend loader contracts

"""Optional backend loader contracts for the UPDE run dispatcher."""

from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pytest
from numpy.typing import NDArray

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


def _install_old_spo_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a ``spo_kernel`` whose stepper predates ``run_omega_schedule``.

    Such a build cannot be produced from the current sources, so this
    negative contract keeps a stand-in for the old compiled module.
    """
    old_module = ModuleType("spo_kernel")

    class OldStepper:
        """Stepper of an old build: constructible, without schedule support."""

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

    old_module.__dict__["PyUPDEStepper"] = OldStepper
    monkeypatch.setitem(sys.modules, "spo_kernel", old_module)


class TestOptionalBackendLoaders:
    def test_rust_schedule_loader_fails_when_symbol_is_absent(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_old_spo_kernel(monkeypatch)
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

    def test_named_rust_schedule_on_an_old_kernel_is_refused(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A named owner that fails for want of a symbol raises; ``auto`` falls back."""
        _install_old_spo_kernel(monkeypatch)
        monkeypatch.setattr(run_mod, "_SCHEDULE_BACKEND_CACHE", {})
        monkeypatch.setattr(run_mod, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(run_mod, "AVAILABLE_BACKENDS", ["rust", "python"])
        phases, _, knm, alpha = _arrays()
        arguments = (phases, _schedule(), knm, alpha, 0.5, 0.125, 0.01)

        with pytest.raises(RuntimeError, match="UPDE backend 'rust' failed") as info:
            run_mod.upde_run_omega_schedule(*arguments, backend="rust")
        assert "run_omega_schedule" in str(info.value.__cause__)

        reference = run_mod.upde_run_omega_schedule(*arguments, backend="python")
        np.testing.assert_array_equal(
            run_mod.upde_run_omega_schedule(*arguments), reference
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

    def test_julia_loaders_return_backend_functions(self) -> None:
        """With a working Julia install the loaders return real callables.

        The missing-``Main`` and broken-set-up cases belong to the shared probe
        ``upde._julia_runtime.require_juliacall_main``, which ``_run`` delegates
        to; they are exercised in ``test_julia_runtime_broken_setup.py``.
        """
        pytest.importorskip("juliacall")
        assert callable(run_mod._load_julia_fn())
        assert callable(run_mod._load_julia_schedule_fn())
