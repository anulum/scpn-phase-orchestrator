# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — UPDE schedule dispatch contracts

"""Schedule-dispatch and validation contracts for the UPDE run dispatcher."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.upde import _run as run_mod

FloatArray = NDArray[np.float64]
BackendFn = Callable[..., FloatArray]


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


class TestScheduleDispatch:
    def test_schedule_backend_cache_loads_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: dict[str, int] = {"go": 0}

        def _ok_go() -> BackendFn:
            calls["go"] += 1
            return _backend_identity

        monkeypatch.setattr(run_mod, "_SCHEDULE_BACKEND_CACHE", {})
        monkeypatch.setattr(run_mod, "_SCHEDULE_LOADERS", {"go": _ok_go})

        assert run_mod._load_schedule_backend("go") is _backend_identity
        assert run_mod._load_schedule_backend("go") is _backend_identity
        assert calls["go"] == 1

    def test_dispatch_schedule_skips_unsupported_and_failed_backends(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: dict[str, int] = {"rust": 0, "go": 0}

        def _fail_rust() -> BackendFn:
            calls["rust"] += 1
            raise AttributeError("old rust stepper")

        def _ok_go() -> BackendFn:
            calls["go"] += 1
            return _backend_identity

        monkeypatch.setattr(run_mod, "_SCHEDULE_BACKEND_CACHE", {})
        monkeypatch.setattr(run_mod, "ACTIVE_BACKEND", "webgpu")
        monkeypatch.setattr(run_mod, "AVAILABLE_BACKENDS", ["rust", "go", "python"])
        monkeypatch.setattr(
            run_mod,
            "_SCHEDULE_LOADERS",
            {"rust": _fail_rust, "go": _ok_go},
        )

        assert run_mod._dispatch_schedule() is _backend_identity
        assert calls == {"rust": 1, "go": 1}

    def test_dispatch_schedule_returns_none_when_all_loaders_fail(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _fail() -> BackendFn:
            raise OSError("backend unavailable")

        monkeypatch.setattr(run_mod, "_SCHEDULE_BACKEND_CACHE", {})
        monkeypatch.setattr(run_mod, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(run_mod, "AVAILABLE_BACKENDS", ["rust", "python"])
        monkeypatch.setattr(run_mod, "_SCHEDULE_LOADERS", {"rust": _fail})

        assert run_mod._dispatch_schedule() is None


class TestRunEntryPoints:
    def test_upde_run_uses_python_reference_when_dispatch_has_no_backend(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        phases, omegas, knm, alpha = _arrays()
        monkeypatch.setattr(run_mod, "_dispatch", lambda: None)

        result = run_mod.upde_run(
            phases,
            omegas,
            knm,
            alpha,
            0.0,
            0.0,
            0.01,
            0,
        )

        np.testing.assert_allclose(result, phases)
        assert result is not phases

    def test_upde_run_rejects_self_coupling_before_backend_dispatch(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        phases, omegas, knm, alpha = _arrays()
        knm[0, 0] = 0.5
        monkeypatch.setattr(run_mod, "_dispatch", lambda: _backend_identity)

        with pytest.raises(ValueError, match="self-coupling diagonal"):
            run_mod.upde_run(phases, omegas, knm, alpha, 0.0, 0.0, 0.01, 1)

    def test_schedule_run_uses_python_reference_without_schedule_backend(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        phases, _, knm, alpha = _arrays()
        schedule = _schedule()
        monkeypatch.setattr(run_mod, "_dispatch_schedule", lambda: None)

        result = run_mod.upde_run_omega_schedule(
            phases,
            schedule,
            knm,
            alpha,
            0.0,
            0.0,
            0.01,
        )

        expected = run_mod.upde_run(
            phases,
            schedule[0],
            knm,
            alpha,
            0.0,
            0.0,
            0.01,
            1,
        )
        expected = run_mod.upde_run(
            expected,
            schedule[1],
            knm,
            alpha,
            0.0,
            0.0,
            0.01,
            1,
        )
        np.testing.assert_allclose(result, expected)

    def test_schedule_run_uses_loaded_backend(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        phases, _, knm, alpha = _arrays()
        monkeypatch.setattr(run_mod, "_dispatch_schedule", lambda: _backend_identity)

        result = run_mod.upde_run_omega_schedule(
            phases,
            _schedule(),
            knm,
            alpha,
            0.0,
            0.0,
            0.01,
        )

        np.testing.assert_allclose(result, phases + 0.25)

    @pytest.mark.parametrize(
        "exc",
        [AttributeError("old backend"), ImportError("gone")],
    )
    def test_schedule_run_falls_back_when_loaded_backend_lacks_symbol(
        self,
        monkeypatch: pytest.MonkeyPatch,
        exc: Exception,
    ) -> None:
        phases, _, knm, alpha = _arrays()

        def _raise(*_args: object) -> FloatArray:
            raise exc

        monkeypatch.setattr(run_mod, "_dispatch_schedule", lambda: _raise)

        result = run_mod.upde_run_omega_schedule(
            phases,
            _schedule(),
            knm,
            alpha,
            0.0,
            0.0,
            0.01,
        )

        assert result.shape == phases.shape
        assert np.all(np.isfinite(result))

    @pytest.mark.parametrize(
        ("schedule", "match"),
        [
            (np.array([[True, False]], dtype=np.bool_), "real-valued"),
            (np.array([[1.0 + 1j, 2.0]], dtype=np.complex128), "real-valued"),
            (np.array([1.0, 2.0], dtype=np.float64), "two-dimensional"),
            (np.empty((0, 2), dtype=np.float64), "at least one step"),
            (np.array([[1.0, 2.0, 3.0]], dtype=np.float64), "column count"),
            (np.array([[1.0, np.inf]], dtype=np.float64), "NaN/Inf"),
        ],
    )
    def test_schedule_run_rejects_malformed_schedule_before_dispatch(
        self,
        monkeypatch: pytest.MonkeyPatch,
        schedule: NDArray[Any],
        match: str,
    ) -> None:
        phases, _, knm, alpha = _arrays()
        monkeypatch.setattr(run_mod, "_dispatch_schedule", lambda: _backend_identity)

        with pytest.raises(ValueError, match=match):
            run_mod.upde_run_omega_schedule(
                phases,
                schedule,
                knm,
                alpha,
                0.0,
                0.0,
                0.01,
            )
