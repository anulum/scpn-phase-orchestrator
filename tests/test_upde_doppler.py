# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Šotek. All rights reserved.
# © Code 2020-2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Doppler UPDE tests

"""Behavioural tests for the PHA-C.2 Doppler UPDE contract."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import scpn_phase_orchestrator.upde.doppler as doppler_module
import scpn_phase_orchestrator.upde.engine as engine_module
from benchmarks.upde_doppler_benchmark import benchmark_upde_doppler_polyglot_gate
from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _doppler_go as doppler_go_module,
)
from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _doppler_julia as doppler_julia_module,
)
from scpn_phase_orchestrator.experimental.accelerators.upde import (
    _doppler_mojo as doppler_mojo_module,
)
from scpn_phase_orchestrator.upde import DopplerEngine as ExportedDopplerEngine
from scpn_phase_orchestrator.upde._ref_kernel import upde_run_omega_schedule_python
from scpn_phase_orchestrator.upde.doppler import (
    DopplerEngine,
    doppler_run,
    doppler_run_python,
    doppler_term,
)
from scpn_phase_orchestrator.upde.engine import UPDEEngine

TWO_PI = 2.0 * np.pi


def _force_python(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_module, "ACTIVE_BACKEND", "python")
    monkeypatch.setattr(
        doppler_module, "_backend_map", lambda: {"python": doppler_run_python}
    )


def _two_body_knm(k: float = 1.0) -> np.ndarray:
    return np.array([[0.0, k], [k, 0.0]], dtype=np.float64)


def _zero_alpha(n: int = 2) -> np.ndarray:
    return np.zeros((n, n), dtype=np.float64)


def _direct_backend_args() -> tuple[object, ...]:
    """Return a valid direct Doppler backend call payload."""
    return (
        np.zeros(2, dtype=np.float64),
        np.zeros((1, 2), dtype=np.float64),
        _two_body_knm(),
        _zero_alpha(),
        np.zeros((1, 2), dtype=np.float64),
        1.0,
        1.0e-9,
        0.0,
        0.0,
        0.01,
        "rk4",
        1,
        1.0e-6,
        1.0e-3,
    )


class _CorruptDopplerGoSymbol:
    """Fake Go symbol that writes a corrupt phase into the output buffer."""

    restype: object
    argtypes: object

    def __init__(self, value: float) -> None:
        self._value = value

    def __call__(self, phases_ptr: object, *_args: object) -> int:
        cast(Any, phases_ptr)[0] = self._value
        return 0


class _CorruptDopplerGoLibrary:
    """Fake Go library exposing the Doppler schedule symbol."""

    def __init__(self, value: float) -> None:
        self.UPDERunDopplerSchedule = _CorruptDopplerGoSymbol(value)


class _FailingDopplerGoSymbol:
    """Fake Go symbol returning a non-zero Doppler backend status."""

    restype: object
    argtypes: object

    def __call__(self, *_args: object) -> int:
        return 7


class _FailingDopplerGoLibrary:
    """Fake Go library exposing a failing Doppler schedule symbol."""

    def __init__(self) -> None:
        self.UPDERunDopplerSchedule = _FailingDopplerGoSymbol()


def _wrapped_abs_delta(phases: np.ndarray) -> float:
    delta = (float(phases[0] - phases[1]) + np.pi) % TWO_PI - np.pi
    return abs(delta)


def test_public_lazy_export_exposes_doppler_engine() -> None:
    assert ExportedDopplerEngine is DopplerEngine


def test_doppler_adapter_modules_import_by_full_path() -> None:
    assert (
        importlib.import_module(
            "scpn_phase_orchestrator.experimental.accelerators.upde._doppler_go"
        )
        is doppler_go_module
    )
    assert (
        importlib.import_module(
            "scpn_phase_orchestrator.experimental.accelerators.upde._doppler_julia"
        )
        is doppler_julia_module
    )
    assert (
        importlib.import_module(
            "scpn_phase_orchestrator.experimental.accelerators.upde._doppler_mojo"
        )
        is doppler_mojo_module
    )


def test_doppler_term_matches_two_body_velocity_formula() -> None:
    term = doppler_term(
        np.array([300.0, -300.0], dtype=np.float64),
        _two_body_knm(),
        doppler_strength=1.0,
        doppler_epsilon=1.0e-12,
    )

    np.testing.assert_allclose(term, np.array([2.0, -2.0]), atol=1.0e-12)


def test_vector_velocity_uses_magnitude_or_axis_projection() -> None:
    velocities = np.array([[3.0, 4.0], [-3.0, 4.0]], dtype=np.float64)

    magnitude_term = doppler_term(velocities, _two_body_knm(), doppler_epsilon=1.0e-12)
    projected_term = doppler_term(
        velocities,
        _two_body_knm(),
        doppler_epsilon=1.0e-12,
        velocity_axis=np.array([1.0, 0.0], dtype=np.float64),
    )

    np.testing.assert_allclose(magnitude_term, np.zeros(2), atol=1.0e-12)
    np.testing.assert_allclose(projected_term, np.array([2.0, -2.0]), atol=1.0e-12)


def test_doppler_engine_step_cancels_counterpropagating_frequency_shift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_python(monkeypatch)
    knm = _two_body_knm()
    velocities = np.array([300.0, -300.0], dtype=np.float64)
    omega = -doppler_term(velocities, knm, doppler_epsilon=1.0e-12)
    engine = DopplerEngine(
        2,
        omega=omega,
        k_nm=knm,
        alpha=0.0,
        dt=0.1,
        velocities=velocities,
        doppler_epsilon=1.0e-12,
        solver="euler",
    )
    engine._rust = None

    out = engine.step()

    np.testing.assert_allclose(out, np.zeros(2), atol=1.0e-12)
    np.testing.assert_allclose(engine.doppler_term, -omega, atol=1.0e-12)
    np.testing.assert_allclose(engine.velocity_current, velocities, atol=1.0e-12)
    assert engine.time == pytest.approx(0.1)


def test_zero_doppler_strength_reduces_to_standard_upde_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_python(monkeypatch)
    phases = np.array([0.2, 0.7, 1.1], dtype=np.float64)
    omega_schedule = np.array([[0.3, 0.1, 0.2], [0.4, -0.2, 0.15]], dtype=np.float64)
    velocity_schedule = np.array(
        [[10.0, -5.0, 2.0], [9.0, -4.5, 2.5]], dtype=np.float64
    )
    knm = np.array(
        [[0.0, 0.3, 0.1], [0.3, 0.0, 0.2], [0.1, 0.2, 0.0]], dtype=np.float64
    )
    alpha = np.zeros((3, 3), dtype=np.float64)

    got = doppler_run(
        phases,
        omega_schedule,
        knm,
        alpha,
        velocity_schedule,
        doppler_strength=0.0,
        doppler_epsilon=1.0e-9,
        dt=0.01,
        method="rk4",
        backend="python",
    )
    expected = upde_run_omega_schedule_python(
        phases,
        omega_schedule,
        knm,
        alpha,
        0.0,
        0.0,
        0.01,
        "rk4",
        1,
        1.0e-6,
        1.0e-3,
    )

    np.testing.assert_allclose(got, expected, atol=1.0e-12)


def test_mach_one_counterpropagating_lock_requires_doppler_correction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_python(monkeypatch)
    knm = _two_body_knm(k=5.0)
    alpha = _zero_alpha()
    velocities = np.array([300.0, -300.0], dtype=np.float64)
    omega = -doppler_term(velocities, knm, doppler_epsilon=1.0e-12)
    phases = np.array([0.5, 0.0], dtype=np.float64)
    dt = 1.0e-3
    n_steps = 2_000

    baseline = UPDEEngine(2, dt=dt, method="euler", omega=omega)
    baseline._rust = None
    without_doppler = baseline.run(phases, knm=knm, alpha=alpha, n_steps=n_steps)

    corrected = DopplerEngine(
        2,
        omega=omega,
        k_nm=knm,
        alpha=alpha,
        dt=dt,
        velocities=velocities,
        doppler_epsilon=1.0e-12,
        solver="euler",
        phases=phases,
    )
    corrected._rust = None
    with_doppler = corrected.run(n_steps=n_steps)

    assert _wrapped_abs_delta(without_doppler) > 0.1
    assert _wrapped_abs_delta(with_doppler) < 0.01


def test_callable_velocity_schedule_updates_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_python(monkeypatch)
    knm = _two_body_knm()

    def velocities(t: float) -> np.ndarray:
        return np.array([300.0 - t, -300.0 + t], dtype=np.float64)

    engine = DopplerEngine(
        2,
        omega=np.zeros(2),
        k_nm=knm,
        alpha=0.0,
        dt=0.01,
        velocities=velocities,
        doppler_epsilon=1.0e-9,
        solver="euler",
    )
    engine._rust = None

    engine.run(n_steps=3)

    np.testing.assert_allclose(engine.velocity_current, velocities(0.02), atol=1.0e-12)
    assert engine.time == pytest.approx(0.03)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"velocities": np.array([True, False])}, "velocities"),
        ({"velocities": np.array([1.0, 2.0]), "doppler_epsilon": 0.0}, "epsilon"),
        ({"velocities": np.array([1.0, 2.0]), "k_nm": np.eye(2)}, "diagonal"),
    ],
)
def test_doppler_engine_invalid_boundaries_fail_closed(
    kwargs: dict[str, object],
    match: str,
) -> None:
    params: dict[str, object] = {
        "n": 2,
        "omega": np.zeros(2),
        "k_nm": _two_body_knm(),
        "alpha": 0.0,
        "dt": 0.01,
        "velocities": np.array([1.0, -1.0]),
    }
    params.update(kwargs)

    with pytest.raises(ValueError, match=match):
        DopplerEngine(**params)


def test_doppler_polyglot_benchmark_reports_available_language_slots() -> None:
    out = benchmark_upde_doppler_polyglot_gate(n=4, n_steps=3, calls=1, seed=11)

    assert out["suite"] == "upde_doppler_polyglot_gate"
    assert out["acceptance_passed"] == 1
    assert out["all_available_passed"] == 1
    assert out["parity_pass_count"] >= 1


def test_doppler_go_adapter_validates_before_loading_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        doppler_go_module,
        "_load_lib",
        lambda: (_ for _ in ()).throw(AssertionError("runtime loaded")),
    )

    with pytest.raises(ValueError, match="diagonal"):
        doppler_go_module.doppler_run_go(
            np.zeros(2),
            np.zeros((1, 2)),
            np.eye(2),
            np.zeros((2, 2)),
            np.zeros((1, 2)),
            1.0,
            1.0e-9,
            0.0,
            0.0,
            0.01,
            "rk4",
            1,
            1.0e-6,
            1.0e-3,
        )


def test_doppler_julia_and_mojo_adapters_validate_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        doppler_julia_module,
        "_ensure",
        lambda: (_ for _ in ()).throw(AssertionError("julia loaded")),
    )
    monkeypatch.setattr(
        doppler_mojo_module,
        "_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("mojo loaded")),
    )
    args = (
        np.zeros(2),
        np.zeros((1, 2)),
        _two_body_knm(),
        np.zeros((2, 2)),
        np.zeros((1, 3)),
        1.0,
        1.0e-9,
        0.0,
        0.0,
        0.01,
        "rk4",
        1,
        1.0e-6,
        1.0e-3,
    )

    with pytest.raises(ValueError, match="velocity_schedule"):
        doppler_julia_module.doppler_run_julia(*args)
    with pytest.raises(ValueError, match="velocity_schedule"):
        doppler_mojo_module.doppler_run_mojo(*args)


@pytest.mark.parametrize(
    ("bad_value", "match"),
    [
        (float("nan"), "contains NaN/Inf"),
        (TWO_PI, r"phases must be in \[0, 2\*pi\)"),
    ],
)
def test_doppler_go_adapter_rejects_corrupt_backend_output(
    monkeypatch: pytest.MonkeyPatch,
    bad_value: float,
    match: str,
) -> None:
    """The direct Go adapter must validate the untrusted backend output."""
    monkeypatch.setattr(
        doppler_go_module,
        "_load_lib",
        lambda: _CorruptDopplerGoLibrary(bad_value),
    )

    with pytest.raises(ValueError, match=match):
        doppler_go_module.doppler_run_go(*_direct_backend_args())


def test_doppler_go_adapter_rejects_missing_symbol_and_nonzero_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The direct Go adapter must surface missing symbols and failing rc values."""
    monkeypatch.setattr(
        doppler_go_module,
        "_load_lib",
        lambda: SimpleNamespace(),
    )
    with pytest.raises(ImportError, match="UPDERunDopplerSchedule"):
        doppler_go_module.doppler_run_go(*_direct_backend_args())

    monkeypatch.setattr(
        doppler_go_module,
        "_load_lib",
        lambda: _FailingDopplerGoLibrary(),
    )
    with pytest.raises(ValueError, match="rc=7"):
        doppler_go_module.doppler_run_go(*_direct_backend_args())


def test_doppler_julia_adapter_rejects_corrupt_backend_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The direct Julia adapter must reject phases outside the principal branch."""
    monkeypatch.setattr(
        doppler_julia_module,
        "_ensure",
        lambda: SimpleNamespace(
            upde_run_doppler_schedule=lambda *_args: np.array(
                [TWO_PI, 0.0], dtype=np.float64
            )
        ),
    )

    with pytest.raises(ValueError, match=r"phases must be in \[0, 2\*pi\)"):
        doppler_julia_module.doppler_run_julia(*_direct_backend_args())


def test_doppler_julia_adapter_rejects_missing_schedule_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The direct Julia adapter must fail closed on a missing entrypoint."""
    monkeypatch.setattr(doppler_julia_module, "_ensure", lambda: SimpleNamespace())

    with pytest.raises(ImportError, match="upde_run_doppler_schedule"):
        doppler_julia_module.doppler_run_julia(*_direct_backend_args())


def test_doppler_mojo_adapter_rejects_corrupt_backend_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The direct Mojo adapter must validate parsed stdout values."""
    monkeypatch.setattr(
        doppler_mojo_module,
        "_run",
        lambda *_args, **_kwargs: [float("inf"), 0.0],
    )

    with pytest.raises(ValueError, match="contains NaN/Inf"):
        doppler_mojo_module.doppler_run_mojo(*_direct_backend_args())
