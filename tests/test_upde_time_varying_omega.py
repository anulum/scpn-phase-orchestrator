# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — UPDE time-varying omega tests

from __future__ import annotations

import numpy as np
import pytest

import scpn_phase_orchestrator.upde._run as run_module
import scpn_phase_orchestrator.upde.engine as engine_module
from scpn_phase_orchestrator.upde.engine import (
    UPDEEngine,
    upde_run,
    upde_run_omega_schedule,
)


def _force_python(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_module, "ACTIVE_BACKEND", "python")
    monkeypatch.setattr(run_module, "ACTIVE_BACKEND", "python")


def _zero_problem(n: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phases = np.zeros(n, dtype=np.float64)
    knm = np.zeros((n, n), dtype=np.float64)
    alpha = np.zeros((n, n), dtype=np.float64)
    return phases, knm, alpha


def test_constructor_fixed_omega_drives_step_without_explicit_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_python(monkeypatch)
    phases, knm, alpha = _zero_problem()
    engine = UPDEEngine(2, dt=0.1, omega=np.array([1.0, 2.0]))

    out = engine.step(phases, knm=knm, alpha=alpha)

    np.testing.assert_allclose(out, np.array([0.1, 0.2]), atol=1.0e-12)
    np.testing.assert_allclose(engine.omega_current, np.array([1.0, 2.0]))
    assert engine.time == pytest.approx(0.1)


def test_callable_omega_run_matches_linear_analytic_solution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_python(monkeypatch)
    phases, knm, alpha = _zero_problem()
    dt = 1.0e-6
    n_steps = 20
    omega0 = np.array([1.0, 2.0], dtype=np.float64)
    slope = np.array([-0.05, 0.02], dtype=np.float64)

    def omega(t: float) -> np.ndarray:
        return omega0 + slope * t

    engine = UPDEEngine(2, dt=dt, method="euler", omega=omega)
    out = engine.run(phases, knm=knm, alpha=alpha, n_steps=n_steps)

    total_time = n_steps * dt
    expected = omega0 * total_time + 0.5 * slope * total_time * total_time
    np.testing.assert_allclose(out, expected, rtol=1.0e-6, atol=1.0e-12)
    np.testing.assert_allclose(engine.omega_current, omega((n_steps - 1) * dt))
    assert engine.time == pytest.approx(total_time)


def test_explicit_step_omega_overrides_configured_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_python(monkeypatch)
    phases, knm, alpha = _zero_problem()
    engine = UPDEEngine(2, dt=0.1, omega=lambda _t: np.zeros(2))

    out = engine.step(phases, np.array([3.0, 4.0]), knm, 0.0, 0.0, alpha)

    np.testing.assert_allclose(out, np.array([0.3, 0.4]), atol=1.0e-12)
    np.testing.assert_allclose(engine.omega_current, np.array([3.0, 4.0]))


def test_schedule_dispatch_matches_manual_per_step_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_python(monkeypatch)
    phases = np.array([0.1, 0.7, 1.4], dtype=np.float64)
    schedule = np.array(
        [[1.0, 1.2, 0.9], [0.8, 1.1, 1.4], [1.3, 0.7, 1.0]],
        dtype=np.float64,
    )
    knm = np.array(
        [[0.0, 0.2, 0.1], [0.2, 0.0, 0.15], [0.1, 0.15, 0.0]],
        dtype=np.float64,
    )
    alpha = np.zeros((3, 3), dtype=np.float64)

    manual = phases.copy()
    for row in schedule:
        manual = upde_run(manual, row, knm, alpha, 0.0, 0.0, 0.01, 1, "rk4")
    scheduled = upde_run_omega_schedule(
        phases,
        schedule,
        knm,
        alpha,
        0.0,
        0.0,
        0.01,
        "rk4",
    )

    np.testing.assert_allclose(scheduled, manual, atol=1.0e-12)


@pytest.mark.parametrize(
    "bad_omega",
    [
        np.array([True, False]),
        np.array([1.0, np.nan]),
        np.array([[1.0, 2.0]]),
        np.array([1.0 + 0.0j, 2.0 + 0.0j]),
    ],
)
def test_step_rejects_invalid_resolved_omega(
    monkeypatch: pytest.MonkeyPatch,
    bad_omega: np.ndarray,
) -> None:
    _force_python(monkeypatch)
    phases, knm, alpha = _zero_problem()
    engine = UPDEEngine(2, dt=0.1)

    with pytest.raises((TypeError, ValueError), match="omegas"):
        engine.step(phases, bad_omega, knm, 0.0, 0.0, alpha)


def test_schedule_dispatch_rejects_invalid_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_python(monkeypatch)
    phases, knm, alpha = _zero_problem()

    with pytest.raises(ValueError, match="omega_schedule"):
        upde_run_omega_schedule(
            phases,
            np.array([[True, False]], dtype=bool),
            knm,
            alpha,
            0.0,
            0.0,
            0.1,
        )


def test_run_requires_explicit_or_configured_omega(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_python(monkeypatch)
    phases, knm, alpha = _zero_problem()
    engine = UPDEEngine(2, dt=0.1)

    with pytest.raises(ValueError, match="omegas are required"):
        engine.run(phases, knm=knm, alpha=alpha, n_steps=1)
