# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Reference UPDE kernel contracts

"""Numerical contracts for the dependency-free Python reference UPDE kernel."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator._compat import TWO_PI
from scpn_phase_orchestrator.upde import _ref_kernel


def test_reference_kernel_fallback_methods_match_manual_dynamics() -> None:
    phases = np.array([0.1, 0.4, 0.9], dtype=np.float64)
    omegas = np.array([0.2, -0.1, 0.05], dtype=np.float64)
    knm = np.array(
        [
            [0.0, 0.3, 0.1],
            [0.2, 0.0, 0.4],
            [0.5, 0.1, 0.0],
        ],
        dtype=np.float64,
    )
    alpha = np.array(
        [
            [0.0, 0.01, -0.02],
            [0.03, 0.0, 0.04],
            [-0.01, 0.02, 0.0],
        ],
        dtype=np.float64,
    )

    euler = _ref_kernel.upde_run_python(
        phases,
        omegas,
        knm,
        alpha,
        zeta=0.2,
        psi=0.7,
        dt=0.01,
        n_steps=3,
        method="euler",
        n_substeps=2,
        atol=1e-8,
        rtol=1e-6,
    )
    manual = phases.copy()
    for _ in range(3):
        for _ in range(2):
            diff = manual[np.newaxis, :] - manual[:, np.newaxis] - alpha
            deriv = omegas + np.sum(knm * np.sin(diff), axis=1)
            deriv += 0.2 * np.sin(0.7 - manual)
            manual = manual + 0.005 * deriv
        manual %= TWO_PI
    np.testing.assert_allclose(euler, manual, rtol=1e-12, atol=1e-12)

    rk4 = _ref_kernel.upde_run_python(
        phases,
        omegas,
        knm,
        alpha,
        zeta=0.0,
        psi=0.0,
        dt=0.01,
        n_steps=2,
        method="rk4",
        n_substeps=3,
        atol=1e-8,
        rtol=1e-6,
    )
    rk45 = _ref_kernel.upde_run_python(
        phases,
        omegas,
        knm,
        alpha,
        zeta=0.1,
        psi=0.5,
        dt=0.01,
        n_steps=2,
        method="rk45",
        n_substeps=1,
        atol=1e-8,
        rtol=1e-6,
    )
    assert rk4.shape == phases.shape
    assert rk45.shape == phases.shape
    assert np.all((rk4 >= 0.0) & (rk4 < TWO_PI))
    assert np.all((rk45 >= 0.0) & (rk45 < TWO_PI))


def test_reference_kernel_zero_steps_returns_independent_copy() -> None:
    phases = np.array([0.25, 1.5], dtype=np.float64)

    result = _ref_kernel.upde_run_python(
        phases,
        np.ones(2),
        np.ones((2, 2)),
        np.zeros((2, 2)),
        zeta=0.0,
        psi=0.0,
        dt=0.01,
        n_steps=0,
        method="euler",
        n_substeps=1,
        atol=1e-8,
        rtol=1e-6,
    )

    np.testing.assert_array_equal(result, phases)
    assert result is not phases
    result[0] = 999.0
    assert phases[0] == 0.25


def test_reference_kernel_euler_without_drive_matches_linear_phase_advance() -> None:
    phases = np.array([TWO_PI - 0.05, 0.25], dtype=np.float64)
    omegas = np.array([1.0, -0.5], dtype=np.float64)

    result = _ref_kernel.upde_run_python(
        phases,
        omegas,
        np.zeros((2, 2)),
        np.zeros((2, 2)),
        zeta=0.0,
        psi=1.25,
        dt=0.1,
        n_steps=2,
        method="euler",
        n_substeps=1,
        atol=1e-8,
        rtol=1e-6,
    )

    expected = (phases + 2 * 0.1 * omegas) % TWO_PI
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_reference_kernel_rk45_retry_exhaustion_returns_finite_wrapped_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_dp_stages(phases, omegas, knm, alpha, zeta, psi, dt):
        calls.append(dt)
        return phases + 10.0, phases - 10.0

    monkeypatch.setattr(_ref_kernel, "_dp_stages", fake_dp_stages)

    result = _ref_kernel.upde_run_python(
        np.array([0.1, 0.2], dtype=np.float64),
        np.ones(2),
        np.zeros((2, 2)),
        np.zeros((2, 2)),
        zeta=0.0,
        psi=0.0,
        dt=0.5,
        n_steps=1,
        method="rk45",
        n_substeps=1,
        atol=1e-12,
        rtol=1e-12,
    )

    assert len(calls) == 4
    assert all(
        next_dt < prev_dt for prev_dt, next_dt in zip(calls, calls[1:], strict=False)
    )
    np.testing.assert_allclose(result, (np.array([0.1, 0.2]) + 10.0) % TWO_PI)


def test_reference_kernel_rk45_zero_error_caps_next_internal_dt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_dp_stages(phases, omegas, knm, alpha, zeta, psi, dt):
        calls.append(dt)
        return phases + dt, phases + dt

    monkeypatch.setattr(_ref_kernel, "_dp_stages", fake_dp_stages)

    result = _ref_kernel.upde_run_python(
        np.array([0.1], dtype=np.float64),
        np.ones(1),
        np.zeros((1, 1)),
        np.zeros((1, 1)),
        zeta=0.0,
        psi=0.0,
        dt=0.25,
        n_steps=2,
        method="rk45",
        n_substeps=1,
        atol=1e-8,
        rtol=1e-6,
    )

    assert calls == [0.25, 1.25]
    np.testing.assert_allclose(result, np.array([(0.1 + 0.25 + 1.25) % TWO_PI]))


@pytest.mark.parametrize(
    ("method", "n_substeps", "match"),
    [
        ("bogus", 1, "unknown method"),
        ("euler", 0, "n_substeps"),
    ],
)
def test_reference_kernel_rejects_invalid_configuration(
    method: str,
    n_substeps: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _ref_kernel.upde_run_python(
            np.zeros(2),
            np.zeros(2),
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            0.0,
            0.0,
            0.01,
            1,
            method,
            n_substeps,
            1e-8,
            1e-6,
        )
