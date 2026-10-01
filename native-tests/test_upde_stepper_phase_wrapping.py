# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mandatory native dense torus contracts

"""Cross the real PyO3 dense boundary without a replacement native producer."""

from __future__ import annotations

import numpy as np
import pytest
from spo_kernel import PyUPDEStepper


@pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
@pytest.mark.parametrize("surface", ["step", "run", "schedule", "doppler", "moving"])
def test_native_phase_projection(method: str, surface: str) -> None:
    """Publish positive zero and preserve interior phases in every native run route."""
    tau = 2.0 * np.pi
    interior = np.nextafter(tau, 0.0)
    phases = np.array([0.0, -tau, -2.0 * tau, -0.0, tau, interior, 0.25])
    n = phases.size
    omega = np.array([-1e-15, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0])
    coupling = np.zeros(n * n)
    lag = np.zeros(n * n)
    velocity = np.ones(n)
    positions = np.arange(n, dtype=np.float64)
    arrays = (phases, omega, coupling, lag, velocity, positions)
    before = tuple(array.tobytes() for array in arrays)
    native = PyUPDEStepper(n, dt=0.01, method=method)
    if surface == "step":
        result = native.step(phases, omega, coupling, 0.0, 0.0, lag)
    elif surface == "run":
        result = native.run(phases, omega, coupling, 0.0, 0.0, lag, 1)
    elif surface == "schedule":
        result = native.run_omega_schedule(phases, omega, coupling, 0.0, 0.0, lag, 1)
    elif surface == "doppler":
        result = native.run_doppler_schedule(
            phases, omega, coupling, 0.0, 0.0, lag, velocity, 1.0, 1e-9, 1
        )
    else:
        packed = native.run_moving_frame_schedule(
            phases,
            positions,
            omega,
            coupling,
            0.0,
            0.0,
            lag,
            velocity,
            1.0,
            0,
            1.0,
            1.0,
            1e-12,
            1.0,
            1e-9,
            1,
        )
        np.testing.assert_array_equal(packed[n:], positions + 0.01)
        result = packed[:n]
    np.testing.assert_array_equal(result[:5], np.zeros(5))
    assert not np.any(np.signbit(result[:5]))
    assert result[5] == interior
    assert result[6] == pytest.approx(0.27, rel=0.0, abs=2e-16)
    assert np.all((result >= 0.0) & (result < tau))
    assert not np.shares_memory(result, phases)
    assert before == tuple(array.tobytes() for array in arrays)


@pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
@pytest.mark.parametrize("primed", [False, True])
def test_native_finite_output_refusal_preserves_proposal(
    method: str, primed: bool
) -> None:
    """Preserve cold/previously published diagnostics on real finite overflow.

    Parameters
    ----------
    method : str
        Actual native integration method.
    primed : bool
        Establish a nonzero order-parameter cache through a valid public step.
    """
    native = PyUPDEStepper(1, dt=0.01, method=method)
    phases = np.zeros(1)
    coupling = np.zeros(1)
    if primed:
        native.step(np.array([0.3]), np.zeros(1), coupling, 0.0, 0.0, coupling)
    previous_order = native.order_parameter()
    previous_dt = native.last_dt
    with pytest.raises(ValueError, match="output phases contain NaN/Inf"):
        native.step(phases, np.array([1e308]), coupling, 1e308, np.pi / 2.0, coupling)
    assert native.last_dt == previous_dt
    assert native.order_parameter() == previous_order
    np.testing.assert_array_equal(phases, np.zeros(1))
    recovered = native.step(phases, np.ones(1), coupling, 0.0, 0.0, coupling)
    np.testing.assert_allclose(recovered, [previous_dt], rtol=0.0, atol=2e-17)
