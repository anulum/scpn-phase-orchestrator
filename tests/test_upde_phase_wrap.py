# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Public NumPy phase projection contracts

"""Test scpn_phase_orchestrator.upde._phase_wrap via public UPDE consumers."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.coupling.spatial_modulator import SpatialCouplingModulator
from scpn_phase_orchestrator.upde import engine
from scpn_phase_orchestrator.upde.doppler import doppler_run
from scpn_phase_orchestrator.upde.moving_frame import moving_frame_run

FloatArray = NDArray[np.float64]


@pytest.fixture
def python_backend() -> Iterator[None]:
    """Select and restore the facade's real documented Python backend."""
    previous = engine.ACTIVE_BACKEND
    engine.ACTIVE_BACKEND = "python"
    try:
        yield
    finally:
        engine.ACTIVE_BACKEND = previous


@pytest.mark.usefixtures("python_backend")
@pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
@pytest.mark.parametrize("surface", ["fixed", "scheduled", "doppler", "moving"])
def test_phase_projection_at_public_consumers(method: str, surface: str) -> None:
    """Preserve interior phases and buffers across actual torus cut crossings."""
    period = 2.0 * np.pi
    interior = np.nextafter(period, 0.0)
    phases = np.array([0.0, -period, -2.0 * period, -0.0, period, interior, 0.25])
    omegas = np.array([-1e-15, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0])
    n = phases.size
    knm = np.zeros((n, n))
    alpha = np.zeros((n, n))
    schedule = omegas[None, :]
    velocities = np.ones((1, n))
    positions = np.arange(n, dtype=np.float64)
    arrays = (phases, omegas, knm, alpha, schedule, velocities, positions)
    before = tuple(array.tobytes() for array in arrays)
    if surface == "fixed":
        result = engine.upde_run(
            phases, omegas, knm, alpha, 0.0, 0.0, 0.01, 1, method=method
        )
    elif surface == "scheduled":
        result = engine.upde_run_omega_schedule(
            phases, schedule, knm, alpha, 0.0, 0.0, 0.01, method=method
        )
    elif surface == "doppler":
        result = doppler_run(
            phases,
            schedule,
            knm,
            alpha,
            velocities,
            dt=0.01,
            method=method,
            backend="python",
        )
    else:
        packed = moving_frame_run(
            phases,
            positions,
            schedule,
            knm,
            alpha,
            velocities,
            SpatialCouplingModulator(K_base=1.0),
            dt=0.01,
            method=method,
            backend="python",
        )
        np.testing.assert_array_equal(packed[n:], positions + 0.01)
        result = packed[:n]
    np.testing.assert_array_equal(result[:5], np.zeros(5))
    assert not np.any(np.signbit(result[:5]))
    assert result[5] == interior
    assert result[6] == pytest.approx(0.27, rel=0.0, abs=2e-16)
    assert np.all((result >= 0.0) & (result < period))
    assert not np.shares_memory(result, phases)
    assert tuple(array.tobytes() for array in arrays) == before


@pytest.mark.usefixtures("python_backend")
@pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
def test_zero_steps_preserve_unwrapped_input(method: str) -> None:
    """Keep the no-integration copy contract independent of canonical projection."""
    phases = np.array([-2.0 * np.pi, -0.0, 9.0])
    zero = np.zeros_like(phases)
    matrix = np.zeros((phases.size, phases.size))
    result = engine.upde_run(
        phases, zero, matrix, matrix, 0.0, 0.0, 0.01, 0, method=method
    )
    assert result.tobytes() == phases.tobytes()
    assert not np.shares_memory(result, phases)


@pytest.mark.usefixtures("python_backend")
@pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
def test_nonfinite_inputs_still_refuse(method: str) -> None:
    """Reject real invalid phase inputs before the projection can conceal them."""
    phases = np.array([np.nan, 0.2])
    omega = np.ones(2)
    matrix = np.zeros((2, 2))
    with pytest.raises(ValueError, match="finite"):
        engine.upde_run(phases, omega, matrix, matrix, 0.0, 0.0, 0.01, 1, method=method)
