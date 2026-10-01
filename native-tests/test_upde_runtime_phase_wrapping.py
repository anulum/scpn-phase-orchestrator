# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mandatory compiled UPDE projection parity

"""Verify actual compiled Go, Julia and Mojo via the public UPDE dispatch routes.

The Julia translator preserves native exceptions other than DomainError. Real
finite-input divergence here reaches DomainError; manufacturing a different
native exception with a substituted function would bypass the owning solver.
The unchanged re-raise stays unexecuted, with actual divergence and subsequent
valid calls covering the nearest boundary.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from scpn_phase_orchestrator.coupling.spatial_modulator import SpatialCouplingModulator
from scpn_phase_orchestrator.upde import engine
from scpn_phase_orchestrator.upde.doppler import doppler_run
from scpn_phase_orchestrator.upde.moving_frame import moving_frame_run

pytestmark = pytest.mark.native_runtime


@pytest.fixture(params=["go", "julia", "mojo"])
def runtime_backend(request: pytest.FixtureRequest) -> Iterator[str]:
    """Require an actual loaded backend and restore the public selection afterward."""
    name = request.param
    assert isinstance(name, str)
    assert name in engine.AVAILABLE_BACKENDS, f"Mandatory real runtime missing: {name}"
    previous = engine.ACTIVE_BACKEND
    engine.ACTIVE_BACKEND = name
    try:
        yield name
    finally:
        engine.ACTIVE_BACKEND = previous


@pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
@pytest.mark.parametrize("surface", ["fixed", "scheduled", "doppler", "moving"])
def test_real_runtime_phase_projection(
    runtime_backend: str, method: str, surface: str
) -> None:
    """Exercise every compiled wrap loop with real cut and signed-multiple inputs."""
    tau = 2.0 * np.pi
    interior = np.nextafter(tau, 0.0)
    phases = np.array([0.0, -tau, -2.0 * tau, -0.0, tau, interior, 0.25])
    n = phases.size
    omega = np.array([-1e-15, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0])
    coupling = np.zeros((n, n))
    lag = np.zeros((n, n))
    schedule = omega[None, :]
    velocities = np.ones((1, n))
    positions = np.arange(n, dtype=np.float64)
    arrays = (phases, omega, coupling, lag, schedule, velocities, positions)
    before = tuple(array.tobytes() for array in arrays)
    if surface == "fixed":
        result = engine.upde_run(
            phases, omega, coupling, lag, 0.0, 0.0, 0.01, 1, method=method
        )
    elif surface == "scheduled":
        result = engine.upde_run_omega_schedule(
            phases, schedule, coupling, lag, 0.0, 0.0, 0.01, method=method
        )
    elif surface == "doppler":
        result = doppler_run(
            phases,
            schedule,
            coupling,
            lag,
            velocities,
            dt=0.01,
            method=method,
            backend=runtime_backend,
        )
    else:
        packed = moving_frame_run(
            phases,
            positions,
            schedule,
            coupling,
            lag,
            velocities,
            SpatialCouplingModulator(K_base=1.0),
            dt=0.01,
            method=method,
            backend=runtime_backend,
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
def test_real_runtime_zero_step_copy(runtime_backend: str, method: str) -> None:
    """Retain unwrapped values and input ownership when no integration is requested."""
    phases = np.array([-2.0 * np.pi, -0.0, 9.0])
    omega = np.zeros(3)
    matrix = np.zeros((3, 3))
    result = engine.upde_run(
        phases, omega, matrix, matrix, 0.0, 0.0, 0.01, 0, method=method
    )
    assert runtime_backend == engine.ACTIVE_BACKEND
    assert result.tobytes() == phases.tobytes()
    assert not np.shares_memory(result, phases)


@pytest.mark.parametrize("phase", [1e20, -1e20, 1e100, -1e308, 1e308])
def test_real_runtime_large_finite_remainder(
    runtime_backend: str, phase: float
) -> None:
    """Keep floating-point remainder valid beyond integer quotient representation."""
    phases = np.array([phase])
    zero = np.zeros(1)
    matrix = np.zeros((1, 1))
    result = engine.upde_run(
        phases, zero, matrix, matrix, 0.0, 0.0, 0.01, 1, method="euler"
    )
    assert runtime_backend == engine.ACTIVE_BACKEND
    np.testing.assert_array_equal(result, np.remainder(phases, 2.0 * np.pi))


@pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
@pytest.mark.parametrize("surface", ["fixed", "scheduled", "doppler", "moving"])
def test_real_runtime_finite_output_refusal(
    runtime_backend: str, method: str, surface: str
) -> None:
    """Refuse actual compiled overflow and retain subsequent stateless execution."""
    phases = np.zeros(1)
    omega = np.array([1e308])
    matrix = np.zeros((1, 1))
    before = tuple(a.tobytes() for a in (phases, omega, matrix))
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(ValueError):
        if surface == "fixed":
            engine.upde_run(
                phases,
                omega,
                matrix,
                matrix,
                1e308,
                np.pi / 2.0,
                0.01,
                1,
                method=method,
            )
        elif surface == "scheduled":
            engine.upde_run_omega_schedule(
                phases,
                omega[None, :],
                matrix,
                matrix,
                1e308,
                np.pi / 2.0,
                0.01,
                method=method,
            )
        elif surface == "doppler":
            doppler_run(
                phases,
                omega[None, :],
                matrix,
                matrix,
                np.zeros((1, 1)),
                zeta=1e308,
                psi=np.pi / 2.0,
                dt=0.01,
                method=method,
                backend=runtime_backend,
            )
        else:
            moving_frame_run(
                phases,
                np.zeros(1),
                omega[None, :],
                matrix,
                matrix,
                np.zeros((1, 1)),
                SpatialCouplingModulator(K_base=1.0),
                zeta=1e308,
                psi=np.pi / 2.0,
                dt=0.01,
                method=method,
                backend=runtime_backend,
            )
    assert before == tuple(a.tobytes() for a in (phases, omega, matrix))
    assert runtime_backend == engine.ACTIVE_BACKEND
    result = engine.upde_run(
        phases, np.ones(1), matrix, matrix, 0.0, 0.0, 0.01, 1, method=method
    )
    np.testing.assert_allclose(result, [0.01], rtol=0.0, atol=2e-17)
