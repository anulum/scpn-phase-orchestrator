# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Public Doppler shared matrix contracts

"""Exercise public Doppler shared-storage admission in actual runtime environments."""

from __future__ import annotations

import importlib.util
from functools import partial

import numpy as np
import pytest

from scpn_phase_orchestrator.upde import engine as engine_module
from scpn_phase_orchestrator.upde.doppler import DopplerEngine, doppler_run

CPU_BACKENDS = tuple(
    backend
    for backend in engine_module.AVAILABLE_BACKENDS
    if backend in {"rust", "mojo", "julia", "go", "python"}
)


@pytest.mark.parametrize("backend", CPU_BACKENDS)
@pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
@pytest.mark.parametrize("readonly", [False, True])
def test_public_doppler_run_shared_matrix_admission(
    backend: str, method: str, readonly: bool
) -> None:
    """Real selected Doppler backends retain shared inputs and readonly boundaries."""
    phases = np.array([0.1, 0.7])
    frequencies = np.array([[0.8, 1.2], [0.9, 1.1]])
    velocities = np.array([[0.2, -0.3], [0.3, -0.2]])
    shared = np.array([[0.0, 0.3], [0.4, 0.0]])
    original = shared.copy()
    shared.setflags(write=not readonly)
    runner = partial(
        doppler_run,
        phases,
        frequencies,
        velocity_schedule=velocities,
        doppler_strength=0.3,
        zeta=0.2,
        psi=0.4,
        dt=0.01,
        method=method,
        backend=backend,
    )
    expected = runner(knm=shared.copy(), alpha=shared.copy())
    if readonly and backend == "rust":
        with pytest.raises(ValueError):
            runner(knm=shared, alpha=shared)
    else:
        actual = runner(knm=shared, alpha=shared)
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(shared, original)
    np.testing.assert_array_equal(phases, [0.1, 0.7])
    np.testing.assert_array_equal(frequencies, [[0.8, 1.2], [0.9, 1.1]])
    np.testing.assert_array_equal(velocities, [[0.2, -0.3], [0.3, -0.2]])


@pytest.mark.parametrize("method", ["euler", "rk4", "rk45"])
@pytest.mark.parametrize("operation", ["step", "run"])
@pytest.mark.parametrize("readonly", [False, True])
def test_public_doppler_engine_shared_matrix_admission(
    method: str, operation: str, readonly: bool
) -> None:
    """Native refusals preserve phases/time/omega and permit writable retry."""
    phases = np.array([0.1, 0.7])
    frequencies = np.array([0.8, 1.2])
    velocities = np.array([0.2, -0.3])
    shared = np.array([[0.0, 0.3], [0.4, 0.0]])
    original = shared.copy()
    shared.setflags(write=not readonly)
    engine = DopplerEngine(
        2,
        frequencies,
        shared,
        shared,
        velocities=velocities,
        solver=method,
        phases=phases,
    )
    reference = DopplerEngine(
        2,
        frequencies.copy(),
        shared.copy(),
        shared.copy(),
        velocities=velocities.copy(),
        solver=method,
        phases=phases.copy(),
    )
    omega_before = engine.omega_current.copy()
    runner = engine.step if operation == "step" else engine.run
    reference_runner = reference.step if operation == "step" else reference.run
    expected = reference_runner()
    if readonly and importlib.util.find_spec("spo_kernel") is not None:
        with pytest.raises(ValueError):
            runner()
        assert engine.time == 0.0
        np.testing.assert_array_equal(engine.phases, phases)
        np.testing.assert_array_equal(engine.omega_current, omega_before)
        actual = runner(knm=shared.copy(), alpha=shared.copy())
    else:
        actual = runner()
    np.testing.assert_array_equal(actual, expected)
    assert engine.time == reference.time
    np.testing.assert_array_equal(engine.phases, reference.phases)
    np.testing.assert_array_equal(shared, original)
    np.testing.assert_array_equal(phases, [0.1, 0.7])
