# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Delayed dynamics measurement contracts

"""Delayed dynamics refuse unrepresentable measurements before native dispatch."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators.upde._delay_go import (
    delayed_kuramoto_run_go,
)
from scpn_phase_orchestrator.experimental.accelerators.upde._delay_julia import (
    delayed_kuramoto_run_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.upde._delay_mojo import (
    delayed_kuramoto_run_mojo,
)
from scpn_phase_orchestrator.upde.delay import DelayBuffer, DelayedEngine

FloatArray: TypeAlias = NDArray[np.float64]
RunFn: TypeAlias = Callable[
    [
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        int,
        float,
        float,
        float,
        int,
        int,
    ],
    FloatArray,
]


@pytest.mark.parametrize(
    "run",
    [delayed_kuramoto_run_go, delayed_kuramoto_run_julia, delayed_kuramoto_run_mojo],
)
@pytest.mark.parametrize(
    "argument", [0, 1, 2, 3], ids=["phases", "omegas", "knm", "alpha"]
)
def test_direct_delay_rejects_real_objects_outside_float_range(
    run: RunFn, argument: int
) -> None:
    arrays = [np.zeros(2), np.zeros(2), np.zeros(4), np.zeros(4)]
    invalid = arrays[argument].astype(object)
    invalid[0] = 10**400
    arrays[argument] = invalid
    name = ("phases", "omegas", "knm_flat", "alpha_flat")[argument]
    with pytest.raises(ValueError, match=name) as error:
        run(arrays[0], arrays[1], arrays[2], arrays[3], 2, 0.0, 0.0, 0.05, 1, 5)
    assert isinstance(error.value.__cause__, OverflowError)


@pytest.mark.parametrize(
    "argument", [0, 1, 2, 3], ids=["phases", "omegas", "knm", "alpha"]
)
@pytest.mark.parametrize(
    "dtype", ["U", "S", "bool", "timedelta64[ms]", "datetime64[ms]"]
)
def test_delayed_engine_rejects_measurement_aliases(argument: int, dtype: str) -> None:
    arrays = [np.zeros(2), np.zeros(2), np.zeros((2, 2)), np.zeros((2, 2))]
    arrays[argument] = arrays[argument].astype(dtype)
    name = ("phases", "omegas", "knm", "alpha")[argument]
    with pytest.raises(ValueError, match=name):
        DelayedEngine(2, dt=0.05).run(
            arrays[0], arrays[1], arrays[2], alpha=arrays[3], n_steps=2
        )


@pytest.mark.parametrize(
    "argument", [0, 1, 2, 3], ids=["phases", "omegas", "knm", "alpha"]
)
def test_delayed_engine_rejects_real_objects_outside_float_range(argument: int) -> None:
    arrays = [np.zeros(2), np.zeros(2), np.zeros((2, 2)), np.zeros((2, 2))]
    invalid = arrays[argument].astype(object)
    invalid.flat[0] = 10**400
    arrays[argument] = invalid
    name = ("phases", "omegas", "knm", "alpha")[argument]
    with pytest.raises(ValueError, match=name) as error:
        DelayedEngine(2, dt=0.05).run(
            arrays[0], arrays[1], arrays[2], alpha=arrays[3], n_steps=2
        )
    assert isinstance(error.value.__cause__, OverflowError)


@pytest.mark.parametrize(
    "dtype", ["U", "S", "bool", "timedelta64[ms]", "datetime64[ms]"]
)
def test_delay_buffer_rejects_measurement_aliases(dtype: str) -> None:
    buffer = DelayBuffer(2, max_delay_steps=5)
    with pytest.raises(ValueError, match="phases"):
        buffer.push(np.array([0, 1]).astype(dtype))
    assert buffer.get_delayed(1) is None


def test_delayed_engine_preserves_real_object_trajectories() -> None:
    phases = np.array([0.2, 0.5])
    omegas = np.array([0.1, 0.3])
    knm = np.array([[0.0, 0.5], [0.5, 0.0]])
    alpha = np.zeros((2, 2))
    expected = DelayedEngine(2, dt=0.05).run(
        phases, omegas, knm, alpha=alpha, n_steps=5
    )
    actual = DelayedEngine(2, dt=0.05).run(
        phases.astype(object),
        omegas.astype(object),
        knm.astype(object),
        alpha=alpha.astype(object),
        n_steps=5,
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
    assert np.all((actual >= 0.0) & (actual < 2.0 * np.pi))


def test_delayed_step_preserves_external_drive_rotation() -> None:
    phases = np.array([0.2, 0.5])
    omegas = np.array([0.1, 0.3])
    dt, zeta, psi = 0.05, 0.7, 1.2
    expected = (phases + dt * (omegas + zeta * np.sin(psi - phases))) % (2 * np.pi)
    actual = DelayedEngine(2, dt=dt).step(
        phases, omegas, np.zeros((2, 2)), zeta=zeta, psi=psi
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
