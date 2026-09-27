# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Lyapunov measurement contracts

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators.monitor._lyapunov_go import (
    lyapunov_spectrum_go,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor._lyapunov_julia import (
    lyapunov_spectrum_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor._lyapunov_mojo import (
    lyapunov_spectrum_mojo,
)
from scpn_phase_orchestrator.monitor.lyapunov import LyapunovGuard, lyapunov_spectrum

FloatArray: TypeAlias = NDArray[np.float64]
SpectrumFn: TypeAlias = Callable[
    [FloatArray, FloatArray, FloatArray, FloatArray, float, int, int, float, float],
    FloatArray,
]


@pytest.mark.parametrize("argument", ["phases", "knm"])
@pytest.mark.parametrize(
    "dtype", ["timedelta64[ms]", "timedelta64[ns]", "datetime64[ms]"]
)
def test_guard_rejects_temporal_arrays_without_changing_history(
    argument: str, dtype: str
) -> None:
    phases = np.array([0.0, 1.0])
    knm = np.array([[0.0, 1.0], [1.0, 0.0]])
    guard = LyapunovGuard()
    first = guard.evaluate(phases, knm)
    invalid_phases = phases.astype(dtype) if argument == "phases" else phases
    invalid_knm = knm.astype(dtype) if argument == "knm" else knm
    with pytest.raises(ValueError, match=argument):
        guard.evaluate(invalid_phases, invalid_knm)
    repeated = guard.evaluate(phases, knm)
    assert repeated.V == first.V
    assert repeated.dV_dt == 0.0


@pytest.mark.parametrize("argument", ["phases", "knm"])
@pytest.mark.parametrize("item", [np.timedelta64(1, "ms"), np.datetime64("2026-01-01")])
def test_guard_rejects_temporal_objects(argument: str, item: object) -> None:
    phases = np.array([0.0, 1.0], dtype=object)
    knm = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=object)
    if argument == "phases":
        phases[1] = item
    else:
        knm[0, 1] = item
    with pytest.raises(ValueError, match=argument):
        LyapunovGuard().evaluate(phases, knm)


def test_guard_preserves_numeric_object_energy_and_basin() -> None:
    phases = np.array([0.0, 1.0])
    knm = np.array([[0.0, 1.0], [1.0, 0.0]])
    expected = LyapunovGuard().evaluate(phases, knm)
    actual = LyapunovGuard().evaluate(phases.astype(object), knm.astype(object))
    assert actual == expected
    assert pytest.approx(-np.cos(1.0) / 2) == actual.V
    assert actual.in_basin


@pytest.mark.parametrize("argument", ["phases_init", "omegas", "knm", "alpha"])
@pytest.mark.parametrize("dtype", ["timedelta64[ms]", "datetime64[ms]"])
def test_spectrum_rejects_temporal_arrays_before_backend_dispatch(
    argument: str, dtype: str
) -> None:
    phases = np.array([0.0, 1.0])
    omegas = np.ones(2)
    knm = np.array([[0.0, 1.0], [1.0, 0.0]])
    alpha = np.zeros((2, 2))
    with pytest.raises(ValueError, match=argument):
        lyapunov_spectrum(
            phases.astype(dtype) if argument == "phases_init" else phases,
            omegas.astype(dtype) if argument == "omegas" else omegas,
            knm.astype(dtype) if argument == "knm" else knm,
            alpha.astype(dtype) if argument == "alpha" else alpha,
            n_steps=0,
        )


@pytest.mark.parametrize(
    "backend", [lyapunov_spectrum_go, lyapunov_spectrum_julia, lyapunov_spectrum_mojo]
)
@pytest.mark.parametrize("argument", ["phases_init", "omegas", "knm", "alpha"])
@pytest.mark.parametrize("dtype", ["timedelta64[ms]", "datetime64[ms]", "object_time"])
def test_direct_spectrum_bridges_reject_temporal_measurements(
    backend: SpectrumFn, argument: str, dtype: str
) -> None:
    """Every direct native bridge validates source units before runtime loading."""
    phases = np.array([0.0, 1.0])
    omegas = np.ones(2)
    knm = np.array([[0.0, 1.0], [1.0, 0.0]])
    alpha = np.zeros((2, 2))
    raw = {"phases_init": phases, "omegas": omegas, "knm": knm, "alpha": alpha}[
        argument
    ]
    invalid = raw.astype(object if dtype == "object_time" else dtype)
    if dtype == "object_time":
        invalid.flat[0] = np.timedelta64(1, "ms")
    with pytest.raises(ValueError, match=argument):
        backend(
            invalid if argument == "phases_init" else phases,
            invalid if argument == "omegas" else omegas,
            invalid if argument == "knm" else knm,
            invalid if argument == "alpha" else alpha,
            0.01,
            2,
            1,
            0.0,
            0.0,
        )


@pytest.mark.parametrize("argument", ["dt", "n_steps"])
def test_public_spectrum_rejects_temporal_scalar_controls(argument: str) -> None:
    """Temporal scalars cannot bypass Real or Integral scalar checks."""
    duration = np.timedelta64(1, "ms")
    with pytest.raises(ValueError, match=argument):
        lyapunov_spectrum(
            np.array([0.0, 1.0]),
            np.ones(2),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.zeros((2, 2)),
            dt=cast("float", duration) if argument == "dt" else 0.01,
            n_steps=cast("int", duration) if argument == "n_steps" else 2,
        )
