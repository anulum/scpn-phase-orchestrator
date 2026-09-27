# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Splitting measurement types

"""Exercise original measurement types through the public splitting engine."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators.upde._splitting_go import (
    splitting_run_go,
)
from scpn_phase_orchestrator.experimental.accelerators.upde._splitting_julia import (
    splitting_run_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.upde._splitting_mojo import (
    splitting_run_mojo,
)
from scpn_phase_orchestrator.upde.splitting import SplittingEngine


@pytest.mark.parametrize("name", ["phases", "omegas", "knm", "alpha"])
@pytest.mark.parametrize(
    "kind", ["timedelta", "datetime", "object_temporal", "mixed_boolean"]
)
def test_measurement_alias_refusal(name: str, kind: str) -> None:
    """Reject temporal storage and mixed booleans before numerical integration."""
    shape = (3, 3) if name in {"knm", "alpha"} else (3,)
    if kind == "timedelta":
        bad = np.zeros(shape, dtype="timedelta64[ns]")
    elif kind == "datetime":
        bad = np.zeros(shape, dtype="datetime64[ns]")
    elif kind == "object_temporal":
        bad = np.array(
            [np.timedelta64(1, "ns")] * int(np.prod(shape)), dtype=object
        ).reshape(shape)
    else:
        bad = np.ones(shape, dtype=object)
        bad.flat[0] = True
    values = {
        "phases": np.zeros(3),
        "omegas": np.ones(3),
        "knm": np.zeros((3, 3)),
        "alpha": np.zeros((3, 3)),
    }
    values[name] = cast(NDArray[np.float64], bad)
    engine = SplittingEngine(3, 0.01)
    with pytest.raises(ValueError):
        engine.step(
            values["phases"], values["omegas"], values["knm"], 0.0, 0.0, values["alpha"]
        )


@pytest.mark.parametrize("name", ["n_oscillators", "dt", "zeta", "psi", "n_steps"])
def test_temporal_control_refusal(name: str) -> None:
    """A duration scalar cannot stand in for a dimensionless or numerical control."""
    temporal = np.timedelta64(1, "ns")
    with pytest.raises(ValueError):
        if name == "n_oscillators":
            SplittingEngine(cast(int, temporal), 0.01)
        elif name == "dt":
            SplittingEngine(3, cast(float, temporal))
        else:
            engine = SplittingEngine(3, 0.01)
            engine.run(
                np.zeros(3),
                np.ones(3),
                np.zeros((3, 3)),
                cast(float, temporal) if name == "zeta" else 0.0,
                cast(float, temporal) if name == "psi" else 0.0,
                np.zeros((3, 3)),
                cast(int, temporal) if name == "n_steps" else 1,
            )


def test_real_object_state_matches_numeric_state() -> None:
    """Object storage containing genuine numbers preserves the numerical trajectory."""
    engine = SplittingEngine(3, 0.01)
    phases = np.array([0.0, 0.1, 0.2])
    omega = np.ones(3)
    coupling = np.zeros((3, 3))
    expected = engine.run(phases, omega, coupling, 0.0, 0.0, coupling, 2)
    actual = engine.run(
        phases.astype(object),
        omega.astype(object),
        coupling.astype(object),
        0.0,
        0.0,
        coupling.astype(object),
        2,
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "backend", [splitting_run_go, splitting_run_julia, splitting_run_mojo]
)
@pytest.mark.parametrize(
    "name",
    ["phases", "omegas", "knm_flat", "alpha_flat", "n", "zeta", "psi", "dt", "n_steps"],
)
def test_direct_backend_temporal_refusal(
    backend: Callable[..., object], name: str
) -> None:
    """Direct accelerator calls reject original temporal arrays and controls."""
    values: dict[str, object] = {
        "phases": np.zeros(3),
        "omegas": np.ones(3),
        "knm_flat": np.zeros(9),
        "alpha_flat": np.zeros(9),
        "n": 3,
        "zeta": 0.0,
        "psi": 0.0,
        "dt": 0.01,
        "n_steps": 2,
    }
    if name in {"phases", "omegas", "knm_flat", "alpha_flat"}:
        values[name] = np.zeros(9 if "flat" in name else 3, dtype="timedelta64[ns]")
    else:
        values[name] = np.timedelta64(1, "ns")
    with pytest.raises(ValueError):
        backend(**values)
