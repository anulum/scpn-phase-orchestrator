# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — geometric engine config validation tests

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.geometric import TorusEngine


@pytest.mark.parametrize("n_oscillators", [False, 0, -1, 1.5, "4"])
def test_torus_engine_rejects_invalid_oscillator_count(
    n_oscillators: Any,
) -> None:
    with pytest.raises(ValueError, match="n_oscillators must be >= 1"):
        TorusEngine(n_oscillators=n_oscillators, dt=0.01)


@pytest.mark.parametrize("dt", [False, 0.0, -0.01, float("nan"), float("inf"), "0.01"])
def test_torus_engine_rejects_invalid_timestep(dt: Any) -> None:
    with pytest.raises(ValueError, match="dt must be positive"):
        TorusEngine(n_oscillators=4, dt=dt)


@pytest.mark.parametrize("n_steps", [False, -1, 1.5, "10"])
def test_torus_engine_run_rejects_invalid_step_count(n_steps: Any) -> None:
    engine = TorusEngine(n_oscillators=4, dt=0.01)
    phases = np.zeros(4, dtype=np.float64)
    omegas = np.ones(4, dtype=np.float64)
    knm = np.zeros((4, 4), dtype=np.float64)
    alpha = np.zeros((4, 4), dtype=np.float64)

    with pytest.raises(ValueError, match="n_steps must be >= 0"):
        engine.run(phases, omegas, knm, 0.0, 0.0, alpha, n_steps=n_steps)


def test_torus_engine_normalises_accepted_numpy_scalars() -> None:
    engine = TorusEngine(n_oscillators=np.int64(4), dt=np.float64(0.01))

    assert engine._n == 4
    assert pytest.approx(0.01) == engine._dt


@pytest.mark.parametrize(
    ("field", "bad_value", "match"),
    [
        ("phases", np.zeros((5,), dtype=np.float64), "phases shape"),
        ("omegas", np.zeros((3,), dtype=np.float64), "omegas shape"),
        ("knm", np.zeros((4, 3), dtype=np.float64), "knm shape"),
        ("alpha", np.zeros((3, 4), dtype=np.float64), "alpha shape"),
    ],
)
def test_torus_run_rejects_state_shape_mismatch(
    field: str,
    bad_value: np.ndarray,
    match: str,
) -> None:
    engine = TorusEngine(n_oscillators=4, dt=0.01)
    values = {
        "phases": np.zeros(4, dtype=np.float64),
        "omegas": np.ones(4, dtype=np.float64),
        "knm": np.zeros((4, 4), dtype=np.float64),
        "alpha": np.zeros((4, 4), dtype=np.float64),
    }
    values[field] = bad_value

    with pytest.raises(ValueError, match=match):
        engine.run(
            values["phases"],
            values["omegas"],
            values["knm"],
            0.0,
            0.0,
            values["alpha"],
            n_steps=1,
        )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("phases", np.nan),
        ("omegas", np.inf),
        ("knm", np.nan),
        ("alpha", np.inf),
    ],
)
def test_torus_run_rejects_non_finite_state_arrays(
    field: str,
    bad_value: float,
) -> None:
    engine = TorusEngine(n_oscillators=4, dt=0.01)
    phases = np.zeros(4, dtype=np.float64)
    omegas = np.ones(4, dtype=np.float64)
    knm = np.zeros((4, 4), dtype=np.float64)
    alpha = np.zeros((4, 4), dtype=np.float64)
    if field in {"knm", "alpha"}:
        locals()[field][0, 1] = bad_value
    else:
        locals()[field][0] = bad_value

    with pytest.raises(ValueError, match=field):
        engine.run(phases, omegas, knm, 0.0, 0.0, alpha, n_steps=1)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("zeta", False),
        ("zeta", np.nan),
        ("zeta", np.inf),
        ("zeta", "1.0"),
        ("psi", False),
        ("psi", np.nan),
        ("psi", np.inf),
        ("psi", "0.0"),
    ],
)
def test_torus_run_rejects_invalid_scalar_inputs(
    field: str,
    bad_value: Any,
) -> None:
    engine = TorusEngine(n_oscillators=4, dt=0.01)
    kwargs = {"zeta": 0.0, "psi": 0.0}
    kwargs[field] = bad_value

    with pytest.raises(ValueError, match=field):
        engine.run(
            np.zeros(4, dtype=np.float64),
            np.ones(4, dtype=np.float64),
            np.zeros((4, 4), dtype=np.float64),
            kwargs["zeta"],
            kwargs["psi"],
            np.zeros((4, 4), dtype=np.float64),
            n_steps=1,
        )
