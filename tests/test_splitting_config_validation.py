# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — splitting engine config validation tests

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.splitting import SplittingEngine


@pytest.mark.parametrize("n_oscillators", [False, 0, -1, 1.5, "4"])
def test_splitting_engine_rejects_invalid_oscillator_count(
    n_oscillators: Any,
) -> None:
    with pytest.raises(ValueError, match="n_oscillators must be >= 1"):
        SplittingEngine(n_oscillators=n_oscillators, dt=0.01)


@pytest.mark.parametrize("dt", [False, 0.0, float("nan"), float("inf"), "0.01"])
def test_splitting_engine_rejects_invalid_timestep(dt: Any) -> None:
    with pytest.raises(ValueError, match="dt must be a finite non-zero real"):
        SplittingEngine(n_oscillators=4, dt=dt)


def test_splitting_engine_preserves_negative_timestep_support() -> None:
    engine = SplittingEngine(n_oscillators=4, dt=np.float64(-0.01))

    assert engine._n == 4
    assert pytest.approx(-0.01) == engine._dt


def test_splitting_engine_normalises_accepted_numpy_scalars() -> None:
    engine = SplittingEngine(n_oscillators=np.int64(4), dt=np.float64(0.01))

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
def test_splitting_run_rejects_state_shape_mismatch(
    field: str,
    bad_value: np.ndarray,
    match: str,
) -> None:
    engine = SplittingEngine(n_oscillators=4, dt=0.01)
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
def test_splitting_run_rejects_non_finite_state_arrays(
    field: str,
    bad_value: float,
) -> None:
    engine = SplittingEngine(n_oscillators=4, dt=0.01)
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
        ("phases", np.array(["0.0", "0.1", "0.2", "0.3"], dtype=object)),
        ("omegas", np.array(["1.0", "1.1", "1.2", "1.3"], dtype=object)),
        (
            "knm",
            np.array(
                [
                    ["0.0", "0.1", "0.2", "0.3"],
                    ["0.1", "0.0", "0.2", "0.3"],
                    ["0.1", "0.2", "0.0", "0.3"],
                    ["0.1", "0.2", "0.3", "0.0"],
                ],
                dtype=object,
            ),
        ),
        ("alpha", np.zeros((4, 4), dtype=object).astype(str)),
    ],
)
def test_splitting_run_rejects_numeric_string_state_arrays(
    field: str,
    bad_value: np.ndarray,
) -> None:
    engine = SplittingEngine(n_oscillators=4, dt=0.01)
    values = {
        "phases": np.zeros(4, dtype=np.float64),
        "omegas": np.ones(4, dtype=np.float64),
        "knm": np.zeros((4, 4), dtype=np.float64),
        "alpha": np.zeros((4, 4), dtype=np.float64),
    }
    values[field] = bad_value

    with pytest.raises(ValueError, match="numeric-string"):
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
def test_splitting_run_rejects_invalid_scalar_inputs(
    field: str,
    bad_value: Any,
) -> None:
    engine = SplittingEngine(n_oscillators=4, dt=0.01)
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


@pytest.mark.parametrize("n_steps", [False, 0, -1, 1.5, "10"])
def test_splitting_run_rejects_invalid_step_count(n_steps: Any) -> None:
    engine = SplittingEngine(n_oscillators=4, dt=0.01)

    with pytest.raises(ValueError, match="n_steps must be >= 1"):
        engine.run(
            np.zeros(4, dtype=np.float64),
            np.ones(4, dtype=np.float64),
            np.zeros((4, 4), dtype=np.float64),
            0.0,
            0.0,
            np.zeros((4, 4), dtype=np.float64),
            n_steps=n_steps,
        )
