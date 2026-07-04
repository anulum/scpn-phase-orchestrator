# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — geometric engine config validation tests

from __future__ import annotations

from typing import Any, cast

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


@pytest.mark.parametrize(
    ("constructor_kwargs", "match"),
    [
        ({"n_oscillators": "4", "dt": 0.01}, "n_oscillators.*numeric-string"),
        ({"n_oscillators": 4, "dt": "0.01"}, "dt.*numeric-string"),
    ],
)
def test_torus_engine_rejects_constructor_numeric_string_aliases(
    constructor_kwargs: dict[str, Any],
    match: str,
) -> None:
    """Reject constructor numeric-string aliases before numeric coercion."""
    with pytest.raises(ValueError, match=match):
        TorusEngine(**constructor_kwargs)


def test_torus_engine_normalises_accepted_numpy_scalars() -> None:
    engine = TorusEngine(
        n_oscillators=cast("int", np.int64(4)),
        dt=cast("float", np.float64(0.01)),
    )

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
        ("phases", np.array(["0.0", "0.1", "0.2", "0.3"], dtype=object)),
        ("omegas", np.array(["1.0", "1.1", "1.2", "1.3"], dtype=object)),
        (
            "knm",
            np.array(
                [
                    ["0.0", "0.1", "0.2", "0.3"],
                    ["0.4", "0.0", "0.5", "0.6"],
                    ["0.7", "0.8", "0.0", "0.9"],
                    ["1.0", "1.1", "1.2", "0.0"],
                ],
                dtype=object,
            ),
        ),
        ("alpha", np.full((4, 4), "0.0", dtype=object)),
    ],
)
def test_torus_run_rejects_numeric_string_state_aliases(
    field: str,
    bad_value: np.ndarray,
) -> None:
    """Reject state arrays that would otherwise be accepted by ``float``."""
    engine = TorusEngine(n_oscillators=4, dt=0.01)
    values: dict[str, object] = {
        "phases": np.zeros(4, dtype=np.float64),
        "omegas": np.ones(4, dtype=np.float64),
        "knm": np.zeros((4, 4), dtype=np.float64),
        "alpha": np.zeros((4, 4), dtype=np.float64),
    }
    values[field] = bad_value

    with pytest.raises(ValueError, match=rf"{field}.*numeric-string"):
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


@pytest.mark.parametrize(
    ("field", "bad_value", "match"),
    [
        ("zeta", "0.5", "zeta.*numeric-string"),
        ("psi", "0.25", "psi.*numeric-string"),
        ("n_steps", "2", "n_steps.*numeric-string"),
    ],
)
def test_torus_run_rejects_numeric_string_scalar_aliases(
    field: str,
    bad_value: Any,
    match: str,
) -> None:
    """Reject scalar control numeric-string aliases before dispatch."""
    engine = TorusEngine(n_oscillators=4, dt=0.01)
    kwargs: dict[str, Any] = {"zeta": 0.0, "psi": 0.0, "n_steps": 1}
    kwargs[field] = bad_value

    with pytest.raises(ValueError, match=match):
        engine.run(
            np.zeros(4, dtype=np.float64),
            np.ones(4, dtype=np.float64),
            np.zeros((4, 4), dtype=np.float64),
            kwargs["zeta"],
            kwargs["psi"],
            np.zeros((4, 4), dtype=np.float64),
            n_steps=kwargs["n_steps"],
        )


def test_order_parameter_rejects_numeric_string_phase_aliases() -> None:
    """Reject order-parameter phase aliases before complex projection."""
    engine = TorusEngine(n_oscillators=4, dt=0.01)

    with pytest.raises(ValueError, match="phases.*numeric-string"):
        engine.order_parameter(np.array(["0.0", "0.1", "0.2", "0.3"], dtype=object))


def test_torus_run_rejects_non_numeric_state_arrays() -> None:
    """Reject non-coercible public state arrays before backend dispatch."""
    engine = TorusEngine(n_oscillators=4, dt=0.01)

    with pytest.raises(ValueError, match="phases must be a finite float array"):
        engine.run(
            np.array(["bad", "bad", "bad", "bad"], dtype=object),
            np.ones(4, dtype=np.float64),
            np.zeros((4, 4), dtype=np.float64),
            0.0,
            0.0,
            np.zeros((4, 4), dtype=np.float64),
            n_steps=1,
        )
