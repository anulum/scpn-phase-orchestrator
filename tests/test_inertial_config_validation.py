# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — inertial engine config validation tests

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.inertial import InertialKuramotoEngine


@pytest.mark.parametrize("n", [False, 0, -1, 1.5, "4"])
def test_inertial_engine_rejects_invalid_oscillator_count(n: Any) -> None:
    with pytest.raises(ValueError, match="n must be >= 1"):
        InertialKuramotoEngine(n=n)


@pytest.mark.parametrize("dt", [False, 0.0, -0.01, float("nan"), float("inf"), "0.01"])
def test_inertial_engine_rejects_invalid_timestep(dt: Any) -> None:
    with pytest.raises(ValueError, match="dt must be positive"):
        InertialKuramotoEngine(n=4, dt=dt)


def test_inertial_engine_normalises_accepted_numpy_scalars() -> None:
    engine = InertialKuramotoEngine(n=np.int64(4), dt=np.float64(0.01))

    assert engine._n == 4
    assert engine._dt == pytest.approx(0.01)


@pytest.mark.parametrize(
    ("field", "bad_value", "match"),
    [
        ("theta", np.array(["0.1", "0.2"]), "theta.*numeric-string"),
        ("omega_dot", np.array(["0.0", "0.1"]), "omega_dot.*numeric-string"),
        ("power", np.array(["1.0", "-1.0"]), "power.*numeric-string"),
        (
            "knm",
            np.array([["0.0", "0.2"], ["0.2", "0.0"]]),
            "knm.*numeric-string",
        ),
        ("inertia", np.array(["1.0", "1.0"]), "inertia.*numeric-string"),
        ("damping", np.array(["0.1", "0.1"]), "damping.*numeric-string"),
    ],
)
def test_inertial_step_rejects_numeric_string_state_arrays_before_coercion(
    field: str,
    bad_value: np.ndarray,
    match: str,
) -> None:
    engine = InertialKuramotoEngine(n=2, dt=0.01)
    payload: dict[str, object] = {
        "theta": np.array([0.1, 0.2], dtype=np.float64),
        "omega_dot": np.array([0.0, 0.1], dtype=np.float64),
        "power": np.array([1.0, -1.0], dtype=np.float64),
        "knm": np.array([[0.0, 0.2], [0.2, 0.0]], dtype=np.float64),
        "inertia": np.array([1.0, 1.0], dtype=np.float64),
        "damping": np.array([0.1, 0.1], dtype=np.float64),
    }
    payload[field] = bad_value

    with pytest.raises(ValueError, match=match):
        engine.step(
            cast(Any, payload["theta"]),
            cast(Any, payload["omega_dot"]),
            cast(Any, payload["power"]),
            cast(Any, payload["knm"]),
            cast(Any, payload["inertia"]),
            cast(Any, payload["damping"]),
        )


def test_inertial_run_rejects_numeric_string_step_count_before_coercion() -> None:
    engine = InertialKuramotoEngine(n=2, dt=0.01)

    with pytest.raises(ValueError, match="n_steps.*numeric-string"):
        engine.run(
            np.array([0.1, 0.2], dtype=np.float64),
            np.array([0.0, 0.1], dtype=np.float64),
            np.array([1.0, -1.0], dtype=np.float64),
            np.array([[0.0, 0.2], [0.2, 0.0]], dtype=np.float64),
            np.array([1.0, 1.0], dtype=np.float64),
            np.array([0.1, 0.1], dtype=np.float64),
            n_steps="2",
        )


@pytest.mark.parametrize(
    ("method_name", "payload"),
    [
        ("frequency_deviation", np.array(["0.0", "0.1"])),
        ("coherence", np.array(["0.1", "0.2"])),
    ],
)
def test_inertial_public_metrics_reject_numeric_string_arrays(
    method_name: str,
    payload: np.ndarray,
) -> None:
    engine = InertialKuramotoEngine(n=2, dt=0.01)
    method = getattr(engine, method_name)

    with pytest.raises(ValueError, match="numeric-string"):
        method(payload)
