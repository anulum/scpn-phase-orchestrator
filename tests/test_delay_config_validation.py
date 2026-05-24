# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — delay constructor validation tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.delay import DelayBuffer, DelayedEngine


@pytest.mark.parametrize("n_oscillators", [False, 0, -1, 1.5, "4"])
def test_delay_buffer_rejects_invalid_oscillator_count(n_oscillators: object) -> None:
    with pytest.raises(ValueError, match="n_oscillators"):
        DelayBuffer(n_oscillators=n_oscillators, max_delay_steps=3)


@pytest.mark.parametrize("max_delay_steps", [True, 0, -1, 1.5, "3"])
def test_delay_buffer_rejects_invalid_max_delay(max_delay_steps: object) -> None:
    with pytest.raises(ValueError, match="max_delay_steps"):
        DelayBuffer(n_oscillators=4, max_delay_steps=max_delay_steps)


@pytest.mark.parametrize("delay_steps", [False, 0, -1, 1.5, "2"])
def test_delay_buffer_rejects_invalid_delayed_lookup(delay_steps: object) -> None:
    buffer = DelayBuffer(n_oscillators=2, max_delay_steps=3)
    buffer.push(np.array([0.0, 0.5], dtype=np.float64))

    with pytest.raises(ValueError, match="delay_steps"):
        buffer.get_delayed(delay_steps)


def test_delay_buffer_push_rejects_boolean_phase_alias() -> None:
    buffer = DelayBuffer(n_oscillators=2, max_delay_steps=3)

    with pytest.raises(ValueError, match="boolean"):
        buffer.push(np.array([0.0, True], dtype=object))


@pytest.mark.parametrize("n_oscillators", [True, 0, -1, 1.5, "4"])
def test_delayed_engine_rejects_invalid_oscillator_count(n_oscillators: object) -> None:
    with pytest.raises(ValueError, match="n_oscillators"):
        DelayedEngine(n_oscillators=n_oscillators, dt=0.01)


@pytest.mark.parametrize("dt", [False, 0.0, -0.01, float("nan"), float("inf"), "0.01"])
def test_delayed_engine_rejects_invalid_dt(dt: object) -> None:
    with pytest.raises(ValueError, match="dt"):
        DelayedEngine(n_oscillators=4, dt=dt)


@pytest.mark.parametrize("delay_steps", [False, 0, -1, 1.5, "3"])
def test_delayed_engine_rejects_invalid_delay_steps(delay_steps: object) -> None:
    with pytest.raises(ValueError, match="delay_steps"):
        DelayedEngine(n_oscillators=4, dt=0.01, delay_steps=delay_steps)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("phases", np.array([0.0, True, 0.2, 0.3], dtype=object)),
        ("omegas", np.array([1.0, 1.1, False, 1.3], dtype=object)),
        (
            "knm",
            np.array(
                [
                    [0.0, 0.2, 0.1, 0.0],
                    [0.2, 0.0, True, 0.1],
                    [0.1, 0.2, 0.0, 0.2],
                    [0.0, 0.1, 0.2, 0.0],
                ],
                dtype=object,
            ),
        ),
        ("alpha", np.eye(4, dtype=bool)),
    ],
)
def test_delayed_engine_step_rejects_boolean_state_aliases(
    field: str,
    bad_value: np.ndarray,
) -> None:
    engine = DelayedEngine(n_oscillators=4, dt=0.01, delay_steps=2)
    values = {
        "phases": np.zeros(4, dtype=np.float64),
        "omegas": np.ones(4, dtype=np.float64),
        "knm": np.zeros((4, 4), dtype=np.float64),
        "alpha": np.zeros((4, 4), dtype=np.float64),
    }
    values[field] = bad_value

    with pytest.raises(ValueError, match="boolean"):
        engine.step(
            values["phases"],
            values["omegas"],
            values["knm"],
            0.0,
            0.0,
            values["alpha"],
        )
