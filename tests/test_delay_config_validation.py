# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — delay constructor validation tests

"""Reject invalid delay counts, time controls and boolean state aliases."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.upde.delay import DelayBuffer, DelayedEngine

_R_CALL = TypeVar("_R_CALL")


def _call(fn: Callable[..., _R_CALL], *args: object, **kwargs: object) -> _R_CALL:
    """Call ``fn`` with arguments mypy cannot check: invalid or dynamic kwargs."""
    return fn(*args, **kwargs)


@pytest.mark.parametrize("n_oscillators", [False, 0, -1, 1.5, "4"])
def test_delay_buffer_rejects_invalid_oscillator_count(n_oscillators: object) -> None:
    """Reject nonpositive and nonintegral buffer dimensions without coercion."""
    with pytest.raises(ValueError, match="n_oscillators"):
        _call(DelayBuffer, n_oscillators=n_oscillators, max_delay_steps=3)


@pytest.mark.parametrize("max_delay_steps", [True, 0, -1, 1.5, "3"])
def test_delay_buffer_rejects_invalid_max_delay(max_delay_steps: object) -> None:
    """Reject a nonpositive or aliased history capacity."""
    with pytest.raises(ValueError, match="max_delay_steps"):
        _call(DelayBuffer, n_oscillators=4, max_delay_steps=max_delay_steps)


@pytest.mark.parametrize("delay_steps", [False, 0, -1, 1.5, "2"])
def test_delay_buffer_rejects_invalid_delayed_lookup(delay_steps: object) -> None:
    """Reject invalid one-based lookup counts on a populated buffer."""
    buffer = DelayBuffer(n_oscillators=2, max_delay_steps=3)
    buffer.push(np.array([0.0, 0.5], dtype=np.float64))

    with pytest.raises(ValueError, match="delay_steps"):
        _call(buffer.get_delayed, delay_steps)


def test_delay_buffer_push_rejects_boolean_phase_alias() -> None:
    """Reject boolean phase aliases rather than converting them to radians."""
    buffer = DelayBuffer(n_oscillators=2, max_delay_steps=3)

    with pytest.raises(ValueError, match="boolean"):
        buffer.push(np.array([0.0, True], dtype=object))


@pytest.mark.parametrize("n_oscillators", [True, 0, -1, 1.5, "4"])
def test_delayed_engine_rejects_invalid_oscillator_count(n_oscillators: object) -> None:
    """Reject nonpositive and aliased oscillator counts at construction."""
    with pytest.raises(ValueError, match="n_oscillators"):
        _call(DelayedEngine, n_oscillators=n_oscillators, dt=0.01)


@pytest.mark.parametrize("dt", [False, 0.0, -0.01, float("nan"), float("inf"), "0.01"])
def test_delayed_engine_rejects_invalid_dt(dt: object) -> None:
    """Require an actual positive finite Euler timestep."""
    with pytest.raises(ValueError, match="dt"):
        _call(DelayedEngine, n_oscillators=4, dt=dt)


@pytest.mark.parametrize("delay_steps", [False, 0, -1, 1.5, "3"])
def test_delayed_engine_rejects_invalid_delay_steps(delay_steps: object) -> None:
    """Require an actual positive integral source delay."""
    with pytest.raises(ValueError, match="delay_steps"):
        _call(DelayedEngine, n_oscillators=4, dt=0.01, delay_steps=delay_steps)


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
    bad_value: object,
) -> None:
    """Reject boolean aliases in each public numerical state buffer."""
    engine = DelayedEngine(n_oscillators=4, dt=0.01, delay_steps=2)
    values = {
        "phases": np.zeros(4, dtype=np.float64),
        "omegas": np.ones(4, dtype=np.float64),
        "knm": np.zeros((4, 4), dtype=np.float64),
        "alpha": np.zeros((4, 4), dtype=np.float64),
    }
    values[field] = cast("NDArray[np.float64]", bad_value)

    with pytest.raises(ValueError, match="boolean"):
        engine.step(
            values["phases"],
            values["omegas"],
            values["knm"],
            0.0,
            0.0,
            values["alpha"],
        )
