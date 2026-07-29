# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Boundary observer validation contracts

"""
Validation contracts for BoundaryObserver construction, observation, and event-bus
wiring.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.binding.types import BoundaryDef
from scpn_phase_orchestrator.monitor.boundaries import BoundaryObserver


def test_u1_boundary_observer_rejects_inverted_bounds() -> None:
    with pytest.raises(TypeError, match="BoundaryDef"):
        BoundaryObserver([object()])  # type: ignore[list-item]


def test_u1_boundary_observer_rejects_non_list_boundary_defs() -> None:
    with pytest.raises(TypeError, match="list\\[BoundaryDef\\]"):
        BoundaryObserver({})  # type: ignore[arg-type]


def test_u1_boundary_observer_rejects_boolean_lower_bound() -> None:
    with pytest.raises(Exception, match="must be < upper"):
        BoundaryObserver(
            [
                BoundaryDef(
                    name="n",
                    variable="x",
                    lower=True,  # type: ignore[arg-type]
                    upper=1.0,
                    severity="soft",
                )
            ]
        )


def test_u1_boundary_observer_rejects_blank_boundary_name() -> None:
    with pytest.raises(Exception, match="non-empty name and variable"):
        BoundaryObserver(
            [
                BoundaryDef(
                    name=" ",
                    variable="x",
                    lower=0.0,
                    upper=1.0,
                    severity="soft",
                )
            ]
        )


def test_u1_boundary_observer_rejects_blank_boundary_variable() -> None:
    with pytest.raises(Exception, match="non-empty name and variable"):
        BoundaryObserver(
            [
                BoundaryDef(
                    name="n",
                    variable=" ",
                    lower=0.0,
                    upper=1.0,
                    severity="soft",
                )
            ]
        )


def test_u1_boundary_observer_observe_rejects_negative_step() -> None:
    obs = BoundaryObserver([])
    with pytest.raises(ValueError, match="non-negative integer"):
        obs.observe({}, step=-1)


def test_u1_boundary_observer_observe_rejects_boolean_step() -> None:
    obs = BoundaryObserver([])
    with pytest.raises(ValueError, match="non-negative integer"):
        obs.observe({}, step=True)  # type: ignore[arg-type]


def test_u1_boundary_observer_observe_rejects_bool_value() -> None:
    obs = BoundaryObserver([])
    with pytest.raises(ValueError, match="finite float"):
        obs.observe({"R": True})  # type: ignore[dict-item]


def test_u1_boundary_observer_observe_rejects_non_numeric_value() -> None:
    obs = BoundaryObserver([])
    with pytest.raises(ValueError, match="finite float"):
        obs.observe({"R": "bad"})  # type: ignore[dict-item]


def test_u1_boundary_observer_observe_rejects_blank_value_key() -> None:
    obs = BoundaryObserver([])
    with pytest.raises(ValueError, match="non-empty strings"):
        obs.observe({" ": 0.1})


def test_u1_boundary_observer_observe_rejects_non_dict_values() -> None:
    obs = BoundaryObserver([])
    with pytest.raises(TypeError, match="dict\\[str, float\\]"):
        obs.observe([("R", 0.1)])  # type: ignore[arg-type]


def test_u1_boundary_observer_set_event_bus_rejects_wrong_type() -> None:
    obs = BoundaryObserver([])
    with pytest.raises(TypeError, match="EventBus"):
        obs.set_event_bus(object())  # type: ignore[arg-type]


def test_u1_boundary_observer_set_event_bus_rejects_none() -> None:
    obs = BoundaryObserver([])
    with pytest.raises(TypeError, match="EventBus"):
        obs.set_event_bus(None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bound",
    [True, np.bool_(True), "0.5", 0.5 + 0j, float("nan"), float("inf")],
)
def test_boundary_observer_rejects_coercive_or_non_finite_bounds(bound: Any) -> None:
    bdef = BoundaryDef(
        name="floor",
        variable="x",
        lower=bound,
        upper=None,
        severity="soft",
    )

    with pytest.raises(ValueError, match="lower bound must be finite real"):
        BoundaryObserver([bdef])


def test_boundary_observer_rejects_coercive_upper_bound() -> None:
    bdef = BoundaryDef(
        name="ceiling",
        variable="x",
        lower=None,
        upper="1.0",  # type: ignore[arg-type]
        severity="hard",
    )

    with pytest.raises(ValueError, match="upper bound must be finite real"):
        BoundaryObserver([bdef])


def test_boundary_observer_revalidates_tampered_bound_order() -> None:
    bdef = BoundaryDef(
        name="band",
        variable="x",
        lower=None,
        upper=None,
        severity="hard",
    )
    object.__setattr__(bdef, "lower", 2.0)
    object.__setattr__(bdef, "upper", 1.0)

    with pytest.raises(ValueError, match="requires lower < upper"):
        BoundaryObserver([bdef])


@pytest.mark.parametrize("field", ["name", "variable"])
def test_boundary_observer_rejects_non_text_identifiers(field: str) -> None:
    kwargs: dict[str, Any] = {
        "name": "floor",
        "variable": "x",
        "lower": None,
        "upper": None,
        "severity": "soft",
    }
    kwargs[field] = 7

    with pytest.raises(ValueError, match="non-empty name and variable"):
        BoundaryObserver([BoundaryDef(**kwargs)])


def test_boundary_observer_rejects_non_text_severity() -> None:
    bdef = BoundaryDef(
        name="floor",
        variable="x",
        lower=0.0,
        upper=None,
        severity=None,  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="severity must be a string"):
        BoundaryObserver([bdef])


def test_boundary_observer_snapshots_caller_owned_definitions() -> None:
    definitions = [
        BoundaryDef(
            name="floor",
            variable="x",
            lower=0.0,
            upper=None,
            severity="soft",
        )
    ]
    observer = BoundaryObserver(definitions)
    definitions.clear()

    state = observer.observe({"x": -1.0})

    assert len(state.soft_violations) == 1


def test_boundary_observer_accepts_numpy_real_measurements_and_step() -> None:
    observer = BoundaryObserver(
        [
            BoundaryDef(
                name="floor",
                variable="x",
                lower=np.float32(0.0),
                upper=None,
                severity="soft",
            )
        ]
    )

    state = observer.observe({"x": np.float32(-1.0)}, step=np.int64(3))  # type: ignore[dict-item,arg-type]

    assert len(state.soft_violations) == 1
    assert observer._step == 3
