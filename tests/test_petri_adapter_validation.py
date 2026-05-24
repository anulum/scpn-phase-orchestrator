# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Petri adapter validation contracts

"""
Validation contracts for PetriNetAdapter constructor, context, mapping, and regime
normalisation boundaries.
"""

from __future__ import annotations

import pytest

from scpn_phase_orchestrator.supervisor.petri_adapter import PetriNetAdapter
from scpn_phase_orchestrator.supervisor.petri_net import (
    Marking,
    PetriNet,
    Place,
    Transition,
)


def test_u1_petri_adapter_step_rejects_blank_ctx_metric_name() -> None:
    net = PetriNet(
        places=[Place("nominal")],
        transitions=[],
    )
    adapter = PetriNetAdapter(
        net=net,
        initial_marking=Marking({"nominal": 1}),
        place_to_regime={"nominal": "NOMINAL"},
    )
    with pytest.raises(Exception, match="metric names must be non-empty strings"):
        adapter.step({" ": 0.1})  # type: ignore[dict-item]


def test_u1_petri_adapter_step_rejects_boolean_ctx_metric_value() -> None:
    net = PetriNet(
        places=[Place("nominal")],
        transitions=[],
    )
    adapter = PetriNetAdapter(
        net=net,
        initial_marking=Marking({"nominal": 1}),
        place_to_regime={"nominal": "NOMINAL"},
    )
    with pytest.raises(Exception, match="must be finite real"):
        adapter.step({"metric": True})  # type: ignore[dict-item]


def test_u1_petri_adapter_step_rejects_non_finite_ctx_metric_value() -> None:
    net = PetriNet(
        places=[Place("nominal")],
        transitions=[],
    )
    adapter = PetriNetAdapter(
        net=net,
        initial_marking=Marking({"nominal": 1}),
        place_to_regime={"nominal": "NOMINAL"},
    )
    with pytest.raises(Exception, match="must be finite real"):
        adapter.step({"metric": float("nan")})


def test_u1_petri_adapter_step_rejects_non_mapping_context() -> None:
    net = PetriNet(
        places=[Place("nominal")],
        transitions=[],
    )
    adapter = PetriNetAdapter(
        net=net,
        initial_marking=Marking({"nominal": 1}),
        place_to_regime={"nominal": "NOMINAL"},
    )
    with pytest.raises(Exception, match="ctx must be a mapping"):
        adapter.step([("metric", 0.1)])  # type: ignore[arg-type]


def test_u1_petri_adapter_rejects_integer_regime_mapping_value() -> None:
    net = PetriNet(
        places=[Place("nominal")],
        transitions=[],
    )
    with pytest.raises(Exception, match="must be non-empty string"):
        PetriNetAdapter(
            net=net,
            initial_marking=Marking({"nominal": 1}),
            place_to_regime={"nominal": True},  # type: ignore[dict-item]
        )


def test_u1_petri_adapter_rejects_non_petri_net() -> None:
    with pytest.raises(Exception, match="net must be a PetriNet"):
        PetriNetAdapter(  # type: ignore[arg-type]
            net=object(),
            initial_marking=Marking({"nominal": 1}),
            place_to_regime={"nominal": "NOMINAL"},
        )


def test_u1_petri_adapter_rejects_non_marking_initial_marking() -> None:
    net = PetriNet(
        places=[Place("nominal")],
        transitions=[],
    )
    with pytest.raises(Exception, match="initial_marking must be a Marking"):
        PetriNetAdapter(
            net=net,
            initial_marking=object(),  # type: ignore[arg-type]
            place_to_regime={"nominal": "NOMINAL"},
        )


def test_u1_petri_adapter_rejects_non_mapping_place_to_regime() -> None:
    net = PetriNet(
        places=[Place("nominal")],
        transitions=[],
    )
    with pytest.raises(Exception, match="place_to_regime must be a mapping"):
        PetriNetAdapter(
            net=net,
            initial_marking=Marking({"nominal": 1}),
            place_to_regime=[("nominal", "NOMINAL")],  # type: ignore[arg-type]
        )


def test_u1_petri_adapter_rejects_invalid_event_bus_type() -> None:
    net = PetriNet(
        places=[Place("nominal")],
        transitions=[],
    )
    with pytest.raises(Exception, match="event_bus must be an EventBus"):
        PetriNetAdapter(
            net=net,
            initial_marking=Marking({"nominal": 1}),
            place_to_regime={"nominal": "NOMINAL"},
            event_bus=object(),  # type: ignore[arg-type]
        )


def test_u1_petri_adapter_rejects_unknown_place_in_mapping() -> None:
    net = PetriNet(
        places=[Place("nominal")],
        transitions=[],
    )
    with pytest.raises(Exception, match="unknown place"):
        PetriNetAdapter(
            net=net,
            initial_marking=Marking({"nominal": 1}),
            place_to_regime={"missing": "NOMINAL"},
        )


def test_u1_petri_adapter_rejects_non_string_regime_mapping_value() -> None:
    net = PetriNet(
        places=[Place("nominal")],
        transitions=[Transition(name="noop", inputs=[], outputs=[])],
    )
    with pytest.raises(Exception, match="non-empty string"):
        PetriNetAdapter(
            net=net,
            initial_marking=Marking(tokens={"nominal": 1}),
            place_to_regime={"nominal": 1},  # type: ignore[dict-item]
        )


def test_u1_petri_adapter_rejects_empty_mapping() -> None:
    net = PetriNet(
        places=[Place("nominal")],
        transitions=[Transition(name="noop", inputs=[], outputs=[])],
    )
    with pytest.raises(Exception, match="must not be empty"):
        PetriNetAdapter(
            net=net,
            initial_marking=Marking(tokens={"nominal": 1}),
            place_to_regime={},
        )


def test_u1_petri_adapter_step_rejects_whitespace_ctx_metric() -> None:
    net = PetriNet(
        places=[Place("nominal")],
        transitions=[Transition(name="noop", inputs=[], outputs=[])],
    )
    adapter = PetriNetAdapter(
        net=net,
        initial_marking=Marking(tokens={"nominal": 1}),
        place_to_regime={"nominal": "nominal"},
    )
    with pytest.raises(Exception, match="non-empty strings"):
        adapter.step({" ": 0.0})


def test_u1_petri_adapter_step_rejects_boolean_ctx_value() -> None:
    net = PetriNet(
        places=[Place("nominal")],
        transitions=[Transition(name="noop", inputs=[], outputs=[])],
    )
    adapter = PetriNetAdapter(
        net=net,
        initial_marking=Marking(tokens={"nominal": 1}),
        place_to_regime={"nominal": "nominal"},
    )
    with pytest.raises(Exception, match="finite real"):
        adapter.step({"stability_proxy": True})  # type: ignore[dict-item]


def test_u1_petri_adapter_accepts_whitespace_wrapped_regime_name() -> None:
    net = PetriNet(
        places=[Place("nominal")],
        transitions=[Transition(name="noop", inputs=[], outputs=[])],
    )
    adapter = PetriNetAdapter(
        net=net,
        initial_marking=Marking(tokens={"nominal": 1}),
        place_to_regime={"nominal": " nominal "},
    )
    assert adapter.step({"stability_proxy": 0.0}).value == "nominal"


def test_u1_petri_adapter_accepts_whitespace_wrapped_place_key() -> None:
    net = PetriNet(
        places=[Place("nominal")],
        transitions=[Transition(name="noop", inputs=[], outputs=[])],
    )
    adapter = PetriNetAdapter(
        net=net,
        initial_marking=Marking(tokens={"nominal": 1}),
        place_to_regime={" nominal ": "nominal"},
    )
    assert adapter.step({"stability_proxy": 0.0}).value == "nominal"


def test_u1_petri_adapter_accepts_uppercase_regime_name() -> None:
    net = PetriNet(
        places=[Place("nominal")],
        transitions=[Transition(name="noop", inputs=[], outputs=[])],
    )
    adapter = PetriNetAdapter(
        net=net,
        initial_marking=Marking(tokens={"nominal": 1}),
        place_to_regime={"nominal": "NOMINAL"},
    )
    assert adapter.step({"stability_proxy": 0.0}).value == "nominal"
