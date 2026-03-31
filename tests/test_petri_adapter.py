# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Petri adapter tests

from __future__ import annotations

import pytest

from scpn_phase_orchestrator.supervisor.events import EventBus
from scpn_phase_orchestrator.supervisor.petri_adapter import PetriNetAdapter
from scpn_phase_orchestrator.supervisor.petri_net import (
    Arc,
    Guard,
    Marking,
    PetriNet,
    Place,
    Transition,
)
from scpn_phase_orchestrator.supervisor.regimes import Regime


def _protocol_net():
    places = [Place("warmup"), Place("nominal"), Place("cooldown"), Place("done")]
    transitions = [
        Transition(
            "start",
            inputs=[Arc("warmup")],
            outputs=[Arc("nominal")],
            guard=Guard("stability_proxy", ">", 0.6),
        ),
        Transition(
            "wind_down",
            inputs=[Arc("nominal")],
            outputs=[Arc("cooldown")],
            guard=Guard("R_0", "<", 0.3),
        ),
        Transition(
            "finish",
            inputs=[Arc("cooldown")],
            outputs=[Arc("done")],
        ),
    ]
    return PetriNet(places, transitions)


def _place_regime_map():
    return {
        "warmup": "NOMINAL",
        "nominal": "NOMINAL",
        "cooldown": "RECOVERY",
        "done": "NOMINAL",
    }


def test_initial_regime():
    adapter = PetriNetAdapter(
        _protocol_net(),
        Marking(tokens={"warmup": 1}),
        _place_regime_map(),
    )
    assert adapter._active_regime() == Regime.NOMINAL


def test_step_no_transition():
    adapter = PetriNetAdapter(
        _protocol_net(),
        Marking(tokens={"warmup": 1}),
        _place_regime_map(),
    )
    regime = adapter.step({"stability_proxy": 0.3})
    assert regime == Regime.NOMINAL
    assert adapter.marking["warmup"] == 1


def test_step_fires_start():
    adapter = PetriNetAdapter(
        _protocol_net(),
        Marking(tokens={"warmup": 1}),
        _place_regime_map(),
    )
    regime = adapter.step({"stability_proxy": 0.8, "R_0": 0.5})
    assert regime == Regime.NOMINAL
    assert adapter.marking["nominal"] == 1
    assert adapter.marking["warmup"] == 0


def test_full_protocol_sequence():
    adapter = PetriNetAdapter(
        _protocol_net(),
        Marking(tokens={"warmup": 1}),
        _place_regime_map(),
    )
    # warmup -> nominal
    adapter.step({"stability_proxy": 0.8, "R_0": 0.5})
    assert adapter.marking["nominal"] == 1

    # nominal -> cooldown
    regime = adapter.step({"stability_proxy": 0.8, "R_0": 0.2})
    assert regime == Regime.RECOVERY
    assert adapter.marking["cooldown"] == 1

    # cooldown -> done
    regime = adapter.step({})
    assert regime == Regime.NOMINAL
    assert adapter.marking["done"] == 1


def test_event_bus_receives_petri_events():
    bus = EventBus()
    adapter = PetriNetAdapter(
        _protocol_net(),
        Marking(tokens={"warmup": 1}),
        _place_regime_map(),
        event_bus=bus,
    )
    adapter.step({"stability_proxy": 0.8})
    assert bus.count == 1
    assert bus.history[0].kind == "petri_transition"
    assert bus.history[0].detail == "start"


def test_no_event_when_no_transition():
    bus = EventBus()
    adapter = PetriNetAdapter(
        _protocol_net(),
        Marking(tokens={"warmup": 1}),
        _place_regime_map(),
        event_bus=bus,
    )
    adapter.step({"stability_proxy": 0.3})
    assert bus.count == 0


def test_invalid_regime_name():
    with pytest.raises(ValueError, match="unknown regime"):
        PetriNetAdapter(
            _protocol_net(),
            Marking(tokens={"warmup": 1}),
            {"warmup": "IMAGINARY"},
        )


def test_highest_priority_regime_wins():
    places = [Place("a"), Place("b")]
    net = PetriNet(places, [])
    adapter = PetriNetAdapter(
        net,
        Marking(tokens={"a": 1, "b": 1}),
        {"a": "NOMINAL", "b": "CRITICAL"},
    )
    assert adapter._active_regime() == Regime.CRITICAL


def test_marking_property():
    adapter = PetriNetAdapter(
        _protocol_net(),
        Marking(tokens={"warmup": 1}),
        _place_regime_map(),
    )
    assert adapter.marking["warmup"] == 1


def test_net_property():
    net = _protocol_net()
    adapter = PetriNetAdapter(net, Marking(), _place_regime_map())
    assert adapter.net is net


def test_empty_marking_gives_nominal():
    adapter = PetriNetAdapter(
        _protocol_net(),
        Marking(),
        _place_regime_map(),
    )
    assert adapter._active_regime() == Regime.NOMINAL


def test_unmapped_place_defaults_nominal():
    places = [Place("unknown_place")]
    net = PetriNet(places, [])
    adapter = PetriNetAdapter(
        net,
        Marking(tokens={"unknown_place": 1}),
        {},
    )
    assert adapter._active_regime() == Regime.NOMINAL


def test_adapter_multiple_steps():
    adapter = PetriNetAdapter(
        _protocol_net(),
        Marking(tokens={"warmup": 1}),
        _place_regime_map(),
    )
    for _ in range(5):
        adapter.step({"stability_proxy": 0.3})
    assert adapter.marking["warmup"] == 1


def test_case_insensitive_regime():
    adapter = PetriNetAdapter(
        _protocol_net(),
        Marking(tokens={"warmup": 1}),
        {"warmup": "nominal", "nominal": "Nominal"},
    )
    assert adapter._active_regime() == Regime.NOMINAL


def test_cooldown_place_to_recovery():
    adapter = PetriNetAdapter(
        _protocol_net(),
        Marking(tokens={"cooldown": 1}),
        _place_regime_map(),
    )
    assert adapter._active_regime() == Regime.RECOVERY


def test_step_increments():
    bus = EventBus()
    adapter = PetriNetAdapter(
        _protocol_net(),
        Marking(tokens={"warmup": 1}),
        _place_regime_map(),
        event_bus=bus,
    )
    adapter.step({"stability_proxy": 0.8})
    assert bus.history[0].step == 1
    adapter.step({"R_0": 0.2})
    assert bus.history[1].step == 2


def test_degraded_regime_mapping():
    places = [Place("degraded_place")]
    net = PetriNet(places, [])
    adapter = PetriNetAdapter(
        net,
        Marking(tokens={"degraded_place": 1}),
        {"degraded_place": "DEGRADED"},
    )
    assert adapter._active_regime() == Regime.DEGRADED


def test_multiple_marked_priority():
    places = [Place("a"), Place("b"), Place("c")]
    net = PetriNet(places, [])
    adapter = PetriNetAdapter(
        net,
        Marking(tokens={"a": 1, "b": 1, "c": 1}),
        {"a": "NOMINAL", "b": "DEGRADED", "c": "RECOVERY"},
    )
    assert adapter._active_regime() == Regime.RECOVERY


def test_supervisor_fires_petri_transition():
    """End-to-end: SupervisorPolicy → PetriNetAdapter → RegimeManager."""
    from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
    from scpn_phase_orchestrator.supervisor.policy import SupervisorPolicy
    from scpn_phase_orchestrator.supervisor.regimes import RegimeManager
    from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

    bus = EventBus()
    rm = RegimeManager(event_bus=bus)
    adapter = PetriNetAdapter(
        _protocol_net(),
        Marking(tokens={"warmup": 1}),
        _place_regime_map(),
        event_bus=bus,
    )
    supervisor = SupervisorPolicy(rm, petri_adapter=adapter)
    upde = UPDEState(
        layers=[LayerState(R=0.8, psi=0.0)],
        cross_layer_alignment=[[1.0]],
        stability_proxy=0.8,
        regime_id="nominal",
    )
    _actions = supervisor.decide(
        upde, BoundaryState(), petri_ctx={"stability_proxy": 0.8}
    )
    assert adapter.marking["nominal"] == 1
    assert adapter.marking["warmup"] == 0
    assert any(e.kind == "petri_transition" for e in bus.history)


# Pipeline wiring is proven by test_supervisor_fires_petri_transition above:
# SupervisorPolicy → PetriNetAdapter → RegimeManager → EventBus.
