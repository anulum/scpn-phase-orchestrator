# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Petri protocol input contracts

"""Public protocol refusal preserves marking and subsequent transition admission."""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from scpn_phase_orchestrator.binding.types import (
    ProtocolNetSpec,
    ProtocolTransitionSpec,
)
from scpn_phase_orchestrator.exceptions import PolicyError
from scpn_phase_orchestrator.supervisor.events import EventBus
from scpn_phase_orchestrator.supervisor.petri_adapter import PetriNetAdapter
from scpn_phase_orchestrator.supervisor.petri_net import petri_net_from_protocol
from scpn_phase_orchestrator.supervisor.regimes import Regime


def _protocol(guard: str = "load > 0.6") -> ProtocolNetSpec:
    """Return a guarded nominal-to-recovery protocol with one initial token."""
    return ProtocolNetSpec(
        places=["nominal", "recovery"],
        initial={"nominal": 1},
        place_regime={"nominal": "NOMINAL", "recovery": "RECOVERY"},
        transitions=[
            ProtocolTransitionSpec(
                name="recover",
                inputs=[{"place": "nominal"}],
                outputs=[{"place": "recovery"}],
                guard=guard,
            )
        ],
    )


@pytest.mark.parametrize("threshold", ["not-a-number", "null", "0,5"])
def test_protocol_builder_refuses_nonnumeric_guard_without_mutating_spec(
    threshold: str,
) -> None:
    """Guard parsing failure preserves the source protocol and its token seed."""
    protocol = _protocol("load > " + threshold)
    original = asdict(protocol)
    with pytest.raises(PolicyError, match="threshold must be finite"):
        petri_net_from_protocol(protocol)
    assert asdict(protocol) == original

    valid_protocol = _protocol()
    net, marking = petri_net_from_protocol(valid_protocol)
    admitted, transition = net.step(marking, {"load": 0.8})
    assert transition is not None and transition.name == "recover"
    assert admitted.tokens == {"recovery": 1}
    assert marking.tokens == {"nominal": 1}
    assert valid_protocol.initial == {"nominal": 1}


@pytest.mark.parametrize("payload", ["null", "[]", "true", "1", '"0.8"'])
def test_net_context_container_refusal_preserves_marking_and_recovers(
    payload: str,
) -> None:
    """Enabled and step reject JSON non-mappings before consuming any token."""
    net, marking = petri_net_from_protocol(_protocol())
    context = json.loads(payload)
    with pytest.raises(PolicyError, match="ctx must be a mapping"):
        net.enabled(marking, context)
    with pytest.raises(PolicyError, match="ctx must be a mapping"):
        net.step(marking, context)
    assert marking.tokens == {"nominal": 1}

    enabled = net.enabled(marking, {"load": 0.8})
    assert [transition.name for transition in enabled] == ["recover"]
    admitted, transition = net.step(marking, {"load": 0.8})
    assert transition is not None and transition.name == "recover"
    assert admitted.tokens == {"recovery": 1}
    assert marking.tokens == {"nominal": 1}


@pytest.mark.parametrize("place", ["", " "])
def test_adapter_blank_place_refusal_preserves_marking_and_event_bus(
    place: str,
) -> None:
    """Invalid regime-map keys cannot consume tokens or emit transition events."""
    protocol = _protocol()
    net, marking = petri_net_from_protocol(protocol)
    mapping = {place: "RECOVERY"}
    bus = EventBus()
    with pytest.raises(PolicyError, match="place mapping key must be non-empty"):
        PetriNetAdapter(net, marking, mapping, event_bus=bus)
    assert mapping == {place: "RECOVERY"}
    assert marking.tokens == {"nominal": 1}
    assert bus.count == 0

    adapter = PetriNetAdapter(net, marking, protocol.place_regime, event_bus=bus)
    assert adapter.step({"load": 0.8}) is Regime.RECOVERY
    assert adapter.marking.tokens == {"recovery": 1}
    assert marking.tokens == {"nominal": 1}
    assert bus.count == 1
    assert bus.history[0].kind == "petri_transition"
    assert bus.history[0].detail == "recover"
    assert bus.history[0].step == 1
