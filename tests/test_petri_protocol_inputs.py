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
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from scpn_phase_orchestrator.binding.loader import load_binding_spec
from scpn_phase_orchestrator.binding.types import (
    ProtocolNetSpec,
    ProtocolTransitionSpec,
)
from scpn_phase_orchestrator.binding.validator import validate_binding_spec
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


_INVALID_ARCS: list[tuple[str, dict[str, object], str]] = [
    (
        "missing-place",
        {"weight": 1},
        " must be a mapping {place: <name>, weight: <int>}, got {'weight': 1}",
    ),
    ("unknown-place", {"place": "missing"}, ": unknown place 'missing'"),
    *[
        (
            f"weight-{weight!r}",
            {"place": "nominal", "weight": weight},
            ": weight must be a positive integer",
        )
        for weight in [0, -1, True, False, 1.5, "1"]
    ],
]


@pytest.mark.parametrize(
    ("protocol", "expected_errors"),
    [
        pytest.param(
            replace(_protocol(), places=["nominal", "recovery", ""]),
            ["protocol_net.places must be non-empty strings"],
            id="blank-place",
        ),
        pytest.param(
            replace(_protocol(), initial={"missing": 1}),
            ["protocol_net.initial: unknown place 'missing'"],
            id="unknown-initial-place",
        ),
        *[
            pytest.param(
                replace(_protocol(), initial={"nominal": tokens}),
                ["protocol_net.initial['nominal'] must be a non-negative integer"],
                id=f"invalid-token-{tokens}",
            )
            for tokens in [-1, True, False]
        ],
        pytest.param(
            replace(_protocol(), place_regime={"missing": "RECOVERY"}),
            ["protocol_net.place_regime: unknown place 'missing'"],
            id="unknown-regime-place",
        ),
        pytest.param(
            replace(
                _protocol(),
                places=["nominal", "recovery", ""],
                initial={"missing": -1, "nominal": True},
                place_regime={"missing": "RECOVERY"},
            ),
            [
                "protocol_net.places must be non-empty strings",
                "protocol_net.initial: unknown place 'missing'",
                "protocol_net.initial['missing'] must be a non-negative integer",
                "protocol_net.initial['nominal'] must be a non-negative integer",
                "protocol_net.place_regime: unknown place 'missing'",
            ],
            id="accumulated-diagnostics",
        ),
        *[
            pytest.param(
                _protocol(guard),
                [
                    "protocol_net.transition 'recover': guard "
                    f"{guard!r} is not 'metric op threshold': {reason}"
                ],
                id=f"guard-{label}",
            )
            for label, guard, reason in [
                (
                    "syntax",
                    "load >",
                    "guard must be 'metric op threshold', got 'load >'",
                ),
                (
                    "threshold",
                    "load > nope",
                    "threshold must be finite, got 'nope'",
                ),
                (
                    "nonfinite",
                    "load > nan",
                    "threshold must be finite, got nan",
                ),
                (
                    "operator",
                    "load != 0.6",
                    "operator must be one of ['<', '<=', '==', '>', '>='], got '!='",
                ),
            ]
        ],
        pytest.param(
            _protocol("nominal > 0.6"),
            [
                "protocol_net.transition 'recover': guard metric 'nominal' is a "
                "place name; guards read context metrics, and token availability "
                "is set by input arcs"
            ],
            id="place-as-guard-metric",
        ),
        *[
            pytest.param(
                replace(
                    _protocol(),
                    transitions=[
                        ProtocolTransitionSpec(
                            name="recover",
                            inputs=[arc]
                            if side == "inputs"
                            else [{"place": "nominal"}],
                            outputs=[arc]
                            if side == "outputs"
                            else [{"place": "recovery"}],
                            guard="load > 0.6",
                        )
                    ],
                ),
                [f"protocol_net.transition 'recover' {side}[0]{reason}"],
                id=f"{side}-{label}",
            )
            for side in ["inputs", "outputs"]
            for label, arc, reason in _INVALID_ARCS
        ],
        *[
            pytest.param(
                replace(_protocol(), place_regime=mapping),
                [f"protocol_net is refused by the runtime: {reason}"],
                id=f"runtime-{label}",
            )
            for label, mapping, reason in [
                ("empty-map", {}, "place_to_regime must not be empty"),
                (
                    "blank-regime",
                    {"nominal": " "},
                    "regime mapping value for place 'nominal' must be "
                    "non-empty string, got ' '",
                ),
                (
                    "unknown-regime",
                    {"nominal": "missing"},
                    "unknown regime 'missing' for place 'nominal'",
                ),
            ]
        ],
        pytest.param(
            replace(
                _protocol(),
                transitions=[
                    ProtocolTransitionSpec(
                        name="",
                        inputs=[{"place": "nominal"}],
                        outputs=[{"place": "recovery"}],
                    )
                ],
            ),
            [
                "protocol_net is refused by the runtime: transition names must "
                "not be empty, got ''"
            ],
            id="runtime-empty-transition-name",
        ),
        pytest.param(
            replace(
                _protocol(),
                transitions=[
                    ProtocolTransitionSpec(
                        name="recover",
                        inputs=[{"weight": 1}, {"place": "missing", "weight": False}],
                        outputs=[{"place": "recovery", "weight": 0}],
                        guard="nominal > 0.6",
                    )
                ],
            ),
            [
                "protocol_net.transition 'recover': guard metric 'nominal' is a "
                "place name; guards read context metrics, and token availability "
                "is set by input arcs",
                "protocol_net.transition 'recover' inputs[0] must be a mapping "
                "{place: <name>, weight: <int>}, got {'weight': 1}",
                "protocol_net.transition 'recover' inputs[1]: unknown place 'missing'",
                "protocol_net.transition 'recover' inputs[1]: weight must be a "
                "positive integer",
                "protocol_net.transition 'recover' outputs[0]: weight must be a "
                "positive integer",
            ],
            id="accumulated-transition-diagnostics",
        ),
    ],
)
def test_binding_protocol_admission_preserves_spec_and_recovers(
    protocol: ProtocolNetSpec, expected_errors: list[str]
) -> None:
    """Binding refusals preserve the spec before corrected runtime event delivery."""
    binding_path = (
        Path(__file__).resolve().parents[1]
        / "domainpacks/minimal_domain/binding_spec.yaml"
    )
    base = load_binding_spec(binding_path)
    malformed = replace(base, protocol_net=protocol)
    original = asdict(malformed)

    assert validate_binding_spec(malformed) == expected_errors
    assert validate_binding_spec(malformed) == expected_errors
    assert asdict(malformed) == original
    assert base.protocol_net is None

    repaired = replace(malformed, protocol_net=_protocol())
    assert validate_binding_spec(repaired) == []
    admitted_protocol = repaired.protocol_net
    assert admitted_protocol is not None
    net, marking = petri_net_from_protocol(admitted_protocol)
    bus = EventBus()
    adapter = PetriNetAdapter(
        net, marking, admitted_protocol.place_regime, event_bus=bus
    )
    assert adapter.step({"load": 0.6}) is Regime.NOMINAL
    assert adapter.marking.tokens == {"nominal": 1}
    assert bus.count == 0
    assert adapter.step({"load": 0.8}) is Regime.RECOVERY
    assert adapter.marking.tokens == {"recovery": 1}
    assert marking.tokens == {"nominal": 1}
    assert admitted_protocol.initial == {"nominal": 1}
    assert bus.count == 1
    assert bus.history[0].kind == "petri_transition"
    assert bus.history[0].detail == "recover"
    assert bus.history[0].step == 2
    assert asdict(malformed) == original


def test_binding_unguarded_weighted_transition_consumes_exact_tokens() -> None:
    """An admitted unguarded transition waits for its full input weight."""
    binding_path = (
        Path(__file__).resolve().parents[1]
        / "domainpacks/minimal_domain/binding_spec.yaml"
    )
    protocol = replace(
        _protocol(),
        transitions=[
            ProtocolTransitionSpec(
                name="recover",
                inputs=[{"place": "nominal", "weight": 2}],
                outputs=[{"place": "recovery", "weight": 3}],
            )
        ],
    )
    binding = replace(load_binding_spec(binding_path), protocol_net=protocol)
    original = asdict(binding)
    assert validate_binding_spec(binding) == []
    net, marking = petri_net_from_protocol(protocol)
    bus = EventBus()
    adapter = PetriNetAdapter(net, marking, protocol.place_regime, event_bus=bus)
    assert adapter.step({}) is Regime.NOMINAL
    assert adapter.marking.tokens == {"nominal": 1}
    assert bus.count == 0

    supplied = marking.copy()
    supplied["nominal"] = 2
    admitted = PetriNetAdapter(net, supplied, protocol.place_regime, event_bus=bus)
    assert admitted.step({}) is Regime.RECOVERY
    assert admitted.marking.tokens == {"recovery": 3}
    assert supplied.tokens == {"nominal": 2}
    assert marking.tokens == {"nominal": 1}
    assert bus.count == 1
    assert bus.history[0].kind == "petri_transition"
    assert bus.history[0].detail == "recover"
    assert bus.history[0].step == 1
    assert asdict(binding) == original


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
