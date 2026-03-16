# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Actuation mapper tests

from __future__ import annotations

from scpn_phase_orchestrator.actuation.mapper import ActuationMapper, ControlAction
from scpn_phase_orchestrator.binding.types import ActuatorMapping


def _mapper():
    mappings = [
        ActuatorMapping(name="K_glob", knob="K", scope="global", limits=(0.0, 1.0)),
        ActuatorMapping(
            name="zeta_glob", knob="zeta", scope="global", limits=(0.0, 0.5)
        ),
    ]
    return ActuationMapper(mappings)


def test_map_actions_produces_commands():
    mapper = _mapper()
    actions = [
        ControlAction(
            knob="K", scope="global", value=0.3, ttl_s=5.0, justification="test"
        )
    ]
    cmds = mapper.map_actions(actions)
    assert len(cmds) == 1
    assert cmds[0]["actuator"] == "K_glob"
    assert cmds[0]["value"] == 0.3


def test_map_actions_clamps_to_limits():
    mapper = _mapper()
    actions = [
        ControlAction(
            knob="K", scope="global", value=5.0, ttl_s=1.0, justification="over"
        )
    ]
    cmds = mapper.map_actions(actions)
    assert cmds[0]["value"] == 1.0


def test_validate_rejects_out_of_range():
    mapper = _mapper()
    action = ControlAction(
        knob="K", scope="global", value=2.0, ttl_s=1.0, justification="x"
    )
    assert mapper.validate_action(action) is False


def test_validate_accepts_in_range():
    mapper = _mapper()
    action = ControlAction(
        knob="K", scope="global", value=0.5, ttl_s=1.0, justification="x"
    )
    assert mapper.validate_action(action) is True


def test_validate_rejects_invalid_knob():
    mapper = _mapper()
    action = ControlAction(
        knob="omega", scope="global", value=0.1, ttl_s=1.0, justification="x"
    )
    assert mapper.validate_action(action) is False


def test_no_matching_actuator_produces_empty():
    mapper = _mapper()
    actions = [
        ControlAction(
            knob="Psi", scope="layer_0", value=0.1, ttl_s=1.0, justification="x"
        )
    ]
    cmds = mapper.map_actions(actions)
    assert cmds == []
