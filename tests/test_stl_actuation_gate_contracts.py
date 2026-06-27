# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — STL runtime actuation gate contracts

"""Contracts for the non-actuating STL runtime actuation gate."""

from __future__ import annotations

from typing import cast

import pytest

from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.binding.types import ActuatorMapping
from scpn_phase_orchestrator.monitor.stl import actuation_gate as gate_module
from scpn_phase_orchestrator.monitor.stl.actuation_gate import (
    STLRuntimeActuationGate,
    validate_stl_runtime_actuation_gate,
)
from scpn_phase_orchestrator.monitor.stl.projection import (
    STLActionProjectionTemplate,
    STLProjectedActionPlan,
)


def _action(
    *,
    knob: str = "K",
    scope: str = "global",
    value: float = 0.7,
) -> ControlAction:
    """Return a projected control action for the runtime gate."""
    return ControlAction(
        knob=knob,
        scope=scope,
        value=value,
        ttl_s=0.5,
        justification="STL review candidate",
    )


def _plan(*actions: ControlAction) -> STLProjectedActionPlan:
    """Return a projected non-actuating STL action plan."""
    return STLProjectedActionPlan(
        spec="eventually (R >= 0.8)",
        actuating=False,
        approved_actions=tuple(actions),
        rejected_candidates=(),
    )


def _template(
    *,
    knob: str = "K",
    scope: str = "global",
    value_bounds: tuple[float, float] = (0.0, 1.0),
) -> STLActionProjectionTemplate:
    """Return a policy-approved STL projection template."""
    return STLActionProjectionTemplate(
        action="raise_coupling",
        knob=knob,
        scope=scope,
        base_value=0.5,
        step=1.0,
        ttl_s=0.5,
        previous_value=0.5,
        value_bounds=value_bounds,
        rate_limit=0.2,
    )


def test_gate_fails_closed_when_action_template_is_missing() -> None:
    """Projected actions without a runtime template remain non-actuating."""
    gate = validate_stl_runtime_actuation_gate(_plan(_action()), ())

    assert gate == STLRuntimeActuationGate(
        spec="eventually (R >= 0.8)",
        non_actuating=True,
        execution_disabled=True,
        accepted=False,
        action_count=1,
        mapper_valid_action_count=0,
        mapped_command_count=0,
        commands=(),
        blocked_reasons=("actuation_template_missing",),
    )


def test_gate_deduplicates_template_construction_failures() -> None:
    """Invalid template bounds fail before any runtime command can be mapped."""
    bool_bounds = cast("tuple[float, float]", (False, True))
    template = _template(value_bounds=bool_bounds)
    gate = validate_stl_runtime_actuation_gate(
        _plan(_action(), _action()),
        (template,),
    )

    assert gate.accepted is False
    assert gate.mapper_valid_action_count == 0
    assert gate.mapped_command_count == 0
    assert gate.blocked_reasons == ("actuation_mapper_rejected_template",)


def test_gate_fails_closed_when_mapper_rejects_runtime_knob() -> None:
    """Runtime mappings with unsupported actuation knobs are rejected."""
    gate = validate_stl_runtime_actuation_gate(
        _plan(_action(knob="theta_gain")),
        (_template(knob="theta_gain"),),
    )

    assert gate.accepted is False
    assert gate.action_count == 1
    assert gate.mapper_valid_action_count == 0
    assert gate.mapped_command_count == 0
    assert gate.blocked_reasons == ("actuation_mapper_rejected_template",)


def test_gate_reports_invalid_projected_runtime_action() -> None:
    """Out-of-bounds projected action values fail runtime validation."""
    gate = validate_stl_runtime_actuation_gate(
        _plan(_action(value=1.5)),
        (_template(value_bounds=(0.0, 1.0)),),
    )

    assert gate.accepted is False
    assert gate.action_count == 1
    assert gate.mapper_valid_action_count == 0
    assert gate.mapped_command_count == 0
    assert gate.commands == ()
    assert gate.blocked_reasons == ("runtime_action_validation_failed",)


class _IncompleteMapper:
    """Mapper test double that validates actions but drops command records."""

    def __init__(self, mappings: list[ActuatorMapping]) -> None:
        self.mapping_count = len(mappings)

    def validate_action(self, action: ControlAction) -> bool:
        """Return ``True`` so command loss is isolated to mapping."""
        return action.knob == "K"

    def map_actions(
        self,
        actions: list[ControlAction],
    ) -> list[dict[str, object]]:
        """Drop validated actions to simulate a mapper contract breach."""
        if not actions:
            raise AssertionError("expected validated runtime actions")
        return []


def test_gate_reports_incomplete_runtime_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validated actions without command records fail the audit gate."""
    monkeypatch.setattr(gate_module, "ActuationMapper", _IncompleteMapper)

    gate = validate_stl_runtime_actuation_gate(
        _plan(_action()),
        (_template(),),
    )

    assert gate.accepted is False
    assert gate.mapper_valid_action_count == 1
    assert gate.mapped_command_count == 0
    assert gate.blocked_reasons == ("actuation_mapping_incomplete",)


def test_gate_audit_record_copies_command_mappings() -> None:
    """Audit records copy commands without enabling runtime actuation."""
    gate = validate_stl_runtime_actuation_gate(
        _plan(_action()),
        (_template(),),
    )

    record = gate.to_audit_record()
    commands = cast("list[dict[str, object]]", record["commands"])
    commands[0]["value"] = 0.0

    assert gate.accepted is True
    assert gate.non_actuating is True
    assert gate.execution_disabled is True
    assert gate.commands[0]["value"] == 0.7
    assert record["blocked_reasons"] == []
