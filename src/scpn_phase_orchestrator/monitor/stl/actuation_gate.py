# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — STL runtime actuation gate

"""Runtime actuation gate validation for projected STL control actions."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from scpn_phase_orchestrator.actuation.mapper import ActuationMapper
from scpn_phase_orchestrator.binding.types import ActuatorMapping

from .projection import STLActionProjectionTemplate, STLProjectedActionPlan


@dataclass(frozen=True)
class STLRuntimeActuationGate:
    """Non-actuating runtime-stack validation of projected STL actions.

    The gate verifies projected proposals against the same actuator mapping
    boundary used by runtime actuation, but it never enables execution. This
    makes the closed-loop STL plan auditable through the safety/actuation stack
    without converting a review artefact into a live controller command.
    """

    spec: str
    non_actuating: bool
    execution_disabled: bool
    accepted: bool
    action_count: int
    mapper_valid_action_count: int
    mapped_command_count: int
    commands: tuple[dict[str, object], ...]
    blocked_reasons: tuple[str, ...]

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-serialisable runtime gate record.

        Returns
        -------
        dict[str, object]
            Return a JSON-serialisable runtime gate record.
        """
        return {
            "spec": self.spec,
            "non_actuating": self.non_actuating,
            "execution_disabled": self.execution_disabled,
            "accepted": self.accepted,
            "action_count": self.action_count,
            "mapper_valid_action_count": self.mapper_valid_action_count,
            "mapped_command_count": self.mapped_command_count,
            "commands": [dict(command) for command in self.commands],
            "blocked_reasons": list(self.blocked_reasons),
        }


def validate_stl_runtime_actuation_gate(
    projected_plan: STLProjectedActionPlan,
    templates: Sequence[STLActionProjectionTemplate],
) -> STLRuntimeActuationGate:
    """Validate projected STL actions through runtime actuation mapping.

    This is an audit gate only: returned commands are deterministic evidence
    that proposals can be represented by the configured actuation stack, while
    ``execution_disabled`` and ``non_actuating`` remain true for every outcome.
    Invalid runtime knobs, missing mappings, and empty projected plans fail
    closed with explicit blocker reasons.

    Parameters
    ----------
    projected_plan : STLProjectedActionPlan
        The projected STL action plan to validate.
    templates : Sequence[STLActionProjectionTemplate]
        STL action-projection templates.

    Returns
    -------
    STLRuntimeActuationGate
        The runtime actuation-gate validation result.
    """
    actions = projected_plan.approved_actions
    if not actions:
        return STLRuntimeActuationGate(
            spec=projected_plan.spec,
            non_actuating=True,
            execution_disabled=True,
            accepted=False,
            action_count=0,
            mapper_valid_action_count=0,
            mapped_command_count=0,
            commands=(),
            blocked_reasons=("no_runtime_actions",),
        )

    templates_by_surface = {
        (template.knob, template.scope): template for template in templates
    }
    mappings: list[ActuatorMapping] = []
    blocked_reasons: list[str] = []
    for action in actions:
        template = templates_by_surface.get((action.knob, action.scope))
        if template is None:
            blocked_reasons.append("actuation_template_missing")
            continue
        try:
            mappings.append(
                ActuatorMapping(
                    name=_runtime_actuator_name(template.knob, template.scope),
                    knob=template.knob,
                    scope=template.scope,
                    limits=template.value_bounds,
                    rate_limit_per_step=template.rate_limit,
                )
            )
        except (TypeError, ValueError):
            blocked_reasons.append("actuation_mapper_rejected_template")

    if not mappings:
        return STLRuntimeActuationGate(
            spec=projected_plan.spec,
            non_actuating=True,
            execution_disabled=True,
            accepted=False,
            action_count=len(actions),
            mapper_valid_action_count=0,
            mapped_command_count=0,
            commands=(),
            blocked_reasons=tuple(dict.fromkeys(blocked_reasons)),
        )

    try:
        mapper = ActuationMapper(mappings)
    except ValueError:
        return STLRuntimeActuationGate(
            spec=projected_plan.spec,
            non_actuating=True,
            execution_disabled=True,
            accepted=False,
            action_count=len(actions),
            mapper_valid_action_count=0,
            mapped_command_count=0,
            commands=(),
            blocked_reasons=("actuation_mapper_rejected_template",),
        )

    valid_actions = tuple(
        action for action in actions if mapper.validate_action(action)
    )
    if len(valid_actions) != len(actions):
        blocked_reasons.append("runtime_action_validation_failed")
    commands = tuple(
        _normalise_runtime_command(command)
        for command in mapper.map_actions(list(valid_actions))
    )
    if len(commands) != len(valid_actions):
        blocked_reasons.append("actuation_mapping_incomplete")
    accepted = (
        len(valid_actions) == len(actions)
        and len(commands) == len(actions)
        and not blocked_reasons
    )
    return STLRuntimeActuationGate(
        spec=projected_plan.spec,
        non_actuating=True,
        execution_disabled=True,
        accepted=accepted,
        action_count=len(actions),
        mapper_valid_action_count=len(valid_actions),
        mapped_command_count=len(commands),
        commands=commands,
        blocked_reasons=tuple(dict.fromkeys(blocked_reasons)),
    )


def _normalise_runtime_command(command: dict[str, object]) -> dict[str, object]:
    return {
        "actuator": command["actuator"],
        "knob": command["knob"],
        "scope": command["scope"],
        "value": command["value"],
        "ttl_s": command["ttl_s"],
    }


def _runtime_actuator_name(knob: str, scope: str) -> str:
    surface = re.sub(r"[^A-Za-z0-9_]+", "_", f"{knob}_{scope}").strip("_")
    return f"stl_runtime_{surface or 'action'}"
