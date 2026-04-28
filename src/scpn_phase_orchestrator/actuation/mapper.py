# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Actuation mapper

from __future__ import annotations

from dataclasses import dataclass

from scpn_phase_orchestrator.binding.types import VALID_KNOBS, ActuatorMapping

__all__ = ["ControlAction", "ActuationMapper"]


@dataclass
class ControlAction:
    """A single control command targeting a specific knob and scope."""

    knob: str  # K, alpha, zeta, or Psi
    scope: str  # "global" or "layer_{n}"
    value: float
    ttl_s: float
    justification: str


class ActuationMapper:
    """Convert ControlActions to actuator-specific command dicts."""

    def __init__(self, actuator_mappings: list[ActuatorMapping]):
        self._by_knob: dict[str, list[ActuatorMapping]] = {}
        for am in actuator_mappings:
            self._by_knob.setdefault(am.knob, []).append(am)

    def map_actions(self, actions: list[ControlAction]) -> list[dict]:
        """Convert ControlActions into actuator command dicts, clamping to limits."""
        commands = []
        for action in actions:
            mappings = self._by_knob.get(action.knob, [])
            for am in mappings:
                if am.scope == action.scope or action.scope == "global":
                    commands.append(
                        {
                            "actuator": am.name,
                            "knob": action.knob,
                            "scope": action.scope,
                            "value": max(am.limits[0], min(action.value, am.limits[1])),
                            "ttl_s": action.ttl_s,
                        }
                    )
        return commands

    def validate_action(self, action: ControlAction) -> bool:
        """Return True if knob is valid and value is within limits."""
        if action.knob not in VALID_KNOBS:
            return False
        mappings = self._by_knob.get(action.knob, [])
        for am in mappings:
            if am.scope == action.scope or action.scope == "global":
                lo, hi = am.limits
                if lo <= action.value <= hi:
                    return True
        return False
