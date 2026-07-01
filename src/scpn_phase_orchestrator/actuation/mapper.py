# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Actuation mapper

"""Map validated control actions onto configured actuator records.

The mapper is deliberately data-only: it validates binding-level actuator
metadata, clamps finite action values to each actuator limit, and returns
command dictionaries for a transport or hardware layer to consume. Invalid
action values are not sent onward, and invalid mapping definitions fail at
construction time.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any

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
            if not isinstance(am, ActuatorMapping):
                raise ValueError("actuator_mappings entries must be ActuatorMapping")
            _validate_mapping(am)
            self._by_knob.setdefault(am.knob, []).append(am)

    def map_actions(self, actions: list[ControlAction]) -> list[dict[str, Any]]:
        """Convert ControlActions into actuator command dicts, clamping to limits.

        Parameters
        ----------
        actions : list[ControlAction]
            The control actions.

        Returns
        -------
        list[dict[str, Any]]
            One command dict (``actuator``, ``knob``, ``scope``, ``value``,
            ``ttl_s``) per matching actuator, with the value clamped to that
            actuator's limits; actions with a non-finite value are dropped.
        """
        commands = []
        for action in actions:
            if not _finite_real(action.value):
                continue
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
        """Return True if knob is valid and value is within limits.

        Parameters
        ----------
        action : ControlAction
            The control action.

        Returns
        -------
        bool
            True if knob is valid and value is within limits.
        """
        if action.knob not in VALID_KNOBS:
            return False
        if not _finite_real(action.value):
            return False
        mappings = self._by_knob.get(action.knob, [])
        for am in mappings:
            if am.scope == action.scope or action.scope == "global":
                lo, hi = am.limits
                if lo <= action.value <= hi:
                    return True
        return False


def _validate_mapping(mapping: ActuatorMapping) -> None:
    """Validate an actuator mapping in place, raising ``ValueError`` if malformed."""
    if mapping.knob not in VALID_KNOBS:
        raise ValueError("actuator mapping knob must be a valid control knob")
    if not isinstance(mapping.scope, str) or not mapping.scope.strip():
        raise ValueError("actuator mapping scope must be a non-empty string")
    if len(mapping.limits) != 2:
        raise ValueError("actuator mapping limits must contain two values")
    lo, hi = mapping.limits
    if not _finite_real(lo) or not _finite_real(hi) or lo >= hi:
        raise ValueError("actuator mapping limits must be finite and increasing")


def _finite_real(value: object) -> bool:
    """Return whether ``value`` is a finite real scalar (booleans excluded)."""
    return isinstance(value, Real) and not isinstance(value, bool) and isfinite(value)
