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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scpn_phase_orchestrator.binding.types import ActuatorMapping

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
        from scpn_phase_orchestrator.binding.types import ActuatorMapping

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
            actuator's limits; actions with a non-finite value, or a TTL that
            is not a finite, non-negative real, are dropped.
        """
        commands = []
        for action in actions:
            if not _finite_real(action.value) or not _valid_ttl(action.ttl_s):
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
        """Return True if knob, TTL and value are valid for a mapped actuator.

        The TTL must be a finite, non-negative real number of seconds, the
        contract the policy validators already apply.

        Parameters
        ----------
        action : ControlAction
            The control action.

        Returns
        -------
        bool
            True if the knob is valid, the TTL is admissible and the value is
            within the limits of a matching actuator.
        """
        from scpn_phase_orchestrator.binding.types import VALID_KNOBS

        if action.knob not in VALID_KNOBS:
            return False
        if not _finite_real(action.value) or not _valid_ttl(action.ttl_s):
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
    from scpn_phase_orchestrator.binding.types import VALID_KNOBS

    if mapping.knob not in VALID_KNOBS:
        raise ValueError("actuator mapping knob must be a valid control knob")
    if not isinstance(mapping.scope, str) or not mapping.scope.strip():
        raise ValueError("actuator mapping scope must be a non-empty string")
    # ActuatorMapping unpacks ``limits`` into two values when it is built, and
    # this unpacking raises ValueError for any other length.
    lo, hi = mapping.limits
    if not _finite_real(lo) or not _finite_real(hi) or lo >= hi:
        raise ValueError("actuator mapping limits must be finite and increasing")


def _valid_ttl(value: object) -> bool:
    """Return whether ``value`` is a finite, non-negative real TTL in seconds.

    Plain ``float`` and ``int`` take an exact-type fast path, because this runs
    once per action in ``map_actions``; other reals go through the ABC check.
    """
    if type(value) is float or type(value) is int:
        return isfinite(value) and value >= 0
    return isinstance(value, Real) and _finite_real(value) and float(value) >= 0.0


def _finite_real(value: object) -> bool:
    """Return whether ``value`` is a finite real scalar (booleans excluded)."""
    return isinstance(value, Real) and not isinstance(value, bool) and isfinite(value)
