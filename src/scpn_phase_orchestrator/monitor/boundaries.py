# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Boundary observer

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from scpn_phase_orchestrator.binding.types import BoundaryDef

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from scpn_phase_orchestrator.supervisor.events import EventBus

__all__ = ["BoundaryState", "BoundaryObserver"]


@dataclass
class BoundaryState:
    """Snapshot of boundary violations partitioned by severity."""

    violations: list[str] = field(default_factory=list)
    soft_violations: list[str] = field(default_factory=list)
    hard_violations: list[str] = field(default_factory=list)


class BoundaryObserver:
    """Check measured values against boundary definitions."""

    def __init__(self, boundary_defs: list[BoundaryDef]):
        self._defs = boundary_defs
        self._event_bus: EventBus | None = None
        self._step = 0

    def set_event_bus(self, event_bus: EventBus) -> None:
        """Attach an event bus for posting boundary_breach events."""
        self._event_bus = event_bus

    def observe(
        self, values: dict[str, float], *, step: int | None = None
    ) -> BoundaryState:
        """Check *values* against all boundary definitions, return violations."""
        if step is not None:
            self._step = step
        state = BoundaryState()
        for bdef in self._defs:
            val = values.get(bdef.variable)
            if val is None:
                continue

            violated = False
            if bdef.lower is not None and val < bdef.lower:
                violated = True
            if bdef.upper is not None and val > bdef.upper:
                violated = True

            if not violated:
                continue

            msg = (
                f"{bdef.name}: {bdef.variable}={val:.4g} "
                f"outside [{bdef.lower}, {bdef.upper}]"
            )
            state.violations.append(msg)
            if bdef.severity == "soft":
                state.soft_violations.append(msg)
            elif bdef.severity == "hard":
                state.hard_violations.append(msg)
            else:
                logger.warning(
                    "unknown severity %r on %s, treating as hard",
                    bdef.severity,
                    bdef.name,
                )
                state.hard_violations.append(msg)

        if state.violations and self._event_bus is not None:
            from scpn_phase_orchestrator.supervisor.events import RegimeEvent

            self._event_bus.post(
                RegimeEvent(
                    kind="boundary_breach",
                    step=self._step,
                    detail="; ".join(state.violations),
                )
            )

        return state
