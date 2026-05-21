# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Boundary observer

"""Boundary observer utilities for compartment and event-bus safety checks.

The observer evaluates declared soft and hard partitions against runtime state
without mutating the monitored values. Missing state variables are ignored so
partially observed deployments can still emit useful diagnostics, while unknown
severity policy is treated as a fail-hard configuration error before monitoring
starts. Events are emitted through the supplied bus only after checks are
classified, preserving a clear separation between detection and actuation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from math import isfinite
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
        if not isinstance(boundary_defs, list):
            raise TypeError("boundary_defs must be a list[BoundaryDef]")
        for idx, bdef in enumerate(boundary_defs):
            if not isinstance(bdef, BoundaryDef):
                raise TypeError(f"boundary_defs[{idx}] must be BoundaryDef, got {bdef!r}")
            if not bdef.name.strip() or not bdef.variable.strip():
                raise ValueError(
                    f"boundary_defs[{idx}] requires non-empty name and variable"
                )
            if bdef.lower is not None and not isfinite(float(bdef.lower)):
                raise ValueError(
                    f"boundary_defs[{idx}] lower bound must be finite, got {bdef.lower!r}"
                )
            if bdef.upper is not None and not isfinite(float(bdef.upper)):
                raise ValueError(
                    f"boundary_defs[{idx}] upper bound must be finite, got {bdef.upper!r}"
                )
            if (
                bdef.lower is not None
                and bdef.upper is not None
                and float(bdef.lower) > float(bdef.upper)
            ):
                raise ValueError(
                    f"boundary_defs[{idx}] requires lower <= upper, got {bdef.lower!r}>{bdef.upper!r}"
                )
        self._defs = boundary_defs
        self._event_bus: EventBus | None = None
        self._step = 0

    def set_event_bus(self, event_bus: EventBus) -> None:
        """Attach an event bus for posting boundary_breach events."""
        from scpn_phase_orchestrator.supervisor.events import EventBus as _EventBus

        if not isinstance(event_bus, _EventBus):
            raise TypeError(f"event_bus must be EventBus, got {event_bus!r}")
        self._event_bus = event_bus

    def observe(
        self, values: dict[str, float], *, step: int | None = None
    ) -> BoundaryState:
        """Evaluate scalar measurements against configured boundaries.

        Parameters
        ----------
        values
            Mapping from monitored variable name to the current scalar
            measurement.
        step
            Optional supervisor step attached to any posted
            ``boundary_breach`` event. When omitted, the observer reuses
            its previous step counter.

        Returns
        -------
        BoundaryState
            Partitioned violation snapshot containing all violations plus
            soft and hard subsets.

        Notes
        -----
        Missing variables are ignored. Unknown severities are logged and
        treated as hard violations so safety-critical callers fail closed.
        """
        if not isinstance(values, dict):
            raise TypeError(f"values must be dict[str, float], got {values!r}")
        if step is not None:
            if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                raise ValueError(f"step must be a non-negative integer, got {step!r}")
            self._step = step
        state = BoundaryState()
        for name, value in values.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"value keys must be non-empty strings, got {name!r}")
            if isinstance(value, bool):
                raise ValueError(f"values[{name!r}] must be finite float, got {value!r}")
            if not isfinite(float(value)):
                raise ValueError(f"values[{name!r}] must be finite float, got {value!r}")
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
