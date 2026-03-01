# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

from dataclasses import dataclass, field

from scpn_phase_orchestrator.binding.types import BoundaryDef


@dataclass
class BoundaryState:
    violations: list[str] = field(default_factory=list)
    soft_warnings: list[str] = field(default_factory=list)
    hard_violations: list[str] = field(default_factory=list)


class BoundaryObserver:
    """Check measured values against boundary definitions."""

    def __init__(self, boundary_defs: list[BoundaryDef]):
        self._defs = boundary_defs

    def observe(self, values: dict[str, float]) -> BoundaryState:
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
                state.soft_warnings.append(msg)
            else:
                state.hard_violations.append(msg)

        return state
