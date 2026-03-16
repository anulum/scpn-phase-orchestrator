# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Actuation constraints

from __future__ import annotations

from dataclasses import replace

from scpn_phase_orchestrator.actuation.mapper import ControlAction

__all__ = ["ActionProjector"]


class ActionProjector:
    """Clip control actions to value bounds and rate limits.

    Rate limits and value bounds are empirical — see docs/ASSUMPTIONS.md § Rate Limits.
    """

    def __init__(
        self,
        rate_limits: dict[str, float],
        value_bounds: dict[str, tuple[float, float]],
    ):
        self._rate_limits = rate_limits
        self._value_bounds = value_bounds

    def project(self, action: ControlAction, previous_value: float) -> ControlAction:
        lo, hi = self._value_bounds.get(action.knob, (float("-inf"), float("inf")))
        clamped = max(lo, min(action.value, hi))

        rate_limit = self._rate_limits.get(action.knob)
        if rate_limit is not None:
            delta = clamped - previous_value
            if abs(delta) > rate_limit:
                clamped = previous_value + rate_limit * (1.0 if delta > 0 else -1.0)
            clamped = max(lo, min(clamped, hi))

        return replace(action, value=clamped)
