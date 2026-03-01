# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

from dataclasses import replace

from scpn_phase_orchestrator.actuation.mapper import ControlAction


class ActionProjector:
    """Clip control actions to value bounds and rate limits."""

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
