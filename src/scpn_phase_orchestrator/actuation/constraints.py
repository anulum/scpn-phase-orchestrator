# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Actuation constraints

"""Projection constraints for bounded supervisor control actions.

`ActionProjector` is the last deterministic clamp before a control proposal is
handed to actuator mapping. It preserves the requested knob/scope metadata and
only changes the scalar value, first applying configured absolute bounds and
then per-step rate limits relative to the previous actuator value.
"""

from __future__ import annotations

from dataclasses import replace
from math import isfinite
from numbers import Real

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
        if not isinstance(rate_limits, dict):
            raise TypeError("rate_limits must be a dict[str, float]")
        if not isinstance(value_bounds, dict):
            raise TypeError("value_bounds must be a dict[str, tuple[float, float]]")
        for knob, limit in rate_limits.items():
            if not isinstance(knob, str) or not knob.strip():
                raise ValueError(f"rate-limit knob name must be non-empty str, got {knob!r}")
            if isinstance(limit, bool) or not isinstance(limit, Real):
                raise TypeError(f"rate limit for {knob!r} must be finite real, got {limit!r}")
            if not isfinite(float(limit)) or float(limit) < 0.0:
                raise ValueError(f"rate limit for {knob!r} must be finite >= 0, got {limit!r}")
        for knob, bounds in value_bounds.items():
            if not isinstance(knob, str) or not knob.strip():
                raise ValueError(f"value-bound knob name must be non-empty str, got {knob!r}")
            if not isinstance(bounds, tuple) or len(bounds) != 2:
                raise TypeError(
                    f"value bounds for {knob!r} must be a 2-tuple (lo, hi), got {bounds!r}"
                )
            lo, hi = bounds
            if any(isinstance(v, bool) or not isinstance(v, Real) for v in (lo, hi)):
                raise TypeError(
                    f"value bounds for {knob!r} must be finite reals, got {bounds!r}"
                )
            lo_f = float(lo)
            hi_f = float(hi)
            if not isfinite(lo_f) or not isfinite(hi_f):
                raise ValueError(
                    f"value bounds for {knob!r} must be finite reals, got {bounds!r}"
                )
            if lo_f > hi_f:
                raise ValueError(
                    f"value bounds for {knob!r} require lo <= hi, got {bounds!r}"
                )
        self._rate_limits = rate_limits
        self._value_bounds = value_bounds

    def project(self, action: ControlAction, previous_value: float) -> ControlAction:
        """Clamp action value to bounds and rate limit relative to *previous_value*."""
        lo, hi = self._value_bounds.get(action.knob, (float("-inf"), float("inf")))
        clamped = max(lo, min(action.value, hi))

        rate_limit = self._rate_limits.get(action.knob)
        if rate_limit is not None:
            delta = clamped - previous_value
            if abs(delta) > rate_limit:
                clamped = previous_value + rate_limit * (1.0 if delta > 0 else -1.0)
            clamped = max(lo, min(clamped, hi))

        return replace(action, value=clamped)
