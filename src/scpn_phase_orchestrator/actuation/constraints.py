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

from collections.abc import Iterable
from dataclasses import replace
from math import isfinite
from numbers import Real

from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.binding.types import ActuatorMapping

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
                raise ValueError(
                    f"rate-limit knob name must be non-empty str, got {knob!r}"
                )
            if isinstance(limit, bool) or not isinstance(limit, Real):
                raise TypeError(
                    f"rate limit for {knob!r} must be finite real, got {limit!r}"
                )
            if not isfinite(float(limit)) or float(limit) < 0.0:
                raise ValueError(
                    f"rate limit for {knob!r} must be finite >= 0, got {limit!r}"
                )
        for knob, bounds in value_bounds.items():
            if not isinstance(knob, str) or not knob.strip():
                raise ValueError(
                    f"value-bound knob name must be non-empty str, got {knob!r}"
                )
            if not isinstance(bounds, tuple) or len(bounds) != 2:
                raise TypeError(
                    f"value bounds for {knob!r} must be a 2-tuple "
                    f"(lo, hi), got {bounds!r}"
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

    @classmethod
    def from_actuator_mappings(
        cls,
        actuators: Iterable[ActuatorMapping],
    ) -> ActionProjector:
        """Build projector bounds and slew limits from binding-spec actuators.

        `ActionProjector` is knob-indexed. A binding that maps the same knob to
        multiple actuator records must therefore provide identical limits and
        identical `rate_limit_per_step` values for those records; otherwise the
        binding is ambiguous and projection fails closed.
        """
        rate_limits: dict[str, float] = {}
        value_bounds: dict[str, tuple[float, float]] = {}
        for actuator in actuators:
            if not isinstance(actuator, ActuatorMapping):
                raise TypeError(
                    "actuators must contain ActuatorMapping instances, "
                    f"got {actuator!r}"
                )
            bounds = (float(actuator.limits[0]), float(actuator.limits[1]))
            existing_bounds = value_bounds.get(actuator.knob)
            if existing_bounds is not None and existing_bounds != bounds:
                raise ValueError(
                    f"conflicting value bounds for actuator knob {actuator.knob!r}"
                )
            value_bounds[actuator.knob] = bounds
            if actuator.rate_limit_per_step is None:
                continue
            rate_limit = float(actuator.rate_limit_per_step)
            existing_rate = rate_limits.get(actuator.knob)
            if existing_rate is not None and existing_rate != rate_limit:
                raise ValueError(
                    f"conflicting rate limits for actuator knob {actuator.knob!r}"
                )
            rate_limits[actuator.knob] = rate_limit
        return cls(rate_limits=rate_limits, value_bounds=value_bounds)

    def project(self, action: ControlAction, previous_value: float) -> ControlAction:
        """Clamp action value to bounds and rate limit relative to *previous_value*."""
        if not isinstance(action, ControlAction):
            raise TypeError(f"action must be ControlAction, got {action!r}")
        if not isfinite(float(action.value)):
            raise ValueError(f"action.value must be finite real, got {action.value!r}")
        if isinstance(previous_value, bool) or not isinstance(previous_value, Real):
            raise TypeError(
                f"previous_value must be a finite real scalar, got {previous_value!r}"
            )
        if not isfinite(float(previous_value)):
            raise ValueError(
                f"previous_value must be a finite real scalar, got {previous_value!r}"
            )
        lo, hi = self._value_bounds.get(action.knob, (float("-inf"), float("inf")))
        clamped = max(lo, min(action.value, hi))

        rate_limit = self._rate_limits.get(action.knob)
        if rate_limit is not None:
            delta = clamped - previous_value
            if abs(delta) > rate_limit:
                clamped = previous_value + rate_limit * (1.0 if delta > 0 else -1.0)
            clamped = max(lo, min(clamped, hi))

        return replace(action, value=clamped)
