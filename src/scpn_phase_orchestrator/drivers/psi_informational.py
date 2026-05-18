# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Informational Psi driver

from __future__ import annotations

from math import isfinite
from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import TWO_PI

__all__ = ["InformationalDriver"]

FloatArray: TypeAlias = NDArray[np.float64]


def _require_finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite")
    parsed = float(value)
    if not isfinite(parsed):
        raise ValueError(f"{name} must be finite, got {value}")
    return parsed


def _require_finite_real_array(value: object, *, name: str) -> FloatArray:
    array = np.asarray(value)
    dtype = array.dtype
    if (
        np.issubdtype(dtype, np.bool_)
        or np.issubdtype(dtype, np.complexfloating)
        or not np.issubdtype(dtype, np.number)
    ):
        raise ValueError(f"{name} must be finite")
    parsed = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(parsed)):
        raise ValueError(f"{name} must be finite")
    return parsed


class InformationalDriver:
    """External drive Psi_I(t) = 2*pi*cadence_hz*t (mod 2*pi)."""

    def __init__(self, cadence_hz: float):
        if isinstance(cadence_hz, bool):
            raise ValueError("cadence_hz must be finite and positive")
        try:
            parsed_cadence = float(cadence_hz)
        except (TypeError, ValueError) as exc:
            raise ValueError("cadence_hz must be finite and positive") from exc
        if not isfinite(parsed_cadence) or parsed_cadence <= 0.0:
            raise ValueError(
                f"cadence_hz must be finite and positive, got {cadence_hz}"
            )
        self._cadence_hz = parsed_cadence

    def compute(self, t: float) -> float:
        """Return Psi_I at time *t*, wrapped to [0, 2*pi)."""
        t = _require_finite_real(t, name="t")
        return (TWO_PI * self._cadence_hz * t) % TWO_PI

    def compute_batch(self, t_array: FloatArray) -> FloatArray:
        """Vectorised Psi_I over an array of time values."""
        t_array = _require_finite_real_array(t_array, name="t_array")
        result: FloatArray = (TWO_PI * self._cadence_hz * t_array) % TWO_PI
        return result
