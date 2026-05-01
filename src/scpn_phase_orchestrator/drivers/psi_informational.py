# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Informational Psi driver

from __future__ import annotations

from math import isfinite
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import TWO_PI

__all__ = ["InformationalDriver"]

FloatArray: TypeAlias = NDArray[np.float64]


class InformationalDriver:
    """External drive Psi_I(t) = 2*pi*cadence_hz*t (mod 2*pi)."""

    def __init__(self, cadence_hz: float):
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
        return (TWO_PI * self._cadence_hz * t) % TWO_PI

    def compute_batch(self, t_array: FloatArray) -> FloatArray:
        """Vectorised Psi_I over an array of time values."""
        result: FloatArray = (TWO_PI * self._cadence_hz * t_array) % TWO_PI
        return result
