# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Physical Psi driver

from __future__ import annotations

from math import isfinite

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import TWO_PI

__all__ = ["PhysicalDriver"]


class PhysicalDriver:
    """Sinusoidal external drive: Psi_P(t) = amplitude * sin(2*pi*frequency*t)."""

    def __init__(self, frequency: float, amplitude: float = 1.0):
        try:
            parsed_frequency = float(frequency)
            parsed_amplitude = float(amplitude)
        except (TypeError, ValueError) as exc:
            raise ValueError("frequency and amplitude must be finite numbers") from exc
        if not isfinite(parsed_frequency) or parsed_frequency <= 0.0:
            raise ValueError(f"frequency must be finite and positive, got {frequency}")
        if not isfinite(parsed_amplitude) or parsed_amplitude < 0.0:
            raise ValueError(
                f"amplitude must be finite and non-negative, got {amplitude}"
            )
        self._frequency = parsed_frequency
        self._amplitude = parsed_amplitude

    def compute(self, t: float) -> float:
        """Return Psi_P at time *t*."""
        return float(self._amplitude * np.sin(TWO_PI * self._frequency * t))

    def compute_batch(self, t_array: NDArray) -> NDArray:
        """Vectorised Psi_P over an array of time values."""
        result: NDArray = self._amplitude * np.sin(TWO_PI * self._frequency * t_array)
        return result
