# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Physical Psi driver

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import TWO_PI

__all__ = ["PhysicalDriver"]


class PhysicalDriver:
    """Sinusoidal external drive: Psi_P(t) = amplitude * sin(2*pi*frequency*t)."""

    def __init__(self, frequency: float, amplitude: float = 1.0):
        if frequency <= 0.0:
            raise ValueError(f"frequency must be positive, got {frequency}")
        self._frequency = frequency
        self._amplitude = amplitude

    def compute(self, t: float) -> float:
        return float(self._amplitude * np.sin(TWO_PI * self._frequency * t))

    def compute_batch(self, t_array: NDArray) -> NDArray:
        result: NDArray = self._amplitude * np.sin(TWO_PI * self._frequency * t_array)
        return result
