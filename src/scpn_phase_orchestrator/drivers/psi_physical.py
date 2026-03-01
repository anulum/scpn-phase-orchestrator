# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

TWO_PI = 2.0 * np.pi


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
