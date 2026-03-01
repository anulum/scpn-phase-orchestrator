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


class InformationalDriver:
    """External drive Psi_I(t) = 2*pi*cadence_hz*t (mod 2*pi)."""

    def __init__(self, cadence_hz: float):
        if cadence_hz <= 0.0:
            raise ValueError(f"cadence_hz must be positive, got {cadence_hz}")
        self._cadence_hz = cadence_hz

    def compute(self, t: float) -> float:
        return (TWO_PI * self._cadence_hz * t) % TWO_PI

    def compute_batch(self, t_array: NDArray) -> NDArray:
        return (TWO_PI * self._cadence_hz * t_array) % TWO_PI
