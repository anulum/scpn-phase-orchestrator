# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["SymbolicDriver"]


class SymbolicDriver:
    """Deterministic phase sequence driver for symbolic/semiotic channels."""

    def __init__(self, sequence: list[float]):
        if not sequence:
            raise ValueError("sequence must be non-empty")
        self._sequence = np.asarray(sequence, dtype=np.float64)
        self._n = len(sequence)

    def compute(self, step: int) -> float:
        return float(self._sequence[step % self._n])

    def compute_batch(self, steps: NDArray) -> NDArray:
        result: NDArray = self._sequence[steps.astype(int) % self._n]
        return result
