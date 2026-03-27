# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Symbolic Psi driver

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
        """Return symbolic phase at discrete *step* (cyclic)."""
        return float(self._sequence[step % self._n])

    def compute_batch(self, steps: NDArray) -> NDArray:
        """Vectorised symbolic phase lookup over an array of step indices."""
        result: NDArray = self._sequence[steps.astype(int) % self._n]
        return result
