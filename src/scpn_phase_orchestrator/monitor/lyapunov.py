# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Lyapunov stability monitor

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["LyapunovGuard", "LyapunovState"]


@dataclass
class LyapunovState:
    V: float
    dV_dt: float
    in_basin: bool
    max_phase_diff: float


class LyapunovGuard:
    """Lyapunov stability monitor for Kuramoto networks.

    V(θ) = -(K/2N) Σ_{i,j} A_ij cos(θ_i - θ_j)

    dV/dt ≤ 0 for gradient flow (Kuramoto is gradient on V).
    Basin of attraction: max|θ_i - θ_j| < π/2 for connected pairs.

    van Hemmen & Wreszinski 1993, J. Stat. Phys. 72:145-166.
    """

    def __init__(self, basin_threshold: float = np.pi / 2):
        self._basin_threshold = basin_threshold
        self._prev_V: float | None = None

    def evaluate(self, phases: NDArray, knm: NDArray) -> LyapunovState:
        """Compute Lyapunov function, its time derivative, and basin check."""
        n = len(phases)
        if n == 0:
            return LyapunovState(V=0.0, dV_dt=0.0, in_basin=True, max_phase_diff=0.0)

        diff = phases[:, np.newaxis] - phases[np.newaxis, :]
        cos_diff = np.cos(diff)

        # V(θ) = -(1/2N) Σ K_ij cos(θ_i - θ_j)
        V = -0.5 * float(np.sum(knm * cos_diff)) / n

        # Numerical dV/dt from consecutive calls
        dV_dt = 0.0
        if self._prev_V is not None:
            dV_dt = V - self._prev_V
        self._prev_V = V

        # Basin check: max phase difference between connected pairs
        connected = knm > 0
        if np.any(connected):
            abs_diff = np.abs(diff)
            # Wrap to [-π, π]
            abs_diff = np.minimum(abs_diff, 2 * np.pi - abs_diff)
            max_diff = float(np.max(abs_diff[connected]))
        else:
            max_diff = 0.0

        in_basin = max_diff < self._basin_threshold

        return LyapunovState(
            V=V,
            dV_dt=dV_dt,
            in_basin=in_basin,
            max_phase_diff=max_diff,
        )

    def reset(self) -> None:
        self._prev_V = None
