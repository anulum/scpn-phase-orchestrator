# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Coupling lag functions

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["LagModel"]


class LagModel:
    """Phase-lag estimation and alpha matrix construction."""

    @staticmethod
    def estimate_from_distances(distances: NDArray, speed: float) -> NDArray:
        """Build antisymmetric alpha matrix from pairwise distances and speed.

        alpha[i,j] = 2*pi * distances[i,j] / speed.
        Matches the Rust ``LagModel::estimate_from_distances`` algorithm.
        """
        n = distances.shape[0]
        alpha = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                lag = 2.0 * np.pi * distances[i, j] / speed
                alpha[i, j] = lag
                alpha[j, i] = -lag
        return alpha

    def estimate_lag(
        self, signal_a: NDArray, signal_b: NDArray, sample_rate: float
    ) -> float:
        """Cross-correlation peak lag in seconds between two signals."""
        corr = np.correlate(
            signal_a - signal_a.mean(), signal_b - signal_b.mean(), mode="full"
        )
        peak_idx = np.argmax(corr)
        lag_samples = peak_idx - (len(signal_a) - 1)
        return float(lag_samples / sample_rate)

    def build_alpha_matrix(
        self,
        lag_estimates: dict[tuple[int, int], float],
        n_layers: int,
        carrier_freq_hz: float = 1.0,
    ) -> NDArray:
        """Pairwise lag estimates (seconds) to phase-offset matrix (radians).

        alpha[i,j] = 2*pi*carrier_freq_hz*lag[i,j], antisymmetric.
        ``carrier_freq_hz`` defaults to 1.0 for backward compatibility.
        """
        alpha = np.zeros((n_layers, n_layers), dtype=np.float64)
        for (i, j), lag in lag_estimates.items():
            offset = 2.0 * np.pi * carrier_freq_hz * lag
            alpha[i, j] = offset
            alpha[j, i] = -offset
        return alpha
