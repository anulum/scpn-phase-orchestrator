# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class LagModel:
    """Estimates phase lags between signal pairs via cross-correlation."""

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
        self, lag_estimates: dict[tuple[int, int], float], n_layers: int
    ) -> NDArray:
        """Pairwise lag estimates (seconds) to phase-offset matrix (radians).

        Assumes lag in fractions of nominal cycle period.
        alpha[i,j] = 2*pi*lag[i,j], antisymmetric.
        """
        alpha = np.zeros((n_layers, n_layers), dtype=np.float64)
        for (i, j), lag in lag_estimates.items():
            alpha[i, j] = 2.0 * np.pi * lag
            alpha[j, i] = -2.0 * np.pi * lag
        return alpha
