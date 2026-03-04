# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["modulation_index", "pac_matrix", "pac_gate"]


def modulation_index(theta_low: NDArray, amp_high: NDArray, n_bins: int = 18) -> float:
    """Phase-amplitude coupling via Tort et al. 2010, J. Neurophysiol.

    Bins amplitude by phase, computes KL divergence from uniform.
    Returns MI ∈ [0, 1], normalised by log(n_bins).
    """
    if theta_low.size == 0 or amp_high.size == 0:
        return 0.0
    n = min(theta_low.size, amp_high.size)
    theta = theta_low[:n] % (2.0 * np.pi)
    amp = amp_high[:n]

    bin_edges = np.linspace(0.0, 2.0 * np.pi, n_bins + 1)
    bin_indices = np.digitize(theta, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    mean_amp = np.zeros(n_bins, dtype=np.float64)
    for k in range(n_bins):
        mask = bin_indices == k
        if np.any(mask):
            mean_amp[k] = np.mean(amp[mask])

    total = mean_amp.sum()
    if total <= 0.0:
        return 0.0

    p = mean_amp / total
    # KL divergence from uniform q = 1/n_bins
    # D_KL = Σ p_k log(p_k / q_k) = Σ p_k log(p_k * n_bins)
    log_n = np.log(n_bins)
    kl = 0.0
    for pk in p:
        if pk > 0.0:
            kl += pk * np.log(pk * n_bins)

    # MI normalised to [0, 1]
    mi = kl / log_n
    return float(np.clip(mi, 0.0, 1.0))


def pac_matrix(
    phases_history: NDArray,
    amplitudes_history: NDArray,
    n_bins: int = 18,
) -> NDArray:
    """N×N PAC matrix: entry [i,j] = MI(phase_i, amplitude_j).

    Args:
        phases_history: (T, N) phase time series
        amplitudes_history: (T, N) amplitude time series
        n_bins: number of phase bins

    Returns:
        (N, N) modulation index matrix
    """
    if phases_history.ndim != 2 or amplitudes_history.ndim != 2:
        raise ValueError("phases_history and amplitudes_history must be 2-D")
    n = phases_history.shape[1]
    if amplitudes_history.shape[1] != n:
        raise ValueError("phases and amplitudes must have same number of oscillators")
    result = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            result[i, j] = modulation_index(
                phases_history[:, i], amplitudes_history[:, j], n_bins
            )
    return result


def pac_gate(pac_value: float, threshold: float = 0.3) -> bool:
    """Binary gate: True when PAC exceeds threshold."""
    return pac_value >= threshold
