# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Phase Transfer Entropy

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["phase_transfer_entropy", "transfer_entropy_matrix"]

try:
    from spo_kernel import (
        phase_transfer_entropy_rust as _rust_pte,
    )
    from spo_kernel import (
        transfer_entropy_matrix_rust as _rust_tem,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


def phase_transfer_entropy(
    source: NDArray,
    target: NDArray,
    n_bins: int = 16,
) -> float:
    """Transfer entropy from source → target phase series.

    TE(X→Y) = H(Y_t+1 | Y_t) - H(Y_t+1 | Y_t, X_t)

    Estimated via binned phase histograms. Higher TE = source drives target.
    """
    if _HAS_RUST:
        return float(
            _rust_pte(
                np.ascontiguousarray(source, dtype=np.float64),
                np.ascontiguousarray(target, dtype=np.float64),
                n_bins,
            )
        )

    n = len(source)
    if n < 3 or len(target) < 3:
        return 0.0

    n = min(len(source), len(target)) - 1
    bins = np.linspace(0, 2 * np.pi, n_bins + 1)

    src_binned = np.digitize(source[:n], bins) - 1
    tgt_binned = np.digitize(target[:n], bins) - 1
    tgt_next = np.digitize(target[1 : n + 1], bins) - 1

    # Joint and marginal probabilities
    # H(Y_t+1 | Y_t)
    h_yt1_yt = _conditional_entropy(tgt_next, tgt_binned, n_bins)
    # H(Y_t+1 | Y_t, X_t) — condition on both
    joint_cond = tgt_binned * n_bins + src_binned
    h_yt1_yt_xt = _conditional_entropy(tgt_next, joint_cond, n_bins * n_bins)

    return max(0.0, h_yt1_yt - h_yt1_yt_xt)


def _conditional_entropy(
    target: NDArray, condition: NDArray, n_cond_bins: int
) -> float:
    """H(target | condition) via histogram."""
    n = len(target)
    h = 0.0
    for c in range(n_cond_bins):
        mask = condition == c
        count = np.sum(mask)
        if count < 2:
            continue
        vals = target[mask]
        _, counts = np.unique(vals, return_counts=True)
        probs = counts / count
        h -= (count / n) * float(np.sum(probs * np.log(probs + 1e-30)))
    return h


def transfer_entropy_matrix(
    phase_series: NDArray,
    n_bins: int = 16,
) -> NDArray:
    """Pairwise transfer entropy matrix TE(i→j) for all oscillator pairs.

    Args:
        phase_series: (n_oscillators, n_timesteps) phase trajectories.
    """
    n_osc = phase_series.shape[0]
    if _HAS_RUST:
        n_time = phase_series.shape[1]
        flat = np.ascontiguousarray(phase_series.ravel(), dtype=np.float64)
        te_flat = np.asarray(_rust_tem(flat, n_osc, n_time, n_bins))
        return te_flat.reshape(n_osc, n_osc)

    te = np.zeros((n_osc, n_osc), dtype=np.float64)
    for i in range(n_osc):
        for j in range(n_osc):
            if i != j:
                te[i, j] = phase_transfer_entropy(
                    phase_series[i], phase_series[j], n_bins
                )
    return te
