# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Inter-Trial Phase Coherence

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    from spo_kernel import (
        compute_itpc_rust as _rust_itpc,
    )
    from spo_kernel import (
        itpc_persistence_rust as _rust_itpc_persist,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

__all__ = ["compute_itpc", "itpc_persistence"]


def compute_itpc(phases_trials: NDArray) -> NDArray:
    """Inter-Trial Phase Coherence at each time point.

    ITPC = |mean(exp(i*θ))| across trials (Lachaux et al. 1999).

    Args:
        phases_trials: shape (n_trials, n_timepoints), phases in radians.

    Returns:
        ITPC values, shape (n_timepoints,), each in [0, 1].
    """
    phases = np.asarray(phases_trials, dtype=np.float64)
    if phases.ndim == 1:
        return np.array([1.0])
    if phases.shape[0] == 0:
        return np.array([])

    if _HAS_RUST:
        n_trials, n_tp = phases.shape
        flat = np.ascontiguousarray(phases.ravel(), dtype=np.float64)
        return np.asarray(_rust_itpc(flat, n_trials, n_tp))

    result: NDArray = np.abs(np.mean(np.exp(1j * phases), axis=0))
    return result


def itpc_persistence(
    phases_trials: NDArray,
    pause_indices: list[int] | NDArray,
) -> float:
    """ITPC measured at time points after a stimulus pause.

    Distinguishes true neural entrainment from evoked response:
    if ITPC remains high after the driving stimulus stops,
    oscillators have genuinely phase-locked (entrainment).
    If ITPC drops immediately, the response was merely evoked.

    Args:
        phases_trials: shape (n_trials, n_timepoints), phases in radians.
        pause_indices: time-point indices falling within or after a pause.

    Returns:
        Mean ITPC across the specified pause indices.
        Returns 0.0 if pause_indices is empty.
    """
    pause_idx = np.asarray(pause_indices, dtype=int)
    if pause_idx.size == 0:
        return 0.0

    if _HAS_RUST:
        phases = np.asarray(phases_trials, dtype=np.float64)
        if phases.ndim == 1:
            phases = phases.reshape(1, -1)
        n_trials, n_tp = phases.shape
        flat = np.ascontiguousarray(phases.ravel(), dtype=np.float64)
        idx = np.ascontiguousarray(pause_idx, dtype=np.int64)
        return float(_rust_itpc_persist(flat, n_trials, n_tp, idx))

    itpc_full = compute_itpc(phases_trials)
    valid = pause_idx[(pause_idx >= 0) & (pause_idx < itpc_full.size)]
    if valid.size == 0:
        return 0.0
    return float(np.mean(itpc_full[valid]))
