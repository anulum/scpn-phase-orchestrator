# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Financial market Kuramoto regime detection

"""Kuramoto-based financial market synchronization analysis.

Extracts instantaneous phase from price/return time series via Hilbert
transform, computes the Kuramoto order parameter R(t) across assets,
and detects synchronization regimes. R(t) → 1 precedes market crashes
(documented for Black Monday 1987 and 2008 crisis).

arXiv:1109.1167; CEUR-WS Vol-915.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import hilbert

try:
    from spo_kernel import (
        detect_regimes_rust as _rust_regimes,
    )
    from spo_kernel import (
        market_order_parameter_rust as _rust_mop,
    )
    from spo_kernel import (
        market_plv_rust as _rust_plv,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


def extract_phase(series: NDArray) -> NDArray:
    """Extract instantaneous phase from a time series via Hilbert transform.

    Args:
        series: (T,) or (T, N) time series (prices, returns, etc.)

    Returns:
        Phase array, same shape as input, values in [0, 2π)
    """
    analytic = hilbert(series, axis=0)
    phase: NDArray = np.angle(analytic) % (2.0 * np.pi)
    return phase


def market_order_parameter(phases: NDArray) -> NDArray:
    """Kuramoto order parameter R(t) across assets at each timestep.

    Args:
        phases: (T, N) phase matrix — T timesteps, N assets

    Returns:
        (T,) order parameter R(t) in [0, 1]
    """
    phases = np.asarray(phases, dtype=np.float64)
    if _HAS_RUST:
        T, N = phases.shape
        flat = np.ascontiguousarray(phases.ravel())
        return np.asarray(_rust_mop(flat, T, N))
    z = np.exp(1j * phases)
    R: NDArray = np.abs(np.mean(z, axis=1))
    return R


def market_plv(phases: NDArray, window: int = 50) -> NDArray:
    """Windowed Phase-Locking Value matrix between assets.

    Args:
        phases: (T, N) phase matrix
        window: rolling window size

    Returns:
        (T - window + 1, N, N) PLV matrices
    """
    phases = np.asarray(phases, dtype=np.float64)
    T, N = phases.shape

    if _HAS_RUST:
        flat = np.ascontiguousarray(phases.ravel())
        plv_flat = np.asarray(_rust_plv(flat, T, N, window))
        n_windows = T - window + 1
        return plv_flat.reshape(n_windows, N, N)

    n_windows = T - window + 1
    plv_series = np.empty((n_windows, N, N))

    for t in range(n_windows):
        chunk = phases[t : t + window]  # (window, N)
        diff = chunk[:, :, np.newaxis] - chunk[:, np.newaxis, :]
        plv_series[t] = np.abs(np.mean(np.exp(1j * diff), axis=0))

    return plv_series


def detect_regimes(
    R: NDArray,
    sync_threshold: float = 0.7,
    desync_threshold: float = 0.3,
) -> NDArray:
    """Classify market synchronization regimes from R(t).

    Args:
        R: (T,) order parameter time series
        sync_threshold: R above this = SYNCHRONIZED (crash risk)
        desync_threshold: R below this = DESYNCHRONIZED (normal)

    Returns:
        (T,) integer labels: 0=desync, 1=transition, 2=synchronized
    """
    R = np.asarray(R, dtype=np.float64)
    if _HAS_RUST:
        flat = np.ascontiguousarray(R.ravel())
        return np.asarray(
            _rust_regimes(flat, sync_threshold, desync_threshold),
        )
    regimes = np.ones(len(R), dtype=np.int32)  # default: transition
    mask_sync = sync_threshold <= R
    mask_desync = desync_threshold >= R
    regimes[mask_sync] = 2
    regimes[mask_desync] = 0
    return regimes


def sync_warning(
    R: NDArray,
    threshold: float = 0.7,
    lookback: int = 10,
) -> NDArray:
    """Detect synchronization warning signals.

    Returns True at timesteps where R crosses the threshold from below
    (trending toward dangerous synchronization).

    Args:
        R: (T,) order parameter
        threshold: warning level
        lookback: smoothing window

    Returns:
        (T,) boolean warning signal
    """
    if lookback > 1:
        kernel = np.ones(lookback) / lookback
        R_smooth = np.convolve(R, kernel, mode="same")
    else:
        R_smooth = R

    warnings = np.zeros(len(R), dtype=bool)
    for t in range(1, len(R)):
        if R_smooth[t] >= threshold and R_smooth[t - 1] < threshold:
            warnings[t] = True
    return warnings
