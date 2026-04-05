# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Amplitude envelope solver

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

try:
    from spo_kernel import (
        envelope_modulation_depth_rust as _rust_modulation_depth,
    )
    from spo_kernel import (
        extract_envelope_rust as _rust_extract_envelope,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

__all__ = ["extract_envelope", "envelope_modulation_depth", "EnvelopeState"]


def extract_envelope(amplitudes_history: NDArray, window: int = 10) -> NDArray:
    """Sliding-window RMS of amplitude history.

    Args:
        amplitudes_history: (T,) or (T, N) amplitude time series
        window: RMS window length in samples

    Returns:
        (T,) or (T, N) envelope array (shorter by window-1 at the start,
        padded with the first valid value).
    """
    if amplitudes_history.size == 0:
        return amplitudes_history.copy()
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    if _HAS_RUST and amplitudes_history.ndim == 1:
        a = np.ascontiguousarray(amplitudes_history, dtype=np.float64)
        result: NDArray = np.asarray(_rust_extract_envelope(a, window))
        return result

    sq = amplitudes_history.astype(np.float64) ** 2
    if sq.ndim == 1:
        cs = np.cumsum(sq)
        cs = np.insert(cs, 0, 0.0)
        rms = np.sqrt((cs[window:] - cs[:-window]) / window)
        # Pad front with first valid value
        pad = np.full(window - 1, rms[0] if rms.size > 0 else 0.0)
        return np.concatenate([pad, rms])

    # 2-D case: (T, N)
    cs = np.cumsum(sq, axis=0)
    cs = np.vstack([np.zeros((1, sq.shape[1]), dtype=np.float64), cs])
    rms = np.sqrt((cs[window:] - cs[:-window]) / window)
    first = rms[0] if rms.shape[0] > 0 else np.zeros(sq.shape[1])
    return np.vstack([np.tile(first, (window - 1, 1)), rms])


def envelope_modulation_depth(envelope: NDArray) -> float:
    """Modulation depth: (max - min) / (max + min), ∈ [0, 1].

    Returns 0.0 for empty or constant envelopes.
    """
    if envelope.size == 0:
        return 0.0
    if _HAS_RUST:
        flat = np.ascontiguousarray(envelope.ravel(), dtype=np.float64)
        return float(_rust_modulation_depth(flat))
    flat = envelope.ravel()
    vmax = float(np.max(flat))
    vmin = float(np.min(flat))
    denom = vmax + vmin
    if denom <= 0.0:
        return 0.0
    return float((vmax - vmin) / denom)


@dataclass(frozen=True)
class EnvelopeState:
    """Snapshot of amplitude envelope statistics."""

    mean_amplitude: float
    amplitude_spread: float
    modulation_depth: float
    subcritical_count: int
