# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Phase extraction from raw time series

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import hilbert

__all__ = ["extract_phases", "PhaseResult"]


@dataclass
class PhaseResult:
    phases: NDArray
    amplitudes: NDArray
    instantaneous_freq: NDArray
    dominant_freq: float


def extract_phases(
    signal: NDArray,
    fs: float,
    bandpass: tuple[float, float] | None = None,
) -> PhaseResult:
    """Extract instantaneous phase, amplitude, frequency via Hilbert transform.

    Args:
        signal: 1-D real-valued time series.
        fs: sampling frequency in Hz.
        bandpass: optional (low, high) Hz for bandpass filtering before Hilbert.
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"signal must be 1-D, got shape {x.shape}")
    if len(x) < 4:
        raise ValueError(f"signal too short ({len(x)} samples), need >= 4")

    if bandpass is not None:
        x = _bandpass_filter(x, fs, bandpass[0], bandpass[1])

    analytic = hilbert(x)
    phases: NDArray = np.angle(analytic) % (2 * np.pi)
    amplitudes: NDArray = np.abs(analytic)

    # Instantaneous frequency from phase derivative
    dphase = np.diff(np.unwrap(np.angle(analytic)))
    inst_freq = np.concatenate([[0.0], dphase * fs / (2 * np.pi)])

    # Dominant frequency via FFT
    fft_mag = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    dominant_freq = float(freqs[np.argmax(fft_mag[1:]) + 1]) if len(freqs) > 1 else 0.0

    return PhaseResult(
        phases=phases,
        amplitudes=amplitudes,
        instantaneous_freq=inst_freq,
        dominant_freq=dominant_freq,
    )


def _bandpass_filter(x: NDArray, fs: float, low: float, high: float) -> NDArray:
    """Simple FFT-based bandpass filter."""
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft = np.fft.rfft(x)
    mask = (freqs >= low) & (freqs <= high)
    fft[~mask] = 0.0
    result: NDArray = np.fft.irfft(fft, n=n)
    return result
