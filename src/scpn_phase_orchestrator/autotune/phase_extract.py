# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Phase extraction from raw time series

"""Hilbert-transform phase extraction for finite one-dimensional signals.

The extractor converts a raw real-valued signal into instantaneous phase,
amplitude, frequency, and dominant FFT frequency, with an optional FFT-domain
bandpass before Hilbert analysis. Inputs must be one-dimensional and long
enough for stable derivative and FFT estimates. The module returns diagnostic
arrays only and does not infer bindings or apply filtering outside its local
copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.signal import hilbert

__all__ = ["extract_phases", "PhaseResult"]

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass
class PhaseResult:
    """Hilbert-transform output: instantaneous phase, amplitude, and frequency."""

    phases: FloatArray
    amplitudes: FloatArray
    instantaneous_freq: FloatArray
    dominant_freq: float


def extract_phases(
    signal: FloatArray,
    fs: float,
    bandpass: tuple[float, float] | None = None,
) -> PhaseResult:
    """Extract instantaneous phase, amplitude, frequency via Hilbert transform.

    Args:
        signal: 1-D real-valued time series.
        fs: sampling frequency in Hz.
        bandpass: optional (low, high) Hz for bandpass filtering before Hilbert.
    """
    sample_rate = _positive_real(fs, "fs")
    x = _real_signal(signal)
    if x.ndim != 1:
        raise ValueError(f"signal must be 1-D, got shape {x.shape}")
    if len(x) < 4:
        raise ValueError(f"signal too short ({len(x)} samples), need >= 4")
    if not np.all(np.isfinite(x)):
        raise ValueError("signal must contain only finite values")

    if bandpass is not None:
        low, high = _validated_bandpass(bandpass, sample_rate)
        x = _bandpass_filter(x, sample_rate, low, high)

    analytic = hilbert(x)
    phases: FloatArray = np.angle(analytic) % (2 * np.pi)
    amplitudes: FloatArray = np.abs(analytic)

    # Instantaneous frequency from phase derivative
    dphase = np.diff(np.unwrap(np.angle(analytic)))
    inst_freq = np.concatenate([[0.0], dphase * sample_rate / (2 * np.pi)])

    # Dominant frequency via FFT
    fft_mag = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sample_rate)
    ac_mag = fft_mag[1:]
    dominant_freq = (
        float(freqs[np.argmax(ac_mag) + 1])
        if ac_mag.size > 0 and float(np.max(ac_mag)) > 0.0
        else 0.0
    )

    return PhaseResult(
        phases=phases,
        amplitudes=amplitudes,
        instantaneous_freq=inst_freq,
        dominant_freq=dominant_freq,
    )


def _bandpass_filter(x: FloatArray, fs: float, low: float, high: float) -> FloatArray:
    """Apply a simple FFT-based bandpass filter."""
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft = np.fft.rfft(x)
    mask = (freqs >= low) & (freqs <= high)
    fft[~mask] = 0.0
    result: FloatArray = np.fft.irfft(fft, n=n)
    return result


def _real_signal(signal: object) -> FloatArray:
    raw = np.asarray(signal)
    if raw.dtype == np.bool_ or _contains_alias(raw, (bool, np.bool_)):
        raise ValueError("signal must not contain boolean values")
    if np.iscomplexobj(raw) or _contains_alias(raw, (complex, np.complexfloating)):
        raise ValueError("signal must be real-valued")
    try:
        x: FloatArray = raw.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("signal must be real-valued") from exc
    return x


def _validated_bandpass(
    bandpass: tuple[float, float],
    fs: float,
) -> tuple[float, float]:
    if not isinstance(bandpass, tuple) or len(bandpass) != 2:
        raise ValueError("bandpass must be a (low, high) frequency tuple")
    low = _non_negative_real(bandpass[0], "bandpass low")
    high = _positive_real(bandpass[1], "bandpass high")
    nyquist = 0.5 * fs
    if not low < high:
        raise ValueError("bandpass low must be lower than bandpass high")
    if high > nyquist:
        raise ValueError("bandpass high must not exceed Nyquist frequency")
    return low, high


def _positive_real(value: object, name: str) -> float:
    parsed = _real_scalar(value, name)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return parsed


def _non_negative_real(value: object, name: str) -> float:
    parsed = _real_scalar(value, name)
    if parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return parsed


def _real_scalar(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real value")
    parsed = float(value)
    if not isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _contains_alias(raw: NDArray[np.generic], aliases: tuple[type, ...]) -> bool:
    if raw.dtype != object:
        return False
    return any(isinstance(item, aliases) for item in raw.ravel())
