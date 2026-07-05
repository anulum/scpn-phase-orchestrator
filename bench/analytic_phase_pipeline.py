# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — shared raw-signal to analytic-phase pipeline

"""Shared signal-to-analytic-phase pipeline for the domain capstone adapters.

Turning a raw multichannel recording into a decimated per-channel analytic-phase
field is the same three-step operation in every domain — band-pass to the band
that carries the rhythm, take the Hilbert analytic phase, and decimate — only the
passband and rates differ. A scalp-EEG adapter and a cardiac-ECG adapter both need
it, so it lives here once and each adapter supplies its own parameters.

* :func:`bandpass` — a zero-phase Butterworth band-pass along the last axis.
* :func:`analytic_phase` — the per-channel Hilbert analytic phase.
* :func:`decimate_analytic_phase` — anti-aliased decimation of a *wrapped* phase
  field via its continuous ``sin``/``cos`` components (a wrapped phase must never
  be low-pass filtered — its ±π jumps are not band-limited), reconstructed with
  ``atan2``.

The three compose into the per-channel phase field an adapter hands to
:func:`~scpn_phase_orchestrator.monitor.early_warning_suite.observables_from_phases`.
:func:`validate_signals` is the shared input guard so an adapter can check its raw
block and emit a domain-specific message before running the pipeline.
"""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, decimate, hilbert, sosfiltfilt

FloatArray = NDArray[np.float64]

#: Default Butterworth band-pass order (per edge).
DEFAULT_FILTER_ORDER = 4

__all__ = [
    "DEFAULT_FILTER_ORDER",
    "analytic_phase",
    "bandpass",
    "decimate_analytic_phase",
    "validate_signals",
]


def bandpass(
    signals: FloatArray,
    *,
    sampling_rate_hz: float,
    band_hz: tuple[float, float],
    order: int = DEFAULT_FILTER_ORDER,
) -> FloatArray:
    """Zero-phase Butterworth band-pass along the last axis.

    Parameters
    ----------
    signals : FloatArray
        Per-channel samples, shape ``(N, T)``; a one-dimensional array is treated
        as a single channel.
    sampling_rate_hz : float
        Sampling rate of ``signals`` in hertz.
    band_hz : tuple[float, float]
        ``(low, high)`` passband edges in hertz, ``0 < low < high < Nyquist``.
    order : int
        Butterworth order (per edge).

    Returns
    -------
    FloatArray
        The band-passed signal, same shape as the validated input.

    Raises
    ------
    ValueError
        If the signal is malformed or the band is not a valid passband.
    """
    array = validate_signals(signals, "signals")
    fs = _positive_real(sampling_rate_hz, "sampling_rate_hz")
    order_int = _positive_int(order, "order")
    low, high = _validate_band(band_hz, fs)
    nyquist = 0.5 * fs
    sos = butter(order_int, [low / nyquist, high / nyquist], btype="band", output="sos")
    filtered = sosfiltfilt(sos, array, axis=-1)
    return np.ascontiguousarray(filtered, dtype=np.float64)


def analytic_phase(signals: FloatArray) -> FloatArray:
    """Return the per-channel Hilbert analytic phase in radians, shape ``(N, T)``.

    Parameters
    ----------
    signals : FloatArray
        Real band-passed samples, shape ``(N, T)`` (a one-dimensional array is a
        single channel).

    Returns
    -------
    FloatArray
        The instantaneous phase ``angle(hilbert(signals))``.

    Raises
    ------
    ValueError
        If the signal is malformed.
    """
    array = validate_signals(signals, "signals")
    analytic = hilbert(array, axis=-1)
    return np.ascontiguousarray(np.angle(analytic), dtype=np.float64)


def decimate_analytic_phase(phases: FloatArray, *, factor: int) -> FloatArray:
    """Anti-aliased decimation of a phase field via its ``sin``/``cos`` parts.

    A wrapped phase must never be low-pass filtered directly — its ±π
    discontinuities are not band-limited — so the *continuous* components
    ``sin φ`` and ``cos φ`` are decimated (zero-phase FIR anti-alias) and the
    phase reconstructed with ``atan2``.

    Parameters
    ----------
    phases : FloatArray
        Per-channel phase in radians, shape ``(N, T)``.
    factor : int
        Integer decimation factor; ``1`` returns the input unchanged.

    Returns
    -------
    FloatArray
        The reconstructed phase at the decimated rate, shape ``(N, T // factor)``.

    Raises
    ------
    ValueError
        If the phase field is malformed or the factor is not a positive integer.
    """
    array = validate_signals(phases, "phases")
    q = _positive_int(factor, "factor")
    if q == 1:
        return np.ascontiguousarray(array, dtype=np.float64)
    sin_d = decimate(np.sin(array), q, ftype="fir", zero_phase=True, axis=-1)
    cos_d = decimate(np.cos(array), q, ftype="fir", zero_phase=True, axis=-1)
    reconstructed = np.arctan2(sin_d, cos_d)
    return np.ascontiguousarray(reconstructed, dtype=np.float64)


def validate_signals(signals: object, name: str) -> FloatArray:
    """Return ``signals`` as a validated 2-D finite float array, else raise.

    A one-dimensional array is promoted to a single ``(1, T)`` channel. Adapters
    call this before the pipeline so a malformed raw block fails with a clear,
    domain-named message.

    Parameters
    ----------
    signals : object
        The candidate signal block.
    name : str
        Field name used in the error messages.

    Returns
    -------
    FloatArray
        The validated contiguous ``(N, T)`` float array.

    Raises
    ------
    ValueError
        If the block is boolean, complex, non-numeric, not one- or
        two-dimensional, empty, or non-finite.
    """
    raw = np.asarray(signals)
    if raw.dtype == np.bool_:
        raise ValueError(f"{name} must not contain boolean values")
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued")
    try:
        array = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real float array") from exc
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"{name} shape {raw.shape} must be one- or two-dimensional")
    if array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one sample")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _validate_band(band_hz: object, sampling_rate_hz: float) -> tuple[float, float]:
    """Return ``band_hz`` as a valid ``(low, high)`` passband, else raise."""
    if (
        isinstance(band_hz, (str, bytes))
        or not isinstance(band_hz, (tuple, list))
        or len(band_hz) != 2
    ):
        raise ValueError("band_hz must be a (low, high) pair in hertz")
    low = _positive_real(band_hz[0], "band_hz low edge")
    high = _positive_real(band_hz[1], "band_hz high edge")
    if low >= high:
        raise ValueError(f"band_hz low {low} must be below high {high}")
    if high >= 0.5 * sampling_rate_hz:
        raise ValueError(
            f"band_hz high {high} must be below the Nyquist {0.5 * sampling_rate_hz}"
        )
    return low, high


def _positive_int(value: object, name: str) -> int:
    """Return ``value`` as a positive integer, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer, got {result}")
    return result


def _positive_real(value: object, name: str) -> float:
    """Return ``value`` as a strictly positive finite real, else raise."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive real, got {value!r}")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {result}")
    return result
