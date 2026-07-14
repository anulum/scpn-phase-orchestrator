# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Adaptive multi-channel Kuramoto detector

"""Robust, quality-weighted multi-channel Kuramoto order parameter.

The simple mean Kuramoto order parameter :math:`R(t)` collapses when some
channels are noisy or when the target and null classes have similar mean
phase coherence. This module introduces an **adaptive** variant that:

1. Weights each channel by a data-driven quality score (delta-band SNR
   penalised by excess kurtosis, a transient/artifact proxy).
2. Computes the weighted Kuramoto order parameter sample by sample.
3. Pools each epoch with a robust statistic (median) instead of the mean,
   reducing sensitivity to brief artefacts.

The result is a per-epoch score that is more stable across recordings and
channel configurations than the unweighted mean-R detector.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, filtfilt, hilbert

if TYPE_CHECKING:  # pragma: no cover - import only for static typing
    pass

FloatArray = NDArray[np.float64]

__all__ = [
    "compute_adaptive_kuramoto_scores",
    "compute_channel_quality_weights",
    "compute_weighted_kuramoto_r",
]


def _bandpass(
    sig: FloatArray, fs: float, lo: float, hi: float, order: int = 3
) -> FloatArray:
    """Return a zero-phase Butterworth band-pass filtered signal.

    Works for both 1-D signals and multi-channel arrays; the filter is applied
    along the last axis.
    """
    nyq = fs / 2.0
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return cast(FloatArray, filtfilt(b, a, sig, axis=-1))


def _kurtosis_excess(x: FloatArray, axis: int = -1) -> FloatArray:
    """Return excess kurtosis of ``x`` along ``axis``.

    Uses the unbiased k2 estimator from Fisher, which is stable for the
    epoch-level sample sizes used here.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[axis]
    if n < 4:
        # Not enough samples for a meaningful kurtosis; return neutral.
        return np.zeros(x.shape[:axis] + x.shape[axis + 1 :], dtype=np.float64)
    mean = x.mean(axis=axis, keepdims=True)
    std = x.std(axis=axis, keepdims=True)
    # Avoid division by zero on constant signals.
    std = np.where(std == 0, 1.0, std)
    z = (x - mean) / std
    k1 = cast(
        FloatArray,
        (n * (z**4).sum(axis=axis)) / ((n - 1) * (n - 2) * (n - 3)),
    )
    k2 = cast(
        FloatArray,
        (3 * (n - 1) ** 2) / ((n - 2) * (n - 3)),
    )
    return k1 - k2


def compute_channel_quality_weights(
    data: FloatArray,
    fs: float,
    band_hz: tuple[float, float] = (0.5, 4.0),
    epoch_seconds: float = 30.0,
    kurtosis_penalty_scale: float = 0.2,
) -> FloatArray:
    """Return per-channel, per-epoch quality weights in ``[0, 1]``.

    The weight rewards channels with strong band-limited SNR and penalises
    channels with high excess kurtosis (transients / muscle artefacts).

    Parameters
    ----------
    data
        Multi-channel signal, shape ``(n_channels, n_samples)``.
    fs
        Sampling rate in hertz.
    band_hz
        Target frequency band (low, high).
    epoch_seconds
        Epoch length in seconds.
    kurtosis_penalty_scale
        Scaling of the kurtosis penalty; larger values make the penalty
        stronger. ``0`` disables kurtosis penalisation.

    Returns
    -------
    FloatArray
        Weights of shape ``(n_channels, n_epochs)``.
    """
    n_channels, n_samples = data.shape
    epoch_len = int(epoch_seconds * fs)
    n_epochs = n_samples // epoch_len
    if n_epochs == 0:
        raise ValueError("signal shorter than one epoch")

    epoch_data = data[:, : n_epochs * epoch_len].reshape(
        n_channels, n_epochs, epoch_len
    )

    # Band-limited power per channel per epoch (vectorised across channels).
    filtered = _bandpass(data, fs, band_hz[0], band_hz[1])
    filtered_epochs = filtered[:, : n_epochs * epoch_len].reshape(
        n_channels, n_epochs, epoch_len
    )
    band_power = np.mean(filtered_epochs**2, axis=2)

    total_power = np.mean(epoch_data**2, axis=2) + 1e-12
    snr = band_power / total_power

    # Per-channel, per-epoch excess kurtosis.
    kurt = _kurtosis_excess(epoch_data, axis=2)
    # Normalise kurtosis to a soft penalty: kurtosis ~ 0 -> penalty ~ 1,
    # high kurtosis -> penalty -> 0.
    kurtosis_penalty = 1.0 / (1.0 + kurtosis_penalty_scale * np.maximum(kurt, 0.0))

    raw_weights = np.sqrt(np.maximum(snr, 0.0)) * kurtosis_penalty

    # Normalise per epoch so weights sum to one (avoids dependence on channel
    # count and keeps R in [0, 1]).
    per_epoch_sum = raw_weights.sum(axis=0, keepdims=True)
    per_epoch_sum = np.where(per_epoch_sum == 0, 1.0, per_epoch_sum)
    return cast(FloatArray, raw_weights / per_epoch_sum)


def compute_weighted_kuramoto_r(
    phases: FloatArray,
    weights: FloatArray,
    epoch_seconds: float,
    fs: float,
) -> FloatArray:
    """Return per-epoch robust weighted Kuramoto order parameter.

    Parameters
    ----------
    phases
        Instantaneous phases, shape ``(n_channels, n_samples)``.
    weights
        Per-channel, per-epoch weights, shape ``(n_channels, n_epochs)``.
    epoch_seconds
        Epoch length in seconds.
    fs
        Sampling rate in hertz.

    Returns
    -------
    FloatArray
        Per-epoch score of shape ``(n_epochs,)``.
    """
    n_channels, n_samples = phases.shape
    epoch_len = int(epoch_seconds * fs)
    n_epochs = n_samples // epoch_len

    phases_epochs = phases[:, : n_epochs * epoch_len].reshape(
        n_channels, n_epochs, epoch_len
    )
    weights_per_sample = np.repeat(weights[:, :, np.newaxis], epoch_len, axis=2)

    with np.errstate(invalid="ignore"):
        z = weights_per_sample * np.exp(1j * phases_epochs)
        r_t = np.abs(z.sum(axis=0) / weights.sum(axis=0)[:, np.newaxis])

    # Robust temporal pooling: median over the epoch.
    return cast(FloatArray, np.median(r_t, axis=1))


def compute_adaptive_kuramoto_scores(
    data: FloatArray,
    fs: float,
    band_hz: tuple[float, float] = (0.5, 4.0),
    epoch_seconds: float = 30.0,
    kurtosis_penalty_scale: float = 0.2,
    score_precision: int = 6,
) -> tuple[FloatArray, FloatArray]:
    """Return per-epoch adaptive Kuramoto scores and channel weights.

    Parameters
    ----------
    data
        Multi-channel signal, shape ``(n_channels, n_samples)``.
    fs
        Sampling rate in hertz.
    band_hz
        Target frequency band (low, high).
    epoch_seconds
        Epoch length in seconds.
    kurtosis_penalty_scale
        Strength of the kurtosis artefact penalty.
    score_precision
        Decimal places to which scores are rounded.

    Returns
    -------
    tuple[FloatArray, FloatArray]
        ``(scores, weights)`` where ``scores`` has shape ``(n_epochs,)`` and
        ``weights`` has shape ``(n_channels, n_epochs)``.
    """
    n_channels, n_samples = data.shape
    epoch_len = int(epoch_seconds * fs)
    n_epochs = n_samples // epoch_len
    if n_channels < 2:
        raise ValueError("adaptive Kuramoto requires at least 2 channels")
    if n_epochs == 0:
        raise ValueError("signal shorter than one epoch")

    trimmed = data[:, : n_epochs * epoch_len]
    filtered = _bandpass(trimmed, fs, band_hz[0], band_hz[1])
    phases = np.angle(hilbert(filtered, axis=1))

    weights = compute_channel_quality_weights(
        data[:, : n_epochs * epoch_len],
        fs,
        band_hz=band_hz,
        epoch_seconds=epoch_seconds,
        kurtosis_penalty_scale=kurtosis_penalty_scale,
    )
    scores = compute_weighted_kuramoto_r(phases, weights, epoch_seconds, fs)
    return np.round(scores, score_precision), weights
