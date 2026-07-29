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

1. Weights each channel by a data-driven quality score. Two strategies are
   provided:
   - *SNR+kurtosis*: delta-band SNR penalised by excess kurtosis (a
     transient/artefact proxy).
   - *PLV-to-mean-field*: each channel's phase-locking value to the
     instantaneous group phase, rewarding channels that track the mean field.
2. Computes the weighted Kuramoto order parameter sample by sample.
3. Pools each epoch with a robust statistic (median) instead of the mean,
   reducing sensitivity to brief artefacts.

The result is a per-epoch score that is more stable across recordings and
channel configurations than the unweighted mean-R detector when the chosen
weighting matches the domain.
"""

from __future__ import annotations

import math
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
    "compute_phase_locking_weights",
    "compute_weighted_kuramoto_r",
]


def _finite_real_matrix(value: object, name: str) -> FloatArray:
    """Return a non-empty finite real matrix without coercive aliases."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real matrix") from exc
    if raw.ndim != 2 or 0 in raw.shape:
        raise ValueError(f"{name} must be a finite real matrix")
    if raw.dtype.kind == "O":
        if any(
            isinstance(item, (bool, np.bool_, complex, str, bytes, np.str_, np.bytes_))
            or not isinstance(item, (int, float, np.integer, np.floating))
            for item in raw.flat
        ):
            raise ValueError(f"{name} must be a finite real matrix")
    elif raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real matrix")
    result = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite real matrix")
    return np.ascontiguousarray(result)


def _finite_real(value: object, name: str) -> float:
    """Return a finite real scalar without boolean or text coercion."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite real")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real")
    return result


def _positive_finite_real(value: object, name: str) -> float:
    """Return a strictly positive finite real scalar."""
    try:
        result = _finite_real(value, name)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive finite real") from exc
    if result <= 0.0:
        raise ValueError(f"{name} must be a positive finite real")
    return result


def _sampling_contract(fs: object, epoch_seconds: object) -> tuple[float, float, int]:
    """Validate sampling geometry and return normalised values plus epoch length."""
    fs_value = _positive_finite_real(fs, "fs")
    epoch_value = _positive_finite_real(epoch_seconds, "epoch_seconds")
    samples = fs_value * epoch_value
    if not math.isfinite(samples) or samples < 1.0:
        raise ValueError("epoch_seconds must define at least one sample")
    return fs_value, epoch_value, int(samples)


def _validate_band(band_hz: object, fs: float) -> tuple[float, float]:
    """Return a finite ordered band strictly inside the Nyquist interval."""
    if not isinstance(band_hz, tuple) or len(band_hz) != 2:
        raise ValueError("band_hz must be a (low, high) tuple")
    low = _positive_finite_real(band_hz[0], "band_hz low")
    high = _positive_finite_real(band_hz[1], "band_hz high")
    if not low < high < fs / 2.0:
        raise ValueError("band_hz must satisfy 0 < low < high < fs / 2")
    return low, high


def _optional_integer(value: object, name: str) -> int | None:
    """Return ``None`` or an integer without boolean coercion."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _non_negative_integer(value: object, name: str) -> int:
    """Return a non-negative integer without boolean coercion."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _bandpass(
    sig: FloatArray, fs: float, lo: float, hi: float, order: int = 3
) -> FloatArray:
    """Return a zero-phase Butterworth band-pass filtered signal.

    Works for both 1-D signals and multi-channel arrays; the filter is applied
    along the last axis.
    """
    nyq = fs / 2.0
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    padlen = 3 * max(len(a), len(b))
    if sig.shape[-1] <= padlen:
        raise ValueError(f"signal must contain more than {padlen} samples")
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

    Raises
    ------
    ValueError
        If the signal is shorter than one epoch.
    """
    data = _finite_real_matrix(data, "data")
    fs, epoch_seconds, epoch_len = _sampling_contract(fs, epoch_seconds)
    band_hz = _validate_band(band_hz, fs)
    kurtosis_penalty_scale = _finite_real(
        kurtosis_penalty_scale, "kurtosis_penalty_scale"
    )
    if kurtosis_penalty_scale < 0.0:
        raise ValueError("kurtosis_penalty_scale must be non-negative")
    n_channels, n_samples = data.shape
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
    uniform = np.full_like(raw_weights, 1.0 / n_channels)
    return cast(
        FloatArray,
        np.divide(
            raw_weights,
            per_epoch_sum,
            out=uniform,
            where=per_epoch_sum > 0.0,
        ),
    )


def compute_phase_locking_weights(
    phases: FloatArray,
    fs: float,
    epoch_seconds: float = 30.0,
    top_k: int | None = None,
) -> FloatArray:
    """Return per-channel, per-epoch weights based on PLV to the mean field.

    For each epoch, the mean field phase ``psi(t)`` is the phase of the average
    complex exponential across channels. Each channel's weight is proportional
    to its phase-locking value to that mean field:

        PLV_c = | < exp(i (phi_c(t) - psi(t))) >_t |

    Channels that consistently track the group phase receive higher weight;
    noisy or independent channels are down-weighted.

    Parameters
    ----------
    phases
        Instantaneous phases, shape ``(n_channels, n_samples)``.
    fs
        Sampling rate in hertz.
    epoch_seconds
        Epoch length in seconds.
    top_k
        If ``None``, returns per-epoch normalised PLV weights (sum to one per
        epoch). If an integer ``k``, selects the ``k`` channels with the highest
        mean PLV across epochs and returns binary selection weights (``1`` for
        selected channels, ``0`` otherwise). This is the "global top-k" variant.

    Returns
    -------
    FloatArray
        Weights of shape ``(n_channels, n_epochs)``.

    Raises
    ------
    ValueError
        If the signal is shorter than one epoch, or ``top_k`` is outside the
        range ``[1, n_channels]``.
    """
    phases = _finite_real_matrix(phases, "phases")
    fs, epoch_seconds, epoch_len = _sampling_contract(fs, epoch_seconds)
    top_k = _optional_integer(top_k, "top_k")
    n_channels, n_samples = phases.shape
    n_epochs = n_samples // epoch_len
    if n_epochs == 0:
        raise ValueError("signal shorter than one epoch")

    phases_epochs = phases[:, : n_epochs * epoch_len].reshape(
        n_channels, n_epochs, epoch_len
    )
    mean_field = np.angle(np.exp(1j * phases_epochs).mean(axis=0))
    aligned = np.exp(1j * (phases_epochs - mean_field[np.newaxis, :, :]))
    plv = np.abs(aligned.mean(axis=2))

    if top_k is not None:
        if not 1 <= top_k <= n_channels:
            raise ValueError(f"top_k must be between 1 and {n_channels}, got {top_k}")
        mean_plv = plv.mean(axis=1)
        selected = np.argsort(mean_plv)[-top_k:]
        weights = np.zeros_like(plv)
        weights[selected, :] = 1.0
        return cast(FloatArray, weights)

    per_epoch_sum = plv.sum(axis=0, keepdims=True)
    per_epoch_sum = np.where(per_epoch_sum == 0, 1.0, per_epoch_sum)
    return cast(FloatArray, plv / per_epoch_sum)


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

    Raises
    ------
    ValueError
        If either matrix is not finite and real, the sampling geometry is
        invalid, the signal is shorter than one epoch, or the weights do not
        exactly match the channel/epoch geometry with non-negative positive
        mass in every epoch.
    """
    phases = _finite_real_matrix(phases, "phases")
    weights = _finite_real_matrix(weights, "weights")
    fs, epoch_seconds, epoch_len = _sampling_contract(fs, epoch_seconds)
    n_channels, n_samples = phases.shape
    n_epochs = n_samples // epoch_len
    if n_epochs == 0:
        raise ValueError("signal shorter than one epoch")
    expected_shape = (n_channels, n_epochs)
    if weights.shape != expected_shape:
        raise ValueError(f"weights shape must be {expected_shape}, got {weights.shape}")
    if np.any(weights < 0.0):
        raise ValueError("weights must be non-negative")
    weight_sums = weights.sum(axis=0)
    if np.any(weight_sums <= 0.0):
        raise ValueError("weights must have positive mass in every epoch")

    phases_epochs = phases[:, : n_epochs * epoch_len].reshape(
        n_channels, n_epochs, epoch_len
    )
    weights_per_sample = np.repeat(weights[:, :, np.newaxis], epoch_len, axis=2)

    with np.errstate(invalid="ignore"):
        z = weights_per_sample * np.exp(1j * phases_epochs)
        r_t = np.abs(z.sum(axis=0) / weight_sums[:, np.newaxis])

    # Robust temporal pooling: median over the epoch.
    return cast(FloatArray, np.median(r_t, axis=1))


def compute_adaptive_kuramoto_scores(
    data: FloatArray,
    fs: float,
    band_hz: tuple[float, float] = (0.5, 4.0),
    epoch_seconds: float = 30.0,
    kurtosis_penalty_scale: float = 0.2,
    weight_mode: str = "snr_kurtosis",
    top_k: int | None = None,
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
        Strength of the kurtosis artefact penalty (only used when
        ``weight_mode="snr_kurtosis"``).
    weight_mode
        Weighting strategy. ``"snr_kurtosis"`` uses band-limited SNR penalised
        by excess kurtosis; ``"plv_mean_field"`` uses each channel's
        phase-locking value to the instantaneous mean field.
    top_k
        Only used when ``weight_mode="plv_mean_field"``. If set, selects the
        ``top_k`` channels with the highest mean PLV across epochs and computes
        the unweighted mean-R over those channels (global top-k selection).
    score_precision
        Decimal places to which scores are rounded.

    Returns
    -------
    tuple[FloatArray, FloatArray]
        ``(scores, weights)`` where ``scores`` has shape ``(n_epochs,)`` and
        ``weights`` has shape ``(n_channels, n_epochs)``.

    Raises
    ------
    ValueError
        If fewer than two channels are supplied, the signal is shorter than one
        epoch, ``top_k`` is combined with ``weight_mode="snr_kurtosis"``, or
        ``weight_mode`` is not a recognised strategy.
    """
    data = _finite_real_matrix(data, "data")
    fs, epoch_seconds, epoch_len = _sampling_contract(fs, epoch_seconds)
    band_hz = _validate_band(band_hz, fs)
    score_precision = _non_negative_integer(score_precision, "score_precision")
    n_channels, n_samples = data.shape
    n_epochs = n_samples // epoch_len
    if n_channels < 2:
        raise ValueError("adaptive Kuramoto requires at least 2 channels")
    if n_epochs == 0:
        raise ValueError("signal shorter than one epoch")

    trimmed = data[:, : n_epochs * epoch_len]
    filtered = _bandpass(trimmed, fs, band_hz[0], band_hz[1])
    phases = np.angle(hilbert(filtered, axis=1))

    if weight_mode == "snr_kurtosis":
        if top_k is not None:
            raise ValueError(
                "top_k is only supported with weight_mode='plv_mean_field'"
            )
        weights = compute_channel_quality_weights(
            data[:, : n_epochs * epoch_len],
            fs,
            band_hz=band_hz,
            epoch_seconds=epoch_seconds,
            kurtosis_penalty_scale=kurtosis_penalty_scale,
        )
    elif weight_mode == "plv_mean_field":
        weights = compute_phase_locking_weights(
            phases, fs, epoch_seconds=epoch_seconds, top_k=top_k
        )
    else:
        raise ValueError(f"unknown weight_mode: {weight_mode!r}")

    scores = compute_weighted_kuramoto_r(phases, weights, epoch_seconds, fs)
    return np.round(scores, score_precision), weights
