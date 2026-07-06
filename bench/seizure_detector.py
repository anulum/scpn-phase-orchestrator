# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — spectral seizure-specific early-warning detector

"""A domain-specific scalp-EEG seizure detector, scored by the same honest protocol.

The SCPN suite and the AR(1)-Kendall-τ competitor are *generic* early-warning
detectors: critical slowing down reads variance and autocorrelation, synchronisation
reads phase coherence, and both are at chance on scalp EEG at a matched false-alarm
rate. The programme's thesis is that generic detection is a commodity — so the value is
not one detector for every domain, but a **domain-specific** detector per modality, each
scored through the same auditable, matched-false-alarm, permutation-tested moat.

This is the first such detector. It reads a *seizure-specific* preictal feature the
generic suite discards: the **redistribution of spectral power toward higher
frequencies** as cortex desynchronises before a seizure. Where the phase-based suite
band-passes and keeps only the analytic phase (throwing the amplitude away), this
detector keeps the amplitude and tracks the **ratio of beta-band (13–30 Hz) to
delta-band (0.5–4 Hz) power** per channel, scoring its rising trend
(:func:`spectral_rise_score`) toward the seizure. The two bands are fixed *a priori*
from the preictal-spectral literature (Mormann et al. 2007), not tuned to the corpus, so
the result is not p-hacked.

Two aggregations are offered, because a focal seizure is spatially localised:

* ``"mean"`` — the whole-head channel-averaged rise, the natural first choice;
* ``"focal"`` — the *most-rising channel's* rise, motivated a priori by a focal onset
  (e.g. chb01's right-temporal focus), where a whole-head average dilutes the signal.
  The per-segment maximum over channels inflates the statistic under the null, but the
  matched-false-alarm threshold is calibrated on that *same* maximum over the interictal
  nulls, so the multiplicity is absorbed and the comparison stays fair.

Either score is calibrated to a matched false alarm on interictal null segments and
tested by the shared label-permutation core
(:func:`~bench.early_warning_domain.permutation_significance_from_alarms`), so its
p-value is directly comparable to the generic suite and the Dakos competitor: same
segments, same operating point, same test — only the feature is seizure-specific.

The honest expectation, from the seizure-prediction literature, is that most preictal
features do not beat chance under a rigorous test; whether either aggregation does is
exactly what the moat is built to answer.

References
----------
* Mormann, Andrzejak, Elger & Lehnertz 2007, *Brain* 130:314 — "Seizure prediction: the
  long and winding road": a review of preictal features (spectral power among them) and
  the rigorous, above-chance-only standard this protocol shares.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.signal import welch
from scipy.stats import kendalltau

from bench.early_warning_domain import (
    DEFAULT_PERMUTATION_SEED,
    DEFAULT_PERMUTATIONS,
    DEFAULT_TARGET_FALSE_ALARM,
    PermutationSignificance,
    calibrate_score_threshold,
    permutation_significance_from_alarms,
)

if TYPE_CHECKING:  # pragma: no cover - import only for static typing
    from collections.abc import Sequence

    #: A one-dimensional real series coerced with ``np.asarray``.
    RealSeries = Sequence[float] | NDArray[np.float64]

FloatArray = NDArray[np.float64]

#: Delta band (Hz): the low-frequency power that dominates a resting/idling cortex.
DELTA_BAND: tuple[float, float] = (0.5, 4.0)
#: Beta band (Hz): the higher-frequency power that rises as cortex desynchronises.
BETA_BAND: tuple[float, float] = (13.0, 30.0)
#: Welch segment length in samples, capping the per-window periodogram resolution.
_NPERSEG = 256
#: A power floor so a near-silent delta band cannot divide to infinity.
_POWER_FLOOR = 1.0e-12

__all__ = [
    "BETA_BAND",
    "DELTA_BAND",
    "SeizureSignificance",
    "band_power",
    "beta_delta_ratio_trajectory",
    "channel_ratio_trajectories",
    "seizure_significance",
    "segment_rise_score",
    "spectral_rise_score",
]


def _integrate_band(
    freqs: FloatArray, psd: FloatArray, band: tuple[float, float]
) -> FloatArray:
    """Return the PSD summed over ``band`` along its last axis.

    Works for a single spectrum ``(freqs,)`` — returning a scalar array — and a stack of
    per-channel spectra ``(channels, freqs)`` — returning ``(channels,)``. When no bin
    falls inside the band the result is zero.
    """
    low, high = band
    mask = (freqs >= low) & (freqs <= high)
    if not bool(mask.any()):
        return np.zeros(psd.shape[:-1], dtype=np.float64)
    return np.asarray(psd[..., mask].sum(axis=-1), dtype=np.float64)


def band_power(window: FloatArray, *, rate: float, band: tuple[float, float]) -> float:
    """Return the spectral power of one channel window within a frequency band.

    Estimates the power spectral density by Welch's method and integrates it over
    ``band``. The Welch segment length is capped at :data:`_NPERSEG` so a short window
    still yields an estimate.

    Parameters
    ----------
    window : FloatArray
        One channel's samples over the analysis window, shape ``(T,)`` with ``T >= 2``.
    rate : float
        Sampling rate in Hz; must be positive.
    band : tuple[float, float]
        The ``(low, high)`` frequency band in Hz.

    Returns
    -------
    float
        The band power; ``0.0`` when no spectral bin falls inside the band.

    Raises
    ------
    ValueError
        If ``window`` is not a one-dimensional array of at least two samples, or
        ``rate`` is not positive.
    """
    values = np.asarray(window, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("window must be one-dimensional")
    if values.shape[0] < 2:
        raise ValueError("window must have at least two samples")
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("rate must be a positive finite number")
    freqs, psd = welch(values, fs=rate, nperseg=min(values.shape[0], _NPERSEG))
    return float(_integrate_band(freqs, psd, band))


def channel_ratio_trajectories(
    signals: FloatArray,
    *,
    rate: float,
    window: int,
    step: int,
    low_band: tuple[float, float] = DELTA_BAND,
    high_band: tuple[float, float] = BETA_BAND,
) -> FloatArray:
    """Return each channel's windowed beta-to-delta power ratio trajectory.

    Slides a ``window`` across the multichannel signal in hops of ``step``; in each
    window it estimates every channel's spectrum in one vectorised Welch call and takes
    the high-band-to-low-band power ratio per channel, giving one preictal-
    desynchronisation trajectory per channel.

    Parameters
    ----------
    signals : FloatArray
        The multichannel signal, shape ``(channels, samples)``.
    rate : float
        Sampling rate in Hz.
    window : int
        Analysis window length in samples; at least two and no longer than the signal.
    step : int
        Hop between consecutive windows in samples; positive.
    low_band, high_band : tuple[float, float]
        The delta and beta bands (Hz).

    Returns
    -------
    FloatArray
        The per-channel beta/delta ratio, shape ``(channels, n_windows)``.

    Raises
    ------
    ValueError
        If ``signals`` is not a two-dimensional channels-by-samples array, or the window
        or step do not fit it.
    """
    values = np.asarray(signals, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("signals must be two-dimensional (channels × samples)")
    n_channels, n_samples = values.shape
    if n_channels < 1:
        raise ValueError("signals must have at least one channel")
    window_int = _positive_int(window, "window")
    step_int = _positive_int(step, "step")
    if window_int < 2 or window_int > n_samples:
        raise ValueError(f"window {window_int} must lie in [2, {n_samples}]")
    starts = range(0, n_samples - window_int + 1, step_int)
    trajectories = np.empty((n_channels, len(starts)), dtype=np.float64)
    nperseg = min(window_int, _NPERSEG)
    for index, start in enumerate(starts):
        segment = values[:, start : start + window_int]
        freqs, psd = welch(segment, fs=rate, nperseg=nperseg, axis=1)
        high = _integrate_band(freqs, psd, high_band)
        low = _integrate_band(freqs, psd, low_band)
        trajectories[:, index] = high / np.maximum(low, _POWER_FLOOR)
    return trajectories


def beta_delta_ratio_trajectory(
    signals: FloatArray,
    *,
    rate: float,
    window: int,
    step: int,
    low_band: tuple[float, float] = DELTA_BAND,
    high_band: tuple[float, float] = BETA_BAND,
) -> FloatArray:
    """Return the whole-head beta-to-delta ratio, averaged across channels per window.

    The channel mean of :func:`channel_ratio_trajectories` — the whole-head preictal
    desynchronisation trajectory.

    Parameters
    ----------
    signals : FloatArray
        The multichannel signal, shape ``(channels, samples)``.
    rate, window, step, low_band, high_band :
        As in :func:`channel_ratio_trajectories`.

    Returns
    -------
    FloatArray
        The channel-mean beta/delta ratio per window, shape ``(n_windows,)``.
    """
    trajectories = channel_ratio_trajectories(
        signals,
        rate=rate,
        window=window,
        step=step,
        low_band=low_band,
        high_band=high_band,
    )
    return np.ascontiguousarray(trajectories.mean(axis=0))


def spectral_rise_score(trajectory: RealSeries) -> float:
    """Return the Kendall-τ rising trend of a beta-to-delta ratio trajectory.

    A positive τ is the power ratio climbing toward the seizure — the preictal
    desynchronisation the detector is built to read; τ near zero or negative is no rise.

    Parameters
    ----------
    trajectory : sequence of float
        The beta/delta ratio per window, in time order.

    Returns
    -------
    float
        The Kendall τ against time, in ``[-1, 1]``; ``0.0`` when fewer than two windows
        are given or the trend is undefined (a constant trajectory).
    """
    values = np.asarray(trajectory, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("trajectory must be one-dimensional")
    if values.shape[0] < 2:
        return 0.0
    tau = kendalltau(np.arange(values.shape[0]), values)[0]
    return float(tau) if np.isfinite(tau) else 0.0


def segment_rise_score(
    signals: FloatArray,
    *,
    rate: float,
    window: int,
    step: int,
    aggregation: str = "mean",
    low_band: tuple[float, float] = DELTA_BAND,
    high_band: tuple[float, float] = BETA_BAND,
) -> float:
    """Return one segment's preictal rise score under the chosen channel aggregation.

    ``"mean"`` scores the whole-head channel-averaged ratio trajectory; ``"focal"``
    scores the *most-rising channel* — the maximum per-channel rise — for a spatially
    localised focal onset.

    Parameters
    ----------
    signals : FloatArray
        The segment's multichannel signal, shape ``(channels, samples)``.
    rate, window, step, low_band, high_band :
        As in :func:`channel_ratio_trajectories`.
    aggregation : str
        ``"mean"`` (whole head) or ``"focal"`` (most-rising channel).

    Returns
    -------
    float
        The segment's rising-trend score.

    Raises
    ------
    ValueError
        If ``aggregation`` is neither ``"mean"`` nor ``"focal"``.
    """
    trajectories = channel_ratio_trajectories(
        signals,
        rate=rate,
        window=window,
        step=step,
        low_band=low_band,
        high_band=high_band,
    )
    if aggregation == "mean":
        return spectral_rise_score(trajectories.mean(axis=0))
    if aggregation == "focal":
        return max(spectral_rise_score(channel) for channel in trajectories)
    raise ValueError(f"aggregation must be 'mean' or 'focal', got {aggregation!r}")


@dataclass(frozen=True)
class SeizureSignificance:
    """The seizure detector's matched-false-alarm result on an EEG corpus.

    Attributes
    ----------
    aggregation : str
        The channel aggregation used (``"mean"`` or ``"focal"``).
    score_threshold : float
        The matched-false-alarm rising-trend threshold set on the interictal null.
    achieved_false_alarm : float
        The fraction of null segments that alarmed at that threshold.
    significance : PermutationSignificance
        The label-permutation significance of the seizure alarm count — the same test
        the SCPN suite and the AR(1)-Kendall-τ competitor are scored by.
    """

    aggregation: str
    score_threshold: float
    achieved_false_alarm: float
    significance: PermutationSignificance

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe mapping of the seizure-detector result."""
        return {
            "detector": f"spectral_beta_delta_rise_{self.aggregation}",
            "aggregation": self.aggregation,
            "score_threshold": self.score_threshold,
            "achieved_false_alarm": self.achieved_false_alarm,
            "significance": self.significance.to_audit_record(),
        }


def seizure_significance(
    transition_signals: Sequence[FloatArray],
    null_signals: Sequence[FloatArray],
    *,
    rate: float,
    window: int,
    step: int,
    aggregation: str = "mean",
    low_band: tuple[float, float] = DELTA_BAND,
    high_band: tuple[float, float] = BETA_BAND,
    target_fa: float = DEFAULT_TARGET_FALSE_ALARM,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_PERMUTATION_SEED,
) -> SeizureSignificance:
    """Score the spectral seizure detector through the matched-false-alarm protocol.

    Each preictal and interictal segment is reduced to its beta-to-delta rising-trend
    score under the chosen aggregation; the threshold is calibrated on the interictal
    scores to a matched false alarm; a segment alarms when its trend meets it; and the
    seizure alarm count is tested for significance by the shared label-
    permutation core, so the p-value is directly comparable to the SCPN suite and the
    Dakos competitor.

    Parameters
    ----------
    transition_signals : sequence of FloatArray
        Each seizure's pre-onset segment, shape ``(channels, samples)``.
    null_signals : sequence of FloatArray
        Each interictal null segment, same shape convention.
    rate : float
        Sampling rate in Hz.
    window, step : int
        Analysis window length and hop in samples.
    aggregation : str
        ``"mean"`` (whole head) or ``"focal"`` (most-rising channel).
    low_band, high_band : tuple[float, float]
        The delta and beta bands (Hz).
    target_fa : float
        Target false-alarm rate the trend threshold is held at or below.
    n_permutations : int
        Number of random relabellings for the significance test.
    seed : int
        Seed of the resampling, so the p-value is reproducible.

    Returns
    -------
    SeizureSignificance
        The calibrated threshold, achieved false-alarm rate, and permutation
        significance.

    Raises
    ------
    ValueError
        If either segment set is empty.
    """
    if not transition_signals:
        raise ValueError("transition_signals must not be empty")
    if not null_signals:
        raise ValueError("null_signals must not be empty")

    def _score(signal: FloatArray) -> float:
        return segment_rise_score(
            signal,
            rate=rate,
            window=window,
            step=step,
            aggregation=aggregation,
            low_band=low_band,
            high_band=high_band,
        )

    transition_scores = [_score(signal) for signal in transition_signals]
    null_scores = [_score(signal) for signal in null_signals]
    threshold = calibrate_score_threshold(null_scores, target_fa=target_fa)
    transition_alarms = [score >= threshold for score in transition_scores]
    null_alarms = [score >= threshold for score in null_scores]
    achieved = float(np.mean(null_alarms))
    significance = permutation_significance_from_alarms(
        transition_alarms, null_alarms, n_permutations=n_permutations, seed=seed
    )
    return SeizureSignificance(
        aggregation=aggregation,
        score_threshold=threshold,
        achieved_false_alarm=achieved,
        significance=significance,
    )


def _positive_int(value: int, name: str) -> int:
    """Return ``value`` as a positive integer, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value
