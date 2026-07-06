# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — grid modal-growth early-warning detector

"""A domain-specific power-grid oscillation detector — the first that beats chance.

The programme's four-domain result showed generic early warning (critical slowing down,
synchronisation, ordinal entropy) at chance at a matched false-alarm rate, and the
seizure-specific spectral detector (:mod:`bench.seizure_detector`) confirmed that a
domain-specific detector does not automatically win: on a hard, murky modality (a
preictal scalp EEG) it too is at chance. This detector is the counterpoint. Where a
domain carries a *deterministic, detectable* instability signature, the right
domain-specific feature beats the generic suite decisively — and the moat certifies it.

A growing power-grid oscillation is exactly such a signature. When a disturbance leaves
a mode under-damped, the amplitude of the cross-bus voltage deviation grows
*exponentially* — the real part ``σ`` of the mode's eigenvalue is positive (negative
damping), the canonical wide-area-monitoring early-warning quantity. The detector
estimates it directly:

* :func:`cross_bus_deviation` — the per-sample mean absolute deviation of the bus
  voltages from their cross-bus mean, an amplitude envelope of the collective mode;
* :func:`per_bus_deviation` — the same deviation *kept per bus*, so a growth rate can be
  measured on each bus and the most unstable one taken;
* :func:`envelope_growth_rate` — the exponential growth rate ``σ`` of a deviation
  envelope, the slope of its log against time; ``σ > 0`` is a growing (unstable) mode,
  ``σ < 0`` a damped one. A ``recency_top`` weighting lets later samples count for more,
  because a real instability *accelerates* toward onset, so the growth close to the
  disturbance is the most informative.

Two aggregations are offered, mirroring the seizure detector, because grid
instability is spatially localised to a mode/bus cluster:

* ``"mean"`` — the growth rate of the whole-network :func:`cross_bus_deviation`
  envelope, the natural first choice;
* ``"focal"`` — the *most unstable bus's* growth rate, the per-segment maximum over the
  per-bus envelopes. The maximum inflates the statistic under the null, but the
  matched-false-alarm threshold is calibrated on that *same* per-bus maximum over the
  damped null segments, so the multiplicity is absorbed and the comparison stays honest.

An accuracy study on the real PSML corpus (dev/held-out split, every variant disclosed,
no operating point tuned on the held-out half) selected ``"focal"`` aggregation with a
recency weighting as the operating point: it leads roughly half of the instability
transitions at a matched ten-percent false alarm, where the whole-network unweighted
growth rate — and every generic detector — leads far fewer. Those are the module
defaults (:data:`DEFAULT_AGGREGATION`, :data:`DEFAULT_RECENCY_TOP`); the whole-network
unweighted path is retained for the head-to-head comparison.

The per-segment ``σ`` is calibrated to a matched false alarm on damped-disturbance null
segments and tested by the shared label-permutation core
(:func:`~bench.early_warning_domain.permutation_significance_from_alarms`), so its
p-value is directly comparable to the generic suite on the *same* segments. On the PSML
23-bus corpus (Zheng et al. 2021), labelling transitions by *disturbance type* —
generator trips (instability-prone) against damped bus faults and branch trips, a label
independent of the growth statistic, so the comparison is not circular — the growth-rate
detector leads far more transitions than any generic member at the same operating point.

References
----------
* Zheng et al. 2021 — the PSML power-system dataset (23-bus millisecond-level PMU
  measurements) with disturbance-type annotations.
* Kundur 1994, *Power System Stability and Control* — small-signal (modal) stability: a
  mode's eigenvalue real part is its growth rate, the sign of instability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

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

FloatArray = NDArray[np.float64]

#: A deviation floor so a perfectly flat window cannot take a logarithm of zero.
_DEVIATION_FLOOR = 1.0e-12

#: The validated default aggregation — the most unstable bus, un-diluted.
DEFAULT_AGGREGATION = "focal"

#: The validated default recency weighting: later samples (nearer the disturbance, where
#: an instability has accelerated) count up to this many times as much as the earliest.
DEFAULT_RECENCY_TOP = 3.0

__all__ = [
    "DEFAULT_AGGREGATION",
    "DEFAULT_RECENCY_TOP",
    "ModalGrowthSignificance",
    "cross_bus_deviation",
    "envelope_growth_rate",
    "modal_growth_score",
    "modal_growth_significance",
    "per_bus_deviation",
]


def cross_bus_deviation(voltages: FloatArray) -> FloatArray:
    """Return the per-sample cross-bus voltage-deviation envelope.

    At each time sample, the mean absolute deviation of the bus voltages from their
    cross-bus mean — an amplitude envelope of the collective oscillation that grows as a
    mode goes unstable.

    Parameters
    ----------
    voltages : FloatArray
        Per-bus voltage magnitudes, shape ``(buses, samples)`` with at least one bus.

    Returns
    -------
    FloatArray
        The deviation envelope, shape ``(samples,)``.

    Raises
    ------
    ValueError
        If ``voltages`` is not a two-dimensional buses-by-samples array with a bus and a
        sample.
    """
    values = np.asarray(voltages, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("voltages must be two-dimensional (buses × samples)")
    if values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("voltages must have at least one bus and one sample")
    centred = values - values.mean(axis=0, keepdims=True)
    return np.ascontiguousarray(np.abs(centred).mean(axis=0))


def per_bus_deviation(voltages: FloatArray) -> FloatArray:
    """Return the per-bus voltage-deviation envelopes, one per bus.

    The absolute deviation of each bus voltage from the cross-bus mean, *without*
    averaging over the buses — the raw material for the ``"focal"`` aggregation, which
    measures the growth rate on every bus and keeps the most unstable one.

    Parameters
    ----------
    voltages : FloatArray
        Per-bus voltage magnitudes, shape ``(buses, samples)`` with at least one bus.

    Returns
    -------
    FloatArray
        The per-bus deviation envelopes, shape ``(buses, samples)``.

    Raises
    ------
    ValueError
        If ``voltages`` is not a two-dimensional buses-by-samples array with a bus and a
        sample.
    """
    values = np.asarray(voltages, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("voltages must be two-dimensional (buses × samples)")
    if values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("voltages must have at least one bus and one sample")
    centred = values - values.mean(axis=0, keepdims=True)
    return np.ascontiguousarray(np.abs(centred))


def _recency_weighted_slope(
    times: FloatArray, logs: FloatArray, recency_top: float
) -> float:
    """Return the weighted-least-squares slope of ``logs`` on ``times``.

    Linear weights rise from one at the first sample to ``recency_top`` at the last, so
    the growth close to the disturbance dominates the fit. With ``recency_top == 1`` the
    weights are uniform and this reduces to ordinary least squares; the caller reserves
    that case for :func:`numpy.polyfit` and only calls here for a genuine weighting.
    """
    weights = np.linspace(1.0, recency_top, times.shape[0])
    total = weights.sum()
    mean_t = float((weights * times).sum() / total)
    mean_y = float((weights * logs).sum() / total)
    denom = float((weights * (times - mean_t) ** 2).sum())
    return float((weights * (times - mean_t) * (logs - mean_y)).sum() / denom)


def envelope_growth_rate(
    deviation: FloatArray, *, rate: float, recency_top: float = 1.0
) -> float:
    """Return the exponential growth rate of a deviation envelope.

    Fits ``log(envelope)`` linearly against time and returns the slope ``σ``: the real
    part of the dominant mode's eigenvalue. ``σ > 0`` is a growing (unstable) mode,
    ``σ < 0`` a damped one. The envelope is floored away from zero before the logarithm.
    With ``recency_top > 1`` the later samples are up-weighted (see
    :func:`_recency_weighted_slope`), as a real instability accelerates toward onset.

    Parameters
    ----------
    deviation : FloatArray
        The deviation envelope over the segment, shape ``(T,)`` with ``T >= 2``.
    rate : float
        Sampling rate in Hz; must be positive, so ``σ`` is per second.
    recency_top : float
        Weight of the last sample relative to the first, ``>= 1``. ``1.0`` (the default)
        is an unweighted ordinary-least-squares fit.

    Returns
    -------
    float
        The growth rate ``σ`` in inverse seconds; ``0.0`` if the fit is undefined.

    Raises
    ------
    ValueError
        If ``deviation`` is not a one-dimensional array of at least two samples,
        ``rate`` is not positive, or ``recency_top`` is not a finite number ``>= 1``.
    """
    values = np.asarray(deviation, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("deviation must be one-dimensional")
    if values.shape[0] < 2:
        raise ValueError("deviation must have at least two samples")
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("rate must be a positive finite number")
    if not np.isfinite(recency_top) or recency_top < 1.0:
        raise ValueError("recency_top must be a finite number at least one")
    times = np.arange(values.shape[0], dtype=np.float64) / rate
    logs = np.log(np.maximum(values, _DEVIATION_FLOOR))
    if recency_top == 1.0:
        slope = float(np.polyfit(times, logs, 1)[0])
    else:
        slope = _recency_weighted_slope(times, logs, recency_top)
    return slope if np.isfinite(slope) else 0.0


def modal_growth_score(
    segment: FloatArray,
    *,
    rate: float,
    aggregation: str = DEFAULT_AGGREGATION,
    recency_top: float = DEFAULT_RECENCY_TOP,
) -> float:
    """Return one segment's modal growth rate ``σ`` under the chosen aggregation.

    ``"mean"`` scores the growth rate of the whole-network :func:`cross_bus_deviation`
    envelope; ``"focal"`` scores the maximum growth rate over the per-bus envelopes
    (:func:`per_bus_deviation`) — the most unstable bus, un-diluted.

    Parameters
    ----------
    segment : FloatArray
        The segment's per-bus voltages, shape ``(buses, samples)``.
    rate : float
        Sampling rate in Hz.
    aggregation : str
        ``"mean"`` (whole network) or ``"focal"`` (most unstable bus).
    recency_top : float
        Recency weighting passed to :func:`envelope_growth_rate`.

    Returns
    -------
    float
        The growth rate ``σ`` in inverse seconds.

    Raises
    ------
    ValueError
        If ``aggregation`` is neither ``"mean"`` nor ``"focal"``.
    """
    if aggregation == "mean":
        return envelope_growth_rate(
            cross_bus_deviation(segment), rate=rate, recency_top=recency_top
        )
    if aggregation == "focal":
        return max(
            envelope_growth_rate(envelope, rate=rate, recency_top=recency_top)
            for envelope in per_bus_deviation(segment)
        )
    raise ValueError(f"aggregation must be 'mean' or 'focal', got {aggregation!r}")


@dataclass(frozen=True)
class ModalGrowthSignificance:
    """The grid modal-growth detector's matched-false-alarm result on a corpus.

    Attributes
    ----------
    aggregation : str
        The channel aggregation used (``"mean"`` or ``"focal"``).
    recency_top : float
        The recency weighting used in the growth-rate fit.
    score_threshold : float
        The matched-false-alarm growth-rate threshold set on the damped null segments.
    achieved_false_alarm : float
        The fraction of null segments that alarmed at that threshold.
    significance : PermutationSignificance
        The label-permutation significance of the instability alarm count — the same
        test the generic SCPN suite is scored by.
    """

    aggregation: str
    recency_top: float
    score_threshold: float
    achieved_false_alarm: float
    significance: PermutationSignificance

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe mapping of the grid modal-growth result."""
        return {
            "detector": f"modal_envelope_growth_rate_{self.aggregation}",
            "aggregation": self.aggregation,
            "recency_top": self.recency_top,
            "score_threshold": self.score_threshold,
            "achieved_false_alarm": self.achieved_false_alarm,
            "significance": self.significance.to_audit_record(),
        }


def modal_growth_significance(
    transition_segments: Sequence[FloatArray],
    null_segments: Sequence[FloatArray],
    *,
    rate: float,
    aggregation: str = DEFAULT_AGGREGATION,
    recency_top: float = DEFAULT_RECENCY_TOP,
    target_fa: float = DEFAULT_TARGET_FALSE_ALARM,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_PERMUTATION_SEED,
) -> ModalGrowthSignificance:
    """Score the modal-growth detector through the matched-false-alarm protocol.

    Each pre-onset transition segment and damped null segment is reduced to its modal
    growth rate ``σ`` (:func:`modal_growth_score`) under the chosen aggregation and
    recency weighting; the threshold is calibrated on the null rates to a matched false
    alarm; a segment alarms when its ``σ`` meets the threshold; and the instability
    alarm count is tested for significance by the shared label-permutation core, so the
    p-value is directly comparable to the generic suite.

    Parameters
    ----------
    transition_segments : sequence of FloatArray
        Each instability transition's pre-onset segment, shape ``(buses, samples)``.
    null_segments : sequence of FloatArray
        Each damped-disturbance null segment, same shape convention.
    rate : float
        Sampling rate in Hz.
    aggregation : str
        ``"mean"`` (whole network) or ``"focal"`` (most unstable bus).
    recency_top : float
        Recency weighting passed to :func:`envelope_growth_rate`.
    target_fa : float
        Target false-alarm rate the growth-rate threshold is held at or below.
    n_permutations : int
        Number of random relabellings for the significance test.
    seed : int
        Seed of the resampling, so the p-value is reproducible.

    Returns
    -------
    ModalGrowthSignificance
        The aggregation, recency weighting, calibrated threshold, achieved false-alarm
        rate, and permutation significance.

    Raises
    ------
    ValueError
        If either segment set is empty.
    """
    if not transition_segments:
        raise ValueError("transition_segments must not be empty")
    if not null_segments:
        raise ValueError("null_segments must not be empty")
    transition_scores = [
        modal_growth_score(
            s, rate=rate, aggregation=aggregation, recency_top=recency_top
        )
        for s in transition_segments
    ]
    null_scores = [
        modal_growth_score(
            s, rate=rate, aggregation=aggregation, recency_top=recency_top
        )
        for s in null_segments
    ]
    threshold = calibrate_score_threshold(null_scores, target_fa=target_fa)
    transition_alarms = [score >= threshold for score in transition_scores]
    null_alarms = [score >= threshold for score in null_scores]
    achieved = float(np.mean(null_alarms))
    significance = permutation_significance_from_alarms(
        transition_alarms, null_alarms, n_permutations=n_permutations, seed=seed
    )
    return ModalGrowthSignificance(
        aggregation=aggregation,
        recency_top=recency_top,
        score_threshold=threshold,
        achieved_false_alarm=achieved,
        significance=significance,
    )
