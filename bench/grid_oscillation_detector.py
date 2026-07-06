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
* :func:`envelope_growth_rate` — the exponential growth rate ``σ`` of that envelope, the
  slope of its log against time; ``σ > 0`` is a growing (unstable) mode, ``σ < 0`` a
  damped one.

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

__all__ = [
    "ModalGrowthSignificance",
    "cross_bus_deviation",
    "envelope_growth_rate",
    "modal_growth_score",
    "modal_growth_significance",
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


def envelope_growth_rate(deviation: FloatArray, *, rate: float) -> float:
    """Return the exponential growth rate of a deviation envelope.

    Fits ``log(envelope)`` linearly against time and returns the slope ``σ``: the real
    part of the dominant mode's eigenvalue. ``σ > 0`` is a growing (unstable) mode,
    ``σ < 0`` a damped one. The envelope is floored away from zero before the logarithm.

    Parameters
    ----------
    deviation : FloatArray
        The deviation envelope over the segment, shape ``(T,)`` with ``T >= 2``.
    rate : float
        Sampling rate in Hz; must be positive, so ``σ`` is per second.

    Returns
    -------
    float
        The growth rate ``σ`` in inverse seconds; ``0.0`` if the fit is undefined.

    Raises
    ------
    ValueError
        If ``deviation`` is not a one-dimensional array of at least two samples, or
        ``rate`` is not positive.
    """
    values = np.asarray(deviation, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("deviation must be one-dimensional")
    if values.shape[0] < 2:
        raise ValueError("deviation must have at least two samples")
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("rate must be a positive finite number")
    times = np.arange(values.shape[0], dtype=np.float64) / rate
    logs = np.log(np.maximum(values, _DEVIATION_FLOOR))
    slope = np.polyfit(times, logs, 1)[0]
    return float(slope) if np.isfinite(slope) else 0.0


def modal_growth_score(segment: FloatArray, *, rate: float) -> float:
    """Return one segment's modal growth rate ``σ``.

    Composes :func:`cross_bus_deviation` and :func:`envelope_growth_rate`: the growth
    rate of the segment's cross-bus deviation envelope.

    Parameters
    ----------
    segment : FloatArray
        The segment's per-bus voltages, shape ``(buses, samples)``.
    rate : float
        Sampling rate in Hz.

    Returns
    -------
    float
        The growth rate ``σ`` in inverse seconds.
    """
    return envelope_growth_rate(cross_bus_deviation(segment), rate=rate)


@dataclass(frozen=True)
class ModalGrowthSignificance:
    """The grid modal-growth detector's matched-false-alarm result on a corpus.

    Attributes
    ----------
    score_threshold : float
        The matched-false-alarm growth-rate threshold set on the damped null segments.
    achieved_false_alarm : float
        The fraction of null segments that alarmed at that threshold.
    significance : PermutationSignificance
        The label-permutation significance of the instability alarm count — the same
        test the generic SCPN suite is scored by.
    """

    score_threshold: float
    achieved_false_alarm: float
    significance: PermutationSignificance

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe mapping of the grid modal-growth result."""
        return {
            "detector": "modal_envelope_growth_rate",
            "score_threshold": self.score_threshold,
            "achieved_false_alarm": self.achieved_false_alarm,
            "significance": self.significance.to_audit_record(),
        }


def modal_growth_significance(
    transition_segments: Sequence[FloatArray],
    null_segments: Sequence[FloatArray],
    *,
    rate: float,
    target_fa: float = DEFAULT_TARGET_FALSE_ALARM,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_PERMUTATION_SEED,
) -> ModalGrowthSignificance:
    """Score the modal-growth detector through the matched-false-alarm protocol.

    Each pre-onset transition segment and damped null segment is reduced to its modal
    growth rate ``σ`` (:func:`modal_growth_score`); the threshold is calibrated on the
    null rates to a matched false alarm; a segment alarms when its ``σ`` meets the
    threshold; and the instability alarm count is tested for significance by the shared
    label-permutation core, so the p-value is directly comparable to the generic suite.

    Parameters
    ----------
    transition_segments : sequence of FloatArray
        Each instability transition's pre-onset segment, shape ``(buses, samples)``.
    null_segments : sequence of FloatArray
        Each damped-disturbance null segment, same shape convention.
    rate : float
        Sampling rate in Hz.
    target_fa : float
        Target false-alarm rate the growth-rate threshold is held at or below.
    n_permutations : int
        Number of random relabellings for the significance test.
    seed : int
        Seed of the resampling, so the p-value is reproducible.

    Returns
    -------
    ModalGrowthSignificance
        The calibrated threshold, achieved false-alarm rate, and permutation
        significance.

    Raises
    ------
    ValueError
        If either segment set is empty.
    """
    if not transition_segments:
        raise ValueError("transition_segments must not be empty")
    if not null_segments:
        raise ValueError("null_segments must not be empty")
    transition_scores = [modal_growth_score(s, rate=rate) for s in transition_segments]
    null_scores = [modal_growth_score(s, rate=rate) for s in null_segments]
    threshold = calibrate_score_threshold(null_scores, target_fa=target_fa)
    transition_alarms = [score >= threshold for score in transition_scores]
    null_alarms = [score >= threshold for score in null_scores]
    achieved = float(np.mean(null_alarms))
    significance = permutation_significance_from_alarms(
        transition_alarms, null_alarms, n_permutations=n_permutations, seed=seed
    )
    return ModalGrowthSignificance(
        score_threshold=threshold,
        achieved_false_alarm=achieved,
        significance=significance,
    )
