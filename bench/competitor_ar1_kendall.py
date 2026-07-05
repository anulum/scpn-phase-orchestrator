# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — AR(1)-Kendall-tau competitor early-warning detector

"""The Dakos et al. 2008 AR(1)-Kendall-τ detector, for a matched-false-alarm comparison.

The SCPN early-warning suite reports sparse detection at chance across four domains,
but that is a statement about *its* detectors at a matched false-alarm operating point,
not about the wider early-warning-signals literature. The most-cited literature
detector is Dakos et al. 2008 (PNAS 105:14308): the **rising trend of the lag-one
autocorrelation**, quantified by the Kendall rank correlation τ of the windowed AR(1)
against time. Dakos assesses it by per-record trend significance against surrogates —
a different, retrospective question from an operational alarm at a fixed false-alarm
budget.

This module implements that detector as a per-segment score (:func:`ar1_trend_tau`) and
runs it through the *same* matched-false-alarm calibration and label-permutation
significance test the SCPN suite is scored by
(:func:`~bench.early_warning_domain.permutation_significance_from_alarms`), so a
head-to-head is a same-protocol, different-detector comparison: identical transition and
null segments, identical false-alarm budget, identical significance test — only the
per-segment score differs. It answers the question the four-domain result leaves open:
does the canonical literature detector beat chance where the SCPN detectors do not, when
both are held to an honest operating point?

References
----------
* Dakos, Scheffer, van Nes, Brovkin, Petoukhov & Held 2008, *PNAS* 105:14308 — the
  rising lag-one autocorrelation (AR(1)) as an early warning, with its Kendall-τ trend.
* Kendall 1938, *Biometrika* 30, 81 — the rank correlation coefficient τ.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.stats import kendalltau

from bench.early_warning_domain import (
    DEFAULT_PERMUTATION_SEED,
    DEFAULT_PERMUTATIONS,
    DEFAULT_TARGET_FALSE_ALARM,
    PermutationSignificance,
    permutation_significance_from_alarms,
)

if TYPE_CHECKING:  # pragma: no cover - import only for static typing
    from collections.abc import Sequence

FloatArray = NDArray[np.float64]

_VARIANCE_FLOOR = 1.0e-12

__all__ = [
    "Ar1KendallSignificance",
    "ar1_kendall_significance",
    "ar1_trend_tau",
    "calibrate_tau_threshold",
]


def ar1_trend_tau(series: FloatArray, *, window: int, step: int) -> float:
    """Return the Kendall-τ trend of the windowed lag-one autocorrelation.

    The Dakos et al. 2008 early-warning score for one segment: slide a ``window`` across
    the series in hops of ``step``, take each window's lag-one autocorrelation (the same
    mean-detrended lag-one estimate the SCPN critical-slowing-down monitor uses), and
    return the Kendall rank correlation τ between that AR(1) series and window index. A
    positive τ is a rising autocorrelation — critical slowing down approaching a
    transition; τ near zero or negative is no rising trend.

    Parameters
    ----------
    series : FloatArray
        The scalar observable (a detrended proxy residual, or a cross-node order
        parameter), shape ``(T,)``.
    window : int
        Analysis window length in samples; must be at least three for a lag-one
        autocorrelation and no longer than the series.
    step : int
        Hop between consecutive windows in samples; must be positive.

    Returns
    -------
    float
        The Kendall τ of the AR(1) trend, in ``[-1, 1]``; ``0.0`` when fewer than two
        windows fit or the trend is undefined (a constant AR(1) series).

    Raises
    ------
    ValueError
        If ``series`` is malformed or the window does not fit it.
    """
    values = np.asarray(series, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("series must be one-dimensional")
    window_int = _positive_int(window, "window")
    step_int = _positive_int(step, "step")
    n_samples = values.shape[0]
    if window_int < 3:
        raise ValueError(f"window {window_int} must be at least 3 for autocorrelation")
    if window_int > n_samples:
        raise ValueError(f"window {window_int} exceeds the series length {n_samples}")
    starts = range(0, n_samples - window_int + 1, step_int)
    autocorrelations = np.empty(len(starts), dtype=np.float64)
    for index, start in enumerate(starts):
        window_values = values[start : start + window_int]
        centred = window_values - float(np.mean(window_values))
        variance = float(np.dot(centred, centred))
        if variance <= _VARIANCE_FLOOR:
            autocorrelations[index] = 0.0
        else:
            autocorrelations[index] = (
                float(np.dot(centred[:-1], centred[1:])) / variance
            )
    if autocorrelations.shape[0] < 2:
        return 0.0
    tau = kendalltau(np.arange(autocorrelations.shape[0]), autocorrelations)[0]
    return float(tau) if np.isfinite(tau) else 0.0


def calibrate_tau_threshold(
    null_taus: Sequence[float], *, target_fa: float = DEFAULT_TARGET_FALSE_ALARM
) -> float:
    """Return the tightest τ threshold holding the null false-alarm rate at target.

    The competitor analogue of the SCPN suite's continuous matched-false-alarm
    calibration: sort the null segments' τ scores and place the alarm threshold just
    above the ``floor(target_fa · n)``-th largest, so at most that fraction of nulls has
    ``τ ≥ threshold``. There is no clamp — τ already lives in ``[-1, 1]``.

    Parameters
    ----------
    null_taus : sequence of float
        The AR(1)-Kendall-τ score of each no-transition null trial.
    target_fa : float
        Target false-alarm rate the detector is held at or below.

    Returns
    -------
    float
        The matched-false-alarm τ threshold; ``-inf`` (the gate fully open) when every
        null may alarm within the budget.

    Raises
    ------
    ValueError
        If ``null_taus`` is empty.
    """
    if not null_taus:
        raise ValueError("null_taus must not be empty")
    scores = sorted((float(tau) for tau in null_taus), reverse=True)
    allowed = int(np.floor(target_fa * len(scores)))
    if allowed >= len(scores):
        return float(-np.inf)
    return float(np.nextafter(scores[allowed], np.inf))


@dataclass(frozen=True)
class Ar1KendallSignificance:
    """The AR(1)-Kendall-τ detector's matched-false-alarm result on a corpus.

    Attributes
    ----------
    tau_threshold : float
        The matched-false-alarm τ threshold calibrated on the null trials.
    achieved_false_alarm : float
        The fraction of null trials that alarmed at that threshold.
    significance : PermutationSignificance
        The label-permutation significance of the transition alarm count — the same
        test the SCPN suite is scored by.
    """

    tau_threshold: float
    achieved_false_alarm: float
    significance: PermutationSignificance

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe mapping of the competitor result.

        Returns
        -------
        dict[str, object]
            The τ threshold, the achieved false-alarm rate, and the sealed
            significance record.
        """
        return {
            "detector": "ar1_kendall_tau",
            "tau_threshold": self.tau_threshold,
            "achieved_false_alarm": self.achieved_false_alarm,
            "significance": self.significance.to_audit_record(),
        }


def ar1_kendall_significance(
    transition_series: Sequence[FloatArray],
    null_series: Sequence[FloatArray],
    *,
    window: int,
    step: int,
    target_fa: float = DEFAULT_TARGET_FALSE_ALARM,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_PERMUTATION_SEED,
) -> Ar1KendallSignificance:
    """Score the AR(1)-Kendall-τ detector through the matched-false-alarm protocol.

    Each transition segment and null trial is reduced to its AR(1)-Kendall-τ score
    (:func:`ar1_trend_tau`); the threshold is calibrated on the null scores to a matched
    false alarm (:func:`calibrate_tau_threshold`); a segment alarms when its τ meets the
    threshold; and the transition alarm count is tested for significance by the shared
    label-permutation core, so the p-value is directly comparable to the SCPN suite's.

    Parameters
    ----------
    transition_series : sequence of FloatArray
        The scalar observable of each transition segment.
    null_series : sequence of FloatArray
        The scalar observable of each no-transition null trial.
    window, step : int
        Analysis window length and hop for the AR(1) estimate.
    target_fa : float
        Target false-alarm rate the τ threshold is held at or below.
    n_permutations : int
        Number of random relabellings for the significance test.
    seed : int
        Seed of the resampling, so the p-value is reproducible.

    Returns
    -------
    Ar1KendallSignificance
        The calibrated threshold, achieved false-alarm rate, and permutation
        significance.

    Raises
    ------
    ValueError
        If either segment set is empty.
    """
    if not transition_series:
        raise ValueError("transition_series must not be empty")
    if not null_series:
        raise ValueError("null_series must not be empty")
    transition_taus = [
        ar1_trend_tau(series, window=window, step=step) for series in transition_series
    ]
    null_taus = [
        ar1_trend_tau(series, window=window, step=step) for series in null_series
    ]
    threshold = calibrate_tau_threshold(null_taus, target_fa=target_fa)
    transition_alarms = [tau >= threshold for tau in transition_taus]
    null_alarms = [tau >= threshold for tau in null_taus]
    achieved = float(np.mean(null_alarms))
    significance = permutation_significance_from_alarms(
        transition_alarms, null_alarms, n_permutations=n_permutations, seed=seed
    )
    return Ar1KendallSignificance(
        tau_threshold=threshold,
        achieved_false_alarm=achieved,
        significance=significance,
    )


def _positive_int(value: object, name: str) -> int:
    """Return ``value`` as a positive integer, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer, got {result}")
    return result
