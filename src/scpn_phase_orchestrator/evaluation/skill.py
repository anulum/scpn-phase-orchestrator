# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — detector-agnostic early-warning skill primitives

"""Detector-agnostic primitives for scoring early-warning skill honestly.

An early-warning detector is only useful if it fires on genuine pre-transition
segments *more often than its own false-alarm rate on transition-free nulls*. A
raw detection rate hides that: a detector that alarms on 90 % of events but also
on 90 % of quiet nulls has learnt nothing. These primitives make the honest
comparison, on **bounded per-segment scores** alone, so they judge any detector —
the SCPN suite, an AR(1)/Kendall-τ trend baseline, or a black-box deep classifier
— by the single number each emits per segment.

Two operations compose into an audit:

* :func:`calibrate_score_threshold` sets the alarm threshold *from the null
  scores* so at most a target fraction of transition-free segments alarm — the
  matched false-alarm rate. The threshold is placed just above an order statistic
  of the nulls, never on a fixed grid, so a detector whose nulls need a high gate
  is not silently clipped.
* :func:`permutation_significance_from_alarms` then asks whether the event
  segments alarm *more than that matched rate by chance*, under the exchangeability
  null that events and nulls are interchangeable. :func:`surrogate_rank_pvalue`
  is the same one-sided add-one-corrected rank test for a single observed
  statistic against a surrogate ensemble.

The primitives are pure and free of any SCPN detector or observable type; the
suite-specific glue that turns SCPN observables into per-segment scores lives in
the benchmark harness, not here.

References
----------
* Boettiger & Hastings 2012, *J. R. Soc. Interface* 9, 2527 — early-warning
  signals need a null model and a quantified false-positive rate.
* Scheffer et al. 2009, *Nature* 461, 53 — generic early-warning signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "DEFAULT_PERMUTATIONS",
    "DEFAULT_PERMUTATION_SEED",
    "DEFAULT_TARGET_FALSE_ALARM",
    "PermutationSignificance",
    "calibrate_score_threshold",
    "matched_false_alarm_rate",
    "permutation_significance_from_alarms",
    "surrogate_rank_pvalue",
]

#: Fraction of transition-free nulls allowed to alarm when calibrating a threshold.
DEFAULT_TARGET_FALSE_ALARM = 0.10
#: Random relabellings drawn for a permutation p-value.
DEFAULT_PERMUTATIONS = 10000
#: Seed of the permutation resampling, so a p-value is reproducible.
DEFAULT_PERMUTATION_SEED = 0


def _positive_int(value: object, name: str) -> int:
    """Return ``value`` as a positive integer, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer, got {result}")
    return result


def calibrate_score_threshold(
    null_scores: Sequence[float], *, target_fa: float = DEFAULT_TARGET_FALSE_ALARM
) -> float:
    """Return the tightest score threshold holding the null false-alarm rate at target.

    The matched-false-alarm calibrator for a bounded per-segment score — any
    statistic already on its own scale (a Kendall-τ, a growth rate, a classifier
    logit). The null scores are sorted and the alarm threshold placed just above
    the ``floor(target_fa · n)``-th largest, so at most that fraction of nulls has
    ``score ≥ threshold``. A convention of *higher score = more evidence of a
    transition* is assumed; orient a falling statistic by negating it first.

    Parameters
    ----------
    null_scores : sequence of float
        The per-segment score of each transition-free null trial.
    target_fa : float
        Target false-alarm rate the detector is held at or below.

    Returns
    -------
    float
        The matched-false-alarm score threshold; ``-inf`` (the gate fully open)
        when the budget permits every null to alarm.

    Raises
    ------
    ValueError
        If ``null_scores`` is empty or ``target_fa`` is not in ``[0, 1]``.
    """
    if len(null_scores) == 0:
        raise ValueError("null_scores must not be empty")
    if not 0.0 <= target_fa <= 1.0:
        raise ValueError(f"target_fa must be in [0, 1], got {target_fa}")
    scores = sorted((float(score) for score in null_scores), reverse=True)
    allowed = int(np.floor(target_fa * len(scores)))
    if allowed >= len(scores):
        return float(-np.inf)
    return float(np.nextafter(scores[allowed], np.inf))


def matched_false_alarm_rate(null_scores: Sequence[float], threshold: float) -> float:
    """Return the fraction of null scores that alarm at ``threshold``.

    The false-alarm rate a threshold actually achieves on the nulls — reported
    alongside the target so a detector held below target (or, in a degenerate
    corpus, unable to be) is visible rather than assumed.

    Parameters
    ----------
    null_scores : sequence of float
        The per-segment score of each transition-free null trial.
    threshold : float
        The alarm threshold; a null alarms when ``score ≥ threshold``.

    Returns
    -------
    float
        The fraction of nulls with ``score ≥ threshold``.

    Raises
    ------
    ValueError
        If ``null_scores`` is empty.
    """
    if len(null_scores) == 0:
        raise ValueError("null_scores must not be empty")
    alarms = sum(float(score) >= threshold for score in null_scores)
    return alarms / len(null_scores)


@dataclass(frozen=True)
class PermutationSignificance:
    """Whether an event alarm count beats a matched-false-alarm null by chance.

    A threshold calibrated to a matched false alarm makes some fraction of the
    transition-free nulls alarm by construction. The open question a raw event
    alarm count leaves is whether the events alarm *more often than that* — or
    whether the count is what a random relabelling of events and nulls would give.
    This holds the answer as a label-permutation (exchangeability) test.

    Attributes
    ----------
    observed_alarms : int
        Number of event segments that alarmed at the calibrated threshold.
    n_events : int
        Number of event segments tested.
    pooled_alarm_rate : float
        Fraction of the pooled event-and-null segments that alarmed.
    expected_alarms : float
        Alarm count expected under the null (``n_events × pooled_alarm_rate``).
    p_value : float
        One-sided permutation p-value: the fraction of random relabellings whose
        event-slot alarm count reached ``observed_alarms``, with an add-one
        correction so it is never zero.
    n_permutations : int
        Number of random relabellings drawn.
    seed : int
        Seed of the resampling, so the p-value is reproducible.
    """

    observed_alarms: int
    n_events: int
    pooled_alarm_rate: float
    expected_alarms: float
    p_value: float
    n_permutations: int
    seed: int

    def to_record(self) -> dict[str, object]:
        """Return a JSON-safe mapping of the significance test.

        Returns
        -------
        dict[str, object]
            The observed and expected alarm counts, the pooled alarm rate, the
            permutation p-value, and the resampling parameters.
        """
        return {
            "observed_alarms": self.observed_alarms,
            "n_events": self.n_events,
            "pooled_alarm_rate": self.pooled_alarm_rate,
            "expected_alarms": self.expected_alarms,
            "p_value": self.p_value,
            "n_permutations": self.n_permutations,
            "seed": self.seed,
        }


def permutation_significance_from_alarms(
    event_alarms: Sequence[bool],
    null_alarms: Sequence[bool],
    *,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_PERMUTATION_SEED,
) -> PermutationSignificance:
    """Test whether an event alarm count beats a matched-false-alarm null.

    Needs only the binary alarm outcome of each event segment and each null trial,
    so it scores any detector once its per-segment score is thresholded to a
    matched false alarm. Under the null that event segments are exchangeable with
    nulls, drawing ``n_permutations`` random event-sized subsets of the pooled
    outcomes builds the null distribution of the alarm count; the one-sided
    p-value is the fraction reaching the observed count, with an add-one
    correction.

    Parameters
    ----------
    event_alarms : sequence of bool
        Whether each event segment alarmed at its calibrated threshold.
    null_alarms : sequence of bool
        Whether each null trial alarmed at that threshold.
    n_permutations : int
        Number of random relabellings to draw; must be a positive integer.
    seed : int
        Seed of the resampling, so the p-value is reproducible.

    Returns
    -------
    PermutationSignificance
        The observed and expected alarm counts and the permutation p-value.

    Raises
    ------
    ValueError
        If either alarm set is empty or ``n_permutations`` is not positive.
    """
    if len(event_alarms) == 0:
        raise ValueError("event_alarms must not be empty")
    if len(null_alarms) == 0:
        raise ValueError("null_alarms must not be empty")
    draws = _positive_int(n_permutations, "n_permutations")
    events = [bool(alarm) for alarm in event_alarms]
    nulls = [bool(alarm) for alarm in null_alarms]
    observed = int(sum(events))
    n_events = len(events)
    pool = np.array(events + nulls, dtype=bool)
    total = int(pool.shape[0])
    rng = np.random.default_rng(seed)
    reached = 0
    for _ in range(draws):
        subset = rng.permutation(total)[:n_events]
        if int(pool[subset].sum()) >= observed:
            reached += 1
    p_value = (1 + reached) / (draws + 1)
    pooled_rate = float(pool.mean())
    return PermutationSignificance(
        observed_alarms=observed,
        n_events=n_events,
        pooled_alarm_rate=pooled_rate,
        expected_alarms=n_events * pooled_rate,
        p_value=p_value,
        n_permutations=draws,
        seed=int(seed),
    )


def surrogate_rank_pvalue(observed: float, surrogates: Sequence[float]) -> float:
    """Return the one-sided surrogate-rank p-value with the add-one correction.

    The single-statistic counterpart to :func:`permutation_significance_from_alarms`:
    given one observed statistic and an ensemble of surrogate statistics drawn
    under the null (e.g. label-shuffled or phase-randomised), the p-value is the
    fraction of surrogates reaching the observed value, add-one corrected so it is
    never zero. A convention of *higher = stronger* is assumed.

    Parameters
    ----------
    observed : float
        The observed statistic.
    surrogates : sequence of float
        The null-ensemble surrogate statistics.

    Returns
    -------
    float
        ``(1 + #{surrogate ≥ observed}) / (1 + n_surrogates)`` — never zero.

    Raises
    ------
    ValueError
        If ``surrogates`` is empty.
    """
    if len(surrogates) == 0:
        raise ValueError("surrogates must not be empty")
    reached = int(sum(1 for score in surrogates if float(score) >= observed))
    return (1 + reached) / (1 + len(surrogates))
