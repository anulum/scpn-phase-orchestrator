# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — detector-agnostic early-warning skill auditor

"""Audit any early-warning detector's event-vs-null skill honestly.

:func:`audit_detector` takes only two arrays of per-segment scores — one on
genuine pre-transition *event* segments, one on transition-free *null* segments —
and returns a :class:`DetectorAudit`: the threshold that holds the null
false-alarm rate at a target, the rate it actually achieved, how many events
alarmed at it, and the label-permutation p-value that says whether the events
alarm *more than that matched rate by chance*. Because it reads scores, not the
detector's internals, it audits the SCPN suite, an AR(1)/Kendall-τ baseline, and
a black-box deep classifier on the same footing.

:func:`audit_scoring_detector` is the convenience wrapper for a detector still
expressed as a callable: it applies a scoring function to each raw event and null
series to obtain the two score arrays, then defers to :func:`audit_detector`.

The verdict carries a ``beats_chance`` boolean only as a convenience at a caller
chosen ``alpha``; the honest quantity is the p-value itself, always reported. An
audit is a statement about the supplied corpus and detector, not a certification
of field performance — seal it with :mod:`~scpn_phase_orchestrator.evaluation.record`
to make the corpus, scores, and verdict tamper-evident.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from scpn_phase_orchestrator.evaluation.skill import (
    DEFAULT_PERMUTATION_SEED,
    DEFAULT_PERMUTATIONS,
    DEFAULT_TARGET_FALSE_ALARM,
    PermutationSignificance,
    calibrate_score_threshold,
    matched_false_alarm_rate,
    permutation_significance_from_alarms,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

#: The raw-segment type a scoring detector consumes — a window, an array, a frame.
_Segment = TypeVar("_Segment")

__all__ = [
    "DEFAULT_ALPHA",
    "DetectorAudit",
    "audit_detector",
    "audit_scoring_detector",
]

#: Significance level below which ``beats_chance`` is reported true by convention.
DEFAULT_ALPHA = 0.05


@dataclass(frozen=True)
class DetectorAudit:
    """The honest event-vs-null skill of one detector at a matched false alarm.

    Attributes
    ----------
    detector_name : str
        A label for the audited detector, carried into the sealed record.
    target_false_alarm : float
        The false-alarm rate the threshold was calibrated to hold at or below.
    matched_threshold : float
        The calibrated alarm threshold; ``-inf`` means the gate is fully open
        because the target permitted every null to alarm.
    achieved_false_alarm : float
        The false-alarm rate the threshold actually held on the null scores.
    n_events : int
        Number of event segments audited.
    n_events_alarmed : int
        Number of event segments that alarmed at ``matched_threshold``.
    detection_rate : float
        Fraction of event segments that alarmed (``n_events_alarmed / n_events``).
    n_nulls : int
        Number of null segments audited.
    significance : PermutationSignificance
        The label-permutation test of the event alarm count against the null.
    alpha : float
        The significance level at which ``beats_chance`` was decided.
    beats_chance : bool
        Whether ``significance.p_value < alpha`` — a convenience, not the finding;
        the p-value is the honest quantity.
    """

    detector_name: str
    target_false_alarm: float
    matched_threshold: float
    achieved_false_alarm: float
    n_events: int
    n_events_alarmed: int
    detection_rate: float
    n_nulls: int
    significance: PermutationSignificance
    alpha: float
    beats_chance: bool

    @property
    def p_value(self) -> float:
        """The permutation p-value — the audit's headline number."""
        return self.significance.p_value

    def to_record(self) -> dict[str, object]:
        """Return a JSON-safe mapping of the audit verdict.

        The ``matched_threshold`` is emitted as the string ``"-inf"`` when the
        gate is fully open, so the record stays strict JSON (a hash of it rejects
        non-finite numbers).

        Returns
        -------
        dict[str, object]
            The detector label, target and achieved false alarm, matched
            threshold, detection counts, nested permutation significance, and the
            ``beats_chance`` decision.
        """
        threshold: object = self.matched_threshold
        if self.matched_threshold == float("-inf"):
            threshold = "-inf"
        return {
            "detector_name": self.detector_name,
            "target_false_alarm": self.target_false_alarm,
            "matched_threshold": threshold,
            "achieved_false_alarm": self.achieved_false_alarm,
            "n_events": self.n_events,
            "n_events_alarmed": self.n_events_alarmed,
            "detection_rate": self.detection_rate,
            "n_nulls": self.n_nulls,
            "significance": self.significance.to_record(),
            "alpha": self.alpha,
            "beats_chance": self.beats_chance,
        }


def audit_detector(
    *,
    event_scores: Sequence[float],
    null_scores: Sequence[float],
    detector_name: str = "detector",
    target_false_alarm: float = DEFAULT_TARGET_FALSE_ALARM,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_PERMUTATION_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> DetectorAudit:
    """Audit a detector's event-vs-null skill at a matched false-alarm rate.

    Calibrate an alarm threshold on the null scores to hold the false-alarm rate
    at ``target_false_alarm``, count how many event scores alarm at it, and test
    that count against the exchangeability null with a label permutation. Higher
    score means more evidence of a transition; orient a falling statistic by
    negating it before calling.

    Parameters
    ----------
    event_scores : sequence of float
        One per-segment score on each genuine pre-transition event segment.
    null_scores : sequence of float
        One per-segment score on each transition-free null segment.
    detector_name : str
        A label for the audited detector.
    target_false_alarm : float
        The false-alarm rate to hold the threshold at or below, in ``[0, 1]``.
    n_permutations : int
        Random relabellings drawn for the permutation p-value.
    seed : int
        Seed of the permutation resampling.
    alpha : float
        Significance level for the convenience ``beats_chance`` flag, in ``[0, 1]``.

    Returns
    -------
    DetectorAudit
        The calibrated threshold, achieved false alarm, detection rate, and the
        permutation significance of the event alarm count.

    Raises
    ------
    ValueError
        If ``event_scores`` is empty or ``alpha`` is not in ``[0, 1]``. (Empty
        ``null_scores`` and an out-of-range ``target_false_alarm`` are rejected by
        :func:`~scpn_phase_orchestrator.evaluation.skill.calibrate_score_threshold`.)
    """
    if len(event_scores) == 0:
        raise ValueError("event_scores must not be empty")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    threshold = calibrate_score_threshold(null_scores, target_fa=target_false_alarm)
    achieved = matched_false_alarm_rate(null_scores, threshold)
    event_alarms = [float(score) >= threshold for score in event_scores]
    null_alarms = [float(score) >= threshold for score in null_scores]
    significance = permutation_significance_from_alarms(
        event_alarms, null_alarms, n_permutations=n_permutations, seed=seed
    )
    n_events = len(event_alarms)
    n_alarmed = int(sum(event_alarms))
    return DetectorAudit(
        detector_name=detector_name,
        target_false_alarm=target_false_alarm,
        matched_threshold=threshold,
        achieved_false_alarm=achieved,
        n_events=n_events,
        n_events_alarmed=n_alarmed,
        detection_rate=n_alarmed / n_events,
        n_nulls=len(null_alarms),
        significance=significance,
        alpha=alpha,
        beats_chance=significance.p_value < alpha,
    )


def audit_scoring_detector(
    *,
    score: Callable[[_Segment], float],
    event_series: Sequence[_Segment],
    null_series: Sequence[_Segment],
    detector_name: str = "detector",
    target_false_alarm: float = DEFAULT_TARGET_FALSE_ALARM,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_PERMUTATION_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> DetectorAudit:
    """Audit a detector expressed as a per-series scoring callable.

    Apply ``score`` to each raw event and null series to obtain the two score
    arrays, then defer to :func:`audit_detector`. A convenience for a detector
    that still lives as a function of a raw window rather than a precomputed
    score; the honest evaluation is identical.

    Parameters
    ----------
    score : callable
        Maps one raw series to a single per-segment score (higher = more evidence
        of a transition).
    event_series : sequence of sequence of float
        The raw pre-transition event segments.
    null_series : sequence of sequence of float
        The raw transition-free null segments.
    detector_name, target_false_alarm, n_permutations, seed, alpha :
        Forwarded to :func:`audit_detector`.

    Returns
    -------
    DetectorAudit
        The audit of the scored segments.

    Raises
    ------
    ValueError
        If ``event_series`` or ``null_series`` is empty.
    """
    if len(event_series) == 0:
        raise ValueError("event_series must not be empty")
    if len(null_series) == 0:
        raise ValueError("null_series must not be empty")
    event_scores = [float(score(series)) for series in event_series]
    null_scores = [float(score(series)) for series in null_series]
    return audit_detector(
        event_scores=event_scores,
        null_scores=null_scores,
        detector_name=detector_name,
        target_false_alarm=target_false_alarm,
        n_permutations=n_permutations,
        seed=seed,
        alpha=alpha,
    )
