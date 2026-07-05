# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — ensemble early-warning decision over the detector suite

"""Ensemble early warning that fuses the detector suite into one decision.

The early-warning suite carries three complementary passive indicators of an
approaching synchronisation transition, each on its own observable: critical
slowing down (:mod:`~scpn_phase_orchestrator.monitor.critical_slowing_down`,
rising variance / autocorrelation), rising synchronisation
(:mod:`~scpn_phase_orchestrator.monitor.synchronisation`, the Kuramoto order
parameter) and ordinal-transition entropy
(:mod:`~scpn_phase_orchestrator.monitor.explosive_sync`, a regularisation drop).
A fair head-to-head (``bench/early_warning_leadtime.py``) showed no single
indicator dominates, so the integration question is how to combine them without
cheating the false-alarm budget.

``ensemble_warning`` fuses the members' *per-window oriented evidence* — each
member's robust z-score re-signed so that larger always means more anomalous —
that share a common window grid. Two rules are offered:

* ``weighted`` — a weighted mean of the oriented z-scores crossed against a single
  scalar ``fused_threshold``. The threshold is continuous, so it can be calibrated
  to a matched false-alarm rate on a no-transition null exactly like a single
  detector; this is the rule to use for a fair lead-time comparison.
* ``vote`` — an alarm when at least ``min_votes`` members individually breach their
  own gate. Interpretable and conservative, but its knob is discrete (one operating
  point per vote count), so it calibrates coarsely.

Both rules require a sustained ``persistence`` run over the post-baseline windows
(the fused baseline is the widest of the members', so no member alarms while still
inside its baseline). **The gain from fusion must be shown as an improvement in
matched-false-alarm lead time, never as a raw detection rate**: an OR of the
members (``min_votes = 1``) trivially raises the detection rate by spending the
false-alarm budget, which is not an advantage. The monitor is passive: it reads
the members' warnings and emits a fused record; it never actuates.

References
----------
* Scheffer et al. 2009, *Nature* 461, 53 — generic early-warning signals for
  critical transitions (the framework the fused indicators contribute to).
* Kittler, Hatef, Duin & Matas 1998, *IEEE TPAMI* 20, 226 — combining classifiers
  (the sum / vote fusion rules this monitor specialises to z-score evidence).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]

#: Alarm direction whose warning is a rise above baseline.
RISE = "rise"
#: Alarm direction whose warning is a drop below baseline.
DROP = "drop"

#: Fusion rule: a weighted mean of oriented z-scores against a scalar threshold.
WEIGHTED_RULE = "weighted"
#: Fusion rule: at least ``min_votes`` members breach their own gate.
VOTE_RULE = "vote"

__all__ = [
    "DROP",
    "RISE",
    "VOTE_RULE",
    "WEIGHTED_RULE",
    "EnsembleWarning",
    "MemberContribution",
    "MemberEvidence",
    "ensemble_warning",
    "member_from_critical_slowing_down",
    "member_from_synchronisation",
    "member_from_transition_entropy",
]


@dataclass(frozen=True, slots=True)
class MemberEvidence:
    """One suite member's per-window evidence, aligned on the shared grid.

    Attributes
    ----------
    name : str
        Member label, e.g. ``critical_slowing_down``.
    native_direction : str
        :data:`RISE` or :data:`DROP` — the member's own alarm direction, kept so
        the fused record can report each contribution with its native sign.
    window_starts : IntArray
        First sample index of each window; must match across members.
    oriented_z : FloatArray
        The member's robust z-score re-signed so larger means more anomalous
        (``native_robust_z`` for a rise, its negation for a drop).
    native_robust_z : FloatArray
        The member's own signed robust z-score.
    breaches : BoolArray
        The member's own per-window gate decision (its z and relative gates and
        the post-baseline mask), used by the vote rule.
    baseline_median : float
        Median of the member's raw indicator over its baseline windows (for the
        multi-indicator critical-slowing-down member this is the variance
        indicator's baseline).
    z_threshold : float
        The member's robust z-score gate.
    n_baseline_windows : int
        Number of leading windows the member fitted its baseline on.
    """

    name: str
    native_direction: str
    window_starts: IntArray
    oriented_z: FloatArray
    native_robust_z: FloatArray
    breaches: BoolArray
    baseline_median: float
    z_threshold: float
    n_baseline_windows: int


@dataclass(frozen=True, slots=True)
class MemberContribution:
    """A member's snapshot at the reported window of a fused alarm.

    Attributes
    ----------
    name : str
        Member label.
    direction : str
        The member's native alarm direction.
    robust_z : float
        The member's signed robust z-score at the reported window.
    baseline_median : float
        Median of the member's raw indicator over its baseline windows.
    z_threshold : float
        The member's robust z-score gate.
    breached : bool
        Whether the member's own gate held at the reported window.
    """

    name: str
    direction: str
    robust_z: float
    baseline_median: float
    z_threshold: float
    breached: bool


@dataclass(frozen=True)
class EnsembleWarning:
    """Result of a fused ensemble early-warning sweep.

    Attributes
    ----------
    window_starts : IntArray
        First sample index of each analysis window, shape ``(W,)``.
    fused_score : FloatArray
        Weighted mean of the members' oriented z-scores per window, shape
        ``(W,)``; the headline combined evidence (always computed, both rules).
    vote_count : IntArray
        Number of members breaching their own gate per window, shape ``(W,)``.
    rule : str
        :data:`WEIGHTED_RULE` or :data:`VOTE_RULE`.
    fused_threshold : float
        Scalar gate on ``fused_score`` used by the weighted rule.
    min_votes : int
        Vote count required by the vote rule.
    persistence : int
        Consecutive breaching windows required to alarm.
    n_baseline_windows : int
        Fused baseline boundary (the widest member baseline); no window before it
        may alarm.
    member_names : tuple[str, ...]
        Fused member labels, in order.
    contributions : tuple[MemberContribution, ...]
        Each member's snapshot at the reported window (the alarm window when
        triggered, else the closest fused approach).
    warning_triggered : bool
        Whether a sustained fused breach was found.
    warning_window : int | None
        Index of the first window of the triggering run, or ``None``.
    warning_sample : int | None
        Sample index ``window_starts[warning_window]``, or ``None``.
    """

    window_starts: IntArray = field(repr=False)
    fused_score: FloatArray = field(repr=False)
    vote_count: IntArray = field(repr=False)
    rule: str
    fused_threshold: float
    min_votes: int
    persistence: int
    n_baseline_windows: int
    member_names: tuple[str, ...]
    contributions: tuple[MemberContribution, ...]
    warning_triggered: bool
    warning_window: int | None
    warning_sample: int | None

    def summary(self) -> dict[str, float | int | bool | str | None]:
        """Return a flat scalar summary for logging or metric export.

        Returns
        -------
        dict[str, float | int | bool | str | None]
            The fusion rule, window count, the peak fused score, the peak vote
            count, and the alarm verdict.
        """
        return {
            "rule": self.rule,
            "n_windows": int(self.fused_score.shape[0]),
            "n_baseline_windows": self.n_baseline_windows,
            "max_fused_score": float(self.fused_score.max())
            if self.fused_score.size
            else 0.0,
            "max_vote_count": int(self.vote_count.max()) if self.vote_count.size else 0,
            "warning_triggered": self.warning_triggered,
            "warning_window": self.warning_window,
            "warning_sample": self.warning_sample,
        }


def ensemble_warning(
    members: list[MemberEvidence] | tuple[MemberEvidence, ...],
    *,
    rule: str = WEIGHTED_RULE,
    weights: list[float] | tuple[float, ...] | None = None,
    fused_threshold: float = 3.0,
    min_votes: int = 2,
    persistence: int = 2,
) -> EnsembleWarning:
    """Fuse aligned suite-member evidence into one early-warning decision.

    Parameters
    ----------
    members : sequence of MemberEvidence
        At least one member, all sharing an identical ``window_starts`` grid.
    rule : str
        :data:`WEIGHTED_RULE` (weighted-mean oriented z against ``fused_threshold``)
        or :data:`VOTE_RULE` (at least ``min_votes`` members breach).
    weights : sequence of float or None
        Per-member weights for the weighted rule; defaults to equal weights. Each
        must be a positive finite real and the length must match ``members``.
    fused_threshold : float
        Scalar gate on the weighted-mean oriented z-score; must be non-negative.
    min_votes : int
        Members that must breach for the vote rule; ``1 ≤ min_votes ≤ len(members)``.
    persistence : int
        Consecutive breaching windows required to alarm.

    Returns
    -------
    EnsembleWarning
        The fused score, vote count, per-member contributions, and the alarm
        decision.

    Raises
    ------
    ValueError
        If the members are empty or misaligned, the rule is unknown, the weights
        are malformed, or a control is out of range.
    """
    sealed = _validate_members(members)
    rule = _validate_rule(rule)
    fused_threshold = _validate_non_negative_real(fused_threshold, "fused_threshold")
    persistence = _validate_positive_int(persistence, "persistence")
    min_votes = _validate_min_votes(min_votes, len(sealed))
    weight_array = _validate_weights(weights, len(sealed))

    window_starts = sealed[0].window_starts
    n_windows = int(window_starts.shape[0])
    oriented = np.vstack([member.oriented_z for member in sealed])
    fused_score = np.asarray(
        weight_array @ oriented / float(weight_array.sum()), dtype=np.float64
    )
    breach_matrix = np.vstack([member.breaches for member in sealed])
    vote_count = breach_matrix.sum(axis=0).astype(np.int64)

    boundary = max(member.n_baseline_windows for member in sealed)
    post_baseline = np.arange(n_windows) >= boundary
    if rule == WEIGHTED_RULE:
        breaches = post_baseline & (fused_score >= fused_threshold)
    else:
        breaches = post_baseline & (vote_count >= min_votes)

    warning_window = _first_sustained_breach(breaches, persistence)
    warning_triggered = warning_window is not None
    warning_sample = (
        int(window_starts[warning_window]) if warning_window is not None else None
    )
    report = _report_window(fused_score, boundary, warning_window)
    contributions = tuple(_contribution(member, report) for member in sealed)

    return EnsembleWarning(
        window_starts=np.ascontiguousarray(window_starts, dtype=np.int64),
        fused_score=np.ascontiguousarray(fused_score, dtype=np.float64),
        vote_count=np.ascontiguousarray(vote_count, dtype=np.int64),
        rule=rule,
        fused_threshold=fused_threshold,
        min_votes=min_votes,
        persistence=persistence,
        n_baseline_windows=boundary,
        member_names=tuple(member.name for member in sealed),
        contributions=contributions,
        warning_triggered=warning_triggered,
        warning_window=warning_window,
        warning_sample=warning_sample,
    )


def member_from_critical_slowing_down(warning: object) -> MemberEvidence:
    """Adapt a critical-slowing-down warning into fused member evidence.

    The member's oriented z-score is the detector's ``combined_z`` (already a
    rise), and its gate reconstructs the detector's own per-window breach mask.

    Parameters
    ----------
    warning : object
        A
        :class:`~scpn_phase_orchestrator.monitor.critical_slowing_down.CriticalSlowingDownWarning`
        to align onto the shared fusion grid.

    Returns
    -------
    MemberEvidence
        The oriented per-window evidence, with ``combined_z`` as the oriented
        z-score and the detector's own per-window breach mask reconstructed.

    Raises
    ------
    ValueError
        If ``warning`` is not a
        :class:`~scpn_phase_orchestrator.monitor.critical_slowing_down.CriticalSlowingDownWarning`.
    """
    from scpn_phase_orchestrator.monitor.critical_slowing_down import (
        CriticalSlowingDownWarning,
    )

    if not isinstance(warning, CriticalSlowingDownWarning):
        raise ValueError("warning must be a CriticalSlowingDownWarning")
    breaches = _rise_breaches(
        warning.combined_z,
        warning.relative_rise,
        warning.n_baseline_windows,
        warning.z_threshold,
        warning.rise_threshold,
    )
    return MemberEvidence(
        name="critical_slowing_down",
        native_direction=RISE,
        window_starts=warning.window_starts,
        oriented_z=warning.combined_z,
        native_robust_z=warning.combined_z,
        breaches=breaches,
        baseline_median=warning.baseline_variance,
        z_threshold=warning.z_threshold,
        n_baseline_windows=warning.n_baseline_windows,
    )


def member_from_synchronisation(warning: object) -> MemberEvidence:
    """Adapt a rising-synchronisation warning into fused member evidence.

    Parameters
    ----------
    warning : object
        A
        :class:`~scpn_phase_orchestrator.monitor.synchronisation.SynchronisationWarning`
        to align onto the shared fusion grid.

    Returns
    -------
    MemberEvidence
        The oriented per-window evidence, with the order-parameter robust
        z-score as the oriented score and the detector's own breach mask.

    Raises
    ------
    ValueError
        If ``warning`` is not a
        :class:`~scpn_phase_orchestrator.monitor.synchronisation.SynchronisationWarning`.
    """
    from scpn_phase_orchestrator.monitor.synchronisation import SynchronisationWarning

    if not isinstance(warning, SynchronisationWarning):
        raise ValueError("warning must be a SynchronisationWarning")
    breaches = _rise_breaches(
        warning.robust_z,
        warning.relative_rise,
        warning.n_baseline_windows,
        warning.z_threshold,
        warning.rise_threshold,
    )
    return MemberEvidence(
        name="synchronisation",
        native_direction=RISE,
        window_starts=warning.window_starts,
        oriented_z=warning.robust_z,
        native_robust_z=warning.robust_z,
        breaches=breaches,
        baseline_median=warning.baseline_median,
        z_threshold=warning.z_threshold,
        n_baseline_windows=warning.n_baseline_windows,
    )


def member_from_transition_entropy(warning: object) -> MemberEvidence:
    """Adapt an ordinal-transition-entropy warning into fused member evidence.

    The member warns on a *drop*, so its oriented z-score is the negation of the
    detector's signed ``robust_z``.

    Parameters
    ----------
    warning : object
        An
        :class:`~scpn_phase_orchestrator.monitor.explosive_sync.ExplosiveSyncWarning`
        to align onto the shared fusion grid.

    Returns
    -------
    MemberEvidence
        The oriented per-window evidence; because the member warns on a *drop*,
        the oriented z-score is the negation of the detector's signed
        ``robust_z``.

    Raises
    ------
    ValueError
        If ``warning`` is not an
        :class:`~scpn_phase_orchestrator.monitor.explosive_sync.ExplosiveSyncWarning`.
    """
    from scpn_phase_orchestrator.monitor.explosive_sync import ExplosiveSyncWarning

    if not isinstance(warning, ExplosiveSyncWarning):
        raise ValueError("warning must be an ExplosiveSyncWarning")
    n_windows = int(warning.window_starts.shape[0])
    breaches = (
        (np.arange(n_windows) >= warning.n_baseline_windows)
        & (warning.robust_z <= -warning.z_threshold)
        & (warning.relative_drop >= warning.drop_threshold)
    )
    return MemberEvidence(
        name="transition_entropy",
        native_direction=DROP,
        window_starts=warning.window_starts,
        oriented_z=-warning.robust_z,
        native_robust_z=warning.robust_z,
        breaches=breaches,
        baseline_median=warning.baseline_median,
        z_threshold=warning.z_threshold,
        n_baseline_windows=warning.n_baseline_windows,
    )


def _rise_breaches(
    oriented_z: FloatArray,
    relative_rise: FloatArray,
    n_baseline: int,
    z_threshold: float,
    rise_threshold: float,
) -> BoolArray:
    """Reconstruct a rising member's per-window gate (post-baseline mask)."""
    n_windows = int(oriented_z.shape[0])
    return (
        (np.arange(n_windows) >= n_baseline)
        & (oriented_z >= z_threshold)
        & (relative_rise >= rise_threshold)
    )


def _report_window(
    fused_score: FloatArray, boundary: int, warning_window: int | None
) -> int | None:
    """Return the window whose member snapshots the record should pin.

    The alarm window when triggered, else the strongest fused post-baseline
    approach, or ``None`` when the baseline covers every window.
    """
    if warning_window is not None:
        return warning_window
    n_windows = int(fused_score.shape[0])
    if boundary >= n_windows:
        return None
    tail = fused_score[boundary:]
    return boundary + int(np.argmax(tail))


def _contribution(member: MemberEvidence, report: int | None) -> MemberContribution:
    """Return the member's snapshot at the reported window."""
    if report is None:
        return MemberContribution(
            name=member.name,
            direction=member.native_direction,
            robust_z=0.0,
            baseline_median=member.baseline_median,
            z_threshold=member.z_threshold,
            breached=False,
        )
    return MemberContribution(
        name=member.name,
        direction=member.native_direction,
        robust_z=float(member.native_robust_z[report]),
        baseline_median=member.baseline_median,
        z_threshold=member.z_threshold,
        breached=bool(member.breaches[report]),
    )


def _first_sustained_breach(breaches: BoolArray, persistence: int) -> int | None:
    """Return the start index of the first ``persistence``-long breach run."""
    run = 0
    for index in range(int(breaches.shape[0])):
        if breaches[index]:
            run += 1
            if run >= persistence:
                return index - persistence + 1
        else:
            run = 0
    return None


def _validate_members(
    members: object,
) -> tuple[MemberEvidence, ...]:
    """Return the members as a validated, grid-aligned non-empty tuple."""
    if isinstance(members, (str, bytes)) or not isinstance(members, (list, tuple)):
        raise ValueError("members must be a list or tuple of MemberEvidence")
    sealed = tuple(members)
    if not sealed:
        raise ValueError("members must contain at least one MemberEvidence")
    for position, member in enumerate(sealed):
        if not isinstance(member, MemberEvidence):
            raise ValueError(f"members[{position}] must be a MemberEvidence")
    reference = sealed[0].window_starts
    for position, member in enumerate(sealed[1:], start=1):
        if not np.array_equal(member.window_starts, reference):
            raise ValueError(
                f"members[{position}] window grid does not match members[0]"
            )
    return sealed


def _validate_rule(rule: object) -> str:
    """Return ``rule`` if it is a known fusion rule, else raise."""
    if rule not in (WEIGHTED_RULE, VOTE_RULE):
        raise ValueError(
            f"rule must be {WEIGHTED_RULE!r} or {VOTE_RULE!r}, got {rule!r}"
        )
    return rule


def _validate_weights(weights: object, n_members: int) -> FloatArray:
    """Return validated positive weights, defaulting to equal weights."""
    if weights is None:
        return np.ones(n_members, dtype=np.float64)
    if isinstance(weights, (str, bytes)) or not isinstance(weights, (list, tuple)):
        raise ValueError("weights must be a list or tuple of positive reals or None")
    if len(weights) != n_members:
        raise ValueError(f"weights must have one entry per member ({n_members})")
    values = np.empty(n_members, dtype=np.float64)
    for position, weight in enumerate(weights):
        values[position] = _validate_positive_real(weight, f"weights[{position}]")
    return values


def _validate_min_votes(value: object, n_members: int) -> int:
    """Return ``value`` as a vote count in ``[1, n_members]``, else raise."""
    votes = _validate_positive_int(value, "min_votes")
    if votes > n_members:
        raise ValueError(f"min_votes {votes} exceeds the member count {n_members}")
    return votes


def _validate_positive_int(value: object, name: str) -> int:
    """Return ``value`` as a positive integer, else raise ``ValueError``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer, got {result}")
    return result


def _validate_positive_real(value: object, name: str) -> float:
    """Return ``value`` as a strictly positive finite real, else raise."""
    result = _validate_non_negative_real(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive, got {result}")
    return result


def _validate_non_negative_real(value: object, name: str) -> float:
    """Return ``value`` as a non-negative finite real, else raise."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a non-negative real, got {value!r}")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {result}")
    return result
