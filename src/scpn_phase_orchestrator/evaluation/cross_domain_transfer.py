# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — honest cross-domain transfer auditing

"""Audit whether a detector trained on one domain transfers to another, honestly.

A detector that scores well on a *new* domain is easy to over-read: the apparent
skill may be the new domain's own structure (any competent detector would find
it), or it may be a scoring-pipeline artefact rather than genuine transfer of what
the source domain taught. Reporting a bare cross-domain detection rate as
"transfer" is exactly the overclaim this module refuses to make — the recorded
CHB-MIT cross-subject result (AUC ≈ 0.50, no transfer) is what an honest harness
must be able to return.

:func:`audit_cross_domain_transfer` therefore judges transfer against **two
orthogonal null models**, reusing the detector-agnostic
:func:`~scpn_phase_orchestrator.evaluation.auditor.audit_detector` for every arm so
the calibration is identical across them:

* **Transfer arm** — the source-trained detector's scores on the *target* domain's
  event and null segments. Its label-permutation p-value asks whether it separates
  the target's events from its nulls more than the matched false alarm explains.
* **Within-domain arm** — a detector trained on the target scoring the same target
  segments. This is the *ceiling*: how detectable the target is at all. If the
  target is undetectable even within-domain, a low transfer number cannot be held
  against transfer — the question is untestable, not answered.
* **Shuffled-source floor** — an ensemble of controls in which the source signal is
  scrambled before scoring the target. Their skill margins form the null a genuine
  transfer must rise above, ranked by the one-sided
  :func:`~scpn_phase_orchestrator.evaluation.skill.surrogate_rank_pvalue`.

The verdict (:data:`TRANSFER_POSITIVE`, :data:`TRANSFER_NULL`,
:data:`TRANSFER_NEGATIVE`) is decided by :func:`classify_transfer_verdict` from
three booleans — transfer beats its own null, transfer rises above the shuffled
floor, and the target is detectable within-domain. A transfer that beats its own
null but not the scrambled floor is reported ``transfer_null`` (indistinguishable
from a pipeline artefact), never upgraded. Crucially, a target that is detectable
within-domain yet whose transfer arm does not beat its own null is a decisive
``transfer_negative`` — the honest failure the CHB-MIT case demands.

Notes
-----
The shuffled-source floor gate can only be significant at ``alpha`` when the
control ensemble is large enough that ``1 / (1 + n_controls) < alpha`` — the
add-one-corrected rank p-value floors there. With too few controls the best a
transfer can earn is ``transfer_null``, never ``transfer_positive``: the harness
fails toward caution, never toward an unsupported positive claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from scpn_phase_orchestrator.evaluation.auditor import (
    DEFAULT_ALPHA,
    DetectorAudit,
    audit_detector,
)
from scpn_phase_orchestrator.evaluation.skill import (
    DEFAULT_PERMUTATION_SEED,
    DEFAULT_PERMUTATIONS,
    DEFAULT_TARGET_FALSE_ALARM,
    surrogate_rank_pvalue,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "TRANSFER_NEGATIVE",
    "TRANSFER_NULL",
    "TRANSFER_POSITIVE",
    "CrossDomainTransferAudit",
    "ScorePair",
    "audit_cross_domain_transfer",
    "classify_transfer_verdict",
]

#: The transfer detector beats its own null *and* rises above the shuffled floor.
TRANSFER_POSITIVE = "transfer_positive"
#: Transfer is inconclusive — beats its own null but not the scrambled floor, or
#: the target is undetectable even within-domain (the question is untestable).
TRANSFER_NULL = "transfer_null"
#: The target is detectable within-domain, yet the transferred detector does not
#: beat its own null — a decisive failure to transfer.
TRANSFER_NEGATIVE = "transfer_negative"


@dataclass(frozen=True)
class ScorePair:
    """A detector's per-segment scores on one domain's event and null segments.

    Attributes
    ----------
    event_scores : tuple[float, ...]
        One score per genuine pre-transition event segment (higher = more
        evidence of a transition).
    null_scores : tuple[float, ...]
        One score per transition-free null segment.

    Any finite-float sequence is accepted at construction and normalised to a
    tuple, so the pair is hashable and its serialisation is deterministic.
    """

    event_scores: tuple[float, ...]
    null_scores: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "event_scores", tuple(float(score) for score in self.event_scores)
        )
        object.__setattr__(
            self, "null_scores", tuple(float(score) for score in self.null_scores)
        )
        if not self.event_scores:
            raise ValueError("event_scores must not be empty")
        if not self.null_scores:
            raise ValueError("null_scores must not be empty")


@dataclass(frozen=True)
class CrossDomainTransferAudit:
    """The honest verdict on whether a source detector transfers to a target.

    Attributes
    ----------
    source_domain : str
        Label of the domain the transferred detector was trained on.
    target_domain : str
        Label of the domain it is being transferred to.
    verdict : str
        One of :data:`TRANSFER_POSITIVE`, :data:`TRANSFER_NULL`,
        :data:`TRANSFER_NEGATIVE`.
    transfer : DetectorAudit
        The audit of the source detector on the target's segments.
    within_domain : DetectorAudit
        The audit of a target-trained detector on the target's segments — the
        detectability ceiling.
    transfer_margin : float
        The transfer arm's detection rate above its achieved false alarm.
    within_domain_margin : float
        The within-domain arm's detection rate above its achieved false alarm.
    floor_margin_mean : float
        Mean skill margin of the shuffled-source control ensemble.
    floor_pvalue : float
        One-sided rank p-value of ``transfer_margin`` against the shuffled floor.
    n_shuffled_controls : int
        Number of shuffled-source controls in the floor ensemble.
    alpha : float
        Significance level at which each gate was decided.
    """

    source_domain: str
    target_domain: str
    verdict: str
    transfer: DetectorAudit
    within_domain: DetectorAudit
    transfer_margin: float
    within_domain_margin: float
    floor_margin_mean: float
    floor_pvalue: float
    n_shuffled_controls: int
    alpha: float

    @property
    def p_value(self) -> float:
        """The transfer arm's permutation p-value — the audit's headline number."""
        return self.transfer.p_value

    @property
    def transferred(self) -> bool:
        """Whether the verdict is a positive transfer — a convenience flag."""
        return self.verdict == TRANSFER_POSITIVE

    def to_record(self) -> dict[str, object]:
        """Return a JSON-safe mapping of the transfer verdict.

        Returns
        -------
        dict[str, object]
            The source and target labels, the verdict, both nested audit records,
            the transfer and within-domain skill margins, the shuffled-source floor
            summary, and the significance level.
        """
        return {
            "schema": "scpn_cross_domain_transfer_audit_v1",
            "source_domain": self.source_domain,
            "target_domain": self.target_domain,
            "verdict": self.verdict,
            "transfer": self.transfer.to_record(),
            "within_domain": self.within_domain.to_record(),
            "transfer_margin": self.transfer_margin,
            "within_domain_margin": self.within_domain_margin,
            "floor_margin_mean": self.floor_margin_mean,
            "floor_pvalue": self.floor_pvalue,
            "n_shuffled_controls": self.n_shuffled_controls,
            "alpha": self.alpha,
        }


def classify_transfer_verdict(
    *,
    transfer_beats_own_null: bool,
    transfer_above_floor: bool,
    within_domain_detectable: bool,
) -> str:
    """Classify a cross-domain transfer verdict from three honesty gates.

    Parameters
    ----------
    transfer_beats_own_null : bool
        Whether the transfer arm's label-permutation p-value is below ``alpha``.
    transfer_above_floor : bool
        Whether the transfer skill margin rises above the shuffled-source floor.
    within_domain_detectable : bool
        Whether a target-trained detector beats its own null on the target — i.e.
        the target is detectable at all.

    Returns
    -------
    str
        :data:`TRANSFER_POSITIVE` when transfer beats its own null and rises above
        the floor; :data:`TRANSFER_NULL` when it beats its own null but not the
        floor (a possible pipeline artefact) or the target is undetectable
        within-domain (untestable); :data:`TRANSFER_NEGATIVE` when the target is
        detectable within-domain yet transfer does not beat its own null.
    """
    if transfer_beats_own_null and transfer_above_floor:
        return TRANSFER_POSITIVE
    if transfer_beats_own_null:
        # Beats its own exchangeability null but not the scrambled-source floor:
        # the apparent skill is not distinguishable from a pipeline artefact.
        return TRANSFER_NULL
    if not within_domain_detectable:
        # Neither transfer nor a within-domain detector separates the target's
        # events from its nulls — transfer is untestable, not refuted.
        return TRANSFER_NULL
    # The target is detectable within-domain, yet the transferred detector does
    # not beat its own null — a genuine, decisive failure to transfer.
    return TRANSFER_NEGATIVE


def _skill_margin(audit: DetectorAudit) -> float:
    """Return the detection rate above the achieved false alarm — corpus skill."""
    return audit.detection_rate - audit.achieved_false_alarm


def audit_cross_domain_transfer(
    *,
    transfer: ScorePair,
    within_domain: ScorePair,
    shuffled_source: Sequence[ScorePair],
    source_domain: str = "source",
    target_domain: str = "target",
    target_false_alarm: float = DEFAULT_TARGET_FALSE_ALARM,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_PERMUTATION_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> CrossDomainTransferAudit:
    """Audit whether a source-trained detector transfers to a target domain.

    Every arm is scored through
    :func:`~scpn_phase_orchestrator.evaluation.auditor.audit_detector` at the same
    matched false alarm, so the transfer, within-domain, and shuffled-source
    controls are calibrated identically. The verdict is decided by
    :func:`classify_transfer_verdict` from the transfer arm's own-null p-value, the
    rank p-value of the transfer margin against the shuffled-source floor, and the
    within-domain arm's detectability.

    Parameters
    ----------
    transfer : ScorePair
        The source-trained detector's scores on the target's event and null
        segments.
    within_domain : ScorePair
        A target-trained detector's scores on the same target segments — the
        detectability ceiling.
    shuffled_source : sequence of ScorePair
        Controls in which the source signal is scrambled before scoring the target;
        their skill margins form the floor a genuine transfer must beat. At least
        one is required, but ``1 / (1 + len) < alpha`` is needed for a positive
        verdict to be reachable.
    source_domain, target_domain : str
        Labels carried into the verdict record.
    target_false_alarm : float
        The false-alarm rate every arm's threshold is calibrated to hold.
    n_permutations : int
        Random relabellings drawn for each arm's label-permutation p-value.
    seed : int
        Seed of the permutation resampling, so the verdict is reproducible.
    alpha : float
        Significance level at which each gate is decided.

    Returns
    -------
    CrossDomainTransferAudit
        The transfer verdict with both audit arms and the shuffled-source floor.

    Raises
    ------
    ValueError
        If ``shuffled_source`` is empty.
    """
    if len(shuffled_source) == 0:
        raise ValueError("shuffled_source must provide at least one control")
    transfer_audit = audit_detector(
        event_scores=transfer.event_scores,
        null_scores=transfer.null_scores,
        detector_name=f"{source_domain}->{target_domain}",
        target_false_alarm=target_false_alarm,
        n_permutations=n_permutations,
        seed=seed,
        alpha=alpha,
    )
    within_audit = audit_detector(
        event_scores=within_domain.event_scores,
        null_scores=within_domain.null_scores,
        detector_name=f"{target_domain}(within)",
        target_false_alarm=target_false_alarm,
        n_permutations=n_permutations,
        seed=seed,
        alpha=alpha,
    )
    floor_margins = [
        _skill_margin(
            audit_detector(
                event_scores=control.event_scores,
                null_scores=control.null_scores,
                detector_name=f"{source_domain}(shuffled)",
                target_false_alarm=target_false_alarm,
                n_permutations=n_permutations,
                seed=seed,
                alpha=alpha,
            )
        )
        for control in shuffled_source
    ]
    transfer_margin = _skill_margin(transfer_audit)
    within_margin = _skill_margin(within_audit)
    floor_pvalue = surrogate_rank_pvalue(transfer_margin, floor_margins)
    verdict = classify_transfer_verdict(
        transfer_beats_own_null=transfer_audit.p_value < alpha,
        transfer_above_floor=floor_pvalue < alpha,
        within_domain_detectable=within_audit.p_value < alpha,
    )
    return CrossDomainTransferAudit(
        source_domain=source_domain,
        target_domain=target_domain,
        verdict=verdict,
        transfer=transfer_audit,
        within_domain=within_audit,
        transfer_margin=transfer_margin,
        within_domain_margin=within_margin,
        floor_margin_mean=sum(floor_margins) / len(floor_margins),
        floor_pvalue=floor_pvalue,
        n_shuffled_controls=len(floor_margins),
        alpha=alpha,
    )
