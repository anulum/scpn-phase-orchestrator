# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — leave-one-domain-out transfer harness

"""Aggregate cross-domain transfer honestly across a leave-one-domain-out sweep.

A single ``audit_cross_domain_transfer`` call answers *does a detector trained
elsewhere transfer to this one target?*. A
claim of **domain-general** transfer is a stronger, and far more easily overstated,
thing: it needs every domain to survive being held out while the pooled remainder
is transferred onto it. This module runs that leave-one-domain-out (LODO) sweep and
aggregates the per-fold verdicts under a rule that, like the single-pair auditor it
builds on, **never upgrades** — it fails toward caution and stays able to return a
decisive negative.

Each :class:`LeaveOneDomainOutFold` carries the already-scored arms for one held-out
target domain: the pooled-source detector's scores on that target, a target-trained
detector's scores on the same segments (the within-domain detectability ceiling),
and a shuffled-source floor ensemble. Scoring the arms is the caller's job — exactly
as it is for the single-pair auditor — so this harness stays a pure, deterministic
aggregation with no hidden training step.

:func:`leave_one_domain_out_transfer` audits every fold and hands the per-fold
verdicts to :func:`classify_lodo_verdict`, which decides the sweep verdict:

* :data:`LODO_NEGATIVE` — at least one fold is a decisive
  :data:`~scpn_phase_orchestrator.evaluation.cross_domain_transfer.TRANSFER_NEGATIVE`
  (the target is detectable within-domain yet the transfer carried no skill). One
  such domain refutes generality; this is the verdict the recorded CHB-MIT
  cross-subject negative must produce, never a laundered aggregate positive.
* :data:`LODO_GENERALISES` — *every* fold is a positive transfer. Only an unbroken
  sweep of positives earns the general claim.
* :data:`LODO_UNTESTABLE` — no fold was detectable even within-domain, so transfer
  is untestable across the whole sweep, not refuted.
* :data:`LODO_INCONCLUSIVE` — anything in between: a mix of positives and nulls with
  no decisive negative and no clean sweep. The evidence neither supports nor refutes
  domain-general transfer.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from scpn_phase_orchestrator.evaluation.auditor import DEFAULT_ALPHA
from scpn_phase_orchestrator.evaluation.cross_domain_transfer import (
    TRANSFER_NEGATIVE,
    TRANSFER_POSITIVE,
    CrossDomainTransferAudit,
    ScorePair,
    audit_cross_domain_transfer,
)
from scpn_phase_orchestrator.evaluation.skill import (
    DEFAULT_PERMUTATION_SEED,
    DEFAULT_PERMUTATIONS,
    DEFAULT_TARGET_FALSE_ALARM,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "LODO_GENERALISES",
    "LODO_INCONCLUSIVE",
    "LODO_NEGATIVE",
    "LODO_UNTESTABLE",
    "LeaveOneDomainOutFold",
    "LeaveOneDomainOutReport",
    "classify_lodo_verdict",
    "leave_one_domain_out_transfer",
]

#: At least one held-out domain is detectable within-domain yet the pooled-source
#: transfer carried no skill on it — a decisive failure of domain-general transfer.
LODO_NEGATIVE = "lodo_negative"
#: Every held-out domain received a positive transfer — the only evidence that earns
#: a domain-general claim.
LODO_GENERALISES = "lodo_generalises"
#: No held-out domain was detectable even within-domain, so transfer is untestable
#: across the sweep rather than refuted.
LODO_UNTESTABLE = "lodo_untestable"
#: A mix of positives and nulls with no decisive negative and no clean sweep — the
#: evidence neither supports nor refutes domain-general transfer.
LODO_INCONCLUSIVE = "lodo_inconclusive"


@dataclass(frozen=True)
class LeaveOneDomainOutFold:
    """The precomputed transfer arms for one held-out target domain.

    Attributes
    ----------
    target_domain : str
        Label of the domain held out on this fold; the pooled remainder of the
        sweep is the transfer source.
    transfer : ScorePair
        The pooled-source detector's scores on the held-out target's event and null
        segments.
    within_domain : ScorePair
        A target-trained detector's scores on the same target segments — the
        detectability ceiling that makes a transfer failure decisive.
    shuffled_source : tuple[ScorePair, ...]
        Controls in which the source signal is scrambled before scoring the target;
        their skill margins form the floor a genuine transfer must beat. At least
        one is required, normalised to a tuple at construction.
    """

    target_domain: str
    transfer: ScorePair
    within_domain: ScorePair
    shuffled_source: tuple[ScorePair, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "shuffled_source", tuple(self.shuffled_source))
        if not self.target_domain:
            raise ValueError("target_domain must be non-empty")
        if not self.shuffled_source:
            raise ValueError("shuffled_source must provide at least one control")


@dataclass(frozen=True)
class LeaveOneDomainOutReport:
    """The aggregated verdict of a leave-one-domain-out transfer sweep.

    Attributes
    ----------
    verdict : str
        One of :data:`LODO_NEGATIVE`, :data:`LODO_GENERALISES`,
        :data:`LODO_UNTESTABLE`, :data:`LODO_INCONCLUSIVE`.
    folds : tuple[CrossDomainTransferAudit, ...]
        The per-fold transfer audits, in the order the folds were supplied.
    n_domains : int
        Number of held-out domains in the sweep.
    n_testable : int
        Number of folds whose held-out target is detectable within-domain — the
        folds on which a transfer failure would be decisive.
    n_positive : int
        Number of folds whose transfer verdict is a positive transfer.
    alpha : float
        Significance level at which every fold gate was decided.
    """

    verdict: str
    folds: tuple[CrossDomainTransferAudit, ...]
    n_domains: int
    n_testable: int
    n_positive: int
    alpha: float

    @property
    def domain_verdicts(self) -> dict[str, str]:
        """Map each held-out target domain to its single-fold transfer verdict."""
        return {fold.target_domain: fold.verdict for fold in self.folds}

    @property
    def verdict_counts(self) -> dict[str, int]:
        """Return the count of each single-fold transfer verdict in the sweep."""
        counts = Counter(fold.verdict for fold in self.folds)
        return {verdict: counts[verdict] for verdict in sorted(counts)}

    def to_record(self) -> dict[str, object]:
        """Return a JSON-safe mapping of the leave-one-domain-out verdict.

        Returns
        -------
        dict[str, object]
            The sweep verdict, the per-domain fold verdicts, the verdict counts,
            the testable and positive fold tallies, the significance level, and the
            full per-fold transfer audit records.
        """
        return {
            "schema": "scpn_leave_one_domain_out_transfer_v1",
            "verdict": self.verdict,
            "n_domains": self.n_domains,
            "n_testable": self.n_testable,
            "n_positive": self.n_positive,
            "alpha": self.alpha,
            "domain_verdicts": self.domain_verdicts,
            "verdict_counts": self.verdict_counts,
            "folds": [fold.to_record() for fold in self.folds],
        }


def classify_lodo_verdict(
    fold_verdicts: Sequence[str],
    *,
    n_testable: int,
) -> str:
    """Aggregate per-fold transfer verdicts into a leave-one-domain-out verdict.

    The rule never upgrades: a single decisive negative refutes generality, and only
    an unbroken sweep of positives earns the general claim.

    Parameters
    ----------
    fold_verdicts : sequence of str
        The single-fold transfer verdicts, one per held-out domain. Must be
        non-empty; the harness enforces at least two folds.
    n_testable : int
        Number of folds whose held-out target is detectable within-domain.

    Returns
    -------
    str
        :data:`LODO_NEGATIVE` if any fold is a decisive transfer negative;
        :data:`LODO_GENERALISES` if every fold is a positive transfer;
        :data:`LODO_UNTESTABLE` if no fold was detectable within-domain;
        :data:`LODO_INCONCLUSIVE` otherwise.
    """
    if any(verdict == TRANSFER_NEGATIVE for verdict in fold_verdicts):
        return LODO_NEGATIVE
    if all(verdict == TRANSFER_POSITIVE for verdict in fold_verdicts):
        return LODO_GENERALISES
    if n_testable == 0:
        return LODO_UNTESTABLE
    return LODO_INCONCLUSIVE


def leave_one_domain_out_transfer(
    folds: Sequence[LeaveOneDomainOutFold],
    *,
    target_false_alarm: float = DEFAULT_TARGET_FALSE_ALARM,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_PERMUTATION_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> LeaveOneDomainOutReport:
    """Run a leave-one-domain-out cross-domain transfer sweep and aggregate it.

    Each fold is audited through ``audit_cross_domain_transfer`` with identical
    calibration, and the per-fold verdicts are aggregated by
    :func:`classify_lodo_verdict`. The source of every fold is labelled as the pooled
    remainder of the sweep (``pooled-not-<target>``).

    Parameters
    ----------
    folds : sequence of LeaveOneDomainOutFold
        The precomputed transfer arms, one per held-out target domain. At least two
        distinct target domains are required.
    target_false_alarm : float
        The false-alarm rate every arm's threshold is calibrated to hold.
    n_permutations : int
        Random relabellings drawn for each arm's label-permutation p-value.
    seed : int
        Seed of the permutation resampling, so the sweep is reproducible.
    alpha : float
        Significance level at which each gate is decided.

    Returns
    -------
    LeaveOneDomainOutReport
        The aggregated sweep verdict with every per-fold transfer audit.

    Raises
    ------
    ValueError
        If fewer than two folds are supplied, or two folds name the same target
        domain.
    """
    if len(folds) < 2:
        raise ValueError("leave-one-domain-out requires at least two domains")
    targets = [fold.target_domain for fold in folds]
    if len(set(targets)) != len(targets):
        raise ValueError("each fold must hold out a distinct target domain")
    audits = tuple(
        audit_cross_domain_transfer(
            transfer=fold.transfer,
            within_domain=fold.within_domain,
            shuffled_source=fold.shuffled_source,
            source_domain=f"pooled-not-{fold.target_domain}",
            target_domain=fold.target_domain,
            target_false_alarm=target_false_alarm,
            n_permutations=n_permutations,
            seed=seed,
            alpha=alpha,
        )
        for fold in folds
    )
    n_testable = sum(1 for audit in audits if audit.within_domain.beats_chance)
    n_positive = sum(1 for audit in audits if audit.verdict == TRANSFER_POSITIVE)
    verdict = classify_lodo_verdict(
        [audit.verdict for audit in audits],
        n_testable=n_testable,
    )
    return LeaveOneDomainOutReport(
        verdict=verdict,
        folds=audits,
        n_domains=len(audits),
        n_testable=n_testable,
        n_positive=n_positive,
        alpha=alpha,
    )
