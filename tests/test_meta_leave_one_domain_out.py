# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — leave-one-domain-out transfer harness tests

from __future__ import annotations

import json

import numpy as np
import pytest

from scpn_phase_orchestrator.evaluation.cross_domain_transfer import (
    TRANSFER_NEGATIVE,
    TRANSFER_NULL,
    TRANSFER_POSITIVE,
    ScorePair,
)
from scpn_phase_orchestrator.meta.leave_one_domain_out import (
    LODO_GENERALISES,
    LODO_INCONCLUSIVE,
    LODO_NEGATIVE,
    LODO_UNTESTABLE,
    LeaveOneDomainOutFold,
    LeaveOneDomainOutReport,
    classify_lodo_verdict,
    leave_one_domain_out_transfer,
)

_N_PERMUTATIONS = 400
_N_CONTROLS = 24


def _pair(
    rng: np.random.Generator,
    event_loc: float,
    *,
    n_events: int = 60,
    n_nulls: int = 300,
) -> ScorePair:
    """Build a score pair whose events sit at ``event_loc`` above zero-mean nulls."""
    return ScorePair(
        event_scores=rng.normal(event_loc, 1.0, n_events),
        null_scores=rng.normal(0.0, 1.0, n_nulls),
    )


def _controls(
    rng: np.random.Generator, event_loc: float, count: int = _N_CONTROLS
) -> tuple[ScorePair, ...]:
    """Build a shuffled-source floor ensemble of ``count`` independent controls."""
    return tuple(_pair(rng, event_loc) for _ in range(count))


def _positive_fold(domain: str, seed: int) -> LeaveOneDomainOutFold:
    """A fold whose pooled-source transfer genuinely carries skill to the target."""
    rng = np.random.default_rng(seed)
    return LeaveOneDomainOutFold(
        target_domain=domain,
        transfer=_pair(rng, 2.0),
        within_domain=_pair(rng, 2.5),
        shuffled_source=_controls(rng, 0.0),
    )


def _negative_fold(domain: str, seed: int) -> LeaveOneDomainOutFold:
    """A CHB-MIT-shaped fold: detectable within-domain, no transfer skill."""
    rng = np.random.default_rng(seed)
    return LeaveOneDomainOutFold(
        target_domain=domain,
        transfer=_pair(rng, 0.0),
        within_domain=_pair(rng, 2.5),
        shuffled_source=_controls(rng, 0.0),
    )


def _undetectable_fold(domain: str, seed: int) -> LeaveOneDomainOutFold:
    """A fold whose target is undetectable even within-domain — untestable."""
    rng = np.random.default_rng(seed)
    return LeaveOneDomainOutFold(
        target_domain=domain,
        transfer=_pair(rng, 0.0),
        within_domain=_pair(rng, 0.0),
        shuffled_source=_controls(rng, 0.0),
    )


class TestLeaveOneDomainOutFold:
    def test_normalises_shuffled_source_to_tuple(self) -> None:
        rng = np.random.default_rng(1)
        fold = LeaveOneDomainOutFold(
            target_domain="grid",
            transfer=_pair(rng, 2.0),
            within_domain=_pair(rng, 2.5),
            shuffled_source=list(_controls(rng, 0.0, 3)),
        )
        assert isinstance(fold.shuffled_source, tuple)
        assert len(fold.shuffled_source) == 3

    def test_empty_target_domain_rejected(self) -> None:
        rng = np.random.default_rng(2)
        with pytest.raises(ValueError, match="target_domain must be non-empty"):
            LeaveOneDomainOutFold(
                target_domain="",
                transfer=_pair(rng, 2.0),
                within_domain=_pair(rng, 2.5),
                shuffled_source=_controls(rng, 0.0, 2),
            )

    def test_empty_shuffled_source_rejected(self) -> None:
        rng = np.random.default_rng(3)
        with pytest.raises(ValueError, match="shuffled_source must provide"):
            LeaveOneDomainOutFold(
                target_domain="grid",
                transfer=_pair(rng, 2.0),
                within_domain=_pair(rng, 2.5),
                shuffled_source=(),
            )

    def test_is_frozen(self) -> None:
        fold = _positive_fold("grid", 4)
        with pytest.raises(AttributeError):
            # Deliberate frozen-attribute write to assert immutability at runtime.
            fold.target_domain = "other"  # type: ignore[misc]


class TestClassifyLodoVerdict:
    def test_any_negative_is_lodo_negative(self) -> None:
        # A decisive negative dominates even a positive on another domain.
        assert (
            classify_lodo_verdict(
                [TRANSFER_POSITIVE, TRANSFER_NEGATIVE],
                n_testable=2,
            )
            == LODO_NEGATIVE
        )

    def test_all_positive_is_lodo_generalises(self) -> None:
        assert (
            classify_lodo_verdict(
                [TRANSFER_POSITIVE, TRANSFER_POSITIVE],
                n_testable=2,
            )
            == LODO_GENERALISES
        )

    def test_all_null_and_untestable_is_lodo_untestable(self) -> None:
        assert (
            classify_lodo_verdict(
                [TRANSFER_NULL, TRANSFER_NULL],
                n_testable=0,
            )
            == LODO_UNTESTABLE
        )

    def test_mixed_positive_and_null_is_inconclusive(self) -> None:
        assert (
            classify_lodo_verdict(
                [TRANSFER_POSITIVE, TRANSFER_NULL],
                n_testable=1,
            )
            == LODO_INCONCLUSIVE
        )


class TestLeaveOneDomainOutTransfer:
    def test_all_folds_positive_generalises(self) -> None:
        report = leave_one_domain_out_transfer(
            [_positive_fold("grid", 7), _positive_fold("eeg", 8)],
            n_permutations=_N_PERMUTATIONS,
        )
        assert report.verdict == LODO_GENERALISES
        assert report.n_domains == 2
        assert report.n_positive == 2
        assert report.n_testable == 2
        assert all(audit.transferred for audit in report.folds)

    def test_one_detectable_untransferred_fold_is_negative(self) -> None:
        # REGRESSION-LOCK: the recorded CHB-MIT cross-subject negative must surface
        # as a decisive LODO_NEGATIVE, never be laundered into an aggregate positive.
        report = leave_one_domain_out_transfer(
            [_positive_fold("grid", 7), _negative_fold("chbmit", 20260716)],
            n_permutations=_N_PERMUTATIONS,
        )
        assert report.verdict == LODO_NEGATIVE
        assert report.domain_verdicts["chbmit"] == TRANSFER_NEGATIVE
        assert report.n_testable == 2
        assert report.n_positive == 1

    def test_all_undetectable_folds_are_untestable(self) -> None:
        report = leave_one_domain_out_transfer(
            [_undetectable_fold("grid", 11), _undetectable_fold("eeg", 12)],
            n_permutations=_N_PERMUTATIONS,
        )
        assert report.verdict == LODO_UNTESTABLE
        assert report.n_testable == 0
        assert report.n_positive == 0
        assert all(audit.verdict == TRANSFER_NULL for audit in report.folds)

    def test_mixed_positive_and_untestable_is_inconclusive(self) -> None:
        report = leave_one_domain_out_transfer(
            [_positive_fold("grid", 7), _undetectable_fold("eeg", 11)],
            n_permutations=_N_PERMUTATIONS,
        )
        assert report.verdict == LODO_INCONCLUSIVE
        assert report.n_positive == 1
        assert report.n_testable == 1

    def test_source_domain_labelled_pooled_remainder(self) -> None:
        report = leave_one_domain_out_transfer(
            [_positive_fold("grid", 7), _positive_fold("eeg", 8)],
            n_permutations=200,
        )
        assert report.folds[0].source_domain == "pooled-not-grid"
        assert report.folds[1].source_domain == "pooled-not-eeg"

    def test_fewer_than_two_folds_rejected(self) -> None:
        with pytest.raises(ValueError, match="requires at least two domains"):
            leave_one_domain_out_transfer([_positive_fold("grid", 7)])

    def test_duplicate_target_domain_rejected(self) -> None:
        with pytest.raises(ValueError, match="distinct target domain"):
            leave_one_domain_out_transfer(
                [_positive_fold("grid", 7), _positive_fold("grid", 8)]
            )


class TestLeaveOneDomainOutReport:
    def _report(self) -> LeaveOneDomainOutReport:
        return leave_one_domain_out_transfer(
            [_positive_fold("grid", 7), _negative_fold("chbmit", 20260716)],
            n_permutations=_N_PERMUTATIONS,
        )

    def test_domain_verdicts_maps_each_target(self) -> None:
        report = self._report()
        assert set(report.domain_verdicts) == {"grid", "chbmit"}
        assert report.domain_verdicts["grid"] == TRANSFER_POSITIVE
        assert report.domain_verdicts["chbmit"] == TRANSFER_NEGATIVE

    def test_verdict_counts_are_sorted_and_total(self) -> None:
        report = self._report()
        counts = report.verdict_counts
        assert sum(counts.values()) == report.n_domains
        assert list(counts) == sorted(counts)
        assert counts[TRANSFER_NEGATIVE] == 1
        assert counts[TRANSFER_POSITIVE] == 1

    def test_to_record_is_json_safe(self) -> None:
        report = self._report()
        record = report.to_record()
        assert record["schema"] == "scpn_leave_one_domain_out_transfer_v1"
        assert record["verdict"] == LODO_NEGATIVE
        assert record["n_domains"] == 2
        assert isinstance(record["folds"], list)
        assert len(record["folds"]) == 2
        # Every nested fold audit round-trips through JSON without loss.
        encoded = json.dumps(record)
        assert json.loads(encoded)["verdict"] == LODO_NEGATIVE

    def test_is_frozen(self) -> None:
        report = self._report()
        with pytest.raises(AttributeError):
            # Deliberate frozen-attribute write to assert immutability at runtime.
            report.verdict = LODO_GENERALISES  # type: ignore[misc]
