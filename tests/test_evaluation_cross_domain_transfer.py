# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — honest cross-domain transfer auditor tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.evaluation.cross_domain_transfer import (
    TRANSFER_NEGATIVE,
    TRANSFER_NULL,
    TRANSFER_POSITIVE,
    CrossDomainTransferAudit,
    ScorePair,
    audit_cross_domain_transfer,
    classify_transfer_verdict,
)


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
    rng: np.random.Generator, event_loc: float, count: int
) -> list[ScorePair]:
    """Build a shuffled-source floor ensemble of ``count`` independent controls."""
    return [_pair(rng, event_loc) for _ in range(count)]


class TestAuditCrossDomainTransfer:
    def test_known_negative_returns_transfer_negative(self) -> None:
        # DECISIVE honesty test: the target is detectable within-domain, but the
        # source detector carries no skill on it (CHB-MIT cross-subject shape).
        rng = np.random.default_rng(20260716)
        within = _pair(rng, 2.5)
        transfer = _pair(rng, 0.0)
        controls = _controls(rng, 0.0, 24)
        audit = audit_cross_domain_transfer(
            transfer=transfer,
            within_domain=within,
            shuffled_source=controls,
            source_domain="chbmit_subject_a",
            target_domain="chbmit_subject_b",
            n_permutations=400,
        )
        assert audit.within_domain.beats_chance is True
        assert audit.transfer.beats_chance is False
        assert audit.verdict == TRANSFER_NEGATIVE
        assert audit.transferred is False

    def test_known_positive_returns_transfer_positive(self) -> None:
        rng = np.random.default_rng(7)
        transfer = _pair(rng, 2.0)
        within = _pair(rng, 2.5)
        controls = _controls(rng, 0.0, 40)
        audit = audit_cross_domain_transfer(
            transfer=transfer,
            within_domain=within,
            shuffled_source=controls,
            n_permutations=400,
        )
        assert audit.transfer.beats_chance is True
        assert audit.floor_pvalue < 0.05
        assert audit.transfer_margin > audit.floor_margin_mean
        assert audit.verdict == TRANSFER_POSITIVE
        assert audit.transferred is True

    def test_undetectable_target_returns_transfer_null(self) -> None:
        rng = np.random.default_rng(11)
        transfer = _pair(rng, 0.0)
        within = _pair(rng, 0.0)
        controls = _controls(rng, 0.0, 20)
        audit = audit_cross_domain_transfer(
            transfer=transfer,
            within_domain=within,
            shuffled_source=controls,
            n_permutations=400,
        )
        assert audit.within_domain.beats_chance is False
        assert audit.transfer.beats_chance is False
        assert audit.verdict == TRANSFER_NULL

    def test_skill_indistinguishable_from_floor_returns_transfer_null(self) -> None:
        # Transfer beats its own null, but the scrambled-source floor is at least
        # as skilful — the apparent transfer is a pipeline artefact, not real.
        rng = np.random.default_rng(3)
        transfer = _pair(rng, 1.0)
        within = _pair(rng, 2.5)
        controls = _controls(rng, 1.1, 40)
        audit = audit_cross_domain_transfer(
            transfer=transfer,
            within_domain=within,
            shuffled_source=controls,
            n_permutations=400,
        )
        assert audit.transfer.beats_chance is True
        assert audit.within_domain.beats_chance is True
        assert audit.floor_pvalue >= 0.05
        assert audit.verdict == TRANSFER_NULL
        assert audit.transferred is False

    def test_empty_shuffled_source_rejected(self) -> None:
        rng = np.random.default_rng(9)
        with pytest.raises(ValueError, match="shuffled_source must provide"):
            audit_cross_domain_transfer(
                transfer=_pair(rng, 2.0),
                within_domain=_pair(rng, 2.5),
                shuffled_source=[],
                n_permutations=50,
            )


class TestClassifyTransferVerdict:
    def test_beats_null_and_above_floor_is_positive(self) -> None:
        assert (
            classify_transfer_verdict(
                transfer_beats_own_null=True,
                transfer_above_floor=True,
                within_domain_detectable=True,
            )
            == TRANSFER_POSITIVE
        )

    def test_beats_null_but_not_floor_is_null(self) -> None:
        assert (
            classify_transfer_verdict(
                transfer_beats_own_null=True,
                transfer_above_floor=False,
                within_domain_detectable=True,
            )
            == TRANSFER_NULL
        )

    def test_no_skill_and_undetectable_is_null(self) -> None:
        assert (
            classify_transfer_verdict(
                transfer_beats_own_null=False,
                transfer_above_floor=False,
                within_domain_detectable=False,
            )
            == TRANSFER_NULL
        )

    def test_no_skill_but_detectable_is_negative(self) -> None:
        assert (
            classify_transfer_verdict(
                transfer_beats_own_null=False,
                transfer_above_floor=False,
                within_domain_detectable=True,
            )
            == TRANSFER_NEGATIVE
        )


class TestScorePair:
    def test_normalises_sequences_to_float_tuples(self) -> None:
        pair = ScorePair(event_scores=[1, 2.5], null_scores=(0, -1))
        assert pair.event_scores == (1.0, 2.5)
        assert pair.null_scores == (0.0, -1.0)
        assert all(isinstance(value, float) for value in pair.event_scores)

    def test_empty_event_scores_rejected(self) -> None:
        with pytest.raises(ValueError, match="event_scores must not be empty"):
            ScorePair(event_scores=[], null_scores=[0.0])

    def test_empty_null_scores_rejected(self) -> None:
        with pytest.raises(ValueError, match="null_scores must not be empty"):
            ScorePair(event_scores=[0.0], null_scores=[])

    def test_is_frozen(self) -> None:
        pair = ScorePair(event_scores=[1.0], null_scores=[0.0])
        with pytest.raises(AttributeError):
            # Deliberate frozen-attribute write to assert immutability at runtime.
            pair.event_scores = (2.0,)  # type: ignore[misc]


class TestCrossDomainTransferAuditRecord:
    def _audit(self, seed: int) -> CrossDomainTransferAudit:
        rng = np.random.default_rng(seed)
        return audit_cross_domain_transfer(
            transfer=_pair(rng, 2.0),
            within_domain=_pair(rng, 2.5),
            shuffled_source=_controls(rng, 0.0, 20),
            source_domain="grid_a",
            target_domain="grid_b",
            n_permutations=200,
        )

    def test_to_record_is_json_safe_and_nests_both_arms(self) -> None:
        audit = self._audit(5)
        record = audit.to_record()
        assert record["schema"] == "scpn_cross_domain_transfer_audit_v1"
        assert record["source_domain"] == "grid_a"
        assert record["target_domain"] == "grid_b"
        assert record["verdict"] == audit.verdict
        assert isinstance(record["transfer"], dict)
        assert isinstance(record["within_domain"], dict)
        assert record["n_shuffled_controls"] == 20
        assert record["floor_margin_mean"] == audit.floor_margin_mean

    def test_p_value_property_is_transfer_headline(self) -> None:
        audit = self._audit(6)
        assert audit.p_value == audit.transfer.p_value

    def test_is_frozen(self) -> None:
        audit = self._audit(8)
        with pytest.raises(AttributeError):
            # Deliberate frozen-attribute write to assert immutability at runtime.
            audit.verdict = TRANSFER_NEGATIVE  # type: ignore[misc]
