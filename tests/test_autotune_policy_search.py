# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for replay-only autotune policy search

from __future__ import annotations

from typing import cast, get_type_hints

import pytest

from scpn_phase_orchestrator.autotune import (
    KnobPolicyCandidate,
    OfflinePolicySearchConfig,
    PolicyProposalConfig,
    ReplayPolicyEvaluator,
    ReplayPolicySearchResult,
    RewardObservation,
    search_replay_policy,
)


class TestReplayPolicySearchContract:
    def test_public_contract_is_typed(self) -> None:
        hints = get_type_hints(search_replay_policy)

        assert hints["seed"] is KnobPolicyCandidate
        assert "Callable" in str(hints["evaluator"])
        assert "RewardObservation" in str(hints["evaluator"])
        assert "OfflinePolicySearchConfig" in str(hints["search_config"])
        assert hints["return"] is ReplayPolicySearchResult


class TestReplayPolicySearch:
    def test_evaluates_generated_candidates_and_accepts_best_candidate(self) -> None:
        seed = KnobPolicyCandidate(K=0.2)
        observed: list[KnobPolicyCandidate] = []

        def evaluator(candidate: KnobPolicyCandidate) -> RewardObservation:
            observed.append(candidate)
            coherence = 0.9 if pytest.approx(0.3) == candidate.K else 0.6
            return RewardObservation(coherence=coherence, previous_coherence=0.5)

        result = search_replay_policy(
            seed,
            evaluator,
            search_config=OfflinePolicySearchConfig(
                K_step=0.1,
                alpha_step=0.0,
                zeta_step=0.0,
                Psi_step=0.0,
                channel_weight_step=0.0,
            ),
            proposal_config=PolicyProposalConfig(min_coherence=0.8),
        )

        assert observed == list(result.candidates)
        assert result.proposal.accepted
        assert result.proposal.selected is not None
        assert pytest.approx(0.3) == result.proposal.selected.candidate.K

    def test_rejects_when_best_candidate_misses_coherence_gate(self) -> None:
        def evaluator(candidate: KnobPolicyCandidate) -> RewardObservation:
            return RewardObservation(coherence=0.55, previous_coherence=0.5)

        result = search_replay_policy(
            KnobPolicyCandidate(K=0.2),
            evaluator,
            proposal_config=PolicyProposalConfig(min_coherence=0.8),
        )

        assert not result.proposal.accepted
        assert result.proposal.selected is None
        assert "coherence" in result.proposal.reasons[0]

    def test_rejects_searches_that_generate_no_candidates(self) -> None:
        def evaluator(candidate: KnobPolicyCandidate) -> RewardObservation:
            return RewardObservation(coherence=0.8)

        with pytest.raises(ValueError, match="generated no candidates"):
            search_replay_policy(
                KnobPolicyCandidate(K=0.2),
                evaluator,
                search_config=OfflinePolicySearchConfig(
                    include_baseline=False,
                    K_step=0.0,
                    alpha_step=0.0,
                    zeta_step=0.0,
                    Psi_step=0.0,
                    channel_weight_step=0.0,
                ),
            )

    def test_audit_record_serialises_search_and_proposal(self) -> None:
        def evaluator(candidate: KnobPolicyCandidate) -> RewardObservation:
            return RewardObservation(coherence=0.82, previous_coherence=0.7)

        result = search_replay_policy(
            KnobPolicyCandidate(K=0.2, channel_weights=(1.0, 0.5)),
            evaluator,
            search_config=OfflinePolicySearchConfig(
                K_step=0.0,
                alpha_step=0.0,
                zeta_step=0.0,
                Psi_step=0.0,
                channel_weight_step=0.0,
            ),
        )

        record = result.to_audit_record()
        seed_record = cast("dict[str, object]", record["seed"])
        proposal_record = cast("dict[str, object]", record["proposal"])

        assert seed_record["K"] == 0.2
        assert seed_record["channel_weights"] == [1.0, 0.5]
        assert proposal_record["accepted"] is True

    def test_evaluator_alias_accepts_candidate_to_observation_callable(self) -> None:
        def evaluator(candidate: KnobPolicyCandidate) -> RewardObservation:
            return RewardObservation(coherence=0.75)

        typed_evaluator: ReplayPolicyEvaluator = evaluator

        assert typed_evaluator(KnobPolicyCandidate()).coherence == 0.75
