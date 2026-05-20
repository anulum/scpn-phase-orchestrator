# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for replay-only autotune policy search

from __future__ import annotations

from typing import cast, get_type_hints

import numpy as np
import pytest

from scpn_phase_orchestrator.autotune import (
    AdaptiveReplayPolicySearchConfig,
    AdaptiveReplayPolicySearchResult,
    KnobPolicyCandidate,
    OfflinePolicySearchConfig,
    PolicyProposalConfig,
    ReplayPolicyEvaluator,
    ReplayPolicySearchResult,
    RewardObservation,
    search_adaptive_replay_policy,
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

    def test_adaptive_public_contract_is_typed(self) -> None:
        hints = get_type_hints(search_adaptive_replay_policy)

        assert hints["seed"] is KnobPolicyCandidate
        assert "Callable" in str(hints["evaluator"])
        assert "AdaptiveReplayPolicySearchConfig" in str(hints["adaptive_config"])
        assert hints["return"] is AdaptiveReplayPolicySearchResult


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
                cross_channel_gain_step=0.0,
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
                    cross_channel_gain_step=0.0,
                ),
            )

    def test_audit_record_serialises_search_and_proposal(self) -> None:
        def evaluator(candidate: KnobPolicyCandidate) -> RewardObservation:
            return RewardObservation(coherence=0.82, previous_coherence=0.7)

        result = search_replay_policy(
            KnobPolicyCandidate(
                K=0.2,
                channel_weights=(1.0, 0.5),
                cross_channel_gains=(0.3, 0.4),
            ),
            evaluator,
            search_config=OfflinePolicySearchConfig(
                K_step=0.0,
                alpha_step=0.0,
                zeta_step=0.0,
                Psi_step=0.0,
                channel_weight_step=0.0,
                cross_channel_gain_step=0.0,
            ),
        )

        record = result.to_audit_record()
        seed_record = cast("dict[str, object]", record["seed"])
        proposal_record = cast("dict[str, object]", record["proposal"])

        assert seed_record["K"] == 0.2
        assert seed_record["channel_weights"] == [1.0, 0.5]
        assert seed_record["cross_channel_gains"] == [0.3, 0.4]
        assert proposal_record["accepted"] is True

    def test_audit_record_serialises_array_knobs_deterministically(self) -> None:
        seed = KnobPolicyCandidate(
            K=np.array([[0.0, 0.2], [0.3, 0.0]], dtype=np.float64),
            alpha=np.array([0.1, 0.2], dtype=np.float64),
            zeta=0.05,
            Psi=0.01,
        )

        def evaluator(candidate: KnobPolicyCandidate) -> RewardObservation:
            assert candidate == seed
            return RewardObservation(coherence=0.82, previous_coherence=0.7)

        result = search_replay_policy(
            seed,
            evaluator,
            search_config=OfflinePolicySearchConfig(
                K_step=0.0,
                alpha_step=0.0,
                zeta_step=0.0,
                Psi_step=0.0,
                channel_weight_step=0.0,
                cross_channel_gain_step=0.0,
            ),
        )

        record = result.to_audit_record()
        seed_record = cast("dict[str, object]", record["seed"])

        assert seed_record["K"] == [[0.0, 0.2], [0.3, 0.0]]
        assert seed_record["alpha"] == [0.1, 0.2]

    def test_audit_record_serialises_mutated_array_candidates(self) -> None:
        seed = KnobPolicyCandidate(
            K=np.array([0.0, 0.3], dtype=np.float64),
            alpha=np.array([0.0, 0.1], dtype=np.float64),
            channel_weights=(0.2,),
            cross_channel_gains=(0.15,),
        )

        def evaluator(candidate: KnobPolicyCandidate) -> RewardObservation:
            coherence = 0.75 + float(np.asarray(candidate.K).mean()) * 0.1
            return RewardObservation(coherence=coherence, previous_coherence=0.6)

        result = search_replay_policy(
            seed,
            evaluator,
            search_config=OfflinePolicySearchConfig(
                K_step=0.05,
                alpha_step=0.0,
                zeta_step=0.0,
                Psi_step=0.0,
                channel_weight_step=0.0,
                cross_channel_gain_step=0.0,
            ),
        )

        candidate_records = cast(
            "list[object]",
            result.to_audit_record()["candidates"],
        )
        assert isinstance(candidate_records, list)
        assert any(isinstance(record["K"], list) for record in candidate_records)
        assert any(
            isinstance(record["channel_weights"], list)
            for record in candidate_records
        )

    def test_evaluator_alias_accepts_candidate_to_observation_callable(self) -> None:
        def evaluator(candidate: KnobPolicyCandidate) -> RewardObservation:
            return RewardObservation(coherence=0.75)

        typed_evaluator: ReplayPolicyEvaluator = evaluator

        assert typed_evaluator(KnobPolicyCandidate()).coherence == 0.75


class TestAdaptiveReplayPolicySearch:
    def test_refines_seed_across_replay_rounds(self) -> None:
        evaluated: list[float] = []

        def evaluator(candidate: KnobPolicyCandidate) -> RewardObservation:
            assert isinstance(candidate.K, float)
            evaluated.append(candidate.K)
            coherence = max(0.0, 1.0 - abs(candidate.K - 0.4))
            return RewardObservation(coherence=coherence, previous_coherence=0.5)

        result = search_adaptive_replay_policy(
            KnobPolicyCandidate(K=0.0),
            evaluator,
            adaptive_config=AdaptiveReplayPolicySearchConfig(
                base_search_config=OfflinePolicySearchConfig(
                    K_step=0.2,
                    alpha_step=0.0,
                    zeta_step=0.0,
                    Psi_step=0.0,
                    channel_weight_step=0.0,
                    cross_channel_gain_step=0.0,
                    max_abs_knob=1.0,
                ),
                iterations=2,
                step_decay=1.0,
            ),
            proposal_config=PolicyProposalConfig(min_coherence=0.8),
        )

        assert len(result.rounds) == 2
        assert 0.2 in evaluated
        assert 0.4 in evaluated
        assert result.proposal.accepted
        assert result.proposal.selected is not None
        assert result.proposal.selected.candidate.K == 0.4

    def test_final_adaptive_proposal_keeps_rejection_gates(self) -> None:
        def evaluator(candidate: KnobPolicyCandidate) -> RewardObservation:
            return RewardObservation(coherence=0.5, previous_coherence=0.4)

        result = search_adaptive_replay_policy(
            KnobPolicyCandidate(K=0.0),
            evaluator,
            adaptive_config=AdaptiveReplayPolicySearchConfig(iterations=1),
            proposal_config=PolicyProposalConfig(min_coherence=0.8),
        )

        assert not result.proposal.accepted
        assert result.proposal.selected is None
        assert "coherence" in result.proposal.reasons[0]

    def test_adaptive_search_rejects_empty_candidate_round(self) -> None:
        def evaluator(candidate: KnobPolicyCandidate) -> RewardObservation:
            return RewardObservation(coherence=0.8)

        with pytest.raises(ValueError, match="generated no candidates"):
            search_adaptive_replay_policy(
                KnobPolicyCandidate(K=0.2),
                evaluator,
                adaptive_config=AdaptiveReplayPolicySearchConfig(
                    base_search_config=OfflinePolicySearchConfig(
                        include_baseline=False,
                        K_step=0.0,
                        alpha_step=0.0,
                        zeta_step=0.0,
                        Psi_step=0.0,
                        channel_weight_step=0.0,
                        cross_channel_gain_step=0.0,
                    ),
                ),
            )

    def test_adaptive_audit_record_serialises_rounds_and_config(self) -> None:
        def evaluator(candidate: KnobPolicyCandidate) -> RewardObservation:
            return RewardObservation(coherence=0.8, previous_coherence=0.7)

        result = search_adaptive_replay_policy(
            KnobPolicyCandidate(K=0.1),
            evaluator,
            adaptive_config=AdaptiveReplayPolicySearchConfig(iterations=1),
        )

        record = result.to_audit_record()
        config_record = cast("dict[str, object]", record["config"])
        base_config = cast("dict[str, object]", config_record["base_search_config"])
        rounds_record = cast("list[object]", record["rounds"])

        assert config_record["iterations"] == 1
        assert "cross_channel_gain_step" in base_config
        assert len(rounds_record) == 1
        assert cast("dict[str, object]", record["proposal"])["accepted"] is True

    def test_adaptive_min_step_zeroes_decayed_search_steps(self) -> None:
        evaluated: list[float] = []

        def evaluator(candidate: KnobPolicyCandidate) -> RewardObservation:
            assert isinstance(candidate.K, float)
            evaluated.append(candidate.K)
            return RewardObservation(coherence=0.7, previous_coherence=0.6)

        result = search_adaptive_replay_policy(
            KnobPolicyCandidate(K=0.0),
            evaluator,
            adaptive_config=AdaptiveReplayPolicySearchConfig(
                base_search_config=OfflinePolicySearchConfig(
                    K_step=0.05,
                    alpha_step=0.0,
                    zeta_step=0.0,
                    Psi_step=0.0,
                    channel_weight_step=0.0,
                    cross_channel_gain_step=0.0,
                ),
                iterations=2,
                step_decay=0.5,
                min_step=0.04,
            ),
        )

        assert [round_result.candidates for round_result in result.rounds] == [
            (
                KnobPolicyCandidate(K=0.0),
                KnobPolicyCandidate(K=-0.05),
                KnobPolicyCandidate(K=0.05),
            ),
            (KnobPolicyCandidate(K=0.0),),
        ]
        assert evaluated == [0.0, -0.05, 0.05, 0.0]

    def test_adaptive_config_validates_bounds(self) -> None:
        with pytest.raises(ValueError, match="iterations"):
            AdaptiveReplayPolicySearchConfig(iterations=0)
        with pytest.raises(ValueError, match="step_decay"):
            AdaptiveReplayPolicySearchConfig(step_decay=0.0)
        with pytest.raises(ValueError, match="improvement_tolerance"):
            AdaptiveReplayPolicySearchConfig(improvement_tolerance=-0.1)
        with pytest.raises(ValueError, match="min_step"):
            AdaptiveReplayPolicySearchConfig(min_step=-0.1)
