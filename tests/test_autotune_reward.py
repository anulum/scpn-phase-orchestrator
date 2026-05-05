# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for autotune reward evaluation

from __future__ import annotations

from typing import get_type_hints

import numpy as np
import pytest

from scpn_phase_orchestrator.autotune import (
    AutotuneRewardReport,
    KnobPolicyCandidate,
    RewardConfig,
    RewardObservation,
    evaluate_knob_policy,
    rank_replay_candidates,
)


class TestAutotuneRewardContract:
    def test_public_contracts_are_typed(self) -> None:
        hints = get_type_hints(evaluate_knob_policy)

        assert hints["candidate"] is KnobPolicyCandidate
        assert hints["observation"] is RewardObservation
        assert hints["return"] is AutotuneRewardReport

    def test_replay_ranking_contract_is_typed(self) -> None:
        hints = get_type_hints(rank_replay_candidates)

        assert "Sequence" in str(hints["replay_candidates"])
        assert "AutotuneRewardReport" in str(hints["return"])

    def test_report_serialises_arrays_for_audit(self) -> None:
        candidate = KnobPolicyCandidate(
            K=np.array([[0.0, 0.2], [0.2, 0.0]], dtype=np.float64),
            alpha=np.zeros((2, 2), dtype=np.float64),
            zeta=0.05,
            Psi=0.1,
            channel_weights=(1.0, 0.5),
        )

        report = evaluate_knob_policy(
            candidate,
            RewardObservation(coherence=0.8, previous_coherence=0.7),
        )
        record = report.to_audit_record()

        assert record["candidate"]["K"] == [[0.0, 0.2], [0.2, 0.0]]
        assert record["candidate"]["channel_weights"] == [1.0, 0.5]
        assert record["observation"]["coherence"] == 0.8


class TestAutotuneRewardScoring:
    def test_better_coherence_scores_higher_under_same_actuation(self) -> None:
        candidate = KnobPolicyCandidate(K=0.1, alpha=0.0, zeta=0.0, Psi=0.0)

        low = evaluate_knob_policy(
            candidate,
            RewardObservation(coherence=0.4, previous_coherence=0.35),
        )
        high = evaluate_knob_policy(
            candidate,
            RewardObservation(coherence=0.8, previous_coherence=0.35),
        )

        assert high.reward > low.reward
        assert high.components["coherence_gain"] > low.components["coherence_gain"]

    def test_bad_coherence_and_unsafe_rollouts_are_penalised(self) -> None:
        candidate = KnobPolicyCandidate(K=0.1)
        safe = evaluate_knob_policy(candidate, RewardObservation(coherence=0.7))
        unsafe = evaluate_knob_policy(
            candidate,
            RewardObservation(coherence=0.2, unsafe=True, regime_changed=True),
        )

        assert unsafe.reward < safe.reward
        assert unsafe.components["bad_coherence"] < 0.0
        assert unsafe.components["unsafe"] < 0.0
        assert unsafe.components["regime_churn"] < 0.0

    def test_larger_actuation_energy_scores_lower(self) -> None:
        observation = RewardObservation(coherence=0.75, previous_coherence=0.7)

        small = evaluate_knob_policy(KnobPolicyCandidate(K=0.1), observation)
        large = evaluate_knob_policy(
            KnobPolicyCandidate(
                K=np.full((3, 3), 2.0, dtype=np.float64),
                alpha=0.5,
                zeta=0.3,
                Psi=0.2,
            ),
            observation,
        )

        assert large.components["actuation"] < small.components["actuation"]
        assert large.reward < small.reward

    def test_custom_penalty_weights_are_applied(self) -> None:
        candidate = KnobPolicyCandidate(K=1.0)
        observation = RewardObservation(coherence=0.2)

        mild = evaluate_knob_policy(
            candidate,
            observation,
            RewardConfig(bad_coherence_penalty=1.0, unsafe_penalty=1.0),
        )
        strict = evaluate_knob_policy(
            candidate,
            observation,
            RewardConfig(bad_coherence_penalty=5.0, unsafe_penalty=1.0),
        )

        assert strict.components["bad_coherence"] < mild.components["bad_coherence"]
        assert strict.reward < mild.reward


class TestAutotuneReplayRanking:
    def test_ranks_replay_candidates_by_reward(self) -> None:
        replay = (
            (
                KnobPolicyCandidate(K=0.2),
                RewardObservation(coherence=0.55, previous_coherence=0.4),
            ),
            (
                KnobPolicyCandidate(K=0.1),
                RewardObservation(coherence=0.82, previous_coherence=0.4),
            ),
            (
                KnobPolicyCandidate(K=0.5),
                RewardObservation(coherence=0.2, previous_coherence=0.4),
            ),
        )

        ranked = rank_replay_candidates(replay)

        assert len(ranked) == 3
        assert ranked[0].observation.coherence == 0.82
        assert ranked[-1].observation.coherence == 0.2
        assert ranked[0].reward >= ranked[1].reward >= ranked[2].reward

    def test_replay_ranking_top_k_limits_reports(self) -> None:
        replay = (
            (KnobPolicyCandidate(K=0.1), RewardObservation(coherence=0.8)),
            (KnobPolicyCandidate(K=0.2), RewardObservation(coherence=0.7)),
        )

        ranked = rank_replay_candidates(replay, top_k=1)

        assert len(ranked) == 1
        assert ranked[0].observation.coherence == 0.8

    def test_replay_ranking_filters_unsafe_by_default(self) -> None:
        replay = (
            (
                KnobPolicyCandidate(K=0.1),
                RewardObservation(coherence=0.9, unsafe=True),
            ),
            (KnobPolicyCandidate(K=0.2), RewardObservation(coherence=0.6)),
        )

        ranked = rank_replay_candidates(replay)

        assert len(ranked) == 1
        assert not ranked[0].observation.unsafe

    def test_replay_ranking_can_include_unsafe_rollouts_for_audit(self) -> None:
        replay = (
            (
                KnobPolicyCandidate(K=0.1),
                RewardObservation(coherence=0.9, unsafe=True),
            ),
            (KnobPolicyCandidate(K=0.2), RewardObservation(coherence=0.6)),
        )

        ranked = rank_replay_candidates(replay, require_safe=False)

        assert len(ranked) == 2
        assert any(report.observation.unsafe for report in ranked)


class TestAutotuneRewardValidation:
    def test_rejects_non_probability_observations(self) -> None:
        with pytest.raises(ValueError, match="coherence"):
            RewardObservation(coherence=1.2)

    def test_rejects_negative_channel_weights(self) -> None:
        with pytest.raises(ValueError, match="channel weight"):
            evaluate_knob_policy(
                KnobPolicyCandidate(channel_weights=(1.0, -0.1)),
                RewardObservation(coherence=0.8),
            )

    def test_rejects_non_finite_knobs(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            evaluate_knob_policy(
                KnobPolicyCandidate(K=np.array([np.nan], dtype=np.float64)),
                RewardObservation(coherence=0.8),
            )

    def test_rejects_negative_config_weights(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            RewardConfig(actuation_penalty=-0.1)

    def test_replay_ranking_rejects_empty_replays(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            rank_replay_candidates(())

    def test_replay_ranking_rejects_empty_safe_filter_result(self) -> None:
        replay = (
            (
                KnobPolicyCandidate(K=0.1),
                RewardObservation(coherence=0.9, unsafe=True),
            ),
        )

        with pytest.raises(ValueError, match="no safe"):
            rank_replay_candidates(replay)

    def test_replay_ranking_rejects_non_positive_top_k(self) -> None:
        replay = ((KnobPolicyCandidate(K=0.1), RewardObservation(coherence=0.9)),)

        with pytest.raises(ValueError, match="top_k"):
            rank_replay_candidates(replay, top_k=0)
