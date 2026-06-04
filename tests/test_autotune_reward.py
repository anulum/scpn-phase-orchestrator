# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for autotune reward evaluation

from __future__ import annotations

from typing import cast, get_type_hints

import numpy as np
import pytest

import scpn_phase_orchestrator.autotune.policy_search as policy_search_module
import scpn_phase_orchestrator.autotune.reward as reward_module
from scpn_phase_orchestrator.autotune import (
    AutotunePolicyProposal,
    AutotuneRewardReport,
    KnobPolicyCandidate,
    OfflinePolicySearchConfig,
    PolicyProposalConfig,
    RewardConfig,
    RewardObservation,
    SafetyConstraintConfig,
    evaluate_knob_policy,
    generate_offline_policy_candidates,
    propose_replay_policy,
    rank_replay_candidates,
)


class TestAutotuneRewardContract:
    def test_public_contracts_are_typed(self) -> None:
        hints = get_type_hints(evaluate_knob_policy)

        assert reward_module.evaluate_knob_policy is evaluate_knob_policy
        assert hints["candidate"] is KnobPolicyCandidate
        assert hints["observation"] is RewardObservation
        assert hints["return"] is AutotuneRewardReport

    def test_replay_ranking_contract_is_typed(self) -> None:
        hints = get_type_hints(rank_replay_candidates)

        assert "Sequence" in str(hints["replay_candidates"])
        assert "AutotuneRewardReport" in str(hints["return"])

    def test_offline_generator_contract_is_typed(self) -> None:
        hints = get_type_hints(generate_offline_policy_candidates)

        assert hints["seed"] is KnobPolicyCandidate
        assert "OfflinePolicySearchConfig" in str(hints["config"])
        assert "KnobPolicyCandidate" in str(hints["return"])

    def test_policy_proposal_contract_is_typed(self) -> None:
        hints = get_type_hints(propose_replay_policy)

        assert policy_search_module.propose_replay_policy is propose_replay_policy
        assert "Sequence" in str(hints["replay_candidates"])
        assert "PolicyProposalConfig" in str(hints["proposal_config"])
        assert hints["return"] is AutotunePolicyProposal

    def test_report_serialises_arrays_for_audit(self) -> None:
        candidate = KnobPolicyCandidate(
            K=np.array([[0.0, 0.2], [0.2, 0.0]], dtype=np.float64),
            alpha=np.zeros((2, 2), dtype=np.float64),
            zeta=0.05,
            Psi=0.1,
            channel_weights=(1.0, 0.5),
            cross_channel_gains=(0.25, 0.4),
        )

        report = evaluate_knob_policy(
            candidate,
            RewardObservation(coherence=0.8, previous_coherence=0.7),
        )
        record = report.to_audit_record()

        assert record["candidate"]["K"] == [[0.0, 0.2], [0.2, 0.0]]
        assert record["candidate"]["channel_weights"] == [1.0, 0.5]
        assert record["candidate"]["cross_channel_gains"] == [0.25, 0.4]
        assert record["observation"]["coherence"] == 0.8
        assert record["observation"]["lyapunov_exponent"] is None
        assert record["observation"]["stl_robustness"] is None
        assert record["observation"]["safety_cost"] == 0.0


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

    def test_empty_array_knobs_have_zero_actuation_energy(self) -> None:
        candidate = KnobPolicyCandidate(K=np.array([], dtype=np.float64))
        observation = RewardObservation(coherence=0.75, previous_coherence=0.75)

        report = evaluate_knob_policy(candidate, observation)

        assert report.components["actuation"] == 0.0
        assert report.reward == pytest.approx(report.components["target_tracking"])

    def test_reward_component_order_controls_scoring_terms(self) -> None:
        candidate = KnobPolicyCandidate(K=0.1)
        observation = RewardObservation(coherence=0.6, previous_coherence=0.4)

        full = evaluate_knob_policy(candidate, observation)
        coherence_only = evaluate_knob_policy(
            candidate,
            observation,
            RewardConfig(component_order=("coherence_gain",)),
        )
        penalty_only = evaluate_knob_policy(
            candidate,
            observation,
            RewardConfig(component_order=("bad_coherence",)),
        )

        assert full.reward == pytest.approx(
            coherence_only.reward
            + full.components["target_tracking"]
            + full.components["actuation"]
            + full.components["regime_churn"]
            + full.components["unsafe"]
            + full.components["bad_coherence"]
        )
        assert penalty_only.reward == full.components["bad_coherence"]
        assert penalty_only.reward < coherence_only.reward

    def test_reward_component_order_rejects_unknown_and_duplicate_terms(self) -> None:
        with pytest.raises(ValueError, match="unknown reward component"):
            RewardConfig(component_order=("coherence_gain", "not_a_component"))

        with pytest.raises(ValueError, match="duplicate reward component"):
            RewardConfig(component_order=("coherence_gain", "coherence_gain"))

    def test_lyapunov_stl_and_safety_cost_terms_penalise_unsafe_replay(
        self,
    ) -> None:
        candidate = KnobPolicyCandidate(K=0.4)

        stable = evaluate_knob_policy(
            candidate,
            RewardObservation(
                coherence=0.8,
                previous_coherence=0.75,
                lyapunov_exponent=-0.02,
                stl_robustness=0.1,
                safety_cost=0.0,
            ),
        )
        unstable = evaluate_knob_policy(
            candidate,
            RewardObservation(
                coherence=0.8,
                previous_coherence=0.75,
                lyapunov_exponent=0.04,
                stl_robustness=-0.03,
                safety_cost=0.2,
            ),
        )

        assert stable.components["lyapunov_stability"] == 0.0
        assert stable.components["stl_robustness"] == 0.0
        assert stable.components["safety_cost"] == 0.0
        assert unstable.components["lyapunov_stability"] < 0.0
        assert unstable.components["stl_robustness"] < 0.0
        assert unstable.components["safety_cost"] < 0.0
        assert unstable.reward < stable.reward


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


class TestAutotuneOfflinePolicySearch:
    def test_generates_coordinate_candidates_around_scalar_seed(self) -> None:
        seed = KnobPolicyCandidate(
            K=0.4,
            alpha=0.2,
            zeta=0.1,
            Psi=0.3,
            channel_weights=(1.0, 0.5),
            cross_channel_gains=(0.3, 0.6),
        )

        candidates = generate_offline_policy_candidates(
            seed,
            OfflinePolicySearchConfig(
                K_step=0.1,
                alpha_step=0.0,
                zeta_step=0.0,
                Psi_step=0.0,
                channel_weight_step=0.2,
                cross_channel_gain_step=0.1,
            ),
        )

        assert candidates[0] == seed
        assert any(pytest.approx(0.3) == candidate.K for candidate in candidates)
        assert any(pytest.approx(0.5) == candidate.K for candidate in candidates)
        assert any(candidate.channel_weights == (0.8, 0.5) for candidate in candidates)
        assert any(candidate.channel_weights == (1.0, 0.7) for candidate in candidates)
        assert any(
            candidate.cross_channel_gains == pytest.approx((0.2, 0.6))
            for candidate in candidates
        )
        assert any(
            candidate.cross_channel_gains == pytest.approx((0.3, 0.7))
            for candidate in candidates
        )

    def test_generator_can_exclude_baseline_and_deduplicates_zero_steps(self) -> None:
        seed = KnobPolicyCandidate(K=0.4)

        candidates = generate_offline_policy_candidates(
            seed,
            OfflinePolicySearchConfig(
                include_baseline=False,
                K_step=0.0,
                alpha_step=0.0,
                zeta_step=0.0,
                Psi_step=0.0,
                channel_weight_step=0.0,
                cross_channel_gain_step=0.0,
            ),
        )

        assert candidates == ()

    def test_generator_clips_knobs_and_channel_weights(self) -> None:
        seed = KnobPolicyCandidate(
            K=np.array([[0.45, -0.45]], dtype=np.float64),
            channel_weights=(0.05, 0.48),
            cross_channel_gains=(0.04, 0.49),
        )

        candidates = generate_offline_policy_candidates(
            seed,
            OfflinePolicySearchConfig(
                K_step=0.1,
                alpha_step=0.0,
                zeta_step=0.0,
                Psi_step=0.0,
                channel_weight_step=0.1,
                cross_channel_gain_step=0.1,
                max_abs_knob=0.5,
            ),
        )

        generated_arrays = [candidate.K for candidate in candidates]
        assert any(
            isinstance(value, np.ndarray) and np.max(value) == 0.5
            for value in generated_arrays
        )
        assert any(candidate.channel_weights == (0.0, 0.48) for candidate in candidates)
        assert any(candidate.channel_weights == (0.05, 0.5) for candidate in candidates)
        assert any(
            candidate.cross_channel_gains == (0.0, 0.49) for candidate in candidates
        )
        assert any(
            candidate.cross_channel_gains == (0.04, 0.5) for candidate in candidates
        )

    def test_generated_candidates_feed_replay_ranking(self) -> None:
        seed = KnobPolicyCandidate(K=0.2)
        candidates = generate_offline_policy_candidates(
            seed,
            OfflinePolicySearchConfig(
                K_step=0.1,
                alpha_step=0.0,
                zeta_step=0.0,
                Psi_step=0.0,
                channel_weight_step=0.0,
            ),
        )
        replay = tuple(
            (
                candidate,
                RewardObservation(
                    coherence=0.5 + 0.1 * index,
                    previous_coherence=0.4,
                ),
            )
            for index, candidate in enumerate(candidates)
        )

        ranked = rank_replay_candidates(replay)

        assert ranked[0].candidate == candidates[-1]
        assert ranked[0].reward >= ranked[-1].reward


class TestAutotunePolicyProposal:
    def test_accepts_best_safe_candidate_when_gates_pass(self) -> None:
        replay = (
            (KnobPolicyCandidate(K=0.1), RewardObservation(coherence=0.65)),
            (KnobPolicyCandidate(K=0.2), RewardObservation(coherence=0.85)),
            (KnobPolicyCandidate(K=0.3), RewardObservation(coherence=0.75)),
        )

        proposal = propose_replay_policy(
            replay,
            proposal_config=PolicyProposalConfig(
                min_reward=-1.0,
                min_coherence=0.7,
                max_alternatives=1,
            ),
        )

        assert proposal.accepted
        assert proposal.selected is not None
        assert proposal.selected.observation.coherence == 0.85
        assert len(proposal.alternatives) == 1
        assert proposal.reasons == ()

    def test_rejects_candidate_below_reward_gate(self) -> None:
        replay = ((KnobPolicyCandidate(K=0.1), RewardObservation(coherence=0.8)),)

        proposal = propose_replay_policy(
            replay,
            proposal_config=PolicyProposalConfig(min_reward=1.0),
        )

        assert not proposal.accepted
        assert proposal.selected is None
        assert "reward" in proposal.reasons[0]

    def test_rejects_candidate_below_coherence_gate(self) -> None:
        replay = ((KnobPolicyCandidate(K=0.1), RewardObservation(coherence=0.5)),)

        proposal = propose_replay_policy(
            replay,
            proposal_config=PolicyProposalConfig(min_coherence=0.8),
        )

        assert not proposal.accepted
        assert proposal.selected is None
        assert "coherence" in proposal.reasons[0]

    def test_proposal_serialises_for_audit(self) -> None:
        replay = (
            (KnobPolicyCandidate(K=0.1), RewardObservation(coherence=0.8)),
            (KnobPolicyCandidate(K=0.2), RewardObservation(coherence=0.7)),
        )

        record = propose_replay_policy(replay).to_audit_record()

        assert record["accepted"] is True
        assert record["selected"]["observation"]["coherence"] == 0.8
        assert record["alternatives"][0]["observation"]["coherence"] == 0.7
        assert record["config"]["require_safe"] is True

    def test_rejects_selected_unsafe_candidate_when_policy_allows_audit(self) -> None:
        replay = (
            (
                KnobPolicyCandidate(K=0.1),
                RewardObservation(coherence=0.95, previous_coherence=0.5, unsafe=True),
            ),
            (
                KnobPolicyCandidate(K=100.0),
                RewardObservation(coherence=0.55, previous_coherence=0.5),
            ),
        )

        proposal = propose_replay_policy(
            replay,
            proposal_config=PolicyProposalConfig(require_safe=False),
        )

        assert not proposal.accepted
        assert proposal.selected is None
        assert proposal.reasons == ("selected rollout is marked unsafe",)
        assert proposal.alternatives[0].observation.coherence == 0.55

    def test_safety_constraints_choose_safe_candidate_over_higher_reward(
        self,
    ) -> None:
        replay = (
            (
                KnobPolicyCandidate(K=0.1),
                RewardObservation(
                    coherence=0.95,
                    previous_coherence=0.5,
                    lyapunov_exponent=0.08,
                    stl_robustness=0.2,
                    safety_cost=0.05,
                ),
            ),
            (
                KnobPolicyCandidate(K=0.2),
                RewardObservation(
                    coherence=0.82,
                    previous_coherence=0.5,
                    lyapunov_exponent=-0.01,
                    stl_robustness=0.04,
                    safety_cost=0.05,
                ),
            ),
        )

        proposal = propose_replay_policy(
            replay,
            proposal_config=PolicyProposalConfig(
                safety_constraints=SafetyConstraintConfig(
                    max_lyapunov_exponent=0.0,
                    min_stl_robustness=0.0,
                    max_safety_cost=0.1,
                    require_lyapunov=True,
                    require_stl=True,
                    require_safety_cost=True,
                ),
            ),
        )

        assert proposal.accepted
        assert proposal.selected is not None
        assert proposal.selected.candidate.K == 0.2
        assert proposal.selected.observation.lyapunov_exponent == -0.01
        assert proposal.to_audit_record()["config"]["safety_constraints"] == {
            "max_lyapunov_exponent": 0.0,
            "min_stl_robustness": 0.0,
            "max_safety_cost": 0.1,
            "require_lyapunov": True,
            "require_stl": True,
            "require_safety_cost": True,
        }

    def test_safety_constraints_reject_missing_evidence(self) -> None:
        replay = (
            (
                KnobPolicyCandidate(K=0.1),
                RewardObservation(coherence=0.9, previous_coherence=0.5),
            ),
        )

        proposal = propose_replay_policy(
            replay,
            proposal_config=PolicyProposalConfig(
                safety_constraints=SafetyConstraintConfig(
                    max_lyapunov_exponent=0.0,
                    min_stl_robustness=0.0,
                    require_lyapunov=True,
                    require_stl=True,
                ),
            ),
        )

        assert not proposal.accepted
        assert proposal.selected is None
        assert proposal.reasons == (
            "no replay candidate satisfies Lyapunov/STL safety constraints",
        )
        assert proposal.alternatives[0].observation.coherence == 0.9


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

    def test_rejects_negative_cross_channel_gains(self) -> None:
        with pytest.raises(ValueError, match="cross-channel gain"):
            evaluate_knob_policy(
                KnobPolicyCandidate(cross_channel_gains=(1.0, -0.1)),
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

    def test_rejects_invalid_bad_coherence_threshold(self) -> None:
        with pytest.raises(ValueError, match="bad_coherence_threshold"):
            RewardConfig(bad_coherence_threshold=1.5)

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

    def test_replay_ranking_rejects_boolean_top_k_and_require_safe(self) -> None:
        replay = ((KnobPolicyCandidate(K=0.1), RewardObservation(coherence=0.9)),)

        with pytest.raises(TypeError, match="top_k"):
            rank_replay_candidates(replay, top_k=cast(int, True))

        with pytest.raises(TypeError, match="require_safe"):
            rank_replay_candidates(replay, require_safe=cast(bool, np.bool_(True)))

    def test_offline_generator_rejects_negative_steps(self) -> None:
        with pytest.raises(ValueError, match="K_step"):
            OfflinePolicySearchConfig(K_step=-0.1)

    def test_offline_generator_rejects_zero_clip_bound(self) -> None:
        with pytest.raises(ValueError, match="max_abs_knob"):
            OfflinePolicySearchConfig(max_abs_knob=0.0)

    def test_offline_generator_rejects_boolean_config_aliases(self) -> None:
        with pytest.raises(ValueError, match="K_step"):
            OfflinePolicySearchConfig(K_step=cast(float, True))

        with pytest.raises(TypeError, match="include_baseline"):
            OfflinePolicySearchConfig(include_baseline=cast(bool, np.bool_(True)))

    def test_policy_proposal_rejects_invalid_coherence_gate(self) -> None:
        with pytest.raises(ValueError, match="min_coherence"):
            PolicyProposalConfig(min_coherence=1.1)

    @pytest.mark.parametrize("min_reward", [float("nan"), float("inf")])
    def test_policy_proposal_rejects_non_finite_reward_gate(
        self,
        min_reward: float,
    ) -> None:
        with pytest.raises(ValueError, match="min_reward"):
            PolicyProposalConfig(min_reward=min_reward)

    def test_policy_proposal_rejects_negative_alternative_limit(self) -> None:
        with pytest.raises(ValueError, match="max_alternatives"):
            PolicyProposalConfig(max_alternatives=-1)

    def test_policy_proposal_rejects_boolean_and_non_integral_controls(self) -> None:
        with pytest.raises(TypeError, match="max_alternatives"):
            PolicyProposalConfig(max_alternatives=cast(int, True))

        with pytest.raises(TypeError, match="max_alternatives"):
            PolicyProposalConfig(max_alternatives=cast(int, 1.5))

        with pytest.raises(TypeError, match="require_safe"):
            PolicyProposalConfig(require_safe=cast(bool, np.bool_(True)))

    def test_reward_observation_rejects_boolean_probability_aliases(self) -> None:
        with pytest.raises(ValueError, match="coherence"):
            RewardObservation(coherence=cast(float, True))

        with pytest.raises(ValueError, match="previous_coherence"):
            RewardObservation(coherence=0.5, previous_coherence=cast(float, False))

    def test_reward_observation_rejects_numpy_boolean_flags(self) -> None:
        with pytest.raises(TypeError, match="unsafe"):
            RewardObservation(coherence=0.5, unsafe=cast(bool, np.bool_(True)))

        with pytest.raises(TypeError, match="regime_changed"):
            RewardObservation(
                coherence=0.5,
                regime_changed=cast(bool, np.bool_(True)),
            )

    def test_reward_observation_rejects_invalid_safety_evidence(self) -> None:
        with pytest.raises(ValueError, match="lyapunov_exponent"):
            RewardObservation(coherence=0.5, lyapunov_exponent=cast(float, True))

        with pytest.raises(ValueError, match="stl_robustness"):
            RewardObservation(coherence=0.5, stl_robustness=float("nan"))

        with pytest.raises(ValueError, match="safety_cost"):
            RewardObservation(coherence=0.5, safety_cost=-0.01)

    def test_safety_constraint_config_rejects_invalid_bounds_and_flags(self) -> None:
        with pytest.raises(ValueError, match="max_lyapunov_exponent"):
            SafetyConstraintConfig(max_lyapunov_exponent=float("nan"))

        with pytest.raises(ValueError, match="max_safety_cost"):
            SafetyConstraintConfig(max_safety_cost=-0.1)

        with pytest.raises(TypeError, match="require_stl"):
            SafetyConstraintConfig(require_stl=cast(bool, np.bool_(True)))

        with pytest.raises(ValueError, match="require_lyapunov"):
            SafetyConstraintConfig(require_lyapunov=True)

        with pytest.raises(TypeError, match="safety_constraints"):
            PolicyProposalConfig(
                safety_constraints=cast(SafetyConstraintConfig, object()),
            )

    @pytest.mark.parametrize(
        "candidate",
        [
            KnobPolicyCandidate(K=True),
            KnobPolicyCandidate(alpha=np.array([True], dtype=object)),
            KnobPolicyCandidate(zeta=np.array([1.0 + 0.0j], dtype=object)),
            KnobPolicyCandidate(Psi=1.0 + 0.0j),
            KnobPolicyCandidate(channel_weights=(cast(float, True),)),
            KnobPolicyCandidate(cross_channel_gains=(cast(float, np.bool_(True)),)),
        ],
    )
    def test_rejects_boolean_and_complex_candidate_aliases(
        self,
        candidate: KnobPolicyCandidate,
    ) -> None:
        with pytest.raises(
            ValueError,
            match="boolean|real-valued|channel weight|cross-channel gain",
        ):
            evaluate_knob_policy(candidate, RewardObservation(coherence=0.8))
