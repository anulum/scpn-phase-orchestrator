# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Replay-only learner proposal tests

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import cast

import numpy as np
import pytest

import scpn_phase_orchestrator.autotune.learners as learner_mod
from scpn_phase_orchestrator.autotune.learners import (
    LearnerPolicyProposal,
    generate_hybrid_physics_proposal,
    generate_ppo_like_proposal,
    generate_sac_like_proposal,
)
from scpn_phase_orchestrator.autotune.policy_search import ReplayPolicySearchResult
from scpn_phase_orchestrator.autotune.reward import (
    KnobPolicyCandidate,
    PolicyProposalConfig,
    RewardObservation,
    SafetyConstraintConfig,
)


def _safe_observation(candidate: KnobPolicyCandidate) -> RewardObservation:
    return RewardObservation(
        coherence=0.82,
        previous_coherence=0.79,
        unsafe=False,
        regime_changed=False,
    )


def _unsafe_observation(candidate: KnobPolicyCandidate) -> RewardObservation:
    return RewardObservation(
        coherence=0.91,
        previous_coherence=0.11,
        unsafe=True,
        regime_changed=True,
    )


def test_ppo_like_proposal_is_deterministic_and_non_actuating() -> None:
    seed = KnobPolicyCandidate(K=1.0, alpha=0.0, zeta=0.1, Psi=0.0)

    proposal = generate_ppo_like_proposal(seed, _safe_observation, seed_value=7)
    second = generate_ppo_like_proposal(seed, _safe_observation, seed_value=7)

    assert proposal.learner_kind == "ppo_like_replay"
    assert proposal.to_audit_record() == second.to_audit_record()
    assert proposal.actuation_permitted is False
    assert proposal.policy_search.proposal.accepted is True


def test_sac_like_proposal_rejects_unsafe_replay() -> None:
    seed = KnobPolicyCandidate(K=1.0, alpha=0.0, zeta=0.1, Psi=0.0)

    proposal = generate_sac_like_proposal(seed, _unsafe_observation, seed_value=11)

    assert proposal.learner_kind == "sac_like_replay"
    assert proposal.actuation_permitted is False
    assert proposal.policy_search.proposal.accepted is False


def test_ppo_like_proposal_rejects_replay_without_required_safety_evidence() -> None:
    seed = KnobPolicyCandidate(K=1.0, alpha=0.0, zeta=0.1, Psi=0.0)

    proposal = generate_ppo_like_proposal(
        seed,
        _safe_observation,
        seed_value=7,
        proposal_config=PolicyProposalConfig(
            safety_constraints=SafetyConstraintConfig(
                max_lyapunov_exponent=0.0,
                min_stl_robustness=0.0,
                require_lyapunov=True,
                require_stl=True,
            ),
        ),
    )

    assert proposal.actuation_permitted is False
    assert proposal.policy_search.proposal.accepted is False
    assert proposal.policy_search.proposal.reasons == (
        "no replay candidate satisfies Lyapunov/STL safety constraints",
    )


def test_hybrid_physics_proposal_records_prior() -> None:
    seed = KnobPolicyCandidate(K=1.0, alpha=0.0, zeta=0.1, Psi=0.0)

    proposal = generate_hybrid_physics_proposal(
        seed,
        _safe_observation,
        critical_coupling_estimate=1.4,
    )

    record = proposal.to_audit_record()
    assert record["learner_kind"] == "hybrid_physics_replay"
    assert record["physics_prior"]["critical_coupling_estimate"] == 1.4
    assert record["actuation_permitted"] is False


def test_hybrid_physics_proposal_rejects_invalid_prior() -> None:
    seed = KnobPolicyCandidate(K=1.0, alpha=0.0, zeta=0.1, Psi=0.0)

    with pytest.raises(ValueError, match="critical_coupling_estimate"):
        generate_hybrid_physics_proposal(
            seed,
            _safe_observation,
            critical_coupling_estimate=-1.0,
        )

    with pytest.raises(ValueError, match="critical_coupling_estimate"):
        generate_hybrid_physics_proposal(
            seed,
            _safe_observation,
            critical_coupling_estimate=True,
        )


def test_hybrid_physics_proposal_rejects_invalid_seed_k_before_prior_math() -> None:
    with pytest.raises(ValueError, match="seed.K"):
        generate_hybrid_physics_proposal(
            KnobPolicyCandidate(K=cast("float", True)),
            _safe_observation,
            critical_coupling_estimate=1.4,
        )

    with pytest.raises(ValueError, match="seed.K"):
        generate_hybrid_physics_proposal(
            KnobPolicyCandidate(K=cast("float", 1.0 + 0.0j)),
            _safe_observation,
            critical_coupling_estimate=1.4,
        )


@pytest.mark.parametrize("seed_value", [True, -1, 1.25])
def test_replay_learner_seed_value_rejects_non_integer_aliases(
    seed_value: object,
) -> None:
    seed = KnobPolicyCandidate(K=1.0, alpha=0.0, zeta=0.1, Psi=0.0)

    with pytest.raises(ValueError, match="seed_value"):
        generate_ppo_like_proposal(
            seed,
            _safe_observation,
            seed_value=cast(int, seed_value),
        )


def test_replay_learner_accepts_numpy_integer_seed_canonically() -> None:
    seed = KnobPolicyCandidate(K=1.0, alpha=0.0, zeta=0.1, Psi=0.0)

    proposal = generate_sac_like_proposal(
        seed,
        _safe_observation,
        seed_value=np.int64(11),
    )

    assert proposal.to_audit_record()["learner_parameters"]["seed_value"] == 11


def test_learner_policy_proposal_never_permits_actuation() -> None:
    seed = KnobPolicyCandidate(K=1.0, alpha=0.0, zeta=0.1, Psi=0.0)
    proposal = generate_ppo_like_proposal(seed, _safe_observation, seed_value=7)

    with pytest.raises(ValueError, match="replay-only"):
        LearnerPolicyProposal(
            learner_kind=proposal.learner_kind,
            policy_search=proposal.policy_search,
            actuation_permitted=True,
        )
    with pytest.raises(ValueError, match="actuation_permitted"):
        LearnerPolicyProposal(
            learner_kind=proposal.learner_kind,
            policy_search=proposal.policy_search,
            actuation_permitted=cast(bool, np.bool_(False)),
        )


def test_learner_kind_is_canonicalised_for_audit_records() -> None:
    seed = KnobPolicyCandidate(K=1.0, alpha=0.0, zeta=0.1, Psi=0.0)
    proposal = generate_ppo_like_proposal(seed, _safe_observation, seed_value=7)
    wrapped = LearnerPolicyProposal(
        learner_kind="  custom_replay  ",
        policy_search=proposal.policy_search,
    )

    assert wrapped.learner_kind == "custom_replay"
    assert wrapped.to_audit_record()["learner_kind"] == "custom_replay"


def test_audit_record_is_json_serialisable() -> None:
    seed = KnobPolicyCandidate(
        K=1.0,
        alpha=0.0,
        zeta=0.1,
        Psi=0.0,
        channel_weights=(0.2, 0.3),
        cross_channel_gains=(0.01,),
    )

    proposal = generate_sac_like_proposal(seed, _safe_observation, seed_value=11)

    json.dumps(proposal.to_audit_record(), allow_nan=False, sort_keys=True)


def test_audit_record_coerces_numpy_scalars_and_rejects_non_finite_values() -> None:
    seed = KnobPolicyCandidate(K=1.0, alpha=0.0, zeta=0.1, Psi=0.0)
    proposal = generate_hybrid_physics_proposal(
        seed,
        _safe_observation,
        critical_coupling_estimate=1.4,
    )
    wrapped = LearnerPolicyProposal(
        learner_kind=proposal.learner_kind,
        policy_search=proposal.policy_search,
        learner_parameters={
            "clip_range": np.float64(0.125),
        },
        physics_prior={"critical_coupling_estimate": np.float64(1.4)},
    )

    record = wrapped.to_audit_record()

    assert record["learner_parameters"]["clip_range"] == 0.125
    assert record["physics_prior"]["critical_coupling_estimate"] == 1.4
    json.dumps(record, allow_nan=False, sort_keys=True)

    invalid_value = LearnerPolicyProposal(
        learner_kind=proposal.learner_kind,
        policy_search=proposal.policy_search,
        learner_parameters={
            "diagnostics": [np.float64(np.inf), np.float64(np.nan)],
        },
    )
    with pytest.raises(ValueError, match="finite"):
        invalid_value.to_audit_record()


def test_audit_record_rejects_invalid_mapping_keys_and_complex_values() -> None:
    seed = KnobPolicyCandidate(K=1.0, alpha=0.0, zeta=0.1, Psi=0.0)
    proposal = generate_hybrid_physics_proposal(
        seed,
        _safe_observation,
        critical_coupling_estimate=1.4,
    )

    invalid_key = LearnerPolicyProposal(
        learner_kind=proposal.learner_kind,
        policy_search=proposal.policy_search,
        learner_parameters=cast(Mapping[str, object], {True: 1.0}),
    )
    with pytest.raises(ValueError, match="audit mapping keys"):
        invalid_key.to_audit_record()

    invalid_value = LearnerPolicyProposal(
        learner_kind=proposal.learner_kind,
        policy_search=proposal.policy_search,
        learner_parameters={"complex_gain": 1.0 + 0.0j},
    )
    with pytest.raises(ValueError, match="complex"):
        invalid_value.to_audit_record()


def test_learner_policy_proposal_rejects_invalid_identity_and_payloads() -> None:
    seed = KnobPolicyCandidate(K=1.0, alpha=0.0, zeta=0.1, Psi=0.0)
    proposal = generate_ppo_like_proposal(seed, _safe_observation, seed_value=7)

    with pytest.raises(ValueError, match="learner_kind"):
        LearnerPolicyProposal(
            learner_kind=" ",
            policy_search=proposal.policy_search,
        )

    with pytest.raises(TypeError, match="learner_parameters"):
        LearnerPolicyProposal(
            learner_kind=proposal.learner_kind,
            policy_search=proposal.policy_search,
            learner_parameters=cast(Mapping[str, object], ["not", "mapping"]),
        )


def test_audit_record_guard_rejects_non_mapping_payload() -> None:
    with pytest.raises(TypeError, match="audit record must be a mapping"):
        learner_mod._json_safe_record(["not", "a", "mapping"])


def test_failed_search_closes_without_alternatives_when_ranking_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unsafe_observation(candidate: KnobPolicyCandidate) -> RewardObservation:
        return RewardObservation(
            coherence=0.9,
            previous_coherence=0.1,
            unsafe=True,
        )

    def reject_rank(*_args, **_kwargs):
        raise ValueError("ranking input rejected")

    monkeypatch.setattr(learner_mod, "rank_replay_candidates", reject_rank)
    seed = KnobPolicyCandidate(K=1.0, alpha=0.0, zeta=0.1, Psi=0.0)

    proposal = generate_ppo_like_proposal(seed, unsafe_observation, seed_value=7)

    assert proposal.policy_search.proposal.accepted is False
    assert proposal.policy_search.proposal.selected is None
    assert proposal.policy_search.proposal.alternatives == ()
    assert proposal.policy_search.proposal.reasons == (
        "no safe replay candidates remain after filtering",
    )


def test_failed_search_collects_alternatives_when_ranking_fallback_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def safe_observation(candidate: KnobPolicyCandidate) -> RewardObservation:
        return RewardObservation(coherence=0.9, previous_coherence=0.8)

    def force_search_failure(*_args, **_kwargs) -> None:
        raise ValueError("search blocked")

    monkeypatch.setattr(learner_mod, "search_replay_policy", force_search_failure)
    monkeypatch.setattr(
        learner_mod,
        "rank_replay_candidates",
        learner_mod.rank_replay_candidates,
    )

    seed = KnobPolicyCandidate(K=1.0, alpha=0.0, zeta=0.1, Psi=0.0)
    proposal = generate_ppo_like_proposal(
        seed,
        safe_observation,
        seed_value=7,
        proposal_config=None,
    )

    assert proposal.policy_search.proposal.accepted is False
    assert proposal.policy_search.proposal.selected is None
    assert len(proposal.policy_search.proposal.alternatives) == 3
    assert proposal.policy_search.proposal.reasons == ("search blocked",)


def test_learner_proposal_rejects_non_policy_search() -> None:
    with pytest.raises(TypeError, match="policy_search must be a ReplayPolicy"):
        LearnerPolicyProposal(
            learner_kind="ppo",
            policy_search=cast(ReplayPolicySearchResult, "not-a-result"),
        )


def test_learner_proposal_rejects_non_mapping_physics_prior() -> None:
    seed = KnobPolicyCandidate(K=1.0, alpha=0.0, zeta=0.1, Psi=0.0)
    proposal = generate_ppo_like_proposal(seed, _safe_observation, seed_value=7)

    with pytest.raises(TypeError, match="physics_prior must be a mapping"):
        LearnerPolicyProposal(
            learner_kind=proposal.learner_kind,
            policy_search=proposal.policy_search,
            physics_prior=cast("Mapping[str, object]", ["not-a-mapping"]),
        )


@pytest.mark.parametrize(
    ("value", "expected_type"),
    [(np.bool_(True), bool), (np.int64(5), int)],
)
def test_json_safe_value_coerces_numpy_bool_and_integer(
    value: object, expected_type: type
) -> None:
    result = learner_mod._json_safe_value(value)
    assert type(result) is expected_type
    assert result == expected_type(value)


def test_mean_seed_k_rejects_non_float_convertible_object_array() -> None:
    seed = KnobPolicyCandidate(K=cast("float", np.array(["x", "y"], dtype=object)))

    with pytest.raises(ValueError, match="seed.K must be real-valued"):
        learner_mod._mean_seed_k(seed)


@pytest.mark.parametrize(
    "array",
    [np.array([], dtype=np.float64), np.array([np.inf], dtype=np.float64)],
)
def test_mean_seed_k_rejects_empty_or_non_finite_array(array: np.ndarray) -> None:
    seed = KnobPolicyCandidate(K=array)

    with pytest.raises(ValueError, match="seed.K must be finite and non-empty"):
        learner_mod._mean_seed_k(seed)
