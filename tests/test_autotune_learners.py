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
from scpn_phase_orchestrator.autotune.reward import (
    KnobPolicyCandidate,
    RewardObservation,
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


def test_audit_record_coerces_numpy_scalars_and_non_finite_values() -> None:
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
            "diagnostics": [np.float64(np.inf), np.float64(np.nan)],
        },
        physics_prior={"critical_coupling_estimate": np.float64(1.4)},
    )

    record = wrapped.to_audit_record()

    assert record["learner_parameters"]["clip_range"] == 0.125
    assert record["learner_parameters"]["diagnostics"] == ["inf", "nan"]
    assert record["physics_prior"]["critical_coupling_estimate"] == 1.4
    json.dumps(record, allow_nan=False, sort_keys=True)


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
