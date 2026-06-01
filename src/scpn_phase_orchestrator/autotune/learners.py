# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Replay-only learner proposal interfaces

"""Learner-shaped replay-only autotune proposal generators."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import TypeAlias

import numpy as np

from scpn_phase_orchestrator.autotune.policy_search import (
    ReplayPolicyEvaluator,
    ReplayPolicySearchResult,
    search_replay_policy,
)
from scpn_phase_orchestrator.autotune.reward import (
    AutotunePolicyProposal,
    AutotuneRewardReport,
    KnobPolicyCandidate,
    OfflinePolicySearchConfig,
    PolicyProposalConfig,
    RewardConfig,
    RewardObservation,
    generate_offline_policy_candidates,
    rank_replay_candidates,
)

__all__ = [
    "LearnerPolicyProposal",
    "generate_hybrid_physics_proposal",
    "generate_ppo_like_proposal",
    "generate_sac_like_proposal",
]

AuditMapping: TypeAlias = Mapping[str, object]


@dataclass(frozen=True)
class LearnerPolicyProposal:
    """Replay-trained learner proposal record for audit review only."""

    learner_kind: str
    policy_search: ReplayPolicySearchResult
    actuation_permitted: bool = False
    learner_parameters: AuditMapping = field(default_factory=dict)
    physics_prior: AuditMapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.learner_kind, str) or not self.learner_kind.strip():
            raise ValueError("learner_kind must be a non-empty string")
        if not isinstance(self.policy_search, ReplayPolicySearchResult):
            raise TypeError("policy_search must be a ReplayPolicySearchResult")
        if self.actuation_permitted:
            raise ValueError("learner policy proposals are replay-only")
        if not isinstance(self.learner_parameters, Mapping):
            raise TypeError("learner_parameters must be a mapping")
        if not isinstance(self.physics_prior, Mapping):
            raise TypeError("physics_prior must be a mapping")

    def to_audit_record(self) -> dict[str, object]:
        """Return an audit-serialisable learner proposal record."""
        return _json_safe_record(
            {
                "learner_kind": self.learner_kind,
                "actuation_permitted": self.actuation_permitted,
                "learner_parameters": dict(self.learner_parameters),
                "physics_prior": dict(self.physics_prior),
                "policy_search": self.policy_search.to_audit_record(),
            }
        )


def _json_safe_record(value: object) -> dict[str, object]:
    safe = _json_safe_value(value)
    if not isinstance(safe, dict):
        raise TypeError("audit record must be a mapping")
    return safe


def _json_safe_value(value: object) -> object:
    if isinstance(value, Mapping):
        record: dict[str, object] = {}
        for key, nested_value in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("audit mapping keys must be non-empty strings")
            record[key.strip()] = _json_safe_value(nested_value)
        return record
    if isinstance(value, tuple | list):
        return [_json_safe_value(nested_value) for nested_value in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return repr(value)
    if isinstance(value, (complex, np.complexfloating)):
        raise ValueError("audit records must not contain complex values")
    return value


def generate_ppo_like_proposal(
    seed: KnobPolicyCandidate,
    evaluator: ReplayPolicyEvaluator,
    *,
    seed_value: int | None = None,
    reward_config: RewardConfig | None = None,
    proposal_config: PolicyProposalConfig | None = None,
) -> LearnerPolicyProposal:
    """Generate a deterministic PPO-shaped proposal from replay evaluations."""
    seed_value = _validate_seed_value(seed_value)
    clip_range = _uniform(seed_value, low=0.08, high=0.18)
    search_config = OfflinePolicySearchConfig(
        K_step=clip_range,
        alpha_step=clip_range * 0.5,
        zeta_step=clip_range * 0.5,
        Psi_step=clip_range * 0.25,
        channel_weight_step=clip_range * 0.25,
        cross_channel_gain_step=clip_range * 0.25,
        max_abs_knob=2.0,
    )
    return LearnerPolicyProposal(
        learner_kind="ppo_like_replay",
        policy_search=_safe_replay_search(
            seed,
            evaluator,
            search_config,
            reward_config,
            proposal_config,
        ),
        learner_parameters={
            "clip_range": clip_range,
            "seed_value": seed_value,
        },
    )


def generate_sac_like_proposal(
    seed: KnobPolicyCandidate,
    evaluator: ReplayPolicyEvaluator,
    *,
    seed_value: int | None = None,
    reward_config: RewardConfig | None = None,
    proposal_config: PolicyProposalConfig | None = None,
) -> LearnerPolicyProposal:
    """Generate a deterministic SAC-shaped proposal from replay evaluations."""
    seed_value = _validate_seed_value(seed_value)
    entropy_temperature = _uniform(seed_value, low=0.03, high=0.12)
    search_config = OfflinePolicySearchConfig(
        K_step=0.04 + entropy_temperature,
        alpha_step=0.04 + entropy_temperature,
        zeta_step=0.02 + entropy_temperature * 0.5,
        Psi_step=0.02 + entropy_temperature * 0.5,
        channel_weight_step=entropy_temperature * 0.5,
        cross_channel_gain_step=entropy_temperature * 0.5,
        max_abs_knob=2.0,
    )
    return LearnerPolicyProposal(
        learner_kind="sac_like_replay",
        policy_search=_safe_replay_search(
            seed,
            evaluator,
            search_config,
            reward_config,
            proposal_config,
        ),
        learner_parameters={
            "entropy_temperature": entropy_temperature,
            "seed_value": seed_value,
        },
    )


def generate_hybrid_physics_proposal(
    seed: KnobPolicyCandidate,
    evaluator: ReplayPolicyEvaluator,
    *,
    critical_coupling_estimate: float,
    seed_value: int | None = None,
    reward_config: RewardConfig | None = None,
    proposal_config: PolicyProposalConfig | None = None,
) -> LearnerPolicyProposal:
    """Generate a replay proposal shaped by a critical-coupling prior."""
    seed_value = _validate_seed_value(seed_value)
    critical_coupling_estimate = _positive_real(
        critical_coupling_estimate,
        "critical_coupling_estimate",
    )

    current_k = float(np.asarray(seed.K, dtype=np.float64).mean())
    prior_gap = critical_coupling_estimate - current_k
    prior_step = min(0.5, max(0.01, abs(prior_gap) * 0.25))
    jitter = _uniform(seed_value, low=0.0, high=0.02)
    search_config = OfflinePolicySearchConfig(
        K_step=prior_step + jitter,
        alpha_step=0.03,
        zeta_step=0.03,
        Psi_step=0.02,
        channel_weight_step=0.02,
        cross_channel_gain_step=0.02,
        max_abs_knob=max(critical_coupling_estimate * 2.0, 2.0),
    )
    return LearnerPolicyProposal(
        learner_kind="hybrid_physics_replay",
        policy_search=_safe_replay_search(
            seed,
            evaluator,
            search_config,
            reward_config,
            proposal_config,
        ),
        learner_parameters={
            "prior_gap": prior_gap,
            "prior_step": prior_step,
            "seed_value": seed_value,
        },
        physics_prior={
            "critical_coupling_estimate": float(critical_coupling_estimate),
        },
    )


def _safe_replay_search(
    seed: KnobPolicyCandidate,
    evaluator: ReplayPolicyEvaluator,
    search_config: OfflinePolicySearchConfig,
    reward_config: RewardConfig | None,
    proposal_config: PolicyProposalConfig | None,
) -> ReplayPolicySearchResult:
    try:
        return search_replay_policy(
            seed,
            evaluator,
            search_config=search_config,
            reward_config=reward_config,
            proposal_config=proposal_config,
        )
    except ValueError as exc:
        candidates = generate_offline_policy_candidates(seed, search_config)
        replay_observations = tuple(
            (candidate, evaluator(candidate)) for candidate in candidates
        )
        rejected = _rejected_policy_proposal(
            replay_observations,
            reward_config,
            proposal_config,
            str(exc),
        )
        return ReplayPolicySearchResult(
            seed=seed,
            candidates=candidates,
            proposal=rejected,
        )


def _rejected_policy_proposal(
    replay_observations: tuple[tuple[KnobPolicyCandidate, RewardObservation], ...],
    reward_config: RewardConfig | None,
    proposal_config: PolicyProposalConfig | None,
    reason: str,
) -> AutotunePolicyProposal:
    active_config = proposal_config or PolicyProposalConfig()
    alternatives: tuple[AutotuneRewardReport, ...] = ()
    try:
        alternatives = rank_replay_candidates(
            replay_observations,
            reward_config,
            require_safe=False,
            top_k=active_config.max_alternatives,
        )
    except ValueError:
        alternatives = ()
    return AutotunePolicyProposal(
        accepted=False,
        selected=None,
        alternatives=alternatives,
        reasons=(reason,),
        config=active_config,
    )


def _uniform(seed_value: int | None, *, low: float, high: float) -> float:
    if seed_value is None:
        return float((low + high) * 0.5)
    return float(np.random.default_rng(seed_value).uniform(low, high))


def _validate_seed_value(seed_value: int | None) -> int | None:
    if seed_value is None:
        return None
    if isinstance(seed_value, (bool, np.bool_)) or not isinstance(
        seed_value,
        (Integral, np.integer),
    ):
        raise ValueError("seed_value must be a non-negative integer or None")
    seed_int = int(seed_value)
    if seed_int < 0:
        raise ValueError("seed_value must be a non-negative integer or None")
    return seed_int


def _positive_real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (Real, np.floating),
    ):
        raise ValueError(f"{name} must be finite and positive")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return parsed
