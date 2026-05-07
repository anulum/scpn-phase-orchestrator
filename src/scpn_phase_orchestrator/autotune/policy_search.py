# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Replay-only autotune policy search

"""Replay-only policy search helpers for autotune candidates."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.autotune.reward import (
    AutotunePolicyProposal,
    KnobPolicyCandidate,
    OfflinePolicySearchConfig,
    PolicyProposalConfig,
    RewardConfig,
    RewardObservation,
    generate_offline_policy_candidates,
    propose_replay_policy,
    rank_replay_candidates,
)

__all__ = [
    "AdaptiveReplayPolicySearchConfig",
    "AdaptiveReplayPolicySearchResult",
    "ReplayPolicyEvaluator",
    "ReplayPolicySearchResult",
    "search_adaptive_replay_policy",
    "search_replay_policy",
]

ReplayPolicyEvaluator: TypeAlias = Callable[[KnobPolicyCandidate], RewardObservation]
FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class AdaptiveReplayPolicySearchConfig:
    """Bounded adaptive search settings for replay-only autotune."""

    base_search_config: OfflinePolicySearchConfig = field(
        default_factory=OfflinePolicySearchConfig
    )
    iterations: int = 3
    step_decay: float = 0.5
    improvement_tolerance: float = 0.0
    min_step: float = 0.0

    def __post_init__(self) -> None:
        if self.iterations < 1:
            raise ValueError("iterations must be positive")
        if not np.isfinite(self.step_decay) or not 0.0 < self.step_decay <= 1.0:
            raise ValueError("step_decay must be finite and within (0, 1]")
        if (
            not np.isfinite(self.improvement_tolerance)
            or self.improvement_tolerance < 0.0
        ):
            raise ValueError("improvement_tolerance must be finite and non-negative")
        if not np.isfinite(self.min_step) or self.min_step < 0.0:
            raise ValueError("min_step must be finite and non-negative")


@dataclass(frozen=True)
class ReplayPolicySearchResult:
    """Replay-only policy-search result suitable for audit review."""

    seed: KnobPolicyCandidate
    candidates: tuple[KnobPolicyCandidate, ...]
    proposal: AutotunePolicyProposal

    def to_audit_record(self) -> dict[str, object]:
        """Return a serialisable search record."""
        return {
            "seed": _candidate_to_record(self.seed),
            "candidates": [
                _candidate_to_record(candidate) for candidate in self.candidates
            ],
            "proposal": self.proposal.to_audit_record(),
        }


@dataclass(frozen=True)
class AdaptiveReplayPolicySearchResult:
    """Multi-round replay-only policy-search result for audit review."""

    seed: KnobPolicyCandidate
    rounds: tuple[ReplayPolicySearchResult, ...]
    proposal: AutotunePolicyProposal
    config: AdaptiveReplayPolicySearchConfig

    def to_audit_record(self) -> dict[str, object]:
        """Return a serialisable adaptive-search record."""
        return {
            "seed": _candidate_to_record(self.seed),
            "rounds": [round_result.to_audit_record() for round_result in self.rounds],
            "proposal": self.proposal.to_audit_record(),
            "config": {
                "iterations": self.config.iterations,
                "step_decay": self.config.step_decay,
                "improvement_tolerance": self.config.improvement_tolerance,
                "min_step": self.config.min_step,
                "base_search_config": _search_config_to_record(
                    self.config.base_search_config
                ),
            },
        }


def search_replay_policy(
    seed: KnobPolicyCandidate,
    evaluator: ReplayPolicyEvaluator,
    search_config: OfflinePolicySearchConfig | None = None,
    reward_config: RewardConfig | None = None,
    proposal_config: PolicyProposalConfig | None = None,
) -> ReplayPolicySearchResult:
    """Generate, replay-evaluate, and propose an autotune policy.

    The evaluator must be a replay or simulation adapter. This helper never
    applies control actions directly; it only turns candidate observations into
    the existing reviewable proposal record.
    """
    candidates = generate_offline_policy_candidates(seed, search_config)
    if not candidates:
        raise ValueError("replay policy search generated no candidates")

    replay_observations = tuple(
        (candidate, evaluator(candidate)) for candidate in candidates
    )
    proposal = propose_replay_policy(
        replay_observations,
        reward_config=reward_config,
        proposal_config=proposal_config,
    )
    return ReplayPolicySearchResult(
        seed=seed,
        candidates=candidates,
        proposal=proposal,
    )


def search_adaptive_replay_policy(
    seed: KnobPolicyCandidate,
    evaluator: ReplayPolicyEvaluator,
    adaptive_config: AdaptiveReplayPolicySearchConfig | None = None,
    reward_config: RewardConfig | None = None,
    proposal_config: PolicyProposalConfig | None = None,
) -> AdaptiveReplayPolicySearchResult:
    """Run bounded adaptive replay-only policy search.

    Each round generates deterministic coordinate-search candidates around the
    current best replay candidate, then shrinks the coordinate step sizes. The
    final proposal is still built from replay observations and existing review
    gates; no candidate is applied directly.
    """
    active_config = adaptive_config or AdaptiveReplayPolicySearchConfig()
    active_proposal_config = proposal_config or PolicyProposalConfig()
    current_seed = seed
    search_config = active_config.base_search_config
    rounds: list[ReplayPolicySearchResult] = []
    replay_observations: list[tuple[KnobPolicyCandidate, RewardObservation]] = []
    best_reward = -np.inf

    for _ in range(active_config.iterations):
        candidates = generate_offline_policy_candidates(current_seed, search_config)
        if not candidates:
            raise ValueError("adaptive replay policy search generated no candidates")
        round_observations = tuple(
            (candidate, evaluator(candidate)) for candidate in candidates
        )
        replay_observations.extend(round_observations)
        round_proposal = propose_replay_policy(
            round_observations,
            reward_config=reward_config,
            proposal_config=active_proposal_config,
        )
        rounds.append(
            ReplayPolicySearchResult(
                seed=current_seed,
                candidates=candidates,
                proposal=round_proposal,
            )
        )

        ranked = rank_replay_candidates(
            round_observations,
            reward_config,
            require_safe=active_proposal_config.require_safe,
            top_k=1,
        )
        candidate_best = ranked[0]
        if candidate_best.reward > best_reward + active_config.improvement_tolerance:
            best_reward = candidate_best.reward
            current_seed = candidate_best.candidate
        search_config = _decay_search_config(search_config, active_config)

    final_proposal = propose_replay_policy(
        tuple(replay_observations),
        reward_config=reward_config,
        proposal_config=active_proposal_config,
    )
    return AdaptiveReplayPolicySearchResult(
        seed=seed,
        rounds=tuple(rounds),
        proposal=final_proposal,
        config=active_config,
    )


def _candidate_to_record(candidate: KnobPolicyCandidate) -> dict[str, object]:
    return {
        "K": _serialise_knob(candidate.K),
        "alpha": _serialise_knob(candidate.alpha),
        "zeta": _serialise_knob(candidate.zeta),
        "Psi": _serialise_knob(candidate.Psi),
        "channel_weights": list(candidate.channel_weights),
        "cross_channel_gains": list(candidate.cross_channel_gains),
    }


def _serialise_knob(value: float | FloatArray) -> object:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return float(array)
    return cast("list[object]", array.tolist())


def _decay_search_config(
    search_config: OfflinePolicySearchConfig,
    adaptive_config: AdaptiveReplayPolicySearchConfig,
) -> OfflinePolicySearchConfig:
    return OfflinePolicySearchConfig(
        K_step=_decay_step(search_config.K_step, adaptive_config),
        alpha_step=_decay_step(search_config.alpha_step, adaptive_config),
        zeta_step=_decay_step(search_config.zeta_step, adaptive_config),
        Psi_step=_decay_step(search_config.Psi_step, adaptive_config),
        channel_weight_step=_decay_step(
            search_config.channel_weight_step,
            adaptive_config,
        ),
        cross_channel_gain_step=_decay_step(
            search_config.cross_channel_gain_step,
            adaptive_config,
        ),
        include_baseline=search_config.include_baseline,
        max_abs_knob=search_config.max_abs_knob,
    )


def _decay_step(
    step: float,
    adaptive_config: AdaptiveReplayPolicySearchConfig,
) -> float:
    decayed = step * adaptive_config.step_decay
    if 0.0 < decayed < adaptive_config.min_step:
        return 0.0
    return decayed


def _search_config_to_record(config: OfflinePolicySearchConfig) -> dict[str, object]:
    return {
        "K_step": config.K_step,
        "alpha_step": config.alpha_step,
        "zeta_step": config.zeta_step,
        "Psi_step": config.Psi_step,
        "channel_weight_step": config.channel_weight_step,
        "cross_channel_gain_step": config.cross_channel_gain_step,
        "include_baseline": config.include_baseline,
        "max_abs_knob": config.max_abs_knob,
    }
