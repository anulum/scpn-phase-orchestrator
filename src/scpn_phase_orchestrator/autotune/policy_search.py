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
from numbers import Integral, Real
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
        if not isinstance(self.base_search_config, OfflinePolicySearchConfig):
            raise TypeError("base_search_config must be OfflinePolicySearchConfig")
        object.__setattr__(
            self,
            "iterations",
            _require_positive_integer(self.iterations, "iterations"),
        )
        object.__setattr__(
            self,
            "step_decay",
            _require_bounded_real(
                self.step_decay,
                "step_decay",
                lower=0.0,
                upper=1.0,
                lower_inclusive=False,
            ),
        )
        object.__setattr__(
            self,
            "improvement_tolerance",
            _require_non_negative_real(
                self.improvement_tolerance,
                "improvement_tolerance",
            ),
        )
        object.__setattr__(
            self,
            "min_step",
            _require_non_negative_real(self.min_step, "min_step"),
        )


@dataclass(frozen=True)
class ReplayPolicySearchResult:
    """Replay-only policy-search result suitable for audit review."""

    seed: KnobPolicyCandidate
    candidates: tuple[KnobPolicyCandidate, ...]
    proposal: AutotunePolicyProposal

    def __post_init__(self) -> None:
        _validate_candidate(self.seed, "seed")
        if not isinstance(self.candidates, tuple):
            raise TypeError("candidates must be a tuple of KnobPolicyCandidate")
        if not self.candidates:
            raise ValueError("candidates must not be empty")
        for index, candidate in enumerate(self.candidates):
            _validate_candidate(candidate, f"candidates[{index}]")
        if not isinstance(self.proposal, AutotunePolicyProposal):
            raise TypeError("proposal must be AutotunePolicyProposal")

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

    def __post_init__(self) -> None:
        _validate_candidate(self.seed, "seed")
        if not isinstance(self.rounds, tuple):
            raise TypeError("rounds must be a tuple of ReplayPolicySearchResult")
        if not self.rounds:
            raise ValueError("rounds must not be empty")
        for index, round_result in enumerate(self.rounds):
            if not isinstance(round_result, ReplayPolicySearchResult):
                raise TypeError(
                    f"rounds[{index}] must be ReplayPolicySearchResult"
                )
        if not isinstance(self.proposal, AutotunePolicyProposal):
            raise TypeError("proposal must be AutotunePolicyProposal")
        if not isinstance(self.config, AdaptiveReplayPolicySearchConfig):
            raise TypeError("config must be AdaptiveReplayPolicySearchConfig")

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
    _validate_candidate(seed, "seed")
    if not callable(evaluator):
        raise TypeError("evaluator must be callable")
    active_search_config = cast(
        "OfflinePolicySearchConfig | None",
        _optional_config(
            search_config,
            OfflinePolicySearchConfig,
            "search_config",
        ),
    )
    active_reward_config = cast(
        "RewardConfig | None",
        _optional_config(
            reward_config,
            RewardConfig,
            "reward_config",
        ),
    )
    active_proposal_config = cast(
        "PolicyProposalConfig | None",
        _optional_config(
            proposal_config,
            PolicyProposalConfig,
            "proposal_config",
        ),
    )

    candidates = generate_offline_policy_candidates(seed, active_search_config)
    if not candidates:
        raise ValueError("replay policy search generated no candidates")

    replay_observations = tuple(
        (candidate, _evaluate_candidate(evaluator, candidate))
        for candidate in candidates
    )
    proposal = propose_replay_policy(
        replay_observations,
        reward_config=active_reward_config,
        proposal_config=active_proposal_config,
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
    _validate_candidate(seed, "seed")
    if not callable(evaluator):
        raise TypeError("evaluator must be callable")
    active_config = (
        AdaptiveReplayPolicySearchConfig()
        if adaptive_config is None
        else cast(
            "AdaptiveReplayPolicySearchConfig",
            _require_config_type(
                adaptive_config,
                AdaptiveReplayPolicySearchConfig,
                "adaptive_config",
            ),
        )
    )
    active_reward_config = cast(
        "RewardConfig | None",
        _optional_config(
            reward_config,
            RewardConfig,
            "reward_config",
        ),
    )
    active_proposal_config = (
        PolicyProposalConfig()
        if proposal_config is None
        else cast(
            "PolicyProposalConfig",
            _require_config_type(
                proposal_config,
                PolicyProposalConfig,
                "proposal_config",
            ),
        )
    )
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
            (candidate, _evaluate_candidate(evaluator, candidate))
            for candidate in candidates
        )
        replay_observations.extend(round_observations)
        round_proposal = propose_replay_policy(
            round_observations,
            reward_config=active_reward_config,
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
            active_reward_config,
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
        reward_config=active_reward_config,
        proposal_config=active_proposal_config,
    )
    return AdaptiveReplayPolicySearchResult(
        seed=seed,
        rounds=tuple(rounds),
        proposal=final_proposal,
        config=active_config,
    )


def _candidate_to_record(candidate: KnobPolicyCandidate) -> dict[str, object]:
    _validate_candidate(candidate, "candidate")
    return {
        "K": _serialise_knob(candidate.K),
        "alpha": _serialise_knob(candidate.alpha),
        "zeta": _serialise_knob(candidate.zeta),
        "Psi": _serialise_knob(candidate.Psi),
        "channel_weights": list(candidate.channel_weights),
        "cross_channel_gains": list(candidate.cross_channel_gains),
    }


def _serialise_knob(value: float | FloatArray) -> object:
    array = _real_array(value, "candidate knob")
    if array.ndim == 0:
        return float(array)
    return cast("list[object]", array.tolist())


def _evaluate_candidate(
    evaluator: ReplayPolicyEvaluator,
    candidate: KnobPolicyCandidate,
) -> RewardObservation:
    observation = evaluator(candidate)
    if not isinstance(observation, RewardObservation):
        raise TypeError("evaluator must return RewardObservation")
    return observation


def _optional_config(
    value: object | None,
    expected_type: type[object],
    label: str,
) -> object | None:
    if value is None:
        return None
    return _require_config_type(value, expected_type, label)


def _require_config_type(
    value: object,
    expected_type: type[object],
    label: str,
) -> object:
    if not isinstance(value, expected_type):
        raise TypeError(f"{label} must be {expected_type.__name__}")
    return value


def _validate_candidate(candidate: object, label: str) -> None:
    if not isinstance(candidate, KnobPolicyCandidate):
        raise TypeError(f"{label} must be KnobPolicyCandidate")
    for knob_name, value in [
        ("K", candidate.K),
        ("alpha", candidate.alpha),
        ("zeta", candidate.zeta),
        ("Psi", candidate.Psi),
    ]:
        _real_array(value, f"{label}.{knob_name}")
    for index, weight in enumerate(candidate.channel_weights):
        _require_non_negative_real(
            weight,
            f"{label}.channel_weights[{index}]",
        )
    for index, gain in enumerate(candidate.cross_channel_gains):
        _require_non_negative_real(
            gain,
            f"{label}.cross_channel_gains[{index}]",
        )


def _require_positive_integer(value: object, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be a positive integer")
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{label} must be positive")
    return parsed


def _require_non_negative_real(value: object, label: str) -> float:
    return _require_bounded_real(
        value,
        label,
        lower=0.0,
        upper=None,
        lower_inclusive=True,
    )


def _require_bounded_real(
    value: object,
    label: str,
    *,
    lower: float,
    upper: float | None,
    lower_inclusive: bool,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{label} must be finite")
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    lower_ok = parsed >= lower if lower_inclusive else parsed > lower
    upper_ok = True if upper is None else parsed <= upper
    if not lower_ok or not upper_ok:
        if upper is None:
            interval = f"{'[' if lower_inclusive else '('}{lower}, inf)"
        else:
            interval = f"{'[' if lower_inclusive else '('}{lower}, {upper}]"
        raise ValueError(f"{label} must be within {interval}")
    return parsed


def _real_array(value: object, label: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype == np.bool_ or _object_array_contains(raw, (bool, np.bool_)):
        raise ValueError(f"{label} must not contain boolean values")
    if np.iscomplexobj(raw) or _object_array_contains(
        raw,
        (complex, np.complexfloating),
    ):
        raise ValueError(f"{label} must be real-valued")
    try:
        array: FloatArray = raw.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be real-valued") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be finite")
    return array


def _object_array_contains(raw: np.ndarray, aliases: tuple[type, ...]) -> bool:
    if raw.dtype != object:
        return False
    return any(isinstance(item, aliases) for item in raw.ravel())


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
