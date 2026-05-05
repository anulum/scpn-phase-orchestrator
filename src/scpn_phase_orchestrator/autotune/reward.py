# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Autotune reward evaluation

"""Auditable reward scoring for candidate autotune policies."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "AutotuneRewardReport",
    "KnobPolicyCandidate",
    "RewardConfig",
    "RewardObservation",
    "evaluate_knob_policy",
    "rank_replay_candidates",
]

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class KnobPolicyCandidate:
    """Candidate phase-control knobs proposed by autotune tooling."""

    K: float | FloatArray = 0.0
    alpha: float | FloatArray = 0.0
    zeta: float | FloatArray = 0.0
    Psi: float | FloatArray = 0.0
    channel_weights: tuple[float, ...] = ()


@dataclass(frozen=True)
class RewardObservation:
    """Observed rollout metrics used to score one policy candidate."""

    coherence: float
    previous_coherence: float | None = None
    unsafe: bool = False
    regime_changed: bool = False

    def __post_init__(self) -> None:
        _require_probability(self.coherence, "coherence")
        if self.previous_coherence is not None:
            _require_probability(self.previous_coherence, "previous_coherence")


@dataclass(frozen=True)
class RewardConfig:
    """Weights for auditable coherence-minus-risk autotune reward."""

    target_coherence: float = 1.0
    bad_coherence_threshold: float = 0.35
    coherence_weight: float = 1.0
    bad_coherence_penalty: float = 2.0
    actuation_penalty: float = 0.01
    churn_penalty: float = 0.1
    unsafe_penalty: float = 10.0
    component_order: tuple[str, ...] = field(
        default=(
            "coherence_gain",
            "target_tracking",
            "bad_coherence",
            "actuation",
            "regime_churn",
            "unsafe",
        )
    )

    def __post_init__(self) -> None:
        _require_probability(self.target_coherence, "target_coherence")
        _require_probability(self.bad_coherence_threshold, "bad_coherence_threshold")
        for label, value in [
            ("coherence_weight", self.coherence_weight),
            ("bad_coherence_penalty", self.bad_coherence_penalty),
            ("actuation_penalty", self.actuation_penalty),
            ("churn_penalty", self.churn_penalty),
            ("unsafe_penalty", self.unsafe_penalty),
        ]:
            _require_non_negative_finite(value, label)


@dataclass(frozen=True)
class AutotuneRewardReport:
    """Reward result suitable for policy search and audit logs."""

    reward: float
    components: dict[str, float]
    candidate: KnobPolicyCandidate
    observation: RewardObservation
    config: RewardConfig

    def to_audit_record(self) -> dict[str, object]:
        """Return a serialisable reward record."""
        return {
            "reward": self.reward,
            "components": dict(self.components),
            "candidate": {
                "K": _serialise_array(self.candidate.K),
                "alpha": _serialise_array(self.candidate.alpha),
                "zeta": _serialise_array(self.candidate.zeta),
                "Psi": _serialise_array(self.candidate.Psi),
                "channel_weights": list(self.candidate.channel_weights),
            },
            "observation": {
                "coherence": self.observation.coherence,
                "previous_coherence": self.observation.previous_coherence,
                "unsafe": self.observation.unsafe,
                "regime_changed": self.observation.regime_changed,
            },
            "config": {
                "target_coherence": self.config.target_coherence,
                "bad_coherence_threshold": self.config.bad_coherence_threshold,
                "coherence_weight": self.config.coherence_weight,
                "bad_coherence_penalty": self.config.bad_coherence_penalty,
                "actuation_penalty": self.config.actuation_penalty,
                "churn_penalty": self.config.churn_penalty,
                "unsafe_penalty": self.config.unsafe_penalty,
            },
        }


def evaluate_knob_policy(
    candidate: KnobPolicyCandidate,
    observation: RewardObservation,
    config: RewardConfig | None = None,
) -> AutotuneRewardReport:
    """Score a candidate policy from coherence and safety metrics.

    The reward is intentionally model-free and side-effect free. Training
    systems can use it to rank replay candidates before any future policy
    learner is allowed to propose production control actions.
    """
    active_config = config or RewardConfig()
    _validate_candidate(candidate)
    previous = (
        active_config.bad_coherence_threshold
        if observation.previous_coherence is None
        else observation.previous_coherence
    )
    coherence_gain = active_config.coherence_weight * (observation.coherence - previous)
    target_tracking = -active_config.coherence_weight * max(
        0.0, active_config.target_coherence - observation.coherence
    )
    bad_deficit = max(
        0.0, active_config.bad_coherence_threshold - observation.coherence
    )
    bad_coherence = -active_config.bad_coherence_penalty * bad_deficit
    actuation = -active_config.actuation_penalty * _actuation_energy(candidate)
    regime_churn = -active_config.churn_penalty if observation.regime_changed else 0.0
    unsafe = -active_config.unsafe_penalty if observation.unsafe else 0.0

    components = {
        "coherence_gain": coherence_gain,
        "target_tracking": target_tracking,
        "bad_coherence": bad_coherence,
        "actuation": actuation,
        "regime_churn": regime_churn,
        "unsafe": unsafe,
    }
    reward = float(sum(components[name] for name in active_config.component_order))
    return AutotuneRewardReport(
        reward=reward,
        components=components,
        candidate=candidate,
        observation=observation,
        config=active_config,
    )


def rank_replay_candidates(
    replay_candidates: Sequence[tuple[KnobPolicyCandidate, RewardObservation]],
    config: RewardConfig | None = None,
    *,
    top_k: int | None = None,
    require_safe: bool = True,
) -> tuple[AutotuneRewardReport, ...]:
    """Rank replay-evaluated policy candidates by reward.

    This helper is the non-actuating bridge between reward scoring and future
    policy learners. It consumes replay or simulation observations, filters
    unsafe rollouts by default, and returns audit-ready reports sorted from
    highest to lowest reward.
    """
    if not replay_candidates:
        raise ValueError("replay candidate ranking requires at least one candidate")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive when provided")

    reports = [
        evaluate_knob_policy(candidate, observation, config)
        for candidate, observation in replay_candidates
        if not require_safe or not observation.unsafe
    ]
    if not reports:
        raise ValueError("no safe replay candidates remain after filtering")

    ranked = sorted(
        reports,
        key=lambda report: (
            report.reward,
            report.components["coherence_gain"],
            report.components["actuation"],
        ),
        reverse=True,
    )
    if top_k is None:
        return tuple(ranked)
    return tuple(ranked[:top_k])


def _actuation_energy(candidate: KnobPolicyCandidate) -> float:
    parts = [
        _mean_square(candidate.K),
        _mean_square(candidate.alpha),
        _mean_square(candidate.zeta),
        _mean_square(candidate.Psi),
    ]
    if candidate.channel_weights:
        parts.append(
            _mean_square(np.asarray(candidate.channel_weights, dtype=np.float64))
        )
    return float(sum(parts))


def _mean_square(value: float | FloatArray) -> float:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("candidate knobs must be finite")
    if array.size == 0:
        return 0.0
    return float(np.mean(np.square(array)))


def _validate_candidate(candidate: KnobPolicyCandidate) -> None:
    for weight in candidate.channel_weights:
        _require_non_negative_finite(weight, "channel weight")


def _require_probability(value: float, label: str) -> None:
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be finite and within [0, 1]")


def _require_non_negative_finite(value: float, label: str) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")


def _serialise_array(value: float | FloatArray) -> float | list[object]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return float(array)
    return cast("list[object]", array.tolist())
