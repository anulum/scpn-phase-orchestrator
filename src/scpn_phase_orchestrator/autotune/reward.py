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
from numbers import Integral, Real
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "AutotunePolicyProposal",
    "AutotuneRewardReport",
    "KnobPolicyCandidate",
    "OfflinePolicySearchConfig",
    "PolicyProposalConfig",
    "RewardConfig",
    "RewardObservation",
    "evaluate_knob_policy",
    "generate_offline_policy_candidates",
    "propose_replay_policy",
    "rank_replay_candidates",
]

FloatArray: TypeAlias = NDArray[np.float64]
_COMPONENT_NAMES = frozenset(
    {
        "coherence_gain",
        "target_tracking",
        "bad_coherence",
        "actuation",
        "regime_churn",
        "unsafe",
    }
)


@dataclass(frozen=True)
class KnobPolicyCandidate:
    """Candidate phase-control knobs proposed by autotune tooling."""

    K: float | FloatArray = 0.0
    alpha: float | FloatArray = 0.0
    zeta: float | FloatArray = 0.0
    Psi: float | FloatArray = 0.0
    channel_weights: tuple[float, ...] = ()
    cross_channel_gains: tuple[float, ...] = ()


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
        _require_bool(self.unsafe, "unsafe")
        _require_bool(self.regime_changed, "regime_changed")


@dataclass(frozen=True)
class OfflinePolicySearchConfig:
    """Deterministic candidate-generation settings for replay searches."""

    K_step: float = 0.05
    alpha_step: float = 0.05
    zeta_step: float = 0.05
    Psi_step: float = 0.05
    channel_weight_step: float = 0.05
    cross_channel_gain_step: float = 0.05
    include_baseline: bool = True
    max_abs_knob: float | None = None

    def __post_init__(self) -> None:
        _require_bool(self.include_baseline, "include_baseline")
        for label, value in [
            ("K_step", self.K_step),
            ("alpha_step", self.alpha_step),
            ("zeta_step", self.zeta_step),
            ("Psi_step", self.Psi_step),
            ("channel_weight_step", self.channel_weight_step),
            ("cross_channel_gain_step", self.cross_channel_gain_step),
        ]:
            _require_non_negative_finite(value, label)
        if self.max_abs_knob is not None:
            _require_non_negative_finite(self.max_abs_knob, "max_abs_knob")
            if self.max_abs_knob == 0.0:
                raise ValueError("max_abs_knob must be positive when provided")


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
        _validate_component_order(self.component_order)


@dataclass(frozen=True)
class PolicyProposalConfig:
    """Acceptance gates for replay-trained policy proposals."""

    min_reward: float = -np.inf
    min_coherence: float = 0.0
    max_alternatives: int = 3
    require_safe: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.min_reward, (bool, np.bool_)) or not isinstance(
            self.min_reward, Real
        ):
            raise ValueError("min_reward must be finite or -inf")
        if not np.isfinite(float(self.min_reward)) and self.min_reward != -np.inf:
            raise ValueError("min_reward must be finite or -inf")
        _require_probability(self.min_coherence, "min_coherence")
        if isinstance(self.max_alternatives, (bool, np.bool_)) or not isinstance(
            self.max_alternatives, Integral
        ):
            raise TypeError("max_alternatives must be a non-negative integer")
        if int(self.max_alternatives) < 0:
            raise ValueError("max_alternatives must be non-negative")
        object.__setattr__(self, "max_alternatives", int(self.max_alternatives))
        _require_bool(self.require_safe, "require_safe")


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
                "cross_channel_gains": list(self.candidate.cross_channel_gains),
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


@dataclass(frozen=True)
class AutotunePolicyProposal:
    """Replay-trained policy proposal record for human or CI review."""

    accepted: bool
    selected: AutotuneRewardReport | None
    alternatives: tuple[AutotuneRewardReport, ...]
    reasons: tuple[str, ...]
    config: PolicyProposalConfig

    def to_audit_record(self) -> dict[str, object]:
        """Return a serialisable policy proposal record."""
        return {
            "accepted": self.accepted,
            "selected": (
                self.selected.to_audit_record() if self.selected is not None else None
            ),
            "alternatives": [
                alternative.to_audit_record() for alternative in self.alternatives
            ],
            "reasons": list(self.reasons),
            "config": {
                "min_reward": (
                    None
                    if self.config.min_reward == -np.inf
                    else self.config.min_reward
                ),
                "min_coherence": self.config.min_coherence,
                "max_alternatives": self.config.max_alternatives,
                "require_safe": self.config.require_safe,
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
    if top_k is not None:
        if isinstance(top_k, (bool, np.bool_)) or not isinstance(top_k, Integral):
            raise TypeError("top_k must be a positive integer when provided")
        top_k = int(top_k)
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive when provided")
    _require_bool(require_safe, "require_safe")

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


def propose_replay_policy(
    replay_candidates: Sequence[tuple[KnobPolicyCandidate, RewardObservation]],
    reward_config: RewardConfig | None = None,
    proposal_config: PolicyProposalConfig | None = None,
) -> AutotunePolicyProposal:
    """Build a reviewable policy proposal from replay-ranked candidates.

    The proposal is an audit artefact only. A caller still has to pass it
    through domain-specific policy review before any candidate is deployed.
    """
    active_config = proposal_config or PolicyProposalConfig()
    ranked = rank_replay_candidates(
        replay_candidates,
        reward_config,
        require_safe=active_config.require_safe,
    )
    selected = ranked[0]
    reasons: list[str] = []
    if selected.reward < active_config.min_reward:
        reasons.append(
            f"selected reward {selected.reward:.6g} below minimum "
            f"{active_config.min_reward:.6g}"
        )
    if selected.observation.coherence < active_config.min_coherence:
        reasons.append(
            f"selected coherence {selected.observation.coherence:.6g} below "
            f"minimum {active_config.min_coherence:.6g}"
        )
    if selected.observation.unsafe:
        reasons.append("selected rollout is marked unsafe")

    accepted = not reasons
    alternatives = tuple(ranked[1 : 1 + active_config.max_alternatives])
    return AutotunePolicyProposal(
        accepted=accepted,
        selected=selected if accepted else None,
        alternatives=alternatives,
        reasons=tuple(reasons),
        config=active_config,
    )


def generate_offline_policy_candidates(
    seed: KnobPolicyCandidate,
    config: OfflinePolicySearchConfig | None = None,
) -> tuple[KnobPolicyCandidate, ...]:
    """Generate deterministic replay-search candidates around a seed policy.

    The generator performs a bounded coordinate search over the universal knobs,
    channel weights, and cross-channel coupling gains. It does not inspect plant
    state and it does not apply actions; callers must evaluate the returned
    candidates through replay or simulation before ranking them.
    """
    active_config = config or OfflinePolicySearchConfig()
    _validate_candidate(seed)

    candidates: list[KnobPolicyCandidate] = []
    if active_config.include_baseline:
        candidates.append(seed)

    for knob, step in [
        ("K", active_config.K_step),
        ("alpha", active_config.alpha_step),
        ("zeta", active_config.zeta_step),
        ("Psi", active_config.Psi_step),
    ]:
        if step > 0.0:
            candidates.extend(
                _mutate_knob(seed, knob, delta, active_config.max_abs_knob)
                for delta in (-step, step)
            )

    if active_config.channel_weight_step > 0.0:
        for index in range(len(seed.channel_weights)):
            for delta in (
                -active_config.channel_weight_step,
                active_config.channel_weight_step,
            ):
                candidates.append(
                    _mutate_channel_weight(
                        seed,
                        index,
                        delta,
                        active_config.max_abs_knob,
                    )
                )

    if active_config.cross_channel_gain_step > 0.0:
        for index in range(len(seed.cross_channel_gains)):
            for delta in (
                -active_config.cross_channel_gain_step,
                active_config.cross_channel_gain_step,
            ):
                candidates.append(
                    _mutate_cross_channel_gain(
                        seed,
                        index,
                        delta,
                        active_config.max_abs_knob,
                    )
                )

    return _deduplicate_candidates(candidates)


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
    if candidate.cross_channel_gains:
        parts.append(
            _mean_square(np.asarray(candidate.cross_channel_gains, dtype=np.float64))
        )
    return float(sum(parts))


def _mean_square(value: float | FloatArray) -> float:
    array = _real_knob_array(value, "candidate knobs")
    if not np.all(np.isfinite(array)):
        raise ValueError("candidate knobs must be finite")
    if array.size == 0:
        return 0.0
    return float(np.mean(np.square(array)))


def _validate_candidate(candidate: KnobPolicyCandidate) -> None:
    for label, value in [
        ("K", candidate.K),
        ("alpha", candidate.alpha),
        ("zeta", candidate.zeta),
        ("Psi", candidate.Psi),
    ]:
        _real_knob_array(value, label)
    for weight in candidate.channel_weights:
        _require_non_negative_finite(weight, "channel weight")
    for gain in candidate.cross_channel_gains:
        _require_non_negative_finite(gain, "cross-channel gain")


def _mutate_knob(
    seed: KnobPolicyCandidate,
    knob: str,
    delta: float,
    max_abs_knob: float | None,
) -> KnobPolicyCandidate:
    values = {
        "K": seed.K,
        "alpha": seed.alpha,
        "zeta": seed.zeta,
        "Psi": seed.Psi,
    }
    values[knob] = _offset_knob(values[knob], delta, max_abs_knob)
    return KnobPolicyCandidate(
        K=values["K"],
        alpha=values["alpha"],
        zeta=values["zeta"],
        Psi=values["Psi"],
        channel_weights=seed.channel_weights,
        cross_channel_gains=seed.cross_channel_gains,
    )


def _offset_knob(
    value: float | FloatArray,
    delta: float,
    max_abs_knob: float | None,
) -> float | FloatArray:
    array = _real_knob_array(value, "candidate knobs") + delta
    if max_abs_knob is not None:
        array = np.clip(array, -max_abs_knob, max_abs_knob)
    if array.ndim == 0:
        return float(array)
    return array.astype(np.float64)


def _mutate_channel_weight(
    seed: KnobPolicyCandidate,
    index: int,
    delta: float,
    max_abs_knob: float | None,
) -> KnobPolicyCandidate:
    weights = list(seed.channel_weights)
    value = max(0.0, weights[index] + delta)
    if max_abs_knob is not None:
        value = min(max_abs_knob, value)
    weights[index] = value
    return KnobPolicyCandidate(
        K=seed.K,
        alpha=seed.alpha,
        zeta=seed.zeta,
        Psi=seed.Psi,
        channel_weights=tuple(weights),
        cross_channel_gains=seed.cross_channel_gains,
    )


def _mutate_cross_channel_gain(
    seed: KnobPolicyCandidate,
    index: int,
    delta: float,
    max_abs_knob: float | None,
) -> KnobPolicyCandidate:
    gains = list(seed.cross_channel_gains)
    value = max(0.0, gains[index] + delta)
    if max_abs_knob is not None:
        value = min(max_abs_knob, value)
    gains[index] = value
    return KnobPolicyCandidate(
        K=seed.K,
        alpha=seed.alpha,
        zeta=seed.zeta,
        Psi=seed.Psi,
        channel_weights=seed.channel_weights,
        cross_channel_gains=tuple(gains),
    )


def _deduplicate_candidates(
    candidates: Sequence[KnobPolicyCandidate],
) -> tuple[KnobPolicyCandidate, ...]:
    deduplicated: list[KnobPolicyCandidate] = []
    seen: set[tuple[object, ...]] = set()
    for candidate in candidates:
        key = (
            repr(_serialise_array(candidate.K)),
            repr(_serialise_array(candidate.alpha)),
            repr(_serialise_array(candidate.zeta)),
            repr(_serialise_array(candidate.Psi)),
            candidate.channel_weights,
            candidate.cross_channel_gains,
        )
        if key not in seen:
            seen.add(key)
            deduplicated.append(candidate)
    return tuple(deduplicated)


def _require_probability(value: float, label: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{label} must be finite and within [0, 1]")
    parsed = float(value)
    if not np.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{label} must be finite and within [0, 1]")


def _require_non_negative_finite(value: float, label: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{label} must be finite and non-negative")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")


def _serialise_array(value: float | FloatArray) -> float | list[object]:
    array = _real_knob_array(value, "candidate knobs")
    if array.ndim == 0:
        return float(array)
    return cast("list[object]", array.tolist())


def _real_knob_array(value: object, label: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype == np.bool_ or _contains_alias(raw, (bool, np.bool_)):
        raise ValueError(f"{label} must not contain boolean values")
    if np.iscomplexobj(raw) or _contains_alias(raw, (complex, np.complexfloating)):
        raise ValueError(f"{label} must be real-valued")
    try:
        array: FloatArray = raw.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be real-valued") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be finite")
    return array


def _contains_alias(raw: NDArray[np.generic], aliases: tuple[type, ...]) -> bool:
    if raw.dtype != object:
        return False
    return any(isinstance(item, aliases) for item in raw.ravel())


def _require_bool(value: object, label: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean")


def _validate_component_order(component_order: tuple[str, ...]) -> None:
    if not isinstance(component_order, tuple) or not component_order:
        raise ValueError("component_order must be a non-empty tuple")
    seen: set[str] = set()
    for component in component_order:
        if not isinstance(component, str):
            raise TypeError("component_order entries must be strings")
        if component not in _COMPONENT_NAMES:
            raise ValueError(f"unknown reward component {component!r}")
        if component in seen:
            raise ValueError(f"duplicate reward component {component!r}")
        seen.add(component)
