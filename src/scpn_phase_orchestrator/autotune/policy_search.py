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
from dataclasses import dataclass
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
)

__all__ = [
    "ReplayPolicyEvaluator",
    "ReplayPolicySearchResult",
    "search_replay_policy",
]

ReplayPolicyEvaluator: TypeAlias = Callable[[KnobPolicyCandidate], RewardObservation]
FloatArray: TypeAlias = NDArray[np.float64]


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


def _candidate_to_record(candidate: KnobPolicyCandidate) -> dict[str, object]:
    return {
        "K": _serialise_knob(candidate.K),
        "alpha": _serialise_knob(candidate.alpha),
        "zeta": _serialise_knob(candidate.zeta),
        "Psi": _serialise_knob(candidate.Psi),
        "channel_weights": list(candidate.channel_weights),
    }


def _serialise_knob(value: float | FloatArray) -> object:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return float(array)
    return cast("list[object]", array.tolist())
