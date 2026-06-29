# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — replay policy-search validation tests

"""Validation tests for the replay policy-search result and knob validators.

Covers the ``__post_init__`` guards of ``ReplayPolicySearchResult`` and
``AdaptiveReplayPolicySearchResult`` and the low-level knob validators
(``_validate_candidate``, ``_real_array``, ``_require_non_negative_real``,
``_object_array_contains``) reached through invalid knob arrays.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

import scpn_phase_orchestrator.autotune.policy_search as _policy_search
from scpn_phase_orchestrator.autotune import (
    AdaptiveReplayPolicySearchConfig,
    AdaptiveReplayPolicySearchResult,
    KnobPolicyCandidate,
    OfflinePolicySearchConfig,
    PolicyProposalConfig,
    ReplayPolicySearchResult,
    RewardObservation,
    search_adaptive_replay_policy,
    search_replay_policy,
)
from scpn_phase_orchestrator.autotune.policy_search import (
    _object_array_contains,
    _real_array,
    _require_non_negative_real,
    _validate_candidate,
)
from scpn_phase_orchestrator.autotune.reward import AutotunePolicyProposal

assert _policy_search is not None


def _evaluator(candidate: KnobPolicyCandidate) -> RewardObservation:
    """Return a constant reward observation for any candidate."""
    return RewardObservation(coherence=0.9, previous_coherence=0.5)


def _valid_result() -> ReplayPolicySearchResult:
    """Return a valid single-round replay policy-search result."""
    return search_replay_policy(
        KnobPolicyCandidate(K=0.2),
        _evaluator,
        search_config=OfflinePolicySearchConfig(
            K_step=0.1,
            alpha_step=0.0,
            zeta_step=0.0,
            Psi_step=0.0,
            channel_weight_step=0.0,
            cross_channel_gain_step=0.0,
        ),
        proposal_config=PolicyProposalConfig(min_coherence=0.8),
    )


def _valid_adaptive() -> AdaptiveReplayPolicySearchResult:
    """Return a valid multi-round adaptive replay policy-search result."""
    return search_adaptive_replay_policy(
        KnobPolicyCandidate(K=0.2),
        _evaluator,
        adaptive_config=AdaptiveReplayPolicySearchConfig(iterations=1),
        proposal_config=PolicyProposalConfig(min_coherence=0.8),
    )


# --------------------------------------------------------------------------- #
# ReplayPolicySearchResult.__post_init__                                      #
# --------------------------------------------------------------------------- #
def test_result_rejects_non_tuple_candidates() -> None:
    result = _valid_result()
    with pytest.raises(TypeError, match="candidates must be a tuple"):
        ReplayPolicySearchResult(
            seed=result.seed,
            candidates=cast("tuple[KnobPolicyCandidate, ...]", list(result.candidates)),
            proposal=result.proposal,
        )


def test_result_rejects_empty_candidates() -> None:
    result = _valid_result()
    with pytest.raises(ValueError, match="candidates must not be empty"):
        ReplayPolicySearchResult(
            seed=result.seed, candidates=(), proposal=result.proposal
        )


def test_result_rejects_a_non_proposal() -> None:
    result = _valid_result()
    with pytest.raises(TypeError, match="proposal must be AutotunePolicyProposal"):
        ReplayPolicySearchResult(
            seed=result.seed,
            candidates=result.candidates,
            proposal=cast("AutotunePolicyProposal", "not a proposal"),
        )


# --------------------------------------------------------------------------- #
# AdaptiveReplayPolicySearchResult.__post_init__                              #
# --------------------------------------------------------------------------- #
def test_adaptive_rejects_non_tuple_rounds() -> None:
    adaptive = _valid_adaptive()
    with pytest.raises(TypeError, match="rounds must be a tuple"):
        AdaptiveReplayPolicySearchResult(
            seed=adaptive.seed,
            rounds=cast("tuple[ReplayPolicySearchResult, ...]", list(adaptive.rounds)),
            proposal=adaptive.proposal,
            config=adaptive.config,
        )


def test_adaptive_rejects_empty_rounds() -> None:
    adaptive = _valid_adaptive()
    with pytest.raises(ValueError, match="rounds must not be empty"):
        AdaptiveReplayPolicySearchResult(
            seed=adaptive.seed,
            rounds=(),
            proposal=adaptive.proposal,
            config=adaptive.config,
        )


def test_adaptive_rejects_a_non_round_element() -> None:
    adaptive = _valid_adaptive()
    with pytest.raises(
        TypeError, match=r"rounds\[0\] must be ReplayPolicySearchResult"
    ):
        AdaptiveReplayPolicySearchResult(
            seed=adaptive.seed,
            rounds=cast("tuple[ReplayPolicySearchResult, ...]", ("not a round",)),
            proposal=adaptive.proposal,
            config=adaptive.config,
        )


def test_adaptive_rejects_a_non_proposal() -> None:
    adaptive = _valid_adaptive()
    with pytest.raises(TypeError, match="proposal must be AutotunePolicyProposal"):
        AdaptiveReplayPolicySearchResult(
            seed=adaptive.seed,
            rounds=adaptive.rounds,
            proposal=cast("AutotunePolicyProposal", "not a proposal"),
            config=adaptive.config,
        )


def test_adaptive_rejects_a_non_config() -> None:
    adaptive = _valid_adaptive()
    with pytest.raises(
        TypeError, match="config must be AdaptiveReplayPolicySearchConfig"
    ):
        AdaptiveReplayPolicySearchResult(
            seed=adaptive.seed,
            rounds=adaptive.rounds,
            proposal=adaptive.proposal,
            config=cast("AdaptiveReplayPolicySearchConfig", "not a config"),
        )


# --------------------------------------------------------------------------- #
# Low-level knob validators                                                   #
# --------------------------------------------------------------------------- #
def test_validate_candidate_rejects_a_non_candidate() -> None:
    with pytest.raises(TypeError, match="seed must be KnobPolicyCandidate"):
        _validate_candidate("not a candidate", "seed")


def test_require_non_negative_real_rejects_a_non_finite_value() -> None:
    with pytest.raises(ValueError, match="weight must be finite"):
        _require_non_negative_real(float("nan"), "weight")


def test_real_array_rejects_complex_values() -> None:
    with pytest.raises(ValueError, match="K must be real-valued"):
        _real_array(np.array([1.0 + 0.0j, 2.0 + 0.0j]), "K")


def test_real_array_rejects_non_numeric_values() -> None:
    with pytest.raises(ValueError, match="K must be real-valued"):
        _real_array(np.array(["a", "b"], dtype=object), "K")


def test_real_array_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="K must be finite"):
        _real_array(np.array([1.0, np.nan]), "K")


def test_object_array_contains_detects_an_alias() -> None:
    raw = np.array([True, 1], dtype=object)
    assert _object_array_contains(raw, (bool, np.bool_)) is True


def test_object_array_contains_is_false_for_numeric_arrays() -> None:
    assert _object_array_contains(np.array([1.0, 2.0]), (bool, np.bool_)) is False
