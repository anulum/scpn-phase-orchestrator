# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — reward validation edge tests

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

import scpn_phase_orchestrator.autotune.reward as reward_mod
from scpn_phase_orchestrator.autotune.reward import (
    KnobPolicyCandidate,
    PolicyProposalConfig,
    RewardConfig,
    RewardObservation,
    SafetyConstraintConfig,
    evaluate_knob_policy,
    propose_replay_policy,
)


def test_safety_constraint_config_requires_stl_and_safety_cost_bounds() -> None:
    with pytest.raises(ValueError, match="require_stl"):
        SafetyConstraintConfig(require_stl=True)

    with pytest.raises(ValueError, match="require_safety_cost"):
        SafetyConstraintConfig(require_safety_cost=True)


def test_policy_proposal_config_rejects_boolean_min_reward() -> None:
    with pytest.raises(ValueError, match="min_reward"):
        PolicyProposalConfig(min_reward=True)


def test_reward_config_rejects_empty_and_non_string_component_orders() -> None:
    with pytest.raises(ValueError, match="component_order"):
        RewardConfig(component_order=())

    with pytest.raises(TypeError, match="component_order entries"):
        RewardConfig(component_order=cast(tuple[str, ...], ("coherence_gain", 42)))


def test_reward_validation_rejects_non_numeric_knob_payloads() -> None:
    with pytest.raises(ValueError, match="real-valued"):
        evaluate_knob_policy(
            KnobPolicyCandidate(K=np.asarray(["not-a-number"], dtype=object)),
            RewardObservation(coherence=0.8),
        )


def test_reward_mean_square_defensively_rejects_non_finite_helper_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def non_finite_array(_value: object, _label: str) -> np.ndarray:
        return np.asarray([np.inf], dtype=np.float64)

    monkeypatch.setattr(reward_mod, "_real_knob_array", non_finite_array)

    with pytest.raises(ValueError, match="candidate knobs"):
        reward_mod._mean_square(0.0)


def test_policy_proposal_reports_stl_and_safety_cost_constraint_reasons() -> None:
    proposal = propose_replay_policy(
        (
            (
                KnobPolicyCandidate(K=0.1),
                RewardObservation(coherence=0.9, stl_robustness=-0.2, safety_cost=0.5),
            ),
        ),
        proposal_config=PolicyProposalConfig(
            safety_constraints=SafetyConstraintConfig(
                min_stl_robustness=0.0,
                max_safety_cost=0.1,
            ),
        ),
    )

    assert proposal.accepted is False
    assert proposal.reasons == (
        "no replay candidate satisfies Lyapunov/STL safety constraints",
    )
    assert proposal.alternatives[0].observation.stl_robustness == -0.2
