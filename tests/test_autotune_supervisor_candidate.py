# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — supervisor candidate bundle tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.actuation.control_barrier import NeuralBarrier
from scpn_phase_orchestrator.autotune.reward import (
    AutotuneRewardReport,
    KnobPolicyCandidate,
    RewardConfig,
    RewardObservation,
    SafetyConstraintConfig,
)
from scpn_phase_orchestrator.autotune.supervisor_candidate import (
    NumericProvenance,
    SupervisorCandidateBundle,
    build_supervisor_candidate_bundle,
)


def _reward(candidate: KnobPolicyCandidate) -> AutotuneRewardReport:
    k_value = float(np.sum(np.asarray(candidate.K, dtype=np.float64)))
    alpha_value = float(np.sum(np.asarray(candidate.alpha, dtype=np.float64)))
    reward = 2.0 * k_value + alpha_value
    return AutotuneRewardReport(
        reward=reward,
        components={"coupling": 2.0 * k_value, "adaptive": alpha_value},
        candidate=candidate,
        observation=RewardObservation(coherence=0.8),
        config=RewardConfig(component_order=("coherence_gain",)),
    )


def _barrier() -> NeuralBarrier:
    return NeuralBarrier(
        weights=(np.asarray([[1.0, 0.0]], dtype=np.float64),),
        biases=(np.asarray([0.0], dtype=np.float64),),
    )


def test_build_supervisor_candidate_bundle_seals_safe_improvement_record() -> None:
    baseline = KnobPolicyCandidate(K=0.0, alpha=0.0)
    candidate = KnobPolicyCandidate(K=0.2, alpha=0.3)
    incumbent = KnobPolicyCandidate(K=0.1, alpha=0.0)

    bundle = build_supervisor_candidate_bundle(
        candidate,
        baseline,
        incumbent,
        _reward,
        observations=(
            RewardObservation(
                coherence=0.8,
                lyapunov_exponent=-0.1,
                stl_robustness=0.2,
                safety_cost=0.01,
            ),
        ),
        constraints=SafetyConstraintConfig(
            max_lyapunov_exponent=0.0,
            min_stl_robustness=0.0,
            max_safety_cost=0.1,
        ),
        safety_tier="review",
        numeric_provenance=NumericProvenance(
            active_backend="python",
            parity_tolerance=1.0e-12,
        ),
        barrier=_barrier(),
        replay_states=np.asarray([[0.2, 0.0], [0.3, 0.1]], dtype=np.float64),
    )

    assert isinstance(bundle, SupervisorCandidateBundle)
    assert bundle.safe_and_improved is True
    assert bundle.evidence_kind == bundle.safety.evidence_kind
    assert bundle.comparison.reward_delta == pytest.approx(0.5)

    record = bundle.to_audit_record()
    assert record["schema"] == "studio.supervisor_candidate.v1"
    assert record["digest"] == bundle.digest
    assert record["candidate"]["K"] == [0.2]
    assert record["numeric_provenance"] == {
        "active_backend": "python",
        "parity_tolerance": 1.0e-12,
    }
    assert record["comparison"]["component_deltas"] == {
        "adaptive": 0.3,
        "coupling": 0.2,
    }


def test_build_supervisor_candidate_bundle_records_unsafe_non_improvement() -> None:
    bundle = build_supervisor_candidate_bundle(
        KnobPolicyCandidate(K=0.05),
        KnobPolicyCandidate(K=0.0),
        KnobPolicyCandidate(K=0.2),
        _reward,
        observations=(RewardObservation(coherence=0.7, safety_cost=0.5),),
        constraints=SafetyConstraintConfig(max_safety_cost=0.1),
        safety_tier="offline",
        numeric_provenance=NumericProvenance(
            active_backend="rust", parity_tolerance=0.0
        ),
    )

    assert bundle.safe_and_improved is False
    assert bundle.safety.safe is False
    assert bundle.comparison.improved is False
    assert bundle.to_audit_record()["safety"]["safe"] is False


def test_numeric_provenance_rejects_invalid_fields() -> None:
    with pytest.raises(ValueError, match="active_backend"):
        NumericProvenance(active_backend="", parity_tolerance=0.0)

    with pytest.raises(ValueError, match="parity_tolerance"):
        NumericProvenance(active_backend="python", parity_tolerance=-1.0)
