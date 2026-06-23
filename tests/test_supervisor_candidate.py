# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — supervisor-candidate bundle tests

"""Tests for the auditable supervisor-candidate evidence bundle.

The bundle is checked over the numeric-provenance validation, the
incumbent comparison, the assembled reward/attribution/safety sub-records, the
barrier path, the candidate serialisation of array-valued knobs, the
``studio.supervisor_candidate.v1`` record shape, and the content-address seal.
"""

from __future__ import annotations

from collections.abc import Callable

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
    SupervisorCandidateComparison,
    build_supervisor_candidate_bundle,
)

_OBSERVATION = RewardObservation(coherence=0.8)
_CONFIG = RewardConfig()
_PROVENANCE = NumericProvenance(active_backend="python", parity_tolerance=1e-9)


def _evaluator(scale: float) -> Callable[[KnobPolicyCandidate], AutotuneRewardReport]:
    def evaluate(candidate: KnobPolicyCandidate) -> AutotuneRewardReport:
        value = scale * float(candidate.alpha)
        return AutotuneRewardReport(
            reward=value,
            components={"main": value},
            candidate=candidate,
            observation=_OBSERVATION,
            config=_CONFIG,
        )

    return evaluate


def _bundle(scale: float = 2.0):
    return build_supervisor_candidate_bundle(
        KnobPolicyCandidate(alpha=1.0),
        KnobPolicyCandidate(alpha=0.0),
        KnobPolicyCandidate(alpha=0.5),
        _evaluator(scale),
        observations=[_OBSERVATION],
        constraints=SafetyConstraintConfig(),
        safety_tier="research",
        numeric_provenance=_PROVENANCE,
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"active_backend": "", "parity_tolerance": 1e-9}, "active_backend"),
        ({"active_backend": "python", "parity_tolerance": -1.0}, "parity_tolerance"),
    ],
)
def test_numeric_provenance_validation(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        NumericProvenance(**kwargs)  # type: ignore[arg-type]


def test_numeric_provenance_audit_record() -> None:
    record = _PROVENANCE.to_audit_record()
    assert record == {"active_backend": "python", "parity_tolerance": 1e-9}


def test_comparison_reports_improvement_and_deltas() -> None:
    bundle = _bundle(scale=2.0)
    comparison = bundle.comparison
    assert comparison.candidate_reward == pytest.approx(2.0)
    assert comparison.incumbent_reward == pytest.approx(1.0)
    assert comparison.reward_delta == pytest.approx(1.0)
    assert comparison.improved is True
    assert comparison.component_deltas["main"] == pytest.approx(1.0)


def test_comparison_not_improved_when_candidate_is_worse() -> None:
    bundle = build_supervisor_candidate_bundle(
        KnobPolicyCandidate(alpha=0.2),
        KnobPolicyCandidate(alpha=0.0),
        KnobPolicyCandidate(alpha=1.0),
        _evaluator(1.0),
        observations=[_OBSERVATION],
        constraints=SafetyConstraintConfig(),
        safety_tier="research",
        numeric_provenance=_PROVENANCE,
    )
    assert bundle.comparison.improved is False
    assert bundle.safe_and_improved is False


def test_comparison_handles_disjoint_components() -> None:
    incumbent = AutotuneRewardReport(
        reward=1.0,
        components={"only_incumbent": 1.0},
        candidate=KnobPolicyCandidate(),
        observation=_OBSERVATION,
        config=_CONFIG,
    )
    candidate = AutotuneRewardReport(
        reward=2.0,
        components={"only_candidate": 2.0},
        candidate=KnobPolicyCandidate(),
        observation=_OBSERVATION,
        config=_CONFIG,
    )
    comparison = SupervisorCandidateComparison(
        incumbent_reward=incumbent.reward,
        candidate_reward=candidate.reward,
        reward_delta=candidate.reward - incumbent.reward,
        component_deltas={"only_candidate": 2.0, "only_incumbent": -1.0},
        improved=True,
    )
    record = comparison.to_audit_record()
    assert list(record["component_deltas"]) == ["only_candidate", "only_incumbent"]


def test_bundle_assembles_all_sub_records() -> None:
    bundle = _bundle()
    assert bundle.reward.reward == pytest.approx(2.0)
    assert bundle.attribution.attributions[0].knob == "alpha"
    assert bundle.safety.safe is True
    assert bundle.safe_and_improved is True
    assert bundle.evidence_kind == bundle.safety.evidence_kind == "measured"


def test_bundle_record_is_schema_tagged_and_sealed() -> None:
    bundle = _bundle()
    record = bundle.to_audit_record()
    assert record["schema"] == "studio.supervisor_candidate.v1"
    assert record["safety_tier"] == "research"
    assert record["digest"] == bundle.digest
    assert record["numeric_provenance"] == {
        "active_backend": "python",
        "parity_tolerance": 1e-9,
    }
    assert set(record) >= {
        "candidate",
        "reward",
        "attribution",
        "safety",
        "comparison",
        "evidence_kind",
    }
    # All six knob fields are serialised, including the uppercase-led K and Psi.
    assert set(record["candidate"]) == {  # type: ignore[arg-type]
        "K",
        "alpha",
        "zeta",
        "Psi",
        "channel_weights",
        "cross_channel_gains",
    }


def test_bundle_digest_is_deterministic() -> None:
    assert _bundle().digest == _bundle().digest


def test_bundle_with_barrier_certifies_replay_states() -> None:
    barrier = NeuralBarrier(weights=(np.array([[1.0]]),), biases=(np.array([0.0]),))
    bundle = build_supervisor_candidate_bundle(
        KnobPolicyCandidate(alpha=1.0),
        KnobPolicyCandidate(alpha=0.0),
        KnobPolicyCandidate(alpha=0.5),
        _evaluator(2.0),
        observations=[_OBSERVATION],
        constraints=SafetyConstraintConfig(),
        safety_tier="research",
        numeric_provenance=_PROVENANCE,
        barrier=barrier,
        replay_states=np.array([[0.5], [0.9]]),
    )
    assert bundle.safety.barrier_worst_margin == pytest.approx(0.5)
    assert bundle.safety.barrier_violations == 0


def test_bundle_serialises_array_valued_alpha() -> None:
    def evaluate(candidate: KnobPolicyCandidate) -> AutotuneRewardReport:
        value = float(np.sum(np.asarray(candidate.alpha, dtype=float)))
        return AutotuneRewardReport(
            reward=value,
            components={"main": value},
            candidate=candidate,
            observation=_OBSERVATION,
            config=_CONFIG,
        )

    bundle = build_supervisor_candidate_bundle(
        KnobPolicyCandidate(alpha=np.array([1.0, 2.0])),
        KnobPolicyCandidate(alpha=np.array([0.0, 0.0])),
        KnobPolicyCandidate(alpha=np.array([0.5, 0.5])),
        evaluate,
        observations=[_OBSERVATION],
        constraints=SafetyConstraintConfig(),
        safety_tier="research",
        numeric_provenance=_PROVENANCE,
    )
    record = bundle.to_audit_record()
    assert record["candidate"]["alpha"] == [1.0, 2.0]  # type: ignore[index]
