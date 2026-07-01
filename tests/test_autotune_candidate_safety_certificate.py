# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — candidate safety certificate tests

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.actuation.control_barrier import (
    BarrierCertificate,
    NeuralBarrier,
)
from scpn_phase_orchestrator.autotune.candidate_safety_certificate import (
    CandidateSafetyCertificate,
    certify_candidate_safety,
)
from scpn_phase_orchestrator.autotune.reward import (
    KnobPolicyCandidate,
    RewardObservation,
    SafetyConstraintConfig,
)

FloatArray = NDArray[np.float64]


def _candidate() -> KnobPolicyCandidate:
    return KnobPolicyCandidate(K=0.1, alpha=0.2, zeta=0.0, Psi=0.0)


def _barrier(offset: float = 0.0) -> NeuralBarrier:
    return NeuralBarrier(
        weights=(np.asarray([[1.0, 0.0]], dtype=np.float64),),
        biases=(np.asarray([offset], dtype=np.float64),),
    )


def test_certify_candidate_safety_builds_formally_proven_record() -> None:
    constraints = SafetyConstraintConfig(
        max_lyapunov_exponent=0.0,
        min_stl_robustness=0.05,
        max_safety_cost=0.2,
        require_lyapunov=True,
        require_stl=True,
        require_safety_cost=True,
    )
    certificate = certify_candidate_safety(
        _candidate(),
        (
            RewardObservation(
                coherence=0.8,
                lyapunov_exponent=-0.1,
                stl_robustness=0.2,
                safety_cost=0.1,
            ),
            RewardObservation(
                coherence=0.9,
                lyapunov_exponent=-0.05,
                stl_robustness=0.15,
                safety_cost=0.12,
            ),
        ),
        constraints,
        barrier=_barrier(),
        replay_states=np.asarray([[0.12, 1.0], [0.2, -1.0]], dtype=np.float64),
        forward_invariance=BarrierCertificate(
            verified=True,
            cells_checked=4,
            boundary_cells=2,
            worst_margin=0.03,
            boundary_shell=0.1,
            gamma=0.2,
        ),
    )

    assert isinstance(certificate, CandidateSafetyCertificate)
    assert certificate.safe is True
    assert certificate.evidence_kind == "formally-proven"
    assert certificate.barrier_worst_margin == pytest.approx(0.12)
    assert certificate.barrier_violations == 0
    assert certificate.lyapunov_margin == pytest.approx(0.05)
    assert certificate.stl_margin == pytest.approx(0.1)
    assert certificate.safety_cost_margin == pytest.approx(0.08)

    record = certificate.to_audit_record()
    assert record["digest"] == certificate.digest
    assert record["safe"] is True
    assert record["constraint_verdicts"] == {
        "lyapunov": True,
        "safety_cost": True,
        "stl": True,
    }


def test_certify_candidate_safety_fails_closed_for_measured_violations() -> None:
    certificate = certify_candidate_safety(
        _candidate(),
        (
            RewardObservation(
                coherence=0.6,
                lyapunov_exponent=0.2,
                stl_robustness=-0.1,
                safety_cost=0.5,
            ),
        ),
        SafetyConstraintConfig(
            max_lyapunov_exponent=0.0,
            min_stl_robustness=0.0,
            max_safety_cost=0.1,
        ),
        barrier=_barrier(),
        replay_states=np.asarray([[-0.2, 0.0], [0.3, 0.0]], dtype=np.float64),
        forward_invariance=BarrierCertificate(
            verified=False,
            cells_checked=2,
            boundary_cells=1,
            worst_margin=-0.1,
            boundary_shell=0.05,
            gamma=0.2,
        ),
    )

    assert certificate.safe is False
    assert certificate.evidence_kind == "measured"
    assert certificate.forward_invariance_verified is False
    assert certificate.barrier_worst_margin == pytest.approx(-0.2)
    assert certificate.barrier_violations == 1
    assert certificate.constraint_verdicts == {
        "lyapunov": False,
        "stl": False,
        "safety_cost": False,
    }


def test_certify_candidate_safety_requires_observations_and_barrier_states() -> None:
    with pytest.raises(ValueError, match="observations"):
        certify_candidate_safety(_candidate(), (), SafetyConstraintConfig())

    with pytest.raises(ValueError, match="replay_states"):
        certify_candidate_safety(
            _candidate(),
            (RewardObservation(coherence=0.8),),
            SafetyConstraintConfig(),
            barrier=_barrier(),
        )

    with pytest.raises(ValueError, match="at least one state"):
        certify_candidate_safety(
            _candidate(),
            (RewardObservation(coherence=0.8),),
            SafetyConstraintConfig(),
            barrier=_barrier(),
            replay_states=np.empty((0, 2), dtype=np.float64),
        )


def test_certify_candidate_safety_rejects_malformed_replay_states() -> None:
    with pytest.raises(ValueError, match="finite numeric state table"):
        certify_candidate_safety(
            _candidate(),
            (RewardObservation(coherence=0.8),),
            SafetyConstraintConfig(),
            barrier=_barrier(),
            replay_states=cast(FloatArray, [["not-a-number"]]),
        )

    with pytest.raises(ValueError, match="only finite values"):
        certify_candidate_safety(
            _candidate(),
            (RewardObservation(coherence=0.8),),
            SafetyConstraintConfig(),
            barrier=_barrier(),
            replay_states=np.asarray([[np.nan, 0.0]], dtype=np.float64),
        )


def test_certify_candidate_safety_fails_closed_when_required_evidence_is_missing() -> (
    None
):
    certificate = certify_candidate_safety(
        _candidate(),
        (RewardObservation(coherence=0.8),),
        SafetyConstraintConfig(
            max_lyapunov_exponent=0.0,
            min_stl_robustness=0.0,
            require_lyapunov=True,
        ),
    )

    assert certificate.safe is False
    assert certificate.lyapunov_margin is None
    assert certificate.stl_margin is None
    assert certificate.constraint_verdicts == {
        "lyapunov": False,
        "stl": True,
        "safety_cost": True,
    }


class _NonFiniteBarrier:
    def value(self, state: FloatArray) -> float:
        return float("nan")


def test_certify_candidate_safety_rejects_non_finite_barrier_margins() -> None:
    with pytest.raises(ValueError, match="finite barrier margin"):
        certify_candidate_safety(
            _candidate(),
            (RewardObservation(coherence=0.8),),
            SafetyConstraintConfig(),
            barrier=cast(NeuralBarrier, _NonFiniteBarrier()),
            replay_states=np.asarray([[1.0, 0.0]], dtype=np.float64),
        )
