# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — candidate safety certificate tests

"""Tests for binding an autotune candidate to a safety certificate.

The certificate is checked over the barrier margin, the Lyapunov/STL/safety-cost
constraints (including the fail-closed require-evidence path), the
measured-versus-formally-proven evidence selection, the content-address seal, and
the input-validation paths.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.actuation.control_barrier import (
    BarrierCertificate,
    NeuralBarrier,
)
from scpn_phase_orchestrator.autotune.candidate_safety_certificate import (
    certify_candidate_safety,
)
from scpn_phase_orchestrator.autotune.reward import (
    KnobPolicyCandidate,
    RewardObservation,
    SafetyConstraintConfig,
)

_CANDIDATE = KnobPolicyCandidate(alpha=0.1, zeta=0.05)


def _barrier() -> NeuralBarrier:
    # h(x) = x[0]; the safe set is {x : x[0] >= 0}.
    return NeuralBarrier(weights=(np.array([[1.0]]),), biases=(np.array([0.0]),))


def _certificate(boundary_shell: float = 0.0, *, verified: bool = True) -> BarrierCertificate:
    return BarrierCertificate(
        verified=verified,
        cells_checked=100,
        boundary_cells=10,
        worst_margin=0.5,
        boundary_shell=boundary_shell,
        gamma=0.5,
    )


def _obs(**kwargs: float | None) -> RewardObservation:
    return RewardObservation(coherence=0.8, **kwargs)


def test_barrier_safe_replay_has_no_violations() -> None:
    result = certify_candidate_safety(
        _CANDIDATE,
        [_obs()],
        SafetyConstraintConfig(),
        barrier=_barrier(),
        replay_states=np.array([[0.5], [1.0], [0.2]]),
    )
    assert result.barrier_worst_margin == pytest.approx(0.2)
    assert result.barrier_violations == 0
    assert result.safe is True


def test_barrier_violations_are_counted_and_unsafe() -> None:
    result = certify_candidate_safety(
        _CANDIDATE,
        [_obs()],
        SafetyConstraintConfig(),
        barrier=_barrier(),
        replay_states=np.array([[0.5], [-0.3], [-0.1]]),
    )
    assert result.barrier_worst_margin == pytest.approx(-0.3)
    assert result.barrier_violations == 2
    assert result.safe is False


def test_barrier_requires_replay_states() -> None:
    with pytest.raises(ValueError, match="replay_states"):
        certify_candidate_safety(
            _CANDIDATE, [_obs()], SafetyConstraintConfig(), barrier=_barrier()
        )


def test_empty_replay_states_raise() -> None:
    with pytest.raises(ValueError, match="at least one state"):
        certify_candidate_safety(
            _CANDIDATE,
            [_obs()],
            SafetyConstraintConfig(),
            barrier=_barrier(),
            replay_states=np.empty((0, 1)),
        )


def test_no_barrier_leaves_margin_unset() -> None:
    result = certify_candidate_safety(_CANDIDATE, [_obs()], SafetyConstraintConfig())
    assert result.barrier_worst_margin is None
    assert result.barrier_violations == 0
    assert result.safe is True


def test_lyapunov_constraint_pass_and_fail() -> None:
    constraints = SafetyConstraintConfig(max_lyapunov_exponent=0.0)
    safe = certify_candidate_safety(
        _CANDIDATE, [_obs(lyapunov_exponent=-0.2), _obs(lyapunov_exponent=-0.1)], constraints
    )
    assert safe.lyapunov_margin == pytest.approx(0.1)
    assert safe.constraint_verdicts["lyapunov"] is True
    unsafe = certify_candidate_safety(
        _CANDIDATE, [_obs(lyapunov_exponent=0.3)], constraints
    )
    assert unsafe.lyapunov_margin == pytest.approx(-0.3)
    assert unsafe.constraint_verdicts["lyapunov"] is False
    assert unsafe.safe is False


def test_stl_constraint_uses_worst_robustness() -> None:
    constraints = SafetyConstraintConfig(min_stl_robustness=0.0)
    result = certify_candidate_safety(
        _CANDIDATE, [_obs(stl_robustness=0.4), _obs(stl_robustness=0.1)], constraints
    )
    assert result.stl_margin == pytest.approx(0.1)
    assert result.constraint_verdicts["stl"] is True


def test_safety_cost_constraint_uses_worst_cost() -> None:
    constraints = SafetyConstraintConfig(max_safety_cost=0.5)
    result = certify_candidate_safety(
        _CANDIDATE, [_obs(safety_cost=0.2), _obs(safety_cost=0.7)], constraints
    )
    assert result.safety_cost_margin == pytest.approx(-0.2)
    assert result.constraint_verdicts["safety_cost"] is False


def test_required_constraint_with_missing_evidence_fails_closed() -> None:
    constraints = SafetyConstraintConfig(
        max_lyapunov_exponent=0.0, require_lyapunov=True
    )
    result = certify_candidate_safety(
        _CANDIDATE, [_obs(lyapunov_exponent=-0.1), _obs()], constraints
    )
    assert result.lyapunov_margin is None
    assert result.constraint_verdicts["lyapunov"] is False
    assert result.safe is False


def test_optional_constraint_with_missing_evidence_passes() -> None:
    constraints = SafetyConstraintConfig(min_stl_robustness=0.0)
    result = certify_candidate_safety(_CANDIDATE, [_obs(), _obs()], constraints)
    assert result.stl_margin is None
    assert result.constraint_verdicts["stl"] is True


def test_unconfigured_constraint_passes_with_no_margin() -> None:
    result = certify_candidate_safety(_CANDIDATE, [_obs()], SafetyConstraintConfig())
    assert result.lyapunov_margin is None
    assert result.constraint_verdicts == {
        "lyapunov": True,
        "stl": True,
        "safety_cost": True,
    }


def test_forward_invariance_within_shell_is_formally_proven() -> None:
    result = certify_candidate_safety(
        _CANDIDATE,
        [_obs()],
        SafetyConstraintConfig(),
        barrier=_barrier(),
        replay_states=np.array([[0.5], [0.6]]),
        forward_invariance=_certificate(boundary_shell=0.1),
    )
    assert result.forward_invariance_verified is True
    assert result.evidence_kind == "formally-proven"


def test_forward_invariance_outside_shell_stays_measured() -> None:
    result = certify_candidate_safety(
        _CANDIDATE,
        [_obs()],
        SafetyConstraintConfig(),
        barrier=_barrier(),
        replay_states=np.array([[0.05], [0.6]]),
        forward_invariance=_certificate(boundary_shell=0.1),
    )
    assert result.evidence_kind == "measured"


def test_unverified_forward_invariance_is_measured() -> None:
    result = certify_candidate_safety(
        _CANDIDATE,
        [_obs()],
        SafetyConstraintConfig(),
        barrier=_barrier(),
        replay_states=np.array([[0.5]]),
        forward_invariance=_certificate(boundary_shell=0.0, verified=False),
    )
    assert result.forward_invariance_verified is False
    assert result.evidence_kind == "measured"


def test_no_forward_invariance_is_measured() -> None:
    result = certify_candidate_safety(_CANDIDATE, [_obs()], SafetyConstraintConfig())
    assert result.forward_invariance_verified is None
    assert result.evidence_kind == "measured"


def test_empty_observations_raise() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        certify_candidate_safety(_CANDIDATE, [], SafetyConstraintConfig())


def test_digest_is_deterministic_and_seals_the_body() -> None:
    args = (
        _CANDIDATE,
        [_obs(lyapunov_exponent=-0.1)],
        SafetyConstraintConfig(max_lyapunov_exponent=0.0),
    )
    first = certify_candidate_safety(*args)
    second = certify_candidate_safety(*args)
    assert first.digest == second.digest
    record = first.to_audit_record()
    assert record["digest"] == first.digest
    assert record["evidence_kind"] == "measured"
    assert list(record["constraint_verdicts"]) == sorted(record["constraint_verdicts"])  # type: ignore[arg-type]


def test_digest_changes_with_safety_outcome() -> None:
    safe = certify_candidate_safety(
        _CANDIDATE,
        [_obs(lyapunov_exponent=-0.1)],
        SafetyConstraintConfig(max_lyapunov_exponent=0.0),
    )
    unsafe = certify_candidate_safety(
        _CANDIDATE,
        [_obs(lyapunov_exponent=0.5)],
        SafetyConstraintConfig(max_lyapunov_exponent=0.0),
    )
    assert safe.digest != unsafe.digest
