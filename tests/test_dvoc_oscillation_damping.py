# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — closed-loop Koopman-MPC damping tests

"""Tests for the closed-loop Koopman-MPC oscillation-damping pipeline.

The suite proves the discretised plant is stable and underdamped, that the
end-to-end pipeline turns a PRC-flagged poorly-damped mode into a better-damped
one with both screening records sealed, and the full input-validation surface.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.actuation.koopman_mpc import KoopmanMPCConfig
from scpn_phase_orchestrator.runtime.dvoc_oscillation_damping import (
    OscillationDampingResult,
    _weakest_damping,
    damp_oscillation,
    underdamped_oscillator,
)

_CAPTURED_AT = "2026-06-22T00:00:00Z"


def _grid_plant() -> tuple[np.ndarray, np.ndarray]:
    return underdamped_oscillator(frequency_hz=0.5, damping_ratio=0.02, dt=0.02)


# --------------------------------------------------------------------------- #
# Plant construction                                                          #
# --------------------------------------------------------------------------- #
def test_underdamped_oscillator_is_stable_and_underdamped() -> None:
    state_matrix, input_matrix = _grid_plant()
    eigenvalues = np.linalg.eigvals(state_matrix)
    assert state_matrix.shape == (2, 2)
    assert input_matrix.shape == (2, 1)
    # Complex-conjugate poles just inside the unit circle: stable but ringing.
    assert np.all(np.abs(eigenvalues) < 1.0)
    assert np.max(np.abs(eigenvalues.imag)) > 0.0
    assert np.all(np.abs(eigenvalues) > 0.9)


@pytest.mark.parametrize(
    ("frequency_hz", "damping_ratio", "dt", "match"),
    [
        (0.0, 0.02, 0.02, "must be positive"),
        (0.5, 0.02, 0.0, "must be positive"),
        (0.5, -0.1, 0.02, "must be non-negative"),
    ],
)
def test_underdamped_oscillator_rejects_bad_parameters(
    frequency_hz: float, damping_ratio: float, dt: float, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        underdamped_oscillator(
            frequency_hz=frequency_hz, damping_ratio=damping_ratio, dt=dt
        )


# --------------------------------------------------------------------------- #
# End-to-end damping                                                          #
# --------------------------------------------------------------------------- #
def test_damp_oscillation_improves_the_weakest_damping() -> None:
    state_matrix, input_matrix = _grid_plant()
    result = damp_oscillation(
        state_matrix,
        input_matrix,
        initial_state=np.array([1.0, 0.0]),
        horizon=300,
        fs=50.0,
        captured_at=_CAPTURED_AT,
    )
    assert isinstance(result, OscillationDampingResult)
    # The open-loop ringdown is poorly damped and flagged; control improves it.
    assert result.uncontrolled_damping_ratio < 0.05
    assert result.controlled_damping_ratio > result.uncontrolled_damping_ratio
    assert result.damping_improved
    assert any(finding.flagged for finding in result.before_evidence.findings)
    assert not any(finding.flagged for finding in result.after_evidence.findings)
    # The Koopman predictor recovers the linear plant almost exactly.
    assert result.fit_residual < 1.0e-6
    # The controlled ringdown ends far closer to the origin.
    assert abs(result.controlled_signal[-1]) < abs(result.uncontrolled_signal[-1])


def test_damp_oscillation_seals_both_evidence_records() -> None:
    state_matrix, input_matrix = _grid_plant()
    result = damp_oscillation(
        state_matrix,
        input_matrix,
        initial_state=np.array([1.0, 0.0]),
        horizon=250,
        fs=50.0,
        captured_at=_CAPTURED_AT,
        event_prefix="unit-test",
    )
    assert result.before_evidence.event_id == "unit-test-open-loop"
    assert result.after_evidence.event_id == "unit-test-closed-loop"
    assert result.before_evidence.captured_at == _CAPTURED_AT


def test_damp_oscillation_exports_hash_sealed_audit_record() -> None:
    state_matrix, input_matrix = _grid_plant()
    result = damp_oscillation(
        state_matrix,
        input_matrix,
        initial_state=np.array([1.0, 0.0]),
        horizon=250,
        fs=50.0,
        captured_at=_CAPTURED_AT,
        event_prefix="audit-test",
    )

    record = result.to_audit_record()

    assert record["schema"] == "scpn_dvoc_oscillation_damping_audit_v1"
    assert record["claim_boundary"] == "review_only_offline_no_live_actuation"
    assert record["review_only"] is True
    assert record["damping_improved"] is True
    assert record["damping_delta"] == pytest.approx(
        result.controlled_damping_ratio - result.uncontrolled_damping_ratio
    )
    assert record["before_evidence_hash"] == result.before_evidence.content_hash
    assert record["after_evidence_hash"] == result.after_evidence.content_hash
    before_record = record["before"]
    after_record = record["after"]
    assert isinstance(before_record, dict)
    assert isinstance(after_record, dict)
    assert before_record["content_hash"] == result.before_evidence.content_hash
    assert after_record["content_hash"] == result.after_evidence.content_hash
    assert len(str(record["content_hash"])) == 64
    assert record == result.to_audit_record()


def test_damp_oscillation_accepts_a_custom_config() -> None:
    state_matrix, input_matrix = _grid_plant()
    config = KoopmanMPCConfig(
        horizon=10,
        output_weight=1.0,
        input_weight=1.0e-2,
        input_lower=-20.0,
        input_upper=20.0,
    )
    result = damp_oscillation(
        state_matrix,
        input_matrix,
        initial_state=np.array([1.0, 0.0]),
        horizon=250,
        fs=50.0,
        captured_at=_CAPTURED_AT,
        config=config,
    )
    assert result.damping_improved


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #
def test_damp_oscillation_rejects_a_mismatched_initial_state() -> None:
    state_matrix, input_matrix = _grid_plant()
    with pytest.raises(ValueError, match="initial_state length"):
        damp_oscillation(
            state_matrix,
            input_matrix,
            initial_state=np.array([1.0, 0.0, 0.0]),
            horizon=50,
            fs=50.0,
            captured_at=_CAPTURED_AT,
        )


def test_weakest_damping_treats_a_silent_signal_as_fully_damped() -> None:
    # A signal with no detectable oscillatory mode is reported fully damped.
    assert _weakest_damping(np.zeros(200), fs=50.0) == 1.0


def test_damp_oscillation_rejects_a_nonpositive_horizon() -> None:
    state_matrix, input_matrix = _grid_plant()
    with pytest.raises(ValueError, match="horizon must be at least 1"):
        damp_oscillation(
            state_matrix,
            input_matrix,
            initial_state=np.array([1.0, 0.0]),
            horizon=0,
            fs=50.0,
            captured_at=_CAPTURED_AT,
        )
