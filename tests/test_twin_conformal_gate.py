# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Conformal twin-confidence gate tests

"""Tests for the adaptive conformal twin-confidence admission gate.

Covers configuration/decision contracts, calibration, the conformal threshold and
adaptive-conformal-inference update, regime conditioning and resolution, the
running coverage guarantee, and the nonconformity helper.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor.twin_confidence import TwinConfidenceScore
from scpn_phase_orchestrator.monitor.twin_conformal_gate import (
    ConformalDecision,
    ConformalGateConfig,
    TwinConformalGate,
    _conformal_threshold,
    confidence_nonconformity,
)

# ---------------------------------------------------------------------
# Config + decision contracts
# ---------------------------------------------------------------------


def test_config_defaults_and_audit() -> None:
    config = ConformalGateConfig()
    assert config.target_miscoverage == pytest.approx(0.1)
    record = config.to_audit_record()
    assert record["regime_conditioned"] is True
    assert json.loads(json.dumps(record)) == record


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5, float("nan"), True])
def test_config_rejects_bad_alpha(alpha: object) -> None:
    with pytest.raises(ValueError, match="target_miscoverage"):
        ConformalGateConfig(target_miscoverage=alpha)  # type: ignore[arg-type]


@pytest.mark.parametrize("gamma", [0.0, -0.5, 1.5, float("inf")])
def test_config_rejects_bad_gamma(gamma: float) -> None:
    with pytest.raises(ValueError, match="adaptation_rate"):
        ConformalGateConfig(adaptation_rate=gamma)


def test_decision_audit_serialises_infinite_threshold_as_none() -> None:
    decision = ConformalDecision(
        admitted=True,
        nonconformity_score=0.0,
        threshold=float("inf"),
        effective_miscoverage=0.1,
        empirical_coverage=1.0,
        regime="default",
        tick=1,
        decision_hash="h",
    )
    record = decision.to_audit_record()
    assert record["threshold"] is None
    assert json.loads(json.dumps(record)) == record


# ---------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------


def test_calibrate_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one nominal score"):
        TwinConformalGate().calibrate([])


def test_calibrate_rejects_non_finite() -> None:
    with pytest.raises(ValueError, match="nominal score"):
        TwinConformalGate().calibrate([0.1, float("inf")])


def test_calibrate_rejects_blank_regime() -> None:
    with pytest.raises(ValueError, match="regime"):
        TwinConformalGate().calibrate([0.1], regime="  ")


# ---------------------------------------------------------------------
# Conformal threshold helper
# ---------------------------------------------------------------------


def test_threshold_small_calibration_is_infinite() -> None:
    # n=3, level=0.9 -> rank = ceil(0.9*4)=4 > n -> inf (admit all).
    assert math.isinf(_conformal_threshold(np.array([0.1, 0.2, 0.3]), 0.1))


def test_threshold_normal_quantile() -> None:
    scores = np.arange(1.0, 101.0)  # 1..100
    # level=0.9, n=100 -> rank=ceil(0.9*101)=ceil(90.9)=91 -> scores[90]=91.0
    assert _conformal_threshold(scores, 0.1) == pytest.approx(91.0)


def test_threshold_alpha_one_returns_minimum() -> None:
    # alpha_t=1 -> level=0 -> rank=ceil(0)=0 -> return smallest score.
    assert _conformal_threshold(np.array([2.0, 5.0, 9.0]), 1.0) == pytest.approx(2.0)


# ---------------------------------------------------------------------
# Update / ACI / coverage
# ---------------------------------------------------------------------


def test_update_rejects_non_finite_score() -> None:
    gate = TwinConformalGate()
    gate.calibrate([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="nonconformity_score"):
        gate.update(float("nan"))


def test_update_without_calibration_raises() -> None:
    with pytest.raises(ValueError, match="no calibrated regime"):
        TwinConformalGate().update(0.5)


def test_update_admits_within_band_and_flags_outside() -> None:
    gate = TwinConformalGate(ConformalGateConfig(target_miscoverage=0.1))
    gate.calibrate(list(np.arange(1.0, 101.0)))  # threshold ~91 at start
    inside = gate.update(10.0)
    assert inside.admitted
    assert inside.tick == 1
    assert inside.threshold == pytest.approx(91.0)
    outside = gate.update(95.0)
    assert not outside.admitted
    assert len(outside.decision_hash) == 64


def test_aci_loosens_after_miscoverage() -> None:
    gate = TwinConformalGate(
        ConformalGateConfig(target_miscoverage=0.1, adaptation_rate=0.1)
    )
    gate.calibrate(list(np.arange(1.0, 101.0)))
    first = gate.update(200.0)  # miscovered -> alpha decreases next round
    assert not first.admitted
    second = gate.update(0.0)  # covered -> alpha increases
    # effective miscoverage used on the second tick is below the target (loosened).
    assert second.effective_miscoverage < 0.1


def test_empirical_coverage_tracks_target_on_nominal_stream() -> None:
    rng = np.random.default_rng(1)
    gate = TwinConformalGate(
        ConformalGateConfig(target_miscoverage=0.1, adaptation_rate=0.05)
    )
    gate.calibrate(np.abs(rng.normal(0.0, 1.0, 600)).tolist())
    admitted = 0
    n = 3000
    for _ in range(n):
        if gate.update(abs(rng.normal(0.0, 1.0))).admitted:
            admitted += 1
    assert 0.85 <= admitted / n <= 0.95
    assert gate.empirical_coverage() == pytest.approx(admitted / n)


def test_empirical_coverage_zero_before_any_tick() -> None:
    gate = TwinConformalGate()
    gate.calibrate([0.1, 0.2, 0.3])
    assert gate.empirical_coverage() == 0.0


# ---------------------------------------------------------------------
# Regime conditioning / resolution
# ---------------------------------------------------------------------


def test_regime_conditioned_uses_regime_band() -> None:
    gate = TwinConformalGate()
    gate.calibrate(list(np.arange(0.0, 1.0, 0.01)), regime="sync")  # ~max 0.99
    gate.calibrate(list(np.arange(0.0, 10.0, 0.1)), regime="chaotic")  # ~max 9.9
    assert not gate.update(2.0, regime="sync").admitted
    assert gate.update(2.0, regime="chaotic").admitted


def test_regime_falls_back_to_default_when_uncalibrated() -> None:
    gate = TwinConformalGate()
    gate.calibrate(list(np.arange(1.0, 101.0)))  # default
    decision = gate.update(5.0, regime="never_calibrated")
    assert decision.regime == "default"


def test_regime_none_uses_default() -> None:
    gate = TwinConformalGate()
    gate.calibrate(list(np.arange(1.0, 101.0)))
    assert gate.update(5.0, regime=None).regime == "default"


def test_regime_blank_string_rejected() -> None:
    gate = TwinConformalGate()
    gate.calibrate([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="regime"):
        gate.update(0.5, regime="   ")


def test_unconditioned_gate_uses_named_regime_without_default() -> None:
    gate = TwinConformalGate(ConformalGateConfig(regime_conditioned=False))
    gate.calibrate(list(np.arange(1.0, 101.0)), regime="named")
    decision = gate.update(5.0, regime="named")
    assert decision.regime == "named"


def test_no_applicable_regime_raises() -> None:
    gate = TwinConformalGate()
    gate.calibrate([0.1, 0.2, 0.3], regime="sync")
    with pytest.raises(ValueError, match="no calibrated regime"):
        gate.update(0.5, regime="chaotic")


def test_gate_audit_record() -> None:
    gate = TwinConformalGate()
    gate.calibrate(list(np.arange(1.0, 101.0)), regime="sync")
    gate.update(5.0, regime="sync")
    record = gate.to_audit_record()
    assert record["ticks"] == 1
    assert "sync" in record["regimes"]  # type: ignore[operator]
    assert json.loads(json.dumps(record)) == record


# ---------------------------------------------------------------------
# Nonconformity helper
# ---------------------------------------------------------------------


def test_confidence_nonconformity_uses_composite_z() -> None:
    score = TwinConfidenceScore(
        confidence=0.5,
        status="warning",
        phase_js_divergence=0.1,
        order_wasserstein=0.2,
        phase_js_z=1.0,
        order_w1_z=2.0,
        composite_z=2.236,
        phase_js_within_band=True,
        order_w1_within_band=False,
        backend="python",
        score_hash="x",
    )
    assert confidence_nonconformity(score) == pytest.approx(2.236)


def test_composes_with_twin_confidence_end_to_end() -> None:
    # Calibrate the gate from nominal twin-confidence composite-z, then flag a
    # divergent observation tick — the full twin-confidence → gate pipeline.
    from scpn_phase_orchestrator.monitor.twin_confidence import (
        TwinConfidenceCalibrator,
        phase_order_divergence,
        score_twin_confidence,
    )

    rng = np.random.default_rng(11)
    two_pi = 2.0 * np.pi

    def nominal_tick() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        a = rng.uniform(0.0, two_pi, 128)
        b = a + rng.normal(0.0, 0.05, 128)
        ra = rng.uniform(0.45, 0.55, 32)
        rb = np.clip(ra + rng.normal(0.0, 0.01, 32), 0.0, 1.0)
        return a, b, ra, rb

    calibrator = TwinConfidenceCalibrator()
    for _ in range(60):
        calibrator.observe(phase_order_divergence(*nominal_tick()))
    baseline = calibrator.baseline()

    gate = TwinConformalGate(ConformalGateConfig(target_miscoverage=0.1))
    gate.calibrate(
        [
            score_twin_confidence(
                phase_order_divergence(*nominal_tick()), baseline
            ).composite_z
            for _ in range(120)
        ]
    )

    nominal_score = score_twin_confidence(
        phase_order_divergence(*nominal_tick()), baseline
    )
    assert gate.update(confidence_nonconformity(nominal_score)).admitted

    divergent = score_twin_confidence(
        phase_order_divergence(
            rng.uniform(0.0, two_pi, 128),
            np.full(128, 0.1),
            np.full(32, 0.5),
            np.full(32, 0.05),
        ),
        baseline,
    )
    assert not gate.update(confidence_nonconformity(divergent)).admitted


def test_decision_hash_is_deterministic() -> None:
    gate1 = TwinConformalGate(ConformalGateConfig(adaptation_rate=0.05))
    gate2 = TwinConformalGate(ConformalGateConfig(adaptation_rate=0.05))
    cal = list(np.arange(1.0, 51.0))
    gate1.calibrate(cal)
    gate2.calibrate(cal)
    assert gate1.update(10.0).decision_hash == gate2.update(10.0).decision_hash
