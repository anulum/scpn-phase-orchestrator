# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — split-conformal alarm-stream tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor.conformal_alarm import (
    ConformalAlarmConfig,
    ConformalAlarmDecision,
    ConformalAlarmStream,
)


class TestConfig:
    def test_defaults(self):
        config = ConformalAlarmConfig()
        assert config.target_false_alarm == 0.1
        assert config.adaptation_rate == 0.0
        assert config.regime_conditioned is False

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
    def test_target_false_alarm_out_of_range_rejected(self, alpha):
        with pytest.raises(ValueError, match="target_false_alarm must be in"):
            ConformalAlarmConfig(target_false_alarm=alpha)

    def test_non_finite_target_rejected(self):
        with pytest.raises(ValueError, match="target_false_alarm must be finite"):
            ConformalAlarmConfig(target_false_alarm=float("nan"))

    def test_boolean_target_rejected(self):
        with pytest.raises(ValueError, match="target_false_alarm must be a real"):
            ConformalAlarmConfig(target_false_alarm=True)

    def test_negative_adaptation_rejected(self):
        with pytest.raises(ValueError, match="adaptation_rate must be non-negative"):
            ConformalAlarmConfig(adaptation_rate=-0.01)

    def test_audit_record(self):
        record = ConformalAlarmConfig(
            target_false_alarm=0.2, adaptation_rate=0.05, regime_conditioned=True
        ).to_audit_record()
        assert record == {
            "target_false_alarm": 0.2,
            "adaptation_rate": 0.05,
            "regime_conditioned": True,
        }


class TestDecisionRecord:
    def test_finite_threshold_record(self):
        decision = ConformalAlarmDecision(
            alarm=True,
            score=2.0,
            threshold=1.5,
            effective_false_alarm=0.1,
            empirical_false_alarm=0.2,
            regime="default",
            nominal_ticks=5,
        )
        assert decision.to_audit_record()["threshold"] == 1.5

    def test_infinite_threshold_serialised_as_string(self):
        decision = ConformalAlarmDecision(
            alarm=False,
            score=2.0,
            threshold=float("inf"),
            effective_false_alarm=0.1,
            empirical_false_alarm=0.0,
            regime="default",
            nominal_ticks=1,
        )
        assert decision.to_audit_record()["threshold"] == "inf"


class TestCalibration:
    def test_empty_calibration_rejected(self):
        with pytest.raises(ValueError, match="at least one nominal score"):
            ConformalAlarmStream().calibrate([])

    def test_non_finite_calibration_rejected(self):
        with pytest.raises(ValueError, match="nominal score must be finite"):
            ConformalAlarmStream().calibrate([0.1, float("inf")])

    def test_empty_regime_rejected(self):
        with pytest.raises(ValueError, match="regime must be a non-empty string"):
            ConformalAlarmStream().calibrate([0.1], regime="  ")


class TestAlarmDecision:
    def _stream(self, **config):
        stream = ConformalAlarmStream(ConformalAlarmConfig(**config))
        stream.calibrate([float(x) for x in range(10)])  # 0..9
        return stream

    def test_alarm_fires_above_threshold(self):
        # target 0.1 over 10 -> rank ceil(0.9*11)=10 -> 10th smallest = 9.
        stream = self._stream(target_false_alarm=0.1)
        decision = stream.update(50.0, is_nominal=True)
        assert decision.alarm
        assert decision.threshold == 9.0

    def test_no_alarm_at_or_below_threshold(self):
        stream = self._stream(target_false_alarm=0.1)
        decision = stream.update(9.0, is_nominal=True)
        assert not decision.alarm

    def test_non_finite_score_rejected(self):
        stream = self._stream()
        with pytest.raises(ValueError, match="score must be finite"):
            stream.update(float("nan"), is_nominal=True)

    def test_nominal_tick_updates_counters(self):
        stream = self._stream(target_false_alarm=0.1)
        stream.update(50.0, is_nominal=True)  # alarm
        stream.update(1.0, is_nominal=True)  # no alarm
        assert stream.empirical_false_alarm() == pytest.approx(0.5)

    def test_event_tick_does_not_update_false_alarm(self):
        stream = self._stream(target_false_alarm=0.1)
        stream.update(50.0, is_nominal=False)  # a detection, not a false alarm
        assert stream.empirical_false_alarm() == 0.0
        assert stream.update(0.0, is_nominal=True).nominal_ticks == 1

    def test_unlabelled_tick_scored_without_calibration_touch(self):
        stream = self._stream(target_false_alarm=0.1)
        decision = stream.update(50.0)  # is_nominal=None
        assert decision.alarm
        assert stream.empirical_false_alarm() == 0.0

    def test_empirical_false_alarm_zero_before_any_nominal_tick(self):
        assert self._stream().empirical_false_alarm() == 0.0

    def test_infinite_threshold_when_calibration_too_small(self):
        stream = ConformalAlarmStream(ConformalAlarmConfig(target_false_alarm=0.1))
        stream.calibrate([0.5])  # rank ceil(0.9*2)=2 > 1 -> +inf
        decision = stream.update(1e9, is_nominal=True)
        assert not np.isfinite(decision.threshold)
        assert not decision.alarm

    def test_stream_audit_record(self):
        stream = self._stream(target_false_alarm=0.1)
        stream.update(50.0, is_nominal=True)
        record = stream.to_audit_record()
        assert record["config"]["target_false_alarm"] == 0.1
        regime = record["regimes"]["default"]
        assert regime["calibration_size"] == 10
        assert regime["nominal_ticks"] == 1
        assert regime["empirical_false_alarm"] == 1.0


class TestRegimeResolution:
    def test_uncalibrated_stream_rejects_update(self):
        with pytest.raises(ValueError, match="no calibrated regime is applicable"):
            ConformalAlarmStream().update(1.0)

    def test_conditioned_named_regime_used(self):
        stream = ConformalAlarmStream(
            ConformalAlarmConfig(target_false_alarm=0.1, regime_conditioned=True)
        )
        stream.calibrate([float(x) for x in range(10)], regime="chimera")
        assert (
            stream.update(50.0, regime="chimera", is_nominal=True).regime == "chimera"
        )

    def test_conditioned_falls_back_to_default_when_regime_uncalibrated(self):
        stream = ConformalAlarmStream(
            ConformalAlarmConfig(target_false_alarm=0.1, regime_conditioned=True)
        )
        stream.calibrate([float(x) for x in range(10)])  # default
        assert stream.update(1.0, regime="unknown", is_nominal=True).regime == "default"

    def test_conditioned_empty_regime_rejected(self):
        stream = ConformalAlarmStream(
            ConformalAlarmConfig(target_false_alarm=0.1, regime_conditioned=True)
        )
        stream.calibrate([float(x) for x in range(10)])
        with pytest.raises(ValueError, match="regime must be a non-empty string"):
            stream.update(1.0, regime="   ")

    def test_conditioned_none_regime_uses_default(self):
        stream = ConformalAlarmStream(
            ConformalAlarmConfig(target_false_alarm=0.1, regime_conditioned=True)
        )
        stream.calibrate([float(x) for x in range(10)])
        assert stream.update(1.0, is_nominal=True).regime == "default"

    def test_unconditioned_named_regime_matched(self):
        stream = ConformalAlarmStream(ConformalAlarmConfig(target_false_alarm=0.1))
        stream.calibrate([float(x) for x in range(10)], regime="named")
        assert stream.update(1.0, regime="named", is_nominal=True).regime == "named"

    def test_empirical_false_alarm_named_regime(self):
        stream = ConformalAlarmStream(
            ConformalAlarmConfig(target_false_alarm=0.1, regime_conditioned=True)
        )
        stream.calibrate([float(x) for x in range(10)], regime="r1")
        stream.update(50.0, regime="r1", is_nominal=True)
        assert stream.empirical_false_alarm(regime="r1") == pytest.approx(1.0)


class TestConformalCoverageGuarantee:
    def test_marginal_false_alarm_tracks_target_over_calibration_draws(self):
        # Split-conformal coverage is marginal over calibration draws, so validate
        # the mean nominal false-alarm rate across many independent draws rather
        # than a single fixed calibration.
        rng = np.random.default_rng(0)
        target = 0.1
        rates = []
        for _ in range(80):
            stream = ConformalAlarmStream(
                ConformalAlarmConfig(target_false_alarm=target)
            )
            stream.calibrate(rng.standard_normal(200).tolist())
            for score in rng.standard_normal(500):
                stream.update(float(score), is_nominal=True)
            rates.append(stream.empirical_false_alarm())
        mean_rate = float(np.mean(rates))
        assert 0.08 <= mean_rate <= 0.12

    def test_aci_tracks_target_under_drift_where_fixed_inflates(self):
        rng = np.random.default_rng(11)
        calibration = rng.standard_normal(200).tolist()
        drift = (rng.standard_normal(4000) + np.linspace(0.0, 1.5, 4000)).tolist()

        adaptive = ConformalAlarmStream(
            ConformalAlarmConfig(target_false_alarm=0.1, adaptation_rate=0.02)
        )
        adaptive.calibrate(calibration)
        fixed = ConformalAlarmStream(ConformalAlarmConfig(target_false_alarm=0.1))
        fixed.calibrate(calibration)
        for score in drift:
            adaptive.update(score, is_nominal=True)
            fixed.update(score, is_nominal=True)

        # Adaptive tracks the target; the fixed threshold inflates under drift.
        assert adaptive.empirical_false_alarm() < 0.2
        assert fixed.empirical_false_alarm() > 0.25
