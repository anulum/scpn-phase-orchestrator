# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — NERC PRC-029 ride-through evidence tests

"""Tests for review-only NERC PRC-029 ride-through evidence screening."""

from __future__ import annotations

from typing import cast

import pytest

from scpn_phase_orchestrator.assurance.prc_ride_through import (
    AC_WIND_IBR,
    ASSESSOR_REVIEW_REQUIRED,
    DURATION_EXCEEDS_MINIMUM,
    MANDATORY_OPERATION_REGION,
    MAY_RIDE_THROUGH_ZONE,
    MAY_TRIP_ZONE_OBSERVED,
    OTHER_IBR,
    PRC_RIDE_THROUGH_DISCLAIMER,
    PRC_RIDE_THROUGH_STANDARD,
    WITHIN_MINIMUM_DURATION,
    WITHIN_REVIEW_ENVELOPE,
    PRCRideThroughEvidence,
    screen_ride_through_samples,
)

_FIXED_ARGS = {
    "event_id": "IBR-EVT-001",
    "captured_at": "2026-07-04T13:45:00Z",
    "signal_source": "IBR-17/high-side-transformer",
}


def _screen(
    time_s: tuple[float, ...],
    voltage_pu: tuple[float, ...],
    frequency_hz: tuple[float, ...],
    **overrides: object,
) -> PRCRideThroughEvidence:
    args = {**_FIXED_ARGS, **overrides}
    return screen_ride_through_samples(
        time_s,
        voltage_pu,
        frequency_hz,
        **args,  # type: ignore[arg-type]
    )


class TestRideThroughScreening:
    def test_nominal_trace_stays_inside_review_envelope(self) -> None:
        evidence = _screen(
            (0.0, 1.0, 2.0, 3.0),
            (1.0, 1.0, 0.99, 1.0),
            (60.0, 60.01, 59.99, 60.0),
        )

        assert evidence.verdict == WITHIN_REVIEW_ENVELOPE
        assert evidence.findings == ()
        assert evidence.flagged_count == 0
        assert evidence.ibr_category == OTHER_IBR
        assert evidence.standard == PRC_RIDE_THROUGH_STANDARD
        assert evidence.disclaimer == PRC_RIDE_THROUGH_DISCLAIMER
        assert evidence.sample_count == 4
        assert evidence.duration_s == pytest.approx(3.0)
        assert "PRC-029-1" in evidence.standard
        assert "certification" in evidence.disclaimer

    def test_short_other_ibr_low_voltage_is_recorded_without_flagging(self) -> None:
        evidence = _screen(
            (0.0, 0.25, 0.50),
            (0.82, 0.82, 1.0),
            (60.0, 60.0, 60.0),
        )

        assert evidence.verdict == WITHIN_REVIEW_ENVELOPE
        assert evidence.flagged_count == 0
        finding = evidence.findings[0]
        assert finding.channel == "voltage"
        assert finding.operation_region == MANDATORY_OPERATION_REGION
        assert finding.band == "other_ibr_voltage_lt_0_90_ge_0_70"
        assert finding.minimum_ride_through_s == pytest.approx(6.0)
        assert finding.window_s == pytest.approx(10.0)
        assert finding.observed_min == pytest.approx(0.82)
        assert finding.observed_max == pytest.approx(0.82)
        assert finding.duration_s == pytest.approx(0.5)
        assert finding.window_duration_s == pytest.approx(0.5)
        assert finding.classification == WITHIN_MINIMUM_DURATION
        assert finding.flagged is False

    def test_other_ibr_low_voltage_duration_beyond_table_limit_is_flagged(
        self,
    ) -> None:
        evidence = _screen(
            (0.0, 3.5, 7.0),
            (0.82, 0.82, 1.0),
            (60.0, 60.0, 60.0),
        )

        assert evidence.verdict == ASSESSOR_REVIEW_REQUIRED
        assert evidence.flagged_count == 1
        finding = evidence.findings[0]
        assert finding.channel == "voltage"
        assert finding.classification == DURATION_EXCEEDS_MINIMUM
        assert finding.minimum_ride_through_s == pytest.approx(6.0)
        assert finding.window_duration_s == pytest.approx(7.0)
        assert finding.flagged is True

    def test_frequency_duration_accumulates_over_the_prc029_window(self) -> None:
        evidence = _screen(
            (0.0, 180.0, 260.0, 340.0),
            (1.0, 1.0, 1.0, 1.0),
            (61.4, 60.0, 61.4, 60.0),
        )

        assert evidence.verdict == WITHIN_REVIEW_ENVELOPE
        finding = evidence.findings[0]
        assert finding.channel == "frequency"
        assert finding.band == "frequency_gt_61_2_le_61_8"
        assert finding.minimum_ride_through_s == pytest.approx(299.0)
        assert finding.window_s == pytest.approx(600.0)
        assert finding.window_duration_s == pytest.approx(260.0)
        assert finding.classification == WITHIN_MINIMUM_DURATION

    def test_frequency_duration_beyond_table_limit_is_flagged(self) -> None:
        evidence = _screen(
            (0.0, 220.0, 340.0, 420.0),
            (1.0, 1.0, 1.0, 1.0),
            (61.4, 60.0, 61.4, 60.0),
        )

        assert evidence.verdict == ASSESSOR_REVIEW_REQUIRED
        finding = evidence.findings[0]
        assert finding.channel == "frequency"
        assert finding.classification == DURATION_EXCEEDS_MINIMUM
        assert finding.window_duration_s == pytest.approx(300.0)
        assert finding.flagged is True

    def test_may_trip_frequency_zone_requires_review(self) -> None:
        evidence = _screen(
            (0.0, 0.5, 1.0),
            (1.0, 1.0, 1.0),
            (62.0, 62.0, 60.0),
        )

        assert evidence.verdict == ASSESSOR_REVIEW_REQUIRED
        finding = evidence.findings[0]
        assert finding.operation_region == MAY_RIDE_THROUGH_ZONE
        assert finding.classification == MAY_TRIP_ZONE_OBSERVED
        assert finding.flagged is True

    def test_ac_wind_voltage_table_uses_wind_specific_duration(self) -> None:
        evidence = _screen(
            (0.0, 2.0, 4.0),
            (0.82, 0.82, 1.0),
            (60.0, 60.0, 60.0),
            ibr_category=AC_WIND_IBR,
        )

        finding = evidence.findings[0]
        assert evidence.ibr_category == AC_WIND_IBR
        assert finding.band == "ac_wind_voltage_lt_0_90_ge_0_70"
        assert finding.minimum_ride_through_s == pytest.approx(3.0)
        assert finding.classification == DURATION_EXCEEDS_MINIMUM

    @pytest.mark.parametrize(
        ("category", "voltage", "expected_band", "minimum_s"),
        [
            (AC_WIND_IBR, 1.21, "ac_wind_voltage_gt_1_20", None),
            (AC_WIND_IBR, 1.10, "ac_wind_voltage_ge_1_10_le_1_20", 1.0),
            (AC_WIND_IBR, 1.06, "ac_wind_voltage_gt_1_05_lt_1_10", 1800.0),
            (AC_WIND_IBR, 1.00, None, None),
            (AC_WIND_IBR, 0.09, "ac_wind_voltage_lt_0_10", 0.16),
            (AC_WIND_IBR, 0.20, "ac_wind_voltage_lt_0_25_ge_0_10", 0.16),
            (AC_WIND_IBR, 0.40, "ac_wind_voltage_lt_0_50_ge_0_25", 1.20),
            (AC_WIND_IBR, 0.60, "ac_wind_voltage_lt_0_70_ge_0_50", 2.50),
            (OTHER_IBR, 1.21, "other_ibr_voltage_gt_1_20", None),
            (OTHER_IBR, 1.11, "other_ibr_voltage_gt_1_10_le_1_20", 1.0),
            (OTHER_IBR, 1.06, "other_ibr_voltage_gt_1_05_le_1_10", 1800.0),
            (OTHER_IBR, 1.00, None, None),
            (OTHER_IBR, 0.09, "other_ibr_voltage_lt_0_10", 0.32),
            (OTHER_IBR, 0.20, "other_ibr_voltage_lt_0_25_ge_0_10", 0.32),
            (OTHER_IBR, 0.40, "other_ibr_voltage_lt_0_50_ge_0_25", 1.20),
            (OTHER_IBR, 0.60, "other_ibr_voltage_lt_0_70_ge_0_50", 3.00),
        ],
    )
    def test_voltage_band_catalogue_matches_prc029_tables(
        self,
        category: str,
        voltage: float,
        expected_band: str | None,
        minimum_s: float | None,
    ) -> None:
        evidence = _screen(
            (0.0, 0.5, 1.0),
            (voltage, 1.0, 1.0),
            (60.0, 60.0, 60.0),
            ibr_category=category,
        )

        if expected_band is None:
            assert evidence.findings == ()
            return
        finding = evidence.findings[0]
        assert finding.band == expected_band
        assert finding.minimum_ride_through_s == minimum_s

    @pytest.mark.parametrize(
        ("frequency", "expected_band", "minimum_s"),
        [
            (60.0, None, None),
            (56.9, "frequency_lt_57_0", None),
            (58.0, "frequency_lt_58_8_ge_57_0", 299.0),
        ],
    )
    def test_frequency_band_catalogue_matches_prc029_table(
        self,
        frequency: float,
        expected_band: str | None,
        minimum_s: float | None,
    ) -> None:
        evidence = _screen(
            (0.0, 0.5, 1.0),
            (1.0, 1.0, 1.0),
            (frequency, 60.0, 60.0),
        )

        if expected_band is None:
            assert evidence.findings == ()
            return
        finding = evidence.findings[0]
        assert finding.band == expected_band
        assert finding.minimum_ride_through_s == minimum_s


class TestEvidenceRecord:
    def test_hash_is_deterministic_and_counts_are_read_only(self) -> None:
        first = _screen(
            (0.0, 0.25, 0.50),
            (0.82, 0.82, 1.0),
            (60.0, 60.0, 60.0),
        )
        second = _screen(
            (0.0, 0.25, 0.50),
            (0.82, 0.82, 1.0),
            (60.0, 60.0, 60.0),
        )

        assert first.content_hash == second.content_hash
        assert len(first.content_hash) == 64
        assert set(first.content_hash) <= set("0123456789abcdef")
        counts = cast(dict[str, int], first.channel_counts)
        with pytest.raises(TypeError):
            counts["voltage"] = 99

    def test_to_audit_record_round_trips_fields(self) -> None:
        evidence = _screen(
            (0.0, 3.5, 7.0),
            (0.82, 0.82, 1.0),
            (60.0, 60.0, 60.0),
        )
        record = evidence.to_audit_record()

        assert record["event_id"] == "IBR-EVT-001"
        assert record["verdict"] == ASSESSOR_REVIEW_REQUIRED
        assert record["content_hash"] == evidence.content_hash
        assert record["channel_counts"] == {"voltage": 1}
        assert isinstance(record["findings"], list)
        assert record["findings"][0]["classification"] == DURATION_EXCEEDS_MINIMUM


class TestValidation:
    def test_rejects_empty_event_id(self) -> None:
        with pytest.raises(ValueError, match="event_id"):
            _screen((0.0, 1.0), (1.0, 1.0), (60.0, 60.0), event_id="")

    def test_rejects_unknown_ibr_category(self) -> None:
        with pytest.raises(ValueError, match="ibr_category"):
            _screen(
                (0.0, 1.0),
                (1.0, 1.0),
                (60.0, 60.0),
                ibr_category="battery",
            )

    def test_rejects_mismatched_array_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            _screen((0.0, 1.0), (1.0,), (60.0, 60.0))

    def test_rejects_too_few_samples(self) -> None:
        with pytest.raises(ValueError, match="at least two samples"):
            _screen((0.0,), (1.0,), (60.0,))

    def test_rejects_non_monotonic_time(self) -> None:
        with pytest.raises(ValueError, match="time_s must be strictly increasing"):
            _screen((0.0, 1.0, 0.5), (1.0, 1.0, 1.0), (60.0, 60.0, 60.0))

    def test_rejects_numeric_string_aliases(self) -> None:
        with pytest.raises(ValueError, match="voltage_pu\\[0\\] must be a finite real"):
            screen_ride_through_samples(
                (0.0, 1.0),
                ("1.0", 1.0),
                (60.0, 60.0),
                **_FIXED_ARGS,
            )

    def test_rejects_non_finite_frequency(self) -> None:
        with pytest.raises(ValueError, match="frequency_hz\\[1\\] must be finite"):
            _screen((0.0, 1.0), (1.0, 1.0), (60.0, float("inf")))
