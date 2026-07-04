# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — NERC PRC oscillation compliance evidence tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.assurance.prc_oscillation import (
    ACCEPTABLE,
    FLAGGED_FOR_REVIEW,
    NO_EXCEEDANCE,
    POORLY_DAMPED,
    PRC_OSCILLATION_DISCLAIMER,
    PRC_OSCILLATION_STANDARD,
    UNDAMPED,
    PRCOscillationEvidence,
    screen_oscillation_modes,
)
from scpn_phase_orchestrator.monitor.oscillation_modes import (
    OscillationMode,
    estimate_oscillation_modes,
)

_FIXED_ARGS = {
    "event_id": "EVT-001",
    "captured_at": "2026-06-21T06:30:00Z",
    "signal_source": "BUS-42-freq",
    "sampling_rate_hz": 50.0,
}


def _mode(frequency: float, damping: float, amplitude: float = 1.0) -> OscillationMode:
    return OscillationMode(
        frequency_hz=frequency,
        damping_ratio=damping,
        amplitude=amplitude,
        phase_rad=0.0,
        poorly_damped=damping < 0.03,
    )


def _screen(
    modes: tuple[OscillationMode, ...], **overrides: object
) -> PRCOscillationEvidence:
    args = {**_FIXED_ARGS, **overrides}
    return screen_oscillation_modes(modes, **args)  # type: ignore[arg-type]


class TestClassification:
    def test_undamped_when_damping_at_or_below_zero(self) -> None:
        finding = _screen((_mode(0.5, -0.02),)).findings[0]

        assert finding.classification == UNDAMPED
        assert finding.flagged is True

    def test_marginal_zero_damping_is_undamped(self) -> None:
        finding = _screen((_mode(0.5, 0.0),)).findings[0]

        assert finding.classification == UNDAMPED
        assert finding.flagged is True

    def test_poorly_damped_between_thresholds(self) -> None:
        finding = _screen((_mode(0.3, 0.015),)).findings[0]

        assert finding.classification == POORLY_DAMPED
        assert finding.flagged is True

    def test_acceptable_at_or_above_threshold(self) -> None:
        finding = _screen((_mode(1.2, 0.12),)).findings[0]

        assert finding.classification == ACCEPTABLE
        assert finding.flagged is False

    def test_boundary_at_poorly_damped_threshold_is_acceptable(self) -> None:
        finding = _screen((_mode(0.5, 0.03),)).findings[0]

        assert finding.classification == ACCEPTABLE

    def test_custom_thresholds_reclassify(self) -> None:
        finding = _screen(
            (_mode(0.5, 0.005),), undamped_threshold=0.01, poorly_damped_threshold=0.05
        ).findings[0]

        assert finding.classification == UNDAMPED


class TestScreening:
    def test_flagged_verdict_when_any_mode_breaches(self) -> None:
        evidence = _screen((_mode(0.3, 0.015), _mode(1.2, 0.12)))

        assert evidence.verdict == FLAGGED_FOR_REVIEW
        assert evidence.flagged_count == 1
        assert evidence.worst_damping_ratio == pytest.approx(0.015)

    def test_clean_verdict_when_all_acceptable(self) -> None:
        evidence = _screen((_mode(0.3, 0.10), _mode(1.2, 0.12)))

        assert evidence.verdict == NO_EXCEEDANCE
        assert evidence.flagged_count == 0
        assert evidence.worst_damping_ratio == pytest.approx(0.10)

    def test_findings_preserve_mode_order_and_fields(self) -> None:
        modes = (_mode(0.3, 0.015, 2.0), _mode(1.2, 0.12, 0.6))
        evidence = _screen(modes)

        assert [f.mode_index for f in evidence.findings] == [0, 1]
        assert evidence.findings[0].frequency_hz == pytest.approx(0.3)
        assert evidence.findings[0].amplitude == pytest.approx(2.0)
        assert evidence.findings[1].damping_ratio == pytest.approx(0.12)

    def test_standard_and_disclaimer_attached(self) -> None:
        evidence = _screen((_mode(0.3, 0.015),))

        assert evidence.standard == PRC_OSCILLATION_STANDARD
        assert evidence.disclaimer == PRC_OSCILLATION_DISCLAIMER
        assert evidence.poorly_damped_threshold == pytest.approx(0.03)
        assert evidence.undamped_threshold == pytest.approx(0.0)
        assert "PRC-028-1" in PRC_OSCILLATION_STANDARD
        assert "PRC-030-1" in PRC_OSCILLATION_STANDARD
        assert "proposed" not in PRC_OSCILLATION_STANDARD.lower()
        assert "proposed" not in PRC_OSCILLATION_DISCLAIMER.lower()


class TestEmptyModes:
    def test_no_modes_is_compliant_with_no_findings(self) -> None:
        evidence = _screen(())

        assert evidence.verdict == NO_EXCEEDANCE
        assert evidence.findings == ()
        assert evidence.flagged_count == 0
        assert evidence.worst_damping_ratio is None


class TestEvidenceRecord:
    def test_content_hash_is_lowercase_hex_sha256(self) -> None:
        evidence = _screen((_mode(0.3, 0.015),))

        assert len(evidence.content_hash) == 64
        assert set(evidence.content_hash) <= set("0123456789abcdef")

    def test_hash_is_deterministic_for_identical_inputs(self) -> None:
        first = _screen((_mode(0.3, 0.015),))
        second = _screen((_mode(0.3, 0.015),))

        assert first.content_hash == second.content_hash

    def test_hash_changes_with_content(self) -> None:
        base = _screen((_mode(0.3, 0.015),))
        other_event = _screen((_mode(0.3, 0.015),), event_id="EVT-002")
        other_mode = _screen((_mode(0.3, 0.016),))

        assert base.content_hash != other_event.content_hash
        assert base.content_hash != other_mode.content_hash

    def test_to_audit_record_round_trips_fields(self) -> None:
        evidence = _screen((_mode(0.3, 0.015),))
        record = evidence.to_audit_record()

        assert record["event_id"] == "EVT-001"
        assert record["verdict"] == FLAGGED_FOR_REVIEW
        assert record["content_hash"] == evidence.content_hash
        assert record["standard"] == PRC_OSCILLATION_STANDARD
        assert isinstance(record["findings"], list)
        assert record["findings"][0]["classification"] == POORLY_DAMPED

    def test_mode_finding_audit_record(self) -> None:
        finding = _screen((_mode(0.3, 0.015, 2.0),)).findings[0]
        record = finding.to_audit_record()

        assert record == {
            "mode_index": 0,
            "frequency_hz": pytest.approx(0.3),
            "damping_ratio": pytest.approx(0.015),
            "amplitude": pytest.approx(2.0),
            "classification": POORLY_DAMPED,
            "flagged": True,
        }


class TestValidation:
    def test_rejects_empty_event_id(self) -> None:
        with pytest.raises(ValueError, match="event_id"):
            _screen((_mode(0.3, 0.015),), event_id="  ")

    def test_rejects_non_string_event_id(self) -> None:
        with pytest.raises(ValueError, match="event_id"):
            _screen((_mode(0.3, 0.015),), event_id=123)

    def test_rejects_empty_captured_at(self) -> None:
        with pytest.raises(ValueError, match="captured_at"):
            _screen((_mode(0.3, 0.015),), captured_at="")

    def test_rejects_empty_signal_source(self) -> None:
        with pytest.raises(ValueError, match="signal_source"):
            _screen((_mode(0.3, 0.015),), signal_source="")

    def test_rejects_non_positive_sampling_rate(self) -> None:
        with pytest.raises(ValueError, match="sampling_rate_hz must be positive"):
            _screen((_mode(0.3, 0.015),), sampling_rate_hz=0.0)

    def test_rejects_non_real_sampling_rate(self) -> None:
        with pytest.raises(ValueError, match="finite real"):
            _screen((_mode(0.3, 0.015),), sampling_rate_hz="fast")

    def test_rejects_boolean_threshold(self) -> None:
        with pytest.raises(ValueError, match="finite real"):
            _screen((_mode(0.3, 0.015),), undamped_threshold=True)

    def test_rejects_non_finite_threshold(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            _screen((_mode(0.3, 0.015),), poorly_damped_threshold=float("inf"))

    def test_rejects_unordered_thresholds(self) -> None:
        with pytest.raises(ValueError, match="must be below"):
            _screen(
                (_mode(0.3, 0.015),),
                undamped_threshold=0.05,
                poorly_damped_threshold=0.03,
            )

    def test_rejects_non_mode_element(self) -> None:
        with pytest.raises(ValueError, match="must be an OscillationMode"):
            _screen((object(),))  # type: ignore[arg-type]


class TestPipelineWiring:
    def test_screens_modes_estimated_from_a_ringdown(self) -> None:
        fs = 50.0
        t = np.arange(0.0, 10.0, 1.0 / fs)

        def damped(freq: float, zeta: float, amp: float) -> np.ndarray:
            wn = 2.0 * np.pi * freq / np.sqrt(1.0 - zeta**2)
            return amp * np.exp(-zeta * wn * t) * np.cos(2.0 * np.pi * freq * t)

        signal = damped(0.3, 0.015, 1.0) + damped(1.2, 0.12, 0.6)
        modes = estimate_oscillation_modes(signal, fs, model_order=8)
        evidence = screen_oscillation_modes(
            modes,
            event_id="RINGDOWN-7",
            captured_at="2026-06-21T06:45:00Z",
            signal_source="order-parameter",
            sampling_rate_hz=fs,
        )

        assert evidence.verdict == FLAGGED_FOR_REVIEW
        flagged = [f for f in evidence.findings if f.flagged]
        assert any(f.frequency_hz == pytest.approx(0.3, abs=0.05) for f in flagged)
        assert evidence.worst_damping_ratio is not None
        assert evidence.worst_damping_ratio < 0.03
