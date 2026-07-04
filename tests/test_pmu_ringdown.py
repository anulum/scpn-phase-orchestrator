# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — PMU ringdown evidence ingestion tests

"""Tests for review-only PMU ringdown ingestion and PRC evidence sealing."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor.oscillation_modes import (
    APERIODIC_MODE,
    INTER_AREA_MODE,
)
from scpn_phase_orchestrator.runtime.pmu_ringdown import (
    PMU_RINGDOWN_AUDIT_SCHEMA,
    PMU_RINGDOWN_CLAIM_BOUNDARY,
    PMURingdownEvidence,
    screen_pmu_ringdown_csv,
)

_CAPTURED_AT = "2026-07-04T10:00:00Z"


def _pmu_csv(path: Path, *, samples: int = 500, fs: float = 50.0) -> Path:
    """Write a deterministic inter-area PMU frequency ringdown fixture."""
    time_s = np.arange(samples, dtype=np.float64) / fs
    frequency_hz = 60.0 + 0.08 * np.exp(-0.06 * time_s) * np.cos(
        2.0 * np.pi * 0.5 * time_s
    )
    lines = ["time_s,frequency_hz"]
    lines.extend(
        f"{timestamp:.6f},{frequency:.12f}"
        for timestamp, frequency in zip(time_s, frequency_hz, strict=True)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_frequency_csv(
    path: Path, time_s: np.ndarray, frequency_hz: np.ndarray
) -> Path:
    """Write a two-column PMU CSV from explicit time and frequency arrays."""
    lines = ["time_s,frequency_hz"]
    lines.extend(
        f"{timestamp:.6f},{frequency:.9f}"
        for timestamp, frequency in zip(time_s, frequency_hz, strict=True)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_screen_pmu_ringdown_csv_hash_seals_prc_evidence(tmp_path: Path) -> None:
    """A PMU CSV is converted into deterministic PRC screening evidence."""
    csv_path = _pmu_csv(tmp_path / "pmu-ringdown.csv")

    evidence = screen_pmu_ringdown_csv(
        csv_path,
        event_id="BUS-42-EVT-7",
        captured_at=_CAPTURED_AT,
        signal_source="PMU/BUS-42/frequency",
    )

    assert isinstance(evidence, PMURingdownEvidence)
    assert evidence.schema == PMU_RINGDOWN_AUDIT_SCHEMA
    assert evidence.claim_boundary == PMU_RINGDOWN_CLAIM_BOUNDARY
    assert evidence.review_only is True
    assert evidence.event_id == "BUS-42-EVT-7"
    assert evidence.sample_count == 500
    assert evidence.sampling_rate_hz == pytest.approx(50.0)
    assert evidence.duration_s == pytest.approx(9.98)
    assert evidence.nominal_frequency_hz == pytest.approx(60.0)
    assert evidence.source_sha256 == hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert evidence.prc_evidence.event_id == "BUS-42-EVT-7"
    assert evidence.prc_evidence.signal_source == "PMU/BUS-42/frequency/deviation"
    assert evidence.prc_evidence.mode_family_counts == {INTER_AREA_MODE: 1}
    assert evidence.prc_evidence.flagged_count == 1

    record = evidence.to_audit_record()
    assert record["schema"] == PMU_RINGDOWN_AUDIT_SCHEMA
    assert record["claim_boundary"] == PMU_RINGDOWN_CLAIM_BOUNDARY
    assert record["review_only"] is True
    assert record["source_sha256"] == evidence.source_sha256
    assert record["prc_evidence_hash"] == evidence.prc_evidence.content_hash
    assert len(str(record["content_hash"])) == 64
    assert record == evidence.to_audit_record()


@pytest.mark.parametrize(
    ("contents", "match"),
    [
        ("timestamp,frequency_hz\n0.0,60.0\n", "missing required column time_s"),
        (
            "time_s,frequency_hz\n0.0,60.0\n0.02,not-a-number\n",
            "frequency_hz row 2 must be a finite real",
        ),
        (
            "time_s,frequency_hz\n0.0,60.0\n0.02\n0.04,60.2\n0.06,60.3\n",
            "frequency_hz row 2 must be a finite real",
        ),
        (
            "time_s,frequency_hz\n0.0,60.0\n0.02,inf\n0.04,60.2\n0.06,60.3\n",
            "frequency_hz row 2 must be finite",
        ),
        (
            "time_s,frequency_hz\n0.0,60.0\n0.02,60.1\n0.01,60.2\n0.04,60.3\n",
            "time_s must be strictly increasing",
        ),
        (
            "time_s,frequency_hz\n0.0,60.0\n0.02,60.1\n0.05,60.2\n0.09,60.3\n",
            "time_s must be uniformly sampled",
        ),
    ],
)
def test_screen_pmu_ringdown_csv_rejects_invalid_operator_data(
    tmp_path: Path, contents: str, match: str
) -> None:
    """Malformed operator CSV data fails before evidence is published."""
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        screen_pmu_ringdown_csv(
            csv_path,
            event_id="EVT",
            captured_at=_CAPTURED_AT,
            signal_source="PMU/BUS-42/frequency",
            min_samples=4,
        )


def test_screen_pmu_ringdown_csv_rejects_too_few_samples(tmp_path: Path) -> None:
    """Short PMU captures are rejected before matrix-pencil fitting."""
    csv_path = tmp_path / "short.csv"
    csv_path.write_text(
        "time_s,frequency_hz\n0.00,60.0\n0.02,60.1\n0.04,60.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="at least 8 samples"):
        screen_pmu_ringdown_csv(
            csv_path,
            event_id="EVT",
            captured_at=_CAPTURED_AT,
            signal_source="PMU/BUS-42/frequency",
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"event_id": ""}, "event_id must be a non-empty string"),
        ({"min_samples": True}, "min_samples must be an integer"),
        ({"min_samples": 3}, "min_samples must be at least 4"),
        ({"nominal_frequency_hz": 0.0}, "nominal_frequency_hz must be positive"),
        (
            {"sampling_jitter_tolerance": -1.0e-6},
            "sampling_jitter_tolerance must be non-negative",
        ),
        ({"nominal_frequency_hz": True}, "nominal_frequency_hz must be a finite real"),
        ({"nominal_frequency_hz": float("inf")}, "nominal_frequency_hz must be finite"),
    ],
)
def test_screen_pmu_ringdown_csv_rejects_invalid_controls(
    tmp_path: Path, kwargs: dict[str, object], match: str
) -> None:
    """Invalid operator-supplied screening controls fail closed."""
    csv_path = _pmu_csv(tmp_path / "pmu-ringdown.csv")
    args: dict[str, object] = {
        "path": csv_path,
        "event_id": "EVT",
        "captured_at": _CAPTURED_AT,
        "signal_source": "PMU/BUS-42/frequency",
    }
    args.update(kwargs)

    with pytest.raises(ValueError, match=match):
        screen_pmu_ringdown_csv(**args)  # type: ignore[arg-type]


def test_mean_detrend_recovers_oscillation_under_operating_point_offset(
    tmp_path: Path,
) -> None:
    """Mean detrending surfaces the swing mode buried by a DC frequency offset.

    A real PMU frequency channel sits at a small offset from the nominal
    frequency (the operating point is not exactly 60 Hz). Without detrending
    the matrix pencil fits that offset as a dominant 0 Hz mode and the
    electromechanical oscillation is not the leading finding; mean detrending
    (the default) removes the offset so the swing mode leads. Fail-on-bug:
    ``detrend="none"`` leaves an aperiodic mode on top.
    """
    time_s = np.arange(300, dtype=np.float64) / 10.0
    # 0.27 Hz sustained swing riding on a 0.4 Hz operating-point offset.
    frequency_hz = 60.0 + 0.4 + 0.03 * np.cos(2.0 * np.pi * 0.27 * time_s)
    csv_path = _write_frequency_csv(tmp_path / "offset.csv", time_s, frequency_hz)

    default = screen_pmu_ringdown_csv(
        csv_path,
        event_id="EVT",
        captured_at=_CAPTURED_AT,
        signal_source="PMU/BUS-9/frequency",
    )
    assert default.detrend == "mean"
    leading = default.prc_evidence.findings[0]
    assert leading.mode_family == INTER_AREA_MODE
    assert leading.frequency_hz == pytest.approx(0.27, abs=0.02)

    undetrended = screen_pmu_ringdown_csv(
        csv_path,
        event_id="EVT",
        captured_at=_CAPTURED_AT,
        signal_source="PMU/BUS-9/frequency",
        detrend="none",
    )
    assert undetrended.prc_evidence.findings[0].mode_family == APERIODIC_MODE


def test_analysis_rate_decimates_before_estimation(tmp_path: Path) -> None:
    """Setting an analysis rate block-mean decimates an over-sampled capture."""
    time_s = np.arange(600, dtype=np.float64) / 60.0
    frequency_hz = 60.0 + 0.05 * np.cos(2.0 * np.pi * 0.5 * time_s)
    csv_path = _write_frequency_csv(tmp_path / "fast.csv", time_s, frequency_hz)

    evidence = screen_pmu_ringdown_csv(
        csv_path,
        event_id="EVT",
        captured_at=_CAPTURED_AT,
        signal_source="PMU/BUS-9/frequency",
        analysis_rate_hz=10.0,
    )

    # Raw provenance is preserved; the estimator saw the decimated signal.
    assert evidence.sample_count == 600
    assert evidence.sampling_rate_hz == pytest.approx(60.0)
    assert evidence.analysis_sample_count == 100
    assert evidence.analysis_rate_hz == pytest.approx(10.0)
    assert INTER_AREA_MODE in evidence.prc_evidence.mode_family_counts


def test_analysis_rate_near_capture_rate_skips_decimation(tmp_path: Path) -> None:
    """An analysis rate that rounds to no reduction leaves the signal untouched."""
    csv_path = _pmu_csv(tmp_path / "near.csv", samples=500, fs=50.0)

    evidence = screen_pmu_ringdown_csv(
        csv_path,
        event_id="EVT",
        captured_at=_CAPTURED_AT,
        signal_source="PMU/BUS-9/frequency",
        analysis_rate_hz=40.0,
    )

    # 50 / 40 rounds to a factor of one: no decimation applied.
    assert evidence.analysis_sample_count == 500
    assert evidence.analysis_rate_hz == pytest.approx(50.0)


def test_over_decimation_of_short_capture_is_a_no_op(tmp_path: Path) -> None:
    """A decimation factor exceeding the sample count leaves the signal intact."""
    time_s = np.arange(8, dtype=np.float64) / 50.0
    frequency_hz = 60.0 + 0.05 * np.cos(2.0 * np.pi * 0.5 * time_s)
    csv_path = _write_frequency_csv(tmp_path / "tiny.csv", time_s, frequency_hz)

    evidence = screen_pmu_ringdown_csv(
        csv_path,
        event_id="EVT",
        captured_at=_CAPTURED_AT,
        signal_source="PMU/BUS-9/frequency",
        analysis_rate_hz=1.0,
    )

    # 50 / 1 is a factor of 50 but only 8 samples exist: no block is formed.
    assert evidence.analysis_sample_count == 8
    assert evidence.analysis_rate_hz == pytest.approx(50.0)


def test_over_long_capture_fails_closed_without_decimation(tmp_path: Path) -> None:
    """A capture above the analysis ceiling fails closed with guidance."""
    csv_path = _pmu_csv(tmp_path / "long.csv", samples=500, fs=50.0)

    with pytest.raises(ValueError, match="above the 100 limit; set analysis_rate_hz"):
        screen_pmu_ringdown_csv(
            csv_path,
            event_id="EVT",
            captured_at=_CAPTURED_AT,
            signal_source="PMU/BUS-9/frequency",
            max_analysis_samples=100,
        )

    # The same capture succeeds once decimation brings it under the ceiling.
    evidence = screen_pmu_ringdown_csv(
        csv_path,
        event_id="EVT",
        captured_at=_CAPTURED_AT,
        signal_source="PMU/BUS-9/frequency",
        max_analysis_samples=100,
        analysis_rate_hz=10.0,
    )
    assert evidence.analysis_sample_count == 100


def test_grid_fit_uniformity_accepts_decimal_rounded_timestamps(
    tmp_path: Path,
) -> None:
    """Decimal-rounded operator timestamps pass the best-fit uniformity check.

    Fail-on-bug: the previous ``1e-6`` tolerance rejected every real capture
    whose timestamps are rounded to a few decimals; the realistic default
    accepts them while the tight value still fails.
    """
    time_s = np.round(np.arange(150, dtype=np.float64) / 30.0, 3)
    frequency_hz = 60.0 + 0.03 * np.cos(2.0 * np.pi * 0.27 * time_s)
    csv_path = _write_frequency_csv(tmp_path / "rounded.csv", time_s, frequency_hz)

    evidence = screen_pmu_ringdown_csv(
        csv_path,
        event_id="EVT",
        captured_at=_CAPTURED_AT,
        signal_source="PMU/BUS-9/frequency",
    )
    assert evidence.sample_count == 150

    with pytest.raises(ValueError, match="time_s must be uniformly sampled"):
        screen_pmu_ringdown_csv(
            csv_path,
            event_id="EVT",
            captured_at=_CAPTURED_AT,
            signal_source="PMU/BUS-9/frequency",
            sampling_jitter_tolerance=1.0e-6,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"detrend": "linear"}, "detrend must be one of"),
        ({"detrend": "bogus"}, "detrend must be one of"),
        ({"max_analysis_samples": True}, "max_analysis_samples must be an integer"),
        ({"max_analysis_samples": 3}, "max_analysis_samples must be at least 4"),
        ({"analysis_rate_hz": 0.0}, "analysis_rate_hz must be positive"),
        ({"analysis_rate_hz": float("nan")}, "analysis_rate_hz must be finite"),
    ],
)
def test_screen_pmu_ringdown_csv_rejects_invalid_preprocessing_controls(
    tmp_path: Path, kwargs: dict[str, object], match: str
) -> None:
    """Invalid detrend, ceiling, and analysis-rate controls fail closed."""
    csv_path = _pmu_csv(tmp_path / "pmu-ringdown.csv")
    args: dict[str, object] = {
        "path": csv_path,
        "event_id": "EVT",
        "captured_at": _CAPTURED_AT,
        "signal_source": "PMU/BUS-42/frequency",
    }
    args.update(kwargs)

    with pytest.raises(ValueError, match=match):
        screen_pmu_ringdown_csv(**args)  # type: ignore[arg-type]
