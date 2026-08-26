# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — real-time PMU replay tests

"""Owner tests for :mod:`bench.pmu_stream_replay`.

Synthetic fixtures only: an IEEE-format capture with a known onset and a
synthetic SEALED evidence payload (hashed with the real canonical hasher), so
the seal-verification gate, the sealed-operating-point construction, the causal
replay, the honest lead, and the sealed replay record are all exercised without
the citation-only raw data.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bench.pmu_stream_replay import (
    StreamReplayResult,
    load_sealed_evidence,
    monitor_from_e2g,
    replay,
    replay_case,
    sealed_replay_record,
)
from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash
from scpn_phase_orchestrator.monitor.grid_modal_stream import GridModalStreamMonitor

RATE_HZ = 30.0
NOMINAL_HZ = 60.0
CASE = "ISO-NE_case1"
MODE_HZ = 0.27


def _write_capture(path: Path, channels: list[tuple[str, np.ndarray]]) -> None:
    """Write an IEEE four-row-header PMU CSV with the given frequency channels."""
    times = np.arange(channels[0][1].size) / RATE_HZ
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Time", *(label for label, _ in channels)])
        writer.writerow(["T", *("F" for _ in channels)])
        writer.writerow(["sec", *("Hz" for _ in channels)])
        writer.writerow(["1900-01-01 00:00:00", *("Ln:x" for _ in channels)])
        for index, time_value in enumerate(times):
            writer.writerow(
                [
                    f"{time_value:.6f}",
                    *(f"{samples[index]:.9f}" for _, samples in channels),
                ]
            )


def _capture(path: Path, *, onset_s: float = 100.0, amplitude: float = 0.05) -> None:
    """A three-channel capture: ambient noise, then a growing oscillation."""
    rng = np.random.default_rng(11)
    n = int(180.0 * RATE_HZ)
    t = np.arange(n) / RATE_HZ
    ramp = np.clip((t - onset_s) / 20.0, 0.0, 1.0)
    channels = []
    for index in range(3):
        scale = 1.0 - 0.2 * index
        signal = (
            NOMINAL_HZ
            + 0.001 * rng.standard_normal(n)
            + amplitude * scale * ramp * np.sin(2 * np.pi * MODE_HZ * t + 0.3 * index)
        )
        channels.append((f"Sub:{index + 1}:Ln:1", signal))
    _write_capture(path, channels)


def _sealed_evidence(
    path: Path,
    *,
    threshold: float = 0.05,
    window_seconds: float = 18.5,
    tamper: bool = False,
    drop_hash: bool = False,
    drop_corpus: bool = False,
    drop_detector: bool = False,
    drop_calibration: bool = False,
    drop_significance: bool = False,
    bad_window: bool = False,
) -> Path:
    """Write a minimal evidence-shaped payload sealed with the real hasher."""
    record: dict[str, Any] = {
        "benchmark": "iso_ne_modal_growth_cross_dataset",
        "detector": {"aggregation": "focal", "recency_top": 3.0},
        "corpus": {
            "transitions": [
                {"case": CASE, "mode_hz": MODE_HZ, "window_seconds": window_seconds}
            ]
        },
        "local_calibration": {
            "threshold": threshold,
            "n_null": 6,
            "significance": {"p_value": 0.3318},
        },
    }
    if drop_corpus:
        record.pop("corpus")
    if drop_detector:
        record.pop("detector")
    if drop_calibration:
        record.pop("local_calibration")
    if drop_significance:
        record["local_calibration"].pop("significance")
    if bad_window:
        record["corpus"]["transitions"][0]["window_seconds"] = "wide"
    record["content_hash"] = canonical_record_hash(
        {key: value for key, value in record.items() if key != "content_hash"}
    )
    if tamper:
        record["local_calibration"]["threshold"] = threshold / 2.0
    if drop_hash:
        record.pop("content_hash")
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


class TestLoadSealedEvidence:
    def test_verifies_and_returns_payload(self, tmp_path: Path) -> None:
        path = _sealed_evidence(tmp_path / "evidence.json")
        payload = load_sealed_evidence(path)
        assert payload["benchmark"] == "iso_ne_modal_growth_cross_dataset"

    def test_rejects_missing_hash(self, tmp_path: Path) -> None:
        path = _sealed_evidence(tmp_path / "nohash.json", drop_hash=True)
        with pytest.raises(ValueError, match="no content_hash"):
            load_sealed_evidence(path)

    def test_rejects_tampered_record(self, tmp_path: Path) -> None:
        path = _sealed_evidence(tmp_path / "tampered.json", tamper=True)
        with pytest.raises(ValueError, match="tampered"):
            load_sealed_evidence(path)


class TestMonitorFromE2g:
    def test_carries_the_sealed_operating_point(self, tmp_path: Path) -> None:
        path = _sealed_evidence(
            tmp_path / "evidence.json", threshold=0.07, window_seconds=20.0
        )
        monitor = monitor_from_e2g(path, case_id=CASE, rate=RATE_HZ)
        assert isinstance(monitor, GridModalStreamMonitor)
        assert monitor.threshold == 0.07
        assert monitor.window_seconds == pytest.approx(20.0)
        assert monitor.step_seconds == pytest.approx(5.0)
        assert monitor.aggregation == "focal"

    def test_rejects_unknown_case(self, tmp_path: Path) -> None:
        path = _sealed_evidence(tmp_path / "evidence.json")
        with pytest.raises(ValueError, match="not in the sealed corpus"):
            monitor_from_e2g(path, case_id="ISO-NE_case9", rate=RATE_HZ)

    def test_rejects_missing_corpus(self, tmp_path: Path) -> None:
        path = _sealed_evidence(tmp_path / "evidence.json", drop_corpus=True)
        with pytest.raises(ValueError, match="corpus.transitions"):
            monitor_from_e2g(path, case_id=CASE, rate=RATE_HZ)

    def test_rejects_missing_detector_block(self, tmp_path: Path) -> None:
        path = _sealed_evidence(tmp_path / "evidence.json", drop_detector=True)
        with pytest.raises(ValueError, match="detector/local_calibration"):
            monitor_from_e2g(path, case_id=CASE, rate=RATE_HZ)

    def test_rejects_non_numeric_window(self, tmp_path: Path) -> None:
        path = _sealed_evidence(tmp_path / "evidence.json", bad_window=True)
        with pytest.raises(ValueError, match="window_seconds"):
            monitor_from_e2g(path, case_id=CASE, rate=RATE_HZ)


class TestReplay:
    def test_collects_alarms_causally(self, tmp_path: Path) -> None:
        capture = tmp_path / "capture.csv"
        _capture(capture)
        from bench.early_warning_leadtime_isone import frequency_matrix

        rate, matrix = frequency_matrix(capture)
        monitor = GridModalStreamMonitor(
            rate=rate,
            threshold=0.05,
            window_seconds=18.5,
            step_seconds=4.6,
            persistence=1,
        )
        alarms = replay(matrix, monitor=monitor)
        assert alarms, "growing oscillation must raise at least one alarm"
        assert all(a.time_s >= 18.5 for a in alarms)  # never before one window

    def test_pace_branch_sleeps_between_samples(self, tmp_path: Path) -> None:
        capture = tmp_path / "capture.csv"
        _capture(capture)
        from bench.early_warning_leadtime_isone import frequency_matrix

        rate, matrix = frequency_matrix(capture)
        monitor = GridModalStreamMonitor(
            rate=rate, threshold=1e9, window_seconds=18.5, step_seconds=4.6
        )
        alarms = replay(matrix[:, :50], monitor=monitor, pace_seconds=0.0)
        assert alarms == []

    @pytest.mark.parametrize("bad", [True, -0.1, float("nan"), "x"])
    def test_rejects_invalid_pace(self, bad: object) -> None:
        monitor = GridModalStreamMonitor(
            rate=RATE_HZ, threshold=1.0, window_seconds=2.0, step_seconds=0.5
        )
        with pytest.raises(ValueError, match="pace_seconds"):
            replay(np.zeros((2, 10)), monitor=monitor, pace_seconds=bad)  # type: ignore[arg-type]


class TestReplayCase:
    def test_alarm_leads_the_onset(self, tmp_path: Path) -> None:
        capture = tmp_path / "capture.csv"
        _capture(capture)
        evidence = _sealed_evidence(tmp_path / "evidence.json", threshold=0.05)
        result = replay_case(CASE, capture, evidence, mode_hz=MODE_HZ)
        assert result.case_id == CASE
        assert result.n_samples == int(180.0 * RATE_HZ)
        assert result.alarms
        assert result.first_alarm_seconds == result.alarms[0].time_s
        assert result.lead_seconds == pytest.approx(
            result.onset_seconds - result.alarms[0].time_s
        )
        assert result.wall_seconds >= 0.0

    def test_silent_monitor_reports_no_lead(self, tmp_path: Path) -> None:
        capture = tmp_path / "capture.csv"
        _capture(capture)
        evidence = _sealed_evidence(tmp_path / "evidence.json", threshold=1e9)
        result = replay_case(CASE, capture, evidence, mode_hz=MODE_HZ)
        assert result.alarms == ()
        assert result.first_alarm_seconds is None
        assert result.lead_seconds is None


class TestSealedReplayRecord:
    def _result(self, tmp_path: Path, *, threshold: float) -> StreamReplayResult:
        capture = tmp_path / "capture.csv"
        _capture(capture)
        evidence = _sealed_evidence(tmp_path / "evidence.json", threshold=threshold)
        return replay_case(CASE, capture, evidence, mode_hz=MODE_HZ)

    def test_seals_provenance_and_recomputes(self, tmp_path: Path) -> None:
        result = self._result(tmp_path, threshold=0.05)
        record = sealed_replay_record(
            result,
            evidence_path=tmp_path / "evidence.json",
            source_sha256="cd" * 32,
        )
        provenance = record["operating_point_provenance"]
        assert provenance["calibration_n_null"] == 6  # type: ignore[index]
        assert record["source_sha256"] == "cd" * 32
        sealed = dict(record)
        content_hash = sealed.pop("content_hash")
        assert canonical_record_hash(sealed) == content_hash

    def test_rejects_evidence_without_calibration(self, tmp_path: Path) -> None:
        result = self._result(tmp_path, threshold=0.05)
        bare = _sealed_evidence(tmp_path / "bare.json", drop_calibration=True)
        with pytest.raises(ValueError, match="no local_calibration"):
            sealed_replay_record(result, evidence_path=bare, source_sha256="cd" * 32)

    def test_rejects_evidence_without_significance(self, tmp_path: Path) -> None:
        result = self._result(tmp_path, threshold=0.05)
        bare = _sealed_evidence(tmp_path / "nosig.json", drop_significance=True)
        with pytest.raises(ValueError, match="significance"):
            sealed_replay_record(result, evidence_path=bare, source_sha256="cd" * 32)

    def test_tampering_breaks_the_seal(self, tmp_path: Path) -> None:
        result = self._result(tmp_path, threshold=0.05)
        record = sealed_replay_record(
            result,
            evidence_path=tmp_path / "evidence.json",
            source_sha256="cd" * 32,
        )
        tampered = dict(record)
        content_hash = tampered.pop("content_hash")
        tampered["lead_seconds"] = 999.0
        assert canonical_record_hash(tampered) != content_hash
