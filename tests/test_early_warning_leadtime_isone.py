# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — ISO-NE E2.G evaluation tests

"""Owner tests for :mod:`bench.early_warning_leadtime_isone`.

Synthetic fixtures only: an IEEE-format PMU CSV writer builds captures with a
known ambient stretch and a known growing-oscillation onset, so every branch of
the corpus preparation, onset estimation, frozen transfer, local calibration,
and payload sealing is exercised without the citation-only raw data.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from bench.early_warning_leadtime_isone import (
    CYCLES_PER_WINDOW,
    DOCUMENTED_MODES,
    EXCLUDED_CASES,
    FROZEN_PSML_THRESHOLD,
    CaseScores,
    case_scores,
    estimate_onset,
    evaluate_local_calibration,
    frequency_matrix,
    frozen_transfer_case,
    local_calibration_payload,
    split_scores,
    window_scores,
)

RATE_HZ = 30.0
NOMINAL_HZ = 60.0


def _write_capture(
    path: Path,
    channels: list[tuple[str, np.ndarray]],
    *,
    rate: float = RATE_HZ,
) -> None:
    """Write an IEEE four-row-header PMU CSV with the given frequency channels."""
    n = channels[0][1].size
    times = np.arange(n) / rate
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


def _oscillating_channels(
    *,
    n_channels: int = 3,
    duration_s: float = 180.0,
    onset_s: float = 100.0,
    mode_hz: float = 0.27,
    rate: float = RATE_HZ,
    amplitude: float = 0.05,
    noise: float = 0.001,
    seed: int = 7,
) -> list[tuple[str, np.ndarray]]:
    """Build channels: ambient noise, then a growing oscillation after onset."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * rate)
    t = np.arange(n) / rate
    ramp = np.clip((t - onset_s) / 20.0, 0.0, 1.0)
    channels = []
    for index in range(n_channels):
        scale = 1.0 - 0.2 * index
        signal = (
            NOMINAL_HZ
            + noise * rng.standard_normal(n)
            + amplitude * scale * ramp * np.sin(2 * np.pi * mode_hz * t + 0.3 * index)
        )
        channels.append((f"Sub:{index + 1}:Ln:1", signal))
    return channels


@pytest.fixture()
def capture_path(tmp_path: Path) -> Path:
    """A synthetic case-1-like capture with a known onset."""
    path = tmp_path / "case.csv"
    _write_capture(path, _oscillating_channels())
    return path


class TestFrequencyMatrix:
    def test_loads_rate_and_clean_matrix(self, capture_path: Path) -> None:
        rate, matrix = frequency_matrix(capture_path)
        assert rate == pytest.approx(RATE_HZ, rel=1e-3)
        assert matrix.shape[0] == 3

    def test_rejects_single_clean_channel(self, tmp_path: Path) -> None:
        channels = _oscillating_channels(n_channels=2)
        # Poison the second channel with a dropout so only one stays clean.
        poisoned = channels[1][1].copy()
        poisoned[5] = 0.0
        path = tmp_path / "single.csv"
        _write_capture(path, [channels[0], (channels[1][0], poisoned)])
        with pytest.raises(ValueError, match="clean channels"):
            frequency_matrix(path)

    def test_rejects_duplicate_channels(self, tmp_path: Path) -> None:
        channels = _oscillating_channels(n_channels=1)
        duplicated = [
            ("Sub:1:Ln:1", channels[0][1]),
            ("Sub:1:Ln:2", channels[0][1]),
            ("Sub:1:Ln:3", channels[0][1]),
        ]
        path = tmp_path / "dupes.csv"
        _write_capture(path, duplicated)
        with pytest.raises(ValueError, match="distinct signal"):
            frequency_matrix(path)


class TestEstimateOnset:
    def test_finds_known_onset(self, capture_path: Path) -> None:
        rate, matrix = frequency_matrix(capture_path)
        onset = estimate_onset(matrix, rate=rate, mode_hz=0.27)
        assert 95.0 <= onset <= 125.0

    def test_fails_closed_without_onset(self, tmp_path: Path) -> None:
        channels = _oscillating_channels(amplitude=0.0)
        path = tmp_path / "flat.csv"
        _write_capture(path, channels)
        rate, matrix = frequency_matrix(path)
        with pytest.raises(ValueError, match="no separable onset"):
            estimate_onset(matrix, rate=rate, mode_hz=0.27)

    @pytest.mark.parametrize(
        "name",
        ["baseline_seconds", "factor", "sustain_seconds", "smooth_seconds"],
    )
    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), True, "x"])
    def test_rejects_invalid_controls(
        self, capture_path: Path, name: str, bad: object
    ) -> None:
        rate, matrix = frequency_matrix(capture_path)
        with pytest.raises(ValueError, match=name):
            estimate_onset(matrix, rate=rate, mode_hz=0.27, **{name: bad})  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_mode", [0.0, -0.2, float("inf"), True])
    def test_rejects_invalid_mode(self, capture_path: Path, bad_mode: object) -> None:
        rate, matrix = frequency_matrix(capture_path)
        with pytest.raises(ValueError, match="mode_hz"):
            estimate_onset(matrix, rate=rate, mode_hz=bad_mode)  # type: ignore[arg-type]

    def test_rejects_band_collapse(self, capture_path: Path) -> None:
        rate, matrix = frequency_matrix(capture_path)
        # A mode so high that 0.45*rate clips the band shut at 30 Hz sampling.
        with pytest.raises(ValueError, match="collapses"):
            estimate_onset(matrix, rate=rate, mode_hz=40.0)

    @pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), True, "x"])
    def test_rejects_invalid_search_start(
        self, capture_path: Path, bad: object
    ) -> None:
        rate, matrix = frequency_matrix(capture_path)
        with pytest.raises(ValueError, match="search_start_seconds"):
            estimate_onset(
                matrix,
                rate=rate,
                mode_hz=0.27,
                search_start_seconds=bad,  # type: ignore[arg-type]
            )

    def test_search_start_floors_the_estimate(self, capture_path: Path) -> None:
        rate, matrix = frequency_matrix(capture_path)
        unpinned = estimate_onset(matrix, rate=rate, mode_hz=0.27)
        pinned = estimate_onset(
            matrix, rate=rate, mode_hz=0.27, search_start_seconds=unpinned + 5.0
        )
        assert pinned >= unpinned + 5.0 - 1.0 / rate

    def test_search_start_beyond_capture_fails_closed(self, capture_path: Path) -> None:
        rate, matrix = frequency_matrix(capture_path)
        with pytest.raises(ValueError, match="no separable onset"):
            estimate_onset(matrix, rate=rate, mode_hz=0.27, search_start_seconds=1.0e6)


class TestWindowScores:
    def test_scores_and_end_times(self, capture_path: Path) -> None:
        rate, matrix = frequency_matrix(capture_path)
        ends, scores = window_scores(
            matrix, rate=rate, window_seconds=10.0, step_seconds=5.0
        )
        assert ends.shape == scores.shape
        assert ends[0] == pytest.approx(10.0, rel=1e-2)
        assert np.all(np.isfinite(scores))

    @pytest.mark.parametrize("name", ["window_seconds", "step_seconds"])
    @pytest.mark.parametrize("bad", [0.0, -2.0, float("nan"), True])
    def test_rejects_invalid_windows(
        self, capture_path: Path, name: str, bad: object
    ) -> None:
        rate, matrix = frequency_matrix(capture_path)
        kwargs = {"window_seconds": 10.0, "step_seconds": 5.0, name: bad}
        with pytest.raises(ValueError, match=name):
            window_scores(matrix, rate=rate, **kwargs)  # type: ignore[arg-type]

    def test_rejects_capture_shorter_than_window(self, capture_path: Path) -> None:
        rate, matrix = frequency_matrix(capture_path)
        with pytest.raises(ValueError, match="cannot hold"):
            window_scores(matrix, rate=rate, window_seconds=1e5, step_seconds=1.0)


class TestSplitScores:
    def test_partitions_with_guard(self) -> None:
        ends = np.array([10.0, 30.0, 49.0, 55.0, 90.0, 99.0, 101.0])
        scores = np.arange(7.0)
        nulls, transitions = split_scores(
            ends,
            scores,
            onset_seconds=100.0,
            window_seconds=10.0,
            transition_seconds=60.0,
        )
        # Null: end <= 100-60-10 = 30; transition: 40 < end <= 100.
        assert nulls.tolist() == [0.0, 1.0]
        assert transitions.tolist() == [2.0, 3.0, 4.0, 5.0]

    def test_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="identical shape"):
            split_scores(
                np.array([1.0, 2.0]),
                np.array([1.0]),
                onset_seconds=10.0,
                window_seconds=1.0,
            )


class TestCaseScores:
    def test_prepares_known_case(self, capture_path: Path) -> None:
        prepared = case_scores("ISO-NE_case1", capture_path)
        assert prepared.case_id == "ISO-NE_case1"
        assert prepared.mode_hz == DOCUMENTED_MODES["ISO-NE_case1"]
        assert prepared.window_seconds == pytest.approx(
            CYCLES_PER_WINDOW / prepared.mode_hz
        )
        assert prepared.n_channels == 3
        assert prepared.null_scores.size > 0
        assert prepared.transition_scores.size > 0
        assert prepared.transition_ends.size == prepared.transition_scores.size

    def test_rejects_unknown_case(self, capture_path: Path) -> None:
        with pytest.raises(ValueError, match="case_id"):
            case_scores("ISO-NE_case9", capture_path)


class TestFrozenTransfer:
    def test_counts_crossings(self, capture_path: Path) -> None:
        outcome = frozen_transfer_case("ISO-NE_case1", capture_path)
        assert outcome.case_id == "ISO-NE_case1"
        assert outcome.n_null > 0
        assert outcome.n_transition > 0
        assert 0 <= outcome.null_crossings <= outcome.n_null
        assert 0 <= outcome.transition_crossings <= outcome.n_transition

    def test_rejects_unknown_case(self, capture_path: Path) -> None:
        with pytest.raises(ValueError, match="case_id"):
            frozen_transfer_case("nope", capture_path)


def _prepared_corpus(tmp_path: Path) -> list[CaseScores]:
    """Three synthetic cases mirroring the fixed corpus structure."""
    prepared = []
    for case_id, (mode_hz, onset_s, duration_s) in {
        "ISO-NE_case1": (0.27, 100.0, 180.0),
        "ISO-NE_case2": (0.15, 200.0, 360.0),
        "ISO-NE_case3": (1.13, 100.0, 180.0),
    }.items():
        path = tmp_path / f"{case_id}.csv"
        _write_capture(
            path,
            _oscillating_channels(
                duration_s=duration_s, onset_s=onset_s, mode_hz=mode_hz
            ),
        )
        prepared.append(case_scores(case_id, path))
    return prepared


class TestLocalCalibration:
    def test_detects_synthetic_growth_at_matched_fa(self, tmp_path: Path) -> None:
        prepared = _prepared_corpus(tmp_path)
        result = evaluate_local_calibration(prepared)
        assert result.n_null >= 5
        assert result.achieved_false_alarm <= result.target_false_alarm + 1e-9
        assert len(result.led) == 3
        assert len(result.lead_seconds) == 3
        for detected, lead in zip(result.led, result.lead_seconds, strict=True):
            if detected:
                assert np.isfinite(lead)
            else:
                assert np.isnan(lead)
        assert result.significance.n_transitions == 3

    def test_rejects_empty_corpus(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            evaluate_local_calibration([])

    def test_rejects_starved_null_calibration(self, tmp_path: Path) -> None:
        prepared = _prepared_corpus(tmp_path)
        starved = [
            CaseScores(
                case_id=case.case_id,
                mode_hz=case.mode_hz,
                rate_hz=case.rate_hz,
                n_channels=case.n_channels,
                onset_seconds=case.onset_seconds,
                window_seconds=case.window_seconds,
                null_scores=case.null_scores[:1],
                transition_scores=case.transition_scores,
                transition_ends=case.transition_ends,
            )
            for case in prepared
        ]
        with pytest.raises(ValueError, match="refusing to calibrate"):
            evaluate_local_calibration(starved)


class TestPayload:
    def test_seals_and_is_deterministic(self, tmp_path: Path) -> None:
        prepared = _prepared_corpus(tmp_path)
        frozen = []
        for case in prepared:
            frozen.append(
                frozen_transfer_case(case.case_id, tmp_path / f"{case.case_id}.csv")
            )
        result = evaluate_local_calibration(prepared)
        digests = {case.case_id: "ab" * 32 for case in prepared}
        payload = local_calibration_payload(
            prepared, frozen, result, source_digests=digests
        )
        assert payload["program"] == "E2.G"
        assert payload["corpus"]["excluded"] == EXCLUDED_CASES  # type: ignore[index]
        assert payload["frozen_transfer"][0]["threshold"] == FROZEN_PSML_THRESHOLD  # type: ignore[index]
        again = local_calibration_payload(
            prepared, frozen, result, source_digests=digests
        )
        assert payload["content_hash"] == again["content_hash"]

    def test_rejects_misaligned_inputs(self, tmp_path: Path) -> None:
        prepared = _prepared_corpus(tmp_path)
        frozen = [
            frozen_transfer_case(
                prepared[0].case_id, tmp_path / f"{prepared[0].case_id}.csv"
            )
        ]
        result = evaluate_local_calibration(prepared)
        with pytest.raises(ValueError, match="one-to-one"):
            local_calibration_payload(prepared, frozen, result, source_digests={})
