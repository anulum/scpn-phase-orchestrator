# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — WECC 240-bus E2.G evaluation tests

"""Owner tests for :mod:`bench.early_warning_leadtime_wecc`.

Synthetic fixtures only: a TSAT-format export writer builds captures with a
known ambient stretch and a known forced oscillation from the pre-registered
forcing start, so every branch of the parser, segmentation, onset handling,
frozen transfer, and payload sealing is exercised without the citation-only
raw contest data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bench.early_warning_leadtime_isone import evaluate_local_calibration
from bench.early_warning_leadtime_wecc import (
    BASELINE_SECONDS,
    DOCUMENTED_MODES,
    FORCING_SECONDS,
    NULL_END_SECONDS,
    CaseDetection,
    _estimate_case_onset,
    case_records,
    case_scores,
    evaluate_detection,
    frozen_transfer_case,
    local_calibration_payload,
    split_scores,
    voltage_matrix,
)
from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash

RATE_HZ = 30.0


def _write_export(
    path: Path,
    channels: list[tuple[str, np.ndarray]],
    *,
    rate: float = RATE_HZ,
) -> None:
    """Write a TSAT-style quoted-header export with the given channels."""
    n = channels[0][1].size
    times = np.arange(n) / rate
    header = " ".join(f"'{label}'" for label in ("Time", *(c for c, _ in channels)))
    rows = [header]
    for index, time_value in enumerate(times):
        rows.append(
            " ".join(
                [
                    f"{time_value:.6f}",
                    *(f"{samples[index]:.9f}" for _, samples in channels),
                ]
            )
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _forced_oscillation_channels(
    *,
    n_channels: int = 3,
    duration_s: float = 90.0,
    forcing_s: float = FORCING_SECONDS,
    mode_hz: float = 0.82,
    rate: float = RATE_HZ,
    ambient_amplitude: float = 0.006,
    start_amplitude: float = 0.002,
    growth_time_s: float = 2.0,
    cap_amplitude: float = 0.5,
    noise: float = 0.002,
    seed: int = 11,
) -> list[tuple[str, np.ndarray]]:
    """Build channels: steady in-band ambient, then exponential forced growth.

    The constant in-band component keeps the onset estimator's 3x-baseline
    crossing meaningfully after the forcing start (as a real system's ambient
    mode content does), so the pre-registered early-warning region is
    non-empty and the growth is strong enough inside it to lead.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * rate)
    times = np.arange(n) / rate
    envelope = np.where(
        times < forcing_s,
        0.0,
        np.minimum(
            start_amplitude * np.exp((times - forcing_s) / growth_time_s),
            cap_amplitude,
        ),
    )
    channels: list[tuple[str, np.ndarray]] = []
    for index in range(n_channels):
        base = 1.0 + 0.001 * index
        ambient = ambient_amplitude * np.sin(
            2.0 * np.pi * mode_hz * times + 1.1 * index + 0.5
        )
        wave = envelope * np.sin(2.0 * np.pi * mode_hz * times + 0.3 * index)
        samples = base + ambient + wave + rng.normal(0.0, noise, size=n)
        channels.append((f"{4000 + index}|BUS {index}", samples))
    return channels


def _ambient_channels(
    *,
    n_channels: int = 3,
    duration_s: float = 90.0,
    mode_hz: float = 1.19,
    rate: float = RATE_HZ,
    ambient_amplitude: float = 0.006,
    noise: float = 0.002,
    seed: int = 23,
) -> list[tuple[str, np.ndarray]]:
    """Build steady-ambient channels with no growing oscillation anywhere."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * rate)
    times = np.arange(n) / rate
    channels: list[tuple[str, np.ndarray]] = []
    for index in range(n_channels):
        ambient = ambient_amplitude * np.sin(
            2.0 * np.pi * mode_hz * times + 1.1 * index
        )
        samples = 1.0 + ambient + rng.normal(0.0, noise, size=n)
        channels.append((f"{5000 + index}|AMB {index}", samples))
    return channels


@pytest.fixture(scope="module")
def forced_export(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A resolvable forced-oscillation export at the WECC_case1 mode."""
    path = tmp_path_factory.mktemp("wecc") / "BusVolMag.txt"
    _write_export(path, _forced_oscillation_channels())
    return path


@pytest.fixture(scope="module")
def ambient_export(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A pure-ambient export with no in-band onset anywhere."""
    path = tmp_path_factory.mktemp("wecc_ambient") / "BusVolMag.txt"
    _write_export(path, _ambient_channels())
    return path


class TestVoltageMatrix:
    """Parser validations and the happy path."""

    def test_happy_path_rate_and_shape(self, forced_export: Path) -> None:
        rate, matrix = voltage_matrix(forced_export)
        assert rate == pytest.approx(RATE_HZ, rel=1e-3)
        assert matrix.shape == (3, int(90.0 * RATE_HZ))

    def test_too_few_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "short.txt"
        path.write_text("'Time' 'a|b'\n0.0 1.0\n", encoding="utf-8")
        with pytest.raises(ValueError, match="header row and >=2 sample rows"):
            voltage_matrix(path)

    def test_header_without_quotes(self, tmp_path: Path) -> None:
        path = tmp_path / "noquotes.txt"
        path.write_text("Time a b\n0.0 1.0 1.0\n0.1 1.0 1.0\n", encoding="utf-8")
        with pytest.raises(ValueError, match="first header column must be 'Time'"):
            voltage_matrix(path)

    def test_header_first_column_not_time(self, tmp_path: Path) -> None:
        path = tmp_path / "nottime.txt"
        path.write_text(
            "'Tick' 'a|b' 'c|d'\n0.0 1.0 1.0\n0.1 1.0 1.0\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="first header column must be 'Time'"):
            voltage_matrix(path)

    def test_malformed_numeric_block(self, tmp_path: Path) -> None:
        path = tmp_path / "malformed.txt"
        path.write_text(
            "'Time' 'a|b' 'c|d'\n0.0 1.0 1.0\n0.1 oops 1.0\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="malformed numeric block"):
            voltage_matrix(path)

    def test_single_column_block_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "onecol.txt"
        path.write_text("'Time'\n0.0\n0.1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="disagrees with"):
            voltage_matrix(path)

    def test_column_count_mismatch(self, tmp_path: Path) -> None:
        path = tmp_path / "mismatch.txt"
        path.write_text(
            "'Time' 'a|b' 'c|d'\n0.0 1.0 1.0 9.0\n0.1 1.0 1.0 9.0\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="disagrees with"):
            voltage_matrix(path)

    def test_backward_time_axis_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "backwards.txt"
        path.write_text(
            "'Time' 'a|b' 'c|d'\n0.0 1.0 2.0\n0.2 1.1 2.1\n0.1 1.2 2.2\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="strictly increasing"):
            voltage_matrix(path)

    def test_duplicate_timestamp_keeps_post_event_row(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(9)
        live_a = 1.0 + rng.normal(0.0, 0.001, size=5)
        live_b = 2.0 + rng.normal(0.0, 0.001, size=5)
        rows = ["'Time' 'a|b' 'c|d'"]
        times = [0.0, 0.1, 0.1, 0.2, 0.3]
        for index, time_value in enumerate(times):
            rows.append(f"{time_value:.6f} {live_a[index]:.9f} {live_b[index]:.9f}")
        path = tmp_path / "discontinuity.txt"
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        rate, matrix = voltage_matrix(path)
        assert matrix.shape == (2, 4)
        # The pre-event row (index 1) is dropped; the post-event row survives.
        assert matrix[0, 1] == pytest.approx(live_a[2], abs=1e-9)
        assert matrix[1, 1] == pytest.approx(live_b[2], abs=1e-9)

    def test_too_few_clean_channels(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(3)
        live = 1.0 + rng.normal(0.0, 0.001, size=8)
        flat = np.full(8, 1.0)
        broken = live.copy()
        broken[3] = np.nan
        path = tmp_path / "dirty.txt"
        _write_export(path, [("1|L", live), ("2|F", flat), ("3|N", broken)])
        with pytest.raises(ValueError, match="clean channels"):
            voltage_matrix(path)

    def test_duplicate_channels_collapse(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(5)
        live = 1.0 + rng.normal(0.0, 0.001, size=8)
        path = tmp_path / "dupes.txt"
        _write_export(path, [("1|A", live), ("2|B", live.copy())])
        with pytest.raises(ValueError, match="distinct"):
            voltage_matrix(path)


class TestSplitScores:
    """Pre-registered segmentation arithmetic and validations."""

    def test_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="identical shape"):
            split_scores(np.array([1.0, 2.0]), np.array([1.0]), onset_seconds=40.0)

    @pytest.mark.parametrize("onset", [True, "40", float("nan"), 0.0, -1.0])
    def test_invalid_onset_rejected(self, onset: object) -> None:
        ends = np.array([10.0, 20.0])
        with pytest.raises(ValueError, match="onset_seconds"):
            split_scores(ends, ends, onset_seconds=onset)  # type: ignore[arg-type]

    def test_segmentation_masks(self) -> None:
        ends = np.array([20.0, 25.0, 26.0, 30.0, 31.0, 40.0, 41.0])
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        nulls, transitions = split_scores(ends, scores, onset_seconds=40.0)
        assert nulls.tolist() == [1.0, 2.0]
        assert transitions.tolist() == [5.0, 6.0]
        assert NULL_END_SECONDS == 25.0
        assert FORCING_SECONDS == 30.0


class TestOnsetHandling:
    """Resolved, unresolved-allowed, and unresolved-strict onset paths."""

    def test_resolved_onset_after_forcing(self, forced_export: Path) -> None:
        prepared = case_scores("WECC_case1", forced_export)
        assert prepared.onset_seconds > FORCING_SECONDS
        assert prepared.transition_scores.size > 0
        assert prepared.null_scores.size > 0
        assert prepared.window_seconds == pytest.approx(
            5.0 / DOCUMENTED_MODES["WECC_case1"]
        )

    def test_unresolved_onset_strict_raises(self, ambient_export: Path) -> None:
        with pytest.raises(ValueError, match="no separable onset"):
            case_scores("WECC_case2", ambient_export)

    def test_unresolved_onset_allowed_reports_empty_transition(
        self, ambient_export: Path
    ) -> None:
        prepared = case_scores(
            "WECC_case2", ambient_export, allow_unresolved_onset=True
        )
        assert np.isnan(prepared.onset_seconds)
        assert prepared.transition_scores.size == 0
        assert prepared.transition_ends.size == 0
        assert prepared.null_scores.size > 0

    def test_other_estimation_error_propagates_despite_allow(self) -> None:
        matrix = np.vstack([np.ones(64), np.ones(64) * 2.0])
        with pytest.raises(ValueError, match="positive finite number"):
            _estimate_case_onset(
                matrix, rate=30.0, mode_hz=-1.0, allow_unresolved_onset=True
            )

    def test_baseline_constant_is_ambient_only(self) -> None:
        assert BASELINE_SECONDS == 25.0


class TestCaseScores:
    """Corpus-membership validation."""

    def test_unknown_case_id(self, forced_export: Path) -> None:
        with pytest.raises(ValueError, match="case_id must be one of"):
            case_scores("WECC_case99", forced_export)

    def test_corpus_is_the_thirteen_contest_cases(self) -> None:
        assert len(DOCUMENTED_MODES) == 13
        assert set(DOCUMENTED_MODES) == {f"WECC_case{n}" for n in range(1, 14)}


class TestFrozenTransfer:
    """G-a branch counts and validations."""

    def test_unknown_case_id(self, forced_export: Path) -> None:
        with pytest.raises(ValueError, match="case_id must be one of"):
            frozen_transfer_case("WECC_case0", forced_export)

    def test_happy_counts(self, forced_export: Path) -> None:
        outcome = frozen_transfer_case("WECC_case1", forced_export)
        assert outcome.case_id == "WECC_case1"
        assert outcome.n_null > 0
        assert outcome.n_transition > 0
        assert 0 <= outcome.null_crossings <= outcome.n_null
        assert 0 <= outcome.transition_crossings <= outcome.n_transition

    def test_unresolved_onset_allowed_empty_transition(
        self, ambient_export: Path
    ) -> None:
        outcome = frozen_transfer_case(
            "WECC_case2", ambient_export, allow_unresolved_onset=True
        )
        assert outcome.n_transition == 0
        assert outcome.transition_crossings == 0
        assert outcome.n_null > 0


class TestDetectionBranch:
    """Secondary detection branch (amendment A.16.1)."""

    def test_case_records_detection_windows(self, forced_export: Path) -> None:
        prepared, detection = case_records("WECC_case1", forced_export)
        assert detection.case_id == prepared.case_id
        assert detection.detection_scores.shape == detection.detection_ends.shape
        assert detection.detection_scores.size > 0
        assert bool(np.all(detection.detection_ends > FORCING_SECONDS))

    def test_alignment_mismatch(self, forced_export: Path) -> None:
        prepared, detection = case_records("WECC_case1", forced_export)
        with pytest.raises(ValueError, match="align one-to-one"):
            evaluate_detection([prepared], [], threshold=0.1)
        renamed = CaseDetection(
            case_id="WECC_case2",
            detection_scores=detection.detection_scores,
            detection_ends=detection.detection_ends,
        )
        with pytest.raises(ValueError, match="align one-to-one"):
            evaluate_detection([prepared], [renamed], threshold=0.1)

    @pytest.mark.parametrize("threshold", [True, "0.1", float("nan")])
    def test_invalid_threshold(self, forced_export: Path, threshold: object) -> None:
        prepared, detection = case_records("WECC_case1", forced_export)
        with pytest.raises(ValueError, match="threshold"):
            evaluate_detection(
                [prepared],
                [detection],
                threshold=threshold,  # type: ignore[arg-type]
            )

    def test_forced_case_detected_with_latency(self, forced_export: Path) -> None:
        prepared, detection = case_records("WECC_case1", forced_export)
        result = evaluate_local_calibration([prepared])
        outcome = evaluate_detection(
            [prepared], [detection], threshold=result.threshold
        )
        assert outcome.detected == (True,)
        assert outcome.latency_seconds[0] > 0.0
        assert outcome.threshold == pytest.approx(result.threshold)
        assert outcome.significance.n_transitions == 1

    def test_never_alarming_case_reports_nan_latency(
        self, ambient_export: Path
    ) -> None:
        prepared, detection = case_records(
            "WECC_case2", ambient_export, allow_unresolved_onset=True
        )
        high = float(np.max(detection.detection_scores)) + 1.0
        outcome = evaluate_detection([prepared], [detection], threshold=high)
        assert outcome.detected == (False,)
        assert np.isnan(outcome.latency_seconds[0])


class TestPayload:
    """Sealed payload assembly, alignment, and JSON safety."""

    @pytest.fixture(scope="class")
    def corpus(
        self, forced_export: Path, ambient_export: Path
    ) -> tuple[list, list, list, object, object]:
        records = [
            case_records("WECC_case1", forced_export),
            case_records("WECC_case2", ambient_export, allow_unresolved_onset=True),
        ]
        cases = [record[0] for record in records]
        detections = [record[1] for record in records]
        frozen = [
            frozen_transfer_case("WECC_case1", forced_export),
            frozen_transfer_case(
                "WECC_case2", ambient_export, allow_unresolved_onset=True
            ),
        ]
        result = evaluate_local_calibration(cases)
        detection_result = evaluate_detection(
            cases, detections, threshold=result.threshold
        )
        return cases, detections, frozen, result, detection_result

    def test_alignment_length_mismatch(self, corpus: tuple) -> None:
        cases, detections, frozen, result, detection_result = corpus
        with pytest.raises(ValueError, match="align one-to-one"):
            local_calibration_payload(
                cases,
                frozen[:1],
                result,
                detections=detections,
                detection_result=detection_result,
                source_digests={},
            )

    def test_alignment_order_mismatch(self, corpus: tuple) -> None:
        cases, detections, frozen, result, detection_result = corpus
        with pytest.raises(ValueError, match="align one-to-one"):
            local_calibration_payload(
                cases,
                list(reversed(frozen)),
                result,
                detections=detections,
                detection_result=detection_result,
                source_digests={},
            )

    def test_detection_alignment_mismatch(self, corpus: tuple) -> None:
        cases, detections, frozen, result, detection_result = corpus
        with pytest.raises(ValueError, match="align one-to-one"):
            local_calibration_payload(
                cases,
                frozen,
                result,
                detections=list(reversed(detections)),
                detection_result=detection_result,
                source_digests={},
            )

    def test_sealed_payload_content(self, corpus: tuple) -> None:
        cases, detections, frozen, result, detection_result = corpus
        digests = {"WECC_case1": "ab" * 32, "WECC_case2": "cd" * 32}
        payload = local_calibration_payload(
            cases,
            frozen,
            result,
            detections=detections,
            detection_result=detection_result,
            source_digests=digests,
        )
        unsealed = {
            key: value for key, value in payload.items() if key != "content_hash"
        }
        assert payload["content_hash"] == canonical_record_hash(unsealed)
        transitions = payload["corpus"]["transitions"]
        assert transitions[0]["onset_resolved"] is True
        assert transitions[0]["onset_seconds"] > FORCING_SECONDS
        assert transitions[0]["source_sha256"] == "ab" * 32
        assert transitions[0]["n_detection_windows"] > 0
        assert transitions[1]["onset_resolved"] is False
        assert transitions[1]["onset_seconds"] is None
        calibration = payload["local_calibration"]
        assert calibration["led"] == [True, False]
        assert calibration["lead_seconds"][0] > 0.0
        assert calibration["lead_seconds"][1] is None
        secondary = payload["detection_secondary"]
        assert secondary["threshold"] == pytest.approx(result.threshold)
        assert secondary["detected"] == list(detection_result.detected)
        assert len(secondary["latency_seconds"]) == 2
        assert "secondary" in secondary["note"]
        assert "DESCRIPTIVE ONLY" in secondary["significance_caveat"]
        bounds = secondary["chance_detection_upper_bound"]
        assert [entry["case"] for entry in bounds] == [
            "WECC_case1",
            "WECC_case2",
        ]
        for entry, detection in zip(bounds, detections, strict=True):
            expected = 1.0 - (1.0 - result.achieved_false_alarm) ** int(
                detection.detection_scores.size
            )
            assert entry["p_chance_upper"] == pytest.approx(expected)
            assert entry["n_windows"] == int(detection.detection_scores.size)
        protocol = payload["protocol"]
        assert protocol["forcing_seconds"] == FORCING_SECONDS
        assert protocol["null_end_seconds"] == NULL_END_SECONDS
        assert protocol["baseline_seconds"] == BASELINE_SECONDS
        assert payload["benchmark"] == "wecc_240_osl_modal_growth_cross_dataset"
