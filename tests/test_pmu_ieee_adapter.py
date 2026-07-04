# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — IEEE multi-header PMU CSV adapter tests

"""Tests for the IEEE-format multi-header PMU CSV ingestion adapter.

The adapter converts the wide, multi-header concentrator layout used by phasor
measurement units and the oscillation-detection literature into the ringdown
screener's two-column input. The tests exercise the header-block location, the
frequency-channel enumeration and dropout accounting, the cleanest-channel
selection with its band and swing rules, the derived-CSV writer with its
provenance, and the failure paths for malformed captures — then confirm the
derived CSV reparses through the screener and recovers a planted oscillation.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scpn_phase_orchestrator.runtime.pmu_ieee_adapter import (
    IEEE_FREQUENCY_QUANTITY,
    AdaptedIngesterCSV,
    PMUFrequencyChannel,
    adapt_ieee_pmu_csv,
    read_ieee_pmu_recording,
    write_ingester_csv,
)
from scpn_phase_orchestrator.runtime.pmu_ringdown import screen_pmu_ringdown_csv


def _write_ieee_document(path: Path, rows: Sequence[Sequence[object]]) -> None:
    """Write ``rows`` verbatim as an IEEE-format PMU CSV."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for row in rows:
            writer.writerow(list(row))


def _standard_recording(
    path: Path,
    *,
    channels: Sequence[tuple[str, str, str, Sequence[float]]],
    times: Sequence[float],
    include_units: bool = True,
    include_secondary: bool = True,
) -> None:
    """Write a four-row-header IEEE PMU document with the given channels.

    Each channel is ``(label, quantity, unit, samples)``; the time column is
    prepended automatically. Header rows are label, quantity, optional unit, and
    optional secondary-label, followed by one numeric row per timestamp.
    """
    label_row = ["Time", *(label for label, _, _, _ in channels)]
    quantity_row = ["T", *(quantity for _, quantity, _, _ in channels)]
    header: list[list[object]] = [label_row, quantity_row]
    if include_units:
        header.append(["sec", *(unit for _, _, unit, _ in channels)])
    if include_secondary:
        header.append(["1900-01-01 00:00:00", *("Ln:x" for _ in channels)])
    data: list[list[object]] = []
    for index, time_value in enumerate(times):
        data.append([time_value, *(samples[index] for _, _, _, samples in channels)])
    _write_ieee_document(path, [*header, *data])


def _ringing(
    times: Sequence[float], *, frequency_hz: float, offset: float
) -> list[float]:
    """Return a decaying sinusoid on ``times`` at ``offset`` plus nominal 60 Hz."""
    array = np.asarray(times, dtype=np.float64)
    envelope = np.exp(-0.05 * array)
    oscillation = 0.02 * envelope * np.sin(2.0 * np.pi * frequency_hz * array)
    return list(60.0 + offset + oscillation)


# --------------------------------------------------------------------------- #
# PMUFrequencyChannel                                                          #
# --------------------------------------------------------------------------- #


def _channel(**overrides: object) -> PMUFrequencyChannel:
    """Return a frequency channel with defaulted, override-able fields."""
    base = PMUFrequencyChannel(
        label="Sub:1:Ln:1",
        column_index=1,
        samples=np.array([60.0, 60.1, 59.9], dtype=np.float64),
        zero_count=0,
        nonfinite_count=0,
        mean_hz=60.0,
        min_hz=59.9,
        max_hz=60.1,
    )
    return replace(base, **overrides)


def test_channel_derived_properties_report_swing_identity_and_cleanliness() -> None:
    channel = _channel()
    assert channel.peak_to_peak_hz == pytest.approx(0.2)
    assert channel.identifier == "Sub:1:Ln:1#1"
    assert channel.is_clean is True


@pytest.mark.parametrize(
    ("zero_count", "nonfinite_count", "expected"),
    [(0, 0, True), (1, 0, False), (0, 1, False), (2, 3, False)],
)
def test_channel_is_clean_requires_no_dropout_or_nonfinite(
    zero_count: int, nonfinite_count: int, expected: bool
) -> None:
    channel = _channel(zero_count=zero_count, nonfinite_count=nonfinite_count)
    assert channel.is_clean is expected


@pytest.mark.parametrize(
    ("mean_hz", "expected"),
    [(60.0, True), (61.9, True), (62.1, False), (float("nan"), False)],
)
def test_channel_band_membership_tracks_finite_mean(
    mean_hz: float, expected: bool
) -> None:
    channel = _channel(mean_hz=mean_hz)
    assert channel.is_within_band(60.0, 2.0) is expected


# --------------------------------------------------------------------------- #
# read_ieee_pmu_recording                                                      #
# --------------------------------------------------------------------------- #


def test_reads_time_vector_and_frequency_channels_from_standard_document(
    tmp_path: Path,
) -> None:
    times = [0.0, 0.1, 0.2, 0.3]
    _standard_recording(
        tmp_path / "cap.csv",
        times=times,
        channels=[
            ("Sub:1:Ln:1", "F", "HZ", [60.01, 60.02, 60.0, 59.99]),
            ("Sub:1:Ln:1", "VM", "KV", [200.0, 200.1, 200.2, 200.3]),
            ("Sub:2:Ln:2", "F", "HZ", [59.98, 60.05, 60.1, 59.9]),
        ],
    )
    recording = read_ieee_pmu_recording(tmp_path / "cap.csv")
    assert recording.source_name == "cap.csv"
    assert (
        recording.source_sha256
        == hashlib.sha256((tmp_path / "cap.csv").read_bytes()).hexdigest()
    )
    np.testing.assert_allclose(recording.times, times)
    assert [channel.column_index for channel in recording.channels] == [1, 3]
    assert [channel.label for channel in recording.channels] == [
        "Sub:1:Ln:1",
        "Sub:2:Ln:2",
    ]
    assert recording.channels[0].zero_count == 0
    assert recording.channels[0].mean_hz == pytest.approx(60.005)


def test_parses_capture_without_units_row(tmp_path: Path) -> None:
    _standard_recording(
        tmp_path / "no_units.csv",
        times=[0.0, 0.1, 0.2],
        channels=[("Sub:1:Ln:1", "F", "HZ", [60.0, 60.1, 59.9])],
        include_units=False,
    )
    recording = read_ieee_pmu_recording(tmp_path / "no_units.csv")
    assert len(recording.channels) == 1
    assert recording.channels[0].mean_hz == pytest.approx(60.0)


def test_missing_quantity_row_is_rejected(tmp_path: Path) -> None:
    _write_ieee_document(
        tmp_path / "bad.csv",
        [["Time", "Sub:1:Ln:1"], ["0.0", "60.0"], ["0.1", "60.1"]],
    )
    with pytest.raises(ValueError, match="no quantity-type row"):
        read_ieee_pmu_recording(tmp_path / "bad.csv")


def test_quantity_row_without_label_row_is_rejected(tmp_path: Path) -> None:
    _write_ieee_document(
        tmp_path / "bad.csv",
        [["T", "F"], ["sec", "HZ"], ["0.0", "60.0"], ["0.1", "60.1"]],
    )
    with pytest.raises(ValueError, match="no label row above"):
        read_ieee_pmu_recording(tmp_path / "bad.csv")


def test_header_without_data_rows_is_rejected(tmp_path: Path) -> None:
    _write_ieee_document(
        tmp_path / "bad.csv",
        [["Time", "Sub:1:Ln:1"], ["T", "F"], ["sec", "HZ"]],
    )
    with pytest.raises(ValueError, match="no numeric sample rows"):
        read_ieee_pmu_recording(tmp_path / "bad.csv")


def test_document_without_frequency_channels_is_rejected(tmp_path: Path) -> None:
    _standard_recording(
        tmp_path / "bad.csv",
        times=[0.0, 0.1],
        channels=[("Sub:1:Ln:1", "VM", "KV", [200.0, 200.1])],
    )
    with pytest.raises(ValueError, match="no frequency channels"):
        read_ieee_pmu_recording(tmp_path / "bad.csv")


def test_frequency_channel_reported_in_wrong_unit_is_rejected(tmp_path: Path) -> None:
    _standard_recording(
        tmp_path / "bad.csv",
        times=[0.0, 0.1],
        channels=[("Sub:1:Ln:1", "F", "RPM", [60.0, 60.1])],
    )
    with pytest.raises(ValueError, match="not 'HZ'"):
        read_ieee_pmu_recording(tmp_path / "bad.csv")


def test_truncated_unit_row_skips_the_unit_cross_check(tmp_path: Path) -> None:
    # Unit row covers only the time column, so the frequency column's unit is
    # absent and the cross-check is skipped rather than raising.
    _write_ieee_document(
        tmp_path / "cap.csv",
        [
            ["Time", "Sub:1:Ln:1", "Sub:2:Ln:2"],
            ["T", "F", "F"],
            ["sec"],
            ["0.0", "60.0", "59.9"],
            ["0.1", "60.1", "60.0"],
        ],
    )
    recording = read_ieee_pmu_recording(tmp_path / "cap.csv")
    assert [channel.column_index for channel in recording.channels] == [1, 2]


def test_ragged_data_row_is_rejected(tmp_path: Path) -> None:
    _write_ieee_document(
        tmp_path / "bad.csv",
        [
            ["Time", "Sub:1:Ln:1", "Sub:2:Ln:2"],
            ["T", "F", "F"],
            ["sec", "HZ", "HZ"],
            ["0.0", "60.0", "59.9"],
            ["0.1", "60.1"],
        ],
    )
    with pytest.raises(ValueError, match="fewer than the 3 required"):
        read_ieee_pmu_recording(tmp_path / "bad.csv")


def test_non_numeric_sample_is_rejected(tmp_path: Path) -> None:
    _standard_recording(
        tmp_path / "bad.csv",
        times=[0.0, 0.1],
        channels=[("Sub:1:Ln:1", "F", "HZ", [60.0, 60.1])],
    )
    document = (tmp_path / "bad.csv").read_text(encoding="utf-8").splitlines()
    document[-1] = "0.1,not-a-number"
    (tmp_path / "bad.csv").write_text("\n".join(document) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be numeric"):
        read_ieee_pmu_recording(tmp_path / "bad.csv")


def test_blank_data_rows_are_skipped(tmp_path: Path) -> None:
    _write_ieee_document(
        tmp_path / "cap.csv",
        [
            ["Time", "Sub:1:Ln:1"],
            ["T", "F"],
            ["sec", "HZ"],
            ["0.0", "60.0"],
            [],
            ["0.1", "60.1"],
            ["", ""],
        ],
    )
    recording = read_ieee_pmu_recording(tmp_path / "cap.csv")
    assert recording.times.shape[0] == 2
    np.testing.assert_allclose(recording.times, [0.0, 0.1])


def test_channel_accounts_for_zero_dropouts_and_nonfinite_samples(
    tmp_path: Path,
) -> None:
    _write_ieee_document(
        tmp_path / "cap.csv",
        [
            ["Time", "Clean", "Dropout", "Dead"],
            ["T", "F", "F", "F"],
            ["sec", "HZ", "HZ", "HZ"],
            ["0.0", "60.0", "0.0", "nan"],
            ["0.1", "60.1", "60.1", "nan"],
            ["0.2", "59.9", "59.9", "inf"],
        ],
    )
    recording = read_ieee_pmu_recording(tmp_path / "cap.csv")
    clean, dropout, dead = recording.channels
    assert clean.zero_count == 0 and clean.nonfinite_count == 0
    assert dropout.zero_count == 1 and dropout.is_clean is False
    assert dead.nonfinite_count == 3
    assert np.isnan(dead.mean_hz) and np.isnan(dead.min_hz) and np.isnan(dead.max_hz)


def test_channel_label_falls_back_when_label_is_blank_or_absent(
    tmp_path: Path,
) -> None:
    # The label row is shorter than the second frequency column and the first
    # frequency column's label is blank, so both fall back to a column label.
    _write_ieee_document(
        tmp_path / "cap.csv",
        [
            ["Time", ""],
            ["T", "F", "F"],
            ["sec", "HZ", "HZ"],
            ["0.0", "60.0", "59.9"],
            ["0.1", "60.1", "60.0"],
        ],
    )
    recording = read_ieee_pmu_recording(tmp_path / "cap.csv")
    assert [channel.label for channel in recording.channels] == [
        "column_1",
        "column_2",
    ]


# --------------------------------------------------------------------------- #
# IEEEPMURecording.select_cleanest_channel                                     #
# --------------------------------------------------------------------------- #


def test_selection_prefers_the_largest_in_band_swing(tmp_path: Path) -> None:
    times = list(np.linspace(0.0, 1.0, 11))
    _standard_recording(
        tmp_path / "cap.csv",
        times=times,
        channels=[
            ("Small", "F", "HZ", list(60.0 + 0.005 * np.sin(times))),
            ("Large", "F", "HZ", list(60.0 + 0.05 * np.sin(times))),
            ("OutOfBand", "F", "HZ", [50.0] * len(times)),
        ],
    )
    recording = read_ieee_pmu_recording(tmp_path / "cap.csv")
    selected = recording.select_cleanest_channel(
        nominal_frequency_hz=60.0, plausible_band_hz=2.0
    )
    assert selected.label == "Large"


def test_selection_breaks_ties_toward_the_lowest_column(tmp_path: Path) -> None:
    times = list(np.linspace(0.0, 1.0, 6))
    swing = list(60.0 + 0.02 * np.sin(times))
    _standard_recording(
        tmp_path / "cap.csv",
        times=times,
        channels=[("First", "F", "HZ", swing), ("Second", "F", "HZ", swing)],
    )
    recording = read_ieee_pmu_recording(tmp_path / "cap.csv")
    selected = recording.select_cleanest_channel()
    assert selected.label == "First"
    assert selected.column_index == 1


def test_selection_without_a_qualifying_channel_is_rejected(tmp_path: Path) -> None:
    _write_ieee_document(
        tmp_path / "cap.csv",
        [
            ["Time", "Dropout", "OutOfBand"],
            ["T", "F", "F"],
            ["sec", "HZ", "HZ"],
            ["0.0", "0.0", "50.0"],
            ["0.1", "60.1", "50.0"],
        ],
    )
    recording = read_ieee_pmu_recording(tmp_path / "cap.csv")
    with pytest.raises(ValueError, match=r"1 carried dropouts"):
        recording.select_cleanest_channel(
            nominal_frequency_hz=60.0, plausible_band_hz=2.0
        )


@pytest.mark.parametrize(
    ("nominal", "band"),
    [(0.0, 2.0), (-1.0, 2.0), (float("inf"), 2.0), (60.0, 0.0), (60.0, float("nan"))],
)
def test_selection_rejects_invalid_controls(
    tmp_path: Path, nominal: float, band: float
) -> None:
    _standard_recording(
        tmp_path / "cap.csv",
        times=[0.0, 0.1],
        channels=[("Sub:1:Ln:1", "F", "HZ", [60.0, 60.1])],
    )
    recording = read_ieee_pmu_recording(tmp_path / "cap.csv")
    with pytest.raises(ValueError, match="positive finite"):
        recording.select_cleanest_channel(
            nominal_frequency_hz=nominal, plausible_band_hz=band
        )


# --------------------------------------------------------------------------- #
# write_ingester_csv                                                           #
# --------------------------------------------------------------------------- #


def test_writes_ingester_csv_with_source_linked_provenance(tmp_path: Path) -> None:
    _standard_recording(
        tmp_path / "cap.csv",
        times=[0.0, 0.1, 0.2],
        channels=[("Sub:3:Ln:5", "F", "HZ", [60.0, 60.1, 59.9])],
    )
    recording = read_ieee_pmu_recording(tmp_path / "cap.csv")
    channel = recording.channels[0]
    export = write_ingester_csv(recording, channel, tmp_path / "out.csv")
    assert isinstance(export, AdaptedIngesterCSV)
    assert export.source_name == "cap.csv"
    assert export.source_sha256 == recording.source_sha256
    assert export.output_name == "out.csv"
    assert export.channel_label == "Sub:3:Ln:5"
    assert export.channel_column_index == 1
    assert export.time_column == "time_s"
    assert export.frequency_column == "frequency_hz"
    assert export.row_count == 3
    written = (tmp_path / "out.csv").read_bytes()
    assert export.output_sha256 == hashlib.sha256(written).hexdigest()
    with (tmp_path / "out.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["time_s", "frequency_hz"]
        parsed = [(float(r["time_s"]), float(r["frequency_hz"])) for r in reader]
    np.testing.assert_allclose([t for t, _ in parsed], [0.0, 0.1, 0.2])
    np.testing.assert_allclose([f for _, f in parsed], [60.0, 60.1, 59.9])


def test_write_honours_custom_column_names(tmp_path: Path) -> None:
    _standard_recording(
        tmp_path / "cap.csv",
        times=[0.0, 0.1],
        channels=[("Sub:1:Ln:1", "F", "HZ", [60.0, 60.1])],
    )
    recording = read_ieee_pmu_recording(tmp_path / "cap.csv")
    export = write_ingester_csv(
        recording,
        recording.channels[0],
        tmp_path / "out.csv",
        time_column="t",
        frequency_column="f",
    )
    assert export.time_column == "t"
    assert export.frequency_column == "f"
    with (tmp_path / "out.csv").open(encoding="utf-8", newline="") as handle:
        assert csv.DictReader(handle).fieldnames == ["t", "f"]


@pytest.mark.parametrize(
    ("time_column", "frequency_column"),
    [(" ", "frequency_hz"), ("time_s", "")],
)
def test_write_rejects_blank_column_names(
    tmp_path: Path, time_column: str, frequency_column: str
) -> None:
    _standard_recording(
        tmp_path / "cap.csv",
        times=[0.0, 0.1],
        channels=[("Sub:1:Ln:1", "F", "HZ", [60.0, 60.1])],
    )
    recording = read_ieee_pmu_recording(tmp_path / "cap.csv")
    with pytest.raises(ValueError, match="non-empty string"):
        write_ingester_csv(
            recording,
            recording.channels[0],
            tmp_path / "out.csv",
            time_column=time_column,
            frequency_column=frequency_column,
        )


def test_write_rejects_channel_not_aligned_with_time_vector(tmp_path: Path) -> None:
    _standard_recording(
        tmp_path / "cap.csv",
        times=[0.0, 0.1, 0.2],
        channels=[("Sub:1:Ln:1", "F", "HZ", [60.0, 60.1, 59.9])],
    )
    recording = read_ieee_pmu_recording(tmp_path / "cap.csv")
    misaligned = _channel(samples=np.array([60.0, 60.1], dtype=np.float64))
    with pytest.raises(ValueError, match="not aligned"):
        write_ingester_csv(recording, misaligned, tmp_path / "out.csv")


# --------------------------------------------------------------------------- #
# adapt_ieee_pmu_csv (end to end)                                             #
# --------------------------------------------------------------------------- #


def test_adapt_selects_writes_and_reparses_through_the_screener(
    tmp_path: Path,
) -> None:
    times = list(np.arange(0.0, 30.0, 0.1))
    _standard_recording(
        tmp_path / "cap.csv",
        times=times,
        channels=[
            ("Flat", "F", "HZ", [60.02] * len(times)),
            ("Ring", "F", "HZ", _ringing(times, frequency_hz=0.4, offset=0.02)),
        ],
    )
    export = adapt_ieee_pmu_csv(
        tmp_path / "cap.csv",
        tmp_path / "out.csv",
        nominal_frequency_hz=60.0,
        plausible_band_hz=2.0,
    )
    assert export.channel_label == "Ring"
    evidence = screen_pmu_ringdown_csv(
        tmp_path / "out.csv",
        event_id="synthetic",
        captured_at="2026-07-04",
        signal_source="adapter-test",
        nominal_frequency_hz=60.0,
        detrend="mean",
    )
    recovered = [
        finding.frequency_hz
        for finding in evidence.prc_evidence.findings
        if 0.3 < finding.frequency_hz < 0.5
    ]
    assert recovered, "adapter output did not preserve the planted 0.4 Hz mode"


def test_adapt_propagates_a_selection_failure(tmp_path: Path) -> None:
    _standard_recording(
        tmp_path / "cap.csv",
        times=[0.0, 0.1],
        channels=[("Sub:1:Ln:1", "F", "HZ", [50.0, 50.0])],
    )
    with pytest.raises(ValueError, match="no frequency channel within"):
        adapt_ieee_pmu_csv(tmp_path / "cap.csv", tmp_path / "out.csv")


def test_public_constants_expose_the_quantity_tokens() -> None:
    assert IEEE_FREQUENCY_QUANTITY == "F"
