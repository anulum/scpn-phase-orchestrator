# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — grid CSV ingestion parses the digested bytes

"""Grid CSV screens parse the bytes their source digest covers.

Spreadsheet "CSV UTF-8" exports prepend a byte-order mark. Decoded as plain
UTF-8 it glues itself to the first column name, so the first column is
reported missing. The screens now decode the digested bytes with
``utf-8-sig``: a BOM export screens exactly like the plain file, and
``source_sha256`` still covers the file's exact bytes, BOM included.
"""

from __future__ import annotations

import codecs
import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest

from scpn_phase_orchestrator.runtime.ibr_ride_through import (
    screen_ibr_ride_through_csv,
)
from scpn_phase_orchestrator.runtime.pmu_ieee_adapter import read_ieee_pmu_recording
from scpn_phase_orchestrator.runtime.pmu_ringdown import screen_pmu_ringdown_csv

_BOM = codecs.BOM_UTF8


def _ringdown_body() -> bytes:
    times = np.arange(200) / 30.0
    freqs = 60.0 + 0.05 * np.exp(-0.2 * times) * np.sin(2 * np.pi * 0.7 * times)
    rows = "".join(f"{t:.6f},{f:.6f}\n" for t, f in zip(times, freqs, strict=True))
    return ("time_s,frequency_hz\n" + rows).encode("utf-8")


def _ride_through_body() -> bytes:
    return (
        b"time_s,voltage_pu,frequency_hz\n0.0,0.82,60.0\n3.5,0.82,60.0\n7.0,1.00,60.0\n"
    )


def _write(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


@pytest.mark.parametrize("prefix", [b"", _BOM], ids=["plain", "bom"])
def test_ringdown_screens_bom_export_like_plain_file(
    tmp_path: Path, prefix: bytes
) -> None:
    path = _write(tmp_path / "ringdown.csv", prefix + _ringdown_body())
    evidence = screen_pmu_ringdown_csv(
        path, event_id="E1", captured_at="2026-09-24T00:00:00Z", signal_source="S"
    )
    reference = screen_pmu_ringdown_csv(
        _write(tmp_path / "reference.csv", _ringdown_body()),
        event_id="E1",
        captured_at="2026-09-24T00:00:00Z",
        signal_source="S",
    )
    assert evidence.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert evidence.sample_count == 200
    assert evidence.prc_evidence.content_hash == reference.prc_evidence.content_hash


@pytest.mark.parametrize("prefix", [b"", _BOM], ids=["plain", "bom"])
def test_ride_through_screens_bom_export_like_plain_file(
    tmp_path: Path, prefix: bytes
) -> None:
    path = _write(tmp_path / "ride.csv", prefix + _ride_through_body())
    evidence = screen_ibr_ride_through_csv(
        path, event_id="E2", captured_at="2026-09-24T00:00:00Z", signal_source="S"
    )
    reference = screen_ibr_ride_through_csv(
        _write(tmp_path / "reference.csv", _ride_through_body()),
        event_id="E2",
        captured_at="2026-09-24T00:00:00Z",
        signal_source="S",
    )
    assert evidence.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert evidence.sample_count == 3
    assert (
        evidence.prc029_evidence.content_hash == reference.prc029_evidence.content_hash
    )


def test_ieee_recording_reads_bom_export_like_plain_file(tmp_path: Path) -> None:
    times = [round(0.1 * i, 3) for i in range(20)]
    rows = [
        ["Time", "BUS1 Freq"],
        ["T", "F"],
        ["sec", "Hz"],
        *[[t, 60.0 + 0.01 * i] for i, t in enumerate(times)],
    ]
    plain = tmp_path / "plain.csv"
    with plain.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)
    bom = _write(tmp_path / "bom.csv", _BOM + plain.read_bytes())

    reference = read_ieee_pmu_recording(plain)
    recording = read_ieee_pmu_recording(bom)
    assert recording.source_sha256 == hashlib.sha256(bom.read_bytes()).hexdigest()
    assert np.array_equal(recording.times, reference.times)
    assert [c.label for c in recording.channels] == [
        c.label for c in reference.channels
    ]
