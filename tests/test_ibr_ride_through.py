# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — IBR ride-through CSV ingestion tests

"""Tests for review-only IBR ride-through CSV ingestion."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scpn_phase_orchestrator.assurance.prc_ride_through import (
    ASSESSOR_REVIEW_REQUIRED,
    OTHER_IBR,
)
from scpn_phase_orchestrator.runtime.ibr_ride_through import (
    IBR_RIDE_THROUGH_AUDIT_SCHEMA,
    IBR_RIDE_THROUGH_CLAIM_BOUNDARY,
    IBRRideThroughCsvEvidence,
    screen_ibr_ride_through_csv,
)

_CAPTURED_AT = "2026-07-04T13:45:00Z"


def _ride_through_csv(path: Path) -> Path:
    """Write a deterministic operator ride-through fixture."""
    path.write_text(
        "\n".join(
            [
                "time_s,voltage_pu,frequency_hz",
                "0.0,0.82,60.0",
                "3.5,0.82,60.0",
                "7.0,1.00,60.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_screen_ibr_ride_through_csv_hash_seals_prc029_evidence(
    tmp_path: Path,
) -> None:
    """A reviewed operator CSV is converted into deterministic PRC-029 evidence."""
    csv_path = _ride_through_csv(tmp_path / "ride-through.csv")

    evidence = screen_ibr_ride_through_csv(
        csv_path,
        event_id="IBR-EVT-001",
        captured_at=_CAPTURED_AT,
        signal_source="IBR-17/high-side-transformer",
    )

    assert isinstance(evidence, IBRRideThroughCsvEvidence)
    assert evidence.schema == IBR_RIDE_THROUGH_AUDIT_SCHEMA
    assert evidence.claim_boundary == IBR_RIDE_THROUGH_CLAIM_BOUNDARY
    assert evidence.review_only is True
    assert evidence.ibr_category == OTHER_IBR
    assert evidence.sample_count == 3
    assert evidence.duration_s == pytest.approx(7.0)
    assert evidence.source_sha256 == hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert evidence.prc029_evidence.verdict == ASSESSOR_REVIEW_REQUIRED
    assert evidence.prc029_evidence.flagged_count == 1

    record = evidence.to_audit_record()
    assert record["schema"] == IBR_RIDE_THROUGH_AUDIT_SCHEMA
    assert record["claim_boundary"] == IBR_RIDE_THROUGH_CLAIM_BOUNDARY
    assert record["review_only"] is True
    assert record["source_sha256"] == evidence.source_sha256
    assert record["prc029_evidence_hash"] == evidence.prc029_evidence.content_hash
    assert len(str(record["content_hash"])) == 64
    assert record == evidence.to_audit_record()


@pytest.mark.parametrize(
    ("contents", "match"),
    [
        (
            "timestamp,voltage_pu,frequency_hz\n0.0,1.0,60.0\n",
            "missing required column time_s",
        ),
        (
            "time_s,voltage_pu,frequency_hz\n0.0,1.0,60.0\n1.0,bad,60.0\n",
            "voltage_pu row 2 must be a finite real",
        ),
        (
            "time_s,voltage_pu,frequency_hz\n0.0,1.0,60.0\n1.0,1.0\n",
            "frequency_hz row 2 must be a finite real",
        ),
        (
            "time_s,voltage_pu,frequency_hz\n0.0,1.0,60.0\n1.0,1.0,nan\n",
            "frequency_hz row 2 must be finite",
        ),
        (
            "time_s,voltage_pu,frequency_hz\n0.0,1.0,60.0\n0.0,1.0,60.0\n",
            "time_s must be strictly increasing",
        ),
    ],
)
def test_screen_ibr_ride_through_csv_rejects_invalid_operator_data(
    tmp_path: Path, contents: str, match: str
) -> None:
    """Malformed operator CSV data fails before evidence is published."""
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        screen_ibr_ride_through_csv(
            csv_path,
            event_id="IBR-EVT-001",
            captured_at=_CAPTURED_AT,
            signal_source="IBR-17/high-side-transformer",
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"event_id": ""}, "event_id must be a non-empty string"),
        ({"time_column": ""}, "time_column must be a non-empty string"),
        ({"ibr_category": "solar"}, "ibr_category"),
    ],
)
def test_screen_ibr_ride_through_csv_rejects_invalid_controls(
    tmp_path: Path, kwargs: dict[str, object], match: str
) -> None:
    """Invalid operator-supplied screening controls fail closed."""
    csv_path = _ride_through_csv(tmp_path / "ride-through.csv")
    args: dict[str, object] = {
        "path": csv_path,
        "event_id": "IBR-EVT-001",
        "captured_at": _CAPTURED_AT,
        "signal_source": "IBR-17/high-side-transformer",
    }
    args.update(kwargs)

    with pytest.raises(ValueError, match=match):
        screen_ibr_ride_through_csv(**args)  # type: ignore[arg-type]
