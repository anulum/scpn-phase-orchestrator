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

from scpn_phase_orchestrator.monitor.oscillation_modes import INTER_AREA_MODE
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
