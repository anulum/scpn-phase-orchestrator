# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — committed Sleep-EDF staging evidence integrity tests

"""Integrity tests for the committed Sleep-EDF N3-vs-Wake audit evidence.

`examples/real_data/sleepedf_staging/sleepedf_n3_vs_wake_audit.json` is produced
by `bench/sleep_staging_sleepedf.py` from a public PhysioNet Sleep-EDF Expanded
recording. These tests guard the committed artefact without the raw EDF files
(which are citation-only and not redistributed): they recompute the content
seal, pin the source-file digests, and assert the documented honest-audit
result.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash

_EVIDENCE_DIR = (
    Path(__file__).resolve().parents[1] / "examples" / "real_data" / "sleepedf_staging"
)
_AUDIT_PATH = _EVIDENCE_DIR / "sleepedf_n3_vs_wake_audit.json"
_SUMMARY_PATH = _EVIDENCE_DIR / "sleepedf_n3_vs_wake_summary.json"

#: SHA-256 of the raw PSG EDF used to generate the committed evidence.
_PSG_SHA256 = "2b40a18adf76af69a42d6db1f30f31d26b369f6d27ca0050ef30147ef892b131"
#: SHA-256 of the raw hypnogram EDF used to generate the committed evidence.
_HYPNOGRAM_SHA256 = "a4cf67694ade1b52a0ddd06d5817fd45d2d3e8bac5302f640f3e9cfbbf12a996"
#: Content hash of the committed sealed audit record.
_AUDIT_CONTENT_HASH = "836b9deda96455b31734c319cb3f30e87fb1ec005fe4a397a555619b57e690d0"


@pytest.fixture(scope="module")
def audit() -> dict[str, Any]:
    """Return the committed Sleep-EDF N3-vs-Wake sealed audit record."""
    return json.loads(_AUDIT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def summary() -> dict[str, Any]:
    """Return the committed Sleep-EDF N3-vs-Wake summary record."""
    return json.loads(_SUMMARY_PATH.read_text(encoding="utf-8"))


def test_audit_content_seal_recomputes(audit: dict[str, Any]) -> None:
    """The record's ``content_hash`` matches its canonical payload."""
    payload = copy.deepcopy(audit)
    sealed = payload.pop("content_hash")
    assert canonical_record_hash(payload) == sealed


def test_audit_content_hash_matches_committed_value(audit: dict[str, Any]) -> None:
    """The committed record's content hash has not drifted."""
    assert audit["content_hash"] == _AUDIT_CONTENT_HASH


def test_summary_pins_source_file_digests(summary: dict[str, Any]) -> None:
    """The summary pins the SHA-256 digests of the citation-only source EDFs."""
    assert summary["source_files"]["psg_sha256"] == _PSG_SHA256
    assert summary["source_files"]["hypnogram_sha256"] == _HYPNOGRAM_SHA256


def test_summary_matches_audit_record(
    audit: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    """The summary carries the same audit identifiers and hash as the sealed record."""
    assert summary["audit_content_hash"] == audit["content_hash"]
    assert summary["corpus_id"] == audit["corpus_id"]
    assert summary["captured_at"] == audit["captured_at"]


def test_audit_verdict_matches_documented_result(audit: dict[str, Any]) -> None:
    """The sealed audit reports the documented N3-vs-Wake honest result."""
    a = audit["audit"]
    assert a["detector_name"] == "normalized_delta_envelope"
    assert a["n_events"] == 220
    assert a["n_nulls"] == 1997
    assert a["target_false_alarm"] == pytest.approx(0.10, abs=1.0e-6)
    assert a["achieved_false_alarm"] == pytest.approx(0.0996, abs=1.0e-3)
    assert a["detection_rate"] == pytest.approx(0.959, abs=1.0e-3)
    assert a["beats_chance"] is True
    assert a["significance"]["p_value"] < 0.001
    assert a["significance"]["seed"] == 42
    assert a["significance"]["n_permutations"] == 10_000


def test_summary_counts_and_scores(summary: dict[str, Any]) -> None:
    """The summary carries the documented epoch counts and mean scores."""
    assert summary["n_n3"] == 220
    assert summary["n_wake"] == 1997
    assert summary["n_epochs"] == 2650
    assert summary["score_mean_n3"] == pytest.approx(0.837, abs=1.0e-3)
    assert summary["score_mean_wake"] == pytest.approx(0.654, abs=1.0e-3)
