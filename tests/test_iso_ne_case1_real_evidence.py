# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — committed real ISO-NE PRC evidence integrity tests

"""Integrity tests for the committed real ISO-NE case-1 PMU ringdown evidence.

`examples/real_data/iso_ne_case1/pmu_ringdown_prc_evidence.json` is the first
non-synthetic sealed artefact the shipped chain produced: the real ISO-NE forced
oscillation (documented near 0.27 Hz) run through `spo pmu-ieee-adapt` and then
`spo pmu-ringdown`. These tests guard that committed artefact without the raw
capture (which is citation-only and not redistributed): they recompute both the
top-level and nested content seals to prove the record was not hand-edited, pin
the source digest of the derived series, and assert the screener recovered and
flagged the documented inter-area mode.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash

_EVIDENCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "real_data"
    / "iso_ne_case1"
    / "pmu_ringdown_prc_evidence.json"
)

#: SHA-256 of the derived single-channel ``time_s,frequency_hz`` series that the
#: adapter emits from ISO-NE case 1 (channel Sub:9:Ln:20). The adapter is
#: deterministic, so regenerating the series from the raw capture reproduces this
#: digest; pinning it here ties the committed evidence to a fixed provenance.
_DERIVED_SERIES_SHA256 = (
    "2ed93167ced93d75d61fef5dc9fb7a878ceb631d74a76fa352e046f64e32f915"
)


@pytest.fixture(scope="module")
def evidence() -> dict[str, Any]:
    """Return the committed ISO-NE case-1 PMU ringdown evidence record."""
    return json.loads(_EVIDENCE_PATH.read_text(encoding="utf-8"))


def test_top_level_content_seal_recomputes(evidence: dict[str, Any]) -> None:
    """The record's ``content_hash`` matches its canonical payload."""
    payload = copy.deepcopy(evidence)
    sealed = payload.pop("content_hash")
    assert canonical_record_hash(payload) == sealed


def test_nested_prc_evidence_seal_recomputes(evidence: dict[str, Any]) -> None:
    """The nested PRC evidence seal recomputes and matches the mirrored hash."""
    prc = copy.deepcopy(evidence["prc_evidence"])
    sealed = prc.pop("content_hash")
    assert canonical_record_hash(prc) == sealed
    assert evidence["prc_evidence_hash"] == evidence["prc_evidence"]["content_hash"]


def test_provenance_pins_the_derived_series_digest(evidence: dict[str, Any]) -> None:
    """The evidence records the fixed digest of the adapter's derived series."""
    assert evidence["source_sha256"] == _DERIVED_SERIES_SHA256
    assert evidence["source_name"] == "iso_ne_case1_frequency_Sub9Ln20.csv"


def test_review_only_claim_boundary_and_schema(evidence: dict[str, Any]) -> None:
    """The artefact carries the review-only claim boundary and audit schema."""
    assert evidence["schema"] == "scpn_pmu_ringdown_prc_audit_v1"
    assert evidence["claim_boundary"] == "review_only_offline_no_live_actuation"
    assert evidence["review_only"] is True


def test_screening_parameters_match_the_documented_run(
    evidence: dict[str, Any],
) -> None:
    """The recorded pre-processing controls match the documented reproduction."""
    assert evidence["nominal_frequency_hz"] == 60.0
    assert evidence["detrend"] == "mean"
    assert evidence["sample_count"] == 5400
    assert evidence["analysis_sample_count"] == 900
    assert evidence["analysis_rate_hz"] == pytest.approx(5.0, abs=1.0e-3)


def test_recovers_and_flags_the_documented_inter_area_mode(
    evidence: dict[str, Any],
) -> None:
    """A flagged inter-area mode near the documented 0.27 Hz event is present."""
    findings = evidence["prc_evidence"]["findings"]
    documented = [
        finding
        for finding in findings
        if finding["mode_family"] == "inter_area"
        and finding["flagged"] is True
        and 0.25 <= finding["frequency_hz"] <= 0.30
    ]
    assert documented, "the documented ~0.27 Hz inter-area mode is not flagged"
    assert evidence["prc_evidence"]["verdict"] == "flagged_for_review"
    assert evidence["prc_evidence"]["flagged_count"] == 3
