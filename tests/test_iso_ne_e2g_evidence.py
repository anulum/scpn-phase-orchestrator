# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — committed ISO-NE E2.G evidence tests

"""Integrity tests for the committed ISO-NE E2.G cross-dataset evidence.

`examples/real_data/iso_ne_forced_oscillation/` seals the E2.G portability test
of the PSML-certified modal-growth detector on real ISO-NE captures. These
tests guard the committed derived artefact without the citation-only raw data:
they recompute the content hash to prove the record was not hand-edited, pin
the source digests (case 1 must equal the digest sealed in the E1 artefact),
and assert the honest headline — the frozen operating point does not port, the
locally calibrated shape leads one of three events, and no significance is
claimed at n = 3.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from bench.early_warning_leadtime_isone import (
    EXCLUDED_CASES,
    FROZEN_PSML_THRESHOLD,
)
from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash

_DIR = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "real_data"
    / "iso_ne_forced_oscillation"
)

#: Raw-source digests; case 1 is the SAME bytes the E1 artefact sealed in July
#: 2026 (`examples/real_data/iso_ne_case1/README.md`), chaining the two legs.
_SOURCE_SHA256 = {
    "ISO-NE_case1": (
        "ca5001bb64cfecced20ea71a6a007a5db8ad96acdcfa13cb021358f0f2575de0"
    ),
    "ISO-NE_case2": (
        "e503003f2df7e02262a9412700ed3be6833bb9f95e120ceea21596b4a6f53f1f"
    ),
    "ISO-NE_case3": (
        "f30d399b017aeee04987dadbf9cec0117444b4372456c93aa367e86ccba3772f"
    ),
}

_CONTENT_HASH = "66edc9e10e81b2a51d2fedf87d5e12f6175810b0b083559a761481342c1f51ba"


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    """The committed sealed payload, parsed once per module."""
    path = _DIR / "iso_ne_modal_growth_cross_dataset.json"
    return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def test_content_hash_recomputes_from_the_payload(payload: dict[str, Any]) -> None:
    """The sealed hash must recompute from the committed record alone."""
    record = copy.deepcopy(payload)
    sealed = record.pop("content_hash")
    assert sealed == _CONTENT_HASH
    assert canonical_record_hash(record) == sealed


def test_tampered_payload_is_rejected(payload: dict[str, Any]) -> None:
    """Any edit to the record must break the seal."""
    record = copy.deepcopy(payload)
    record.pop("content_hash")
    record["local_calibration"]["led"] = [True, True, True]
    assert canonical_record_hash(record) != _CONTENT_HASH


def test_source_digests_chain_to_the_raw_captures(payload: dict[str, Any]) -> None:
    """Each transition pins its raw CSV digest; case 1 chains to the E1 seal."""
    transitions = {entry["case"]: entry for entry in payload["corpus"]["transitions"]}
    assert set(transitions) == set(_SOURCE_SHA256)
    for case, digest in _SOURCE_SHA256.items():
        assert transitions[case]["source_sha256"] == digest


def test_exclusions_are_sealed(payload: dict[str, Any]) -> None:
    """The pre-registered corpus exclusions are part of the sealed record."""
    assert payload["corpus"]["excluded"] == EXCLUDED_CASES


def test_frozen_transfer_headline_is_honest(payload: dict[str, Any]) -> None:
    """G-a seals the non-portability: ambient FA far above the certified rate."""
    by_case = {entry["case"]: entry for entry in payload["frozen_transfer"]}
    assert by_case["ISO-NE_case2"]["threshold"] == FROZEN_PSML_THRESHOLD
    assert by_case["ISO-NE_case2"]["n_null"] == 219
    assert by_case["ISO-NE_case2"]["null_crossings"] == 71
    ambient_fa = 71 / 219
    assert ambient_fa > 0.25  # certified per-window FA was 0.0909


def test_local_calibration_headline_is_honest(payload: dict[str, Any]) -> None:
    """G-b seals one led event, a positive lead, held FA, and p above 0.05."""
    calibration = payload["local_calibration"]
    assert calibration["led"] == [False, False, True]
    leads = calibration["lead_seconds"]
    assert leads[0] is None and leads[1] is None
    assert leads[2] == pytest.approx(57.123, abs=0.01)
    assert calibration["achieved_false_alarm"] <= calibration["target_false_alarm"]
    significance = calibration["significance"]
    assert significance["observed_led"] == 1
    assert significance["n_transitions"] == 3
    assert significance["p_value"] > 0.05  # no significance claim at n=3
