# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — committed real CHB-MIT chb01 early-warning evidence tests

"""Integrity tests for the committed real CHB-MIT chb01 early-warning evidence.

`examples/real_data/chb01_seizures/` is the empirical capstone of the
early-warning detector suite: the suite and its weighted fusion run on real
annotated seizures from the CHB-MIT Scalp EEG Database (Shoeb 2009), calibrated
to a matched false-alarm rate, and sealed per detector per seizure. These tests
guard the committed derived artefact without the raw EEG (citation-only, not
redistributed): they recompute every sealed record's content hash to prove it was
not hand-edited, pin the digests of the one leading detection, and assert the
honest sparse-detection result — one of six evaluated seizures led, no robust
early-warning advantage, and the early-onset seizure excluded rather than counted
as a silent null.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash

_DIR = Path(__file__).resolve().parents[1] / "examples" / "real_data" / "chb01_seizures"
#: The six evaluated seizures (chb01_21 is excluded for an early onset).
_SEIZURES = (
    "chb01_03",
    "chb01_04",
    "chb01_15",
    "chb01_16",
    "chb01_18",
    "chb01_26",
)
#: The single seizure that produced a leading alarm at matched false alarm.
_LED_SEIZURE = "chb01_04"
#: Digests of the leading detections on ``chb01_04``. The pipeline and detectors
#: are deterministic, so regenerating the artefact reproduces these; pinning them
#: ties the committed evidence to a fixed derived provenance.
_PINNED = {
    "synchronisation": (
        "639db152b4948be6e8444a3e959735c62a79501d347963cb91e7e3786a16759b"
    ),
    "ensemble_weighted": (
        "a687a87e4db37e8178775960d15f863efa4a055cfd8efee53e59d1a1186943bf"
    ),
}


@pytest.fixture(scope="module")
def aggregate() -> dict[str, Any]:
    """Return the committed aggregate results record."""
    return json.loads(
        (_DIR / "early_warning_leadtime_eeg_results.json").read_text(encoding="utf-8")
    )


def _evidence(record_id: str) -> dict[str, Any]:
    """Return the committed sealed evidence for one seizure."""
    return json.loads(
        (_DIR / f"{record_id}_early_warning_evidence.json").read_text(encoding="utf-8")
    )


def test_every_detector_seal_recomputes() -> None:
    """Each detector record's ``content_hash`` matches its canonical payload."""
    for record_id in _SEIZURES:
        detectors = _evidence(record_id)["detectors"]
        for name, record in detectors.items():
            payload = copy.deepcopy(record)
            sealed = payload.pop("content_hash")
            assert canonical_record_hash(payload) == sealed, f"{record_id}/{name}"


def test_only_the_led_seizure_fires_and_pins_its_digest() -> None:
    """``chb01_04`` alone is led (synchronisation + fusion), with pinned digests."""
    detectors = _evidence(_LED_SEIZURE)["detectors"]
    assert detectors["synchronisation"]["content_hash"] == _PINNED["synchronisation"]
    assert (
        detectors["ensemble_weighted"]["content_hash"] == _PINNED["ensemble_weighted"]
    )
    assert detectors["synchronisation"]["lead_is_early"] is True
    assert detectors["ensemble_weighted"]["lead_is_early"] is True
    # Critical slowing down and transition entropy do not fire even here.
    assert detectors["critical_slowing_down"]["warning_triggered"] is False
    assert detectors["transition_entropy"]["warning_triggered"] is False


def test_the_other_evaluated_seizures_are_sealed_silences() -> None:
    """No detector fires on the five seizures other than ``chb01_04``."""
    for record_id in _SEIZURES:
        if record_id == _LED_SEIZURE:
            continue
        detectors = _evidence(record_id)["detectors"]
        for name, record in detectors.items():
            assert record["warning_triggered"] is False, f"{record_id}/{name} fired"
            assert record["lead_is_early"] is False


def test_aggregate_records_the_honest_sparse_verdict(
    aggregate: dict[str, Any],
) -> None:
    """The aggregate carries the sparse-detection verdict and its parameters."""
    assert aggregate["verdict"].startswith("SPARSE DETECTION, NO ROBUST ADVANTAGE")
    assert aggregate["n_null_trials"] == 20
    assert aggregate["segment_seconds"] == 900.0
    assert aggregate["matched_false_alarm_thresholds"] == {
        "critical_slowing_down": 4.5,
        "synchronisation": 3.75,
        "transition_entropy": 0.25,
        "ensemble_weighted": 3.0,
    }


def test_early_onset_seizure_is_excluded_not_a_silent_null(
    aggregate: dict[str, Any],
) -> None:
    """``chb01_21`` (onset 327 s) is excluded, not sealed as a silent null."""
    excluded = [entry["record_id"] for entry in aggregate["excluded_seizures"]]
    assert excluded == ["chb01_21"]
    assert not (_DIR / "chb01_21_early_warning_evidence.json").exists()


def test_records_carry_the_review_only_claim_boundary() -> None:
    """Every sealed record keeps the review-only evidence-mapping disclaimer."""
    detectors = _evidence(_LED_SEIZURE)["detectors"]
    for record in detectors.values():
        assert "technical evidence-mapping artefact" in record["disclaimer"]
