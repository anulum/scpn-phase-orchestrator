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
to an exact matched false-alarm rate, and sealed per detector per seizure. These
tests guard the committed derived artefact without the raw EEG (citation-only, not
redistributed): they recompute every sealed record's content hash to prove it was
not hand-edited, pin the digests of the led detections, assert every detector was
held at or below the target false-alarm rate, and assert the honest
sparse-detection result.
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
#: The two seizures with a leading alarm at the matched false-alarm rate.
_LED = ("chb01_04", "chb01_26")
#: Digests of the leading detections. The pipeline, detectors, and continuous
#: calibration are deterministic, so regenerating the artefact reproduces these;
#: pinning them ties the committed evidence to a fixed derived provenance.
_PINNED = {
    "chb01_04": {
        "critical_slowing_down": (
            "afd7dc50b40b5806eaf8e4bd1a01a9f91ffd94211b4e6c25ec3ed09a85afb53f"
        ),
        "synchronisation": (
            "778400e07ea94f4fcdb610248e801e9fe22c5f2fcf7fd9d70a2bf08b3f33592e"
        ),
        "ensemble_weighted": (
            "31509033a2cee6a97aac26fdd988ad39e2a76d251dd5bf77c4ae5054d9b3c8dc"
        ),
    },
    "chb01_26": {
        "critical_slowing_down": (
            "edb417a45708ae7f932109199e4dc35f9d9f7ad4a741138569b989489c3eba1c"
        ),
    },
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


def test_the_led_seizures_pin_their_digests() -> None:
    """The led seizures carry the pinned leading-detection digests."""
    for record_id, pins in _PINNED.items():
        detectors = _evidence(record_id)["detectors"]
        for name, digest in pins.items():
            assert detectors[name]["content_hash"] == digest, f"{record_id}/{name}"
            assert detectors[name]["warning_triggered"] is True
            assert detectors[name]["lead_is_early"] is True


def test_the_other_evaluated_seizures_are_sealed_silences() -> None:
    """No detector leads the four seizures other than the two led records."""
    for record_id in _SEIZURES:
        if record_id in _LED:
            continue
        detectors = _evidence(record_id)["detectors"]
        for name, record in detectors.items():
            assert record["lead_is_early"] is False, f"{record_id}/{name} led"


def test_aggregate_records_the_honest_sparse_verdict(
    aggregate: dict[str, Any],
) -> None:
    """The aggregate carries the sparse-detection verdict and its parameters."""
    assert aggregate["verdict"].startswith("SPARSE DETECTION, NO ROBUST ADVANTAGE")
    assert aggregate["n_null_trials"] == 20
    assert aggregate["segment_seconds"] == 900.0


def test_every_detector_is_held_at_or_below_the_target_false_alarm(
    aggregate: dict[str, Any],
) -> None:
    """The continuous calibration holds each detector at or below the 10 % target."""
    for rate in aggregate["achieved_false_alarm"].values():
        assert rate <= 0.10 + 1.0e-9


def test_no_detector_beats_the_matched_false_alarm(aggregate: dict[str, Any]) -> None:
    """No detector's lead count is significant at the matched false-alarm rate."""
    sig = aggregate["permutation_significance"]
    assert set(sig) == {
        "critical_slowing_down",
        "synchronisation",
        "transition_entropy",
        "ensemble_weighted",
    }
    for detector in sig.values():
        assert detector["p_value"] > 0.05  # none beats chance
    # Critical slowing down leads the most (2/6) yet is still not significant.
    assert sig["critical_slowing_down"]["observed_led"] == 2


def test_early_onset_seizure_is_excluded_not_a_silent_null(
    aggregate: dict[str, Any],
) -> None:
    """``chb01_21`` (onset 327 s) is excluded, not sealed as a silent null."""
    excluded = [entry["record_id"] for entry in aggregate["excluded_seizures"]]
    assert excluded == ["chb01_21"]
    assert not (_DIR / "chb01_21_early_warning_evidence.json").exists()


def test_records_carry_the_review_only_claim_boundary() -> None:
    """Every sealed record keeps the review-only evidence-mapping disclaimer."""
    detectors = _evidence("chb01_04")["detectors"]
    for record in detectors.values():
        assert "technical evidence-mapping artefact" in record["disclaimer"]
