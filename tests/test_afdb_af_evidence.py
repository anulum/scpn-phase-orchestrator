# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — committed real MIT-BIH AFDB early-warning evidence tests

"""Integrity tests for the committed real MIT-BIH AFDB early-warning evidence.

`examples/real_data/afdb_atrial_fibrillation/` is the second-domain proof that the
early-warning design is domain-adaptable: the *same* suite and matched-false-alarm
harness that screened scalp-EEG seizures screen atrial-fibrillation onsets in the
two-lead surface ECG, sealed per detector per onset. These tests guard the
committed derived artefact without the raw ECG (citation-only, not redistributed):
they recompute every sealed record's content hash to prove it was not hand-edited,
pin the digests of the two led onsets, and assert the honest sparse-detection
result — two of six evaluated onsets led, no robust fusion advantage.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash

_DIR = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "real_data"
    / "afdb_atrial_fibrillation"
)
#: The six evaluated AF onsets.
_ONSETS = ("04043", "04048", "04746", "04908", "05091", "07879")
#: The two records that produced a leading alarm at matched false alarm.
_LED = ("04043", "04908")
#: Digests of the leading detections. The pipeline and detectors are
#: deterministic, so regenerating the artefact reproduces these; pinning them ties
#: the committed evidence to a fixed derived provenance.
_PINNED = {
    "04043": {
        "synchronisation": (
            "fa30c57dd3de860db7cf6aab89db832ca63046cd93649ac0420ec0fc081b211e"
        ),
        "ensemble_weighted": (
            "d21b3a7f0c327b364799983110d86012cc840a9a0d179470649d7a7e14417940"
        ),
    },
    "04908": {
        "critical_slowing_down": (
            "bcc926773edd5edc8fd06823317676b445536f4f50ccc3497dacb8f776e93876"
        ),
        "synchronisation": (
            "1899f61ef4fb08415f9ae48c0dd3d62bab7ca2c272f4248d9a35d95e161dcddd"
        ),
        "ensemble_weighted": (
            "f3c728b46d912f0eb1276207f9afe54fa6582bf6c9ee3086e2ce710e8422f5f3"
        ),
    },
}


@pytest.fixture(scope="module")
def aggregate() -> dict[str, Any]:
    """Return the committed aggregate results record."""
    return json.loads(
        (_DIR / "early_warning_leadtime_cardiac_results.json").read_text(
            encoding="utf-8"
        )
    )


def _evidence(record_id: str) -> dict[str, Any]:
    """Return the committed sealed evidence for one AF onset."""
    return json.loads(
        (_DIR / f"{record_id}_early_warning_evidence.json").read_text(encoding="utf-8")
    )


def test_every_detector_seal_recomputes() -> None:
    """Each detector record's ``content_hash`` matches its canonical payload."""
    for record_id in _ONSETS:
        detectors = _evidence(record_id)["detectors"]
        for name, record in detectors.items():
            payload = copy.deepcopy(record)
            sealed = payload.pop("content_hash")
            assert canonical_record_hash(payload) == sealed, f"{record_id}/{name}"


def test_the_led_onsets_pin_their_digests() -> None:
    """The two led onsets carry the pinned leading-detection digests."""
    for record_id, pins in _PINNED.items():
        detectors = _evidence(record_id)["detectors"]
        for name, digest in pins.items():
            assert detectors[name]["content_hash"] == digest, f"{record_id}/{name}"
            assert detectors[name]["warning_triggered"] is True
            assert detectors[name]["lead_is_early"] is True


def test_the_other_onsets_are_sealed_silences() -> None:
    """No detector leads the four onsets other than the two led records."""
    for record_id in _ONSETS:
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
    assert aggregate["matched_false_alarm_thresholds"] == {
        "critical_slowing_down": 5.75,
        "synchronisation": 4.25,
        "transition_entropy": 2.75,
        "ensemble_weighted": 2.5,
    }


def test_the_fusion_has_no_robust_advantage(aggregate: dict[str, Any]) -> None:
    """The fusion leads no more onsets than its best single member."""
    led = dict.fromkeys(("critical_slowing_down", "synchronisation"), 0)
    fusion_led = 0
    for onset in aggregate["af_onsets"]:
        leads = onset["lead_seconds"]
        for name in led:
            if leads.get(name) is not None:
                led[name] += 1
        if leads.get("ensemble_weighted") is not None:
            fusion_led += 1
    assert fusion_led <= max(led.values())


def test_records_carry_the_review_only_claim_boundary() -> None:
    """Every sealed record keeps the review-only evidence-mapping disclaimer."""
    detectors = _evidence(_LED[0])["detectors"]
    for record in detectors.values():
        assert "technical evidence-mapping artefact" in record["disclaimer"]
