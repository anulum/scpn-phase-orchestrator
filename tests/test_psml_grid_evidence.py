# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — committed real PSML grid early-warning evidence tests

"""Integrity tests for the committed real PSML power-grid early-warning evidence.

`examples/real_data/psml_grid_oscillation/` is the third-domain proof that the
early-warning design is domain-adaptable: the *same* suite and matched-false-alarm
harness that screened scalp-EEG seizures and cardiac AF onsets screen a growing
power-grid oscillation, sealed per detector per instability. These tests guard the
committed derived artefact without the raw co-simulation data (citation-only, not
redistributed): they recompute every sealed record's content hash to prove it was
not hand-edited, pin the digests of two led instabilities, and assert the honest
sparse-detection result — critical slowing down leads the most, the fusion no more.
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
    / "psml_grid_oscillation"
)
#: Digests of two led instabilities. The pipeline and detectors are deterministic,
#: so regenerating the artefact reproduces these; pinning them ties the committed
#: evidence to a fixed derived provenance.
_PINNED = {
    "row_176": {
        "critical_slowing_down": (
            "41088a7dc67629c4007a37670c0d36a7003750c98f5debbf6d4e0746026c71fb"
        ),
        "transition_entropy": (
            "ab7732b98fff8438c27c2c4cbcc0f6560039e0f89d45ec46161f807a5dfc4e10"
        ),
        "ensemble_weighted": (
            "468a869ec3b6358d259772e0877260a59f3f557f383dd6aec6fd48f6e39748da"
        ),
    },
    "row_175": {
        "critical_slowing_down": (
            "8d7e341729d715601ccef41fc9a43bda974ff75923f530471357c714a2f70e60"
        ),
        "ensemble_weighted": (
            "e5e9ffc675d80cf08476157a3ae704d999f906f750c729f3b939ddf1491be4fd"
        ),
    },
}


@pytest.fixture(scope="module")
def aggregate() -> dict[str, Any]:
    """Return the committed aggregate results record."""
    return json.loads(
        (_DIR / "early_warning_leadtime_grid_results.json").read_text(encoding="utf-8")
    )


def _evidence(record_id: str) -> dict[str, Any]:
    """Return the committed sealed evidence for one instability."""
    return json.loads(
        (_DIR / f"{record_id}_early_warning_evidence.json").read_text(encoding="utf-8")
    )


def test_every_detector_seal_recomputes(aggregate: dict[str, Any]) -> None:
    """Each detector record's ``content_hash`` matches its canonical payload."""
    for transition in aggregate["transitions"]:
        detectors = _evidence(transition["record_id"])["detectors"]
        for name, record in detectors.items():
            payload = copy.deepcopy(record)
            sealed = payload.pop("content_hash")
            assert canonical_record_hash(payload) == sealed, (
                f"{transition['record_id']}/{name}"
            )


def test_the_led_instabilities_pin_their_digests() -> None:
    """The two pinned instabilities carry the leading-detection digests."""
    for record_id, pins in _PINNED.items():
        detectors = _evidence(record_id)["detectors"]
        for name, digest in pins.items():
            assert detectors[name]["content_hash"] == digest, f"{record_id}/{name}"
            assert detectors[name]["warning_triggered"] is True
            assert detectors[name]["lead_is_early"] is True


def test_aggregate_records_the_honest_sparse_verdict(
    aggregate: dict[str, Any],
) -> None:
    """The aggregate carries the sparse-detection verdict and its parameters."""
    assert aggregate["verdict"].startswith("SPARSE DETECTION, NO ROBUST ADVANTAGE")
    assert aggregate["n_transitions_found"] == 27
    assert aggregate["n_null_scenarios_found"] == 187
    assert aggregate["n_transitions_evaluated"] == 12
    assert aggregate["n_null_trials"] == 24
    assert aggregate["segment_seconds"] == 2.0


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
    # Critical slowing down leads the most (3/12) yet is still not significant.
    assert sig["critical_slowing_down"]["observed_led"] == 3


def test_critical_slowing_down_leads_and_the_fusion_has_no_advantage(
    aggregate: dict[str, Any],
) -> None:
    """Critical slowing down leads the most instabilities; the fusion leads no more."""
    led = dict.fromkeys(("critical_slowing_down", "synchronisation"), 0)
    entropy_led = 0
    fusion_led = 0
    for transition in aggregate["transitions"]:
        leads = transition["lead_seconds"]
        for name in led:
            if leads.get(name) is not None:
                led[name] += 1
        if leads.get("transition_entropy") is not None:
            entropy_led += 1
        if leads.get("ensemble_weighted") is not None:
            fusion_led += 1
    best_member = max(led["critical_slowing_down"], led["synchronisation"], entropy_led)
    assert led["critical_slowing_down"] == best_member  # CSD is the best member
    assert fusion_led <= best_member  # no robust fusion advantage


def test_records_carry_the_review_only_claim_boundary() -> None:
    """Every sealed record keeps the review-only evidence-mapping disclaimer."""
    detectors = _evidence("row_176")["detectors"]
    for record in detectors.values():
        assert "technical evidence-mapping artefact" in record["disclaimer"]
