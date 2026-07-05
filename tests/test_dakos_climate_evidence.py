# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — committed real Dakos climate early-warning evidence tests

"""Integrity tests for the committed real palaeoclimate early-warning evidence.

`examples/real_data/dakos_climate_transitions/` is the fourth-domain proof — and the
first single-series one — that the early-warning design is domain-adaptable: the
single-series critical-slowing-down harness screens the eight Dakos et al. 2008
palaeoclimate abrupt-transition records the same way the multi-node suite screens EEG,
ECG and grid signals. These tests guard the committed derived artefact without the raw
proxy series (citation-only, not redistributed): they recompute every sealed record's
content hash to prove it was not hand-edited, pin the digest of the one led transition
and of a sealed silence, and assert the honest single-indicator result — critical
slowing down leads one of six evaluated transitions at a matched false-alarm rate.
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
    / "dakos_climate_transitions"
)
#: The one led transition (Eocene–Oligocene greenhouse end) and a sealed silence
#: (Younger Dryas termination). The pipeline and detector are deterministic, so
#: regenerating the artefact reproduces these digests; pinning them ties the committed
#: evidence to a fixed derived provenance.
_PINNED = {
    "eocene_oligocene_greenhouse_end": {
        "digest": "c3a310f963737c2b238b554a50df5bc8ccdbeaca180d80d192a83cf77ca0bb7a",
        "warning_triggered": True,
        "lead_is_early": True,
    },
    "younger_dryas_termination": {
        "digest": "330d25eba5657c09bf88ddad3b7c8cb0c20bdee116a962be75b332f16d3602bd",
        "warning_triggered": False,
        "lead_is_early": False,
    },
}


@pytest.fixture(scope="module")
def aggregate() -> dict[str, Any]:
    """Return the committed aggregate results record."""
    return json.loads(
        (_DIR / "early_warning_leadtime_climate_results.json").read_text(
            encoding="utf-8"
        )
    )


def _evidence(record_id: str) -> dict[str, Any]:
    """Return the committed sealed evidence for one transition."""
    return json.loads(
        (_DIR / f"{record_id}_early_warning_evidence.json").read_text(encoding="utf-8")
    )


def test_every_detector_seal_recomputes(aggregate: dict[str, Any]) -> None:
    """Each record's ``content_hash`` matches its canonical payload."""
    for transition in aggregate["transitions"]:
        record = _evidence(transition["record_id"])["detector"]
        payload = copy.deepcopy(record)
        sealed = payload.pop("content_hash")
        assert canonical_record_hash(payload) == sealed, transition["record_id"]


def test_the_led_and_silent_records_pin_their_digests() -> None:
    """The pinned led transition and sealed silence carry their fixed digests."""
    for record_id, pin in _PINNED.items():
        record = _evidence(record_id)["detector"]
        assert record["content_hash"] == pin["digest"], record_id
        assert record["warning_triggered"] is pin["warning_triggered"]
        assert record["lead_is_early"] is pin["lead_is_early"]


def test_aggregate_records_the_honest_single_indicator_verdict(
    aggregate: dict[str, Any],
) -> None:
    """The aggregate carries the single-indicator verdict and its parameters."""
    assert aggregate["verdict"].startswith("SINGLE-INDICATOR DETECTION")
    assert aggregate["benchmark"] == "early_warning_leadtime_climate"
    assert len(aggregate["transitions"]) == 6
    assert aggregate["n_null_trials"] == 50
    assert aggregate["segment_samples"] == 60
    excluded = {row["record_id"] for row in aggregate["excluded_records"]}
    assert excluded == {"glaciation_IV_termination", "north_africa_desertification"}


def test_the_detector_is_held_at_or_below_the_target_false_alarm(
    aggregate: dict[str, Any],
) -> None:
    """The continuous calibration holds the detector at or below the 10 % target."""
    for rate in aggregate["achieved_false_alarm"].values():
        assert rate <= 0.10 + 1.0e-9


def test_exactly_one_transition_is_led(aggregate: dict[str, Any]) -> None:
    """Critical slowing down leads exactly one of the six evaluated transitions."""
    led = [t for t in aggregate["transitions"] if t["lead_years"] is not None]
    assert len(led) == 1
    assert led[0]["record_id"] == "eocene_oligocene_greenhouse_end"
    assert led[0]["lead_years"] > 0.0


def test_the_lead_count_is_not_significant(aggregate: dict[str, Any]) -> None:
    """The one lead is consistent with chance at the matched false-alarm rate."""
    sig = aggregate["permutation_significance"]
    assert sig["observed_led"] == 1
    assert sig["n_transitions"] == 6
    assert sig["n_permutations"] == 10000
    assert sig["expected_led"] == pytest.approx(0.643, abs=1.0e-2)
    # The permutation p-value is far above 0.05 — the single lead does not beat the
    # matched false-alarm rate, so detection here is at chance.
    assert sig["p_value"] > 0.05


def test_records_carry_the_review_only_claim_boundary() -> None:
    """Every sealed record keeps the review-only evidence-mapping disclaimer."""
    for record_id in _PINNED:
        record = _evidence(record_id)["detector"]
        assert "technical evidence-mapping artefact" in record["disclaimer"]
