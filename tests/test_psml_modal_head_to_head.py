# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — committed grid modal-vs-generic head-to-head evidence tests

"""Integrity tests for the committed grid modal-vs-generic head-to-head evidence.

`examples/real_data/psml_modal_growth/` is the flagship deliverable: on the real PSML
23-bus corpus, the domain-specific modal envelope-growth detector (focal aggregation,
recency-weighted growth rate) against the whole generic SCPN early-warning suite, on the
identical non-circular disturbance-type split at a matched false alarm. These tests
guard the committed derived artefact without the raw PSML data (citation-only, not
redistributed): they recompute its content hash from the committed payload alone (no
raw re-run, so no cross-platform float drift), pin the digest, and assert the honest
positive result — the modal detector leads far more instability transitions than any
generic member, every one of which is at chance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash

_ARTEFACT = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "real_data"
    / "psml_modal_growth"
    / "grid_modal_head_to_head.json"
)

#: The payload is sealed by canonical_record_hash over its body; recomputing the hash
#: from the committed payload alone proves it was not hand-edited (no raw re-run).
_PINNED_HASH = "bc6895879088b31b763c566aa315f0edcfc842f537e81102cbd5706cf3ef7bf2"


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    """Return the committed sealed head-to-head evidence record."""
    return json.loads(_ARTEFACT.read_text(encoding="utf-8"))


def test_content_hash_recomputes(payload: dict[str, Any]) -> None:
    body = {key: value for key, value in payload.items() if key != "content_hash"}
    assert payload["content_hash"] == canonical_record_hash(body)


def test_content_hash_is_pinned(payload: dict[str, Any]) -> None:
    assert payload["content_hash"] == _PINNED_HASH


def test_corpus_is_the_clean_non_circular_split(payload: dict[str, Any]) -> None:
    corpus = payload["corpus"]
    assert corpus["n_transitions"] == 90
    assert corpus["n_nulls"] == 88
    assert corpus["n_dropped_bad_rate"] == 3  # disclosed, not silently dropped
    assert corpus["sampling_rate_hz"] == pytest.approx(238.095, abs=1e-2)
    assert "disturbance type" in corpus["labelling"]  # non-circular label


def test_operating_point_is_the_validated_winner(payload: dict[str, Any]) -> None:
    operating_point = payload["operating_point"]
    assert operating_point["aggregation"] == "focal"
    assert operating_point["recency_top"] == 3.0
    assert "held-out" in operating_point["selection"]


def test_modal_beats_every_generic_member(payload: dict[str, Any]) -> None:
    modal = payload["modal"]["significance"]
    assert payload["modal"]["detector"] == "modal_envelope_growth_rate_focal"
    assert modal["observed_led"] == 36
    assert modal["n_transitions"] == 90
    assert modal["p_value"] < 0.01  # beats chance decisively
    generic = payload["generic_suite"]
    assert set(generic) == {
        "critical_slowing_down",
        "synchronisation",
        "transition_entropy",
        "ensemble_weighted",
    }
    for record in generic.values():
        assert modal["observed_led"] > record["observed_led"]  # more leads than each
        assert record["p_value"] > 0.05  # every generic member is at chance
    assert "beating" in payload["verdict"]


def test_held_out_validation_is_unbiased_and_significant(
    payload: dict[str, Any],
) -> None:
    held = payload["held_out_validation"]
    assert held["n_transitions"] == 45
    assert held["observed_led"] == 24
    assert held["p_value"] < 0.01  # the unbiased held-out estimate still beats chance
