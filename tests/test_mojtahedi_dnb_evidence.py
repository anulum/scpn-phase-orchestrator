# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — committed single-cell DNB early-warning evidence tests

"""Integrity tests for the committed real single-cell DNB early-warning evidence.

`examples/real_data/mojtahedi_fate/` is the fifth-domain proof — and the first molecular
one — that the early-warning design is domain-adaptable: the dynamical-network-biomarker
index screens the Mojtahedi et al. 2016 leukaemic fate bifurcation through the same
matched-false-alarm and label-permutation protocol the four physical domains use. These
tests guard the committed derived artefact: they recompute its content hash to prove it
was not hand-edited, pin the digest, assert the honest single-domain result (one of
three lineages clears the matched operating point, not significant), and confirm a fresh
run of the embedded-summary capstone reproduces the committed artefact byte-for-byte.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from bench.early_warning_dnb import evaluate_mojtahedi
from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash

_ARTEFACT = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "real_data"
    / "mojtahedi_fate"
    / "early_warning_dnb_mojtahedi.json"
)

#: The pipeline is deterministic (seed 0), so regenerating the artefact reproduces this
#: digest; pinning it ties the committed evidence to a fixed derived provenance.
_PINNED_HASH = "353b2e7c6c62252b168047e6a7eb4d8a9881c8dc3e4d85477f3694302fab3f26"


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    """Return the committed sealed DNB evidence record."""
    return json.loads(_ARTEFACT.read_text(encoding="utf-8"))


def test_content_hash_recomputes(payload: dict[str, Any]) -> None:
    # Recompute the seal from the payload minus its hash: a hand-edit breaks it.
    body = {key: value for key, value in payload.items() if key != "content_hash"}
    assert payload["content_hash"] == canonical_record_hash(body)


def test_content_hash_is_pinned(payload: dict[str, Any]) -> None:
    assert payload["content_hash"] == _PINNED_HASH


def test_records_the_honest_single_cell_result(payload: dict[str, Any]) -> None:
    significance = payload["permutation_significance"]
    assert significance["observed_led"] == 1  # only the erythroid arm clears
    assert significance["n_transitions"] == 3
    assert significance["p_value"] > 0.05  # not significant across three lineages
    assert payload["target_false_alarm"] == 0.1
    lineages = {row["lineage_id"]: row for row in payload["lineages"]}
    assert lineages["erythroid_epo"]["alarmed"] is True
    assert lineages["combined_epo_gmcsf"]["alarmed"] is False


def test_committed_artefact_matches_a_fresh_run(payload: dict[str, Any]) -> None:
    # The embedded-summary capstone is self-contained, so a fresh run must reproduce the
    # committed artefact exactly — proof the sealed evidence was not drifted or edited.
    assert evaluate_mojtahedi() == payload
