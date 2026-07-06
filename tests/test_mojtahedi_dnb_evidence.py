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


def _reproduces(fresh: Any, committed: Any, *, rel: float = 1.0e-9) -> bool:
    """Return whether ``fresh`` reproduces ``committed`` up to a float tolerance.

    Every non-float field (labels, counts, decisions) must match exactly; float fields
    match within ``rel`` — far tighter than any meaningful drift, but loose enough to
    absorb the last-ULP differences between BLAS backends (``np.polyfit`` on macOS
    Accelerate vs Linux OpenBLAS), which shift a raw slope in its final bit.
    """
    if isinstance(fresh, float) or isinstance(committed, float):
        return fresh == pytest.approx(committed, rel=rel, abs=1.0e-12)
    if isinstance(fresh, dict):
        return (
            isinstance(committed, dict)
            and fresh.keys() == committed.keys()
            and all(_reproduces(fresh[key], committed[key], rel=rel) for key in fresh)
        )
    if isinstance(fresh, list):
        return (
            isinstance(committed, list)
            and len(fresh) == len(committed)
            and all(
                _reproduces(a, b, rel=rel)
                for a, b in zip(fresh, committed, strict=True)
            )
        )
    return bool(fresh == committed)


def test_committed_reproduces_from_a_fresh_run(payload: dict[str, Any]) -> None:
    # The embedded-summary capstone is self-contained, so a fresh run reproduces the
    # committed artefact — proof the sealed evidence was not drifted or edited. The
    # content hash is bit-exact only on the reference platform (a raw slope's final bit
    # is BLAS-dependent), so the numeric fields are compared within a floating-point
    # tolerance and every decision — the alarm counts and the permutation p-value —
    # reproduces exactly.
    fresh = evaluate_mojtahedi()
    fresh_body = {key: value for key, value in fresh.items() if key != "content_hash"}
    committed_body = {k: v for k, v in payload.items() if k != "content_hash"}
    assert _reproduces(fresh_body, committed_body)
    assert fresh["permutation_significance"]["observed_led"] == 1
    assert fresh["lineages"][0]["alarmed"] is True
