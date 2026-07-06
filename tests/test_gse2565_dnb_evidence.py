# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — committed bulk DNB (GSE2565) early-warning evidence tests

"""Integrity tests for the committed real bulk-DNB early-warning evidence.

`examples/real_data/gse2565_lung/` is the bulk-transcriptomic companion to the
single-cell Mojtahedi proof: the GSE2565 phosgene lung-injury DNB benchmark, tested with
a **selection-controlled** surrogate null that re-selects the module on each shuffled
surrogate. These tests guard the committed derived artefact without the raw expression
matrix (citation-only, not redistributed): they recompute its content hash to prove it
was not hand-edited, pin the digest, and assert the honest result — the phosgene-exposed
arm's DNB rise does not beat the selection-controlled surrogates, barely differing from
the air control.
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
    / "gse2565_lung"
    / "early_warning_dnb_gse2565.json"
)

#: The pipeline is deterministic (seed 0, deterministic module selection), so
#: regenerating from the raw corpus reproduces this digest.
_PINNED_HASH = "e5180ff4a7896acceba7da628207705129c8a22744a6e57e604898c623970b1f"


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    """Return the committed sealed bulk-DNB evidence record."""
    return json.loads(_ARTEFACT.read_text(encoding="utf-8"))


def test_content_hash_recomputes(payload: dict[str, Any]) -> None:
    body = {key: value for key, value in payload.items() if key != "content_hash"}
    assert payload["content_hash"] == canonical_record_hash(body)


def test_content_hash_is_pinned(payload: dict[str, Any]) -> None:
    assert payload["content_hash"] == _PINNED_HASH


def test_records_the_honest_selection_controlled_result(
    payload: dict[str, Any],
) -> None:
    arms = {arm["group"]: arm for arm in payload["arms"]}
    assert set(arms) == {"CG", "Air"}
    exposed = arms["CG"]
    assert exposed["p_value"] > 0.05  # not significant against reselecting surrogates
    assert exposed["alarmed"] is False
    # the exposed rise barely exceeds the control's, both near their surrogate means
    assert exposed["observed_slope"] < exposed["surrogate_p90"]
    assert payload["n_surrogates_per_arm"] == 1000
    assert payload["target_false_alarm"] == 0.1
