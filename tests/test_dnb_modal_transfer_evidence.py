# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — DNB modal-growth transfer sealed-evidence integrity

"""Integrity tests over the committed DNB modal-growth transfer artefact.

These pin the honest sealed characterisation without any raw expression data: the
``content_hash`` recomputes from the committed payload, and the short rising-limb
lengths, the gate keeping every trajectory (its uninformativeness), and the
growth-rate/slope order agreement are asserted exactly as sealed. The artefact records
that the grid modal-growth moat cannot even be *posed* on the DNB early-warning corpora.
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
    / "mojtahedi_fate"
    / "dnb_modal_transfer.json"
)


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    return json.loads(_ARTEFACT.read_text(encoding="utf-8"))


def test_content_hash_recomputes(payload: dict[str, Any]) -> None:
    sealed = dict(payload)
    stored = sealed.pop("content_hash")
    assert canonical_record_hash(sealed) == stored


def test_benchmark_and_resolution_threshold(payload: dict[str, Any]) -> None:
    assert payload["benchmark"] == "dnb_modal_transfer"
    assert payload["min_resolvable_points"] == 6


def test_rising_limbs_are_too_short_to_resolve_a_growth_form(
    payload: dict[str, Any],
) -> None:
    # three single-cell lineages at three points, one bulk arm at four — all below six
    single_cell = payload["single_cell_mojtahedi"]
    assert len(single_cell) == 3
    assert all(record["n_points"] == 3 for record in single_cell)
    assert payload["bulk_gse2565"]["n_points"] == 4
    longest = max(
        record["n_points"] for record in (*single_cell, payload["bulk_gse2565"])
    )
    assert longest < payload["min_resolvable_points"]


def test_the_gate_keeps_every_trajectory(payload: dict[str, Any]) -> None:
    # the point of the finding: on so few points the gate is uninformative — it keeps
    # every monotone rise (R² in [0.5, 1]) and rejects none, so it cannot discriminate
    records = [*payload["single_cell_mojtahedi"], payload["bulk_gse2565"]]
    assert all(record["gate_keeps"] is True for record in records)
    assert all(0.5 <= record["exponential_fit_r2"] <= 1.0 for record in records)


def test_growth_rate_only_re_orders_the_slope(payload: dict[str, Any]) -> None:
    # the exponential growth rate carries no more than the linear slope: the two agree
    # on the ordering of the single-cell lineages, so the transfer adds no information
    single_cell = payload["single_cell_mojtahedi"]
    by_slope = [r["label"] for r in sorted(single_cell, key=lambda r: r["slope"])]
    by_rate = [r["label"] for r in sorted(single_cell, key=lambda r: r["growth_rate"])]
    assert by_slope == by_rate


def test_verdict_states_the_moat_cannot_be_posed(payload: dict[str, Any]) -> None:
    assert "cannot be posed" in payload["verdict"]
    assert "critical-slowing-down" in payload["verdict"]
