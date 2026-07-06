# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — EEG modal-growth transfer sealed-evidence integrity

"""Integrity tests over the committed EEG modal-growth transfer artefact.

These pin the honest sealed result without any raw EEG: the ``content_hash`` recomputes
from the committed payload (never from a fresh pipeline run — Welch/BLAS drift across
platforms), and the reported lead counts, the fit-quality gate's false-alarm collapse,
the parameter-grid robustness, and the exploratory-only disclosure are asserted exactly
as sealed. The artefact records that the grid modal-growth moat does *not* transfer to
scalp EEG.
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
    / "chb01_seizures"
    / "seizure_modal_transfer.json"
)


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    return json.loads(_ARTEFACT.read_text(encoding="utf-8"))


def test_content_hash_recomputes(payload: dict[str, Any]) -> None:
    sealed = dict(payload)
    stored = sealed.pop("content_hash")
    assert canonical_record_hash(sealed) == stored


def test_corpus_is_the_six_usable_chb01_seizures(payload: dict[str, Any]) -> None:
    assert payload["corpus"]["n_transitions"] == 6
    assert payload["corpus"]["n_nulls"] == 20
    assert payload["benchmark"] == "seizure_modal_transfer"


@pytest.mark.parametrize("aggregation", ["mean", "focal"])
def test_modal_growth_leads_nothing(payload: dict[str, Any], aggregation: str) -> None:
    # the exponential-growth score is at chance — below the spectral rank trend
    assert payload["modal_growth"][aggregation]["led"] == 0
    assert payload["modal_growth"][aggregation]["significance"]["p_value"] == 1.0


@pytest.mark.parametrize("aggregation", ["mean", "focal"])
def test_the_fit_quality_gate_collapses_the_false_alarm(
    payload: dict[str, Any], aggregation: str
) -> None:
    # the gate, built to reject non-exponential transients, rejects the signal too:
    # every gated score is clamped, so nothing alarms and the false alarm falls to zero
    gated = payload["modal_growth_r2_gated"][aggregation]
    assert gated["led"] == 0
    assert gated["achieved_false_alarm"] == 0.0


def test_spectral_rank_trend_leads_more_but_stays_at_chance(
    payload: dict[str, Any],
) -> None:
    for aggregation in ("mean", "focal"):
        record = payload["spectral_kendall_tau"][aggregation]
        assert record["led"] == 1  # above modal-growth's zero
        assert record["significance"]["p_value"] > 0.05  # yet still at chance


def test_robustness_grid_is_at_chance_everywhere(payload: dict[str, Any]) -> None:
    # the failure is a model mismatch, not a parameter artefact: no window/recency/gate
    # setting lifts the modal focal score off zero leads
    rows = payload["robustness_modal_focal"]
    assert len(rows) == 12  # 3 windows × 2 recency × 2 gate
    assert all(row["led"] == 0 for row in rows)


def test_the_shorter_horizon_is_disclosed_but_not_claimed(
    payload: dict[str, Any],
) -> None:
    exploratory = payload["exploratory_shorter_horizon"]
    assert exploratory["segment_seconds"] == 600.0
    # the shorter horizon does surface a spectral lead, but it is explicitly unclaimed
    assert exploratory["spectral_focal"]["led"] == 3
    assert "not claimed" in exploratory["claim"]


def test_verdict_states_the_non_transfer(payload: dict[str, Any]) -> None:
    assert "does not transfer" in payload["verdict"]
    assert "grid-specific" in payload["verdict"]
