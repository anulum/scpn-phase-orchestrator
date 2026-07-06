# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — sealed real-PSML advisory example integrity

"""Integrity tests over the committed real-PSML grid early-warning advisory example.

These pin the sealed streaming-to-decision example without any raw PMU data: the
``content_hash`` recomputes from the committed payload, the alarm genuinely crossed the
certified threshold, the record is structurally non-actuating and carries the honest
recall, and it is the live-deployment case — sealed with no ground-truth onset, so no
lead is claimed. The raw voltages are never redistributed; only the derived advisory is.
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
    / "grid_early_warning_advisory.json"
)


@pytest.fixture(scope="module")
def advisory() -> dict[str, Any]:
    return json.loads(_ARTEFACT.read_text(encoding="utf-8"))


def test_content_hash_recomputes(advisory: dict[str, Any]) -> None:
    sealed = dict(advisory)
    stored = sealed.pop("content_hash")
    assert canonical_record_hash(sealed) == stored


def test_the_alarm_crossed_the_certified_threshold(advisory: dict[str, Any]) -> None:
    assert advisory["detector"] == "grid_modal_growth_stream"
    assert advisory["aggregation"] == "focal"
    assert advisory["r2_gate"] == 0.5
    # the sealed alarm is a genuine crossing of the certified matched-false-alarm point
    assert advisory["growth_rate"] >= advisory["growth_rate_threshold"]
    assert advisory["verdict"] == "grid_early_warning_advisory_raised"


def test_the_advisory_is_structurally_non_actuating(advisory: dict[str, Any]) -> None:
    assert advisory["non_actuating"] is True
    assert advisory["actuating"] is False


def test_the_honest_recall_is_sealed(advisory: dict[str, Any]) -> None:
    # 11 of 45 held-out transitions led — the recall an operator must see
    assert advisory["certified_recall"] == pytest.approx(11 / 45)
    assert advisory["certified_false_alarm"] < 0.12
    assert advisory["certified_recall"] < 0.5  # a reason to look, not proof


def test_it_is_the_live_deployment_case_with_no_claimed_lead(
    advisory: dict[str, Any],
) -> None:
    # a live operator does not know the onset, so no lead is claimed
    assert advisory["transition_onset_sample"] is None
    assert advisory["lead_samples"] is None
    assert advisory["lead_is_early"] is False
