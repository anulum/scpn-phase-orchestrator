# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Studio strange-loop panel tests

"""Studio facade contract tests for the strange-loop review panel."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import cast

import pytest

import scpn_phase_orchestrator.studio as studio
from scpn_phase_orchestrator.supervisor import evaluate_strange_loop_drift_scenarios


def _records() -> list[dict[str, object]]:
    """Return production strange-loop drift scenario audit records."""
    return [
        result.to_audit_record() for result in evaluate_strange_loop_drift_scenarios()
    ]


def _record() -> dict[str, object]:
    """Return one production strange-loop drift scenario audit record."""
    return _records()[0]


def _copy_mapping(payload: dict[str, object]) -> dict[str, object]:
    """Return a mutable JSON-like mapping copy."""
    return cast("dict[str, object]", deepcopy(payload))


def test_strange_loop_panel_renders_offline_scenario_evidence() -> None:
    """The public Studio facade renders passive strange-loop review evidence."""
    records = _records()

    panel = studio.build_strange_loop_studio_panel(records)

    assert panel["panel_kind"] == "studio_strange_loop_panel"
    assert panel["supervisor"] == "strange_loop"
    assert panel["claim_boundary"] == "strange_loop_drift_review_not_live_actuation"
    assert panel["non_actuating"] is True
    assert panel["execution_disabled"] is True
    assert panel["actuation_permitted"] is False
    assert panel["scenario_count"] == len(records)
    assert panel["passed_count"] == len(records)
    assert panel["failed_scenario_ids"] == []
    assert panel["triggered_modes"] == (
        "control_loop_oscillation",
        "over_control",
        "policy_drift",
        "stable",
    )
    assert panel["maxima"]["drift_score"] == pytest.approx(
        max(cast("float", record["max_drift_score"]) for record in records)
    )
    assert panel["maxima"]["oscillation_score"] == pytest.approx(
        max(cast("float", record["max_oscillation_score"]) for record in records)
    )
    assert panel["maxima"]["overcontrol_score"] == pytest.approx(
        max(cast("float", record["max_overcontrol_score"]) for record in records)
    )
    assert panel["minima"]["control_coherence"] == pytest.approx(
        min(cast("float", record["min_control_coherence"]) for record in records)
    )
    assert "actions_to_apply" not in panel
    assert "control_actions" not in panel
    decoded_panel = json.loads(json.dumps(panel, allow_nan=False))
    assert decoded_panel["panel_kind"] == panel["panel_kind"]


def test_strange_loop_panel_reports_failed_expected_trigger_evidence() -> None:
    """Failed trigger checks remain visible as passive review evidence."""
    failed_record = _copy_mapping(_record())
    failed_record["passed_expected_trigger"] = False

    panel = studio.build_strange_loop_studio_panel([failed_record])

    assert panel["passed_count"] == 0
    assert panel["failed_scenario_ids"] == [failed_record["scenario_id"]]
    assert panel["operator_summary"] == (
        "strange-loop scenario review: 0/1 expected triggers passed"
    )
    assert panel["actuation_permitted"] is False


@pytest.mark.parametrize(
    ("records", "match"),
    [
        ("bad", "must be a sequence"),
        ([], "must not be empty"),
        ([42], "must be mappings"),
    ],
)
def test_strange_loop_panel_rejects_malformed_record_sequence(
    records: object,
    match: str,
) -> None:
    """Top-level record sequence validation fails closed before rendering."""
    with pytest.raises(ValueError, match=match):
        studio.build_strange_loop_studio_panel(
            cast("list[dict[str, object]]", records)
        )


@pytest.mark.parametrize(
    ("field_name", "bad_value", "match"),
    [
        ("claim_boundary", "live_actuation", "claim boundary"),
        ("non_actuating", False, "non_actuating"),
        ("passed_expected_trigger", "yes", "passed_expected_trigger"),
        ("execution_disabled", False, "execution_disabled"),
        ("expected_trigger", "unknown", "expected_trigger"),
        ("scenario_hash", "bad", "scenario_hash"),
        ("max_drift_score", True, "finite non-negative real"),
    ],
)
def test_strange_loop_panel_rejects_malformed_record_fields(
    field_name: str,
    bad_value: object,
    match: str,
) -> None:
    """Record boundary, flag, trigger, hash, and score validation fail closed."""
    record = _copy_mapping(_record())
    record[field_name] = bad_value

    with pytest.raises(ValueError, match=match):
        studio.build_strange_loop_studio_panel([record])
