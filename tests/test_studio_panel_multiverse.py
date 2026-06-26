# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Studio multiverse counterfactual panel tests

"""Studio facade contract tests for the multiverse counterfactual panel."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import cast

import numpy as np
import pytest

import scpn_phase_orchestrator.studio as studio
from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.supervisor.multiverse import (
    simulate_multiverse_counterfactual_branches,
)
from scpn_phase_orchestrator.supervisor.multiverse_risk import (
    MultiverseRiskThresholds,
    evaluate_multiverse_branch_risk,
)


def _production_multiverse_records(
    *,
    thresholds: MultiverseRiskThresholds | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Return production rollout and risk-gate audit records for Studio review."""
    phases = np.array([0.10, 1.20, 2.40], dtype=np.float64)
    omegas = np.array([0.05, -0.02, 0.01], dtype=np.float64)
    baseline_k = np.array(
        [[0.0, 0.15, 0.15], [0.15, 0.0, 0.15], [0.15, 0.15, 0.0]],
        dtype=np.float64,
    )
    baseline_alpha = np.zeros((3, 3), dtype=np.float64)
    manifest = simulate_multiverse_counterfactual_branches(
        phases=phases,
        omegas=omegas,
        baseline_k=baseline_k,
        baseline_alpha=baseline_alpha,
        branch_action_sets=(
            (),
            (ControlAction("K", "global", 0.25, 1.0, "coupling review"),),
        ),
        horizon=8,
        dt=0.02,
    ).to_audit_record()
    risk = evaluate_multiverse_branch_risk(
        manifest,
        thresholds
        or MultiverseRiskThresholds(
            min_mean_R=0.0,
            min_final_R=0.0,
            max_action_count=0,
        ),
    ).to_audit_record()
    return manifest, risk


def _manual_manifest() -> dict[str, object]:
    """Return a compact valid multiverse manifest for validation mutations."""
    return {
        "schema_name": "multiverse_counterfactual_rollout",
        "schema_version": "0.1.0",
        "branch_count": 1,
        "horizon": 4,
        "backend": "numpy_vectorized",
        "non_actuating": True,
        "execution_disabled": True,
        "claim_boundary": "counterfactual_branch_rollout_not_live_actuation",
        "manifest_hash": "a" * 64,
        "branch_records": [
            {
                "branch_id": "safe",
                "branch_hash": "b" * 64,
                "action_count": 0,
                "action_labels": [],
                "topology_edge_count": 2,
                "topology_scale": 0.3,
                "final_R": 0.8,
                "mean_R": 0.7,
                "min_R": 0.6,
                "max_R": 0.9,
                "final_psi": 0.1,
            }
        ],
    }


def _manual_risk() -> dict[str, object]:
    """Return a compact valid multiverse risk report for validation mutations."""
    return {
        "schema_name": "multiverse_branch_risk_gate",
        "schema_version": "0.1.0",
        "branch_count": 1,
        "approved_count": 1,
        "rejected_count": 0,
        "safest_branch_id": "safe",
        "safest_branch_hash": "b" * 64,
        "rejection_reasons": [],
        "claim_boundary": "counterfactual_branch_risk_gate_not_live_actuation",
        "non_actuating": True,
        "execution_disabled": True,
        "report_hash": "c" * 64,
        "branch_decisions": [
            {
                "branch_id": "safe",
                "branch_hash": "b" * 64,
                "final_R": 0.8,
                "mean_R": 0.7,
                "min_R": 0.6,
                "max_R": 0.9,
                "action_count": 0,
                "topology_edge_count": 2,
                "topology_scale": 0.3,
                "approved": True,
                "rejection_reasons": [],
            }
        ],
    }


def _branch_records(manifest: dict[str, object]) -> list[dict[str, object]]:
    """Return the manifest branch-record list with a strict test-time type."""
    return cast("list[dict[str, object]]", manifest["branch_records"])


def _risk_decisions(risk: dict[str, object]) -> list[dict[str, object]]:
    """Return the risk-decision list with a strict test-time type."""
    return cast("list[dict[str, object]]", risk["branch_decisions"])


def _copy_mapping(payload: dict[str, object]) -> dict[str, object]:
    """Return a mutable JSON-like copy for negative-path mutations."""
    return cast("dict[str, object]", deepcopy(payload))


def test_multiverse_panel_joins_rollout_and_risk_gate_evidence() -> None:
    """The public Studio facade renders joined rollout/risk review evidence."""
    manifest, risk = _production_multiverse_records()

    panel = studio.build_multiverse_counterfactual_studio_panel(manifest, risk)

    assert panel["panel_kind"] == "studio_multiverse_counterfactual_panel"
    assert panel["claim_boundary"] == "counterfactual_branch_rollout_not_live_actuation"
    assert panel["risk_claim_boundary"] == (
        "counterfactual_branch_risk_gate_not_live_actuation"
    )
    assert panel["non_actuating"] is True
    assert panel["execution_disabled"] is True
    assert panel["actuation_permitted"] is False
    assert panel["branch_count"] == 2
    assert panel["approved_count"] == 1
    assert panel["rejected_count"] == 1
    assert panel["safest_branch_id"] == "branch_000"
    assert panel["rejected_branch_ids"] == ["branch_001"]
    assert panel["branch_rows"][0]["risk_approved"] is True
    assert panel["branch_rows"][1]["risk_approved"] is False
    assert panel["branch_rows"][1]["risk_rejection_reasons"] == [
        "action_count_exceeds_limit"
    ]
    assert panel["coherence_range"]["minimum"] <= panel["coherence_range"]["maximum"]
    assert panel["coherence_range"]["final_minimum"] <= (
        panel["coherence_range"]["final_maximum"]
    )
    assert panel["operator_summary"] == (
        "multiverse branch review: 1/2 branches approved"
    )
    assert "actions_to_apply" not in panel
    assert "control_actions" not in panel
    decoded_panel = json.loads(json.dumps(panel, allow_nan=False))
    assert decoded_panel["panel_kind"] == panel["panel_kind"]
    assert len(decoded_panel["branch_rows"]) == len(panel["branch_rows"])


def test_multiverse_panel_preserves_all_rejected_review_boundary() -> None:
    """All-rejected risk reports remain renderable but never become executable."""
    manifest, risk = _production_multiverse_records(
        thresholds=MultiverseRiskThresholds(min_mean_R=1.0, min_final_R=1.0)
    )

    panel = studio.build_multiverse_counterfactual_studio_panel(manifest, risk)

    assert panel["approved_count"] == 0
    assert panel["rejected_count"] == 2
    assert panel["safest_branch_id"] is None
    assert panel["safest_branch_hash"] is None
    assert panel["rejected_branch_ids"] == ["branch_000", "branch_001"]
    assert panel["actuation_permitted"] is False
    assert panel["operator_summary"] == (
        "multiverse branch review: 0/2 branches approved"
    )


@pytest.mark.parametrize(
    ("manifest", "match"),
    [
        ("not-a-mapping", "multiverse manifest must be a mapping"),
        ({**_manual_manifest(), "schema_name": "wrong"}, "schema_name"),
        ({**_manual_manifest(), "backend": "qpu_live"}, "backend"),
        ({**_manual_manifest(), "non_actuating": False}, "non_actuating"),
        ({**_manual_manifest(), "execution_disabled": False}, "execution_disabled"),
        ({**_manual_manifest(), "claim_boundary": "live_control"}, "claim boundary"),
        ({**_manual_manifest(), "branch_records": "bad"}, "branch_records"),
        ({**_manual_manifest(), "branch_records": []}, "non-empty"),
        ({**_manual_manifest(), "branch_records": [42]}, "entries"),
        ({**_manual_manifest(), "branch_count": 2}, "branch_count"),
    ],
)
def test_multiverse_panel_rejects_malformed_rollout_manifest_shape(
    manifest: object,
    match: str,
) -> None:
    """Rollout-level schema, boundary, and branch-count violations fail closed."""
    with pytest.raises(ValueError, match=match):
        studio.build_multiverse_counterfactual_studio_panel(
            cast("dict[str, object]", manifest),
            _manual_risk(),
        )


@pytest.mark.parametrize(
    ("field_name", "bad_value", "match"),
    [
        ("branch_id", "safe", "unique branch_id"),
        ("action_labels", ["too", "many"], "action_count"),
        ("min_R", 0.75, "min_R <= mean_R <= max_R"),
        ("final_R", 0.95, "contain final_R"),
        ("branch_hash", "bad", "branch_hash"),
        ("topology_edge_count", -1, "topology_edge_count"),
        ("topology_scale", -0.1, "topology_scale"),
    ],
)
def test_multiverse_panel_rejects_malformed_rollout_branch_records(
    field_name: str,
    bad_value: object,
    match: str,
) -> None:
    """Branch record validation catches unsafe or internally inconsistent rows."""
    manifest = _copy_mapping(_manual_manifest())
    branch = dict(_branch_records(manifest)[0])
    branch[field_name] = bad_value
    if field_name == "branch_id":
        manifest["branch_count"] = 2
        manifest["branch_records"] = [_branch_records(manifest)[0], branch]
    else:
        manifest["branch_records"] = [branch]

    with pytest.raises(ValueError, match=match):
        studio.build_multiverse_counterfactual_studio_panel(manifest, _manual_risk())


@pytest.mark.parametrize(
    ("risk", "match"),
    [
        ("not-a-mapping", "multiverse risk report must be a mapping"),
        ({**_manual_risk(), "schema_name": "wrong"}, "schema_name"),
        ({**_manual_risk(), "non_actuating": False}, "non_actuating"),
        ({**_manual_risk(), "execution_disabled": False}, "execution_disabled"),
        ({**_manual_risk(), "claim_boundary": "live_gate"}, "claim boundary"),
        ({**_manual_risk(), "branch_decisions": "bad"}, "branch_decisions"),
        ({**_manual_risk(), "branch_decisions": []}, "non-empty"),
        ({**_manual_risk(), "branch_decisions": [42]}, "entries"),
        ({**_manual_risk(), "branch_count": 2}, "branch_count"),
        ({**_manual_risk(), "approved_count": 0, "rejected_count": 0}, "sum"),
        ({**_manual_risk(), "approved_count": 0, "rejected_count": 1}, "approved"),
        (
            {**_manual_risk(), "approved_count": 1, "rejected_count": 1},
            "sum",
        ),
    ],
)
def test_multiverse_panel_rejects_malformed_risk_report_shape(
    risk: object,
    match: str,
) -> None:
    """Risk-report schema, boundary, and count violations fail closed."""
    with pytest.raises(ValueError, match=match):
        studio.build_multiverse_counterfactual_studio_panel(
            _manual_manifest(),
            cast("dict[str, object]", risk),
        )


@pytest.mark.parametrize(
    ("field_name", "bad_value", "match"),
    [
        ("branch_id", "safe", "unique branch_id"),
        ("min_R", 0.75, "R interval"),
        ("branch_hash", "bad", "branch_hash"),
        ("action_count", -1, "action_count"),
        ("topology_edge_count", -1, "topology_edge_count"),
        ("topology_scale", -0.1, "topology_scale"),
        ("approved", "yes", "approved"),
        ("rejection_reasons", "bad", "rejection_reasons"),
    ],
)
def test_multiverse_panel_rejects_malformed_risk_decisions(
    field_name: str,
    bad_value: object,
    match: str,
) -> None:
    """Risk decision validation catches invalid rows before table rendering."""
    risk = _copy_mapping(_manual_risk())
    decision = dict(_risk_decisions(risk)[0])
    decision[field_name] = bad_value
    if field_name == "branch_id":
        risk["branch_count"] = 2
        risk["approved_count"] = 2
        risk["branch_decisions"] = [_risk_decisions(risk)[0], decision]
    else:
        risk["branch_decisions"] = [decision]

    with pytest.raises(ValueError, match=match):
        studio.build_multiverse_counterfactual_studio_panel(_manual_manifest(), risk)


def test_multiverse_panel_rejects_rejected_count_drift() -> None:
    """Rejected-count drift is caught even when total counts still sum correctly."""
    risk = _copy_mapping(_manual_risk())
    risk["approved_count"] = 1
    risk["rejected_count"] = 0
    decision = dict(_risk_decisions(risk)[0])
    decision["approved"] = False
    risk["branch_decisions"] = [decision]

    with pytest.raises(ValueError, match="rejected"):
        studio.build_multiverse_counterfactual_studio_panel(_manual_manifest(), risk)


def test_multiverse_panel_accepts_missing_optional_risk_topology_scale() -> None:
    """A risk decision may omit topology scale when edge counts carry the risk."""
    risk = _copy_mapping(_manual_risk())
    decision = dict(_risk_decisions(risk)[0])
    decision["topology_scale"] = None
    risk["branch_decisions"] = [decision]

    panel = studio.build_multiverse_counterfactual_studio_panel(
        _manual_manifest(),
        risk,
    )

    assert panel["branch_rows"][0]["topology_scale"] == 0.3
    assert panel["branch_rows"][0]["risk_approved"] is True


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("branch_hash", "d" * 64),
        ("final_R", 0.7),
        ("mean_R", 0.65),
        ("min_R", 0.5),
        ("max_R", 0.95),
        ("action_count", 1),
    ],
)
def test_multiverse_panel_rejects_rollout_risk_join_mismatch(
    field_name: str,
    bad_value: object,
) -> None:
    """Risk decisions must replay the matching rollout branch facts exactly."""
    risk = _copy_mapping(_manual_risk())
    decision = dict(_risk_decisions(risk)[0])
    decision[field_name] = bad_value
    risk["branch_decisions"] = [decision]

    with pytest.raises(ValueError, match="risk decision"):
        studio.build_multiverse_counterfactual_studio_panel(_manual_manifest(), risk)
