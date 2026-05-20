# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for multiverse branch risk gate

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from scpn_phase_orchestrator.supervisor.multiverse_risk import (
    MultiverseRiskThresholds,
    evaluate_multiverse_branch_risk,
)


def _realistic_manifest() -> dict[str, Any]:
    """Return a manifest with mixed accepted and rejected branch candidates."""
    return {
        "branches": [
            {
                "branch_id": "b-low-risk",
                "final_R": 0.94,
                "mean_R": 0.91,
                "min_R": 0.70,
                "max_R": 0.98,
                "action_count": 8,
                "topology_edge_count": 13,
                "branch_hash": "b1",
            },
            {
                "branch_id": "b-edge-heavy",
                "final_R": 0.96,
                "mean_R": 0.89,
                "min_R": 0.88,
                "max_R": 0.99,
                "action_count": 3,
                "topology_scale": 0.4,
                "branch_hash": "b2",
            },
            {
                "branch_id": "b-rejected",
                "final_R": 0.60,
                "mean_R": 0.52,
                "min_R": 0.50,
                "max_R": 0.70,
                "action_count": 16,
                "topology_edge_count": 15,
                "branch_hash": "b3",
            },
        ]
    }


class TestMultiverseRiskGate:
    def test_evaluate_multiverse_risk_gate_reports_counts_and_reasoning(self) -> None:
        thresholds = MultiverseRiskThresholds(
            min_mean_R=0.7,
            min_final_R=0.8,
            max_action_count=10,
            max_topology_edge_count=13,
            max_topology_scale=0.5,
        )
        report = evaluate_multiverse_branch_risk(_realistic_manifest(), thresholds)

        assert report.approved_count == 2
        assert report.rejected_count == 1
        assert report.safest_branch_id == "b-low-risk"
        assert report.safest_branch_hash == "b1"
        assert {
            reason
            for branch in report.branch_decisions
            if not branch.approved
            for reason in branch.rejection_reasons
        } == {
            "action_count_exceeds_limit",
            "topology_edge_count_exceeds_limit",
            "final_R_below_minimum",
            "mean_R_below_minimum",
        }
        assert "topology_edge_count_exceeds_limit" in report.rejection_reasons
        assert "final_R_below_minimum" in report.rejection_reasons

    def test_approved_branch_without_edge_count_uses_topology_scale_and_becomes_safest(
        self,
    ) -> None:
        thresholds = MultiverseRiskThresholds(
            min_mean_R=0.7,
            min_final_R=0.8,
            max_action_count=10,
            max_topology_scale=1.0,
        )
        manifest = {
            "branches": [
                {
                    "branch_id": "scale-safe",
                    "final_R": 0.90,
                    "mean_R": 0.86,
                    "min_R": 0.81,
                    "max_R": 0.94,
                    "action_count": 2,
                    "topology_scale": 0.9,
                    "branch_hash": "hs",
                },
                {
                    "branch_id": "edge-safe",
                    "final_R": 0.95,
                    "mean_R": 0.86,
                    "min_R": 0.82,
                    "max_R": 0.96,
                    "action_count": 1,
                    "topology_edge_count": 20,
                    "branch_hash": "he",
                },
            ]
        }

        report = evaluate_multiverse_branch_risk(manifest, thresholds)
        assert report.approved_count == 2
        assert report.safest_branch_id == "edge-safe"
        assert report.safest_branch_hash == "he"

    def test_report_hash_is_deterministic(self) -> None:
        manifest = _realistic_manifest()

        first = evaluate_multiverse_branch_risk(manifest)
        second = evaluate_multiverse_branch_risk(manifest)

        assert first.report_hash == second.report_hash
        assert len(first.report_hash) == 64

    def test_report_is_json_safe_and_non_actuating(self) -> None:
        thresholds = MultiverseRiskThresholds()
        report = evaluate_multiverse_branch_risk(_realistic_manifest(), thresholds)
        record = report.to_audit_record()

        assert isinstance(record, dict)
        assert record["non_actuating"] is True
        assert record["execution_disabled"] is True
        assert (
            record["claim_boundary"]
            == "counterfactual_branch_risk_gate_not_live_actuation"
        )
        assert "actions_to_apply" not in record
        assert "control_actions" not in record
        assert "branch_decisions" in record
        assert all(isinstance(branch, dict) for branch in record["branch_decisions"])
        json.loads(json.dumps(record, allow_nan=False))

    @pytest.mark.parametrize(
        ("bad_manifest", "message"),
        [
            ("not-a-mapping", "manifest must be a mapping"),
            ({"branches": "not-a-list"}, "must be a list or tuple"),
            ({"branches": []}, "must be non-empty"),
            ({"branches": [{}]}, "branch_id"),
            (
                {
                    "branches": [
                        {
                            "branch_id": "bad",
                            "final_R": float("nan"),
                            "mean_R": 1.0,
                            "min_R": 0.2,
                            "max_R": 0.2,
                            "action_count": 1,
                            "topology_edge_count": 1,
                            "branch_hash": "x",
                        }
                    ]
                },
                "finite real",
            ),
            (
                {
                    "branches": [
                        {
                            "branch_id": "bad",
                            "final_R": 1.0,
                            "mean_R": 1.0,
                            "min_R": 0.2,
                            "max_R": 0.2,
                            "action_count": 1,
                            "branch_hash": "x",
                        }
                    ]
                },
                "topology_edge_count or topology_scale",
            ),
        ],
    )
    def test_fail_closed_on_invalid_manifests(
        self, bad_manifest: Any, message: str
    ) -> None:
        manifest: Mapping[str, Any] | str
        manifest = bad_manifest if isinstance(bad_manifest, str) else bad_manifest

        with pytest.raises(ValueError, match=message):
            evaluate_multiverse_branch_risk(manifest)
