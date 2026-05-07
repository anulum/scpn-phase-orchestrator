# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Power-grid hierarchy sync demo tests

from __future__ import annotations

from domainpacks.power_grid.hierarchy_sync_demo import (
    build_power_grid_sync_envelopes,
    power_grid_sync_states,
    run_demo,
)


def test_power_grid_hierarchy_sync_demo_emits_parent_plan() -> None:
    payload = run_demo()

    assert payload["domainpack"] == "power_grid"
    assert payload["plan"]["hierarchy"] == "power_grid_edge_cloud_summary_sync"
    assert payload["plan"]["audit_scope"] == "reduced_child_summaries_only"
    assert len(payload["accepted"]) == 2
    assert payload["rejected"] == []
    assert {record["summary"]["name"] for record in payload["accepted"]} == {
        "generation_area",
        "demand_renewable_area",
    }
    parent = payload["plan"]["parent"]
    assert 0.0 <= parent["R"] <= 1.0
    assert parent["regime_id"].startswith("hierarchical_")


def test_power_grid_hierarchy_sync_envelopes_are_reduced_and_ordered() -> None:
    envelopes = build_power_grid_sync_envelopes()

    assert [envelope.sequence for envelope in envelopes] == [11, 12]
    assert [envelope.source_node for envelope in envelopes] == [
        "grid-edge-11",
        "grid-edge-12",
    ]
    for envelope in envelopes:
        record = envelope.to_audit_record()
        summary = record["summary"]
        assert "raw_phases" not in summary
        assert "time_series" not in summary
        assert "knm" not in summary
        assert summary["metadata"]["source"] == "pmu_replay"


def test_power_grid_sync_states_have_distinct_region_dispersion() -> None:
    states = power_grid_sync_states()

    assert set(states) == {"generation_area", "demand_renewable_area"}
    assert states["generation_area"].size == states["demand_renewable_area"].size
    assert states["generation_area"].std() < states["demand_renewable_area"].std()
