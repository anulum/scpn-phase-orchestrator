# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Cardiac hierarchy sync demo tests

from __future__ import annotations

from domainpacks.cardiac_rhythm.hierarchy_sync_demo import (
    build_cardiac_sync_envelopes,
    cardiac_sync_states,
    run_demo,
)


def test_cardiac_hierarchy_sync_demo_emits_parent_plan() -> None:
    payload = run_demo()

    assert payload["domainpack"] == "cardiac_rhythm"
    assert payload["plan"]["hierarchy"] == "cardiac_edge_cloud_summary_sync"
    assert payload["plan"]["audit_scope"] == "reduced_child_summaries_only"
    assert len(payload["accepted"]) == 2
    assert payload["rejected"] == []
    assert {record["summary"]["name"] for record in payload["accepted"]} == {
        "pacemaker_atrial_axis",
        "ventricular_recovery_axis",
    }
    parent = payload["plan"]["parent"]
    assert 0.0 <= parent["R"] <= 1.0
    assert parent["regime_id"].startswith("hierarchical_")


def test_cardiac_hierarchy_sync_envelopes_are_reduced_and_ordered() -> None:
    envelopes = build_cardiac_sync_envelopes()

    assert [envelope.sequence for envelope in envelopes] == [21, 22]
    assert [envelope.source_node for envelope in envelopes] == [
        "cardiac-edge-21",
        "cardiac-edge-22",
    ]
    for envelope in envelopes:
        record = envelope.to_audit_record()
        summary = record["summary"]
        assert "raw_phases" not in summary
        assert "time_series" not in summary
        assert "knm" not in summary
        assert summary["metadata"]["source"] == "conduction_replay"


def test_cardiac_sync_states_have_distinct_conduction_axes() -> None:
    states = cardiac_sync_states()

    assert set(states) == {"pacemaker_atrial_axis", "ventricular_recovery_axis"}
    assert states["pacemaker_atrial_axis"].size == 5
    assert states["ventricular_recovery_axis"].size == 5
    assert (
        states["pacemaker_atrial_axis"].mean()
        < states["ventricular_recovery_axis"].mean()
    )
