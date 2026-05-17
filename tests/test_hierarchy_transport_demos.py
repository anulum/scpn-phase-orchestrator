# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hierarchy transport demos

from __future__ import annotations

from domainpacks.cardiac_rhythm.hierarchy_transport_demo import (
    build_cardiac_transport_envelopes,
    run_demo as run_cardiac_transport_demo,
)
from domainpacks.edge_consensus_nchannel.hierarchy_transport_demo import (
    NODES,
    build_edge_consensus_transport_envelopes,
    run_demo as run_edge_transport_demo,
)
from domainpacks.power_grid.hierarchy_transport_demo import (
    build_power_grid_transport_envelopes,
    run_demo as run_power_grid_transport_demo,
)


def _assert_boundary_consistency(payload: dict[str, object]) -> None:
    assert payload["network_opened"] is True
    assert payload["accepted_count"] == len(payload["accepted_names"])
    for boundary in ("rest_boundary", "websocket_frame", "jsonl_replay"):
        audit = payload[boundary]
        assert audit["boundary"] in {"rest_boundary", "websocket_frame", "jsonl_replay"}


def _assert_boundary_fields(audit: dict[str, object], expected_count: int) -> None:
    assert audit["accepted_count"] == expected_count
    assert audit["rejected_count"] == 0
    assert audit["status"] == "accepted"
    assert isinstance(audit["watermarks"], dict)


def test_power_grid_transport_demo_runs_all_boundaries() -> None:
    payload = run_power_grid_transport_demo()
    envelopes = build_power_grid_transport_envelopes()

    assert payload["domainpack"] == "power_grid"
    assert payload["scenario"] == "hierarchy_summary_transport_replay"
    assert payload["channel_count"] == 2
    assert payload["accepted_names"] == [
        envelope.to_audit_record()["summary"]["name"] for envelope in envelopes
    ]
    assert payload["accepted_count"] == len(envelopes)

    assert payload["rest_boundary"]["boundary"] == "rest_boundary"
    assert payload["websocket_frame"]["boundary"] == "websocket_frame"
    assert payload["jsonl_replay"]["boundary"] == "jsonl_replay"

    _assert_boundary_fields(payload["rest_boundary"], expected_count=2)
    _assert_boundary_fields(payload["websocket_frame"], expected_count=2)
    _assert_boundary_fields(payload["jsonl_replay"], expected_count=2)

    assert payload["rest_boundary"]["watermarks"] == {"grid-edge-11": 11, "grid-edge-12": 12}
    assert payload["websocket_frame"]["watermarks"] == {"grid-edge-11": 11, "grid-edge-12": 12}
    assert payload["jsonl_replay"]["watermarks"] == {"grid-edge-11": 11, "grid-edge-12": 12}
    _assert_boundary_consistency(payload)


def test_cardiac_transport_demo_reports_replayed_frames() -> None:
    payload = run_cardiac_transport_demo()
    envelopes = build_cardiac_transport_envelopes()

    assert payload["domainpack"] == "cardiac_rhythm"
    assert payload["scenario"] == "hierarchy_summary_transport_replay"
    assert payload["channel_count"] == 2
    assert payload["accepted_names"] == [
        envelope.to_audit_record()["summary"]["name"] for envelope in envelopes
    ]
    assert payload["accepted_count"] == len(envelopes)

    _assert_boundary_fields(payload["rest_boundary"], expected_count=2)
    _assert_boundary_fields(payload["websocket_frame"], expected_count=2)
    _assert_boundary_fields(payload["jsonl_replay"], expected_count=2)
    assert payload["rest_boundary"]["watermarks"] == {
        "cardiac-edge-21": 21,
        "cardiac-edge-22": 22,
    }
    assert payload["jsonl_replay"]["watermarks"] == {
        "cardiac-edge-21": 21,
        "cardiac-edge-22": 22,
    }
    assert payload["websocket_frame"]["watermarks"] == {
        "cardiac-edge-21": 21,
        "cardiac-edge-22": 22,
    }
    _assert_boundary_consistency(payload)


def test_edge_consensus_transport_demo_reports_node_level_replay() -> None:
    payload = run_edge_transport_demo()
    envelopes = build_edge_consensus_transport_envelopes()

    assert payload["domainpack"] == "edge_consensus_nchannel"
    assert payload["scenario"] == "heterogeneous_node_transport_replay"
    assert payload["channel_count"] == 6
    assert payload["nodes"] == list(NODES)
    assert payload["accepted_names"] == [
        envelope.to_audit_record()["summary"]["name"] for envelope in envelopes
    ]
    assert payload["accepted_count"] == len(envelopes)

    _assert_boundary_fields(payload["rest_boundary"], expected_count=3)
    _assert_boundary_fields(payload["websocket_frame"], expected_count=3)
    _assert_boundary_fields(payload["jsonl_replay"], expected_count=3)
    assert payload["rest_boundary"]["watermarks"] == {
        "edge-node-31": 31,
        "edge-node-32": 32,
        "edge-node-33": 33,
    }
    assert payload["websocket_frame"]["watermarks"] == {
        "edge-node-31": 31,
        "edge-node-32": 32,
        "edge-node-33": 33,
    }
    assert payload["jsonl_replay"]["watermarks"] == {
        "edge-node-31": 31,
        "edge-node-32": 32,
        "edge-node-33": 33,
    }
    _assert_boundary_consistency(payload)
