# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hierarchy transport smoke demos

from __future__ import annotations

import json

from domainpacks.cardiac_rhythm.hierarchy_transport_demo import (
    run_demo as run_cardiac_transport_demo,
)
from domainpacks.edge_consensus_nchannel.hierarchy_transport_demo import (
    run_demo as run_edge_transport_demo,
)
from domainpacks.power_grid.hierarchy_transport_demo import (
    run_demo as run_power_grid_transport_demo,
)


def _assert_transport_payload(payload: dict[str, object], expected_domain: str) -> None:
    assert payload["domainpack"] == expected_domain
    assert payload["network_opened"] is True
    assert isinstance(payload["accepted_count"], int)
    assert isinstance(payload["accepted_names"], list)
    assert payload["accepted_count"] == len(payload["accepted_names"])
    assert payload["channel_count"] >= 1

    for key in ("rest_boundary", "websocket_frame", "jsonl_replay"):
        boundary = payload[key]
        assert boundary["boundary"] in {
            "rest_boundary",
            "websocket_frame",
            "jsonl_replay",
        }
        assert isinstance(boundary["watermarks"], dict)
        assert boundary["accepted_count"] >= 1
        assert boundary["status"] == "accepted"
        assert isinstance(boundary["ledger"]["accepted"], list)

    # Assert stable JSON serialisation for artifact handoff.
    assert (
        json.loads(json.dumps(payload, sort_keys=True))["domainpack"]
        == expected_domain
    )


def test_power_grid_hierarchy_transport_smoke() -> None:
    payload = run_power_grid_transport_demo()
    _assert_transport_payload(payload, expected_domain="power_grid")
    assert payload["scenario"] == "hierarchy_summary_transport_replay"
    assert payload["channels"] == ["P", "I"]


def test_cardiac_hierarchy_transport_smoke() -> None:
    payload = run_cardiac_transport_demo()
    _assert_transport_payload(payload, expected_domain="cardiac_rhythm")
    assert payload["scenario"] == "hierarchy_summary_transport_replay"
    assert payload["channels"] == ["P", "I"]


def test_edge_consensus_hierarchy_transport_smoke() -> None:
    payload = run_edge_transport_demo()
    _assert_transport_payload(payload, expected_domain="edge_consensus_nchannel")
    assert payload["scenario"] == "heterogeneous_node_transport_replay"
    assert payload["nodes"] == ["leaf_cluster", "regional_gateway", "parent_supervisor"]
