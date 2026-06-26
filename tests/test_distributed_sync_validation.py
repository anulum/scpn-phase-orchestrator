# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — distributed phase synchronisation validation

"""Fail-closed validation contracts for distributed phase synchronisation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pytest

from scpn_phase_orchestrator.runtime.distributed.sync import (
    DistributedSyncConfig,
    PhaseGossipNode,
    PhaseSyncMessage,
    simulate_lossy_phase_gossip,
)


def _wire_record() -> dict[str, Any]:
    message = PhaseSyncMessage.from_phases(
        node_id="edge-b",
        sequence=3,
        phases=np.array([0.2, 0.4]),
        wall_time_s=1.0,
    )
    return cast(dict[str, Any], json.loads(message.to_wire().decode("utf-8")))


def test_phase_sync_message_rejects_invalid_json_syntax() -> None:
    with pytest.raises(json.JSONDecodeError):
        PhaseSyncMessage.from_wire("{not-json")

    node = PhaseGossipNode(DistributedSyncConfig(node_id="edge-a", n_oscillators=2))
    result = node.ingest(b"{not-json")

    assert not result.accepted
    assert "Expecting property name" in result.reason


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ("[1,2,3]", "payload must decode to a mapping"),
        (object(), "payload must be bytes, string, or decoded mapping"),
        (_wire_record() | {"kind": "wrong"}, "phase sync message kind mismatch"),
        (_wire_record() | {"n_oscillators": 3}, "n_oscillators must match"),
        (_wire_record() | {"node_id": ""}, "node_id must be a non-empty string"),
        (_wire_record() | {"sequence": 1.5}, "sequence must be an integer"),
        (
            _wire_record() | {"protocol_version": 1.5},
            "protocol_version must be an integer",
        ),
        (_wire_record() | {"wall_time_s": "soon"}, "wall_time_s must be"),
        (_wire_record() | {"wall_time_s": float("inf")}, "wall_time_s must be finite"),
        (_wire_record() | {"phases": []}, "phases must be a non-empty"),
        (_wire_record() | {"phases": [0.1, float("nan")]}, "phases must contain"),
        (_wire_record() | {"digest": ""}, "digest must be a non-empty string"),
    ],
)
def test_phase_sync_message_rejects_malformed_payloads(
    payload: object,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        PhaseSyncMessage.from_wire(cast(bytes | str | Mapping[str, Any], payload))


def test_phase_sync_config_rejects_weight_timeout_and_node_id_guards() -> None:
    with pytest.raises(ValueError, match="peer_timeout_s"):
        DistributedSyncConfig(node_id="edge-a", n_oscillators=2, peer_timeout_s=0.0)
    with pytest.raises(ValueError, match="local_weight"):
        DistributedSyncConfig(node_id="edge-a", n_oscillators=2, local_weight=0.0)
    with pytest.raises(ValueError, match="peer_weight"):
        DistributedSyncConfig(
            node_id="edge-a",
            n_oscillators=2,
            peer_weight=float("nan"),
        )
    with pytest.raises(ValueError, match="node_id"):
        DistributedSyncConfig(node_id="edge a", n_oscillators=2)


def test_phase_gossip_rejects_protocol_mismatch_and_self_messages() -> None:
    node = PhaseGossipNode(DistributedSyncConfig(node_id="edge-a", n_oscillators=2))
    protocol_mismatch = PhaseSyncMessage.from_phases(
        node_id="edge-b",
        sequence=1,
        phases=np.array([0.1, 0.2]),
        wall_time_s=1.0,
        protocol_version=2,
    )
    self_message = node.observe_local(np.array([0.3, 0.4]), wall_time_s=1.0)

    protocol_result = node.ingest(protocol_mismatch.to_wire())
    self_result = node.ingest(self_message.to_wire())

    assert protocol_result.to_audit_record() == {
        "accepted": False,
        "reason": "protocol_version mismatch",
        "peer_id": "edge-b",
        "sequence": 1,
    }
    assert not self_result.accepted
    assert self_result.reason == "self message ignored"
    assert node.peer_sequences == {}
    assert node.peer_count == 0


def test_phase_gossip_audit_records_and_idle_sync_copy() -> None:
    node = PhaseGossipNode(DistributedSyncConfig(node_id="edge-a", n_oscillators=2))
    local = np.array([0.1, 0.2])

    outbound = node.observe_local(local, wall_time_s=2.0)
    synced = node.synchronise(local, now_s=2.0)

    assert outbound.sequence == 1
    assert np.array_equal(synced, local)
    assert synced is not local
    assert node.to_audit_record() == {
        "kind": "phase_gossip_node",
        "node_id": "edge-a",
        "local_sequence": 1,
        "peer_sequences": {},
        "peer_count": 0,
    }


def test_phase_gossip_rejects_non_finite_synchronise_timestamp() -> None:
    node = PhaseGossipNode(DistributedSyncConfig(node_id="edge-a", n_oscillators=2))

    with pytest.raises(ValueError, match="now_s must be finite"):
        node.synchronise(np.array([0.1, 0.2]), now_s=float("nan"))


@pytest.mark.parametrize(
    ("phases", "match"),
    [
        (np.array([0.1]), r"phases must have shape \(2,\)"),
        (np.array([0.1, float("nan")]), "phases must contain finite values"),
    ],
)
def test_phase_gossip_observe_local_rejects_bad_phase_arrays(
    phases: np.ndarray[Any, np.dtype[np.float64]],
    match: str,
) -> None:
    node = PhaseGossipNode(DistributedSyncConfig(node_id="edge-a", n_oscillators=2))

    with pytest.raises(ValueError, match=match):
        node.observe_local(phases)


def test_phase_gossip_drops_expired_peers_from_sync() -> None:
    node = PhaseGossipNode(
        DistributedSyncConfig(
            node_id="edge-a",
            n_oscillators=2,
            peer_timeout_s=0.25,
        )
    )
    stale = PhaseSyncMessage.from_phases(
        node_id="edge-b",
        sequence=1,
        phases=np.array([2.0, 2.5]),
        wall_time_s=1.0,
    )
    local = np.array([0.1, 0.2])

    assert node.ingest(stale.to_wire()).accepted
    synced = node.synchronise(local, now_s=2.0)

    assert np.array_equal(synced, local)


def test_lossy_phase_gossip_accepts_single_node_and_reports_audit_record() -> None:
    initial = {"edge-a": np.array([0.0, 0.1])}
    result = simulate_lossy_phase_gossip(
        initial,
        rounds=2,
        config=DistributedSyncConfig(node_id="template", n_oscillators=2),
    )

    assert result.initial_mean_pairwise_error == 0.0
    assert result.final_mean_pairwise_error == 0.0
    assert result.accepted_messages == 0
    assert result.rejected_messages == 0
    assert result.to_audit_record() == {
        "kind": "lossy_phase_gossip_replay",
        "rounds": 2,
        "initial_mean_pairwise_error": 0.0,
        "final_mean_pairwise_error": 0.0,
        "accepted_messages": 0,
        "rejected_messages": 0,
        "nodes": ["edge-a"],
    }


def test_lossy_phase_gossip_rejects_invalid_replay_contracts() -> None:
    config = DistributedSyncConfig(node_id="template", n_oscillators=2)

    with pytest.raises(ValueError, match="rounds must be positive"):
        simulate_lossy_phase_gossip(
            {"edge-a": np.array([0.0, 0.1])},
            rounds=0,
            config=config,
        )
    with pytest.raises(ValueError, match="initial_phases"):
        simulate_lossy_phase_gossip({}, rounds=1, config=config)
    with pytest.raises(ValueError, match="node_id"):
        simulate_lossy_phase_gossip(
            {"edge a": np.array([0.0, 0.1])},
            rounds=1,
            config=config,
        )
    with pytest.raises(ValueError, match=r"edge-a\.phases must have shape"):
        simulate_lossy_phase_gossip(
            {"edge-a": np.array([0.0])},
            rounds=1,
            config=config,
        )
