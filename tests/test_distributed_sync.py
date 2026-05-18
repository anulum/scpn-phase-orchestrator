# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — distributed phase synchronisation tests

from __future__ import annotations

import json

import numpy as np
import pytest

from scpn_phase_orchestrator.runtime.distributed.sync import (
    DistributedSyncConfig,
    PhaseGossipNode,
    PhaseSyncMessage,
    simulate_lossy_phase_gossip,
)


def test_phase_sync_message_round_trips_with_digest() -> None:
    message = PhaseSyncMessage.from_phases(
        node_id="edge-a",
        sequence=7,
        phases=np.array([0.1, 1.2, 6.1]),
        wall_time_s=123.0,
    )

    encoded = message.to_wire()
    decoded = PhaseSyncMessage.from_wire(encoded)

    assert decoded == message
    assert json.loads(encoded.decode("utf-8"))["digest"] == message.digest


def test_phase_gossip_rejects_stale_duplicate_and_wrong_dimension() -> None:
    node = PhaseGossipNode(DistributedSyncConfig(node_id="edge-a", n_oscillators=3))
    accepted = PhaseSyncMessage.from_phases(
        node_id="edge-b",
        sequence=1,
        phases=np.array([0.2, 0.4, 0.6]),
        wall_time_s=1.0,
    )
    duplicate = PhaseSyncMessage.from_phases(
        node_id="edge-b",
        sequence=1,
        phases=np.array([0.3, 0.5, 0.7]),
        wall_time_s=1.1,
    )
    wrong_dimension = PhaseSyncMessage.from_phases(
        node_id="edge-c",
        sequence=1,
        phases=np.array([0.1, 0.2]),
        wall_time_s=1.0,
    )

    assert node.ingest(accepted.to_wire()).accepted
    assert not node.ingest(duplicate.to_wire()).accepted
    assert "stale" in node.ingest(duplicate.to_wire()).reason
    assert not node.ingest(wrong_dimension.to_wire()).accepted
    assert "n_oscillators" in node.ingest(wrong_dimension.to_wire()).reason


def test_phase_gossip_applies_bounded_circular_correction() -> None:
    node = PhaseGossipNode(
        DistributedSyncConfig(
            node_id="edge-a",
            n_oscillators=2,
            phase_blend=0.5,
            max_phase_step_rad=0.2,
        )
    )
    local = np.array([0.0, 2.0 * np.pi - 0.05])
    peer = PhaseSyncMessage.from_phases(
        node_id="edge-b",
        sequence=1,
        phases=np.array([1.0, 0.10]),
        wall_time_s=1.0,
    )
    node.ingest(peer.to_wire())

    corrected = node.synchronise(local, now_s=1.2)
    delta = np.abs(np.angle(np.exp(1j * (corrected - local))))

    assert np.all(delta <= 0.2 + 1e-12)
    assert corrected[0] > local[0]
    assert corrected[1] > local[1] or corrected[1] < 0.2


def test_lossy_phase_gossip_reduces_pairwise_phase_error() -> None:
    initial = {
        "edge-a": np.array([0.0, 0.1, 0.2]),
        "edge-b": np.array([1.4, 1.5, 1.6]),
        "edge-c": np.array([2.4, 2.5, 2.6]),
    }

    result = simulate_lossy_phase_gossip(
        initial,
        rounds=18,
        config=DistributedSyncConfig(
            node_id="template",
            n_oscillators=3,
            phase_blend=0.35,
            max_phase_step_rad=0.18,
        ),
        drop_edges={("edge-c", "edge-a")},
    )

    assert result.initial_mean_pairwise_error > result.final_mean_pairwise_error
    assert result.accepted_messages > 0
    assert result.rejected_messages == 0
    assert set(result.final_phases) == set(initial)


def test_phase_gossip_fails_closed_on_tampered_wire_message() -> None:
    message = PhaseSyncMessage.from_phases(
        node_id="edge-b",
        sequence=1,
        phases=np.array([0.2, 0.4]),
        wall_time_s=1.0,
    )
    raw = json.loads(message.to_wire().decode("utf-8"))
    raw["phases"][0] = 0.9

    node = PhaseGossipNode(DistributedSyncConfig(node_id="edge-a", n_oscillators=2))
    result = node.ingest(json.dumps(raw).encode("utf-8"))

    assert not result.accepted
    assert "digest" in result.reason


def test_phase_sync_config_rejects_unphysical_limits() -> None:
    with pytest.raises(ValueError, match="phase_blend"):
        DistributedSyncConfig(node_id="edge-a", n_oscillators=2, phase_blend=1.5)
    with pytest.raises(ValueError, match="max_phase_step_rad"):
        DistributedSyncConfig(
            node_id="edge-a",
            n_oscillators=2,
            max_phase_step_rad=0.0,
        )
