# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — gossip peers are active only inside the timeout window

"""A peer state counts only within ``peer_timeout_s`` of now, on either side.

A single message from a peer whose clock ran an hour ahead used to steer the
local phases for that hour, and expired states were never dropped from the
peer table.
"""

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.runtime.distributed.sync import (
    DistributedSyncConfig,
    PhaseGossipNode,
    PhaseSyncMessage,
)


def _node() -> PhaseGossipNode:
    return PhaseGossipNode(
        DistributedSyncConfig(node_id="local", n_oscillators=3, peer_timeout_s=5.0)
    )


def _message(peer: str, sequence: int, wall_time_s: float) -> bytes:
    return PhaseSyncMessage.from_phases(
        node_id=peer, sequence=sequence, phases=np.full(3, 1.0), wall_time_s=wall_time_s
    ).to_wire()


def test_peer_dated_far_in_the_future_does_not_steer() -> None:
    node = _node()
    assert node.ingest(_message("peer-skewed", 1, 3600.0)).accepted
    local = np.zeros(3)
    for now in (10.0, 1000.0, 3500.0):
        assert np.allclose(node.synchronise(local, now_s=now), local)
    assert node.peer_count == 0


def test_small_forward_skew_inside_the_window_still_steers() -> None:
    node = _node()
    assert node.ingest(_message("peer-a", 1, 12.0)).accepted
    local = np.zeros(3)
    moved = node.synchronise(local, now_s=10.0)
    assert not np.allclose(moved, local)
    assert node.peer_count == 1


def test_expired_peer_is_dropped_but_its_watermark_is_kept() -> None:
    node = _node()
    assert node.ingest(_message("peer-a", 5, 10.0)).accepted
    node.synchronise(np.zeros(3), now_s=12.0)
    assert node.peer_count == 1
    node.synchronise(np.zeros(3), now_s=20.0)
    assert node.peer_count == 0
    assert node.peer_sequences == {"peer-a": 5}
    replay = node.ingest(_message("peer-a", 5, 20.0))
    assert not replay.accepted
    assert replay.reason == "stale or duplicate sequence"
    assert node.ingest(_message("peer-a", 6, 20.0)).accepted
    assert node.peer_count == 1
