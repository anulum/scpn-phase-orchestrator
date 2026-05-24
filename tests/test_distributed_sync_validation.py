# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Distributed sync validation contracts

"""
Validation and serialisation contracts for runtime.distributed.sync messages and
ingest audit records.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.runtime.distributed.sync import (
    DistributedSyncConfig,
    GossipIngestResult,
    PhaseGossipNode,
    PhaseSyncMessage,
    simulate_lossy_phase_gossip,
)


class TestDistributedSyncValidation:
    def test_config_rejects_malformed_inputs(self) -> None:
        with pytest.raises(ValueError, match="node_id must contain only letters"):
            DistributedSyncConfig(node_id="bad node", n_oscillators=4)

        with pytest.raises(ValueError, match="n_oscillators must be positive"):
            DistributedSyncConfig(node_id="edge-a", n_oscillators=0)

        with pytest.raises(ValueError, match="protocol_version must be positive"):
            DistributedSyncConfig(node_id="edge-a", n_oscillators=4, protocol_version=0)

        with pytest.raises(
            ValueError, match="peer_timeout_s must be positive and finite"
        ):
            DistributedSyncConfig(
                node_id="edge-a",
                n_oscillators=4,
                peer_timeout_s=0.0,
            )

    def test_from_wire_fails_closed_on_malformed_mapping(self) -> None:
        node = PhaseGossipNode(DistributedSyncConfig(node_id="edge-a", n_oscillators=2))
        result = node.ingest({"node_id": "edge-b", "sequence": 1})

        assert not result.accepted
        assert "phase sync message kind mismatch" in result.reason

    def test_sync_audit_records_are_deterministic(self) -> None:
        result = simulate_lossy_phase_gossip(
            {
                "edge-a": np.array([0.0, 0.1, 0.2]),
                "edge-b": np.array([1.1, 1.2, 1.3]),
            },
            rounds=2,
            config=DistributedSyncConfig(
                node_id="template",
                n_oscillators=3,
                phase_blend=0.25,
                max_phase_step_rad=0.2,
            ),
        )

        audit = result.to_audit_record()
        assert audit["kind"] == "lossy_phase_gossip_replay"
        assert audit["nodes"] == ["edge-a", "edge-b"]
        assert audit["rounds"] == 2
        assert isinstance(audit["final_mean_pairwise_error"], float)


class TestGossipIngestRecordContract:
    def test_gossip_ingest_audit_record_keeps_reason_and_sequence(self) -> None:
        record = GossipIngestResult(
            accepted=False,
            reason="protocol_version mismatch",
            peer_id="edge-b",
            sequence=7,
        ).to_audit_record()

        assert record == {
            "accepted": False,
            "reason": "protocol_version mismatch",
            "peer_id": "edge-b",
            "sequence": 7,
        }

    def test_phase_sync_message_round_trip_wire_contract(self) -> None:
        message = PhaseSyncMessage.from_phases(
            node_id="edge-a",
            sequence=3,
            phases=np.array([0.1, 1.2]),
            wall_time_s=12.5,
        )
        wire_a = message.to_wire()
        wire_b = message.to_wire()

        assert wire_a == wire_b
