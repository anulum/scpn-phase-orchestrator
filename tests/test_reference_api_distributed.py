# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - Distributed sync API reference documentation tests

from __future__ import annotations

from pathlib import Path

DISTRIBUTED_REFERENCE = Path("docs/reference/api/distributed.md")


def test_distributed_sync_api_reference_meets_depth_baseline() -> None:
    doc = DISTRIBUTED_REFERENCE.read_text(encoding="utf-8")

    assert len(doc.splitlines()) >= 567


def test_distributed_sync_api_reference_documents_protocol_contracts() -> None:
    doc = DISTRIBUTED_REFERENCE.read_text(encoding="utf-8")
    required_phrases = (
        "DistributedSyncConfig",
        "PhaseSyncMessage",
        "PhaseGossipNode",
        "GossipIngestResult",
        "LossyGossipReplayResult",
        "simulate_lossy_phase_gossip",
        "spo.phase_sync",
        "SHA-256 digest",
        "sequence watermarks",
        "protocol_version mismatch",
        "stale or duplicate sequence",
        "bounded circular correction",
        "finite JSON",
        "transport-neutral",
    )

    for phrase in required_phrases:
        assert phrase in doc
