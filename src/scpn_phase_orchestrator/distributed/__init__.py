# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — distributed synchronisation protocols

"""Transport-neutral distributed phase synchronisation API.

The package exports canonical phase-vector gossip messages, deterministic
lossy replay helpers, and local node ingestion logic. It performs digest,
shape, sequence, timeout, and bounded-correction validation, but it never opens
sockets or starts a live transport by itself.
"""

from __future__ import annotations

from scpn_phase_orchestrator.distributed.sync import (
    DistributedSyncConfig,
    GossipIngestResult,
    LossyGossipReplayResult,
    PhaseGossipNode,
    PhaseSyncMessage,
    simulate_lossy_phase_gossip,
)

__all__ = [
    "DistributedSyncConfig",
    "GossipIngestResult",
    "LossyGossipReplayResult",
    "PhaseGossipNode",
    "PhaseSyncMessage",
    "simulate_lossy_phase_gossip",
]
