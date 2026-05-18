# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — runtime distributed synchronisation protocols

"""Runtime namespace for transport-neutral distributed phase synchronisation."""

from __future__ import annotations

from scpn_phase_orchestrator.runtime.distributed.sync import (
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
