# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hierarchical supervisor summaries

"""Reduced-evidence hierarchy summaries, envelopes, ledgers, and consensus.

The hierarchy package enforces a boundary where parent supervisors receive only
bounded child summaries: coherence, phase, regime, confidence, channel, and
metadata. Raw phases, time series, coupling matrices, event payloads, and
actuator targets are rejected from metadata and transport envelopes. The
boundary types and validation live in one core module, with the orchestration
plan, sync transport, and gossip consensus split into their own modules behind a
stable re-export surface. Builders and runtimes are socket-free and return
audit-ready plans or ledgers.
"""

from __future__ import annotations

from .boundary import (
    ChildSupervisorSummary,
    HierarchyEscalation,
    HierarchySyncEnvelope,
)
from .boundary import (
    _child_escalations as _child_escalations,
)
from .boundary import (
    _is_forbidden_hierarchy_key as _is_forbidden_hierarchy_key,
)
from .boundary import (
    _metadata_to_audit_record as _metadata_to_audit_record,
)
from .boundary import (
    _normalise_metadata_value as _normalise_metadata_value,
)
from .boundary import (
    _normalise_previous_sequences as _normalise_previous_sequences,
)
from .boundary import (
    _reject_raw_instance_attributes as _reject_raw_instance_attributes,
)
from .consensus import (
    HierarchyConsensusRound,
    HierarchyConsensusState,
    simulate_hierarchy_gossip_consensus,
)
from .plan import (
    HierarchicalOrchestrationPlan,
    build_hierarchical_orchestration_plan,
)
from .sync import (
    HierarchySyncLedger,
    HierarchyTransportRuntime,
    build_hierarchy_sync_envelope,
    ingest_hierarchy_sync_envelopes,
    load_hierarchy_sync_envelope,
)
from .sync import (
    _load_child_summary as _load_child_summary,
)
from .sync import (
    _load_mapping_record as _load_mapping_record,
)
from .sync import (
    _reject_raw_hierarchy_keys as _reject_raw_hierarchy_keys,
)
from .sync import (
    _reject_unknown_keys as _reject_unknown_keys,
)

__all__ = [
    "ChildSupervisorSummary",
    "HierarchicalOrchestrationPlan",
    "HierarchyConsensusRound",
    "HierarchyConsensusState",
    "HierarchySyncEnvelope",
    "HierarchySyncLedger",
    "HierarchyTransportRuntime",
    "HierarchyEscalation",
    "build_hierarchical_orchestration_plan",
    "build_hierarchy_sync_envelope",
    "ingest_hierarchy_sync_envelopes",
    "load_hierarchy_sync_envelope",
    "simulate_hierarchy_gossip_consensus",
]
