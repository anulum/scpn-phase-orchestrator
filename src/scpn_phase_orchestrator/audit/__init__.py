# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Audit subsystem

"""Audit logging, event streaming, and deterministic replay entry points.

The audit package owns append-only JSONL records, optional hash-chained event
streams, replay reconstruction, and integrity verification. Public helpers
fail closed on malformed signatures or key material while preserving unsigned
development logs for local reproducibility workflows.
"""

from __future__ import annotations

from scpn_phase_orchestrator.audit.logger import AuditLogger
from scpn_phase_orchestrator.audit.replay import ReplayEngine
from scpn_phase_orchestrator.audit.stream import (
    AuditStreamEvent,
    EventStreamWriter,
    read_event_stream,
    verify_event_stream_integrity,
)

__all__ = [
    "AuditLogger",
    "AuditStreamEvent",
    "EventStreamWriter",
    "ReplayEngine",
    "read_event_stream",
    "verify_event_stream_integrity",
]
