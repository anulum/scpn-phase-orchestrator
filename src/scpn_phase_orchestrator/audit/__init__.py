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

from typing import Any

__all__ = [
    "AuditLogger",
    "AuditStreamEvent",
    "EventStreamWriter",
    "ReplayEngine",
    "read_event_stream",
    "verify_event_stream_integrity",
]

_EXPORT_MODULES = {
    "AuditLogger": "scpn_phase_orchestrator.runtime.audit_logger",
    "ReplayEngine": "scpn_phase_orchestrator.runtime.replay",
    "AuditStreamEvent": "scpn_phase_orchestrator.runtime.audit_stream",
    "EventStreamWriter": "scpn_phase_orchestrator.runtime.audit_stream",
    "read_event_stream": "scpn_phase_orchestrator.runtime.audit_stream",
    "verify_event_stream_integrity": "scpn_phase_orchestrator.runtime.audit_stream",
}


def __dir__() -> list[str]:
    """Return module attributes plus lazy public audit exports."""
    return sorted({*globals(), *__all__})


def __getattr__(name: str) -> Any:
    """Lazily expose audit package exports without import cycles."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
