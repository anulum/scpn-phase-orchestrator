# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — assurance evidence from a simulation run record

"""Derive assurance evidence directly from a serialised simulation run record.

A completed run produces a JSON-safe ``SimulationResult.to_record()`` summary.
This module maps the trust-relevant fields of that record — the close-time audit
event-stream integrity result and the conformal twin-confidence admission-gate
decisions — into :class:`~scpn_phase_orchestrator.assurance.evidence.EvidenceItem`
records, so a deployment can assemble a conformity package from a run record
without hand-authoring evidence JSON.

The helper consumes the serialised record (a ``Mapping``), not the
``SimulationResult`` object, so the assurance package stays free of the heavy
runtime/numeric import chain and can run against a persisted run summary. Fields
that are absent or describe an inactive gate produce no evidence item — the
mapping never fabricates evidence for a surface that did not run.
"""

from __future__ import annotations

from collections.abc import Mapping

from scpn_phase_orchestrator.assurance.evidence import (
    AUDIT_LOGGING,
    CONFORMAL_GATE,
    EvidenceItem,
    build_evidence_item,
)


def _is_positive_int(value: object) -> bool:
    """Return whether ``value`` is a positive, non-boolean integer."""
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def build_run_evidence(run_record: Mapping[str, object]) -> tuple[EvidenceItem, ...]:
    """Build assurance evidence from a serialised simulation run record.

    Parameters
    ----------
    run_record:
        A JSON-safe ``SimulationResult.to_record()`` mapping.

    Returns
    -------
    tuple[EvidenceItem, ...]
        Evidence for the trust surfaces the run attests: the audit event-stream
        integrity result (``audit-chain``) when an event stream was written, and
        the conformal admission-gate decisions (``conformal-gate``) when the gate
        scored at least one tick. Empty if the run attests neither.
    """
    items: list[EvidenceItem] = []

    integrity = run_record.get("audit_event_stream_integrity")
    if isinstance(integrity, Mapping):
        items.append(
            build_evidence_item(
                evidence_id="run-audit-stream-integrity",
                category=AUDIT_LOGGING,
                summary="Close-time audit event-stream integrity result for the run",
                record=dict(integrity),
            )
        )

    conformal_total = run_record.get("conformal_admission_total")
    if _is_positive_int(conformal_total):
        items.append(
            build_evidence_item(
                evidence_id="run-conformal-admission",
                category=CONFORMAL_GATE,
                summary="Conformal twin-confidence admission decisions for the run",
                record={
                    "conformal_admission_total": conformal_total,
                    "conformal_admission_rejections": run_record.get(
                        "conformal_admission_rejections"
                    ),
                    "last_conformal_admission": run_record.get(
                        "last_conformal_admission"
                    ),
                },
            )
        )

    return tuple(items)


__all__ = ["build_run_evidence"]
