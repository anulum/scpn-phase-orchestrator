# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — run-record assurance evidence tests

"""Tests for deriving assurance evidence from a simulation run record.

``build_run_evidence`` maps the trust-relevant fields of a serialised
``SimulationResult`` (audit-stream integrity and conformal admission-gate
decisions) into evidence items, and emits nothing for surfaces that did not run.
"""

from __future__ import annotations

import scpn_phase_orchestrator.assurance.run_evidence as _run_evidence
from scpn_phase_orchestrator.assurance import (
    AUDIT_LOGGING,
    CONFORMAL_GATE,
    build_run_evidence,
)

assert _run_evidence is not None


def test_builds_integrity_evidence_when_present() -> None:
    items = build_run_evidence(
        {"audit_event_stream_integrity": {"integrity_ok": True, "verified_records": 5}}
    )

    assert len(items) == 1
    assert items[0].evidence_id == "run-audit-stream-integrity"
    assert items[0].category == AUDIT_LOGGING
    assert items[0].record["integrity_ok"] is True


def test_builds_conformal_evidence_when_gate_active() -> None:
    items = build_run_evidence(
        {
            "conformal_admission_total": 8,
            "conformal_admission_rejections": 2,
            "last_conformal_admission": {"admitted": False},
        }
    )

    assert len(items) == 1
    assert items[0].evidence_id == "run-conformal-admission"
    assert items[0].category == CONFORMAL_GATE
    assert items[0].record["conformal_admission_total"] == 8
    assert items[0].record["conformal_admission_rejections"] == 2


def test_builds_both_when_both_present() -> None:
    items = build_run_evidence(
        {
            "audit_event_stream_integrity": {"integrity_ok": True},
            "conformal_admission_total": 3,
        }
    )

    assert {item.evidence_id for item in items} == {
        "run-audit-stream-integrity",
        "run-conformal-admission",
    }


def test_no_evidence_when_gate_inactive_and_no_integrity() -> None:
    assert (
        build_run_evidence(
            {"conformal_admission_total": 0, "audit_event_stream_integrity": None}
        )
        == ()
    )


def test_boolean_conformal_total_is_not_evidence() -> None:
    # A bool is an int subclass; it must not be read as a gate-active count.
    assert build_run_evidence({"conformal_admission_total": True}) == ()


def test_empty_record_yields_no_evidence() -> None:
    assert build_run_evidence({}) == ()
