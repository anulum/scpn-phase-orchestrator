# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — run-record assurance evidence tests

"""Tests for deriving assurance evidence from a simulation run record.

``build_run_evidence`` maps the trust-relevant fields of a serialised
``SimulationResult`` (audit-stream integrity, conformal admission-gate decisions,
and the closed-loop control-safety envelope) into evidence items, and emits
nothing for surfaces that did not run.
"""

from __future__ import annotations

import scpn_phase_orchestrator.assurance.run_evidence as _run_evidence
from scpn_phase_orchestrator.assurance import (
    AUDIT_LOGGING,
    CONFORMAL_GATE,
    CONTROL_ENVELOPE,
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


def test_builds_control_envelope_when_policy_enabled() -> None:
    items = build_run_evidence(
        {
            "policy_enabled": True,
            "control_mode": "projected",
            "action_total": 4,
            "boundary_violation_total": 0,
            "final_regime": "nominal",
        }
    )

    assert len(items) == 1
    item = items[0]
    assert item.evidence_id == "run-control-envelope"
    assert item.category == CONTROL_ENVELOPE
    assert item.record["control_mode"] == "projected"
    assert item.record["action_total"] == 4
    assert item.record["boundary_violation_total"] == 0
    assert item.record["final_regime"] == "nominal"
    assert item.record["policy_enabled"] is True


def test_no_control_envelope_when_policy_disabled() -> None:
    assert (
        build_run_evidence(
            {"policy_enabled": False, "control_mode": "projected", "action_total": 0}
        )
        == ()
    )


def test_no_control_envelope_when_policy_flag_absent() -> None:
    # An open-loop record omits ``policy_enabled``; control fields alone must not
    # fabricate an envelope item.
    assert (
        build_run_evidence({"control_mode": "projected", "boundary_violation_total": 2})
        == ()
    )


def test_truthy_non_true_policy_flag_is_not_control_envelope() -> None:
    # Only the bool ``True`` counts as an active policy; a truthy int does not.
    assert build_run_evidence({"policy_enabled": 1, "action_total": 3}) == ()


def test_builds_all_three_surfaces_together() -> None:
    items = build_run_evidence(
        {
            "audit_event_stream_integrity": {"integrity_ok": True},
            "conformal_admission_total": 3,
            "policy_enabled": True,
            "control_mode": "projected",
            "action_total": 2,
            "boundary_violation_total": 1,
            "final_regime": "nominal",
        }
    )

    assert {item.evidence_id for item in items} == {
        "run-audit-stream-integrity",
        "run-conformal-admission",
        "run-control-envelope",
    }
    assert {item.category for item in items} == {
        AUDIT_LOGGING,
        CONFORMAL_GATE,
        CONTROL_ENVELOPE,
    }
