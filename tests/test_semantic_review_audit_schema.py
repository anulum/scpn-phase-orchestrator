# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — semantic review generated-audit-schema guard tests

"""Validation-guard coverage for the generated semantic-review audit schema.

`_validate_generated_audit_schema` enforces that a generated review audit carries
every required field at the right type and that its review gate is non-actuating,
manual-review-gated, and auto-execution-disabled. These tests drive each rejection
path by starting from a well-formed audit record and tampering exactly one field.
"""

from __future__ import annotations

from typing import Any

import pytest

from scpn_phase_orchestrator.binding.semantic.review import (
    _validate_generated_audit_schema,
)


def _valid_audit() -> dict[str, Any]:
    """Return a well-formed generated-review audit record."""
    return {
        "compiler": "scpn-semantic",
        "schema_valid": True,
        "validation_errors": [],
        "intent_boundary": {"declared": True},
        "review_gate": {
            "non_actuating": True,
            "manual_review_required": True,
            "auto_execution_enabled": False,
        },
        "confidence": 0.9,
        "confidence_factors": {"retrieval": 0.9},
        "retrieval_evidence": [],
        "notebook_execution": {"status": "skipped"},
    }


def test_valid_audit_passes() -> None:
    # A well-formed record validates without raising.
    _validate_generated_audit_schema(_valid_audit())


def test_rejects_wrong_typed_required_field() -> None:
    audit = _valid_audit()
    audit["compiler"] = 123
    with pytest.raises(ValueError, match="generated audit schema invalid: compiler"):
        _validate_generated_audit_schema(audit)


def test_rejects_actuating_review_gate() -> None:
    audit = _valid_audit()
    audit["review_gate"]["non_actuating"] = False
    with pytest.raises(ValueError, match="review gate must be non-actuating"):
        _validate_generated_audit_schema(audit)


def test_rejects_review_gate_without_manual_review() -> None:
    audit = _valid_audit()
    audit["review_gate"]["manual_review_required"] = False
    with pytest.raises(ValueError, match="manual review required"):
        _validate_generated_audit_schema(audit)


def test_rejects_review_gate_with_auto_execution() -> None:
    audit = _valid_audit()
    audit["review_gate"]["auto_execution_enabled"] = True
    with pytest.raises(ValueError, match="auto execution disabled"):
        _validate_generated_audit_schema(audit)
