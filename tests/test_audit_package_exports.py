# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator audit package facade tests

from __future__ import annotations

import pytest

import scpn_phase_orchestrator.audit as audit


def test_audit_package_lazy_exports_resolve_and_cache_public_contracts() -> None:
    expected_exports = {
        "AuditLogger",
        "AuditStreamEvent",
        "EventStreamWriter",
        "ReplayEngine",
        "read_event_stream",
        "verify_event_stream_integrity",
    }

    assert set(audit.__all__) == expected_exports

    for export_name in expected_exports:
        audit.__dict__.pop(export_name, None)
        resolved = getattr(audit, export_name)

        assert resolved is audit.__dict__[export_name]
        assert getattr(audit, export_name) is resolved


def test_audit_package_rejects_unknown_lazy_export() -> None:
    unknown_name = "Unknown" + "AuditExport"

    with pytest.raises(AttributeError, match="has no attribute 'UnknownAuditExport'"):
        getattr(audit, unknown_name)
