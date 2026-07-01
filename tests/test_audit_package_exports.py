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

_EXPECTED_EXPORTS = {
    "AuditLogger",
    "AuditStreamEvent",
    "EventStreamWriter",
    "ReplayEngine",
    "read_event_stream",
    "verify_event_stream_integrity",
}


def test_audit_package_lazy_exports_resolve_and_cache_public_contracts() -> None:
    assert set(audit.__all__) == _EXPECTED_EXPORTS

    for export_name in _EXPECTED_EXPORTS:
        audit.__dict__.pop(export_name, None)
        resolved = getattr(audit, export_name)

        assert resolved is audit.__dict__[export_name]
        assert getattr(audit, export_name) is resolved


def test_audit_package_dir_lists_lazy_public_exports_before_resolution() -> None:
    for export_name in _EXPECTED_EXPORTS:
        audit.__dict__.pop(export_name, None)

    assert set(dir(audit)) >= _EXPECTED_EXPORTS


def test_audit_package_rejects_unknown_lazy_export() -> None:
    unknown_name = "Unknown" + "AuditExport"

    with pytest.raises(AttributeError, match="has no attribute 'UnknownAuditExport'"):
        getattr(audit, unknown_name)
