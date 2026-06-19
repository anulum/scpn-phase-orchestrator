# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Plugin revocation-list and lifecycle record guards

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from scpn_phase_orchestrator.plugins.registry import (
    PluginExecutionRequestLifecycleRecord,
    PluginExecutionRequestLifecycleSummary,
    PluginExecutionRequestRevocationList,
)

_HEX = "a" * 64


def _revocation_list() -> PluginExecutionRequestRevocationList:
    return PluginExecutionRequestRevocationList(
        schema="scpn_plugin_execution_request_revocation_list_v1",
        version="1.0.0",
        request_hashes=(_HEX,),
        revocation_hashes=(_HEX,),
        revocation_count=1,
        created_by="operator",
        revocation_list_hash=_HEX,
        audit_record={},
    )


class TestRevocationList:
    def test_valid_construction(self) -> None:
        assert _revocation_list().revocation_count == 1

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"schema": "x"}, "revocation list schema must be"),
            ({"version": "2.0.0"}, "revocation list version must be 1.0.0"),
            ({"revocation_count": 2}, "must match request hash count"),
            ({"revocation_hashes": ()}, "must match revocation hash count"),
            ({"created_by": ""}, "revocation list creator"),
            ({"revocation_list_hash": "short"}, "revocation list hash"),
            ({"request_hashes": ("bad",)}, "revoked request hash"),
        ],
    )
    def test_rejects_corrupt_field(self, overrides: dict[str, Any], match: str) -> None:
        with pytest.raises((ValueError, TypeError), match=match):
            replace(_revocation_list(), **overrides)


def _lifecycle() -> PluginExecutionRequestLifecycleRecord:
    return PluginExecutionRequestLifecycleRecord(
        schema="scpn_plugin_execution_request_lifecycle_v1",
        version="1.0.0",
        request_hash=_HEX,
        status="approved",
        plugin="grid_pack",
        kind="monitor",
        name="frequency_drift",
        operator_identity="operator",
        approval_reference="REF-1",
        storage_manifest_hash=None,
        storage_backend=None,
        storage_uri=None,
        revoked=False,
        revocation_list_hash=None,
        revocation_hash=None,
        revoked_by=None,
        revocation_reference=None,
        created_by="operator",
        lifecycle_hash=_HEX,
        audit_record={},
    )


class TestLifecycleRecord:
    def test_valid_construction(self) -> None:
        assert _lifecycle().status == "approved"

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"schema": "x"}, "lifecycle schema must be"),
            ({"version": "2.0.0"}, "lifecycle version must be 1.0.0"),
            ({"request_hash": "short"}, "lifecycle request hash"),
            ({"lifecycle_hash": "short"}, "lifecycle hash"),
            ({"status": "pending"}, "unsupported lifecycle status"),
            ({"plugin": ""}, "plugin"),
            ({"kind": "wizard"}, "unsupported plugin capability kind"),
            ({"revoked": "yes"}, "revoked must be a boolean"),
            ({"storage_manifest_hash": "short"}, "storage manifest hash"),
        ],
    )
    def test_rejects_corrupt_field(self, overrides: dict[str, Any], match: str) -> None:
        with pytest.raises((ValueError, TypeError), match=match):
            replace(_lifecycle(), **overrides)


def _summary() -> PluginExecutionRequestLifecycleSummary:
    return PluginExecutionRequestLifecycleSummary(
        schema="scpn_plugin_execution_request_lifecycle_summary_v1",
        version="1.0.0",
        request_count=1,
        status_counts={"approved": 1},
        lifecycle_hashes=(_HEX,),
        approved_request_hashes=(_HEX,),
        stored_request_hashes=(),
        revoked_request_hashes=(),
        storage_missing_request_hashes=(),
        renewal_required_request_hashes=(),
        created_by="operator",
        summary_hash=_HEX,
        audit_record={},
    )


class TestLifecycleSummary:
    def test_valid_construction(self) -> None:
        assert _summary().request_count == 1

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"schema": "x"}, "lifecycle summary schema must be"),
            ({"version": "2.0.0"}, "lifecycle summary version must be 1.0.0"),
            ({"request_count": 0}, "requires at least one request"),
            ({"created_by": ""}, "lifecycle summary creator"),
            ({"summary_hash": "short"}, "lifecycle summary hash"),
            ({"lifecycle_hashes": ("bad",)}, "lifecycle hash"),
            ({"approved_request_hashes": ("bad",)}, "lifecycle summary request hash"),
        ],
    )
    def test_rejects_corrupt_field(self, overrides: dict[str, Any], match: str) -> None:
        with pytest.raises((ValueError, TypeError), match=match):
            replace(_summary(), **overrides)
