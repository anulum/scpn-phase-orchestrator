# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Plugin storage-adapter and revocation record guards

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from scpn_phase_orchestrator.plugins.registry import (
    PluginExecutionRequestRevocation,
    PluginExecutionRequestStorageAdapterManifest,
)

_HEX = "a" * 64


def _adapter() -> PluginExecutionRequestStorageAdapterManifest:
    return PluginExecutionRequestStorageAdapterManifest(
        schema="scpn_plugin_execution_request_storage_adapter_v1",
        version="1.0.0",
        request_hash=_HEX,
        storage_manifest_hash=_HEX,
        storage_backend="filesystem",
        storage_uri="file:///tmp/store",
        storage_scheme="file",
        adapter_mode="external",
        bundle_hash=_HEX,
        write_performed=True,
        created_by="operator",
        adapter_hash=_HEX,
        audit_record={},
    )


class TestStorageAdapterManifest:
    def test_valid_construction(self) -> None:
        assert _adapter().write_performed is True

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"schema": "x"}, "storage adapter schema must be"),
            ({"version": "2.0.0"}, "storage adapter version must be 1.0.0"),
            ({"request_hash": "short"}, "storage adapter request hash"),
            ({"bundle_hash": "short"}, "storage adapter bundle hash"),
            ({"storage_backend": ""}, "storage backend"),
            ({"storage_uri": ""}, "storage URI"),
            ({"created_by": ""}, "storage adapter creator"),
            ({"write_performed": "yes"}, "write_performed must be a boolean"),
        ],
    )
    def test_rejects_corrupt_field(self, overrides: dict[str, Any], match: str) -> None:
        with pytest.raises((ValueError, TypeError), match=match):
            replace(_adapter(), **overrides)


def _revocation() -> PluginExecutionRequestRevocation:
    return PluginExecutionRequestRevocation(
        schema="scpn_plugin_execution_request_revocation_v1",
        version="1.0.0",
        request_hash=_HEX,
        plan_hash=_HEX,
        approval_hash=_HEX,
        target_hash=_HEX,
        plugin="grid_pack",
        kind="monitor",
        name="frequency_drift",
        operator_identity="operator",
        approval_reference="REF-1",
        revoked_by="operator",
        revocation_reference="RREF-1",
        revocation_reason="superseded",
        revoked=True,
        revocation_hash=_HEX,
        audit_record={},
    )


class TestExecutionRequestRevocation:
    def test_valid_construction(self) -> None:
        assert _revocation().revoked is True

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"schema": "x"}, "revocation schema must be"),
            ({"version": "2.0.0"}, "revocation version must be 1.0.0"),
            ({"request_hash": "short"}, "revocation request hash"),
            ({"revocation_hash": "short"}, "revocation hash"),
            ({"plugin": ""}, "plugin"),
            ({"kind": "wizard"}, "unsupported plugin capability kind"),
        ],
    )
    def test_rejects_corrupt_field(self, overrides: dict[str, Any], match: str) -> None:
        with pytest.raises((ValueError, TypeError), match=match):
            replace(_revocation(), **overrides)
