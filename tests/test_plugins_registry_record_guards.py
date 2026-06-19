# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Plugin execution-request record validation guards

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from scpn_phase_orchestrator.plugins.registry import (
    PluginExecutionRequest,
    PluginExecutionRequestStorageManifest,
)

_HEX = "a" * 64


def _request() -> PluginExecutionRequest:
    return PluginExecutionRequest(
        schema="scpn_plugin_runtime_execution_request_v1",
        version="1.0.0",
        plan_hash=_HEX,
        approval_hash=_HEX,
        target_hash=_HEX,
        plugin="grid_pack",
        kind="monitor",
        name="frequency_drift",
        operator_identity="operator",
        approval_reference="REF-1",
        loading_permitted=True,
        execution_permitted=True,
        require_target_hash_approval=False,
        approved_target_hashes=(),
        allowed_kinds=("monitor",),
        require_package_target=True,
        audit_record={},
    )


class TestPluginExecutionRequest:
    def test_valid_construction(self) -> None:
        assert _request().execution_permitted is True

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"schema": "x"}, "request schema must be"),
            ({"version": "2.0.0"}, "request version must be 1.0.0"),
            ({"plan_hash": "short"}, "request plan hash"),
            ({"plugin": ""}, "plugin"),
            ({"kind": "wizard"}, "unsupported plugin capability kind"),
            ({"loading_permitted": "yes"}, "loading_permitted must be a boolean"),
            ({"execution_permitted": "yes"}, "execution_permitted must be a boolean"),
            (
                {"require_target_hash_approval": "yes"},
                "require_target_hash_approval must be a boolean",
            ),
            (
                {"require_package_target": "yes"},
                "require_package_target must be a boolean",
            ),
            ({"allowed_kinds": ("wizard",)}, "unsupported runtime load kind"),
            ({"approved_target_hashes": ("bad",)}, "approved target hash"),
        ],
    )
    def test_rejects_corrupt_field(self, overrides: dict[str, Any], match: str) -> None:
        with pytest.raises((ValueError, TypeError), match=match):
            replace(_request(), **overrides)


def _storage_manifest() -> PluginExecutionRequestStorageManifest:
    return PluginExecutionRequestStorageManifest(
        schema="scpn_plugin_execution_request_storage_manifest_v1",
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
        storage_uri="file:///tmp/store",
        storage_backend="filesystem",
        retention_policy="thirty_days",
        created_by="operator",
        revoked_request_hashes=(),
        revocation_hash=_HEX,
        manifest_hash=_HEX,
        audit_record={},
    )


class TestPluginExecutionRequestStorageManifest:
    def test_valid_construction(self) -> None:
        assert _storage_manifest().storage_backend == "filesystem"

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"schema": "x"}, "storage manifest schema must be"),
            ({"version": "2.0.0"}, "storage manifest version must be 1.0.0"),
            ({"request_hash": "short"}, "storage manifest request hash"),
            ({"manifest_hash": "short"}, "storage manifest hash"),
            ({"plugin": ""}, "plugin"),
            ({"kind": "wizard"}, "unsupported plugin capability kind"),
            ({"storage_uri": ""}, "storage URI"),
            ({"storage_backend": ""}, "storage backend"),
            ({"retention_policy": ""}, "retention policy"),
            ({"created_by": ""}, "storage manifest creator"),
            ({"revoked_request_hashes": ("bad",)}, "revoked request hash"),
        ],
    )
    def test_rejects_corrupt_field(self, overrides: dict[str, Any], match: str) -> None:
        with pytest.raises((ValueError, TypeError), match=match):
            replace(_storage_manifest(), **overrides)
