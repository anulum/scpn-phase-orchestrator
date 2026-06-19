# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Plugin lifecycle policy report and request validation

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from scpn_phase_orchestrator.plugins.registry import (
    PluginExecutionRequest,
    PluginExecutionRequestLifecyclePolicyReport,
    validate_plugin_execution_request,
)

_HEX = "a" * 64
_OTHER = "b" * 64


def _policy_report() -> PluginExecutionRequestLifecyclePolicyReport:
    return PluginExecutionRequestLifecyclePolicyReport(
        schema="scpn_plugin_execution_request_lifecycle_policy_v1",
        version="1.0.0",
        summary_hash=_HEX,
        request_count=1,
        policy_action_counts={"renew_approval": 1},
        storage_missing_request_hashes=(),
        renewal_required_request_hashes=(_HEX,),
        missing_adapter_request_hashes=(),
        local_storage_request_hashes=(),
        non_local_storage_request_hashes=(),
        external_write_followup_request_hashes=(),
        storage_adapter_hashes=(_HEX,),
        created_by="operator",
        policy_hash=_HEX,
        audit_record={},
    )


class TestLifecyclePolicyReport:
    def test_valid_construction(self) -> None:
        assert _policy_report().request_count == 1

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"schema": "x"}, "lifecycle policy schema must be"),
            ({"version": "2.0.0"}, "lifecycle policy version must be 1.0.0"),
            ({"request_count": 0}, "requires at least one request"),
            ({"summary_hash": "short"}, "lifecycle summary hash"),
            ({"policy_hash": "short"}, "lifecycle policy hash"),
            ({"created_by": ""}, "lifecycle policy creator"),
            ({"storage_adapter_hashes": ("bad",)}, "storage adapter hash"),
            (
                {"renewal_required_request_hashes": ("bad",)},
                "lifecycle policy request hash",
            ),
        ],
    )
    def test_rejects_corrupt_field(self, overrides: dict[str, Any], match: str) -> None:
        with pytest.raises((ValueError, TypeError), match=match):
            replace(_policy_report(), **overrides)


def _request(**overrides: Any) -> PluginExecutionRequest:
    fields: dict[str, Any] = {
        "schema": "scpn_plugin_runtime_execution_request_v1",
        "version": "1.0.0",
        "plan_hash": _HEX,
        "approval_hash": _HEX,
        "target_hash": _HEX,
        "plugin": "grid_pack",
        "kind": "monitor",
        "name": "frequency_drift",
        "operator_identity": "operator",
        "approval_reference": "REF-1",
        "loading_permitted": True,
        "execution_permitted": True,
        "require_target_hash_approval": True,
        "approved_target_hashes": (_HEX,),
        "allowed_kinds": ("monitor",),
        "require_package_target": True,
        "audit_record": {"request_hash": _HEX},
    }
    fields.update(overrides)
    return PluginExecutionRequest(**fields)


class TestValidatePluginExecutionRequest:
    def test_rejects_audit_record_without_request_hash(self) -> None:
        with pytest.raises(ValueError, match="missing request_hash"):
            validate_plugin_execution_request(_request(audit_record={}))

    def test_rejects_revoked_request(self) -> None:
        with pytest.raises(PermissionError, match="has been revoked"):
            validate_plugin_execution_request(
                _request(), revoked_request_hashes=(_HEX,)
            )

    def test_rejects_request_not_requiring_target_hash_approval(self) -> None:
        with pytest.raises(PermissionError, match="must require target hash approval"):
            validate_plugin_execution_request(
                _request(require_target_hash_approval=False, approved_target_hashes=())
            )

    def test_rejects_mismatched_approved_target_hashes(self) -> None:
        with pytest.raises(PermissionError, match="exactly the request target hash"):
            validate_plugin_execution_request(
                _request(target_hash=_OTHER, approved_target_hashes=(_HEX,))
            )
