# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Plugin lifecycle record/summary/policy validation tests

"""Validation-guard coverage for plugin execution-request lifecycle artefacts.

Each test drives one rejection path in
:mod:`scpn_phase_orchestrator.plugins.registry.lifecycle`: the missing-audit-record
guards of the three lifecycle dataclasses, the audit-record schema/version/hash/field
checks of ``validate_..._lifecycle_record`` and ``validate_..._lifecycle_summary``,
the empty-batch guard of the summary builder, and the storage-adapter
membership/duplicate guards of the policy-report builder. A valid request →
record → summary chain is built once and exactly one field is tampered per test.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from scpn_phase_orchestrator.plugins.registry import (
    PluginCapability,
    PluginManifest,
    PluginRuntimeExecutionPolicy,
    build_plugin_execution_approval,
    build_plugin_execution_plan,
    build_plugin_execution_request,
    build_plugin_execution_request_storage_adapter_manifest,
    build_plugin_execution_request_storage_manifest,
)
from scpn_phase_orchestrator.plugins.registry.lifecycle import (
    PluginExecutionRequestLifecycleRecord,
    PluginExecutionRequestLifecycleSummary,
    build_plugin_execution_request_lifecycle_policy_report,
    build_plugin_execution_request_lifecycle_record,
    build_plugin_execution_request_lifecycle_summary,
    validate_plugin_execution_request_lifecycle_record,
    validate_plugin_execution_request_lifecycle_summary,
)
from scpn_phase_orchestrator.plugins.registry.request import PluginExecutionRequest


def _manifest() -> PluginManifest:
    """Build a minimal compatible plugin manifest with an actuator capability."""
    return PluginManifest(
        name="grid_pack",
        version="0.1.0",
        package="grid_pack",
        capabilities=(
            PluginCapability(
                kind="actuator",
                name="breaker",
                target="grid_pack.actuators:BreakerMapper",
                knobs=("K",),
            ),
        ),
        min_spo_version="0.1.0",
    )


def _request(*, operator: str = "operator_alpha") -> PluginExecutionRequest:
    """Build a real approved plugin execution request."""
    draft = build_plugin_execution_plan(
        _manifest(),
        "actuator",
        "breaker",
        policy=PluginRuntimeExecutionPolicy(
            loading_permitted=True, execution_permitted=True
        ),
    )
    plan = build_plugin_execution_plan(
        _manifest(),
        "actuator",
        "breaker",
        policy=PluginRuntimeExecutionPolicy(
            loading_permitted=True,
            execution_permitted=True,
            require_target_hash_approval=True,
            approved_target_hashes=(draft.target_hash,),
        ),
    )
    approval = build_plugin_execution_approval(
        plan,
        operator_identity=operator,
        approval_reference="REQ-2026-06-30",
        approval_reason="operator approved",
    )
    return build_plugin_execution_request(plan, approval)


def _record(
    *, operator: str = "operator_alpha"
) -> PluginExecutionRequestLifecycleRecord:
    """Build a valid approved lifecycle record."""
    return build_plugin_execution_request_lifecycle_record(
        _request(operator=operator), created_by="ops_console"
    )


def _summary(
    *, operator: str = "operator_alpha"
) -> PluginExecutionRequestLifecycleSummary:
    """Build a valid one-record lifecycle summary."""
    return build_plugin_execution_request_lifecycle_summary(
        (_record(operator=operator),), created_by="ops_console"
    )


def _adapter(*, operator: str = "operator_alpha") -> object:
    """Build a real storage adapter manifest for a request."""
    request = _request(operator=operator)
    manifest = build_plugin_execution_request_storage_manifest(
        request,
        storage_uri="file:///var/lib/spo/plugin-requests/grid_pack.json",
        storage_backend="local_file",
        retention_policy="retain_until_revoked",
        created_by="deployment_gate",
    )
    return build_plugin_execution_request_storage_adapter_manifest(request, manifest)


# --- dataclass missing-audit-record guards --------------------------------


def test_lifecycle_record_requires_audit_record() -> None:
    record = _record()
    with pytest.raises(ValueError, match="audit_record must be provided"):
        replace(record, audit_record=None)  # type: ignore[arg-type]


def test_lifecycle_summary_requires_audit_record() -> None:
    summary = _summary()
    with pytest.raises(ValueError, match="audit_record must be provided"):
        replace(summary, audit_record=None)  # type: ignore[arg-type]


def test_lifecycle_policy_report_requires_audit_record() -> None:
    report = build_plugin_execution_request_lifecycle_policy_report(
        _summary(), created_by="ops_console"
    )
    with pytest.raises(ValueError, match="audit_record must be provided"):
        replace(report, audit_record=None)  # type: ignore[arg-type]


# --- validate_..._lifecycle_record ----------------------------------------


def _retamper(
    record: PluginExecutionRequestLifecycleRecord, **changes: object
) -> PluginExecutionRequestLifecycleRecord:
    """Return the record with selected audit-record fields overridden."""
    return replace(record, audit_record={**record.audit_record, **changes})


def test_validate_record_rejects_schema_mismatch() -> None:
    record = _retamper(_record(), schema="wrong")
    with pytest.raises(ValueError, match="lifecycle schema mismatch"):
        validate_plugin_execution_request_lifecycle_record(record)


def test_validate_record_rejects_version_mismatch() -> None:
    record = _retamper(_record(), version="2.0.0")
    with pytest.raises(ValueError, match="lifecycle version must be 1.0.0"):
        validate_plugin_execution_request_lifecycle_record(record)


def test_validate_record_rejects_missing_lifecycle_hash() -> None:
    record = _retamper(_record(), lifecycle_hash=123)
    with pytest.raises(ValueError, match="missing lifecycle_hash"):
        validate_plugin_execution_request_lifecycle_record(record)


def test_validate_record_rejects_field_mismatch() -> None:
    # Tamper a dataclass field (not the audit record), so the lifecycle hash still
    # matches but the field-by-field cross-check fails.
    record = replace(_record(), operator_identity="someone_else")
    with pytest.raises(ValueError, match="operator_identity field mismatch"):
        validate_plugin_execution_request_lifecycle_record(record)


# --- validate_..._lifecycle_summary ---------------------------------------


def _retamper_summary(
    summary: PluginExecutionRequestLifecycleSummary, **changes: object
) -> PluginExecutionRequestLifecycleSummary:
    """Return the summary with selected audit-record fields overridden."""
    return replace(summary, audit_record={**summary.audit_record, **changes})


def test_validate_summary_rejects_schema_mismatch() -> None:
    summary = _retamper_summary(_summary(), schema="wrong")
    with pytest.raises(ValueError, match="lifecycle summary schema mismatch"):
        validate_plugin_execution_request_lifecycle_summary(summary)


def test_validate_summary_rejects_version_mismatch() -> None:
    summary = _retamper_summary(_summary(), version="2.0.0")
    with pytest.raises(ValueError, match="lifecycle summary version must be 1.0.0"):
        validate_plugin_execution_request_lifecycle_summary(summary)


def test_validate_summary_rejects_missing_summary_hash() -> None:
    summary = _retamper_summary(_summary(), summary_hash=123)
    with pytest.raises(ValueError, match="missing summary_hash"):
        validate_plugin_execution_request_lifecycle_summary(summary)


def test_validate_summary_rejects_field_mismatch() -> None:
    summary = replace(_summary(), status_counts={"approved": 99})
    with pytest.raises(ValueError, match="status_counts field mismatch"):
        validate_plugin_execution_request_lifecycle_summary(summary)


# --- builder guards -------------------------------------------------------


def test_summary_builder_rejects_empty_records() -> None:
    with pytest.raises(ValueError, match="requires at least one record"):
        build_plugin_execution_request_lifecycle_summary((), created_by="ops_console")


def test_policy_report_rejects_adapter_outside_summary() -> None:
    summary = _summary(operator="operator_alpha")
    foreign_adapter = _adapter(operator="operator_beta")
    with pytest.raises(ValueError, match="not in lifecycle summary"):
        build_plugin_execution_request_lifecycle_policy_report(
            summary,
            created_by="ops_console",
            storage_adapters=(foreign_adapter,),  # type: ignore[arg-type]
        )


def test_policy_report_rejects_duplicate_adapter() -> None:
    summary = _summary(operator="operator_alpha")
    adapter = _adapter(operator="operator_alpha")
    with pytest.raises(ValueError, match="duplicate storage adapter"):
        build_plugin_execution_request_lifecycle_policy_report(
            summary,
            created_by="ops_console",
            storage_adapters=(adapter, adapter),  # type: ignore[arg-type]
        )
