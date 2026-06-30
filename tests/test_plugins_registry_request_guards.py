# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Plugin execution request validation-guard tests

"""Validation-guard coverage for plugin execution approval and request building.

Each test drives one rejection path in
:mod:`scpn_phase_orchestrator.plugins.registry.request` — the request
``__post_init__`` field guards, ``validate_plugin_execution_request`` audit-record
checks, and the consistency/permission/kind guards of
``build_plugin_execution_approval`` and ``build_plugin_execution_request`` — by
building a valid plan and approval and then tampering exactly one field.
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
)
from scpn_phase_orchestrator.plugins.registry.request import (
    PluginExecutionApproval,
    PluginExecutionRequest,
    build_plugin_execution_request,
    validate_plugin_execution_request,
)
from scpn_phase_orchestrator.plugins.registry.runtime import PluginExecutionPlan

_HEX = "a" * 64
_OTHER_HEX = "b" * 64


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


def _plan() -> PluginExecutionPlan:
    """Build a fully approvable execution plan (loading + execution + target)."""
    draft = build_plugin_execution_plan(
        _manifest(),
        "actuator",
        "breaker",
        policy=PluginRuntimeExecutionPolicy(
            loading_permitted=True,
            execution_permitted=True,
        ),
    )
    return build_plugin_execution_plan(
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


def _approval(plan: PluginExecutionPlan) -> PluginExecutionApproval:
    """Build a granted operator approval for a plan."""
    return build_plugin_execution_approval(
        plan,
        operator_identity="operator_alpha",
        approval_reference="REQ-2026-06-30",
        approval_reason="operator approved",
    )


def _retampered_plan(
    plan: PluginExecutionPlan, **audit_changes: object
) -> PluginExecutionPlan:
    """Return the plan with selected audit-record fields overridden."""
    return replace(plan, audit_record={**plan.audit_record, **audit_changes})


# --- PluginExecutionRequest.__post_init__ ---------------------------------


def test_request_rejects_missing_audit_record() -> None:
    with pytest.raises(ValueError, match="audit_record must be provided"):
        PluginExecutionRequest(
            schema="scpn_plugin_runtime_execution_request_v1",
            version="1.0.0",
            plan_hash=_HEX,
            approval_hash=_HEX,
            target_hash=_HEX,
            plugin="grid_pack",
            kind="actuator",
            name="breaker",
            operator_identity="operator_alpha",
            approval_reference="REQ",
            loading_permitted=True,
            execution_permitted=True,
            require_target_hash_approval=True,
            approved_target_hashes=(_HEX,),
            allowed_kinds=("actuator",),
            require_package_target=True,
            audit_record=None,  # type: ignore[arg-type]
        )


# --- validate_plugin_execution_request ------------------------------------


def test_validate_rejects_audit_record_mismatch() -> None:
    request = build_plugin_execution_request(_plan(), _approval(_plan()))
    drifted = replace(request, operator_identity="operator_beta")

    with pytest.raises(ValueError, match="request audit record mismatch"):
        validate_plugin_execution_request(drifted)


# --- build_plugin_execution_approval --------------------------------------


def test_approval_requires_execution_permitted() -> None:
    plan = _retampered_plan(_plan(), execution_permitted=False)

    with pytest.raises(PermissionError, match="execution must be permitted"):
        _approval(plan)


def test_approval_requires_approved_target_hash() -> None:
    plan = _retampered_plan(
        _plan(), require_target_hash_approval=True, target_hash_approved=False
    )

    with pytest.raises(PermissionError, match="is not approved"):
        _approval(plan)


# --- build_plugin_execution_request: approval consistency -----------------


def test_request_rejects_wrong_approval_schema() -> None:
    plan = _plan()
    approval = replace(_approval(plan), schema="wrong")

    with pytest.raises(ValueError, match="approval schema must be"):
        build_plugin_execution_request(plan, approval)


def test_request_rejects_ungranted_approval() -> None:
    plan = _plan()
    approval = replace(_approval(plan), approved=False)

    with pytest.raises(PermissionError, match="approval must be granted"):
        build_plugin_execution_request(plan, approval)


def test_request_rejects_approval_without_execution_permission() -> None:
    plan = _plan()
    approval = replace(_approval(plan), execution_permitted=False)

    with pytest.raises(PermissionError, match="execution_permitted must be true"):
        build_plugin_execution_request(plan, approval)


def test_request_rejects_plan_hash_mismatch() -> None:
    plan = _plan()
    approval = replace(_approval(plan), plan_hash=_OTHER_HEX)

    with pytest.raises(ValueError, match="plan hash mismatch"):
        build_plugin_execution_request(plan, approval)


def test_request_rejects_target_hash_mismatch() -> None:
    plan = _plan()
    approval = replace(_approval(plan), target_hash=_OTHER_HEX)

    with pytest.raises(ValueError, match="target hash mismatch"):
        build_plugin_execution_request(plan, approval)


def test_request_rejects_plugin_name_mismatch() -> None:
    plan = _plan()
    approval = replace(_approval(plan), plugin="other_pack")

    with pytest.raises(ValueError, match="plugin name mismatch"):
        build_plugin_execution_request(plan, approval)


def test_request_rejects_capability_kind_mismatch() -> None:
    plan = _plan()
    approval = replace(_approval(plan), kind="monitor")

    with pytest.raises(ValueError, match="capability kind mismatch"):
        build_plugin_execution_request(plan, approval)


def test_request_rejects_capability_name_mismatch() -> None:
    plan = _plan()
    approval = replace(_approval(plan), name="other_name")

    with pytest.raises(ValueError, match="capability name mismatch"):
        build_plugin_execution_request(plan, approval)


# --- build_plugin_execution_request: plan audit-record guards -------------


def test_request_requires_loading_permitted() -> None:
    plan = _plan()
    approval = _approval(plan)
    tampered = _retampered_plan(plan, loading_permitted=False)

    with pytest.raises(PermissionError, match="loading must be permitted"):
        build_plugin_execution_request(tampered, approval)


def test_request_requires_execution_permitted() -> None:
    plan = _plan()
    approval = _approval(plan)
    tampered = _retampered_plan(plan, execution_permitted=False)

    with pytest.raises(PermissionError, match="execution must be permitted before"):
        build_plugin_execution_request(tampered, approval)


def test_request_requires_approved_target_hash() -> None:
    plan = _plan()
    approval = _approval(plan)
    tampered = _retampered_plan(plan, target_hash_approved=False)

    with pytest.raises(PermissionError, match="is not approved"):
        build_plugin_execution_request(tampered, approval)


def test_request_rejects_non_sequence_allowed_kinds() -> None:
    plan = _plan()
    approval = _approval(plan)
    tampered = _retampered_plan(plan, allowed_kinds="actuator")

    with pytest.raises(ValueError, match="allowed_kinds must be a sequence"):
        build_plugin_execution_request(tampered, approval)


def test_request_rejects_empty_allowed_kinds() -> None:
    plan = _plan()
    approval = _approval(plan)
    tampered = _retampered_plan(plan, allowed_kinds=[])

    with pytest.raises(ValueError, match="missing allowed_kinds"):
        build_plugin_execution_request(tampered, approval)


def test_request_rejects_unsupported_allowed_kind() -> None:
    plan = _plan()
    approval = _approval(plan)
    tampered = _retampered_plan(plan, allowed_kinds=["nonsense"])

    with pytest.raises(ValueError, match="unsupported runtime load kind"):
        build_plugin_execution_request(tampered, approval)


def test_request_accepts_domainpack_allowed_kind() -> None:
    plan = _plan()
    approval = _approval(plan)
    tampered = _retampered_plan(plan, allowed_kinds=["domainpack"])

    request = build_plugin_execution_request(tampered, approval)

    assert request.allowed_kinds == ("domainpack",)
