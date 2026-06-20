# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI plan, approval, and request payload loaders

"""Execution plan, approval, and request payload loaders for the plugins CLI."""

from __future__ import annotations

from typing import Literal, cast

import click

from scpn_phase_orchestrator.plugins import (
    PluginExecutionApproval,
    PluginExecutionPlan,
    PluginExecutionRequest,
    PluginManifest,
)

from ._shared import (
    _PLUGIN_KIND_OPTIONS,
    _build_plan_payload_for_hash,
    _find_capability,
    _record_hash,
    _require_sha256,
)


def _load_plan_from_payload(
    plan_payload: dict[str, object],
) -> tuple[PluginExecutionPlan, dict[str, object]]:
    if plan_payload.get("schema") != "scpn_plugin_runtime_execution_plan_v1":
        raise click.ClickException(
            "plan schema mismatch: expected scpn_plugin_runtime_execution_plan_v1"
        )

    manifest_payload = plan_payload.get("manifest")
    capability_payload = plan_payload.get("capability")
    if not isinstance(manifest_payload, dict):
        raise click.ClickException("plan payload is missing manifest object")
    if not isinstance(capability_payload, dict):
        raise click.ClickException("plan payload is missing capability object")

    try:
        manifest = PluginManifest.from_mapping(manifest_payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise click.ClickException(f"manifest schema mismatch: {exc}") from exc

    kind = capability_payload.get("kind")
    name = capability_payload.get("name")
    if not isinstance(kind, str) or not isinstance(name, str):
        raise click.ClickException(
            "capability schema mismatch: kind and name are required"
        )

    try:
        capability = _find_capability(manifest, kind, name)
    except click.ClickException as exc:
        raise click.ClickException(f"capability schema mismatch: {exc}") from exc

    raw_argument_count = plan_payload.get("argument_count")
    if not isinstance(raw_argument_count, int) or raw_argument_count < 0:
        raise click.ClickException(
            "plan schema mismatch: argument_count must be a non-negative integer"
        )
    raw_keyword_names = plan_payload.get("keyword_names")
    if not isinstance(raw_keyword_names, list):
        raise click.ClickException("plan schema mismatch: keyword_names must be a list")
    if not all(isinstance(name, str) for name in raw_keyword_names):
        raise click.ClickException(
            "plan schema mismatch: keyword_names must contain strings"
        )

    expected_plan_hash = _record_hash(
        _build_plan_payload_for_hash(plan_payload),
    )
    plan_hash = _require_sha256(plan_payload.get("plan_hash"), "plan_hash")
    if expected_plan_hash != plan_hash:
        raise click.ClickException("plan hash mismatch")

    target_hash = _require_sha256(plan_payload.get("target_hash"), "target_hash")

    audit_record = dict(plan_payload)
    audit_record["target_hash"] = target_hash
    execution_permitted = plan_payload.get("execution_permitted")
    if not isinstance(execution_permitted, bool):
        raise click.ClickException(
            "plan schema mismatch: execution_permitted must be a boolean"
        )
    if not execution_permitted:
        raise click.ClickException(
            "plugin runtime execution must be permitted for approval"
        )

    if plan_payload.get("require_target_hash_approval") is True:
        target_hash_approved = plan_payload.get("target_hash_approved")
        if target_hash_approved is not True:
            raise click.ClickException(
                f"plugin runtime target hash {target_hash} is not approved"
            )

    return PluginExecutionPlan(
        manifest=manifest,
        capability=capability,
        argument_count=raw_argument_count,
        keyword_names=tuple(raw_keyword_names),
        target_hash=target_hash,
        plan_hash=plan_hash,
        audit_record=audit_record,
    ), audit_record


def _load_approval_from_payload(
    approval_payload: dict[str, object],
) -> PluginExecutionApproval:
    if approval_payload.get("schema") != "scpn_plugin_execution_approval_v1":
        raise click.ClickException(
            "approval schema mismatch: expected scpn_plugin_execution_approval_v1"
        )

    plan_hash = _require_sha256(approval_payload.get("plan_hash"), "plan_hash")
    target_hash = _require_sha256(approval_payload.get("target_hash"), "target_hash")
    approval_hash = _require_sha256(
        approval_payload.get("approval_hash"), "approval_hash"
    )
    plugin = approval_payload.get("plugin")
    kind = approval_payload.get("kind")
    name = approval_payload.get("name")
    operator_identity = approval_payload.get("operator_identity")
    approval_reference = approval_payload.get("approval_reference")
    approval_reason = approval_payload.get("approval_reason")
    approved = approval_payload.get("approved")
    execution_permitted = approval_payload.get("execution_permitted")
    version = approval_payload.get("version")

    for field_name, value in (
        ("plugin", plugin),
        ("kind", kind),
        ("name", name),
        ("operator_identity", operator_identity),
        ("approval_reference", approval_reference),
        ("approval_reason", approval_reason),
        ("version", version),
    ):
        if not isinstance(value, str) or not value:
            raise click.ClickException(
                f"approval schema mismatch: {field_name} must be a non-empty string"
            )

    if not isinstance(approved, bool):
        raise click.ClickException(
            "approval schema mismatch: approved must be a boolean"
        )
    if not isinstance(execution_permitted, bool):
        raise click.ClickException(
            "approval schema mismatch: execution_permitted must be a boolean"
        )
    if kind not in _PLUGIN_KIND_OPTIONS:
        raise click.ClickException(
            f"approval schema mismatch: unsupported kind {kind!r}"
        )

    return PluginExecutionApproval(
        schema="scpn_plugin_execution_approval_v1",
        version=str(version),
        plan_hash=plan_hash,
        target_hash=target_hash,
        plugin=str(plugin),
        kind=cast(
            Literal["actuator", "bridge", "domainpack", "extractor", "monitor"],
            str(kind),
        ),
        name=str(name),
        operator_identity=str(operator_identity),
        approval_reference=str(approval_reference),
        approval_reason=str(approval_reason),
        approved=bool(approved),
        execution_permitted=bool(execution_permitted),
        approval_hash=approval_hash,
        audit_record=approval_payload,
    )


def _load_request_from_payload(
    request_payload: dict[str, object],
) -> PluginExecutionRequest:
    if request_payload.get("schema") != "scpn_plugin_runtime_execution_request_v1":
        raise click.ClickException(
            "request schema mismatch: expected scpn_plugin_runtime_execution_request_v1"
        )

    plan_hash = _require_sha256(request_payload.get("plan_hash"), "plan_hash")
    target_hash = _require_sha256(request_payload.get("target_hash"), "target_hash")
    approval_hash = _require_sha256(
        request_payload.get("approval_hash"), "approval_hash"
    )
    plugin = request_payload.get("plugin")
    kind = request_payload.get("kind")
    name = request_payload.get("name")
    operator_identity = request_payload.get("operator_identity")
    approval_reference = request_payload.get("approval_reference")
    loading_permitted = request_payload.get("loading_permitted")
    execution_permitted = request_payload.get("execution_permitted")
    require_target_hash_approval = request_payload.get("require_target_hash_approval")
    require_package_target = request_payload.get("require_package_target")
    approved_target_hashes = request_payload.get("approved_target_hashes")
    allowed_kinds = request_payload.get("allowed_kinds")
    version = request_payload.get("version")

    for field_name, value in (
        ("plugin", plugin),
        ("kind", kind),
        ("name", name),
        ("operator_identity", operator_identity),
        ("approval_reference", approval_reference),
        ("version", version),
    ):
        if not isinstance(value, str) or not value:
            raise click.ClickException(
                f"request schema mismatch: {field_name} must be a non-empty string"
            )
    if kind not in _PLUGIN_KIND_OPTIONS:
        raise click.ClickException(
            f"request schema mismatch: unsupported kind {kind!r}"
        )
    for field_name, value in (
        ("loading_permitted", loading_permitted),
        ("execution_permitted", execution_permitted),
        ("require_target_hash_approval", require_target_hash_approval),
        ("require_package_target", require_package_target),
    ):
        if not isinstance(value, bool):
            raise click.ClickException(
                f"request schema mismatch: {field_name} must be a boolean"
            )
    if not isinstance(approved_target_hashes, list) or not all(
        isinstance(item, str) for item in approved_target_hashes
    ):
        raise click.ClickException(
            "request schema mismatch: approved_target_hashes must be a string list"
        )
    if not isinstance(allowed_kinds, list) or not all(
        isinstance(item, str) and item in _PLUGIN_KIND_OPTIONS for item in allowed_kinds
    ):
        raise click.ClickException(
            "request schema mismatch: allowed_kinds must be valid kind strings"
        )
    normalized_target_hashes = tuple(
        _require_sha256(item, "approved_target_hash") for item in approved_target_hashes
    )

    return PluginExecutionRequest(
        schema="scpn_plugin_runtime_execution_request_v1",
        version=str(version),
        plan_hash=plan_hash,
        approval_hash=approval_hash,
        target_hash=target_hash,
        plugin=str(plugin),
        kind=cast(
            Literal["actuator", "bridge", "domainpack", "extractor", "monitor"],
            str(kind),
        ),
        name=str(name),
        operator_identity=str(operator_identity),
        approval_reference=str(approval_reference),
        loading_permitted=bool(loading_permitted),
        execution_permitted=bool(execution_permitted),
        require_target_hash_approval=bool(require_target_hash_approval),
        approved_target_hashes=normalized_target_hashes,
        allowed_kinds=cast(
            tuple[
                Literal["actuator", "bridge", "domainpack", "extractor", "monitor"],
                ...,
            ],
            tuple(allowed_kinds),
        ),
        require_package_target=bool(require_package_target),
        audit_record=request_payload,
    )
