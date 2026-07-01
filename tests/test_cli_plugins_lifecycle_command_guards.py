# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI plugin lifecycle command rejection-path guards

"""Rejection-path coverage for the plugin lifecycle CLI commands.

Each test drives exactly one guard in
:mod:`scpn_phase_orchestrator.runtime.cli.plugins.lifecycle` that the happy-path
CLI tests in ``test_cli.py`` never reach: the storage-bundle object check and the
builder-mismatch handler of ``lifecycle-status``, the ``request_count`` mismatch of
``lifecycle-renewal-queue``, the empty ``created_by`` and malformed
``policy_action_counts``/``status_counts`` guards of the multi-store dashboard and
drill-down, and the empty-input guard of the drill-down callback. Canonical
request/summary/policy payloads are built once from the registry builders and a
single field is tampered per test so the guard under test is the only thing that
fires.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
import pytest
from click.testing import CliRunner

from scpn_phase_orchestrator.plugins.registry import (
    PluginCapability,
    PluginManifest,
    PluginRuntimeExecutionPolicy,
    build_plugin_execution_approval,
    build_plugin_execution_plan,
    build_plugin_execution_request,
)
from scpn_phase_orchestrator.plugins.registry.lifecycle import (
    build_plugin_execution_request_lifecycle_policy_report,
    build_plugin_execution_request_lifecycle_record,
    build_plugin_execution_request_lifecycle_summary,
)
from scpn_phase_orchestrator.plugins.registry.request import PluginExecutionRequest
from scpn_phase_orchestrator.runtime.cli import main
from scpn_phase_orchestrator.runtime.cli.plugins.lifecycle import (
    plugins_lifecycle_multistore_drilldown,
)

_HEX = "a" * 64


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _manifest() -> PluginManifest:
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
        approval_reference="REQ-2026-07-01",
        approval_reason="operator approved",
    )
    return build_plugin_execution_request(plan, approval)


def _request_payload() -> dict[str, Any]:
    return dict(_request().audit_record)


def _summary_payload() -> dict[str, Any]:
    record = build_plugin_execution_request_lifecycle_record(
        _request(), created_by="ops_console"
    )
    summary = build_plugin_execution_request_lifecycle_summary(
        (record,), created_by="ops_console"
    )
    return dict(summary.audit_record)


def _policy_payload() -> dict[str, Any]:
    record = build_plugin_execution_request_lifecycle_record(
        _request(), created_by="ops_console"
    )
    summary = build_plugin_execution_request_lifecycle_summary(
        (record,), created_by="ops_console"
    )
    report = build_plugin_execution_request_lifecycle_policy_report(
        summary, created_by="ops_console"
    )
    return dict(report.audit_record)


def _foreign_storage_manifest() -> dict[str, Any]:
    """A structurally valid storage manifest that references a different request."""
    return {
        "schema": "scpn_plugin_execution_request_storage_manifest_v1",
        "request_hash": _HEX,
        "plan_hash": _HEX,
        "approval_hash": _HEX,
        "target_hash": _HEX,
        "revocation_hash": _HEX,
        "manifest_hash": _HEX,
        "plugin": "grid_pack",
        "kind": "actuator",
        "name": "breaker",
        "operator_identity": "operator",
        "approval_reference": "REF-1",
        "storage_uri": "file:///tmp/store",
        "storage_backend": "local_file",
        "retention_policy": "retain_until_revoked",
        "created_by": "operator",
        "version": "1.0.0",
        "revoked_request_hashes": [],
    }


def _write(tmp_path: Path, name: str, payload: dict[str, Any]) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


# --- lifecycle-status -----------------------------------------------------


def test_lifecycle_status_rejects_non_object_storage_manifest(
    runner: CliRunner, tmp_path: Path
) -> None:
    request_path = _write(tmp_path, "request.json", _request_payload())
    bundle_path = _write(tmp_path, "bundle.json", {"storage_manifest": "not-an-object"})

    result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-status",
            str(request_path),
            "--storage-bundle",
            str(bundle_path),
            "--created-by",
            "ops_console",
        ],
    )

    assert result.exit_code == 1
    assert "storage_manifest must be an object" in result.output


def test_lifecycle_status_rejects_foreign_storage_manifest(
    runner: CliRunner, tmp_path: Path
) -> None:
    request_path = _write(tmp_path, "request.json", _request_payload())
    bundle_path = _write(
        tmp_path,
        "bundle.json",
        {"storage_manifest": _foreign_storage_manifest()},
    )

    result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-status",
            str(request_path),
            "--storage-bundle",
            str(bundle_path),
            "--created-by",
            "ops_console",
        ],
    )

    assert result.exit_code == 1
    assert "request hash mismatch" in result.output


# --- lifecycle-renewal-queue ----------------------------------------------


def test_lifecycle_renewal_queue_rejects_request_count_mismatch(
    runner: CliRunner, tmp_path: Path
) -> None:
    summary = _summary_payload()
    policy = _policy_payload()
    # Keep summary_hash matching so the summary_hash guard passes, then diverge
    # request_count so the request_count guard is the only branch that fires.
    assert policy["summary_hash"] == summary["summary_hash"]
    policy["request_count"] = summary["request_count"] + 1
    summary_path = _write(tmp_path, "summary.json", summary)
    policy_path = _write(tmp_path, "policy.json", policy)

    result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-renewal-queue",
            str(summary_path),
            "--policy-report",
            str(policy_path),
            "--created-by",
            "ops_console",
        ],
    )

    assert result.exit_code == 1
    assert "request_count does not match lifecycle summary" in result.output


# --- lifecycle-multistore-dashboard ---------------------------------------


def test_multistore_dashboard_rejects_empty_created_by(
    runner: CliRunner, tmp_path: Path
) -> None:
    policy_path = _write(tmp_path, "policy.json", _policy_payload())

    result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-multistore-dashboard",
            str(policy_path),
            "--created-by",
            "",
        ],
    )

    assert result.exit_code == 1
    assert "created_by must be non-empty" in result.output


def test_multistore_dashboard_rejects_non_dict_policy_action_counts(
    runner: CliRunner, tmp_path: Path
) -> None:
    policy = _policy_payload()
    policy["policy_action_counts"] = ["not", "a", "dict"]
    policy_path = _write(tmp_path, "policy.json", policy)

    result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-multistore-dashboard",
            str(policy_path),
            "--created-by",
            "ops_console",
        ],
    )

    assert result.exit_code == 1
    assert "policy_action_counts is malformed" in result.output


def test_multistore_dashboard_rejects_negative_action_count(
    runner: CliRunner, tmp_path: Path
) -> None:
    policy = _policy_payload()
    action_counts = dict(policy["policy_action_counts"])
    action_counts["persist_request"] = -1
    policy["policy_action_counts"] = action_counts
    policy_path = _write(tmp_path, "policy.json", policy)

    result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-multistore-dashboard",
            str(policy_path),
            "--created-by",
            "ops_console",
        ],
    )

    assert result.exit_code == 1
    assert "policy_action_counts is malformed" in result.output


# --- lifecycle-multistore-drilldown ---------------------------------------


def test_multistore_drilldown_rejects_empty_created_by(
    runner: CliRunner, tmp_path: Path
) -> None:
    policy_path = _write(tmp_path, "policy.json", _policy_payload())

    result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-multistore-drilldown",
            str(policy_path),
            "--created-by",
            "",
        ],
    )

    assert result.exit_code == 1
    assert "created_by must be non-empty" in result.output


def test_multistore_drilldown_rejects_empty_policy_input() -> None:
    # click's ``nargs=-1, required=True`` blocks an empty argument list at the CLI
    # boundary, so the callback's own empty-input guard is reachable only by a
    # direct programmatic call.
    with pytest.raises(click.ClickException, match="at least one policy report"):
        plugins_lifecycle_multistore_drilldown.callback(
            policy_json=(),
            created_by="ops_console",
        )


def test_multistore_drilldown_rejects_non_dict_status_counts(
    runner: CliRunner, tmp_path: Path
) -> None:
    policy = _policy_payload()
    policy["status_counts"] = ["not", "a", "dict"]
    policy_path = _write(tmp_path, "policy.json", policy)

    result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-multistore-drilldown",
            str(policy_path),
            "--created-by",
            "ops_console",
        ],
    )

    assert result.exit_code == 1
    assert "status/action counts are malformed" in result.output
