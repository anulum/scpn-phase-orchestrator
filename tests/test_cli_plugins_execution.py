# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — plugin execution approve/request CLI branch tests

"""Branch tests for the plugin approve-execution-plan and request-execution CLIs.

Covers the missing-approval-field guards, the approval-build failure path, the
plan/approval plugin/kind/name mismatch checks, the not-execution-permitted
guard, and the request-builder error and approval-return branches.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

import scpn_phase_orchestrator.runtime.cli.plugins.execution as _execution
from scpn_phase_orchestrator.plugins import PluginExecutionApproval
from scpn_phase_orchestrator.runtime.cli._app import main
from tests.plugin_execution_fixtures import (
    make_manifest,
    write_approval_payload,
    write_plan_payload,
)


@pytest.fixture
def runner() -> CliRunner:
    """Return a Click runner."""
    return CliRunner()


def _valid_pair(tmp_path: Path, **approval: object) -> tuple[Path, Path]:
    """Write a matching plan and approval and return their paths."""
    manifest = make_manifest()
    plan_path = tmp_path / "plan.json"
    write_plan_payload(plan_path, manifest, "extractor", "phase")
    approval_path = tmp_path / "approval.json"
    write_approval_payload(approval_path, manifest, "extractor", "phase", **approval)  # type: ignore[arg-type]
    return plan_path, approval_path


def _mutate(path: Path, **fields: object) -> None:
    """Overwrite ``path`` JSON with ``fields`` merged in."""
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(fields)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _approve(runner: CliRunner, plan_path: Path, **opts: str) -> object:
    """Invoke approve-execution-plan with default-or-overridden options."""
    args = {
        "--operator-id": "operator_42",
        "--approval-reference": "RFC-1",
        "--approval-reason": "change window",
    }
    args.update(opts)
    flat: list[str] = []
    for key, value in args.items():
        flat.extend([key, value])
    return runner.invoke(
        main, ["plugins", "approve-execution-plan", str(plan_path), *flat]
    )


def _request(runner: CliRunner, plan_path: Path, approval_path: Path) -> object:
    """Invoke request-execution."""
    return runner.invoke(
        main, ["plugins", "request-execution", str(plan_path), str(approval_path)]
    )


def test_approve_rejects_an_empty_approval_reference(
    runner: CliRunner, tmp_path: Path
) -> None:
    manifest = make_manifest()
    plan_path = tmp_path / "plan.json"
    write_plan_payload(plan_path, manifest, "extractor", "phase")

    result = _approve(runner, plan_path, **{"--approval-reference": ""})

    assert result.exit_code == 1
    assert "approval reference is required" in result.output


def test_approve_rejects_an_empty_approval_reason(
    runner: CliRunner, tmp_path: Path
) -> None:
    manifest = make_manifest()
    plan_path = tmp_path / "plan.json"
    write_plan_payload(plan_path, manifest, "extractor", "phase")

    result = _approve(runner, plan_path, **{"--approval-reason": ""})

    assert result.exit_code == 1
    assert "approval reason is required" in result.output


def test_approve_reports_a_build_failure(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = make_manifest()
    plan_path = tmp_path / "plan.json"
    write_plan_payload(plan_path, manifest, "extractor", "phase")

    def _raise(*args: object, **kwargs: object) -> object:
        raise ValueError("approval build refused")

    monkeypatch.setattr(_execution, "build_plugin_execution_approval", _raise)

    result = _approve(runner, plan_path)

    assert result.exit_code == 1
    assert "approval build refused" in result.output


def test_request_rejects_a_plugin_mismatch(runner: CliRunner, tmp_path: Path) -> None:
    plan_path, approval_path = _valid_pair(tmp_path)
    _mutate(approval_path, plugin="other_plugin")

    result = _request(runner, plan_path, approval_path)

    assert result.exit_code == 1
    assert "plugin mismatch" in result.output


def test_request_rejects_a_kind_mismatch(runner: CliRunner, tmp_path: Path) -> None:
    plan_path, approval_path = _valid_pair(tmp_path)
    _mutate(approval_path, kind="monitor")

    result = _request(runner, plan_path, approval_path)

    assert result.exit_code == 1
    assert "kind mismatch" in result.output


def test_request_rejects_a_name_mismatch(runner: CliRunner, tmp_path: Path) -> None:
    plan_path, approval_path = _valid_pair(tmp_path)
    _mutate(approval_path, name="other_name")

    result = _request(runner, plan_path, approval_path)

    assert result.exit_code == 1
    assert "name mismatch" in result.output


def test_request_rejects_an_approval_that_forbids_execution(
    runner: CliRunner, tmp_path: Path
) -> None:
    plan_path, approval_path = _valid_pair(tmp_path)
    _mutate(approval_path, execution_permitted=False)

    result = _request(runner, plan_path, approval_path)

    assert result.exit_code == 1
    assert "approval does not permit execution" in result.output


def test_request_reports_a_builder_error(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path, approval_path = _valid_pair(tmp_path)

    def _raise(plan: object, approval: object) -> object:
        raise ValueError("request builder refused")

    monkeypatch.setattr(_execution, "_build_plugin_execution_request", _raise)

    result = _request(runner, plan_path, approval_path)

    assert result.exit_code == 1
    assert "request builder refused" in result.output


def test_request_emits_an_approval_artefact(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path, approval_path = _valid_pair(tmp_path)
    approval_payload = json.loads(approval_path.read_text(encoding="utf-8"))

    def _return_approval(plan: object, approval: PluginExecutionApproval) -> object:
        return approval

    monkeypatch.setattr(_execution, "_build_plugin_execution_request", _return_approval)

    result = _request(runner, plan_path, approval_path)

    assert result.exit_code == 0, result.output
    emitted = json.loads(result.output)
    assert emitted["plan_hash"] == approval_payload["plan_hash"]
