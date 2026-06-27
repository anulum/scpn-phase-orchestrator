# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — remediation payload loader tests

"""Remediation JSON payload-loader contracts for the ``spo plugins`` CLI."""

from __future__ import annotations

import importlib
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias, cast

import click
import pytest

Payload: TypeAlias = dict[str, object]
Loader: TypeAlias = Callable[[Payload], Payload]

_PAYLOADS_PACKAGE = "scpn_phase_orchestrator.runtime.cli._payloads"
_PAYLOADS_DIR = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "scpn_phase_orchestrator"
    / "runtime"
    / "cli"
    / "_payloads"
)
_PREFIX = "scpn_plugin_execution_request_lifecycle_remediation"
_HASHES = {
    "plan_hash": "a" * 64,
    "drilldown_hash": "b" * 64,
    "action_hash": "c" * 64,
    "request_hash": "d" * 64,
    "store_hash": "e" * 64,
    "policy_hash": "f" * 64,
    "summary_hash": "1" * 64,
    "status_hash": "2" * 64,
    "execution_hash": "3" * 64,
    "handoff_hash": "4" * 64,
    "handoff_action_hash": "5" * 64,
}


def _load_payload_module(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> types.ModuleType:
    """Load a payload module under its production package name."""
    module_name = f"{_PAYLOADS_PACKAGE}.{name}"
    spec = importlib.util.spec_from_file_location(
        module_name,
        _PAYLOADS_DIR / f"{name}.py",
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {module_name}")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _load_remediation_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Return the real remediation loader module with shared validators loaded."""
    package = types.ModuleType(_PAYLOADS_PACKAGE)
    package.__path__ = [str(_PAYLOADS_DIR)]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _PAYLOADS_PACKAGE, package)
    _load_payload_module("_shared", monkeypatch)
    return _load_payload_module("remediation", monkeypatch)


def _loader(module: types.ModuleType, name: str) -> Loader:
    """Return a typed remediation loader function from the loaded module."""
    return cast("Loader", getattr(module, name))


def _expect_error(loader: Loader, payload: Payload, message: str) -> None:
    """Assert malformed payloads fail closed with the expected diagnostic."""
    with pytest.raises(click.ClickException) as exc_info:
        loader(payload)
    assert message in str(exc_info.value)


def _action() -> Payload:
    return {
        "action_hash": _HASHES["action_hash"],
        "request_hash": _HASHES["request_hash"],
        "store_hash": _HASHES["store_hash"],
        "policy_hash": _HASHES["policy_hash"],
        "summary_hash": _HASHES["summary_hash"],
        "action_type": "renew_approval",
        "priority": 1,
    }


def _plan_payload() -> Payload:
    return {
        "schema": f"{_PREFIX}_plan_v1",
        "plan_hash": _HASHES["plan_hash"],
        "drilldown_hash": _HASHES["drilldown_hash"],
        "action_count": 1,
        "actions": [_action()],
    }


def _status_payload() -> Payload:
    return {
        "schema": f"{_PREFIX}_action_status_v1",
        "status_hash": _HASHES["status_hash"],
        "action_hash": _HASHES["action_hash"],
        "plan_hash": _HASHES["plan_hash"],
        "state": "pending",
    }


def _dashboard_row() -> Payload:
    return {
        "action_hash": _HASHES["action_hash"],
        "request_hash": _HASHES["request_hash"],
        "state": "completed",
        "action_type": "persist_request",
        "priority": 1,
    }


def _dashboard_payload() -> Payload:
    return {
        "schema": f"{_PREFIX}_execution_dashboard_v1",
        "execution_hash": _HASHES["execution_hash"],
        "plan_hash": _HASHES["plan_hash"],
        "action_count": 1,
        "rows": [_dashboard_row()],
    }


def _handoff_action() -> Payload:
    return {
        "handoff_action_hash": _HASHES["handoff_action_hash"],
        "action_hash": _HASHES["action_hash"],
        "request_hash": _HASHES["request_hash"],
        "action_type": "confirm_external_write",
        "priority": 1,
        "deployment_command_template": (
            "spo plugins lifecycle-remediation-action-status PLAN_JSON ACTION_HASH "
            "--state completed"
        ),
    }


def _handoff_payload() -> Payload:
    return {
        "schema": f"{_PREFIX}_deployment_handoff_v1",
        "handoff_hash": _HASHES["handoff_hash"],
        "plan_hash": _HASHES["plan_hash"],
        "execution_hash": _HASHES["execution_hash"],
        "unresolved_action_count": 1,
        "handoff_actions": [_handoff_action()],
    }


@pytest.mark.parametrize(
    ("loader_name", "payload"),
    (
        ("_load_lifecycle_remediation_plan_payload", _plan_payload()),
        ("_load_lifecycle_remediation_action_status_payload", _status_payload()),
        (
            "_load_lifecycle_remediation_execution_dashboard_payload",
            _dashboard_payload(),
        ),
        (
            "_load_lifecycle_remediation_deployment_handoff_payload",
            _handoff_payload(),
        ),
    ),
)
def test_remediation_loaders_return_valid_payloads(
    monkeypatch: pytest.MonkeyPatch,
    loader_name: str,
    payload: Payload,
) -> None:
    """Valid remediation payloads pass through unchanged."""
    module = _load_remediation_module(monkeypatch)
    assert _loader(module, loader_name)(payload) is payload


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("schema", "expected scpn_plugin_execution_request_lifecycle_remediation"),
        ("count_type", "action_count must be non-negative"),
        ("actions_type", "actions must be a list"),
        ("count_mismatch", "action_count does not match actions"),
        ("action_type", "action must be an object"),
        ("unsupported_action", "unsupported action_type"),
        ("priority", "priority must be a positive integer"),
    ),
)
def test_remediation_plan_loader_rejects_structural_mismatches(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    """Plan validation fails closed on malformed action contracts."""
    payload = _plan_payload()
    if case == "schema":
        payload["schema"] = "wrong"
    elif case == "count_type":
        payload["action_count"] = -1
    elif case == "actions_type":
        payload["actions"] = "not-a-list"
    elif case == "count_mismatch":
        payload["action_count"] = 2
    elif case == "action_type":
        payload["actions"] = ["not-an-object"]
    else:
        action = cast("Payload", cast("list[object]", payload["actions"])[0])
        if case == "unsupported_action":
            action["action_type"] = "unknown"
        elif case == "priority":
            action["priority"] = 0

    module = _load_remediation_module(monkeypatch)
    _expect_error(
        _loader(module, "_load_lifecycle_remediation_plan_payload"),
        payload,
        message,
    )


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("schema", "expected scpn_plugin_execution_request_lifecycle_remediation"),
        ("state", "unsupported state"),
    ),
)
def test_remediation_action_status_loader_rejects_structural_mismatches(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    """Action-status validation rejects unknown schema and state records."""
    payload = _status_payload()
    if case == "schema":
        payload["schema"] = "wrong"
    elif case == "state":
        payload["state"] = "unknown"

    module = _load_remediation_module(monkeypatch)
    _expect_error(
        _loader(module, "_load_lifecycle_remediation_action_status_payload"),
        payload,
        message,
    )


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("schema", "expected scpn_plugin_execution_request_lifecycle_remediation"),
        ("count_type", "action_count must be non-negative"),
        ("rows_type", "rows must be a list"),
        ("count_mismatch", "action_count does not match rows"),
        ("row_type", "row must be object"),
        ("state", "unsupported state"),
        ("action_type", "unsupported action_type"),
        ("priority", "priority must be a positive integer"),
    ),
)
def test_remediation_dashboard_loader_rejects_structural_mismatches(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    """Execution-dashboard validation fails closed on malformed row records."""
    payload = _dashboard_payload()
    if case == "schema":
        payload["schema"] = "wrong"
    elif case == "count_type":
        payload["action_count"] = -1
    elif case == "rows_type":
        payload["rows"] = "not-a-list"
    elif case == "count_mismatch":
        payload["action_count"] = 2
    elif case == "row_type":
        payload["rows"] = ["not-an-object"]
    else:
        row = cast("Payload", cast("list[object]", payload["rows"])[0])
        if case == "state":
            row["state"] = "unknown"
        elif case == "action_type":
            row["action_type"] = "unknown"
        elif case == "priority":
            row["priority"] = 0

    module = _load_remediation_module(monkeypatch)
    _expect_error(
        _loader(module, "_load_lifecycle_remediation_execution_dashboard_payload"),
        payload,
        message,
    )


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("schema", "expected scpn_plugin_execution_request_lifecycle_remediation"),
        ("count_type", "unresolved_action_count must be non-negative"),
        ("actions_type", "handoff_actions must be a list"),
        ("count_mismatch", "unresolved_action_count does not match handoff_actions"),
        ("action_type", "action must be object"),
        ("unsupported_action", "unsupported action_type"),
        ("priority", "priority must be a positive integer"),
        ("template", "deployment_command_template must be non-empty"),
    ),
)
def test_remediation_handoff_loader_rejects_structural_mismatches(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    """Deployment-handoff validation fails closed on malformed handoff actions."""
    payload = _handoff_payload()
    if case == "schema":
        payload["schema"] = "wrong"
    elif case == "count_type":
        payload["unresolved_action_count"] = -1
    elif case == "actions_type":
        payload["handoff_actions"] = "not-a-list"
    elif case == "count_mismatch":
        payload["unresolved_action_count"] = 2
    elif case == "action_type":
        payload["handoff_actions"] = ["not-an-object"]
    else:
        action = cast("Payload", cast("list[object]", payload["handoff_actions"])[0])
        if case == "unsupported_action":
            action["action_type"] = "unknown"
        elif case == "priority":
            action["priority"] = 0
        elif case == "template":
            action["deployment_command_template"] = ""

    module = _load_remediation_module(monkeypatch)
    _expect_error(
        _loader(module, "_load_lifecycle_remediation_deployment_handoff_payload"),
        payload,
        message,
    )
