# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — scheduler payload loader tests

"""Scheduler JSON payload-loader contracts for the ``spo plugins`` CLI."""

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
    "scheduler_hash": "a" * 64,
    "plan_hash": "b" * 64,
    "execution_hash": "c" * 64,
    "handoff_hash": "d" * 64,
    "entry_hash": "e" * 64,
    "handoff_action_hash": "f" * 64,
    "action_hash": "1" * 64,
    "request_hash": "2" * 64,
    "telemetry_hash": "3" * 64,
    "adapter_handoff_hash": "4" * 64,
    "adapter_entry_hash": "5" * 64,
    "acknowledgement_hash": "6" * 64,
}


def _load_payload_module(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> types.ModuleType:
    """Load a payload-loader module without importing the broad CLI package."""
    module_name = f"{_PAYLOADS_PACKAGE}.{name}"
    spec = importlib.util.spec_from_file_location(
        module_name, _PAYLOADS_DIR / f"{name}.py"
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {module_name}")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _load_scheduler_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Return the real scheduler loader module with its shared dependency."""
    package = types.ModuleType(_PAYLOADS_PACKAGE)
    package.__path__ = [str(_PAYLOADS_DIR)]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _PAYLOADS_PACKAGE, package)
    _load_payload_module("_shared", monkeypatch)
    return _load_payload_module("scheduler", monkeypatch)


def _loader(module: types.ModuleType, name: str) -> Loader:
    """Return a typed scheduler loader function from the loaded module."""
    return cast("Loader", getattr(module, name))


def _expect_error(loader: Loader, payload: Payload, message: str) -> None:
    """Assert that a malformed payload fails closed with a useful message."""
    with pytest.raises(click.ClickException) as exc_info:
        loader(payload)
    assert message in str(exc_info.value)


def _queue_entry() -> Payload:
    return {
        "entry_hash": _HASHES["entry_hash"],
        "handoff_action_hash": _HASHES["handoff_action_hash"],
        "action_hash": _HASHES["action_hash"],
        "request_hash": _HASHES["request_hash"],
        "action_type": "renew_approval",
        "priority": 1,
        "schedule_epoch": 0,
        "scheduler_command_template": "spo plugins approve-execution-plan PLAN_JSON",
    }


def _queue_payload() -> Payload:
    return {
        "schema": f"{_PREFIX}_scheduler_queue_v1",
        "scheduler_hash": _HASHES["scheduler_hash"],
        "plan_hash": _HASHES["plan_hash"],
        "execution_hash": _HASHES["execution_hash"],
        "handoff_hash": _HASHES["handoff_hash"],
        "queue_entry_count": 1,
        "queue_entries": [_queue_entry()],
    }


def _telemetry_payload() -> Payload:
    return {
        "schema": f"{_PREFIX}_scheduler_telemetry_v1",
        "telemetry_hash": _HASHES["telemetry_hash"],
        "plan_hash": _HASHES["plan_hash"],
        "execution_hash": _HASHES["execution_hash"],
        "handoff_hash": _HASHES["handoff_hash"],
        "scheduler_hash": _HASHES["scheduler_hash"],
        "queue_entry_count": 1,
        "rows": [{"entry_hash": _HASHES["entry_hash"], "state": "overdue"}],
    }


def _adapter_handoff_payload() -> Payload:
    return {
        "schema": f"{_PREFIX}_scheduler_adapter_handoff_v1",
        "adapter_handoff_hash": _HASHES["adapter_handoff_hash"],
        "telemetry_hash": _HASHES["telemetry_hash"],
        "plan_hash": _HASHES["plan_hash"],
        "execution_hash": _HASHES["execution_hash"],
        "entry_count": 1,
        "entries": [
            {
                "adapter_entry_hash": _HASHES["adapter_entry_hash"],
                "entry_hash": _HASHES["entry_hash"],
                "action_hash": _HASHES["action_hash"],
                "request_hash": _HASHES["request_hash"],
            }
        ],
    }


def _acknowledgement_payload() -> Payload:
    return {
        "schema": f"{_PREFIX}_scheduler_acknowledgement_v1",
        "acknowledgement_hash": _HASHES["acknowledgement_hash"],
        "adapter_handoff_hash": _HASHES["adapter_handoff_hash"],
        "telemetry_hash": _HASHES["telemetry_hash"],
        "plan_hash": _HASHES["plan_hash"],
        "execution_hash": _HASHES["execution_hash"],
        "adapter_entry_hash": _HASHES["adapter_entry_hash"],
        "entry_hash": _HASHES["entry_hash"],
        "action_hash": _HASHES["action_hash"],
        "request_hash": _HASHES["request_hash"],
        "state": "completed",
        "acknowledged_by": "deployment_scheduler",
        "external_reference": "ticket-123",
    }


@pytest.mark.parametrize(
    ("loader_name", "payload"),
    (
        ("_load_lifecycle_remediation_scheduler_queue_payload", _queue_payload()),
        (
            "_load_lifecycle_remediation_scheduler_telemetry_payload",
            _telemetry_payload(),
        ),
        (
            "_load_lifecycle_remediation_scheduler_adapter_handoff_payload",
            _adapter_handoff_payload(),
        ),
        (
            "_load_lifecycle_remediation_scheduler_acknowledgement_payload",
            _acknowledgement_payload(),
        ),
    ),
)
def test_scheduler_loaders_return_valid_payloads(
    monkeypatch: pytest.MonkeyPatch,
    loader_name: str,
    payload: Payload,
) -> None:
    """Valid scheduler payloads pass through unchanged."""
    module = _load_scheduler_module(monkeypatch)
    assert _loader(module, loader_name)(payload) is payload


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("schema", "expected scpn_plugin_execution_request_lifecycle_remediation"),
        ("count_type", "queue_entry_count must be non-negative"),
        ("entries_type", "queue_entries must be a list"),
        ("count_mismatch", "queue_entry_count does not match queue_entries"),
        ("entry_type", "entry must be object"),
        ("action_type", "unsupported action_type"),
        ("priority", "priority must be a positive integer"),
        ("schedule_epoch", "schedule_epoch must be non-negative integer"),
        ("template", "scheduler_command_template must be non-empty"),
    ),
)
def test_scheduler_queue_loader_rejects_structural_mismatches(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    """Queue payload validation fails closed on malformed scheduler contracts."""
    payload = _queue_payload()
    if case == "schema":
        payload["schema"] = "wrong"
    elif case == "count_type":
        payload["queue_entry_count"] = -1
    elif case == "entries_type":
        payload["queue_entries"] = "not-a-list"
    elif case == "count_mismatch":
        payload["queue_entry_count"] = 2
    elif case == "entry_type":
        payload["queue_entries"] = ["not-an-object"]
    else:
        entry = cast("Payload", cast("list[object]", payload["queue_entries"])[0])
        if case == "action_type":
            entry["action_type"] = "unknown"
        elif case == "priority":
            entry["priority"] = 0
        elif case == "schedule_epoch":
            entry["schedule_epoch"] = -1
        elif case == "template":
            entry["scheduler_command_template"] = ""

    module = _load_scheduler_module(monkeypatch)
    _expect_error(
        _loader(module, "_load_lifecycle_remediation_scheduler_queue_payload"),
        payload,
        message,
    )


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("schema", "expected scpn_plugin_execution_request_lifecycle_remediation"),
        ("count_type", "queue_entry_count must be non-negative"),
        ("rows_type", "rows must be a list"),
        ("count_mismatch", "queue_entry_count does not match rows"),
    ),
)
def test_scheduler_telemetry_loader_rejects_structural_mismatches(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    """Telemetry payload validation fails closed on malformed scheduler rows."""
    payload = _telemetry_payload()
    if case == "schema":
        payload["schema"] = "wrong"
    elif case == "count_type":
        payload["queue_entry_count"] = -1
    elif case == "rows_type":
        payload["rows"] = "not-a-list"
    elif case == "count_mismatch":
        payload["queue_entry_count"] = 2

    module = _load_scheduler_module(monkeypatch)
    _expect_error(
        _loader(module, "_load_lifecycle_remediation_scheduler_telemetry_payload"),
        payload,
        message,
    )


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("schema", "expected scpn_plugin_execution_request_lifecycle_remediation"),
        ("count_type", "entry_count must be non-negative"),
        ("entries_type", "entries must be list"),
        ("count_mismatch", "entry_count does not match entries"),
        ("entry_type", "entry must be object"),
    ),
)
def test_scheduler_adapter_handoff_loader_rejects_structural_mismatches(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    """Adapter handoff validation fails closed on malformed entry contracts."""
    payload = _adapter_handoff_payload()
    if case == "schema":
        payload["schema"] = "wrong"
    elif case == "count_type":
        payload["entry_count"] = -1
    elif case == "entries_type":
        payload["entries"] = "not-a-list"
    elif case == "count_mismatch":
        payload["entry_count"] = 2
    elif case == "entry_type":
        payload["entries"] = ["not-an-object"]

    module = _load_scheduler_module(monkeypatch)
    _expect_error(
        _loader(
            module,
            "_load_lifecycle_remediation_scheduler_adapter_handoff_payload",
        ),
        payload,
        message,
    )


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("schema", "expected scpn_plugin_execution_request_lifecycle_remediation"),
        ("state", "unsupported state"),
        ("acknowledged_by", "acknowledged_by must be non-empty"),
        ("external_reference", "external_reference must be non-empty"),
    ),
)
def test_scheduler_acknowledgement_loader_rejects_structural_mismatches(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    """Acknowledgement validation fails closed on malformed worker records."""
    payload = _acknowledgement_payload()
    if case == "schema":
        payload["schema"] = "wrong"
    elif case == "state":
        payload["state"] = "unknown"
    elif case == "acknowledged_by":
        payload["acknowledged_by"] = ""
    elif case == "external_reference":
        payload["external_reference"] = ""

    module = _load_scheduler_module(monkeypatch)
    _expect_error(
        _loader(
            module,
            "_load_lifecycle_remediation_scheduler_acknowledgement_payload",
        ),
        payload,
        message,
    )
