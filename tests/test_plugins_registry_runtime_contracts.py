# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Plugin registry runtime contracts

"""Strict branch-contract tests for plugin registry runtime execution."""

from __future__ import annotations

import types
from typing import TypeAlias, cast

import pytest

from scpn_phase_orchestrator.plugins import (
    LoadedPluginCapability,
    PluginCapability,
    PluginExecutionRequest,
    PluginManifest,
    PluginRuntimeExecutionPolicy,
    PluginRuntimeLoadPolicy,
    build_plugin_execution_approval,
    build_plugin_execution_plan,
    build_plugin_execution_request,
    execute_plugin_capability,
    execute_plugin_execution_request,
    load_plugin_capability,
)
from scpn_phase_orchestrator.plugins.registry import runtime as runtime_mod
from scpn_phase_orchestrator.plugins.registry._shared import PluginKind

CapabilityTuple: TypeAlias = tuple[PluginCapability, ...]


def _manifest(capabilities: CapabilityTuple | None = None) -> PluginManifest:
    """Return a minimal plugin manifest for runtime-contract tests."""
    if capabilities is None:
        capabilities = (
            PluginCapability(
                kind="monitor",
                name="frequency_drift",
                target="runtime_pack.monitors:FrequencyDriftMonitor",
                channels=("frequency",),
            ),
        )
    return PluginManifest(
        name="runtime_pack",
        version="0.1.0",
        package="runtime_pack",
        capabilities=capabilities,
    )


def _execution_policy(
    *,
    allowed_kinds: tuple[PluginKind, ...] = (
        "actuator",
        "bridge",
        "extractor",
        "monitor",
    ),
) -> PluginRuntimeExecutionPolicy:
    """Return an execution policy that permits loading and execution."""
    return PluginRuntimeExecutionPolicy(
        loading_permitted=True,
        execution_permitted=True,
        allowed_kinds=allowed_kinds,
    )


def _approved_request() -> PluginExecutionRequest:
    """Return a valid approved runtime execution request."""
    manifest = _manifest()
    draft = build_plugin_execution_plan(
        manifest,
        "monitor",
        "frequency_drift",
        policy=_execution_policy(),
    )
    plan = build_plugin_execution_plan(
        manifest,
        "monitor",
        "frequency_drift",
        policy=PluginRuntimeExecutionPolicy(
            loading_permitted=True,
            execution_permitted=True,
            require_target_hash_approval=True,
            approved_target_hashes=(draft.target_hash,),
        ),
    )
    approval = build_plugin_execution_approval(
        plan,
        operator_identity="operator_alpha",
        approval_reference="REQ-2026-06-27-runtime",
        approval_reason="runtime contract verification",
    )
    return build_plugin_execution_request(plan, approval)


def test_runtime_loading_rejects_unsupported_kind_before_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime loading rejects invalid kinds before importing a target."""

    def fail_import(_module_name: str) -> object:
        raise AssertionError("invalid kind must fail before import")

    monkeypatch.setattr(
        "scpn_phase_orchestrator.plugins.registry.runtime.importlib.import_module",
        fail_import,
    )

    with pytest.raises(ValueError, match="unsupported plugin capability kind"):
        load_plugin_capability(
            _manifest(),
            cast("PluginKind", "invalid"),
            "frequency_drift",
            policy=PluginRuntimeLoadPolicy(loading_permitted=True),
        )


def test_runtime_loading_rejects_noncallable_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime loading fails closed when the declared target is not callable."""
    module = types.SimpleNamespace(FrequencyDriftMonitor=object())
    monkeypatch.setattr(
        "scpn_phase_orchestrator.plugins.registry.runtime.importlib.import_module",
        lambda _module_name: module,
    )

    with pytest.raises(TypeError, match="must be callable"):
        load_plugin_capability(
            _manifest(),
            "monitor",
            "frequency_drift",
            policy=PluginRuntimeLoadPolicy(loading_permitted=True),
        )


def test_execution_plan_rejects_non_tuple_args() -> None:
    """Execution plans require tuple positional arguments."""
    with pytest.raises(TypeError, match="args must be a tuple"):
        build_plugin_execution_plan(
            _manifest(),
            "monitor",
            "frequency_drift",
            args=cast("tuple[object, ...]", ["bad"]),
            policy=_execution_policy(),
        )


def test_execution_plan_rejects_non_dict_kwargs() -> None:
    """Execution plans require dictionary keyword arguments."""
    with pytest.raises(TypeError, match="kwargs must be a dictionary"):
        build_plugin_execution_plan(
            _manifest(),
            "monitor",
            "frequency_drift",
            kwargs=cast("dict[str, object]", (("scale", 1.0),)),
            policy=_execution_policy(),
        )


def test_execution_plan_rejects_unsupported_kind() -> None:
    """Execution plans reject unsupported runtime kinds."""
    with pytest.raises(ValueError, match="unsupported plugin capability kind"):
        build_plugin_execution_plan(
            _manifest(),
            cast("PluginKind", "invalid"),
            "frequency_drift",
            policy=_execution_policy(),
        )


def test_execution_plan_rejects_kind_excluded_by_policy() -> None:
    """Execution plans enforce allowed runtime kinds before planning."""
    with pytest.raises(ValueError, match="not permitted by runtime load policy"):
        build_plugin_execution_plan(
            _manifest(),
            "monitor",
            "frequency_drift",
            policy=_execution_policy(allowed_kinds=("actuator",)),
        )


def test_execution_plan_rejects_target_outside_manifest_package() -> None:
    """Execution plans enforce package-scoped targets before import."""
    manifest = _manifest(
        (
            PluginCapability(
                kind="monitor",
                name="escape",
                target="other_pack.monitors:Monitor",
                channels=("frequency",),
            ),
        ),
    )

    with pytest.raises(ValueError, match="outside plugin package"):
        build_plugin_execution_plan(
            manifest,
            "monitor",
            "escape",
            policy=_execution_policy(),
        )


def test_execute_plugin_capability_rejects_mutated_noncallable_loaded_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execution rechecks callability of the loaded target before invoking it."""
    manifest = _manifest()
    capability = manifest.capabilities[0]
    loaded = LoadedPluginCapability(
        manifest=manifest,
        capability=capability,
        target_object=object(),
        audit_record={
            "load_hash": "0" * 64,
            "target_hash": build_plugin_execution_plan(
                manifest,
                "monitor",
                "frequency_drift",
                policy=_execution_policy(),
            ).target_hash,
        },
    )

    def load_noncallable_target(
        *_args: object,
        **_kwargs: object,
    ) -> LoadedPluginCapability:
        return loaded

    monkeypatch.setattr(runtime_mod, "load_plugin_capability", load_noncallable_target)

    with pytest.raises(TypeError, match="loaded plugin target must be callable"):
        execute_plugin_capability(
            manifest,
            "monitor",
            "frequency_drift",
            policy=_execution_policy(),
        )


def test_request_execution_rejects_non_tuple_args() -> None:
    """Request-bound execution requires tuple positional arguments."""
    with pytest.raises(TypeError, match="args must be a tuple"):
        execute_plugin_execution_request(
            _manifest(),
            _approved_request(),
            args=cast("tuple[object, ...]", ["bad"]),
        )


def test_request_execution_rejects_non_dict_kwargs() -> None:
    """Request-bound execution requires dictionary keyword arguments."""
    with pytest.raises(TypeError, match="kwargs must be a dictionary"):
        execute_plugin_execution_request(
            _manifest(),
            _approved_request(),
            kwargs=cast("dict[str, object]", (("scale", 1.0),)),
        )


def test_request_execution_rejects_mutated_schema() -> None:
    """Request-bound execution rechecks the request schema before validation."""
    request = _approved_request()
    object.__setattr__(request, "schema", "wrong_schema")

    with pytest.raises(ValueError, match="request schema must be"):
        execute_plugin_execution_request(_manifest(), request)


def test_request_execution_rejects_target_hash_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Request-bound execution compares the plan target hash before import."""
    request = _approved_request()
    object.__setattr__(request, "target_hash", "0" * 64)

    def accept_request(
        checked_request: PluginExecutionRequest,
        *,
        revoked_request_hashes: tuple[str, ...] = (),
    ) -> PluginExecutionRequest:
        return checked_request

    monkeypatch.setattr(
        runtime_mod,
        "validate_plugin_execution_request",
        accept_request,
    )

    with pytest.raises(PermissionError, match="target hash mismatch"):
        execute_plugin_execution_request(_manifest(), request)


def test_select_capability_rejects_missing_declaration() -> None:
    """Capability selection rejects undeclared runtime targets."""
    with pytest.raises(LookupError, match="is not declared"):
        runtime_mod._select_capability(_manifest(), "monitor", "missing")


def test_select_capability_rejects_duplicate_declaration() -> None:
    """Capability selection rejects duplicate kind/name declarations."""
    capability = PluginCapability(
        kind="monitor",
        name="frequency_drift",
        target="runtime_pack.monitors:FrequencyDriftMonitor",
        channels=("frequency",),
    )
    manifest = _manifest((capability, capability))

    with pytest.raises(ValueError, match="declared more than once"):
        runtime_mod._select_capability(manifest, "monitor", "frequency_drift")


def test_parse_target_rejects_missing_module_attribute_separator() -> None:
    """Target parsing requires module:attribute syntax."""
    with pytest.raises(ValueError, match="module:attribute"):
        runtime_mod._parse_target("runtime_pack.monitors.FrequencyDriftMonitor")


def test_resolve_attribute_path_reports_missing_attribute() -> None:
    """Attribute resolution reports non-importable dotted targets."""
    module = types.SimpleNamespace()

    with pytest.raises(AttributeError, match="is not importable"):
        runtime_mod._resolve_attribute_path(
            module,
            "MissingTarget",
            "runtime_pack.monitors:MissingTarget",
        )
