# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Plugin capability runtime loading and execution

"""Runtime loading, execution planning, and execution of plugin capabilities."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from scpn_phase_orchestrator import __version__

from ._shared import _VALID_KINDS, _record_hash, _require_identifier, _require_non_empty
from .manifest import validate_plugin_manifest
from .policy import _DEFAULT_RUNTIME_LOAD_POLICY, PluginRuntimeExecutionPolicy
from .request import validate_plugin_execution_request

if TYPE_CHECKING:
    from ._shared import PluginKind
    from .manifest import PluginCapability, PluginManifest
    from .policy import PluginRuntimeLoadPolicy
    from .request import PluginExecutionRequest


@dataclass(frozen=True)
class LoadedPluginCapability:
    """Resolved plugin runtime target with deterministic audit evidence."""

    manifest: PluginManifest
    capability: PluginCapability
    target_object: object
    audit_record: dict[str, object]


@dataclass(frozen=True)
class ExecutedPluginCapability:
    """Result of an explicitly approved plugin runtime invocation."""

    loaded: LoadedPluginCapability
    result: object
    audit_record: dict[str, object]


@dataclass(frozen=True)
class PluginExecutionPlan:
    """Prepared plugin-capability invocation plan without executing it."""

    manifest: PluginManifest
    capability: PluginCapability
    argument_count: int
    keyword_names: tuple[str, ...]
    target_hash: str
    plan_hash: str
    audit_record: dict[str, object]


def load_plugin_capability(
    manifest: PluginManifest,
    kind: PluginKind,
    name: str,
    *,
    policy: PluginRuntimeLoadPolicy | None = None,
) -> LoadedPluginCapability:
    """Resolve a declared plugin capability under an explicit runtime policy.

    Runtime loading is Python-owned and disabled by default. Callers must pass a
    policy with ``loading_permitted=True`` after their deployment boundary has
    approved the manifest. The target must be declared by the manifest, match the
    requested capability kind/name, and resolve inside the plugin package unless
    the policy explicitly relaxes that check.

    Parameters
    ----------
    manifest : PluginManifest
        The manifest object.
    kind : PluginKind
        The plugin capability kind.
    name : str
        The capability or resource name.
    policy : PluginRuntimeLoadPolicy | None
        The runtime policy, or ``None``.

    Returns
    -------
    LoadedPluginCapability
        A declared plugin capability under an explicit runtime policy.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    PermissionError
        If the operation is not permitted by policy.
    TypeError
        If an argument has the wrong type.
    """
    validate_plugin_manifest(manifest)
    if policy is None:
        policy = _DEFAULT_RUNTIME_LOAD_POLICY
    _require_identifier(name, "capability name")
    if kind not in _VALID_KINDS:
        raise ValueError(f"unsupported plugin capability kind: {kind}")
    if not policy.loading_permitted:
        raise PermissionError("plugin runtime loading is disabled by policy")
    if kind not in policy.allowed_kinds:
        raise ValueError(f"{kind} capability is not permitted by runtime load policy")

    capability = _select_capability(manifest, kind, name)
    module_name, attribute_path = _parse_target(capability.target)
    if policy.require_package_target and not _target_within_package(
        module_name,
        manifest.package,
    ):
        raise ValueError(
            f"capability target {capability.target!r} is outside plugin package "
            f"{manifest.package!r}"
        )

    module = importlib.import_module(module_name)
    target_object = _resolve_attribute_path(module, attribute_path, capability.target)
    if not callable(target_object):
        raise TypeError(f"capability target {capability.target!r} must be callable")

    audit_record = _runtime_load_audit_record(
        manifest=manifest,
        capability=capability,
        policy=policy,
        module_name=module_name,
        callable_target=True,
    )
    return LoadedPluginCapability(
        manifest=manifest,
        capability=capability,
        target_object=target_object,
        audit_record=audit_record,
    )


def build_plugin_execution_plan(
    manifest: PluginManifest,
    kind: PluginKind,
    name: str,
    *,
    args: tuple[object, ...] = (),
    kwargs: dict[str, object] | None = None,
    policy: PluginRuntimeExecutionPolicy | None = None,
) -> PluginExecutionPlan:
    """Build a deterministic runtime invocation plan without executing it.

    Parameters
    ----------
    manifest : PluginManifest
        The manifest object.
    kind : PluginKind
        The plugin capability kind.
    name : str
        The capability or resource name.
    args : tuple[object, ...]
        Positional arguments for the capability.
    kwargs : dict[str, object] | None
        Keyword arguments for the capability, or ``None``.
    policy : PluginRuntimeExecutionPolicy | None
        The runtime policy, or ``None``.

    Returns
    -------
    PluginExecutionPlan
        A deterministic runtime invocation plan without executing it.

    Raises
    ------
    PermissionError
        If the operation is not permitted by policy.
    TypeError
        If an argument has the wrong type.
    ValueError
        If the inputs are invalid or inconsistent.
    """
    if policy is None:
        policy = PluginRuntimeExecutionPolicy()
    if not policy.execution_permitted:
        raise PermissionError("plugin runtime execution is disabled by policy")
    if not isinstance(args, tuple):
        raise TypeError("args must be a tuple")
    if kwargs is None:
        kwargs = {}
    if not isinstance(kwargs, dict):
        raise TypeError("kwargs must be a dictionary")
    for key in kwargs:
        _require_identifier(key, "plugin execution keyword")

    validate_plugin_manifest(manifest)
    _require_identifier(name, "capability name")
    if kind not in _VALID_KINDS:
        raise ValueError(f"unsupported plugin capability kind: {kind}")

    capability = _select_capability(manifest, kind, name)
    load_policy = policy.to_load_policy()
    if kind not in load_policy.allowed_kinds:
        raise ValueError(f"{kind} capability is not permitted by runtime load policy")

    module_name, _attribute_path = _parse_target(capability.target)
    if load_policy.require_package_target and not _target_within_package(
        module_name,
        manifest.package,
    ):
        raise ValueError(
            f"capability target {capability.target!r} is outside plugin package "
            f"{manifest.package!r}"
        )

    target_hash = _preimport_target_hash(
        manifest=manifest,
        capability=capability,
        policy=policy,
    )
    _assert_execution_target_hash_approved(
        manifest=manifest,
        capability=capability,
        policy=policy,
    )

    keyword_names = tuple(sorted(kwargs))
    audit_record = _runtime_execution_plan_audit_record(
        manifest=manifest,
        capability=capability,
        policy=policy,
        target_hash=target_hash,
        argument_count=len(args),
        keyword_names=keyword_names,
    )
    return PluginExecutionPlan(
        manifest=manifest,
        capability=capability,
        argument_count=len(args),
        keyword_names=keyword_names,
        target_hash=target_hash,
        plan_hash=str(audit_record["plan_hash"]),
        audit_record=audit_record,
    )


def execute_plugin_capability(
    manifest: PluginManifest,
    kind: PluginKind,
    name: str,
    *,
    args: tuple[object, ...] = (),
    kwargs: dict[str, object] | None = None,
    policy: PluginRuntimeExecutionPolicy | None = None,
) -> ExecutedPluginCapability:
    """Invoke a declared plugin capability under an explicit execution policy.

    Execution is denied before any target import unless both loading and
    execution are explicitly permitted. Audit metadata records the invocation
    shape without serialising argument values, so secrets and large payloads are
    not copied into the audit record.

    Parameters
    ----------
    manifest : PluginManifest
        The manifest object.
    kind : PluginKind
        The plugin capability kind.
    name : str
        The capability or resource name.
    args : tuple[object, ...]
        Positional arguments for the capability.
    kwargs : dict[str, object] | None
        Keyword arguments for the capability, or ``None``.
    policy : PluginRuntimeExecutionPolicy | None
        The runtime policy, or ``None``.

    Returns
    -------
    ExecutedPluginCapability
        Invoke a declared plugin capability under an explicit execution policy.

    Raises
    ------
    TypeError
        If an argument has the wrong type.
    """
    if kwargs is None:
        kwargs = {}
    if policy is None:
        policy = PluginRuntimeExecutionPolicy()
    plan = build_plugin_execution_plan(
        manifest,
        kind,
        name,
        args=args,
        kwargs=kwargs,
        policy=policy,
    )
    loaded = load_plugin_capability(
        manifest,
        kind,
        name,
        policy=policy.to_load_policy(),
    )
    target = loaded.target_object
    if not callable(target):
        raise TypeError("loaded plugin target must be callable")
    result = target(*args, **kwargs)
    audit_record = _runtime_execute_audit_record(
        loaded=loaded,
        policy=policy,
        args=args,
        kwargs=kwargs,
        result=result,
        plan_hash=plan.plan_hash,
    )
    return ExecutedPluginCapability(
        loaded=loaded,
        result=result,
        audit_record=audit_record,
    )


def execute_plugin_execution_request(
    manifest: PluginManifest,
    request: PluginExecutionRequest,
    *,
    args: tuple[object, ...] = (),
    kwargs: dict[str, object] | None = None,
    revoked_request_hashes: tuple[str, ...] = (),
) -> ExecutedPluginCapability:
    """Invoke a plugin only when the approved request matches this call shape.

    The request-bound path validates manifest identity, invocation shape, plan
    hash, and target hash before importing the plugin module. Argument values
    remain outside audit records; only positional count and keyword names
    participate in the plan hash.

    Parameters
    ----------
    manifest : PluginManifest
        The manifest object.
    request : PluginExecutionRequest
        The plugin execution request.
    args : tuple[object, ...]
        Positional arguments for the capability.
    kwargs : dict[str, object] | None
        Keyword arguments for the capability, or ``None``.
    revoked_request_hashes : tuple[str, ...]
        Hashes of revoked execution requests.

    Returns
    -------
    ExecutedPluginCapability
        Invoke a plugin only when the approved request matches this call shape.

    Raises
    ------
    TypeError
        If an argument has the wrong type.
    ValueError
        If the inputs are invalid or inconsistent.
    PermissionError
        If the operation is not permitted by policy.
    """
    if kwargs is None:
        kwargs = {}
    if not isinstance(args, tuple):
        raise TypeError("args must be a tuple")
    if not isinstance(kwargs, dict):
        raise TypeError("kwargs must be a dictionary")
    if request.schema != "scpn_plugin_runtime_execution_request_v1":
        raise ValueError(
            "request schema must be scpn_plugin_runtime_execution_request_v1"
        )
    validate_plugin_execution_request(
        request,
        revoked_request_hashes=revoked_request_hashes,
    )
    if manifest.name != request.plugin:
        raise ValueError("request plugin does not match manifest")

    policy = request.to_execution_policy()
    plan = build_plugin_execution_plan(
        manifest,
        request.kind,
        request.name,
        args=args,
        kwargs=kwargs,
        policy=policy,
    )
    if plan.plan_hash != request.plan_hash:
        raise PermissionError("execution request plan hash mismatch")
    if plan.target_hash != request.target_hash:
        raise PermissionError("execution request target hash mismatch")

    executed = execute_plugin_capability(
        manifest,
        request.kind,
        request.name,
        args=args,
        kwargs=kwargs,
        policy=policy,
    )
    audit_record = {
        **executed.audit_record,
        "request_hash": request.audit_record["request_hash"],
        "approval_hash": request.approval_hash,
        "operator_identity": request.operator_identity,
        "approval_reference": request.approval_reference,
    }
    audit_record["execution_hash"] = _record_hash(audit_record)
    return ExecutedPluginCapability(
        loaded=executed.loaded,
        result=executed.result,
        audit_record=audit_record,
    )


def _select_capability(
    manifest: PluginManifest,
    kind: PluginKind,
    name: str,
) -> PluginCapability:
    matches = tuple(
        capability
        for capability in manifest.capabilities
        if capability.kind == kind and capability.name == name
    )
    if not matches:
        raise LookupError(f"plugin capability {kind}:{name} is not declared")
    if len(matches) > 1:
        raise ValueError(f"plugin capability {kind}:{name} is declared more than once")
    return matches[0]


def _parse_target(target: str) -> tuple[str, str]:
    if ":" not in target:
        raise ValueError("capability target must use 'module:attribute' syntax")
    module_name, attribute_path = target.split(":", maxsplit=1)
    _require_non_empty(module_name, "capability target module")
    _require_non_empty(attribute_path, "capability target attribute")
    return module_name, attribute_path


def _target_within_package(module_name: str, package: str) -> bool:
    return module_name == package or module_name.startswith(f"{package}.")


def _resolve_attribute_path(
    module: object,
    attribute_path: str,
    target: str,
) -> object:
    current = module
    for attribute in attribute_path.split("."):
        _require_identifier(attribute, "capability target attribute")
        try:
            current = getattr(current, attribute)
        except AttributeError as exc:
            raise AttributeError(
                f"capability target {target!r} is not importable"
            ) from exc
    return current


def _runtime_load_audit_record(
    *,
    manifest: PluginManifest,
    capability: PluginCapability,
    policy: PluginRuntimeLoadPolicy,
    module_name: str,
    callable_target: bool,
) -> dict[str, object]:
    record = {
        "schema": "scpn_plugin_runtime_load_v1",
        "spo_version": __version__,
        "plugin": manifest.name,
        "plugin_version": manifest.version,
        "package": manifest.package,
        "kind": capability.kind,
        "name": capability.name,
        "target": capability.target,
        "module": module_name,
        "version": capability.version,
        "channels": list(capability.channels),
        "knobs": list(capability.knobs),
        "loading_permitted": policy.loading_permitted,
        "load_policy": "python_owned_explicit",
        "require_package_target": policy.require_package_target,
        "allowed_kinds": list(policy.allowed_kinds),
        "callable": callable_target,
    }
    record["target_hash"] = _record_hash(record)
    record["load_hash"] = _record_hash(record)
    return record


def _runtime_execute_audit_record(
    *,
    loaded: LoadedPluginCapability,
    policy: PluginRuntimeExecutionPolicy,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    result: object,
    plan_hash: str | None = None,
) -> dict[str, object]:
    record = {
        "schema": "scpn_plugin_runtime_execute_v1",
        "load_hash": loaded.audit_record["load_hash"],
        "target_hash": loaded.audit_record["target_hash"],
        "plugin": loaded.manifest.name,
        "plugin_version": loaded.manifest.version,
        "package": loaded.manifest.package,
        "kind": loaded.capability.kind,
        "name": loaded.capability.name,
        "target": loaded.capability.target,
        "loading_permitted": policy.loading_permitted,
        "execution_permitted": policy.execution_permitted,
        "target_hash_approved": _execution_target_hash_approved(
            str(loaded.audit_record["target_hash"]),
            policy,
        ),
        "approved_target_hashes": list(policy.approved_target_hashes),
        "load_policy": "python_owned_explicit",
        "argument_count": len(args),
        "keyword_names": sorted(kwargs),
        "result_type": type(result).__name__,
    }
    if plan_hash is not None:
        record["plan_hash"] = plan_hash
    record["execution_hash"] = _record_hash(record)
    return record


def _runtime_execution_plan_audit_record(
    *,
    manifest: PluginManifest,
    capability: PluginCapability,
    policy: PluginRuntimeExecutionPolicy,
    target_hash: str,
    argument_count: int,
    keyword_names: tuple[str, ...],
) -> dict[str, object]:
    record = {
        "schema": "scpn_plugin_runtime_execution_plan_v1",
        "schema_version": "1.0.0",
        "spo_version": __version__,
        "plugin": manifest.name,
        "plugin_version": manifest.version,
        "package": manifest.package,
        "kind": capability.kind,
        "name": capability.name,
        "target": capability.target,
        "target_hash": target_hash,
        "load_hash": target_hash,
        "argument_count": argument_count,
        "keyword_names": list(keyword_names),
        "execution_permitted": policy.execution_permitted,
        "loading_permitted": policy.loading_permitted,
        "require_package_target": policy.require_package_target,
        "allowed_kinds": list(policy.allowed_kinds),
        "require_target_hash_approval": policy.require_target_hash_approval,
        "approved_target_hashes": list(policy.approved_target_hashes),
        "target_hash_approved": _execution_target_hash_approved(
            target_hash=target_hash,
            policy=policy,
        ),
    }
    record["plan_hash"] = _record_hash(record)
    return record


def _assert_execution_target_hash_approved(
    *,
    manifest: PluginManifest,
    capability: PluginCapability,
    policy: PluginRuntimeExecutionPolicy,
) -> None:
    if not policy.require_target_hash_approval:
        return
    expected_hash = _preimport_target_hash(
        manifest=manifest,
        capability=capability,
        policy=policy,
    )
    if expected_hash not in policy.approved_target_hashes:
        raise PermissionError(
            f"plugin runtime target hash {expected_hash} is not approved"
        )


def _preimport_target_hash(
    *,
    manifest: PluginManifest,
    capability: PluginCapability,
    policy: PluginRuntimeExecutionPolicy,
) -> str:
    module_name, _attribute_path = _parse_target(capability.target)
    load_policy = policy.to_load_policy()
    record = _runtime_load_audit_record(
        manifest=manifest,
        capability=capability,
        policy=load_policy,
        module_name=module_name,
        callable_target=True,
    )
    return str(record["target_hash"])


def _execution_target_hash_approved(
    target_hash: str,
    policy: PluginRuntimeExecutionPolicy,
) -> bool:
    if not policy.require_target_hash_approval:
        return False
    return target_hash in policy.approved_target_hashes
