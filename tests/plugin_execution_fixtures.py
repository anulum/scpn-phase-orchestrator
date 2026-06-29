# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — shared plugin execution-plan/approval test fixtures

"""Shared builders for plugin execution-plan and approval CLI fixtures.

Extracted from ``tests/test_cli.py`` so the plugin execution-plan and approval
JSON builders can be reused by focused command tests without growing that module.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from scpn_phase_orchestrator.plugins import (
    PluginCapability,
    PluginManifest,
    PluginRuntimeExecutionPolicy,
    build_plugin_execution_approval,
    build_plugin_execution_plan,
)

DEFAULT_PLUGIN = "cli_plugin"
DEFAULT_KIND = "extractor"
DEFAULT_CAPABILITY = "phase"


def make_manifest(
    name: str = DEFAULT_PLUGIN,
    *,
    kind: str = DEFAULT_KIND,
    capability_name: str = DEFAULT_CAPABILITY,
) -> PluginManifest:
    """Return a single-capability plugin manifest for execution fixtures.

    Parameters
    ----------
    name : str
        Plugin and package name.
    kind : str
        Capability kind.
    capability_name : str
        Capability name.

    Returns
    -------
    PluginManifest
        The manifest with one matching capability.
    """
    return PluginManifest(
        name=name,
        version="0.1.0",
        package=name,
        capabilities=(
            PluginCapability(
                kind=kind,
                name=capability_name,
                target=f"{name}.extractors:PhaseExtractor",
                channels=("P",),
            ),
        ),
    )


def recompute_plan_hash(plan_payload: Mapping[str, object]) -> str:
    """Return the canonical SHA-256 hash of a plan payload.

    Parameters
    ----------
    plan_payload : Mapping[str, object]
        The plan audit record (possibly with derived fields).

    Returns
    -------
    str
        The canonical plan hash over the non-derived fields.
    """
    canonical_payload = dict(plan_payload)
    for key in (
        "plan_hash",
        "manifest",
        "capability",
        "compatible",
        "compatibility_reasons",
    ):
        canonical_payload.pop(key, None)
    canonical = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def normalize_plan_hash(plan_payload: Mapping[str, object]) -> str:
    """Return the recomputed plan hash for a (possibly mutated) plan payload.

    Parameters
    ----------
    plan_payload : Mapping[str, object]
        The plan payload to re-hash.

    Returns
    -------
    str
        The recomputed canonical plan hash.
    """
    payload = dict(plan_payload)
    payload["plan_hash"] = recompute_plan_hash(plan_payload)
    return payload["plan_hash"]


def write_plan_payload(
    path: Path,
    manifest: PluginManifest,
    kind: str,
    name: str,
    *,
    execution_permitted: bool = True,
    require_target_hash_approval: bool = False,
    approved_target_hashes: tuple[str, ...] = (),
) -> dict[str, object]:
    """Build, write, and return a plugin execution-plan JSON payload.

    Parameters
    ----------
    path : Path
        Destination file for the JSON payload.
    manifest : PluginManifest
        Plugin manifest.
    kind, name : str
        Capability selector.
    execution_permitted : bool
        When ``False``, mark the plan as not execution-permitted and re-hash it.
    require_target_hash_approval : bool
        Whether the runtime policy requires target-hash approval.
    approved_target_hashes : tuple[str, ...]
        Approved target hashes recorded in the runtime policy.

    Returns
    -------
    dict[str, object]
        The written plan payload.
    """
    policy = PluginRuntimeExecutionPolicy(
        loading_permitted=True,
        execution_permitted=True,
        require_target_hash_approval=require_target_hash_approval,
        approved_target_hashes=approved_target_hashes,
    )
    plan = build_plugin_execution_plan(manifest, kind, name, policy=policy)
    payload: dict[str, object] = {
        **plan.audit_record,
        "manifest": manifest.to_audit_record(),
        "capability": {
            "kind": plan.capability.kind,
            "name": plan.capability.name,
            "target": plan.capability.target,
            "version": plan.capability.version,
            "channels": list(plan.capability.channels),
            "knobs": list(plan.capability.knobs),
        },
    }
    if not execution_permitted:
        payload["execution_permitted"] = False
        payload["plan_hash"] = normalize_plan_hash(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def write_approval_payload(
    path: Path,
    manifest: PluginManifest,
    kind: str,
    name: str,
    *,
    operator_identity: str = "operator_42",
    approval_reference: str = "RFC-2026-05-20-01",
    approval_reason: str = "Production change window",
    approved: bool = True,
) -> dict[str, object]:
    """Build, write, and return a plugin execution-approval JSON payload.

    Parameters
    ----------
    path : Path
        Destination file for the JSON payload.
    manifest : PluginManifest
        Plugin manifest.
    kind, name : str
        Capability selector.
    operator_identity, approval_reference, approval_reason : str
        Operator approval metadata.
    approved : bool
        Approval decision recorded in the payload.

    Returns
    -------
    dict[str, object]
        The written approval payload.
    """
    policy = PluginRuntimeExecutionPolicy(
        loading_permitted=True,
        execution_permitted=True,
    )
    plan = build_plugin_execution_plan(manifest, kind, name, policy=policy)
    approval = build_plugin_execution_approval(
        plan,
        operator_identity=operator_identity,
        approval_reference=approval_reference,
        approval_reason=approval_reason,
    )
    payload = dict(approval.audit_record)
    payload["approved"] = approved
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
