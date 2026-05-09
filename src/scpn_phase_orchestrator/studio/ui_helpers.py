# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SPO Studio pure UI helpers

"""Pure helper layer for the SPO Studio Streamlit surface."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path

import numpy as np

from scpn_phase_orchestrator.binding import validate_binding_spec
from scpn_phase_orchestrator.binding.digital_twin import (
    DigitalTwinBindingContract,
    DigitalTwinSyncGrpcAdapter,
    DigitalTwinSyncHardwareAdapter,
    DigitalTwinSyncKafkaAdapter,
    DigitalTwinSyncRestAdapter,
    build_digital_twin_adapter_manifest,
    build_digital_twin_binding_contract,
    build_digital_twin_sync_envelope,
)
from scpn_phase_orchestrator.binding.loader import load_binding_spec
from scpn_phase_orchestrator.binding.types import BindingSpec
from scpn_phase_orchestrator.coupling.knm import CouplingState
from scpn_phase_orchestrator.server import SimulationState
from scpn_phase_orchestrator.studio.workflow import (
    BindingProposal,
    ExportManifest,
    ImportedSourceSummary,
    JsonValue,
    RuntimeSnapshot,
    StudioProjectState,
)

__all__ = [
    "StudioKnobState",
    "StudioReplayResult",
    "apply_canvas_binding_rewrite_candidate",
    "apply_knob_update",
    "binding_spec_project_state",
    "build_beginner_guidance",
    "build_canvas_edit_artifact",
    "build_canvas_binding_rewrite_candidate",
    "build_canvas_graph",
    "build_canvas_interaction_state",
    "build_canvas_layout_manifest",
    "build_canvas_topology_patch",
    "build_command_table",
    "build_export_manifests",
    "build_deployment_package",
    "build_deployment_readiness",
    "build_error_report",
    "build_layer_table",
    "build_hardware_target_package",
    "build_live_connector_plan",
    "build_live_connector_run_record",
    "build_owned_live_connector_runtime_record",
    "build_oscillator_edit_artifact",
    "build_oscillator_table",
    "build_operator_checklist",
    "build_regime_chart_payload",
    "build_package_materialisation_plan",
    "build_runtime_snapshot",
    "build_series_chart_payload",
    "build_service_process_manifest",
    "build_verified_hardware_target_package",
    "disabled_export_reasons",
    "discover_domainpacks",
    "run_binding_spec_replay",
]


@dataclass(frozen=True, slots=True)
class StudioKnobState:
    """Review-only knob state used by Studio replay controls."""

    K: float = 1.0
    alpha: float = 0.0
    zeta: float = 0.0
    Psi: float = 0.0

    def __post_init__(self) -> None:
        _finite_range(self.K, "K", low=0.1, high=10.0)
        _finite_range(self.alpha, "alpha", low=0.0, high=5.0)
        _finite_range(self.zeta, "zeta", low=0.0, high=5.0)
        _finite_range(self.Psi, "Psi", low=0.0, high=10.0)

    def to_audit_record(self) -> dict[str, float]:
        """Return a JSON-safe knob record."""
        return {
            "K": float(self.K),
            "alpha": float(self.alpha),
            "zeta": float(self.zeta),
            "Psi": float(self.Psi),
        }


@dataclass(frozen=True, slots=True)
class StudioReplayResult:
    """Replay output rendered by SPO Studio."""

    project_state: StudioProjectState
    r_history: tuple[float, ...]
    regime_history: tuple[str, ...]
    layer_table: tuple[dict[str, object], ...]
    oscillator_table: tuple[dict[str, object], ...]
    canvas_graph: Mapping[str, object]
    connector_plan: Mapping[str, object]
    export_manifests: tuple[ExportManifest, ...]

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe replay audit record."""
        return {
            "project": self.project_state.to_audit_record(),
            "r_history": list(self.r_history),
            "regime_history": list(self.regime_history),
            "layer_table": list(self.layer_table),
            "oscillator_table": list(self.oscillator_table),
            "canvas_graph": dict(self.canvas_graph),
            "connector_plan": dict(self.connector_plan),
            "exports": [
                manifest.to_audit_record() for manifest in self.export_manifests
            ],
        }


def discover_domainpacks(domainpack_dir: Path) -> tuple[str, ...]:
    """Return domainpack names containing a binding spec."""
    if not domainpack_dir.exists():
        return ()
    return tuple(
        sorted(
            path.name
            for path in domainpack_dir.iterdir()
            if path.is_dir() and (path / "binding_spec.yaml").exists()
        )
    )


def apply_knob_update(
    knobs: StudioKnobState,
    *,
    K: float | None = None,
    alpha: float | None = None,
    zeta: float | None = None,
    Psi: float | None = None,
) -> StudioKnobState:
    """Return validated knobs after a UI edit."""
    return StudioKnobState(
        K=knobs.K if K is None else K,
        alpha=knobs.alpha if alpha is None else alpha,
        zeta=knobs.zeta if zeta is None else zeta,
        Psi=knobs.Psi if Psi is None else Psi,
    )


def build_series_chart_payload(
    label: str,
    values: Sequence[float],
) -> list[dict[str, float | int]]:
    """Return dense chart rows for a scalar time-series."""
    _require_non_empty_text(label, "label")
    return [
        {"step": index, label: _finite_number(value, label)}
        for index, value in enumerate(values, 1)
    ]


def build_regime_chart_payload(regimes: Sequence[str]) -> list[dict[str, object]]:
    """Return deterministic chart rows for regime timelines."""
    regime_levels = {
        "critical": 0.0,
        "degraded": 1.0,
        "recovery": 1.5,
        "nominal": 2.0,
    }
    rows: list[dict[str, object]] = []
    for index, regime in enumerate(regimes, 1):
        regime_text = _require_non_empty_text(regime, "regime")
        rows.append(
            {
                "step": index,
                "regime": regime_text,
                "regime_level": regime_levels.get(regime_text, 0.0),
            }
        )
    return rows


def build_layer_table(spec: BindingSpec) -> tuple[dict[str, object], ...]:
    """Return editable layer rows for the Studio oscillator canvas."""
    return tuple(
        {
            "index": int(layer.index),
            "name": layer.name,
            "oscillator_count": len(layer.oscillator_ids),
            "family": layer.family or "",
            "omega_count": len(layer.omegas or ()),
        }
        for layer in sorted(spec.layers, key=lambda item: item.index)
    )


def build_oscillator_table(spec: BindingSpec) -> tuple[dict[str, object], ...]:
    """Return oscillator rows suitable for Streamlit data editing."""
    family_channels = {
        family_name: family.channel
        for family_name, family in spec.oscillator_families.items()
    }
    rows: list[dict[str, object]] = []
    for layer in sorted(spec.layers, key=lambda item: item.index):
        channel = family_channels.get(layer.family or "", "")
        for oscillator_id in layer.oscillator_ids:
            rows.append(
                {
                    "layer": layer.name,
                    "layer_index": int(layer.index),
                    "oscillator_id": oscillator_id,
                    "family": layer.family or "",
                    "channel": channel,
                }
            )
    return tuple(rows)


def build_canvas_graph(spec: BindingSpec) -> dict[str, object]:
    """Return a deterministic layer/coupling graph for Studio canvas review."""
    family_channels = {
        family_name: family.channel
        for family_name, family in spec.oscillator_families.items()
    }
    channel_order = {
        channel: index for index, channel in enumerate(sorted(spec.used_channels()))
    }
    channels = tuple(sorted(channel_order))
    nodes: list[dict[str, object]] = []
    for layer in sorted(spec.layers, key=lambda item: item.index):
        family = layer.family or ""
        channel = family_channels.get(family, "")
        nodes.append(
            {
                "id": f"layer_{layer.index}",
                "label": layer.name,
                "kind": "layer",
                "layer_index": int(layer.index),
                "family": family,
                "channel": channel,
                "oscillator_count": len(layer.oscillator_ids),
                "x": float(layer.index) * 220.0,
                "y": float(channel_order.get(channel, 0)) * 140.0,
            }
        )
    for index, channel in enumerate(channels):
        nodes.append(
            {
                "id": _canvas_channel_id(channel),
                "label": channel,
                "kind": "channel",
                "channel": channel,
                "layer_index": -1,
                "family": "",
                "oscillator_count": 0,
                "x": float(index) * 220.0,
                "y": 420.0,
            }
        )

    edges = [
        {
            "id": f"cross_channel_{index}",
            "source": _canvas_channel_id(coupling.source),
            "target": _canvas_channel_id(coupling.target),
            "kind": "cross_channel_coupling",
            "source_channel": coupling.source,
            "target_channel": coupling.target,
            "strength": float(coupling.strength),
            "mode": coupling.mode,
            "template": coupling.template or "",
        }
        for index, coupling in enumerate(spec.cross_channel_couplings, 1)
        if coupling.source in channel_order and coupling.target in channel_order
    ]
    return {
        "canvas_kind": "layer_coupling_graph",
        "node_count": len(nodes),
        "layer_count": len(spec.layers),
        "channel_count": len(channels),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
    }


def build_canvas_edit_artifact(
    before_graph: Mapping[str, object],
    after_graph: Mapping[str, object],
) -> ExportManifest:
    """Build a review artefact from edited Studio canvas graph rows."""
    before_nodes, before_edges = _normalise_canvas_graph(before_graph, "before_graph")
    after_nodes, after_edges = _normalise_canvas_graph(after_graph, "after_graph")
    payload = json.dumps(
        {
            "artifact": "canvas_edit_review",
            "changed": (before_nodes, before_edges) != (after_nodes, after_edges),
            "node_count_before": len(before_nodes),
            "node_count_after": len(after_nodes),
            "edge_count_before": len(before_edges),
            "edge_count_after": len(after_edges),
            "nodes_before": before_nodes,
            "nodes_after": after_nodes,
            "edges_before": before_edges,
            "edges_after": after_edges,
        },
        sort_keys=True,
        indent=2,
    )
    return ExportManifest.review_artifact(
        target_kind="canvas_edit_review",
        file_name="canvas_edit_review.json",
        payload=payload,
        command="review canvas_edit_review.json before updating binding_spec.yaml",
    )


def build_canvas_layout_manifest(
    *,
    project_name: str,
    graph: Mapping[str, object],
) -> ExportManifest:
    """Build a deterministic canvas layout manifest from node positions."""
    nodes, edges = _normalise_canvas_graph(graph, "canvas_layout")
    positions = []
    for node in sorted(nodes, key=lambda item: str(item.get("id", ""))):
        try:
            node_id = _require_non_empty_text(node.get("id"), "canvas layout id")
            kind = _require_non_empty_text(node.get("kind"), "canvas layout kind")
            label = _require_non_empty_text(node.get("label"), "canvas layout label")
            x = _finite_number(node.get("x"), "canvas layout x")
            y = _finite_number(node.get("y"), "canvas layout y")
        except ValueError as exc:
            raise ValueError(f"canvas layout node is invalid: {exc}") from exc
        positions.append({"id": node_id, "kind": kind, "label": label, "x": x, "y": y})
    payload = json.dumps(
        {
            "manifest_kind": "canvas_layout_manifest",
            "project_name": _require_non_empty_text(project_name, "project_name"),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "positions": positions,
        },
        sort_keys=True,
        indent=2,
    )
    return ExportManifest.review_artifact(
        target_kind="canvas_layout_manifest",
        file_name="canvas_layout_manifest.json",
        payload=payload,
        command="review canvas_layout_manifest.json before restoring Studio layout",
    )


def build_canvas_topology_patch(
    *,
    project_name: str,
    before_graph: Mapping[str, object],
    after_graph: Mapping[str, object],
) -> ExportManifest:
    """Build a review patch for persistent Studio topology edits."""
    before_nodes, before_edges = _normalise_canvas_graph(before_graph, "before_graph")
    after_nodes, after_edges = _normalise_canvas_graph(after_graph, "after_graph")
    _validate_canvas_edge_endpoints(after_nodes, after_edges)

    node_changes = _canvas_item_changes(
        before_nodes,
        after_nodes,
        fields=("id", "kind", "label", "x", "y"),
    )
    edge_changes = _canvas_item_changes(
        before_edges,
        after_edges,
        fields=("id", "kind", "source", "target"),
    )
    changed = any(node_changes[key] or edge_changes[key] for key in node_changes)
    payload = json.dumps(
        {
            "patch_kind": "canvas_topology_patch",
            "project_name": _require_non_empty_text(project_name, "project_name"),
            "status": "review_required",
            "changed": changed,
            "node_count_before": len(before_nodes),
            "node_count_after": len(after_nodes),
            "edge_count_before": len(before_edges),
            "edge_count_after": len(after_edges),
            "node_changes": node_changes,
            "edge_changes": edge_changes,
            "safety": {
                "binding_spec_rewritten": False,
                "actuation_permitted": False,
            },
        },
        sort_keys=True,
        indent=2,
    )
    return ExportManifest.review_artifact(
        target_kind="canvas_topology_patch",
        file_name="canvas_topology_patch.json",
        payload=payload,
        command="review canvas_topology_patch.json before rewriting binding_spec.yaml",
    )


def build_canvas_interaction_state(
    *,
    canvas_artifact: ExportManifest,
    canvas_layout: ExportManifest,
    canvas_patch: ExportManifest,
    canvas_rewrite: Mapping[str, object],
    operator_signoff: bool,
) -> dict[str, object]:
    """Summarise Canvas browser controls for deterministic operator feedback."""
    record = json.loads(canvas_artifact.payload)
    changed = bool(record.get("changed"))
    rewrite_status = _require_non_empty_text(
        canvas_rewrite.get("status"),
        "rewrite_status",
    )
    validation_errors = _string_list(
        canvas_rewrite.get("validation_errors", ()),
        "validation_errors",
    )
    disabled_reasons: list[str] = []
    if rewrite_status != "review_ready":
        disabled_reasons.append("binding rewrite candidate is blocked")
    disabled_reasons.extend(validation_errors)
    if not operator_signoff:
        disabled_reasons.append("operator sign-off required")
    apply_enabled = not disabled_reasons
    return {
        "state_kind": "studio_canvas_interaction_state",
        "changed": changed,
        "rewrite_status": rewrite_status,
        "apply_enabled": apply_enabled,
        "disabled_reasons": disabled_reasons,
        "next_action": _canvas_next_action(
            changed=changed,
            rewrite_status=rewrite_status,
            operator_signoff=operator_signoff,
            apply_enabled=apply_enabled,
        ),
        "status_message": (
            "Canvas edits need review before apply."
            if changed
            else "Canvas graph matches the current binding."
        ),
        "download_manifest": [
            canvas_artifact.file_name,
            canvas_layout.file_name,
            canvas_patch.file_name,
            "binding_rewrite_candidate.yaml",
        ],
        "candidate_yaml_sha256": canvas_rewrite.get("candidate_yaml_sha256", ""),
    }


def build_canvas_binding_rewrite_candidate(
    result: StudioReplayResult,
    *,
    after_graph: Mapping[str, object],
) -> dict[str, object]:
    """Build validated binding YAML candidate from reviewed canvas edits."""
    if not isinstance(result, StudioReplayResult):
        raise ValueError("replay result must be a StudioReplayResult")
    _, after_edges = _normalise_canvas_graph(after_graph, "after_graph")
    before_yaml = result.project_state.binding.yaml_text
    before_digest = sha256(before_yaml.encode("utf-8")).hexdigest()
    unsupported = [
        _require_non_empty_text(edge.get("id"), "canvas edge id")
        for edge in after_edges
        if edge.get("kind") != "cross_channel_coupling"
    ]
    if unsupported:
        return _blocked_binding_rewrite_candidate(
            result,
            before_digest,
            ["only cross_channel_coupling edges can rewrite binding YAML"],
        )

    try:
        candidate_yaml = _rewrite_binding_cross_channel_couplings(
            before_yaml,
            after_edges,
        )
    except ValueError as exc:
        return _blocked_binding_rewrite_candidate(result, before_digest, [str(exc)])

    validation_errors = _validate_candidate_binding_yaml(candidate_yaml)
    return {
        "candidate_kind": "canvas_binding_rewrite_candidate",
        "project_name": result.project_state.project_name,
        "status": "blocked" if validation_errors else "review_ready",
        "binding_spec_rewritten": False,
        "actuation_permitted": False,
        "network_opened": False,
        "before_yaml_sha256": before_digest,
        "candidate_yaml_sha256": sha256(candidate_yaml.encode("utf-8")).hexdigest(),
        "coupling_count_before": int(result.canvas_graph["edge_count"]),
        "coupling_count_after": len(after_edges),
        "validation_errors": validation_errors,
        "candidate_yaml": candidate_yaml,
    }


def apply_canvas_binding_rewrite_candidate(
    candidate: Mapping[str, object],
    *,
    binding_spec_path: str | Path,
    operator_signoff: bool,
    create_backup: bool = True,
) -> dict[str, object]:
    """Apply a reviewed canvas binding candidate with hash and validation gates."""
    path = Path(binding_spec_path)
    candidate_yaml = _require_non_empty_payload(
        candidate.get("candidate_yaml"),
        "candidate_yaml",
    )
    before_digest = _require_sha256_digest(
        candidate.get("before_yaml_sha256"),
        "before_yaml_sha256",
    )
    candidate_digest = _require_sha256_digest(
        candidate.get("candidate_yaml_sha256"),
        "candidate_yaml_sha256",
    )
    blocked_reasons = _binding_apply_blocked_reasons(
        candidate,
        path,
        candidate_yaml,
        before_digest,
        candidate_digest,
        operator_signoff=operator_signoff,
    )
    if blocked_reasons:
        return _binding_apply_record(
            candidate,
            path,
            status="blocked",
            before_digest=before_digest,
            after_digest="",
            backup_path="",
            blocked_reasons=blocked_reasons,
        )

    current_yaml = path.read_text(encoding="utf-8")
    backup_path = ""
    if create_backup:
        backup = _next_binding_backup_path(path, before_digest)
        backup.write_text(current_yaml, encoding="utf-8")
        backup_path = str(backup)
    _atomic_write_text(path, candidate_yaml)
    after_digest = sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    return _binding_apply_record(
        candidate,
        path,
        status="applied",
        before_digest=before_digest,
        after_digest=after_digest,
        backup_path=backup_path,
        blocked_reasons=[],
    )


def build_live_connector_plan(spec: BindingSpec) -> dict[str, object]:
    """Return non-opening connector ownership guidance for Studio."""
    contract = build_digital_twin_binding_contract(spec)
    connector_specs = (
        ("memory", True, False, "review offline memory connector"),
        ("jsonl", True, False, "review JSONL replay connector"),
        ("rest", False, True, "assign connector owner and auth policy"),
        ("grpc", False, True, "assign connector owner and auth policy"),
        ("kafka", False, True, "assign connector owner and auth policy"),
        ("hardware", False, True, "assign connector owner and auth policy"),
    )
    connectors: list[dict[str, object]] = []
    for transport, supports_replay, requires_auth, action in connector_specs:
        compatibility = build_digital_twin_adapter_manifest(
            contract,
            name=f"studio-{transport}",
            transport=transport,
            sync_capabilities=[
                capability.name for capability in contract.sync_capabilities
            ],
            supports_replay=supports_replay,
            requires_auth=requires_auth,
            notes="SPO Studio connector review",
        )
        manifest = compatibility.manifest
        owner_required = transport in {"rest", "grpc", "kafka", "hardware"}
        connectors.append(
            {
                "name": manifest.name,
                "transport": manifest.transport,
                "status": "owner_required" if owner_required else "review_ready",
                "compatible": compatibility.compatible,
                "reasons": list(compatibility.reasons),
                "sync_capabilities": list(manifest.sync_capabilities),
                "supports_replay": manifest.supports_replay,
                "requires_auth": manifest.requires_auth,
                "operator_action": action,
                "network_opened": False,
                "hardware_write_permitted": False,
            }
        )
    return {
        "plan_kind": "studio_live_connector_plan",
        "project_name": spec.name,
        "contract_hash": contract.contract_hash,
        "network_opened": False,
        "actuation_permitted": False,
        "connectors": connectors,
    }


def build_live_connector_run_record(
    connector_plan: Mapping[str, object],
    *,
    transport: str,
    payload: Mapping[str, object],
    dry_run: bool = True,
) -> dict[str, object]:
    """Return a gated live-connector execution record without opening transport."""
    connector = _connector_by_transport(
        connector_plan,
        _require_non_empty_text(transport, "transport"),
    )
    payload_json = _stable_json_payload(payload, "payload")
    connector_status = _require_non_empty_text(connector.get("status"), "status")
    blocked_reasons: list[str] = []
    if connector_status != "review_ready":
        blocked_reasons.append("connector owner and auth policy required")
    if not dry_run:
        blocked_reasons.append("Studio live execution uses dry-run records only")

    status = "blocked" if blocked_reasons else "accepted"
    return {
        "record_kind": "studio_live_connector_run",
        "project_name": _require_non_empty_text(
            connector_plan.get("project_name"),
            "project_name",
        ),
        "transport": connector["transport"],
        "connector_name": connector["name"],
        "status": status,
        "dry_run": bool(dry_run),
        "payload_sha256": sha256(payload_json.encode("utf-8")).hexdigest(),
        "blocked_reasons": blocked_reasons,
        "operator_action": (
            "review dry-run connector payload"
            if status == "accepted"
            else _require_non_empty_text(
                connector.get("operator_action"),
                "operator_action",
            )
        ),
        "network_opened": False,
        "actuation_permitted": False,
        "hardware_write_permitted": False,
    }


def build_owned_live_connector_runtime_record(
    result: StudioReplayResult,
    *,
    transport: str,
    owner: str,
    auth_policy: Mapping[str, object],
    payload: Mapping[str, object],
    sequence: int = 1,
    capability: str = "audit_replay",
    direction: str = "twin_to_spo",
) -> dict[str, object]:
    """Validate an owned live connector boundary without opening transport."""
    if not isinstance(result, StudioReplayResult):
        raise ValueError("replay result must be a StudioReplayResult")
    checked_transport = _require_non_empty_text(transport, "transport")
    checked_payload = _normalise_json_mapping(payload, "payload")
    payload_json = _stable_json_payload(checked_payload, "payload")
    blocked_reasons = _owned_runtime_blocked_reasons(
        result.connector_plan,
        checked_transport,
        owner,
        auth_policy,
    )
    base = _owned_runtime_base_record(
        result,
        transport=checked_transport,
        owner=owner,
        payload_sha256=sha256(payload_json.encode("utf-8")).hexdigest(),
        sequence=sequence,
        capability=capability,
        direction=direction,
    )
    if blocked_reasons:
        return {
            **base,
            "status": "blocked",
            "blocked_reasons": blocked_reasons,
            "response": {},
            "adapter": {},
            "queued_count": 0,
        }

    spec_path = _result_binding_spec_path(result)
    spec = load_binding_spec(spec_path)
    contract = build_digital_twin_binding_contract(spec)
    envelope = build_digital_twin_sync_envelope(
        contract,
        capability=_require_non_empty_text(capability, "capability"),
        direction=_require_non_empty_text(direction, "direction"),
        sequence=_non_negative_int(sequence, "sequence"),
        payload=checked_payload,
    )
    response, adapter_record = _run_owned_live_adapter(
        contract,
        transport=checked_transport,
        envelope_record=envelope.to_audit_record(),
    )
    return {
        **base,
        "status": "accepted" if response.get("accepted") is True else "blocked",
        "blocked_reasons": (
            [] if response.get("accepted") is True else [str(response["reason"])]
        ),
        "response": response,
        "adapter": adapter_record,
        "queued_count": int(adapter_record.get("queued_count", 0)),
    }


def build_hardware_target_package(result: StudioReplayResult) -> dict[str, object]:
    """Return a review-only hardware target package for Studio."""
    if not isinstance(result, StudioReplayResult):
        raise ValueError("replay result must be a StudioReplayResult")
    connector_plan = result.connector_plan
    hardware_connector = _connector_by_transport(connector_plan, "hardware")
    return {
        "package_kind": "studio_hardware_target_package",
        "project_name": result.project_state.project_name,
        "overall_status": "evidence_required",
        "contract_hash": _require_non_empty_text(
            connector_plan.get("contract_hash"),
            "contract_hash",
        ),
        "hardware_write_permitted": False,
        "network_opened": False,
        "targets": ["fpga_verilog", "neuromorphic_schedule"],
        "required_evidence": [
            "generated hardware artefact path",
            "simulator parity report",
            "target toolchain version",
            "operator sign-off",
        ],
        "commands": [
            "review connector_plan.json",
            "generate FPGA Verilog with KuramotoVerilogCompiler",
            "run simulator parity before hardware handoff",
        ],
        "connector": hardware_connector,
        "export_artifacts": [
            manifest.to_audit_record() for manifest in result.export_manifests
        ],
    }


def build_verified_hardware_target_package(
    result: StudioReplayResult,
    *,
    evidence: Mapping[str, object],
) -> dict[str, object]:
    """Return a verified hardware package only when evidence is complete."""
    if not isinstance(result, StudioReplayResult):
        raise ValueError("replay result must be a StudioReplayResult")
    if not isinstance(evidence, Mapping):
        raise ValueError("hardware evidence must be a mapping")

    base_package = build_hardware_target_package(result)
    normalised, invalid_evidence = _normalise_hardware_evidence(evidence)
    verified = not invalid_evidence
    return {
        "package_kind": "studio_verified_hardware_target_package",
        "project_name": result.project_state.project_name,
        "overall_status": "review_ready" if verified else "evidence_required",
        "evidence_status": "verified" if verified else "blocked",
        "contract_hash": base_package["contract_hash"],
        "hardware_write_permitted": False,
        "network_opened": False,
        "targets": list(base_package["targets"]),
        "required_evidence": list(base_package["required_evidence"]),
        "invalid_evidence": invalid_evidence,
        "evidence": normalised,
        "connector": base_package["connector"],
        "commands": (
            [
                "review verified_hardware_target_package.json",
                "compare generated artefact hash before handoff",
                "archive simulator parity report with package",
            ]
            if verified
            else []
        ),
        "safety_gates": [
            "local replay completed",
            "binding validation passed",
            "hardware evidence verified" if verified else "hardware evidence blocked",
            "hardware output remains operator-controlled",
        ],
        "export_artifacts": list(base_package["export_artifacts"]),
    }


def build_beginner_guidance(result: StudioReplayResult) -> dict[str, object]:
    """Return domain-term guidance for first-time Studio operators."""
    if not isinstance(result, StudioReplayResult):
        raise ValueError("replay result must be a StudioReplayResult")
    project = result.project_state
    runtime = project.runtime
    layers = [
        _require_non_empty_text(row.get("name"), "layer")
        for row in result.layer_table
        if isinstance(row, Mapping)
    ]
    channels = [
        _require_non_empty_text(node.get("channel"), "channel")
        for node in result.canvas_graph.get("nodes", ())
        if isinstance(node, Mapping) and node.get("kind") == "channel"
    ]
    validation_errors = list(project.binding.validation_errors)
    canvas_evidence = {
        "layers": int(result.canvas_graph["layer_count"]),
        "channels": int(result.canvas_graph["channel_count"]),
        "couplings": int(result.canvas_graph["edge_count"]),
    }
    return {
        "guide_kind": "beginner_mode",
        "project_name": project.project_name,
        "actuation_permitted": False,
        "runtime_summary": {
            "replay_status": runtime.replay_status,
            "regime": runtime.regime,
            "R": float(runtime.R),
            "domain_signal": (
                "R summarises how closely the reviewed domain signals move together."
            ),
        },
        "concept_cards": [
            {
                "title": "Signals",
                "plain_language": (
                    "Each layer groups domain measurements that Studio reviews as "
                    "oscillators."
                ),
                "evidence": {
                    "layers": layers,
                    "channels": sorted(channels),
                    "source_kind": project.source.source_kind,
                },
            },
            {
                "title": "Coupling",
                "plain_language": (
                    "K raises or lowers how much the reviewed signals influence "
                    "one another during replay."
                ),
                "evidence": {
                    "K": float(runtime.K),
                    "alpha": float(runtime.alpha),
                    "zeta": float(runtime.zeta),
                    "Psi": float(runtime.Psi),
                    "cross_channel_edges": int(result.canvas_graph["edge_count"]),
                },
            },
            {
                "title": "Objectives",
                "plain_language": (
                    "The objective is to keep reviewed good layers coherent while "
                    "validation errors block packaging."
                ),
                "evidence": {
                    "validation_errors": validation_errors,
                    "binding_ready": not validation_errors,
                },
            },
            {
                "title": "Supervisor",
                "plain_language": (
                    "The supervisor reads the replay regime and emits review "
                    "evidence only; live actuation stays disabled."
                ),
                "evidence": {
                    "regime": runtime.regime,
                    "hierarchy_watermarks": dict(runtime.hierarchy_watermarks),
                },
            },
        ],
        "next_actions": (
            ["review binding validation"]
            + (["fix validation errors"] if validation_errors else ["review exports"])
            + ["download project_state.json"]
        ),
        "walkthrough_steps": [
            {
                "step": 1,
                "title": "Load project",
                "status": "complete",
                "operator_action": "review source summary",
                "evidence": {"source_kind": project.source.source_kind},
            },
            {
                "step": 2,
                "title": "Run replay",
                "status": (
                    "complete" if runtime.replay_status == "completed" else "blocked"
                ),
                "operator_action": "run local replay",
                "evidence": {"replay_status": runtime.replay_status},
            },
            {
                "step": 3,
                "title": "Review binding",
                "status": "blocked" if validation_errors else "complete",
                "operator_action": (
                    "fix validation errors"
                    if validation_errors
                    else "review binding and continue"
                ),
                "evidence": {"validation_errors": validation_errors},
            },
            {
                "step": 4,
                "title": "Inspect canvas",
                "status": "complete",
                "operator_action": "inspect layer, channel, and coupling graph",
                "evidence": canvas_evidence,
            },
            {
                "step": 5,
                "title": "Prepare exports",
                "status": "blocked" if validation_errors else "ready",
                "operator_action": (
                    "fix validation errors"
                    if validation_errors
                    else "download review artefacts"
                ),
                "evidence": {
                    "export_count": len(result.export_manifests),
                    "connector_count": len(result.connector_plan.get("connectors", ())),
                },
            },
        ],
    }


def build_runtime_snapshot(
    *,
    final_state: Mapping[str, object],
    knobs: StudioKnobState,
    hierarchy_watermarks: Mapping[str, int] | None = None,
    replay_status: str = "not_started",
) -> RuntimeSnapshot:
    """Build a workflow runtime snapshot from a simulation state dict."""
    layers = _layer_metrics(final_state.get("layers", ()))
    return RuntimeSnapshot(
        R=_finite_number(final_state.get("R_global", 0.0), "R_global"),
        Psi=knobs.Psi,
        K=knobs.K,
        alpha=knobs.alpha,
        zeta=knobs.zeta,
        regime=_require_non_empty_text(final_state.get("regime", "unknown"), "regime"),
        layer_metrics=layers,
        hierarchy_watermarks=dict(hierarchy_watermarks or {}),
        replay_status=replay_status,
    )


def binding_spec_project_state(
    *,
    project_name: str,
    spec_path: Path,
    knobs: StudioKnobState,
    runtime: RuntimeSnapshot,
) -> StudioProjectState:
    """Create a Studio project state from an existing binding spec file."""
    yaml_text = spec_path.read_text(encoding="utf-8")
    spec = load_binding_spec(spec_path)
    validation_errors = tuple(validate_binding_spec(spec))
    source = ImportedSourceSummary.from_payload(
        source_kind="binding_spec_yaml",
        payload=yaml_text.encode("utf-8"),
        channel_count=max(1, len(spec.used_channels())),
        sample_count=sum(len(layer.oscillator_ids) for layer in spec.layers),
    )
    provenance: dict[str, JsonValue] = {
        "source_path": str(spec_path),
        "knobs": dict(knobs.to_audit_record()),
        "validator": "validate_binding_spec",
    }
    binding = BindingProposal(
        yaml_text=yaml_text,
        validation_errors=validation_errors,
        inferred_channels=tuple(sorted(spec.used_channels())),
        confidence_factors={
            "validator_acceptance": 1.0 if not validation_errors else 0.0,
            "layer_coverage": 1.0 if spec.layers else 0.0,
        },
        provenance=provenance,
    )
    exports = build_export_manifests(
        project_name=project_name,
        binding_yaml=yaml_text,
        audit_payload={
            "project_name": project_name,
            "runtime": runtime.to_audit_record(),
        },
        validation_errors=validation_errors,
    )
    return StudioProjectState(
        project_name=project_name,
        source=source,
        binding=binding,
        runtime=runtime,
        exports=exports,
        metadata={
            "domainpack": project_name,
            "safety": "local_replay_only",
        },
    )


def build_export_manifests(
    *,
    project_name: str,
    binding_yaml: str,
    audit_payload: Mapping[str, object],
    validation_errors: Sequence[str],
) -> tuple[ExportManifest, ...]:
    """Build review-only export manifests for Studio."""
    deploy_warnings = disabled_export_reasons(validation_errors)
    audit_export_payload = {
        **dict(audit_payload),
        "enabled": not deploy_warnings,
        "disabled_reasons": list(deploy_warnings),
    }
    audit_json = json.dumps(audit_export_payload, sort_keys=True, indent=2)
    docker_payload = json.dumps(
        {
            "project_name": project_name,
            "image": "scpn-phase-orchestrator:local",
            "command": "spo run binding_spec.yaml --audit audit.jsonl",
            "enabled": not deploy_warnings,
            "disabled_reasons": list(deploy_warnings),
        },
        sort_keys=True,
        indent=2,
    )
    wasm_payload = json.dumps(
        {
            "project_name": project_name,
            "target": "wasm_review_manifest",
            "enabled": not deploy_warnings,
            "disabled_reasons": list(deploy_warnings),
        },
        sort_keys=True,
        indent=2,
    )
    return (
        ExportManifest.review_artifact(
            target_kind="binding_spec",
            file_name="binding_spec.yaml",
            payload=binding_yaml,
            command="spo run binding_spec.yaml --audit audit.jsonl",
            warnings=deploy_warnings,
        ),
        ExportManifest.review_artifact(
            target_kind="audit_summary",
            file_name="spo_studio_audit.json",
            payload=audit_json,
            command="spo audit summary spo_studio_audit.json",
            warnings=deploy_warnings,
        ),
        ExportManifest.review_artifact(
            target_kind="docker_manifest",
            file_name="docker_manifest.json",
            payload=docker_payload,
            command="docker compose config",
            warnings=deploy_warnings,
        ),
        ExportManifest.review_artifact(
            target_kind="wasm_manifest",
            file_name="wasm_manifest.json",
            payload=wasm_payload,
            command="spo export wasm --manifest wasm_manifest.json",
            warnings=deploy_warnings,
        ),
    )


def build_deployment_readiness(
    project_state: StudioProjectState,
) -> dict[str, object]:
    """Return target-specific deployment readiness guidance for Studio."""
    blocked_reasons = _deployment_blocked_reasons(project_state.exports)
    if blocked_reasons:
        return {
            "project_name": project_state.project_name,
            "overall_status": "blocked",
            "operator_next_step": "fix binding validation errors",
            "targets": [
                _blocked_target("docker", blocked_reasons),
                _blocked_target("wasm", blocked_reasons),
                _blocked_target("hardware", blocked_reasons),
            ],
        }

    return {
        "project_name": project_state.project_name,
        "overall_status": "review_ready",
        "operator_next_step": "review target-specific packaging",
        "targets": [
            {
                "target": "docker",
                "status": "ready",
                "required_artifacts": [
                    "binding_spec.yaml",
                    "spo_studio_audit.json",
                    "docker_manifest.json",
                ],
                "commands": [
                    "docker compose config",
                    "docker build -t scpn-phase-orchestrator:local .",
                    "docker run --rm -v $PWD:/workspace "
                    "scpn-phase-orchestrator:local "
                    "spo run binding_spec.yaml --audit audit.jsonl",
                ],
                "operator_action": "run docker manifest review before packaging",
            },
            {
                "target": "wasm",
                "status": "ready",
                "required_artifacts": [
                    "binding_spec.yaml",
                    "spo_studio_audit.json",
                    "wasm_manifest.json",
                ],
                "commands": [
                    "cd spo-kernel && wasm-pack build crates/spo-wasm "
                    "--target web --out-dir ../../../docs/wasm-pkg",
                ],
                "operator_action": "review browser-safe replay constraints",
            },
            {
                "target": "hardware",
                "status": "postponed",
                "required_artifacts": [
                    "binding_spec.yaml",
                    "spo_studio_audit.json",
                    "verified_hardware_target_evidence",
                ],
                "commands": [],
                "operator_action": "attach verified hardware-target evidence",
            },
        ],
    }


def build_deployment_package(
    project_state: StudioProjectState,
) -> dict[str, object]:
    """Return a deterministic deployment package manifest for Studio."""
    readiness = build_deployment_readiness(project_state)
    targets = _readiness_targets(readiness)
    blocked_reasons = _deployment_blocked_reasons(project_state.exports)
    return {
        "package_kind": "studio_deployment_package",
        "project_name": project_state.project_name,
        "overall_status": readiness["overall_status"],
        "ready_targets": [
            target["target"] for target in targets if target["status"] == "ready"
        ],
        "postponed_targets": [
            target["target"] for target in targets if target["status"] == "postponed"
        ],
        "blocked_targets": [
            target["target"] for target in targets if target["status"] == "blocked"
        ],
        "blocked_reasons": list(blocked_reasons),
        "required_artifacts": _unique_artifacts(targets),
        "export_artifacts": [
            {
                "target_kind": manifest.target_kind,
                "file_name": manifest.file_name,
                "payload_sha256": manifest.payload_sha256,
                "safety_posture": manifest.safety_posture,
                "warnings": list(manifest.warnings),
            }
            for manifest in project_state.exports
        ],
        "commands": list(build_command_table(project_state)),
        "safety_gates": [
            "local replay completed",
            (
                "binding validation blocked"
                if blocked_reasons
                else "binding validation passed"
            ),
            "live actuation disabled",
            "hardware output requires verified evidence",
        ],
        "readiness": readiness,
    }


def build_service_process_manifest(
    project_state: StudioProjectState,
) -> dict[str, object]:
    """Return localhost-only service process packaging for Studio deployment."""
    blocked_reasons = _deployment_blocked_reasons(project_state.exports)
    if blocked_reasons:
        return {
            "manifest_kind": "studio_service_process_manifest",
            "project_name": project_state.project_name,
            "overall_status": "blocked",
            "execution_mode": "operator_invoked",
            "network_opened": False,
            "actuation_permitted": False,
            "hardware_write_permitted": False,
            "host_bind": "127.0.0.1",
            "compose_file": "spo_studio_services.compose.yaml",
            "services": [],
            "blocked_reasons": list(blocked_reasons),
            "required_artifacts": [],
            "compose_yaml": "",
            "compose_yaml_sha256": "",
        }

    services = _studio_service_processes()
    compose_yaml = _render_service_compose_yaml(services)
    return {
        "manifest_kind": "studio_service_process_manifest",
        "project_name": project_state.project_name,
        "overall_status": "operator_ready",
        "execution_mode": "operator_invoked",
        "network_opened": False,
        "actuation_permitted": False,
        "hardware_write_permitted": False,
        "host_bind": "127.0.0.1",
        "compose_file": "spo_studio_services.compose.yaml",
        "services": services,
        "blocked_reasons": [],
        "required_artifacts": [
            "binding_spec.yaml",
            "spo_studio_audit.json",
            "docker_manifest.json",
            "owned_connector_runtime.json",
        ],
        "operator_commands": [
            "docker compose -f spo_studio_services.compose.yaml config",
            "docker compose -f spo_studio_services.compose.yaml up spo-studio-ui",
        ],
        "compose_yaml": compose_yaml,
        "compose_yaml_sha256": sha256(compose_yaml.encode("utf-8")).hexdigest(),
    }


def build_package_materialisation_plan(
    project_state: StudioProjectState,
) -> dict[str, object]:
    """Return ordered, operator-invoked package materialisation commands."""
    package = build_deployment_package(project_state)
    command_rows = build_command_table(project_state)
    commands = [
        {
            "step": index,
            "target": _require_non_empty_text(row.get("target"), "target"),
            "command": _require_non_empty_text(row.get("command"), "command"),
            "status": _require_non_empty_text(row.get("status"), "status"),
            "requires_operator": True,
            "writes_artifact": _materialisation_command_writes_artifact(
                row.get("command")
            ),
        }
        for index, row in enumerate(command_rows, 1)
    ]
    readiness = build_deployment_readiness(project_state)
    targets = _readiness_targets(readiness)
    return {
        "plan_kind": "studio_package_materialisation_plan",
        "project_name": project_state.project_name,
        "overall_status": package["overall_status"],
        "execution_mode": "operator_invoked",
        "network_opened": False,
        "hardware_write_permitted": False,
        "commands": commands,
        "blocked_targets": list(package["blocked_targets"]),
        "blocked_reasons": list(package["blocked_reasons"]),
        "postponed_targets": [
            {
                "target": target["target"],
                "reason": _require_non_empty_text(
                    target.get("operator_action"),
                    "operator_action",
                ),
            }
            for target in targets
            if target["status"] == "postponed"
        ],
        "required_artifacts": list(package["required_artifacts"]),
        "safety_gates": list(package["safety_gates"]),
    }


def build_operator_checklist(
    project_state: StudioProjectState,
) -> tuple[dict[str, object], ...]:
    """Return beginner-friendly ordered deployment steps for Studio."""
    readiness = build_deployment_readiness(project_state)
    validation_blocked = readiness["overall_status"] == "blocked"
    steps: list[dict[str, object]] = [
        {
            "step": 1,
            "title": "Run local replay",
            "status": (
                "complete"
                if project_state.runtime.replay_status == "completed"
                else "blocked"
            ),
            "detail": project_state.runtime.replay_status,
        },
        {
            "step": 2,
            "title": "Validate binding",
            "status": "blocked" if validation_blocked else "complete",
            "detail": (
                "; ".join(_deployment_blocked_reasons(project_state.exports))
                if validation_blocked
                else "binding validation passed"
            ),
        },
    ]
    for target in readiness["targets"]:
        if not isinstance(target, Mapping):
            raise ValueError("readiness targets must be mappings")
        target_name = _require_non_empty_text(target.get("target"), "target")
        status = _require_non_empty_text(target.get("status"), "status")
        operator_action = _require_non_empty_text(
            target.get("operator_action"),
            "operator_action",
        )
        blocked_detail = "; ".join(
            str(reason) for reason in target.get("blocked_reasons", ())
        )
        steps.append(
            {
                "step": len(steps) + 1,
                "title": f"Review {target_name} packaging",
                "target": target_name,
                "status": status,
                "detail": blocked_detail or operator_action,
            }
        )
    return tuple(steps)


def build_command_table(
    project_state: StudioProjectState,
) -> tuple[dict[str, object], ...]:
    """Return copyable deployment-review commands for ready targets."""
    readiness = build_deployment_readiness(project_state)
    rows: list[dict[str, object]] = []
    for target in readiness["targets"]:
        if not isinstance(target, Mapping):
            raise ValueError("readiness targets must be mappings")
        status = _require_non_empty_text(target.get("status"), "status")
        if status == "blocked":
            continue
        target_name = _require_non_empty_text(target.get("target"), "target")
        commands = target.get("commands", ())
        if isinstance(commands, str | bytes) or not isinstance(commands, Sequence):
            raise ValueError("target commands must be a sequence of strings")
        for index, command in enumerate(commands, 1):
            rows.append(
                {
                    "target": target_name,
                    "command_index": index,
                    "command": _require_non_empty_text(command, "command"),
                    "status": status,
                }
            )
    return tuple(rows)


def build_error_report(
    *,
    operation: str,
    error: Exception,
    project_name: str = "unknown",
) -> dict[str, object]:
    """Return a path-safe operator report for failed Studio actions."""
    return {
        "project_name": _require_non_empty_text(project_name, "project_name"),
        "operation": _require_non_empty_text(operation, "operation"),
        "status": "blocked",
        "error_type": type(error).__name__,
        "operator_action": "review input artefacts and rerun",
    }


def build_oscillator_edit_artifact(
    before_rows: Sequence[Mapping[str, object]],
    after_rows: Sequence[Mapping[str, object]],
) -> ExportManifest:
    """Build a review artefact from edited oscillator table rows."""
    before = _normalise_table_rows(before_rows, "before_rows")
    after = _normalise_table_rows(after_rows, "after_rows")
    payload = json.dumps(
        {
            "artifact": "oscillator_edit_review",
            "changed": before != after,
            "row_count_before": len(before),
            "row_count_after": len(after),
            "rows_before": before,
            "rows_after": after,
        },
        sort_keys=True,
        indent=2,
    )
    return ExportManifest.review_artifact(
        target_kind="oscillator_edit_review",
        file_name="oscillator_edit_review.json",
        payload=payload,
        command="review oscillator_edit_review.json before updating binding_spec.yaml",
    )


def disabled_export_reasons(validation_errors: Sequence[str]) -> tuple[str, ...]:
    """Return reasons deploy-like exports must stay review-only."""
    errors = tuple(str(error) for error in validation_errors)
    if not errors:
        return ()
    return (
        "binding validation must pass before deploy manifests are enabled",
        *errors,
    )


def run_binding_spec_replay(
    spec_path: Path,
    *,
    steps: int,
    knobs: StudioKnobState,
) -> StudioReplayResult:
    """Run a local binding-spec replay and return Studio-ready payloads."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    spec = load_binding_spec(spec_path)
    sim = SimulationState(spec)
    _apply_replay_knobs(sim, knobs)

    r_history: list[float] = []
    regime_history: list[str] = []
    final_state: Mapping[str, object] = sim.snapshot()
    for _ in range(steps):
        final_state = sim.step()
        r_history.append(_finite_number(final_state["R_global"], "R_global"))
        regime_history.append(_require_non_empty_text(final_state["regime"], "regime"))

    runtime = build_runtime_snapshot(
        final_state=final_state,
        knobs=knobs,
        replay_status="completed",
    )
    project_state = binding_spec_project_state(
        project_name=spec.name,
        spec_path=spec_path,
        knobs=knobs,
        runtime=runtime,
    )
    return StudioReplayResult(
        project_state=project_state,
        r_history=tuple(r_history),
        regime_history=tuple(regime_history),
        layer_table=build_layer_table(spec),
        oscillator_table=build_oscillator_table(spec),
        canvas_graph=build_canvas_graph(spec),
        connector_plan=build_live_connector_plan(spec),
        export_manifests=project_state.exports,
    )


def _apply_replay_knobs(sim: SimulationState, knobs: StudioKnobState) -> None:
    scaled_knm = np.asarray(sim.coupling.knm, dtype=np.float64) * knobs.K
    alpha = np.asarray(sim.coupling.alpha, dtype=np.float64).copy()
    if knobs.alpha:
        alpha = alpha + knobs.alpha
        np.fill_diagonal(alpha, 0.0)
    knm_r = None
    if sim.coupling.knm_r is not None:
        knm_r = np.asarray(sim.coupling.knm_r, dtype=np.float64) * knobs.K
    sim.coupling = CouplingState(
        knm=scaled_knm,
        alpha=alpha,
        active_template=f"{sim.coupling.active_template}:studio_replay",
        knm_r=knm_r,
    )
    if knobs.zeta or knobs.Psi:
        sim.omegas = np.asarray(sim.omegas, dtype=np.float64) + knobs.zeta * knobs.Psi


def _deployment_blocked_reasons(
    exports: Sequence[ExportManifest],
) -> tuple[str, ...]:
    reasons: list[str] = []
    for manifest in exports:
        for warning in manifest.warnings:
            if warning not in reasons:
                reasons.append(warning)
    return tuple(reasons)


def _blocked_target(
    target: str,
    blocked_reasons: Sequence[str],
) -> dict[str, object]:
    return {
        "target": target,
        "status": "blocked",
        "required_artifacts": (),
        "commands": (),
        "operator_action": "resolve blocked reasons before packaging",
        "blocked_reasons": list(blocked_reasons),
    }


def _readiness_targets(
    readiness: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    raw_targets = readiness.get("targets", ())
    if isinstance(raw_targets, str | bytes) or not isinstance(raw_targets, Sequence):
        raise ValueError("readiness targets must be a sequence")
    targets: list[dict[str, object]] = []
    for index, raw_target in enumerate(raw_targets):
        if not isinstance(raw_target, Mapping):
            raise ValueError(f"readiness targets[{index}] must be a mapping")
        targets.append(
            {
                **dict(raw_target),
                "target": _require_non_empty_text(raw_target.get("target"), "target"),
                "status": _require_non_empty_text(raw_target.get("status"), "status"),
            }
        )
    return tuple(targets)


def _unique_artifacts(targets: Sequence[Mapping[str, object]]) -> list[str]:
    artifacts: list[str] = []
    for target in targets:
        required = target.get("required_artifacts", ())
        if isinstance(required, str | bytes) or not isinstance(required, Sequence):
            raise ValueError("required_artifacts must be a sequence")
        for artifact in required:
            name = _require_non_empty_text(artifact, "required_artifact")
            if name not in artifacts:
                artifacts.append(name)
    return artifacts


def _connector_by_transport(
    connector_plan: Mapping[str, object],
    transport: str,
) -> dict[str, object]:
    connectors = connector_plan.get("connectors", ())
    if isinstance(connectors, str | bytes) or not isinstance(connectors, Sequence):
        raise ValueError("connectors must be a sequence")
    for connector in connectors:
        if not isinstance(connector, Mapping):
            raise ValueError("connector entries must be mappings")
        if connector.get("transport") == transport:
            return dict(connector)
    raise ValueError(f"connector transport {transport!r} not found")


def _canvas_next_action(
    *,
    changed: bool,
    rewrite_status: str,
    operator_signoff: bool,
    apply_enabled: bool,
) -> str:
    if apply_enabled:
        return "apply reviewed binding rewrite or download artefacts"
    if rewrite_status != "review_ready":
        return "fix blocked canvas rewrite before apply"
    if changed and not operator_signoff:
        return "review artefacts and sign off before apply"
    if not changed:
        return "download artefacts or continue replay review"
    return "review canvas artefacts"


def _string_list(value: object, name: str) -> list[str]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of strings")
    items: list[str] = []
    for item in value:
        items.append(_require_non_empty_text(item, name))
    return items


def _owned_runtime_blocked_reasons(
    connector_plan: Mapping[str, object],
    transport: str,
    owner: str,
    auth_policy: Mapping[str, object],
) -> list[str]:
    blocked: list[str] = []
    connector = _connector_by_transport(connector_plan, transport)
    if transport not in {"rest", "grpc", "kafka", "hardware"}:
        blocked.append("owned runtime requires a live connector transport")
    if connector.get("compatible") is not True:
        blocked.append("connector manifest is incompatible")
    if not isinstance(owner, str) or not owner.strip():
        blocked.append("owner must be assigned")
    if not isinstance(auth_policy, Mapping):
        blocked.append("auth_policy must be a mapping")
        return blocked
    scheme = auth_policy.get("scheme")
    credential_label = auth_policy.get("credential_label")
    if not isinstance(scheme, str) or not scheme.strip():
        blocked.append("auth_policy.scheme must be assigned")
    if not isinstance(credential_label, str) or not credential_label.strip():
        blocked.append("auth_policy.credential_label must be assigned")
    return blocked


def _studio_service_processes() -> list[dict[str, object]]:
    validate_binding_command = (
        "python -m scpn_phase_orchestrator.cli validate binding_spec.yaml"
    )
    return [
        {
            "name": "spo-studio-ui",
            "image": "scpn-phase-orchestrator:local",
            "command": (
                "streamlit run tools/spo_studio.py "
                "--server.address 127.0.0.1 --server.port 8501"
            ),
            "ports": ["127.0.0.1:8501:8501"],
            "profiles": ["studio"],
            "healthcheck": validate_binding_command,
            "network_opened": False,
            "actuation_permitted": False,
        },
        {
            "name": "spo-binding-validator",
            "image": "scpn-phase-orchestrator:local",
            "command": validate_binding_command,
            "ports": [],
            "profiles": ["validation"],
            "healthcheck": validate_binding_command,
            "network_opened": False,
            "actuation_permitted": False,
        },
        {
            "name": "spo-connector-boundary",
            "image": "scpn-phase-orchestrator:local",
            "command": validate_binding_command,
            "ports": [],
            "profiles": ["connector-boundary-review"],
            "healthcheck": validate_binding_command,
            "network_opened": False,
            "actuation_permitted": False,
        },
    ]


def _render_service_compose_yaml(services: Sequence[Mapping[str, object]]) -> str:
    lines = ["services:"]
    for service in services:
        name = _require_non_empty_text(service.get("name"), "service.name")
        image = _require_non_empty_text(service.get("image"), "image")
        command = json.dumps(_require_non_empty_text(service.get("command"), "command"))
        lines.extend(
            [
                f"  {name}:",
                f"    image: {image}",
                "    working_dir: /workspace",
                "    volumes:",
                "      - .:/workspace:ro",
                f"    command: {command}",
            ]
        )
        ports = service.get("ports", ())
        if isinstance(ports, Sequence) and not isinstance(ports, str | bytes) and ports:
            lines.append("    ports:")
            for port in ports:
                port_text = json.dumps(_require_non_empty_text(port, "port"))
                lines.append(f"      - {port_text}")
        profiles = service.get("profiles", ())
        if (
            isinstance(profiles, Sequence)
            and not isinstance(profiles, str | bytes)
            and profiles
        ):
            lines.append("    profiles:")
            for profile in profiles:
                lines.append(
                    f"      - {json.dumps(_require_non_empty_text(profile, 'profile'))}"
                )
        healthcheck = json.dumps(
            _require_non_empty_text(service.get("healthcheck"), "healthcheck")
        )
        lines.extend(
            [
                "    healthcheck:",
                f'      test: ["CMD-SHELL", {healthcheck}]',
                "      interval: 30s",
                "      timeout: 10s",
                "      retries: 3",
            ]
        )
    return "\n".join(lines) + "\n"


def _owned_runtime_base_record(
    result: StudioReplayResult,
    *,
    transport: str,
    owner: str,
    payload_sha256: str,
    sequence: int,
    capability: str,
    direction: str,
) -> dict[str, object]:
    return {
        "record_kind": "studio_owned_live_connector_runtime",
        "project_name": result.project_state.project_name,
        "transport": transport,
        "owner": owner.strip() if isinstance(owner, str) else "",
        "contract_hash": result.connector_plan.get("contract_hash", ""),
        "capability": capability,
        "direction": direction,
        "sequence": sequence,
        "payload_sha256": payload_sha256,
        "network_opened": False,
        "actuation_permitted": False,
        "hardware_write_permitted": False,
    }


def _result_binding_spec_path(result: StudioReplayResult) -> Path:
    source_path = result.project_state.binding.provenance.get("source_path")
    if not isinstance(source_path, str) or not source_path.strip():
        raise ValueError("project binding provenance must include source_path")
    return Path(source_path)


def _run_owned_live_adapter(
    contract: DigitalTwinBindingContract,
    *,
    transport: str,
    envelope_record: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    headers = {"authorization": "Bearer studio-owned-runtime"}
    if transport == "rest":
        rest = DigitalTwinSyncRestAdapter.for_contract(contract, name="studio-rest")
        response = rest.handle_post(envelope_record, headers=headers)
        return response.to_audit_record(), rest.to_audit_record()
    if transport == "grpc":
        grpc = DigitalTwinSyncGrpcAdapter.for_contract(contract, name="studio-grpc")
        response = grpc.handle_unary(envelope_record, metadata=headers)
        return response.to_audit_record(), grpc.to_audit_record()
    if transport == "kafka":
        kafka = DigitalTwinSyncKafkaAdapter.for_contract(contract, name="studio-kafka")
        response = kafka.handle_message(
            {"topic": kafka.topic, "value": envelope_record},
            headers=headers,
        )
        return response.to_audit_record(), kafka.to_audit_record()
    if transport == "hardware":
        hardware = DigitalTwinSyncHardwareAdapter.for_contract(
            contract,
            name="studio-hardware",
            device_ids=("studio-review-device",),
        )
        response = hardware.handle_frame(
            {
                "device_id": "studio-review-device",
                "safety_interlock": True,
                "value": envelope_record,
            },
            headers=headers,
        )
        return response.to_audit_record(), hardware.to_audit_record()
    raise ValueError(f"connector transport {transport!r} is not a live runtime")


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative int")
    return value


def _materialisation_command_writes_artifact(command: object) -> bool:
    command_text = _require_non_empty_text(command, "command")
    return any(
        marker in command_text
        for marker in (
            "docker build",
            "docker run",
            "wasm-pack build",
        )
    )


def _stable_json_payload(value: object, field_name: str) -> str:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return json.dumps(
        _normalise_json_mapping(value, field_name),
        sort_keys=True,
        separators=(",", ":"),
    )


def _normalise_json_mapping(
    value: Mapping[object, object],
    field_name: str,
) -> dict[str, object]:
    safe: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{field_name} contains an invalid key")
        safe[key] = _normalise_json_value(item, field_name)
    return safe


def _normalise_json_value(value: object, field_name: str) -> object:
    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{field_name} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        return _normalise_json_mapping(value, field_name)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_normalise_json_value(item, field_name) for item in value]
    raise ValueError(f"{field_name} contains a non-JSON-safe value")


def _normalise_hardware_evidence(
    evidence: Mapping[str, object],
) -> tuple[dict[str, object], list[str]]:
    invalid: list[str] = []
    normalised: dict[str, object] = {}
    for field in (
        "generated_artifact_path",
        "simulator_parity_report",
        "target_toolchain",
        "target_toolchain_version",
    ):
        value = evidence.get(field)
        if isinstance(value, str) and value.strip():
            normalised[field] = value.strip()
        else:
            invalid.append(f"{field} is required")

    for field in ("generated_artifact_sha256", "simulator_parity_sha256"):
        value = evidence.get(field)
        if _is_sha256_digest(value):
            normalised[field] = str(value).lower()
        elif value is None:
            invalid.append(f"{field} is required")
        else:
            invalid.append(f"{field} must be a SHA-256 digest")

    parity_status = evidence.get("simulator_parity_status")
    if isinstance(parity_status, str) and parity_status.strip().lower() == "passed":
        normalised["simulator_parity_status"] = "passed"
    else:
        invalid.append("simulator_parity_status must be passed")

    if evidence.get("operator_signoff") is True:
        normalised["operator_signoff"] = True
    else:
        invalid.append("operator_signoff must be true")
    return normalised, invalid


def _blocked_binding_rewrite_candidate(
    result: StudioReplayResult,
    before_digest: str,
    validation_errors: Sequence[str],
) -> dict[str, object]:
    yaml_text = result.project_state.binding.yaml_text
    return {
        "candidate_kind": "canvas_binding_rewrite_candidate",
        "project_name": result.project_state.project_name,
        "status": "blocked",
        "binding_spec_rewritten": False,
        "actuation_permitted": False,
        "network_opened": False,
        "before_yaml_sha256": before_digest,
        "candidate_yaml_sha256": before_digest,
        "coupling_count_before": int(result.canvas_graph["edge_count"]),
        "coupling_count_after": int(result.canvas_graph["edge_count"]),
        "validation_errors": list(validation_errors),
        "candidate_yaml": yaml_text,
    }


def _binding_apply_blocked_reasons(
    candidate: Mapping[str, object],
    path: Path,
    candidate_yaml: str,
    before_digest: str,
    candidate_digest: str,
    *,
    operator_signoff: bool,
) -> list[str]:
    blocked: list[str] = []
    if candidate.get("candidate_kind") != "canvas_binding_rewrite_candidate":
        blocked.append("candidate_kind must be canvas_binding_rewrite_candidate")
    if candidate.get("status") != "review_ready":
        blocked.append("candidate status must be review_ready")
    if operator_signoff is not True:
        blocked.append("operator_signoff must be true")
    if not path.exists() or not path.is_file():
        blocked.append("binding_spec_path must point to an existing file")
    if sha256(candidate_yaml.encode("utf-8")).hexdigest() != candidate_digest:
        blocked.append("candidate YAML SHA-256 does not match candidate metadata")

    validation_errors = _validate_candidate_binding_yaml(candidate_yaml)
    blocked.extend(validation_errors)

    if path.exists() and path.is_file():
        current_yaml = path.read_text(encoding="utf-8")
        current_digest = sha256(current_yaml.encode("utf-8")).hexdigest()
        if current_digest != before_digest:
            blocked.append(
                "current binding_spec.yaml SHA-256 does not match candidate source"
            )
    return blocked


def _binding_apply_record(
    candidate: Mapping[str, object],
    path: Path,
    *,
    status: str,
    before_digest: str,
    after_digest: str,
    backup_path: str,
    blocked_reasons: Sequence[str],
) -> dict[str, object]:
    return {
        "apply_kind": "canvas_binding_rewrite_apply",
        "candidate_kind": candidate.get("candidate_kind", ""),
        "project_name": candidate.get("project_name", ""),
        "status": status,
        "binding_spec_path": str(path),
        "backup_path": backup_path,
        "binding_spec_rewritten": status == "applied",
        "actuation_permitted": False,
        "network_opened": False,
        "before_yaml_sha256": before_digest,
        "after_yaml_sha256": after_digest,
        "candidate_yaml_sha256": candidate.get("candidate_yaml_sha256", ""),
        "blocked_reasons": list(blocked_reasons),
    }


def _next_binding_backup_path(path: Path, before_digest: str) -> Path:
    stem = f"{path.name}.studio-backup-{before_digest[:12]}.bak"
    backup = path.with_name(stem)
    if not backup.exists():
        return backup
    for index in range(1, 1000):
        candidate = path.with_name(f"{stem}.{index}")
        if not candidate.exists():
            return candidate
    raise RuntimeError("could not allocate binding backup path")


def _atomic_write_text(path: Path, text: str) -> None:
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = handle.name
            handle.write(text)
            handle.flush()
        Path(tmp_path).replace(path)
    except OSError:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
        raise


def _rewrite_binding_cross_channel_couplings(
    yaml_text: str,
    edges: Sequence[Mapping[str, object]],
) -> str:
    import yaml

    raw = yaml.safe_load(yaml_text)
    if not isinstance(raw, dict):
        raise ValueError("binding YAML must contain a mapping")
    raw["cross_channel_couplings"] = [
        _canvas_edge_to_cross_channel_coupling(edge) for edge in edges
    ]
    rendered: str = yaml.safe_dump(raw, sort_keys=False)
    return rendered


def _canvas_edge_to_cross_channel_coupling(
    edge: Mapping[str, object],
) -> dict[str, object]:
    source = _canvas_edge_channel(edge, "source")
    target = _canvas_edge_channel(edge, "target")
    if source == target:
        raise ValueError("cross-channel coupling source and target must differ")
    return {
        "source": source,
        "target": target,
        "strength": _finite_range(
            edge.get("strength", 0.0),
            "cross-channel coupling strength",
            low=0.0,
            high=100.0,
        ),
        "mode": _require_non_empty_text(edge.get("mode", "directed"), "mode"),
        "template": _require_non_empty_text(
            edge.get("template", "studio_canvas_review"),
            "template",
        ),
    }


def _canvas_edge_channel(edge: Mapping[str, object], endpoint: str) -> str:
    explicit = edge.get(f"{endpoint}_channel")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    raw_endpoint = _require_non_empty_text(edge.get(endpoint), endpoint)
    if raw_endpoint.startswith("channel_"):
        return raw_endpoint.removeprefix("channel_")
    raise ValueError(f"{endpoint} must reference a channel node")


def _validate_candidate_binding_yaml(candidate_yaml: str) -> list[str]:
    with tempfile.TemporaryDirectory() as tmpdir:
        spec_path = Path(tmpdir) / "binding_spec.yaml"
        spec_path.write_text(candidate_yaml, encoding="utf-8")
        try:
            spec = load_binding_spec(spec_path)
        except Exception as exc:
            return [f"candidate binding failed to load: {type(exc).__name__}"]
        return list(validate_binding_spec(spec))


def _is_sha256_digest(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdefABCDEF" for character in value)


def _require_sha256_digest(value: object, field_name: str) -> str:
    if not _is_sha256_digest(value):
        raise ValueError(f"{field_name} must be a SHA-256 digest")
    return str(value)


def _require_non_empty_payload(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _layer_metrics(value: object) -> tuple[tuple[str, float], ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return ()
    rows: list[tuple[str, float]] = []
    for index, layer in enumerate(value):
        if not isinstance(layer, Mapping):
            continue
        name = _require_non_empty_text(layer.get("name", f"layer_{index}"), "layer")
        rows.append((name, _finite_number(layer.get("R", 0.0), "layer.R")))
    return tuple(rows)


def _normalise_table_rows(
    rows: Sequence[Mapping[str, object]],
    field_name: str,
) -> list[dict[str, object]]:
    if isinstance(rows, str | bytes) or not isinstance(rows, Sequence):
        raise ValueError(f"{field_name} must be a sequence of mappings")
    normalised: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"{field_name}[{index}] must be a mapping")
        safe_row: dict[str, object] = {}
        for key, value in row.items():
            if not isinstance(key, str):
                raise ValueError(f"{field_name}[{index}] contains a non-string key")
            if value is None or isinstance(value, str | int | float | bool):
                if isinstance(value, float) and not isfinite(value):
                    raise ValueError(f"{field_name}[{index}].{key} must be finite")
                safe_row[key] = value
            else:
                raise ValueError(f"{field_name}[{index}].{key} must be JSON-safe")
        normalised.append(safe_row)
    return normalised


def _normalise_canvas_graph(
    graph: Mapping[str, object],
    field_name: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not isinstance(graph, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    nodes = graph.get("nodes", ())
    edges = graph.get("edges", ())
    if isinstance(nodes, str | bytes) or not isinstance(nodes, Sequence):
        raise ValueError("canvas nodes must be a sequence of mappings")
    if isinstance(edges, str | bytes) or not isinstance(edges, Sequence):
        raise ValueError("canvas edges must be a sequence of mappings")
    return (
        _normalise_table_rows(nodes, "canvas nodes"),
        _normalise_table_rows(edges, "canvas edges"),
    )


def _validate_canvas_edge_endpoints(
    nodes: Sequence[Mapping[str, object]],
    edges: Sequence[Mapping[str, object]],
) -> None:
    node_ids = {
        _require_non_empty_text(node.get("id"), "canvas node id") for node in nodes
    }
    for edge in edges:
        edge_id = _require_non_empty_text(edge.get("id"), "canvas edge id")
        source = _require_non_empty_text(edge.get("source"), "canvas edge source")
        target = _require_non_empty_text(edge.get("target"), "canvas edge target")
        if source not in node_ids or target not in node_ids:
            raise ValueError(f"canvas edge {edge_id!r} references unknown endpoint")


def _canvas_item_changes(
    before_items: Sequence[Mapping[str, object]],
    after_items: Sequence[Mapping[str, object]],
    *,
    fields: Sequence[str],
) -> dict[str, list[dict[str, object]]]:
    before = _canvas_item_index(before_items, fields=fields)
    after = _canvas_item_index(after_items, fields=fields)
    before_ids = set(before)
    after_ids = set(after)
    common_ids = before_ids & after_ids
    return {
        "added": [after[item_id] for item_id in sorted(after_ids - before_ids)],
        "removed": [before[item_id] for item_id in sorted(before_ids - after_ids)],
        "modified": [
            {"id": item_id, "before": before[item_id], "after": after[item_id]}
            for item_id in sorted(common_ids)
            if before[item_id] != after[item_id]
        ],
    }


def _canvas_item_index(
    items: Sequence[Mapping[str, object]],
    *,
    fields: Sequence[str],
) -> dict[str, dict[str, object]]:
    indexed: dict[str, dict[str, object]] = {}
    for item in items:
        item_id = _require_non_empty_text(item.get("id"), "canvas item id")
        if item_id in indexed:
            raise ValueError(f"canvas item id {item_id!r} must be unique")
        indexed[item_id] = {
            field: item[field]
            for field in fields
            if field in item and item[field] is not None
        }
    return indexed


def _canvas_channel_id(channel: str) -> str:
    safe = "".join(character if character.isalnum() else "_" for character in channel)
    return f"channel_{safe}"


def _finite_range(value: object, name: str, *, low: float, high: float) -> float:
    number = _finite_number(value, name)
    if not low <= number <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return number


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be finite")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _require_non_empty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()
