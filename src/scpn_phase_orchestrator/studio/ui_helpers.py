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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import numpy as np

from scpn_phase_orchestrator.binding import validate_binding_spec
from scpn_phase_orchestrator.binding.digital_twin import (
    build_digital_twin_adapter_manifest,
    build_digital_twin_binding_contract,
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
    "apply_knob_update",
    "binding_spec_project_state",
    "build_beginner_guidance",
    "build_canvas_edit_artifact",
    "build_canvas_graph",
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
    "build_oscillator_edit_artifact",
    "build_oscillator_table",
    "build_operator_checklist",
    "build_regime_chart_payload",
    "build_runtime_snapshot",
    "build_series_chart_payload",
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
