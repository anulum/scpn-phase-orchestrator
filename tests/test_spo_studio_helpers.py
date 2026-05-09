# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SPO Studio helper tests

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_phase_orchestrator.binding.loader import load_binding_spec
from scpn_phase_orchestrator.studio import BindingProposal
from scpn_phase_orchestrator.studio.ui_helpers import (
    StudioKnobState,
    apply_knob_update,
    binding_spec_project_state,
    build_beginner_guidance,
    build_canvas_edit_artifact,
    build_canvas_graph,
    build_canvas_layout_manifest,
    build_canvas_topology_patch,
    build_command_table,
    build_deployment_package,
    build_deployment_readiness,
    build_error_report,
    build_export_manifests,
    build_hardware_target_package,
    build_layer_table,
    build_live_connector_plan,
    build_operator_checklist,
    build_oscillator_edit_artifact,
    build_oscillator_table,
    build_package_materialisation_plan,
    build_regime_chart_payload,
    build_runtime_snapshot,
    build_series_chart_payload,
    build_verified_hardware_target_package,
    disabled_export_reasons,
    run_binding_spec_replay,
)


def _minimal_spec_path() -> Path:
    return Path("domainpacks/minimal_domain/binding_spec.yaml")


def test_chart_payloads_are_stable_and_dense() -> None:
    r_payload = build_series_chart_payload("R", [0.1, 0.5, 0.9])
    regime_payload = build_regime_chart_payload(["critical", "degraded", "nominal"])

    assert r_payload == [
        {"step": 1, "R": 0.1},
        {"step": 2, "R": 0.5},
        {"step": 3, "R": 0.9},
    ]
    assert regime_payload == [
        {"step": 1, "regime": "critical", "regime_level": 0.0},
        {"step": 2, "regime": "degraded", "regime_level": 1.0},
        {"step": 3, "regime": "nominal", "regime_level": 2.0},
    ]


def test_knob_update_rejects_unsafe_values() -> None:
    knobs = StudioKnobState()
    updated = apply_knob_update(knobs, K=1.5, alpha=0.2, zeta=0.4, Psi=3.0)

    assert updated.to_audit_record() == {
        "K": 1.5,
        "alpha": 0.2,
        "zeta": 0.4,
        "Psi": 3.0,
    }
    with pytest.raises(ValueError, match="K"):
        apply_knob_update(knobs, K=0.0)
    with pytest.raises(ValueError, match="Psi"):
        apply_knob_update(knobs, Psi=-0.1)


def test_binding_spec_tables_describe_oscillators_and_layers() -> None:
    spec = load_binding_spec(_minimal_spec_path())

    oscillators = build_oscillator_table(spec)
    layers = build_layer_table(spec)
    canvas = build_canvas_graph(spec)

    assert oscillators
    assert layers
    assert {"layer", "oscillator_id", "channel", "family"} <= set(oscillators[0])
    assert {"index", "name", "oscillator_count", "family"} <= set(layers[0])
    assert canvas["canvas_kind"] == "layer_coupling_graph"
    assert canvas["layer_count"] == len(spec.layers)
    assert canvas["channel_count"] == 3
    assert canvas["nodes"][0] == {
        "id": "layer_0",
        "label": "lower",
        "kind": "layer",
        "layer_index": 0,
        "family": "",
        "channel": "",
        "oscillator_count": 2,
        "x": 0.0,
        "y": 0.0,
    }
    assert canvas["nodes"][-1]["kind"] == "channel"
    assert canvas["edge_count"] == 0


def test_canvas_graph_exposes_layer_and_cross_channel_edges() -> None:
    spec = load_binding_spec(
        Path("domainpacks/digital_twin_nchannel/binding_spec.yaml")
    )

    canvas = build_canvas_graph(spec)

    assert canvas["layer_count"] == len(spec.layers)
    assert canvas["channel_count"] == len(spec.used_channels())
    assert canvas["edge_count"] == len(spec.cross_channel_couplings)
    assert {f"layer_{layer.index}" for layer in spec.layers} <= {
        node["id"] for node in canvas["nodes"]
    }
    assert {"channel_Thermal", "channel_TwinResidual", "channel_P"} <= {
        node["id"] for node in canvas["nodes"]
    }
    edge = canvas["edges"][0]
    assert {
        "id",
        "source",
        "target",
        "kind",
        "source_channel",
        "target_channel",
        "strength",
        "mode",
        "template",
    } <= set(edge)
    assert edge["kind"] == "cross_channel_coupling"
    assert edge["source"] == "channel_Thermal"
    assert edge["target"] == "channel_P"


def test_canvas_edit_artifact_records_reviewable_node_and_edge_changes() -> None:
    before = {
        "nodes": (
            {
                "id": "layer_0",
                "label": "source",
                "kind": "layer",
                "x": 0.0,
                "y": 0.0,
            },
        ),
        "edges": (),
    }
    after = {
        "nodes": (
            {
                "id": "layer_0",
                "label": "source",
                "kind": "layer",
                "x": 12.5,
                "y": 0.0,
            },
        ),
        "edges": (
            {
                "id": "manual_edge_1",
                "source": "layer_0",
                "target": "layer_1",
                "kind": "review_edge",
            },
        ),
    }

    artifact = build_canvas_edit_artifact(before, after)
    record = json.loads(artifact.payload)

    assert artifact.target_kind == "canvas_edit_review"
    assert artifact.file_name == "canvas_edit_review.json"
    assert artifact.safety_posture == "review_artifact"
    assert record["changed"] is True
    assert record["node_count_before"] == 1
    assert record["node_count_after"] == 1
    assert record["edge_count_after"] == 1
    assert record["nodes_after"][0]["x"] == 12.5


def test_canvas_layout_manifest_persists_positions_without_edges() -> None:
    graph = {
        "nodes": (
            {
                "id": "layer_0",
                "label": "source",
                "kind": "layer",
                "x": 12.5,
                "y": 8.0,
            },
            {
                "id": "channel_P",
                "label": "P",
                "kind": "channel",
                "x": 220.0,
                "y": 420.0,
            },
        ),
        "edges": (
            {
                "id": "edge_1",
                "source": "layer_0",
                "target": "channel_P",
                "kind": "review_edge",
            },
        ),
    }

    artifact = build_canvas_layout_manifest(
        project_name="minimal_domain",
        graph=graph,
    )
    record = json.loads(artifact.payload)

    assert artifact.target_kind == "canvas_layout_manifest"
    assert artifact.file_name == "canvas_layout_manifest.json"
    assert record["manifest_kind"] == "canvas_layout_manifest"
    assert record["project_name"] == "minimal_domain"
    assert record["node_count"] == 2
    assert record["edge_count"] == 1
    assert record["positions"] == [
        {"id": "channel_P", "kind": "channel", "label": "P", "x": 220.0, "y": 420.0},
        {"id": "layer_0", "kind": "layer", "label": "source", "x": 12.5, "y": 8.0},
    ]


def test_canvas_layout_manifest_rejects_missing_coordinates() -> None:
    with pytest.raises(ValueError, match="canvas layout"):
        build_canvas_layout_manifest(
            project_name="minimal_domain",
            graph={"nodes": ({"id": "layer_0", "kind": "layer"},), "edges": ()},
        )


def test_canvas_topology_patch_records_added_and_removed_edges() -> None:
    before = {
        "nodes": (
            {
                "id": "layer_0",
                "label": "source",
                "kind": "layer",
                "x": 0.0,
                "y": 0.0,
            },
            {
                "id": "layer_1",
                "label": "sink",
                "kind": "layer",
                "x": 220.0,
                "y": 0.0,
            },
        ),
        "edges": (
            {
                "id": "edge_old",
                "source": "layer_0",
                "target": "layer_1",
                "kind": "review_edge",
            },
        ),
    }
    after = {
        "nodes": before["nodes"],
        "edges": (
            {
                "id": "edge_new",
                "source": "layer_1",
                "target": "layer_0",
                "kind": "review_edge",
            },
        ),
    }

    artifact = build_canvas_topology_patch(
        project_name="minimal_domain",
        before_graph=before,
        after_graph=after,
    )
    record = json.loads(artifact.payload)

    assert artifact.target_kind == "canvas_topology_patch"
    assert artifact.file_name == "canvas_topology_patch.json"
    assert record["patch_kind"] == "canvas_topology_patch"
    assert record["project_name"] == "minimal_domain"
    assert record["changed"] is True
    assert record["status"] == "review_required"
    assert record["node_changes"] == {
        "added": [],
        "removed": [],
        "modified": [],
    }
    assert record["edge_changes"]["removed"] == [
        {
            "id": "edge_old",
            "kind": "review_edge",
            "source": "layer_0",
            "target": "layer_1",
        }
    ]
    assert record["edge_changes"]["added"] == [
        {
            "id": "edge_new",
            "kind": "review_edge",
            "source": "layer_1",
            "target": "layer_0",
        }
    ]


def test_canvas_topology_patch_rejects_edges_with_unknown_endpoints() -> None:
    with pytest.raises(ValueError, match="unknown endpoint"):
        build_canvas_topology_patch(
            project_name="minimal_domain",
            before_graph={"nodes": ({"id": "layer_0", "kind": "layer"},), "edges": ()},
            after_graph={
                "nodes": ({"id": "layer_0", "kind": "layer"},),
                "edges": (
                    {
                        "id": "edge_bad",
                        "source": "layer_0",
                        "target": "missing",
                        "kind": "review_edge",
                    },
                ),
            },
        )


def test_canvas_edit_artifact_rejects_invalid_canvas_shape() -> None:
    with pytest.raises(ValueError, match="canvas nodes"):
        build_canvas_edit_artifact(
            {"nodes": "bad", "edges": ()},
            {"nodes": (), "edges": ()},
        )


def test_live_connector_plan_declares_owned_non_opened_transports() -> None:
    spec = load_binding_spec(
        Path("domainpacks/digital_twin_nchannel/binding_spec.yaml")
    )

    plan = build_live_connector_plan(spec)

    assert plan["plan_kind"] == "studio_live_connector_plan"
    assert plan["project_name"] == "digital_twin_nchannel"
    assert len(plan["contract_hash"]) == 64
    assert plan["network_opened"] is False
    assert plan["actuation_permitted"] is False
    assert [connector["transport"] for connector in plan["connectors"]] == [
        "memory",
        "jsonl",
        "rest",
        "grpc",
        "kafka",
        "hardware",
    ]
    memory, jsonl, rest, grpc, kafka, hardware = plan["connectors"]
    assert memory["status"] == "review_ready"
    assert jsonl["supports_replay"] is True
    for connector in (rest, grpc, kafka, hardware):
        assert connector["status"] == "owner_required"
        assert connector["requires_auth"] is True
        assert connector["operator_action"] == "assign connector owner and auth policy"
    assert hardware["hardware_write_permitted"] is False
    assert "live_transport_requires_auth" not in json.dumps(plan)


def test_hardware_target_package_requires_evidence_before_readiness() -> None:
    result = run_binding_spec_replay(
        _minimal_spec_path(),
        steps=3,
        knobs=StudioKnobState(K=1.0),
    )

    package = build_hardware_target_package(result)

    assert package["package_kind"] == "studio_hardware_target_package"
    assert package["project_name"] == "minimal_domain"
    assert package["overall_status"] == "evidence_required"
    assert package["hardware_write_permitted"] is False
    assert package["network_opened"] is False
    assert package["targets"] == ["fpga_verilog", "neuromorphic_schedule"]
    assert package["required_evidence"] == [
        "generated hardware artefact path",
        "simulator parity report",
        "target toolchain version",
        "operator sign-off",
    ]
    assert package["commands"] == [
        "review connector_plan.json",
        "generate FPGA Verilog with KuramotoVerilogCompiler",
        "run simulator parity before hardware handoff",
    ]
    assert package["connector"]["transport"] == "hardware"
    assert package["connector"]["status"] == "owner_required"
    assert len(package["contract_hash"]) == 64


def test_verified_hardware_target_package_accepts_complete_evidence() -> None:
    result = run_binding_spec_replay(
        _minimal_spec_path(),
        steps=3,
        knobs=StudioKnobState(K=1.0),
    )
    evidence = {
        "generated_artifact_path": "build/hardware/minimal_domain/fpga_top.v",
        "generated_artifact_sha256": "a" * 64,
        "simulator_parity_report": "reports/minimal_domain_parity.json",
        "simulator_parity_sha256": "b" * 64,
        "simulator_parity_status": "passed",
        "target_toolchain": "yosys-nextpnr",
        "target_toolchain_version": "yosys 0.40 / nextpnr 0.7",
        "operator_signoff": True,
    }

    package = build_verified_hardware_target_package(result, evidence=evidence)

    assert package["package_kind"] == "studio_verified_hardware_target_package"
    assert package["project_name"] == "minimal_domain"
    assert package["overall_status"] == "review_ready"
    assert package["evidence_status"] == "verified"
    assert package["hardware_write_permitted"] is False
    assert package["network_opened"] is False
    assert package["invalid_evidence"] == []
    assert package["evidence"]["generated_artifact_path"] == (
        "build/hardware/minimal_domain/fpga_top.v"
    )
    assert package["evidence"]["simulator_parity_status"] == "passed"
    assert package["commands"] == [
        "review verified_hardware_target_package.json",
        "compare generated artefact hash before handoff",
        "archive simulator parity report with package",
    ]


def test_verified_hardware_target_package_blocks_incomplete_evidence() -> None:
    result = run_binding_spec_replay(
        _minimal_spec_path(),
        steps=3,
        knobs=StudioKnobState(K=1.0),
    )

    package = build_verified_hardware_target_package(
        result,
        evidence={
            "generated_artifact_path": "build/hardware/minimal_domain/fpga_top.v",
            "generated_artifact_sha256": "not-a-sha",
            "simulator_parity_status": "failed",
            "operator_signoff": False,
        },
    )

    assert package["overall_status"] == "evidence_required"
    assert package["evidence_status"] == "blocked"
    assert package["hardware_write_permitted"] is False
    assert package["commands"] == []
    assert (
        "generated_artifact_sha256 must be a SHA-256 digest"
        in package["invalid_evidence"]
    )
    assert "simulator_parity_report is required" in package["invalid_evidence"]
    assert "simulator_parity_status must be passed" in package["invalid_evidence"]
    assert "operator_signoff must be true" in package["invalid_evidence"]


def test_beginner_guidance_explains_runtime_in_domain_terms() -> None:
    result = run_binding_spec_replay(
        _minimal_spec_path(),
        steps=3,
        knobs=StudioKnobState(K=1.3, alpha=0.2, zeta=0.1, Psi=0.5),
    )

    guidance = build_beginner_guidance(result)

    assert guidance["guide_kind"] == "beginner_mode"
    assert guidance["project_name"] == "minimal_domain"
    assert guidance["actuation_permitted"] is False
    assert guidance["runtime_summary"]["replay_status"] == "completed"
    assert guidance["runtime_summary"]["regime"] == result.project_state.runtime.regime
    assert guidance["runtime_summary"]["domain_signal"] == (
        "R summarises how closely the reviewed domain signals move together."
    )
    assert [card["title"] for card in guidance["concept_cards"]] == [
        "Signals",
        "Coupling",
        "Objectives",
        "Supervisor",
    ]
    assert guidance["concept_cards"][0]["evidence"]["layers"] == ["lower", "upper"]
    assert guidance["concept_cards"][0]["evidence"]["channels"] == ["I", "P", "S"]
    assert guidance["concept_cards"][1]["evidence"]["K"] == 1.3
    assert guidance["next_actions"][0] == "review binding validation"
    assert [step["title"] for step in guidance["walkthrough_steps"]] == [
        "Load project",
        "Run replay",
        "Review binding",
        "Inspect canvas",
        "Prepare exports",
    ]
    assert [step["status"] for step in guidance["walkthrough_steps"]] == [
        "complete",
        "complete",
        "complete",
        "complete",
        "ready",
    ]
    assert guidance["walkthrough_steps"][3]["evidence"] == {
        "layers": 2,
        "channels": 3,
        "couplings": 0,
    }
    assert all("Kuramoto" not in json.dumps(card) for card in guidance["concept_cards"])


def test_beginner_guidance_walkthrough_blocks_on_validation_errors() -> None:
    manifests = build_export_manifests(
        project_name="broken",
        binding_yaml="version: 1\n",
        audit_payload={"project_name": "broken"},
        validation_errors=("layer missing",),
    )
    result = run_binding_spec_replay(
        _minimal_spec_path(),
        steps=3,
        knobs=StudioKnobState(K=1.0),
    )
    broken_state = type(result.project_state)(
        project_name=result.project_state.project_name,
        source=result.project_state.source,
        binding=BindingProposal(
            yaml_text=result.project_state.binding.yaml_text,
            validation_errors=("layer missing",),
            inferred_channels=tuple(result.project_state.binding.inferred_channels),
            confidence_factors=dict(result.project_state.binding.confidence_factors),
            provenance=dict(result.project_state.binding.provenance),
        ),
        runtime=result.project_state.runtime,
        exports=manifests,
        metadata=result.project_state.metadata,
    )
    broken_result = type(result)(
        project_state=broken_state,
        r_history=result.r_history,
        regime_history=result.regime_history,
        layer_table=result.layer_table,
        oscillator_table=result.oscillator_table,
        canvas_graph=result.canvas_graph,
        connector_plan=result.connector_plan,
        export_manifests=manifests,
    )

    guidance = build_beginner_guidance(broken_result)

    assert guidance["walkthrough_steps"][2]["status"] == "blocked"
    assert guidance["walkthrough_steps"][2]["evidence"] == {
        "validation_errors": ["layer missing"]
    }
    assert guidance["walkthrough_steps"][4]["status"] == "blocked"


def test_beginner_guidance_blocks_invalid_result_shape() -> None:
    with pytest.raises(ValueError, match="replay result"):
        build_beginner_guidance(object())


def test_runtime_snapshot_and_exports_are_review_safe() -> None:
    spec_path = _minimal_spec_path()
    yaml_text = spec_path.read_text(encoding="utf-8")
    state = binding_spec_project_state(
        project_name="minimal_domain",
        spec_path=spec_path,
        knobs=StudioKnobState(K=1.2, alpha=0.1, zeta=0.0, Psi=0.0),
        runtime=build_runtime_snapshot(
            final_state={
                "R_global": 0.82,
                "regime": "nominal",
                "layers": [{"name": "layer-a", "R": 0.8}],
            },
            knobs=StudioKnobState(K=1.2, alpha=0.1),
            replay_status="completed",
        ),
    )
    exports = build_export_manifests(
        project_name="minimal_domain",
        binding_yaml=yaml_text,
        audit_payload=state.to_audit_record(),
        validation_errors=(),
    )
    restored = json.loads(exports[1].payload)

    assert state.source.source_kind == "binding_spec_yaml"
    assert all(manifest.safety_posture == "review_artifact" for manifest in exports)
    assert restored["project_name"] == "minimal_domain"
    assert disabled_export_reasons(()) == ()


def test_deployment_readiness_records_guided_target_status() -> None:
    state = binding_spec_project_state(
        project_name="minimal_domain",
        spec_path=_minimal_spec_path(),
        knobs=StudioKnobState(K=1.0),
        runtime=build_runtime_snapshot(
            final_state={
                "R_global": 0.72,
                "regime": "nominal",
                "layers": [{"name": "layer-a", "R": 0.7}],
            },
            knobs=StudioKnobState(K=1.0),
            replay_status="completed",
        ),
    )

    readiness = build_deployment_readiness(state)

    assert readiness["project_name"] == "minimal_domain"
    assert readiness["overall_status"] == "review_ready"
    assert readiness["operator_next_step"] == "review target-specific packaging"
    assert [target["target"] for target in readiness["targets"]] == [
        "docker",
        "wasm",
        "hardware",
    ]
    docker, wasm, hardware = readiness["targets"]
    assert docker["status"] == "ready"
    assert docker["required_artifacts"] == [
        "binding_spec.yaml",
        "spo_studio_audit.json",
        "docker_manifest.json",
    ]
    assert docker["commands"] == [
        "docker compose config",
        "docker build -t scpn-phase-orchestrator:local .",
        "docker run --rm -v $PWD:/workspace scpn-phase-orchestrator:local "
        "spo run binding_spec.yaml --audit audit.jsonl",
    ]
    assert wasm["status"] == "ready"
    assert wasm["commands"] == [
        "cd spo-kernel && wasm-pack build crates/spo-wasm --target web "
        "--out-dir ../../../docs/wasm-pkg",
    ]
    assert hardware["status"] == "postponed"
    assert hardware["operator_action"] == "attach verified hardware-target evidence"
    assert hardware["commands"] == []


def test_deployment_package_collects_ready_artifacts_and_safety_gates() -> None:
    state = binding_spec_project_state(
        project_name="minimal_domain",
        spec_path=_minimal_spec_path(),
        knobs=StudioKnobState(K=1.0),
        runtime=build_runtime_snapshot(
            final_state={
                "R_global": 0.72,
                "regime": "nominal",
                "layers": [{"name": "layer-a", "R": 0.7}],
            },
            knobs=StudioKnobState(K=1.0),
            replay_status="completed",
        ),
    )

    package = build_deployment_package(state)

    assert package["package_kind"] == "studio_deployment_package"
    assert package["project_name"] == "minimal_domain"
    assert package["overall_status"] == "review_ready"
    assert package["ready_targets"] == ["docker", "wasm"]
    assert package["postponed_targets"] == ["hardware"]
    assert package["blocked_targets"] == []
    assert package["required_artifacts"] == [
        "binding_spec.yaml",
        "spo_studio_audit.json",
        "docker_manifest.json",
        "wasm_manifest.json",
        "verified_hardware_target_evidence",
    ]
    assert package["safety_gates"] == [
        "local replay completed",
        "binding validation passed",
        "live actuation disabled",
        "hardware output requires verified evidence",
    ]
    assert [artifact["file_name"] for artifact in package["export_artifacts"]] == [
        "binding_spec.yaml",
        "spo_studio_audit.json",
        "docker_manifest.json",
        "wasm_manifest.json",
    ]
    assert all(
        len(artifact["payload_sha256"]) == 64
        for artifact in package["export_artifacts"]
    )
    assert package["commands"][0]["target"] == "docker"


def test_deployment_readiness_blocks_targets_on_validation_errors() -> None:
    manifests = build_export_manifests(
        project_name="broken",
        binding_yaml="version: 1\n",
        audit_payload={"project_name": "broken"},
        validation_errors=("layer missing",),
    )
    state = binding_spec_project_state(
        project_name="minimal_domain",
        spec_path=_minimal_spec_path(),
        knobs=StudioKnobState(K=1.0),
        runtime=build_runtime_snapshot(
            final_state={"R_global": 0.72, "regime": "nominal"},
            knobs=StudioKnobState(K=1.0),
            replay_status="completed",
        ),
    )
    broken_state = type(state)(
        project_name=state.project_name,
        source=state.source,
        binding=state.binding,
        runtime=state.runtime,
        exports=manifests,
        metadata=state.metadata,
    )

    readiness = build_deployment_readiness(broken_state)
    package = build_deployment_package(broken_state)

    assert readiness["overall_status"] == "blocked"
    assert readiness["operator_next_step"] == "fix binding validation errors"
    for target in readiness["targets"]:
        assert target["status"] == "blocked"
        assert "layer missing" in target["blocked_reasons"]
    assert package["overall_status"] == "blocked"
    assert package["ready_targets"] == []
    assert package["blocked_targets"] == ["docker", "wasm", "hardware"]
    assert (
        "binding validation must pass before deploy manifests are enabled"
        in (package["blocked_reasons"])
    )


def test_operator_checklist_guides_ready_and_postponed_targets() -> None:
    state = binding_spec_project_state(
        project_name="minimal_domain",
        spec_path=_minimal_spec_path(),
        knobs=StudioKnobState(K=1.0),
        runtime=build_runtime_snapshot(
            final_state={"R_global": 0.72, "regime": "nominal"},
            knobs=StudioKnobState(K=1.0),
            replay_status="completed",
        ),
    )

    checklist = build_operator_checklist(state)

    assert [step["step"] for step in checklist] == [1, 2, 3, 4, 5]
    assert checklist[0]["status"] == "complete"
    assert checklist[0]["title"] == "Run local replay"
    assert checklist[1]["status"] == "complete"
    assert checklist[2]["status"] == "ready"
    assert checklist[2]["target"] == "docker"
    assert checklist[3]["target"] == "wasm"
    assert checklist[4]["status"] == "postponed"
    assert checklist[4]["target"] == "hardware"


def test_operator_checklist_blocks_after_validation_failure() -> None:
    manifests = build_export_manifests(
        project_name="broken",
        binding_yaml="version: 1\n",
        audit_payload={"project_name": "broken"},
        validation_errors=("layer missing",),
    )
    state = binding_spec_project_state(
        project_name="minimal_domain",
        spec_path=_minimal_spec_path(),
        knobs=StudioKnobState(K=1.0),
        runtime=build_runtime_snapshot(
            final_state={"R_global": 0.72, "regime": "nominal"},
            knobs=StudioKnobState(K=1.0),
            replay_status="completed",
        ),
    )
    broken_state = type(state)(
        project_name=state.project_name,
        source=state.source,
        binding=state.binding,
        runtime=state.runtime,
        exports=manifests,
        metadata=state.metadata,
    )

    checklist = build_operator_checklist(broken_state)

    assert checklist[0]["status"] == "complete"
    assert checklist[1]["status"] == "blocked"
    assert checklist[1]["title"] == "Validate binding"
    assert checklist[2]["status"] == "blocked"
    assert "layer missing" in checklist[2]["detail"]


def test_command_table_exposes_review_commands_only() -> None:
    state = binding_spec_project_state(
        project_name="minimal_domain",
        spec_path=_minimal_spec_path(),
        knobs=StudioKnobState(K=1.0),
        runtime=build_runtime_snapshot(
            final_state={"R_global": 0.72, "regime": "nominal"},
            knobs=StudioKnobState(K=1.0),
            replay_status="completed",
        ),
    )

    rows = build_command_table(state)

    assert [row["target"] for row in rows] == ["docker", "docker", "docker", "wasm"]
    assert rows[0] == {
        "target": "docker",
        "command_index": 1,
        "command": "docker compose config",
        "status": "ready",
    }
    assert rows[-1]["command"].startswith("cd spo-kernel && wasm-pack build")
    assert all(row["target"] != "hardware" for row in rows)


def test_package_materialisation_plan_orders_ready_target_commands() -> None:
    state = binding_spec_project_state(
        project_name="minimal_domain",
        spec_path=_minimal_spec_path(),
        knobs=StudioKnobState(K=1.0),
        runtime=build_runtime_snapshot(
            final_state={"R_global": 0.72, "regime": "nominal"},
            knobs=StudioKnobState(K=1.0),
            replay_status="completed",
        ),
    )

    plan = build_package_materialisation_plan(state)

    assert plan["plan_kind"] == "studio_package_materialisation_plan"
    assert plan["project_name"] == "minimal_domain"
    assert plan["overall_status"] == "review_ready"
    assert plan["execution_mode"] == "operator_invoked"
    assert plan["network_opened"] is False
    assert plan["hardware_write_permitted"] is False
    assert plan["commands"][0] == {
        "step": 1,
        "target": "docker",
        "command": "docker compose config",
        "status": "ready",
        "requires_operator": True,
        "writes_artifact": False,
    }
    assert plan["commands"][-1]["target"] == "wasm"
    assert plan["commands"][-1]["writes_artifact"] is True
    assert plan["postponed_targets"] == [
        {
            "target": "hardware",
            "reason": "attach verified hardware-target evidence",
        }
    ]


def test_package_materialisation_plan_blocks_commands_after_validation_failure() -> (
    None
):
    manifests = build_export_manifests(
        project_name="broken",
        binding_yaml="version: 1\n",
        audit_payload={"project_name": "broken"},
        validation_errors=("layer missing",),
    )
    state = binding_spec_project_state(
        project_name="minimal_domain",
        spec_path=_minimal_spec_path(),
        knobs=StudioKnobState(K=1.0),
        runtime=build_runtime_snapshot(
            final_state={"R_global": 0.72, "regime": "nominal"},
            knobs=StudioKnobState(K=1.0),
            replay_status="completed",
        ),
    )
    broken_state = type(state)(
        project_name=state.project_name,
        source=state.source,
        binding=state.binding,
        runtime=state.runtime,
        exports=manifests,
        metadata=state.metadata,
    )

    plan = build_package_materialisation_plan(broken_state)

    assert plan["overall_status"] == "blocked"
    assert plan["commands"] == []
    assert plan["blocked_targets"] == ["docker", "wasm", "hardware"]
    assert "layer missing" in plan["blocked_reasons"]


def test_command_table_is_empty_when_validation_blocks_targets() -> None:
    manifests = build_export_manifests(
        project_name="broken",
        binding_yaml="version: 1\n",
        audit_payload={"project_name": "broken"},
        validation_errors=("layer missing",),
    )
    state = binding_spec_project_state(
        project_name="minimal_domain",
        spec_path=_minimal_spec_path(),
        knobs=StudioKnobState(K=1.0),
        runtime=build_runtime_snapshot(
            final_state={"R_global": 0.72, "regime": "nominal"},
            knobs=StudioKnobState(K=1.0),
            replay_status="completed",
        ),
    )
    broken_state = type(state)(
        project_name=state.project_name,
        source=state.source,
        binding=state.binding,
        runtime=state.runtime,
        exports=manifests,
        metadata=state.metadata,
    )

    assert build_command_table(broken_state) == ()


def test_error_report_scrubs_exception_text() -> None:
    report = build_error_report(
        operation="run_replay",
        error=ValueError("bad file at /tmp/private/domainpacks/demo.yaml"),
        project_name="minimal_domain",
    )

    assert report["project_name"] == "minimal_domain"
    assert report["operation"] == "run_replay"
    assert report["status"] == "blocked"
    assert report["error_type"] == "ValueError"
    assert report["operator_action"] == "review input artefacts and rerun"
    assert "private" not in json.dumps(report)
    assert "demo.yaml" not in json.dumps(report)


def test_error_report_rejects_empty_operation() -> None:
    with pytest.raises(ValueError, match="operation"):
        build_error_report(operation="", error=RuntimeError("boom"))


def test_deploy_exports_are_disabled_when_validation_fails() -> None:
    errors = ("layer missing", "bad channel")
    reasons = disabled_export_reasons(errors)
    manifests = build_export_manifests(
        project_name="broken",
        binding_yaml="version: 1\n",
        audit_payload={"project_name": "broken"},
        validation_errors=errors,
    )

    assert reasons == (
        "binding validation must pass before deploy manifests are enabled",
        "layer missing",
        "bad channel",
    )
    assert all(manifest.warnings == reasons for manifest in manifests)
    for manifest in manifests:
        payload = (
            json.loads(manifest.payload) if manifest.file_name.endswith(".json") else {}
        )
        if payload:
            assert payload["enabled"] is False
            assert payload["disabled_reasons"] == list(reasons)
    review_targets = {"binding_spec", "audit_summary"}
    deploy_targets = {"docker_manifest", "wasm_manifest"}
    assert {
        manifest.target_kind
        for manifest in manifests
        if manifest.target_kind in review_targets
    } == review_targets
    assert {
        manifest.target_kind
        for manifest in manifests
        if manifest.target_kind in deploy_targets
    } == deploy_targets


def test_oscillator_edit_artifact_records_reviewable_changes() -> None:
    before = (
        {
            "layer": "layer-a",
            "layer_index": 0,
            "oscillator_id": "osc-a",
            "family": "phase",
            "channel": "P",
        },
    )
    after = (
        {
            "layer": "layer-a",
            "layer_index": 0,
            "oscillator_id": "osc-renamed",
            "family": "phase",
            "channel": "P",
        },
        {
            "layer": "layer-b",
            "layer_index": 1,
            "oscillator_id": "osc-b",
            "family": "event",
            "channel": "I",
        },
    )

    artifact = build_oscillator_edit_artifact(before, after)
    record = json.loads(artifact.payload)

    assert artifact.target_kind == "oscillator_edit_review"
    assert artifact.safety_posture == "review_artifact"
    assert record["changed"] is True
    assert record["row_count_before"] == 1
    assert record["row_count_after"] == 2
    assert record["rows_after"][0]["oscillator_id"] == "osc-renamed"


def test_oscillator_edit_artifact_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        build_oscillator_edit_artifact(
            (),
            (
                {
                    "layer": "layer-a",
                    "layer_index": float("nan"),
                    "oscillator_id": "osc-a",
                },
            ),
        )


def test_replay_helper_runs_without_streamlit_dependency() -> None:
    import sys

    result = run_binding_spec_replay(
        _minimal_spec_path(),
        steps=3,
        knobs=StudioKnobState(K=1.0),
    )

    assert result.project_state.runtime.replay_status == "completed"
    assert len(result.r_history) == 3
    assert result.export_manifests
    assert "streamlit" not in sys.modules
