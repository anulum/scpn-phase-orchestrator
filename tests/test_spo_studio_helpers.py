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
from scpn_phase_orchestrator.studio.ui_helpers import (
    StudioKnobState,
    apply_knob_update,
    binding_spec_project_state,
    build_command_table,
    build_deployment_readiness,
    build_export_manifests,
    build_layer_table,
    build_operator_checklist,
    build_oscillator_edit_artifact,
    build_oscillator_table,
    build_regime_chart_payload,
    build_runtime_snapshot,
    build_series_chart_payload,
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

    assert oscillators
    assert layers
    assert {"layer", "oscillator_id", "channel", "family"} <= set(oscillators[0])
    assert {"index", "name", "oscillator_count", "family"} <= set(layers[0])


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

    assert readiness["overall_status"] == "blocked"
    assert readiness["operator_next_step"] == "fix binding validation errors"
    for target in readiness["targets"]:
        assert target["status"] == "blocked"
        assert "layer missing" in target["blocked_reasons"]


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
