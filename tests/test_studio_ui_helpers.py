# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Studio UI helper edge-case tests

from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from math import log
from pathlib import Path

import numpy as np
import pytest

import scpn_phase_orchestrator.studio.ui_helpers as ui
from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.studio.workflow import BindingProposal
from scpn_phase_orchestrator.supervisor import (
    MorphogeneticFieldState,
    evaluate_strange_loop_drift_scenarios,
    render_morphogenetic_field_svg,
)
from scpn_phase_orchestrator.supervisor.multiverse import (
    simulate_multiverse_counterfactual_branches,
)
from scpn_phase_orchestrator.supervisor.multiverse_risk import (
    MultiverseRiskThresholds,
    evaluate_multiverse_branch_risk,
)

MINIMAL_SPEC = Path("domainpacks/minimal_domain/binding_spec.yaml")
DIGITAL_TWIN_SPEC = Path("domainpacks/digital_twin_nchannel/binding_spec.yaml")


def _minimal_result() -> ui.StudioReplayResult:
    return ui.run_binding_spec_replay(
        MINIMAL_SPEC,
        steps=2,
        knobs=ui.StudioKnobState(K=1.0),
    )


def _digital_twin_result() -> ui.StudioReplayResult:
    return ui.run_binding_spec_replay(
        DIGITAL_TWIN_SPEC,
        steps=2,
        knobs=ui.StudioKnobState(K=1.0),
    )


def test_validate_candidate_binding_yaml_maps_expected_load_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_value_error(_path: Path) -> object:
        raise ValueError("invalid yaml")

    monkeypatch.setattr(ui, "load_binding_spec", _raise_value_error)
    errors = ui._validate_candidate_binding_yaml("name: candidate\n")
    assert errors == ["candidate binding failed to load: ValueError"]


def test_validate_candidate_binding_yaml_propagates_unexpected_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_runtime_error(_path: Path) -> object:
        raise RuntimeError("unexpected failure domain")

    monkeypatch.setattr(ui, "load_binding_spec", _raise_runtime_error)
    with pytest.raises(RuntimeError, match="unexpected failure domain"):
        ui._validate_candidate_binding_yaml("name: candidate\n")


def test_replay_result_audit_record_and_domainpack_discovery_are_stable(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing-domainpacks"
    domainpacks = tmp_path / "domainpacks"
    (domainpacks / "zeta").mkdir(parents=True)
    (domainpacks / "alpha").mkdir()
    (domainpacks / "alpha" / "binding_spec.yaml").write_text(
        "name: alpha\n",
        encoding="utf-8",
    )
    (domainpacks / "zeta" / "binding_spec.yaml").write_text(
        "name: zeta\n",
        encoding="utf-8",
    )
    (domainpacks / "not_a_pack.txt").write_text("ignored", encoding="utf-8")
    result = _minimal_result()

    record = result.to_audit_record()

    assert ui.discover_domainpacks(missing) == ()
    assert ui.discover_domainpacks(domainpacks) == ("alpha", "zeta")
    assert record["project"]["project_name"] == "minimal_domain"
    assert record["r_history"] == list(result.r_history)
    assert record["regime_history"] == list(result.regime_history)
    assert record["canvas_graph"]["canvas_kind"] == "layer_coupling_graph"
    assert record["connector_plan"]["plan_kind"] == "studio_live_connector_plan"
    assert [export["file_name"] for export in record["exports"]] == [
        "binding_spec.yaml",
        "spo_studio_audit.json",
        "docker_manifest.json",
        "wasm_manifest.json",
    ]


def test_chart_table_and_canvas_builders_describe_real_binding_spec() -> None:
    spec = ui.load_binding_spec(MINIMAL_SPEC)
    knobs = ui.apply_knob_update(
        ui.StudioKnobState(),
        K=1.5,
        alpha=0.2,
        zeta=0.4,
        Psi=3.0,
    )
    series = ui.build_series_chart_payload("R", [0.1, 0.5, 0.9])
    regimes = ui.build_regime_chart_payload(
        ["critical", "degraded", "recovery", "nominal", "custom"]
    )
    layers = ui.build_layer_table(spec)
    oscillators = ui.build_oscillator_table(spec)
    graph = ui.build_canvas_graph(spec)

    assert knobs.to_audit_record() == {
        "K": 1.5,
        "alpha": 0.2,
        "zeta": 0.4,
        "Psi": 3.0,
    }
    assert series == [
        {"step": 1, "R": 0.1},
        {"step": 2, "R": 0.5},
        {"step": 3, "R": 0.9},
    ]
    assert [row["regime_level"] for row in regimes] == [0.0, 1.0, 1.5, 2.0, 0.0]
    assert [row["name"] for row in layers] == ["lower", "upper"]
    assert [row["oscillator_id"] for row in oscillators] == [
        "osc_0",
        "osc_1",
        "osc_2",
        "osc_3",
    ]
    assert graph["layer_count"] == 2
    assert graph["channel_count"] == 3
    assert graph["edge_count"] == 0


def test_integrated_information_panel_preserves_operator_claim_boundary() -> None:
    n_bins = 8
    records = [
        {
            "monitor": "integrated_information",
            "phi": 0.12,
            "normalised_phi": 0.12 / log(n_bins),
            "total_integration": 0.24,
            "minimum_partition": [[0, 1], [2, 3]],
            "pairwise_mi": [
                [0.5, 0.2, 0.12, 0.12],
                [0.2, 0.5, 0.12, 0.12],
                [0.12, 0.12, 0.5, 0.2],
                [0.12, 0.12, 0.2, 0.5],
            ],
            "n_bins": n_bins,
            "claim_boundary": "engineering_proxy_not_theoretical_iit",
        },
        {
            "monitor": "integrated_information",
            "phi": 0.16,
            "normalised_phi": 0.16 / log(n_bins),
            "total_integration": 0.31,
            "minimum_partition": [[0, 2], [1, 3]],
            "pairwise_mi": [
                [0.6, 0.16, 0.18, 0.16],
                [0.16, 0.6, 0.16, 0.18],
                [0.18, 0.16, 0.6, 0.16],
                [0.16, 0.18, 0.16, 0.6],
            ],
            "n_bins": n_bins,
            "claim_boundary": "engineering_proxy_not_theoretical_iit",
        },
    ]

    panel = ui.build_integrated_information_panel(records)

    assert panel["panel_kind"] == "studio_integrated_information_panel"
    assert panel["claim_boundary"] == "engineering_proxy_not_theoretical_iit"
    assert panel["consciousness_claim_permitted"] is False
    assert panel["actuation_permitted"] is False
    assert panel["record_count"] == 2
    assert panel["latest"]["minimum_partition"] == [[0, 2], [1, 3]]
    assert panel["strongest_partition"]["phi"] == 0.16
    assert panel["phi_range"] == {"min": 0.12, "max": 0.16}


def test_integrated_information_panel_rejects_malformed_monitor_records() -> None:
    n_bins = 8
    valid_record = {
        "monitor": "integrated_information",
        "phi": 0.12,
        "normalised_phi": 0.12 / log(n_bins),
        "total_integration": 0.24,
        "minimum_partition": [[0], [1]],
        "pairwise_mi": [[0.5, 0.12], [0.12, 0.5]],
        "n_bins": n_bins,
        "claim_boundary": "engineering_proxy_not_theoretical_iit",
    }

    with pytest.raises(ValueError, match="claim boundary"):
        ui.build_integrated_information_panel(
            [{**valid_record, "claim_boundary": "consciousness_claim"}]
        )
    with pytest.raises(ValueError, match="normalised_phi"):
        ui.build_integrated_information_panel([{**valid_record, "normalised_phi": 0.9}])
    with pytest.raises(ValueError, match="indices must be integers"):
        ui.build_integrated_information_panel(
            [{**valid_record, "minimum_partition": [[False], [1]]}]
        )
    with pytest.raises(ValueError, match="symmetric"):
        ui.build_integrated_information_panel(
            [{**valid_record, "pairwise_mi": [[0.5, 0.11], [0.12, 0.5]]}]
        )


def test_strange_loop_studio_panel_preserves_non_actuating_scenario_evidence() -> None:
    records = [
        result.to_audit_record() for result in evaluate_strange_loop_drift_scenarios()
    ]

    panel = ui.build_strange_loop_studio_panel(records)

    assert panel["panel_kind"] == "studio_strange_loop_panel"
    assert panel["claim_boundary"] == "strange_loop_drift_review_not_live_actuation"
    assert panel["non_actuating"] is True
    assert panel["execution_disabled"] is True
    assert panel["actuation_permitted"] is False
    assert panel["scenario_count"] == len(records)
    assert panel["passed_count"] == len(records)
    assert panel["failed_scenario_ids"] == []
    assert set(panel["triggered_modes"]) == {
        "stable",
        "policy_drift",
        "control_loop_oscillation",
        "over_control",
    }
    assert panel["maxima"]["drift_score"] == pytest.approx(
        max(record["max_drift_score"] for record in records)
    )
    assert panel["maxima"]["overcontrol_score"] == pytest.approx(
        max(record["max_overcontrol_score"] for record in records)
    )


def test_strange_loop_studio_panel_rejects_malformed_scenario_evidence() -> None:
    record = evaluate_strange_loop_drift_scenarios()[0].to_audit_record()

    with pytest.raises(ValueError, match="execution_disabled"):
        ui.build_strange_loop_studio_panel([{**record, "execution_disabled": False}])
    with pytest.raises(ValueError, match="expected_trigger"):
        ui.build_strange_loop_studio_panel([{**record, "expected_trigger": "unknown"}])
    with pytest.raises(ValueError, match="scenario_hash"):
        ui.build_strange_loop_studio_panel([{**record, "scenario_hash": "bad"}])
    with pytest.raises(ValueError, match="finite non-negative real"):
        ui.build_strange_loop_studio_panel([{**record, "max_drift_score": True}])


def test_morphogenetic_field_panel_preserves_svg_snapshot_evidence() -> None:
    field = np.array(
        [
            [0.0, 0.9, 0.2],
            [0.4, 0.0, 0.7],
            [0.1, 0.3, 0.0],
        ],
        dtype=np.float64,
    )
    artefact = render_morphogenetic_field_svg(
        MorphogeneticFieldState(field),
        top_k=3,
        cell_size=16,
        title="Studio field review",
    ).to_audit_record()

    panel = ui.build_morphogenetic_field_studio_panel(artefact)

    assert panel["panel_kind"] == "studio_morphogenetic_field_panel"
    assert panel["renderer"] == "morphogenetic_field_svg"
    assert panel["actuation_permitted"] is False
    assert panel["top_edge_count"] == 3
    assert panel["shape"] == [3, 3]
    assert panel["snapshot"]["maximum"] == pytest.approx(0.9)
    assert panel["strongest_edge"] == {
        "source": 0,
        "target": 1,
        "weight": pytest.approx(0.9),
    }
    assert panel["field_energy"]["l2_norm"] == pytest.approx(
        float(np.linalg.norm(field))
    )
    assert panel["svg"].startswith("<svg ")


def test_morphogenetic_field_panel_rejects_malformed_svg_evidence() -> None:
    field = np.array([[0.0, 0.8], [0.2, 0.0]], dtype=np.float64)
    artefact = render_morphogenetic_field_svg(
        MorphogeneticFieldState(field),
        top_k=2,
    ).to_audit_record()
    snapshot = dict(artefact["snapshot"])

    with pytest.raises(ValueError, match="format"):
        ui.build_morphogenetic_field_studio_panel({**artefact, "format": "png"})
    with pytest.raises(ValueError, match="svg"):
        ui.build_morphogenetic_field_studio_panel({**artefact, "svg": "<script />"})
    with pytest.raises(ValueError, match="shape"):
        ui.build_morphogenetic_field_studio_panel(
            {**artefact, "snapshot": {**snapshot, "shape": [2, True]}}
        )
    with pytest.raises(ValueError, match="top_edges"):
        ui.build_morphogenetic_field_studio_panel(
            {
                **artefact,
                "snapshot": {
                    **snapshot,
                    "top_edges": [{"source": 0, "target": 0, "weight": 0.5}],
                },
            }
        )


def test_multiverse_counterfactual_panel_preserves_manifest_and_risk_evidence() -> None:
    phases = np.array([0.10, 1.20, 2.40], dtype=np.float64)
    omegas = np.array([0.05, -0.02, 0.01], dtype=np.float64)
    baseline_k = np.array(
        [[0.0, 0.15, 0.15], [0.15, 0.0, 0.15], [0.15, 0.15, 0.0]],
        dtype=np.float64,
    )
    baseline_alpha = np.zeros((3, 3), dtype=np.float64)
    manifest = simulate_multiverse_counterfactual_branches(
        phases=phases,
        omegas=omegas,
        baseline_k=baseline_k,
        baseline_alpha=baseline_alpha,
        branch_action_sets=(
            (),
            (ControlAction("K", "global", 0.25, 1.0, "coupling review"),),
        ),
        horizon=8,
        dt=0.02,
    ).to_audit_record()
    risk = evaluate_multiverse_branch_risk(
        manifest,
        MultiverseRiskThresholds(
            min_mean_R=0.0,
            min_final_R=0.0,
            max_action_count=4,
        ),
    ).to_audit_record()

    panel = ui.build_multiverse_counterfactual_studio_panel(manifest, risk)

    assert panel["panel_kind"] == "studio_multiverse_counterfactual_panel"
    assert panel["claim_boundary"] == "counterfactual_branch_rollout_not_live_actuation"
    assert panel["risk_claim_boundary"] == (
        "counterfactual_branch_risk_gate_not_live_actuation"
    )
    assert panel["non_actuating"] is True
    assert panel["execution_disabled"] is True
    assert panel["actuation_permitted"] is False
    assert panel["branch_count"] == 2
    assert panel["approved_count"] == 2
    assert panel["rejected_count"] == 0
    assert panel["safest_branch_id"] in {"branch_000", "branch_001"}
    assert panel["coherence_range"]["minimum"] <= panel["coherence_range"]["maximum"]
    assert panel["branch_rows"][1]["action_count"] == 1
    assert panel["branch_rows"][1]["risk_approved"] is True
    assert "actions_to_apply" not in panel
    assert "control_actions" not in panel


def test_hybrid_order_studio_panel_preserves_entanglement_review_evidence() -> None:
    from scpn_phase_orchestrator.monitor.hybrid_order import (
        compute_hybrid_entanglement_order_parameter,
    )
    from scpn_phase_orchestrator.monitor.hybrid_order_examples import (
        build_hybrid_order_parameter_scenarios,
    )

    phases = np.array([0.0, 0.4, 0.9, 1.7], dtype=np.float64)
    product_state = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    bell_state = np.array(
        [1 / np.sqrt(2), 0.0, 0.0, 1 / np.sqrt(2)],
        dtype=np.complex128,
    )
    records = [
        compute_hybrid_entanglement_order_parameter(
            phases=phases,
            quantum_state=product_state,
            bipartition=((0,), (1,)),
            simulator_backend="numpy_statevector",
        ).to_audit_record(),
        compute_hybrid_entanglement_order_parameter(
            phases=phases,
            quantum_state=np.outer(bell_state, np.conj(bell_state)),
            bipartition=((0,), (1,)),
            simulator_backend="numpy_density_matrix",
        ).to_audit_record(),
    ]

    panel = ui.build_hybrid_order_studio_panel(
        records,
        scenarios=build_hybrid_order_parameter_scenarios(),
    )

    assert panel["panel_kind"] == "studio_hybrid_order_panel"
    assert panel["claim_boundary"] == "quantum_cosimulation_monitor_not_qpu_execution"
    assert panel["non_actuating"] is True
    assert panel["execution_disabled"] is True
    assert panel["actuation_permitted"] is False
    assert panel["qpu_execution_permitted"] is False
    assert panel["record_count"] == 2
    assert panel["scenario_count"] >= 3
    assert set(panel["simulator_backends"]) == {
        "numpy_density_matrix",
        "numpy_statevector",
    }
    assert panel["entropy_range"]["minimum"] < 0.05
    assert panel["entropy_range"]["maximum"] > 0.9
    assert panel["latest"]["backend"] == "numpy_density_matrix"
    assert panel["strongest_entanglement"]["entanglement_entropy"] == pytest.approx(
        panel["entropy_range"]["maximum"]
    )
    assert panel["candidate_rows"]
    assert {row["state_type"] for row in panel["candidate_rows"]} >= {
        "product",
        "entangled",
    }


def test_hybrid_order_studio_panel_rejects_malformed_review_evidence() -> None:
    from scpn_phase_orchestrator.monitor.hybrid_order import (
        compute_hybrid_entanglement_order_parameter,
    )

    record = compute_hybrid_entanglement_order_parameter(
        phases=np.array([0.0, 1.0], dtype=np.float64),
        quantum_state=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128),
        bipartition=((0,), (1,)),
    ).to_audit_record()

    with pytest.raises(ValueError, match="claim boundary"):
        ui.build_hybrid_order_studio_panel(
            [{**record, "claim_boundary": "qpu_execution_claim"}]
        )
    with pytest.raises(ValueError, match="normalised_entanglement_entropy"):
        ui.build_hybrid_order_studio_panel(
            [{**record, "normalised_entanglement_entropy": 1.4}]
        )
    with pytest.raises(ValueError, match="bipartition"):
        ui.build_hybrid_order_studio_panel([{**record, "bipartition": [[False], [1]]}])
    with pytest.raises(ValueError, match="record_hash"):
        ui.build_hybrid_order_studio_panel([{**record, "record_hash": "bad"}])


def test_multiverse_counterfactual_panel_rejects_malformed_evidence() -> None:
    manifest = {
        "schema_name": "multiverse_counterfactual_rollout",
        "schema_version": "0.1.0",
        "branch_count": 1,
        "horizon": 4,
        "backend": "numpy_vectorized",
        "non_actuating": True,
        "execution_disabled": True,
        "claim_boundary": "counterfactual_branch_rollout_not_live_actuation",
        "manifest_hash": "a" * 64,
        "branch_records": [
            {
                "branch_id": "safe",
                "branch_hash": "b" * 64,
                "action_count": 0,
                "action_labels": [],
                "topology_edge_count": 2,
                "topology_scale": 0.3,
                "final_R": 0.8,
                "mean_R": 0.7,
                "min_R": 0.6,
                "max_R": 0.9,
                "final_psi": 0.1,
            }
        ],
    }
    risk = {
        "schema_name": "multiverse_branch_risk_gate",
        "schema_version": "0.1.0",
        "branch_count": 1,
        "approved_count": 1,
        "rejected_count": 0,
        "safest_branch_id": "safe",
        "safest_branch_hash": "b" * 64,
        "rejection_reasons": [],
        "claim_boundary": "counterfactual_branch_risk_gate_not_live_actuation",
        "non_actuating": True,
        "execution_disabled": True,
        "report_hash": "c" * 64,
        "branch_decisions": [
            {
                "branch_id": "safe",
                "branch_hash": "b" * 64,
                "final_R": 0.8,
                "mean_R": 0.7,
                "min_R": 0.6,
                "max_R": 0.9,
                "action_count": 0,
                "topology_edge_count": 2,
                "topology_scale": 0.3,
                "approved": True,
                "rejection_reasons": [],
            }
        ],
    }

    with pytest.raises(ValueError, match="claim boundary"):
        ui.build_multiverse_counterfactual_studio_panel(
            {**manifest, "claim_boundary": "live_control"}, risk
        )
    with pytest.raises(ValueError, match="branch_count"):
        ui.build_multiverse_counterfactual_studio_panel(
            {**manifest, "branch_count": 2}, risk
        )
    with pytest.raises(ValueError, match="R interval"):
        bad_branch = dict(manifest["branch_records"][0])
        bad_branch["min_R"] = 0.95
        ui.build_multiverse_counterfactual_studio_panel(
            {**manifest, "branch_records": [bad_branch]}, risk
        )
    with pytest.raises(ValueError, match="risk decision"):
        ui.build_multiverse_counterfactual_studio_panel(
            manifest,
            {
                **risk,
                "branch_decisions": [
                    {**risk["branch_decisions"][0], "branch_id": "other"}
                ],
            },
        )


def test_canvas_review_artifacts_capture_layout_and_topology_changes() -> None:
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
        "edges": (),
    }
    after = {
        "nodes": (
            {
                "id": "layer_0",
                "label": "source-reviewed",
                "kind": "layer",
                "x": 12.5,
                "y": 8.0,
            },
            before["nodes"][1],
        ),
        "edges": (
            {
                "id": "edge_new",
                "source": "layer_0",
                "target": "layer_1",
                "kind": "review_edge",
            },
        ),
    }

    edit = json.loads(ui.build_canvas_edit_artifact(before, after).payload)
    layout = json.loads(
        ui.build_canvas_layout_manifest(
            project_name="minimal_domain",
            graph=after,
        ).payload
    )
    patch = json.loads(
        ui.build_canvas_topology_patch(
            project_name="minimal_domain",
            before_graph=before,
            after_graph=after,
        ).payload
    )

    assert edit["changed"] is True
    assert edit["node_count_after"] == 2
    assert layout["edge_count"] == 1
    assert layout["positions"][0]["id"] == "layer_0"
    assert patch["node_changes"]["modified"][0]["id"] == "layer_0"
    assert patch["edge_changes"]["added"][0]["id"] == "edge_new"


def test_canvas_interaction_state_requires_signoff_for_review_ready_changes() -> None:
    result = _digital_twin_result()
    after_graph = {
        "nodes": result.canvas_graph["nodes"],
        "edges": (
            {
                "id": "cross_channel_reviewed",
                "source": "channel_P",
                "target": "channel_Quality",
                "kind": "cross_channel_coupling",
                "source_channel": "P",
                "target_channel": "Quality",
                "strength": 0.07,
                "mode": "excitatory",
                "template": "operator_reviewed_quality_probe",
            },
        ),
    }
    artifact = ui.build_canvas_edit_artifact(result.canvas_graph, after_graph)
    layout = ui.build_canvas_layout_manifest(
        project_name=result.project_state.project_name,
        graph=after_graph,
    )
    patch = ui.build_canvas_topology_patch(
        project_name=result.project_state.project_name,
        before_graph=result.canvas_graph,
        after_graph=after_graph,
    )
    rewrite = ui.build_canvas_binding_rewrite_candidate(result, after_graph=after_graph)

    state = ui.build_canvas_interaction_state(
        canvas_artifact=artifact,
        canvas_layout=layout,
        canvas_patch=patch,
        canvas_rewrite=rewrite,
        operator_signoff=False,
    )

    assert state["rewrite_status"] == "review_ready"
    assert state["apply_enabled"] is False
    assert state["disabled_reasons"] == ["operator sign-off required"]
    assert state["next_action"] == "review artefacts and sign off before apply"


def test_canvas_interaction_state_allows_noop_review_without_signoff() -> None:
    result = _minimal_result()
    artifact = ui.build_canvas_edit_artifact(result.canvas_graph, result.canvas_graph)
    layout = ui.build_canvas_layout_manifest(
        project_name=result.project_state.project_name,
        graph=result.canvas_graph,
    )
    patch = ui.build_canvas_topology_patch(
        project_name=result.project_state.project_name,
        before_graph=result.canvas_graph,
        after_graph=result.canvas_graph,
    )
    rewrite = ui.build_canvas_binding_rewrite_candidate(
        result,
        after_graph=result.canvas_graph,
    )

    state = ui.build_canvas_interaction_state(
        canvas_artifact=artifact,
        canvas_layout=layout,
        canvas_patch=patch,
        canvas_rewrite=rewrite,
        operator_signoff=False,
    )

    assert state["changed"] is False
    assert state["apply_enabled"] is False
    assert state["next_action"] == "download artefacts or continue replay review"
    assert state["status_message"] == "Canvas graph matches the current binding."


def test_canvas_interaction_state_reports_blocked_candidate_reasons() -> None:
    result = _minimal_result()
    after_graph = {
        "nodes": result.canvas_graph["nodes"],
        "edges": (
            {
                "id": "layer_edge",
                "source": "layer_0",
                "target": "layer_1",
                "kind": "review_edge",
            },
        ),
    }
    state = ui.build_canvas_interaction_state(
        canvas_artifact=ui.build_canvas_edit_artifact(result.canvas_graph, after_graph),
        canvas_layout=ui.build_canvas_layout_manifest(
            project_name=result.project_state.project_name,
            graph=after_graph,
        ),
        canvas_patch=ui.build_canvas_topology_patch(
            project_name=result.project_state.project_name,
            before_graph=result.canvas_graph,
            after_graph=after_graph,
        ),
        canvas_rewrite=ui.build_canvas_binding_rewrite_candidate(
            result,
            after_graph=after_graph,
        ),
        operator_signoff=True,
    )

    assert state["apply_enabled"] is False
    assert state["disabled_reasons"] == [
        "binding rewrite candidate is blocked",
        "only cross_channel_coupling edges can rewrite binding YAML",
    ]
    assert state["next_action"] == "fix blocked canvas rewrite before apply"


def test_canvas_binding_rewrite_blocks_invalid_result_and_invalid_couplings() -> None:
    result = _digital_twin_result()

    with pytest.raises(ValueError, match="StudioReplayResult"):
        ui.build_canvas_binding_rewrite_candidate(
            object(),
            after_graph={"nodes": (), "edges": ()},
        )

    same_channel = ui.build_canvas_binding_rewrite_candidate(
        result,
        after_graph={
            "nodes": result.canvas_graph["nodes"],
            "edges": (
                {
                    "id": "self_coupling",
                    "source": "channel_P",
                    "target": "channel_P",
                    "kind": "cross_channel_coupling",
                    "strength": 0.5,
                },
            ),
        },
    )
    malformed_yaml_result = replace(
        result,
        project_state=replace(
            result.project_state,
            binding=BindingProposal(
                yaml_text="- not\n- a\n- mapping\n",
                inferred_channels=("P",),
                provenance=dict(result.project_state.binding.provenance),
            ),
        ),
    )
    malformed_yaml = ui.build_canvas_binding_rewrite_candidate(
        malformed_yaml_result,
        after_graph={"nodes": result.canvas_graph["nodes"], "edges": ()},
    )

    assert same_channel["status"] == "blocked"
    assert same_channel["validation_errors"] == [
        "cross-channel coupling source and target must differ"
    ]
    assert malformed_yaml["status"] == "blocked"
    assert malformed_yaml["validation_errors"] == [
        "binding YAML must contain a mapping"
    ]


def test_canvas_apply_blocks_bad_candidate_metadata_and_allocates_numbered_backup(
    tmp_path: Path,
) -> None:
    result = _digital_twin_result()
    candidate = ui.build_canvas_binding_rewrite_candidate(
        result,
        after_graph={
            "nodes": result.canvas_graph["nodes"],
            "edges": result.canvas_graph["edges"],
        },
    )
    target = tmp_path / "binding_spec.yaml"
    target.write_text(result.project_state.binding.yaml_text, encoding="utf-8")
    default_backup = target.with_name(
        f"{target.name}.studio-backup-{candidate['before_yaml_sha256'][:12]}.bak"
    )
    default_backup.write_text("existing backup", encoding="utf-8")

    blocked = ui.apply_canvas_binding_rewrite_candidate(
        {
            **candidate,
            "candidate_kind": "wrong",
            "status": "blocked",
            "candidate_yaml_sha256": "0" * 64,
        },
        binding_spec_path=tmp_path / "missing.yaml",
        operator_signoff=False,
    )
    applied = ui.apply_canvas_binding_rewrite_candidate(
        candidate,
        binding_spec_path=target,
        operator_signoff=True,
    )

    assert blocked["status"] == "blocked"
    assert blocked["blocked_reasons"] == [
        "candidate_kind must be canvas_binding_rewrite_candidate",
        "candidate status must be review_ready",
        "operator_signoff must be true",
        "binding_spec_path must point to an existing file",
        "candidate YAML SHA-256 does not match candidate metadata",
    ]
    assert applied["status"] == "applied"
    assert applied["backup_path"].endswith(".bak.1")
    assert Path(str(applied["backup_path"])).read_text(encoding="utf-8") == (
        result.project_state.binding.yaml_text
    )


def test_canvas_apply_detects_stale_source_and_uses_default_backup_path(
    tmp_path: Path,
) -> None:
    result = _digital_twin_result()
    candidate = ui.build_canvas_binding_rewrite_candidate(
        result,
        after_graph={
            "nodes": result.canvas_graph["nodes"],
            "edges": result.canvas_graph["edges"],
        },
    )
    stale_target = tmp_path / "stale" / "binding_spec.yaml"
    stale_target.parent.mkdir()
    stale_target.write_text(
        result.project_state.binding.yaml_text + "\n# local edit\n",
        encoding="utf-8",
    )
    fresh_target = tmp_path / "fresh" / "binding_spec.yaml"
    fresh_target.parent.mkdir()
    fresh_target.write_text(result.project_state.binding.yaml_text, encoding="utf-8")

    stale = ui.apply_canvas_binding_rewrite_candidate(
        candidate,
        binding_spec_path=stale_target,
        operator_signoff=True,
    )
    fresh = ui.apply_canvas_binding_rewrite_candidate(
        candidate,
        binding_spec_path=fresh_target,
        operator_signoff=True,
    )

    assert stale["status"] == "blocked"
    assert stale["blocked_reasons"] == [
        "current binding_spec.yaml SHA-256 does not match candidate source"
    ]
    assert fresh["status"] == "applied"
    assert fresh["backup_path"].endswith(".bak")
    assert Path(str(fresh["backup_path"])).exists()


def test_live_connector_run_record_accepts_replay_connector_payload() -> None:
    result = _minimal_result()

    record = ui.build_live_connector_run_record(
        result.connector_plan,
        transport="jsonl",
        payload={"sequence": 1, "kind": "replay_probe"},
        dry_run=True,
    )

    assert record["status"] == "accepted"
    assert record["operator_action"] == "review dry-run connector payload"
    assert len(record["payload_sha256"]) == 64
    assert record["network_opened"] is False
    assert record["actuation_permitted"] is False


def test_hardware_packages_require_and_accept_complete_evidence() -> None:
    result = _minimal_result()
    package = ui.build_hardware_target_package(result)
    incomplete = ui.build_verified_hardware_target_package(
        result,
        evidence={
            "generated_artifact_path": "build/hardware/minimal_domain/fpga_top.v",
            "generated_artifact_sha256": "not-a-sha",
            "simulator_parity_status": "failed",
            "operator_signoff": False,
        },
    )
    complete = ui.build_verified_hardware_target_package(
        result,
        evidence={
            "generated_artifact_path": "build/hardware/minimal_domain/fpga_top.v",
            "generated_artifact_sha256": "a" * 64,
            "simulator_parity_report": "reports/minimal_domain_parity.json",
            "simulator_parity_sha256": "b" * 64,
            "simulator_parity_status": "passed",
            "target_toolchain": "yosys-nextpnr",
            "target_toolchain_version": "yosys 0.40 / nextpnr 0.7",
            "operator_signoff": True,
        },
    )

    assert package["overall_status"] == "evidence_required"
    assert package["connector"]["transport"] == "hardware"
    assert incomplete["evidence_status"] == "blocked"
    assert "simulator_parity_report is required" in incomplete["invalid_evidence"]
    assert (
        "generated_artifact_sha256 must be a SHA-256 digest"
        in (incomplete["invalid_evidence"])
    )
    assert complete["overall_status"] == "review_ready"
    assert complete["evidence"]["simulator_parity_status"] == "passed"
    assert complete["commands"] == [
        "review verified_hardware_target_package.json",
        "compare generated artefact hash before handoff",
        "archive simulator parity report with package",
    ]


def test_beginner_guidance_and_checklist_report_ready_operator_flow() -> None:
    result = ui.run_binding_spec_replay(
        MINIMAL_SPEC,
        steps=2,
        knobs=ui.StudioKnobState(K=1.3, alpha=0.2, zeta=0.1, Psi=0.5),
    )
    guidance = ui.build_beginner_guidance(result)
    checklist = ui.build_operator_checklist(result.project_state)

    assert guidance["guide_kind"] == "beginner_mode"
    assert guidance["runtime_summary"]["replay_status"] == "completed"
    assert [card["title"] for card in guidance["concept_cards"]] == [
        "Signals",
        "Coupling",
        "Objectives",
        "Supervisor",
    ]
    assert guidance["concept_cards"][1]["evidence"]["K"] == 1.3
    assert [step["status"] for step in guidance["walkthrough_steps"]] == [
        "complete",
        "complete",
        "complete",
        "complete",
        "ready",
    ]
    assert [step["target"] for step in checklist[2:]] == ["docker", "wasm", "hardware"]
    assert checklist[4]["status"] == "postponed"


@pytest.mark.parametrize("transport", ["grpc", "kafka", "hardware"])
def test_owned_live_connector_runtime_accepts_each_live_adapter(
    transport: str,
) -> None:
    result = _digital_twin_result()

    record = ui.build_owned_live_connector_runtime_record(
        result,
        transport=transport,
        owner="plant-ops",
        auth_policy={"scheme": "bearer", "credential_label": "studio-local-key"},
        payload={"kind": "owned_runtime_probe", "R": 0.82},
        sequence=3,
    )

    assert record["transport"] == transport
    assert record["status"] == "accepted"
    assert record["response"]["accepted"] is True
    assert record["queued_count"] == 1
    assert record["network_opened"] is False
    assert record["actuation_permitted"] is False
    assert record["hardware_write_permitted"] is False


def test_owned_live_connector_runtime_reports_boundary_validation_errors() -> None:
    result = _digital_twin_result()
    incompatible_plan = {
        **dict(result.connector_plan),
        "connectors": [
            {
                **dict(connector),
                "compatible": False
                if connector.get("transport") == "rest"
                else connector.get("compatible"),
            }
            for connector in result.connector_plan["connectors"]
        ],
    }
    incompatible_result = replace(result, connector_plan=incompatible_plan)

    with pytest.raises(ValueError, match="StudioReplayResult"):
        ui.build_owned_live_connector_runtime_record(
            object(),
            transport="rest",
            owner="plant-ops",
            auth_policy={"scheme": "bearer", "credential_label": "studio-local-key"},
            payload={"kind": "probe"},
        )
    with pytest.raises(ValueError, match="payload contains an invalid key"):
        ui.build_owned_live_connector_runtime_record(
            result,
            transport="rest",
            owner="plant-ops",
            auth_policy={"scheme": "bearer", "credential_label": "studio-local-key"},
            payload={1: "bad"},
        )
    with pytest.raises(ValueError, match="payload contains a non-finite float"):
        ui.build_owned_live_connector_runtime_record(
            result,
            transport="rest",
            owner="plant-ops",
            auth_policy={"scheme": "bearer", "credential_label": "studio-local-key"},
            payload={"bad": float("nan")},
        )

    offline_transport = ui.build_owned_live_connector_runtime_record(
        result,
        transport="memory",
        owner="plant-ops",
        auth_policy={"scheme": "bearer", "credential_label": "studio-local-key"},
        payload={"nested": [{"ok": True}]},
    )
    missing_auth_mapping = ui.build_owned_live_connector_runtime_record(
        result,
        transport="rest",
        owner="plant-ops",
        auth_policy=object(),
        payload={"kind": "probe"},
    )
    incompatible = ui.build_owned_live_connector_runtime_record(
        incompatible_result,
        transport="rest",
        owner="plant-ops",
        auth_policy={"scheme": "bearer", "credential_label": "studio-local-key"},
        payload={"kind": "probe"},
    )
    missing_owner_and_auth = ui.build_owned_live_connector_runtime_record(
        result,
        transport="rest",
        owner="",
        auth_policy={"scheme": "", "credential_label": ""},
        payload={"kind": "probe"},
    )
    rest = ui.build_owned_live_connector_runtime_record(
        result,
        transport="rest",
        owner="plant-ops",
        auth_policy={"scheme": "bearer", "credential_label": "studio-local-key"},
        payload={"kind": "owned_runtime_probe", "nested": [{"ok": True}]},
        sequence=5,
    )

    assert offline_transport["blocked_reasons"] == [
        "owned runtime requires a live connector transport"
    ]
    assert missing_auth_mapping["blocked_reasons"] == ["auth_policy must be a mapping"]
    assert incompatible["blocked_reasons"] == ["connector manifest is incompatible"]
    assert missing_owner_and_auth["blocked_reasons"] == [
        "owner must be assigned",
        "auth_policy.scheme must be assigned",
        "auth_policy.credential_label must be assigned",
    ]
    assert rest["status"] == "accepted"
    assert rest["queued_count"] == 1


def test_owned_live_connector_runtime_requires_source_path_provenance() -> None:
    result = _digital_twin_result()
    result_without_source_path = replace(
        result,
        project_state=replace(
            result.project_state,
            binding=BindingProposal(
                yaml_text=result.project_state.binding.yaml_text,
                inferred_channels=tuple(result.project_state.binding.inferred_channels),
                provenance={},
            ),
        ),
    )

    with pytest.raises(ValueError, match="source_path"):
        ui.build_owned_live_connector_runtime_record(
            result_without_source_path,
            transport="rest",
            owner="plant-ops",
            auth_policy={"scheme": "bearer", "credential_label": "studio-local-key"},
            payload={"kind": "probe"},
        )


def test_live_connector_run_record_rejects_malformed_payload_and_connectors() -> None:
    result = _minimal_result()

    with pytest.raises(ValueError, match="payload must be a mapping"):
        ui.build_live_connector_run_record(
            result.connector_plan,
            transport="jsonl",
            payload=object(),
        )
    with pytest.raises(ValueError, match="connectors must be a sequence"):
        ui.build_live_connector_run_record(
            {**dict(result.connector_plan), "connectors": "bad"},
            transport="jsonl",
            payload={"kind": "probe"},
        )
    with pytest.raises(ValueError, match="connector entries must be mappings"):
        ui.build_live_connector_run_record(
            {**dict(result.connector_plan), "connectors": ("bad",)},
            transport="jsonl",
            payload={"kind": "probe"},
        )
    with pytest.raises(ValueError, match="connector transport 'missing' not found"):
        ui.build_live_connector_run_record(
            result.connector_plan,
            transport="missing",
            payload={"kind": "probe"},
        )


def test_hardware_package_builders_reject_invalid_inputs() -> None:
    result = _minimal_result()

    with pytest.raises(ValueError, match="StudioReplayResult"):
        ui.build_hardware_target_package(object())
    with pytest.raises(ValueError, match="StudioReplayResult"):
        ui.build_verified_hardware_target_package(object(), evidence={})
    with pytest.raises(ValueError, match="hardware evidence must be a mapping"):
        ui.build_verified_hardware_target_package(result, evidence=object())


def test_runtime_snapshot_and_replay_input_guards_are_operator_safe() -> None:
    with pytest.raises(ValueError, match="steps"):
        ui.run_binding_spec_replay(MINIMAL_SPEC, steps=True, knobs=ui.StudioKnobState())
    with pytest.raises(ValueError, match="layer.R"):
        ui.build_runtime_snapshot(
            final_state={"R_global": 0.5, "regime": "nominal", "layers": [{"R": True}]},
            knobs=ui.StudioKnobState(),
        )
    snapshot = ui.build_runtime_snapshot(
        final_state={"R_global": 0.5, "regime": "nominal", "layers": object()},
        knobs=ui.StudioKnobState(),
    )
    with pytest.raises(ValueError, match="R_global"):
        ui.build_runtime_snapshot(
            final_state={"R_global": object(), "regime": "nominal"},
            knobs=ui.StudioKnobState(),
        )
    assert snapshot.layer_metrics == ()


def test_table_and_canvas_normalisers_reject_non_json_safe_rows() -> None:
    with pytest.raises(ValueError, match="non-string key"):
        ui.build_oscillator_edit_artifact(
            (),
            ({1: "bad", "id": "row"},),
        )
    with pytest.raises(ValueError, match="JSON-safe"):
        ui.build_oscillator_edit_artifact(
            (),
            ({"id": "row", "payload": object()},),
        )
    with pytest.raises(ValueError, match="canvas edges must be a sequence"):
        ui.build_canvas_edit_artifact(
            {"nodes": (), "edges": "bad"},
            {"nodes": (), "edges": ()},
        )
    with pytest.raises(ValueError, match="before_graph must be a mapping"):
        ui.build_canvas_edit_artifact(
            object(),
            {"nodes": (), "edges": ()},
        )
    with pytest.raises(ValueError, match="canvas nodes must be a sequence"):
        ui.build_canvas_edit_artifact(
            {"nodes": "bad", "edges": ()},
            {"nodes": (), "edges": ()},
        )
    with pytest.raises(ValueError, match="canvas item id"):
        ui.build_canvas_topology_patch(
            project_name="minimal_domain",
            before_graph={
                "nodes": (
                    {"id": "duplicate", "kind": "layer"},
                    {"id": "duplicate", "kind": "layer"},
                ),
                "edges": (),
            },
            after_graph={"nodes": (), "edges": ()},
        )
    with pytest.raises(ValueError, match="unknown endpoint"):
        ui.build_canvas_topology_patch(
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


def test_value_guards_reject_invalid_canvas_coupling_and_chart_inputs() -> None:
    result = _digital_twin_result()

    with pytest.raises(ValueError, match="label"):
        ui.build_series_chart_payload("", [0.1])
    with pytest.raises(ValueError, match="R"):
        ui.build_series_chart_payload("R", [float("inf")])
    with pytest.raises(ValueError, match="regime"):
        ui.build_regime_chart_payload([""])

    bad_endpoint = ui.build_canvas_binding_rewrite_candidate(
        result,
        after_graph={
            "nodes": result.canvas_graph["nodes"],
            "edges": (
                {
                    "id": "bad_endpoint",
                    "source": "layer_0",
                    "target": "channel_P",
                    "kind": "cross_channel_coupling",
                    "strength": 0.5,
                },
            ),
        },
    )
    bad_strength = ui.build_canvas_binding_rewrite_candidate(
        result,
        after_graph={
            "nodes": result.canvas_graph["nodes"],
            "edges": (
                {
                    "id": "bad_strength",
                    "source": "channel_Thermal",
                    "target": "channel_P",
                    "kind": "cross_channel_coupling",
                    "strength": 101.0,
                },
            ),
        },
    )

    assert bad_endpoint["validation_errors"] == ["source must reference a channel node"]
    assert bad_strength["validation_errors"] == [
        "cross-channel coupling strength must be in [0.0, 100.0]"
    ]


def test_builders_validate_malformed_readiness_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _minimal_result().project_state

    monkeypatch.setattr(
        ui,
        "build_deployment_readiness",
        lambda _state: {"overall_status": "review_ready", "targets": "bad"},
    )
    with pytest.raises(ValueError, match="readiness targets must be a sequence"):
        ui.build_deployment_package(state)

    monkeypatch.setattr(
        ui,
        "build_deployment_readiness",
        lambda _state: {"overall_status": "review_ready", "targets": ("bad",)},
    )
    with pytest.raises(ValueError, match=r"readiness targets\[0\] must be a mapping"):
        ui.build_deployment_package(state)

    monkeypatch.setattr(
        ui,
        "build_deployment_readiness",
        lambda _state: {
            "overall_status": "review_ready",
            "targets": (
                {
                    "target": "docker",
                    "status": "ready",
                    "required_artifacts": "bad",
                },
            ),
        },
    )
    with pytest.raises(ValueError, match="required_artifacts must be a sequence"):
        ui.build_deployment_package(state)

    monkeypatch.setattr(
        ui,
        "build_deployment_readiness",
        lambda _state: {
            "overall_status": "review_ready",
            "targets": (
                {
                    "target": "docker",
                    "status": "ready",
                    "required_artifacts": (),
                    "commands": "bad",
                },
            ),
        },
    )
    with pytest.raises(ValueError, match="target commands must be a sequence"):
        ui.build_command_table(state)

    monkeypatch.setattr(
        ui,
        "build_deployment_readiness",
        lambda _state: {"overall_status": "review_ready", "targets": ("bad",)},
    )
    with pytest.raises(ValueError, match="readiness targets must be mappings"):
        ui.build_operator_checklist(state)


def test_canvas_rewrite_and_apply_validate_required_payload_metadata(
    tmp_path: Path,
) -> None:
    result = _minimal_result()
    yaml_text = result.project_state.binding.yaml_text
    digest = sha256(yaml_text.encode("utf-8")).hexdigest()
    target = tmp_path / "binding_spec.yaml"
    target.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(ValueError, match="candidate_yaml"):
        ui.apply_canvas_binding_rewrite_candidate(
            {
                "candidate_kind": "canvas_binding_rewrite_candidate",
                "status": "review_ready",
                "candidate_yaml": "",
                "before_yaml_sha256": digest,
                "candidate_yaml_sha256": digest,
            },
            binding_spec_path=target,
            operator_signoff=True,
        )
    with pytest.raises(ValueError, match="before_yaml_sha256"):
        ui.apply_canvas_binding_rewrite_candidate(
            {
                "candidate_kind": "canvas_binding_rewrite_candidate",
                "status": "review_ready",
                "candidate_yaml": yaml_text,
                "before_yaml_sha256": "not-a-sha",
                "candidate_yaml_sha256": digest,
            },
            binding_spec_path=target,
            operator_signoff=True,
        )
    with pytest.raises(ValueError, match="validation_errors"):
        ui.build_canvas_interaction_state(
            canvas_artifact=ui.build_canvas_edit_artifact(
                result.canvas_graph,
                result.canvas_graph,
            ),
            canvas_layout=ui.build_canvas_layout_manifest(
                project_name=result.project_state.project_name,
                graph=result.canvas_graph,
            ),
            canvas_patch=ui.build_canvas_topology_patch(
                project_name=result.project_state.project_name,
                before_graph=result.canvas_graph,
                after_graph=result.canvas_graph,
            ),
            canvas_rewrite={
                "status": "review_ready",
                "validation_errors": "bad",
            },
            operator_signoff=True,
        )


def test_command_tables_remain_json_serialisable_for_operator_artifacts() -> None:
    result = _minimal_result()
    service_manifest = ui.build_service_process_manifest(result.project_state)
    package = ui.build_deployment_package(result.project_state)
    plan = ui.build_package_materialisation_plan(result.project_state)

    rendered = json.dumps(
        {
            "service_manifest": service_manifest,
            "package": package,
            "plan": plan,
        },
        sort_keys=True,
    )

    assert "127.0.0.1:8501:8501" in rendered
    assert "docker build -t scpn-phase-orchestrator:local ." in rendered
    assert "network_opened" in rendered


def test_service_manifest_blocks_validation_errors_and_renders_compose_yaml() -> None:
    result = _minimal_result()
    warning_export = replace(
        result.project_state.exports[0],
        warnings=("binding validation failed",),
    )
    blocked_state = replace(
        result.project_state,
        exports=(warning_export,) + tuple(result.project_state.exports[1:]),
    )

    blocked_manifest = ui.build_service_process_manifest(blocked_state)
    ready_manifest = ui.build_service_process_manifest(result.project_state)

    assert blocked_manifest["overall_status"] == "blocked"
    assert blocked_manifest["services"] == []
    assert blocked_manifest["compose_yaml"] == ""
    assert blocked_manifest["compose_yaml_sha256"] == ""
    assert blocked_manifest["blocked_reasons"] == ["binding validation failed"]

    assert ready_manifest["overall_status"] == "operator_ready"
    assert ready_manifest["services"]
    assert (
        ready_manifest["compose_yaml_sha256"]
        == sha256(ready_manifest["compose_yaml"].encode("utf-8")).hexdigest()
    )
    assert "services:" in ready_manifest["compose_yaml"]
    assert "spo-studio-ui" in ready_manifest["compose_yaml"]
    assert "spo-binding-validator" in ready_manifest["compose_yaml"]
    assert "spo-connector-boundary" in ready_manifest["compose_yaml"]
    assert "ports:" in ready_manifest["compose_yaml"]


def test_build_command_table_skips_blocked_targets_from_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _minimal_result()
    project_state = result.project_state

    monkeypatch.setattr(
        ui,
        "build_deployment_readiness",
        lambda _state: {
            "project_name": project_state.project_name,
            "overall_status": "review_ready",
            "targets": (
                {
                    "target": "docker",
                    "status": "ready",
                    "required_artifacts": ("binding_spec.yaml",),
                    "commands": ("echo docker-ready",),
                    "operator_action": "ready for packaging",
                },
                {
                    "target": "wasm",
                    "status": "blocked",
                    "required_artifacts": ("binding_spec.yaml",),
                    "commands": ("echo wasm-blocked",),
                    "operator_action": "resolve blocked reasons",
                },
            ),
        },
    )

    commands = ui.build_command_table(project_state)
    package = ui.build_package_materialisation_plan(project_state)

    assert [row["command"] for row in commands] == ["echo docker-ready"]
    assert package["blocked_targets"] == ["wasm"]
    assert package["commands"][0]["target"] == "docker"


def test_run_owned_live_adapter_routes_supported_transports_and_rejects_unknown() -> (
    None
):
    class _FakeResponse:
        def __init__(self, transport: str):
            self.transport = transport

        def to_audit_record(self) -> dict[str, object]:
            return {"accepted": True, "transport": self.transport}

    class _RestAdapter:
        def __init__(self, name: str) -> None:
            self.name = name

        @classmethod
        def for_contract(cls, _contract: object, name: str) -> _RestAdapter:
            return cls(name)

        def handle_post(
            self,
            envelope_record: dict[str, object],
            headers: dict[str, str],
        ) -> _FakeResponse:
            return _FakeResponse("rest")

        def to_audit_record(self) -> dict[str, object]:
            return {"adapter": self.name}

    class _GrpcAdapter:
        def __init__(self, name: str) -> None:
            self.name = name

        @classmethod
        def for_contract(cls, _contract: object, name: str) -> _GrpcAdapter:
            return cls(name)

        def handle_unary(
            self,
            envelope_record: dict[str, object],
            metadata: dict[str, str],
        ) -> _FakeResponse:
            return _FakeResponse("grpc")

        def to_audit_record(self) -> dict[str, object]:
            return {"adapter": self.name}

    class _KafkaAdapter:
        topic = "studio-topic"

        def __init__(self, name: str) -> None:
            self.name = name

        @classmethod
        def for_contract(cls, _contract: object, name: str) -> _KafkaAdapter:
            return cls(name)

        def handle_message(
            self,
            message: dict[str, object],
            headers: dict[str, str],
        ) -> _FakeResponse:
            assert message["topic"] == self.topic
            return _FakeResponse("kafka")

        def to_audit_record(self) -> dict[str, object]:
            return {"adapter": self.name}

    class _HardwareAdapter:
        def __init__(self, name: str) -> None:
            self.name = name

        @classmethod
        def for_contract(
            cls,
            _contract: object,
            name: str,
            device_ids: tuple[str, ...],
        ) -> _HardwareAdapter:
            assert "studio-review-device" in device_ids
            return cls(name)

        def handle_frame(
            self,
            frame: dict[str, object],
            headers: dict[str, str],
        ) -> _FakeResponse:
            assert frame["device_id"] == "studio-review-device"
            return _FakeResponse("hardware")

        def to_audit_record(self) -> dict[str, object]:
            return {"adapter": self.name}

    with pytest.MonkeyPatch.context() as patch_ctx:
        patch_ctx.setattr(ui, "DigitalTwinSyncRestAdapter", _RestAdapter)
        patch_ctx.setattr(ui, "DigitalTwinSyncGrpcAdapter", _GrpcAdapter)
        patch_ctx.setattr(ui, "DigitalTwinSyncKafkaAdapter", _KafkaAdapter)
        patch_ctx.setattr(ui, "DigitalTwinSyncHardwareAdapter", _HardwareAdapter)

        for transport in ("rest", "grpc", "kafka", "hardware"):
            response, adapter = ui._run_owned_live_adapter(
                contract=object(),
                transport=transport,
                envelope_record={"ok": True},
            )
            assert response["accepted"] is True
            assert response["transport"] == transport
            assert adapter["adapter"] in {
                "studio-rest",
                "studio-grpc",
                "studio-kafka",
                "studio-hardware",
            }

    with pytest.raises(ValueError, match="connector transport"):
        ui._run_owned_live_adapter(
            contract=object(),
            transport="memory",
            envelope_record={"ok": True},
        )


def test_stable_json_payload_normalises_nested_values_and_is_deterministic() -> None:
    payload_a = {
        "zebra": 1,
        "alpha": {
            "nested": [
                {"inner": "text", "items": (1, 2, 3), "value": 2.0},
                {"value": 2.0, "items": [1, 2, 3]},
            ],
            "flag": False,
        },
        "list": [0, {"z": -1}],
    }
    payload_b = {
        "list": [0, {"z": -1}],
        "zebra": 1,
        "alpha": {
            "flag": False,
            "nested": (
                {"items": [1, 2, 3], "value": 2.0, "inner": "text"},
                {"value": 2.0, "items": (1, 2, 3)},
            ),
        },
    }

    canonical_a = ui._stable_json_payload(payload_a, "payload")
    canonical_b = ui._stable_json_payload(payload_b, "payload")
    assert (
        canonical_a
        == canonical_b
        == json.dumps(
            {
                "alpha": {
                    "flag": False,
                    "nested": [
                        {"inner": "text", "items": [1, 2, 3], "value": 2.0},
                        {"value": 2.0, "items": [1, 2, 3]},
                    ],
                },
                "list": [0, {"z": -1}],
                "zebra": 1,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def test_stable_json_payload_reports_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="non-finite float"):
        ui._stable_json_payload({"payload": float("nan")}, "payload")

    with pytest.raises(ValueError, match="contains a non-JSON-safe value"):
        ui._stable_json_payload({"payload": object()}, "payload")

    with pytest.raises(ValueError, match="contains an invalid key"):
        ui._stable_json_payload({1: "bad"}, "payload")
