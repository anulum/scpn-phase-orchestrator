#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SPO Studio (Streamlit GUI)
#
# Streamlit operator surface for local SPO replay and review artefacts.

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st  # type: ignore[import-not-found]

from scpn_phase_orchestrator.autotune.binding_proposal import (
    propose_binding_from_event_log,
    propose_binding_from_graph,
    propose_binding_from_time_series_csv,
)
from scpn_phase_orchestrator.studio.ui_helpers import (
    StudioKnobState,
    StudioReplayResult,
    apply_canvas_binding_rewrite_candidate,
    build_beginner_guidance,
    build_canvas_binding_rewrite_candidate,
    build_canvas_edit_artifact,
    build_canvas_interaction_state,
    build_canvas_layout_manifest,
    build_canvas_topology_patch,
    build_command_table,
    build_deployment_package,
    build_deployment_readiness,
    build_error_report,
    build_hardware_target_package,
    build_live_connector_run_record,
    build_operator_checklist,
    build_oscillator_edit_artifact,
    build_owned_live_connector_runtime_record,
    build_package_materialisation_plan,
    build_regime_chart_payload,
    build_series_chart_payload,
    build_verified_hardware_target_package,
    disabled_export_reasons,
    discover_domainpacks,
    run_binding_spec_replay,
)

st.set_page_config(
    page_title="SPO Studio",
    page_icon="SPO",
    layout="wide",
)

st.title("SPO Studio")


def _domainpack_dir() -> Path:
    candidate = Path(__file__).parent.parent / "domainpacks"
    if candidate.exists():
        return candidate
    return Path("domainpacks")


def _selected_result() -> StudioReplayResult | None:
    value = st.session_state.get("studio_result")
    if isinstance(value, StudioReplayResult):
        return value
    return None


def _render_metrics(result: StudioReplayResult) -> None:
    runtime = result.project_state.runtime
    metrics = st.columns(5)
    metrics[0].metric("R", f"{runtime.R:.3f}")
    metrics[1].metric("Psi", f"{runtime.Psi:.3f}")
    metrics[2].metric("K", f"{runtime.K:.2f}")
    metrics[3].metric("Regime", runtime.regime)
    metrics[4].metric("Replay", runtime.replay_status)


def _render_exports(result: StudioReplayResult) -> None:
    validation_errors = result.project_state.binding.validation_errors
    disabled_reasons = disabled_export_reasons(validation_errors)
    if disabled_reasons:
        st.error("\n".join(disabled_reasons))
    for manifest in result.export_manifests:
        st.download_button(
            label=manifest.file_name,
            data=manifest.payload,
            file_name=manifest.file_name,
            mime=_mime_for_export(manifest.file_name),
            disabled=bool(disabled_reasons)
            and manifest.target_kind in {"docker_manifest", "wasm_manifest"},
            use_container_width=True,
        )


def _mime_for_export(file_name: str) -> str:
    if file_name.endswith(".json"):
        return "application/json"
    if file_name.endswith((".yaml", ".yml")):
        return "application/x-yaml"
    return "text/plain"


def _render_error_report(report: dict[str, object]) -> None:
    st.error(f"{report['operation']} failed: {report['error_type']}")
    st.download_button(
        label="studio_error_report.json",
        data=json.dumps(report, sort_keys=True, indent=2),
        file_name="studio_error_report.json",
        mime="application/json",
        use_container_width=True,
    )


def _run_replay_or_report(
    spec_path: Path,
    *,
    steps: int,
    knobs: StudioKnobState,
    project_name: str,
) -> StudioReplayResult | None:
    try:
        return run_binding_spec_replay(spec_path, steps=steps, knobs=knobs)
    except (OSError, ValueError, TypeError) as exc:
        _render_error_report(
            build_error_report(
                operation="run_replay",
                error=exc,
                project_name=project_name,
            )
        )
        return None


domainpack_dir = _domainpack_dir()
packs = discover_domainpacks(domainpack_dir)
if not packs:
    st.error("No domainpacks with binding_spec.yaml were found.")
    st.stop()

default_index = packs.index("minimal_domain") if "minimal_domain" in packs else 0
with st.sidebar:
    domain = st.selectbox("Domainpack", packs, index=default_index)
    steps = st.slider("Steps", 10, 500, 100, step=10)
    knobs = StudioKnobState(
        K=st.slider("K", 0.1, 10.0, 1.0, step=0.1),
        alpha=st.slider("alpha", 0.0, 5.0, 0.0, step=0.1),
        zeta=st.slider("zeta", 0.0, 5.0, 0.0, step=0.1),
        Psi=st.slider("Psi", 0.0, 10.0, 0.0, step=0.1),
    )
    if st.button("Run Replay", type="primary", use_container_width=True):
        replay_result = _run_replay_or_report(
            domainpack_dir / domain / "binding_spec.yaml",
            steps=steps,
            knobs=knobs,
            project_name=domain,
        )
        if replay_result is not None:
            st.session_state["studio_result"] = replay_result

result = _selected_result()
if result is None:
    result = _run_replay_or_report(
        domainpack_dir / domain / "binding_spec.yaml",
        steps=10,
        knobs=knobs,
        project_name=domain,
    )
    if result is None:
        st.stop()
    st.session_state["studio_result"] = result

project = result.project_state
_render_metrics(result)

tabs = st.tabs(
    [
        "Load",
        "Guide",
        "Binding",
        "Oscillators",
        "Canvas",
        "Live",
        "Autotune",
        "Hierarchy",
        "Connectors",
        "Exports",
    ]
)

with tabs[0]:
    source = project.source.to_audit_record()
    st.json(source, expanded=False)
    uploaded = st.file_uploader(
        "Import source",
        type=("csv", "json", "jsonl"),
        accept_multiple_files=False,
    )
    source_kind = st.segmented_control(
        "Source family",
        ["time_series_csv", "event_log_json", "graph_json"],
        default="time_series_csv",
    )
    if uploaded is not None:
        text = uploaded.getvalue().decode("utf-8")
        try:
            if source_kind == "time_series_csv":
                proposal = propose_binding_from_time_series_csv(
                    text,
                    sample_rate_hz=st.number_input(
                        "Sample rate Hz",
                        min_value=0.001,
                        value=1.0,
                    ),
                    project_name=uploaded.name,
                )
            elif source_kind == "event_log_json":
                proposal = propose_binding_from_event_log(
                    text,
                    project_name=uploaded.name,
                )
            else:
                proposal = propose_binding_from_graph(
                    text,
                    project_name=uploaded.name,
                )
        except (ValueError, TypeError) as exc:
            _render_error_report(
                build_error_report(
                    operation=f"import_{source_kind}",
                    error=exc,
                    project_name=uploaded.name,
                )
            )
        else:
            st.code(proposal.binding.yaml_text, language="yaml")
            st.json(proposal.binding.to_audit_record(), expanded=False)

with tabs[1]:
    guidance = build_beginner_guidance(result)
    summary = guidance["runtime_summary"]
    st.dataframe(
        [
            {"metric": "Replay", "value": summary["replay_status"]},
            {"metric": "Regime", "value": summary["regime"]},
            {"metric": "R", "value": summary["R"]},
            {"metric": "Actuation", "value": "disabled"},
        ],
        hide_index=True,
        use_container_width=True,
    )
    st.info(summary["domain_signal"])
    for card in guidance["concept_cards"]:
        with st.expander(card["title"], expanded=True):
            st.write(card["plain_language"])
            st.json(card["evidence"], expanded=False)
    st.dataframe(
        [
            {"step": index, "action": action}
            for index, action in enumerate(guidance["next_actions"], 1)
        ],
        hide_index=True,
        use_container_width=True,
    )
    st.dataframe(
        guidance["walkthrough_steps"],
        hide_index=True,
        use_container_width=True,
    )
    st.download_button(
        label="beginner_guidance.json",
        data=json.dumps(guidance, sort_keys=True, indent=2),
        file_name="beginner_guidance.json",
        mime="application/json",
        use_container_width=True,
    )

with tabs[2]:
    validation_errors = project.binding.validation_errors
    if validation_errors:
        st.error("\n".join(validation_errors))
    else:
        st.success("Binding validation passed")
    st.code(project.binding.yaml_text, language="yaml")

with tabs[3]:
    edited = st.data_editor(
        list(result.oscillator_table),
        hide_index=True,
        use_container_width=True,
        num_rows="dynamic",
        disabled=("layer_index",),
    )
    edit_artifact = build_oscillator_edit_artifact(result.oscillator_table, edited)
    edit_record = json.loads(edit_artifact.payload)
    if edit_record["changed"]:
        st.warning("Oscillator edits are staged as a review artefact.")
    else:
        st.success("Oscillator table matches the current binding.")
    st.download_button(
        label=edit_artifact.file_name,
        data=edit_artifact.payload,
        file_name=edit_artifact.file_name,
        mime="application/json",
        use_container_width=True,
    )
    st.json(edit_record, expanded=False)

with tabs[4]:
    canvas_graph = result.canvas_graph
    st.dataframe(
        [
            {
                "metric": "layers",
                "value": canvas_graph["layer_count"],
            },
            {
                "metric": "channels",
                "value": canvas_graph["channel_count"],
            },
            {
                "metric": "couplings",
                "value": canvas_graph["edge_count"],
            },
        ],
        hide_index=True,
        use_container_width=True,
    )
    canvas_nodes = st.data_editor(
        list(canvas_graph["nodes"]),
        hide_index=True,
        use_container_width=True,
        num_rows="dynamic",
        disabled=("id", "kind"),
        key="canvas_nodes",
    )
    canvas_edges = st.data_editor(
        list(canvas_graph["edges"]),
        hide_index=True,
        use_container_width=True,
        num_rows="dynamic",
        disabled=("id", "kind"),
        key="canvas_edges",
    )
    canvas_artifact = build_canvas_edit_artifact(
        canvas_graph,
        {"nodes": canvas_nodes, "edges": canvas_edges},
    )
    canvas_layout = build_canvas_layout_manifest(
        project_name=project.project_name,
        graph={"nodes": canvas_nodes, "edges": canvas_edges},
    )
    canvas_patch = build_canvas_topology_patch(
        project_name=project.project_name,
        before_graph=canvas_graph,
        after_graph={"nodes": canvas_nodes, "edges": canvas_edges},
    )
    canvas_rewrite = build_canvas_binding_rewrite_candidate(
        result,
        after_graph={"nodes": canvas_nodes, "edges": canvas_edges},
    )
    apply_signoff = st.checkbox(
        "Apply reviewed binding rewrite candidate to binding_spec.yaml",
        value=False,
        help="Requires a review-ready candidate and an unchanged source SHA-256.",
    )
    canvas_state = build_canvas_interaction_state(
        canvas_artifact=canvas_artifact,
        canvas_layout=canvas_layout,
        canvas_patch=canvas_patch,
        canvas_rewrite=canvas_rewrite,
        operator_signoff=apply_signoff,
    )
    if st.button(
        "Apply binding rewrite",
        disabled=not canvas_state["apply_enabled"],
        use_container_width=True,
    ):
        apply_record = apply_canvas_binding_rewrite_candidate(
            canvas_rewrite,
            binding_spec_path=domainpack_dir / domain / "binding_spec.yaml",
            operator_signoff=apply_signoff,
        )
        if apply_record["status"] == "applied":
            st.success("binding_spec.yaml updated with a reviewed candidate.")
        else:
            st.error("Binding rewrite was blocked.")
        st.json(apply_record, expanded=False)
    if canvas_state["changed"]:
        st.warning(canvas_state["status_message"])
    else:
        st.success(canvas_state["status_message"])
    st.json(
        {
            "next_action": canvas_state["next_action"],
            "apply_enabled": canvas_state["apply_enabled"],
            "disabled_reasons": canvas_state["disabled_reasons"],
        },
        expanded=False,
    )
    st.download_button(
        label=canvas_artifact.file_name,
        data=canvas_artifact.payload,
        file_name=canvas_artifact.file_name,
        mime="application/json",
        use_container_width=True,
    )
    st.download_button(
        label=canvas_layout.file_name,
        data=canvas_layout.payload,
        file_name=canvas_layout.file_name,
        mime="application/json",
        use_container_width=True,
    )
    st.download_button(
        label=canvas_patch.file_name,
        data=canvas_patch.payload,
        file_name=canvas_patch.file_name,
        mime="application/json",
        use_container_width=True,
    )
    st.download_button(
        label="binding_rewrite_candidate.yaml",
        data=canvas_rewrite["candidate_yaml"],
        file_name="binding_rewrite_candidate.yaml",
        mime="application/x-yaml",
        use_container_width=True,
        disabled=canvas_rewrite["status"] != "review_ready",
    )
    st.json(
        {
            "rewrite_status": canvas_state["rewrite_status"],
            "validation_errors": canvas_rewrite["validation_errors"],
            "candidate_yaml_sha256": canvas_state["candidate_yaml_sha256"],
        },
        expanded=False,
    )

with tabs[5]:
    st.line_chart(
        build_series_chart_payload("R", result.r_history),
        x="step",
        y="R",
        use_container_width=True,
    )
    st.area_chart(
        build_regime_chart_payload(result.regime_history),
        x="step",
        y="regime_level",
        use_container_width=True,
    )
    st.dataframe(result.layer_table, hide_index=True, use_container_width=True)

with tabs[6]:
    st.json(
        {
            "replay_status": project.runtime.replay_status,
            "actuation_permitted": False,
            "knobs": {
                "K": project.runtime.K,
                "alpha": project.runtime.alpha,
                "zeta": project.runtime.zeta,
                "Psi": project.runtime.Psi,
            },
            "binding_validation_errors": list(project.binding.validation_errors),
        },
        expanded=False,
    )

with tabs[7]:
    st.json(
        {
            "watermarks": project.runtime.hierarchy_watermarks,
            "layer_metrics": project.runtime.layer_metrics,
            "regime": project.runtime.regime,
        },
        expanded=False,
    )

with tabs[8]:
    connector_plan = result.connector_plan
    st.dataframe(
        list(connector_plan["connectors"]),
        hide_index=True,
        use_container_width=True,
    )
    st.json(
        {
            "contract_hash": connector_plan["contract_hash"],
            "network_opened": connector_plan["network_opened"],
            "actuation_permitted": connector_plan["actuation_permitted"],
        },
        expanded=False,
    )
    st.download_button(
        label="connector_plan.json",
        data=json.dumps(connector_plan, sort_keys=True, indent=2),
        file_name="connector_plan.json",
        mime="application/json",
        use_container_width=True,
    )
    connector_transport = st.selectbox(
        "Connector dry-run transport",
        [connector["transport"] for connector in connector_plan["connectors"]],
    )
    connector_payload = st.text_area(
        "Connector dry-run payload JSON",
        value='{"kind":"replay_probe","sequence":1}',
        height=120,
    )
    if st.button("Build Connector Run Record", use_container_width=True):
        try:
            connector_run = build_live_connector_run_record(
                connector_plan,
                transport=connector_transport,
                payload=json.loads(connector_payload),
                dry_run=True,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            _render_error_report(
                build_error_report(
                    operation="connector_dry_run",
                    error=exc,
                    project_name=project.project_name,
                )
            )
        else:
            st.json(connector_run, expanded=False)
            st.download_button(
                label="connector_run_record.json",
                data=json.dumps(connector_run, sort_keys=True, indent=2),
                file_name="connector_run_record.json",
                mime="application/json",
                use_container_width=True,
            )
    st.subheader("Owned Runtime Boundary")
    runtime_owner = st.text_input("Connector owner", value="")
    runtime_auth_label = st.text_input("Credential label", value="")
    if st.button("Validate Owned Runtime Boundary", use_container_width=True):
        try:
            runtime_record = build_owned_live_connector_runtime_record(
                result,
                transport=connector_transport,
                owner=runtime_owner,
                auth_policy={
                    "scheme": "bearer",
                    "credential_label": runtime_auth_label,
                },
                payload=json.loads(connector_payload),
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            _render_error_report(
                build_error_report(
                    operation="connector_owned_runtime",
                    error=exc,
                    project_name=project.project_name,
                )
            )
        else:
            st.json(runtime_record, expanded=False)
            st.download_button(
                label="owned_connector_runtime.json",
                data=json.dumps(runtime_record, sort_keys=True, indent=2),
                file_name="owned_connector_runtime.json",
                mime="application/json",
                use_container_width=True,
            )

with tabs[9]:
    readiness = build_deployment_readiness(project)
    package = build_deployment_package(project)
    materialisation_plan = build_package_materialisation_plan(project)
    hardware_package = build_hardware_target_package(result)
    st.subheader("Deployment Readiness")
    st.dataframe(
        list(build_operator_checklist(project)),
        hide_index=True,
        use_container_width=True,
    )
    command_rows = build_command_table(project)
    if command_rows:
        st.subheader("Review Commands")
        st.dataframe(
            list(command_rows),
            hide_index=True,
            use_container_width=True,
        )
    st.json(readiness, expanded=False)
    st.download_button(
        label="deployment_readiness.json",
        data=json.dumps(readiness, sort_keys=True, indent=2),
        file_name="deployment_readiness.json",
        mime="application/json",
        use_container_width=True,
    )
    st.download_button(
        label="deployment_package.json",
        data=json.dumps(package, sort_keys=True, indent=2),
        file_name="deployment_package.json",
        mime="application/json",
        use_container_width=True,
    )
    st.download_button(
        label="package_materialisation_plan.json",
        data=json.dumps(materialisation_plan, sort_keys=True, indent=2),
        file_name="package_materialisation_plan.json",
        mime="application/json",
        use_container_width=True,
    )
    st.download_button(
        label="hardware_target_package.json",
        data=json.dumps(hardware_package, sort_keys=True, indent=2),
        file_name="hardware_target_package.json",
        mime="application/json",
        use_container_width=True,
    )
    evidence_text = st.text_area(
        "Verified hardware evidence JSON",
        value="",
        height=120,
    )
    if evidence_text.strip():
        try:
            evidence_payload = json.loads(evidence_text)
            verified_hardware_package = build_verified_hardware_target_package(
                result,
                evidence=evidence_payload,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            _render_error_report(
                build_error_report(
                    operation="verify_hardware_evidence",
                    error=exc,
                    project_name=project.project_name,
                )
            )
        else:
            st.download_button(
                label="verified_hardware_target_package.json",
                data=json.dumps(verified_hardware_package, sort_keys=True, indent=2),
                file_name="verified_hardware_target_package.json",
                mime="application/json",
                use_container_width=True,
            )
    _render_exports(result)
    st.download_button(
        label="project_state.json",
        data=json.dumps(project.to_audit_record(), sort_keys=True, indent=2),
        file_name="project_state.json",
        mime="application/json",
        use_container_width=True,
    )
