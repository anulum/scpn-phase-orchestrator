# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SPO Studio guidance and runtime-snapshot builders

"""Beginner guidance, runtime snapshot, project state, and error report builders."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from scpn_phase_orchestrator.binding import validate_binding_spec
from scpn_phase_orchestrator.binding.loader import load_binding_spec
from scpn_phase_orchestrator.studio.workflow import (
    BindingProposal,
    ImportedSourceSummary,
    JsonValue,
    RuntimeSnapshot,
    StudioProjectState,
)

from ._shared import (
    _canvas_graph_count,
    _finite_number,
    _layer_metrics,
    _require_non_empty_text,
    _require_sequence,
)
from ._state import StudioReplayResult
from .deployment import build_export_manifests

if TYPE_CHECKING:
    from ._state import StudioKnobState


def build_beginner_guidance(result: StudioReplayResult) -> dict[str, object]:
    """Return domain-term guidance for first-time Studio operators.

    Parameters
    ----------
    result : StudioReplayResult
        The Studio replay result.

    Returns
    -------
    dict[str, object]
        Domain-term guidance for first-time Studio operators.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    """
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
        for node in _require_sequence(result.canvas_graph.get("nodes", ()), "nodes")
        if isinstance(node, Mapping) and node.get("kind") == "channel"
    ]
    validation_errors = list(project.binding.validation_errors)
    canvas_evidence = {
        "layers": _canvas_graph_count(result, "layer_count"),
        "channels": _canvas_graph_count(result, "channel_count"),
        "couplings": _canvas_graph_count(result, "edge_count"),
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
                    "cross_channel_edges": _canvas_graph_count(result, "edge_count"),
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
                    "connector_count": len(
                        _require_sequence(
                            result.connector_plan.get("connectors", ()),
                            "connectors",
                        )
                    ),
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
    """Build a workflow runtime snapshot from a simulation state dict.

    Parameters
    ----------
    final_state : Mapping[str, object]
        The final simulation state mapping.
    knobs : StudioKnobState
        The Studio knob state.
    hierarchy_watermarks : Mapping[str, int] | None
        Per-source hierarchy watermarks, or ``None``.
    replay_status : str
        Replay status label.

    Returns
    -------
    RuntimeSnapshot
        A workflow runtime snapshot from a simulation state dict.
    """
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
    """Create a Studio project state from an existing binding spec file.

    Parameters
    ----------
    project_name : str
        Name of the project.
    spec_path : Path
        Filesystem path to the binding-spec file.
    knobs : StudioKnobState
        The Studio knob state.
    runtime : RuntimeSnapshot
        The workflow runtime snapshot.

    Returns
    -------
    StudioProjectState
        A Studio project state from an existing binding spec file.
    """
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


def build_error_report(
    *,
    operation: str,
    error: Exception,
    project_name: str = "unknown",
) -> dict[str, object]:
    """Return a path-safe operator report for failed Studio actions.

    Parameters
    ----------
    operation : str
        The operation label.
    error : Exception
        The exception that was raised.
    project_name : str
        Name of the project.

    Returns
    -------
    dict[str, object]
        A path-safe operator report for failed Studio actions.
    """
    return {
        "project_name": _require_non_empty_text(project_name, "project_name"),
        "operation": _require_non_empty_text(operation, "operation"),
        "status": "blocked",
        "error_type": type(error).__name__,
        "operator_action": "review input artefacts and rerun",
    }
