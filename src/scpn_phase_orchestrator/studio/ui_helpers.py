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
from numbers import Real
from pathlib import Path
from typing import cast

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
from scpn_phase_orchestrator.binding.loader import BindingLoadError, load_binding_spec
from scpn_phase_orchestrator.binding.types import BindingSpec
from scpn_phase_orchestrator.coupling.knm import CouplingState
from scpn_phase_orchestrator.runtime.server import SimulationState
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
    "build_autopoietic_lineage_studio_panel",
    "build_intergenerational_inheritance_studio_panel",
    "build_evolutionary_supervisor_policy_search_studio_panel",
    "build_hybrid_order_studio_panel",
    "build_information_geometry_studio_panel",
    "build_integrated_information_panel",
    "build_sheaf_cohomology_studio_panel",
    "build_layer_table",
    "build_hardware_target_package",
    "build_live_connector_plan",
    "build_live_connector_run_record",
    "build_morphogenetic_field_studio_panel",
    "build_multiverse_counterfactual_studio_panel",
    "build_owned_live_connector_runtime_record",
    "build_oscillator_edit_artifact",
    "build_oscillator_table",
    "build_operator_checklist",
    "build_regime_chart_payload",
    "build_package_materialisation_plan",
    "build_runtime_snapshot",
    "build_series_chart_payload",
    "build_service_process_manifest",
    "build_strange_loop_studio_panel",
    "build_topos_semantic_binding_studio_panel",
    "build_verified_hardware_target_package",
    "disabled_export_reasons",
    "discover_domainpacks",
    "run_binding_spec_replay",
]

_STRANGE_LOOP_CLAIM_BOUNDARY = "strange_loop_drift_review_not_live_actuation"
_STRANGE_LOOP_TRIGGERS = frozenset(
    {"stable", "policy_drift", "control_loop_oscillation", "over_control"}
)
_MULTIVERSE_ROLLOUT_CLAIM_BOUNDARY = "counterfactual_branch_rollout_not_live_actuation"
_MULTIVERSE_RISK_CLAIM_BOUNDARY = "counterfactual_branch_risk_gate_not_live_actuation"
_MULTIVERSE_BACKENDS = frozenset({"numpy_vectorized", "jax_vectorized"})
_HYBRID_ORDER_CLAIM_BOUNDARY = "quantum_cosimulation_monitor_not_qpu_execution"
_HYBRID_ORDER_BACKENDS = frozenset(
    {
        "numpy_statevector_density_matrix",
        "numpy_statevector",
        "numpy_density_matrix",
    }
)
_INFORMATION_GEOMETRY_CLAIM_BOUNDARY = "information_geometry_control_not_live_actuation"
_INFORMATION_GEOMETRY_BACKENDS = frozenset(
    {
        "numpy_jax_compatible_information_geometry",
        "jax_native_information_geometry",
    }
)
_SHEAF_COHOMOLOGY_CLAIM_BOUNDARY = "sheaf_cohomology_review_not_live_actuation"
_SHEAF_RESULT_METHOD = "directed_cellular_sheaf_laplacian"
_SHEAF_CONTROL_METHOD = "sheaf_laplacian_gradient_descent_review"
_TOPOS_PROOF_BOUNDARY = "categorical_validation_prototype_not_formal_topos_proof"
_TOPOS_REPORT_SCHEMAS = frozenset(
    {
        "symbolic_binding_functor",
        "policy_composition_category",
    }
)
_EVOLUTIONARY_SEARCH_BOUNDARY = (
    "offline_evolutionary_supervisor_review_not_live_actuation"
)
_EVOLUTIONARY_EXAMPLE_BOUNDARY = "evolutionary_supervisor_search_not_live_actuation"
_EVOLUTIONARY_SEARCH_SCHEMA = "evolutionary_supervisor_policy_search"
_EVOLUTIONARY_DSL_SCHEMA = "policy_dsl_evolution"
_AUTOPOIETIC_LINEAGE_SCHEMA = "scpn_autopoietic_lineage_sandbox_v1"
_AUTOPOIETIC_LINEAGE_BOUNDARY = "autopoietic_lineage_sandbox_review_not_live_merge"
_INTERGENERATIONAL_HISTORY_SCHEMA = (
    "scpn_intergenerational_policy_inheritance_history_v1"
)
_INTERGENERATIONAL_HISTORY_BOUNDARY = (
    "intergenerational_inheritance_review_not_direct_hot_patch"
)


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
        """Return a JSON-safe knob record.

        Returns
        -------
        dict[str, float]
            A JSON-safe knob record.
        """
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
        """Return a JSON-safe replay audit record.

        Returns
        -------
        dict[str, object]
            A JSON-safe replay audit record.
        """
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
    """Return domainpack names containing a binding spec.

    Parameters
    ----------
    domainpack_dir : Path
        Directory containing domainpacks.

    Returns
    -------
    tuple[str, ...]
        Domainpack names containing a binding spec.
    """
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
    """Return validated knobs after a UI edit.

    Parameters
    ----------
    knobs : StudioKnobState
        The Studio knob state.
    K : float | None
        Coupling-strength knob value, or ``None``.
    alpha : float | None
        Phase-lag knob value, or ``None``.
    zeta : float | None
        Drive-strength knob value, or ``None``.
    Psi : float | None
        Drive reference-phase knob value, or ``None``.

    Returns
    -------
    StudioKnobState
        Validated knobs after a UI edit.
    """
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
    """Return dense chart rows for a scalar time-series.

    Parameters
    ----------
    label : str
        Series or chart label.
    values : Sequence[float]
        Scalar time-series values.

    Returns
    -------
    list[dict[str, float | int]]
        Dense chart rows for a scalar time-series.
    """
    _require_non_empty_text(label, "label")
    return [
        {"step": index, label: _finite_number(value, label)}
        for index, value in enumerate(values, 1)
    ]


def build_regime_chart_payload(regimes: Sequence[str]) -> list[dict[str, object]]:
    """Return deterministic chart rows for regime timelines.

    Parameters
    ----------
    regimes : Sequence[str]
        Per-step regime labels.

    Returns
    -------
    list[dict[str, object]]
        Deterministic chart rows for regime timelines.
    """
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


def build_integrated_information_panel(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Return a Studio panel payload for integrated-information audit records.

    The panel is deliberately pure and non-actuating: it converts validated
    monitor audit records into a deterministic operator payload suitable for
    rendering charts, latest-value tiles, and partition review cards. The input
    must preserve the monitor's explicit claim boundary so Studio cannot display
    the Phi proxy as a theoretical IIT or consciousness claim.

    Parameters
    ----------
    records : Sequence[Mapping[str, object]]
        The records to summarise.

    Returns
    -------
    dict[str, object]
        A Studio panel payload for integrated-information audit records.
    """
    normalised_records = _normalise_integrated_information_records(records)
    latest = normalised_records[-1]
    strongest = max(
        normalised_records,
        key=lambda item: cast("float", item["phi"]),
    )
    phi_values = [cast("float", item["phi"]) for item in normalised_records]
    normalised_phi_values = [
        cast("float", item["normalised_phi"]) for item in normalised_records
    ]
    total_integration_values = [
        cast("float", item["total_integration"]) for item in normalised_records
    ]
    return {
        "panel_kind": "studio_integrated_information_panel",
        "monitor": "integrated_information",
        "record_count": len(normalised_records),
        "claim_boundary": "engineering_proxy_not_theoretical_iit",
        "latest": latest,
        "strongest_partition": strongest,
        "series": normalised_records,
        "phi_range": {
            "min": min(phi_values),
            "max": max(phi_values),
        },
        "normalised_phi_range": {
            "min": min(normalised_phi_values),
            "max": max(normalised_phi_values),
        },
        "total_integration_range": {
            "min": min(total_integration_values),
            "max": max(total_integration_values),
        },
        "operator_summary": (
            "latest Phi proxy "
            f"{cast('float', latest['phi']):.6g}; latest normalised Phi "
            f"{cast('float', latest['normalised_phi']):.6g}; records "
            f"{len(normalised_records)}"
        ),
        "operator_action": (
            "render as an engineering integration proxy; preserve the claim "
            "boundary and review the minimum partition before operational use"
        ),
        "actuation_permitted": False,
        "consciousness_claim_permitted": False,
    }


def build_strange_loop_studio_panel(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Return a Studio panel payload for strange-loop drift scenario records.

    The panel renders precomputed ``StrangeLoopSupervisor`` review evidence.
    It does not observe live actions, execute recommendations, or apply control
    changes. All scenario records must keep the supervisor's non-actuating
    claim boundary and disabled-execution flags intact.

    Parameters
    ----------
    records : Sequence[Mapping[str, object]]
        The records to summarise.

    Returns
    -------
    dict[str, object]
        A Studio panel payload for strange-loop drift scenario records.
    """
    normalised_records = _normalise_strange_loop_records(records)
    drift_scores = [
        cast("float", record["max_drift_score"]) for record in normalised_records
    ]
    oscillation_scores = [
        cast("float", record["max_oscillation_score"]) for record in normalised_records
    ]
    overcontrol_scores = [
        cast("float", record["max_overcontrol_score"]) for record in normalised_records
    ]
    coherence_scores = [
        cast("float", record["min_control_coherence"]) for record in normalised_records
    ]
    failed_ids = [
        cast("str", record["scenario_id"])
        for record in normalised_records
        if record["passed_expected_trigger"] is not True
    ]
    triggered_modes = tuple(
        sorted(
            {cast("str", record["expected_trigger"]) for record in normalised_records}
        )
    )
    return {
        "panel_kind": "studio_strange_loop_panel",
        "supervisor": "strange_loop",
        "scenario_count": len(normalised_records),
        "passed_count": len(normalised_records) - len(failed_ids),
        "failed_scenario_ids": failed_ids,
        "triggered_modes": triggered_modes,
        "claim_boundary": _STRANGE_LOOP_CLAIM_BOUNDARY,
        "non_actuating": True,
        "execution_disabled": True,
        "actuation_permitted": False,
        "series": normalised_records,
        "maxima": {
            "drift_score": max(drift_scores),
            "oscillation_score": max(oscillation_scores),
            "overcontrol_score": max(overcontrol_scores),
        },
        "minima": {
            "control_coherence": min(coherence_scores),
        },
        "operator_summary": (
            "strange-loop scenario review: "
            f"{len(normalised_records) - len(failed_ids)}/"
            f"{len(normalised_records)} expected triggers passed"
        ),
        "operator_action": (
            "render as offline supervisor self-control evidence; keep all "
            "recommendations behind the normal review and safety gate"
        ),
    }


def build_information_geometry_studio_panel(
    records: Sequence[Mapping[str, object]],
    *,
    scenarios: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    """Return a Studio panel for information-geometry control evidence.

    The panel renders already-computed Fisher-Rao/Wasserstein review proposals
    and deterministic scenario fixtures. It validates metric tensors, simplex
    coordinates, natural-gradient tangents, geodesic/curvature metrics, hash
    fields, and disabled-execution boundaries before exposing anything to the
    operator surface. No returned field is an executable control channel.

    Parameters
    ----------
    records : Sequence[Mapping[str, object]]
        The records to summarise.
    scenarios : Sequence[Mapping[str, object]]
        Scenario records.

    Returns
    -------
    dict[str, object]
        A Studio panel for information-geometry control evidence.
    """
    normalised_records = _normalise_information_geometry_records(records)
    normalised_scenarios, candidate_rows = _normalise_information_geometry_scenarios(
        scenarios
    )
    fisher_values = [
        cast("float", record["fisher_rao_distance"]) for record in normalised_records
    ]
    wasserstein_values = [
        cast("float", record["wasserstein_distance"]) for record in normalised_records
    ]
    gradient_values = [
        cast("float", record["natural_gradient_norm"]) for record in normalised_records
    ]
    curvature_values = [
        cast("float", record["curvature_proxy"]) for record in normalised_records
    ]
    metric_values = [
        value
        for record in normalised_records
        for row in cast("tuple[tuple[float, ...], ...]", record["metric_tensor"])
        for value in row
    ]
    metric_diagonal_values = [
        row[index]
        for record in normalised_records
        for index, row in enumerate(
            cast("tuple[tuple[float, ...], ...]", record["metric_tensor"])
        )
    ]
    backends = tuple(
        sorted({cast("str", record["backend"]) for record in normalised_records})
    )
    scenario_domains = tuple(
        sorted({cast("str", scenario["domain"]) for scenario in normalised_scenarios})
    )
    return {
        "panel_kind": "studio_information_geometry_panel",
        "supervisor": "information_geometry_control",
        "proposal_count": len(normalised_records),
        "scenario_count": len(normalised_scenarios),
        "claim_boundary": _INFORMATION_GEOMETRY_CLAIM_BOUNDARY,
        "non_actuating": True,
        "execution_disabled": True,
        "actuation_permitted": False,
        "backends": backends,
        "scenario_domains": scenario_domains,
        "latest": normalised_records[-1],
        "series": normalised_records,
        "candidate_rows": candidate_rows,
        "fisher_rao_range": {
            "minimum": min(fisher_values),
            "maximum": max(fisher_values),
        },
        "wasserstein_range": {
            "minimum": min(wasserstein_values),
            "maximum": max(wasserstein_values),
        },
        "natural_gradient_range": {
            "minimum": min(gradient_values),
            "maximum": max(gradient_values),
        },
        "curvature_range": {
            "minimum": min(curvature_values),
            "maximum": max(curvature_values),
        },
        "metric_tensor_range": {
            "minimum": min(metric_values),
            "maximum": max(metric_values),
        },
        "metric_diagonal_range": {
            "minimum": min(metric_diagonal_values),
            "maximum": max(metric_diagonal_values),
        },
        "operator_summary": (
            "information-geometry review: "
            f"{len(normalised_records)} proposal record(s) across "
            f"{len(backends)} backend(s); max Fisher-Rao distance "
            f"{max(fisher_values):.6g}"
        ),
        "operator_action": (
            "render as non-actuating geometry-aware control evidence; compare "
            "Fisher-Rao/Wasserstein distances, metric conditioning, and "
            "natural-gradient magnitude before any separately gated policy use"
        ),
    }


def build_sheaf_cohomology_studio_panel(
    records: Sequence[Mapping[str, object]],
    *,
    summaries: Sequence[Mapping[str, object]],
    control_proposals: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Return a Studio panel for sheaf-cohomology review evidence.

    The panel renders already-computed sheaf-Laplacian obstruction records,
    residual triage summaries, and review-only control proposals. It validates
    cohomology dimensions, finite obstruction/energy metrics, residual rows,
    disabled execution gates, and monotone accepted projections before exposing
    evidence to Studio. No returned field is an executable control channel.

    Parameters
    ----------
    records : Sequence[Mapping[str, object]]
        The records to summarise.
    summaries : Sequence[Mapping[str, object]]
        Summary records.
    control_proposals : Sequence[Mapping[str, object]]
        Sheaf control-proposal records.

    Returns
    -------
    dict[str, object]
        A Studio panel for sheaf-cohomology review evidence.
    """
    normalised_records = _normalise_sheaf_cohomology_records(records)
    normalised_summaries, residual_rows = _normalise_sheaf_obstruction_summaries(
        summaries
    )
    normalised_proposals = _normalise_sheaf_control_proposals(control_proposals)
    obstruction_scores = [
        cast("float", record["obstruction_score"]) for record in normalised_records
    ]
    consistency_energies = [
        cast("float", record["consistency_energy"]) for record in normalised_records
    ]
    kernel_dimensions = [
        cast("int", record["kernel_dimension"]) for record in normalised_records
    ]
    obstruction_dimensions = [
        cast("int", record["obstruction_dimension"]) for record in normalised_records
    ]
    accepted_count = sum(
        1
        for proposal in normalised_proposals
        if proposal["accepted_for_review"] is True
    )
    critical_count = sum(
        1 for summary in normalised_summaries if summary["severity"] == "critical"
    )
    return {
        "panel_kind": "studio_sheaf_cohomology_panel",
        "supervisor": "sheaf_cohomology_control",
        "claim_boundary": _SHEAF_COHOMOLOGY_CLAIM_BOUNDARY,
        "non_actuating": True,
        "execution_disabled": True,
        "operator_review_required": True,
        "actuation_permitted": False,
        "live_merge_permitted": False,
        "hot_patch_permitted": False,
        "record_count": len(normalised_records),
        "summary_count": len(normalised_summaries),
        "control_proposal_count": len(normalised_proposals),
        "accepted_control_proposal_count": accepted_count,
        "critical_summary_count": critical_count,
        "records": normalised_records,
        "summaries": normalised_summaries,
        "control_proposals": normalised_proposals,
        "top_residual_rows": residual_rows,
        "obstruction_range": {
            "minimum": min(obstruction_scores),
            "maximum": max(obstruction_scores),
        },
        "consistency_energy_range": {
            "minimum": min(consistency_energies),
            "maximum": max(consistency_energies),
        },
        "cohomology_dimension_range": {
            "kernel_minimum": min(kernel_dimensions),
            "kernel_maximum": max(kernel_dimensions),
            "obstruction_minimum": min(obstruction_dimensions),
            "obstruction_maximum": max(obstruction_dimensions),
        },
        "operator_summary": (
            "sheaf-cohomology review: "
            f"{len(normalised_records)} obstruction record(s), "
            f"{len(residual_rows)} residual edge row(s), "
            f"{accepted_count}/{len(normalised_proposals)} accepted proposal(s)"
        ),
        "operator_action": (
            "render as non-actuating sheaf-Laplacian obstruction evidence; "
            "review residual edges and cohomology-dimension changes before any "
            "separately approved operator workflow"
        ),
    }


def build_topos_semantic_binding_studio_panel(
    symbolic_reports: Sequence[Mapping[str, object]],
    policy_reports: Sequence[Mapping[str, object]],
    *,
    examples: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    """Return a Studio panel for Topos semantic-binding evidence.

    The helper renders categorical validation reports and deterministic domain
    obligations as review evidence only. It validates schema names, proof
    boundaries, report hashes, obligation/object/morphism counts, non-actuation
    flags, and example hashes before exposing a compact payload to Studio. The
    payload intentionally makes no formal Topos proof claim and emits no
    executable policy actions.

    Parameters
    ----------
    symbolic_reports : Sequence[Mapping[str, object]]
        Symbolic-binding reports.
    policy_reports : Sequence[Mapping[str, object]]
        Policy-composition reports.
    examples : Sequence[Mapping[str, object]]
        Example records.

    Returns
    -------
    dict[str, object]
        A Studio panel for Topos semantic-binding evidence.
    """
    normalised_symbolic = _normalise_topos_validation_reports(
        symbolic_reports,
        schema_name="symbolic_binding_functor",
        label="symbolic binding report",
    )
    normalised_policy = _normalise_topos_validation_reports(
        policy_reports,
        schema_name="policy_composition_category",
        label="policy composition report",
    )
    normalised_examples, example_rows = _normalise_topos_domain_examples(examples)
    all_reports = (*normalised_symbolic, *normalised_policy)
    object_counts = [cast("int", report["object_count"]) for report in all_reports]
    morphism_counts = [cast("int", report["morphism_count"]) for report in all_reports]
    failed_symbolic = [
        cast("str", report["report_hash"])
        for report in normalised_symbolic
        if report["passed"] is not True
    ]
    failed_policy = [
        cast("str", report["report_hash"])
        for report in normalised_policy
        if report["passed"] is not True
    ]
    return {
        "panel_kind": "studio_topos_semantic_binding_panel",
        "proof_surface": "topos_semantic_binding",
        "symbolic_report_count": len(normalised_symbolic),
        "policy_report_count": len(normalised_policy),
        "example_count": len(normalised_examples),
        "passed_symbolic_report_count": len(normalised_symbolic) - len(failed_symbolic),
        "passed_policy_report_count": len(normalised_policy) - len(failed_policy),
        "failed_symbolic_report_hashes": failed_symbolic,
        "failed_policy_report_hashes": failed_policy,
        "proof_boundary": _TOPOS_PROOF_BOUNDARY,
        "non_actuating": True,
        "actuation_permitted": False,
        "formal_proof_claim_permitted": False,
        "symbolic_reports": normalised_symbolic,
        "policy_reports": normalised_policy,
        "example_rows": example_rows,
        "example_domains": tuple(
            sorted({cast("str", example["domain"]) for example in normalised_examples})
        ),
        "object_count_range": {
            "minimum": min(object_counts),
            "maximum": max(object_counts),
        },
        "morphism_count_range": {
            "minimum": min(morphism_counts),
            "maximum": max(morphism_counts),
        },
        "operator_summary": (
            "Topos semantic-binding review: "
            f"{len(normalised_symbolic)} symbolic report(s), "
            f"{len(normalised_policy)} policy report(s), "
            f"{len(normalised_examples)} domain example(s)"
        ),
        "operator_action": (
            "render as categorical validation prototype evidence only; preserve "
            "the proof boundary and require a separate formal-methods gate "
            "before claiming machine-checked Topos proofs or applying policy"
        ),
    }


def build_autopoietic_lineage_studio_panel(
    manifests: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build a passive Studio panel for autopoietic lineage sandbox review.

    Parameters
    ----------
    manifests : Sequence[Mapping[str, object]]
        The manifest records.

    Returns
    -------
    dict[str, object]
        A passive Studio panel for autopoietic lineage sandbox review.
    """
    normalised_manifests = _normalise_autopoietic_lineage_manifests(manifests)
    replay_corpus_rows = tuple(
        row
        for manifest in normalised_manifests
        for row in cast(tuple[dict[str, object], ...], manifest["replay_corpus"])
    )
    replay_domains = tuple(sorted({str(row["domain"]) for row in replay_corpus_rows}))
    child_rows = tuple(
        child
        for manifest in normalised_manifests
        for child in cast(tuple[dict[str, object], ...], manifest["child_candidates"])
    )
    accepted_child_rows = tuple(
        child for child in child_rows if child["status"] == "accepted_for_review"
    )
    rejected_child_rows = tuple(
        child for child in child_rows if child["status"] == "rejected"
    )

    return {
        "panel_kind": "studio_autopoietic_lineage_panel",
        "supervisor": "autopoietic_lineage_sandbox",
        "manifest_count": len(normalised_manifests),
        "claim_boundary": _AUTOPOIETIC_LINEAGE_BOUNDARY,
        "non_actuating": True,
        "execution_disabled": True,
        "operator_review_required": True,
        "hot_patch_permitted": False,
        "live_merge_permitted": False,
        "actuation_permitted": False,
        "lineage_manifests": normalised_manifests,
        "replay_corpus_rows": replay_corpus_rows,
        "replay_domains": replay_domains,
        "replay_domain_count": len(replay_domains),
        "child_candidate_total": len(child_rows),
        "accepted_child_total": len(accepted_child_rows),
        "rejected_child_total": len(rejected_child_rows),
        "accepted_child_rows": accepted_child_rows,
        "rejected_child_rows": rejected_child_rows,
        "operator_summary": (
            "autopoietic lineage review: "
            f"{len(normalised_manifests)} manifest(s), "
            f"{len(replay_domains)} replay domain(s), "
            f"{len(accepted_child_rows)} accepted child candidate(s)"
        ),
        "operator_action": (
            "render as offline lineage sandbox evidence only; compare replay "
            "domains, policy diffs, and blocked reasons before a separately "
            "reviewed inheritance workflow"
        ),
    }


def build_intergenerational_inheritance_studio_panel(
    histories: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build a passive Studio panel for inheritance-history review.

    Parameters
    ----------
    histories : Sequence[Mapping[str, object]]
        Inheritance-history records.

    Returns
    -------
    dict[str, object]
        A passive Studio panel for inheritance-history review.
    """
    normalised_histories = _normalise_intergenerational_inheritance_histories(histories)
    child_rows = tuple(
        child
        for history in normalised_histories
        for child in cast(tuple[dict[str, object], ...], history["child_rows"])
    )
    replay_domains = tuple(
        sorted(
            {
                str(domain)
                for history in normalised_histories
                for domain in cast(tuple[str, ...], history["replay_domains"])
            }
        )
    )
    fitness_scores = tuple(
        float(cast(float, row["fitness_score"])) for row in child_rows
    )

    return {
        "panel_kind": "studio_intergenerational_inheritance_panel",
        "supervisor": "intergenerational_policy_inheritance",
        "history_count": len(normalised_histories),
        "claim_boundary": _INTERGENERATIONAL_HISTORY_BOUNDARY,
        "non_actuating": True,
        "execution_disabled": True,
        "operator_review_required": True,
        "direct_hot_patch_permitted": False,
        "hot_patch_permitted": False,
        "live_merge_permitted": False,
        "actuation_permitted": False,
        "histories": normalised_histories,
        "inheritance_child_rows": child_rows,
        "history_record_total": len(child_rows),
        "signed_metadata_total": sum(
            int(cast(int, history["signed_metadata_count"]))
            for history in normalised_histories
        ),
        "replay_domains": replay_domains,
        "replay_domain_count": len(replay_domains),
        "fitness_range": {
            "minimum": min(fitness_scores),
            "maximum": max(fitness_scores),
        },
        "operator_summary": (
            "intergenerational inheritance review: "
            f"{len(normalised_histories)} history package(s), "
            f"{len(child_rows)} signed child record(s), "
            f"{len(replay_domains)} replay domain(s)"
        ),
        "operator_action": (
            "render as signed inheritance-history evidence only; require "
            "separate operator approval before any reviewed hot-patch workflow"
        ),
    }


def build_evolutionary_supervisor_policy_search_studio_panel(
    reports: Sequence[Mapping[str, object]],
    *,
    examples: Sequence[Mapping[str, object]] = (),
    dsl_reports: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    """Return a Studio panel for offline evolutionary policy-search evidence.

    The panel renders deterministic replay search reports, optional enriched
    domain examples, and optional policy-DSL mutation reports for operator
    review. It validates hashes, candidate counts, replay/STL summaries, DSL
    mutation records, and all disabled execution gates before exposing data to
    Studio. No returned field permits live merge, hot patching, or actuation.

    Parameters
    ----------
    reports : Sequence[Mapping[str, object]]
        The report records.
    examples : Sequence[Mapping[str, object]]
        Example records.
    dsl_reports : Sequence[Mapping[str, object]]
        Policy-DSL search reports.

    Returns
    -------
    dict[str, object]
        A Studio panel for offline evolutionary policy-search evidence.
    """
    normalised_reports = _normalise_evolutionary_search_reports(reports)
    normalised_examples, example_rows = _normalise_evolutionary_examples(examples)
    normalised_dsl_reports = _normalise_evolutionary_dsl_reports(dsl_reports)
    candidate_counts = [
        cast("int", report["candidate_count"]) for report in normalised_reports
    ] + [cast("int", report["candidate_count"]) for report in normalised_dsl_reports]
    accepted_total = sum(
        cast("int", report["accepted_count"]) for report in normalised_reports
    ) + sum(cast("int", report["accepted_count"]) for report in normalised_dsl_reports)
    rejected_total = sum(
        cast("int", report["rejected_count"]) for report in normalised_reports
    ) + sum(cast("int", report["rejected_count"]) for report in normalised_dsl_reports)
    best_rows = [
        report["best_candidate"]
        for report in normalised_reports
        if report["best_candidate"] is not None
    ]
    replay_reward_values = [
        cast(
            "float",
            cast("Mapping[str, object]", report["replay_summary"])["mean_reward"],
        )
        for report in normalised_reports
    ]
    return {
        "panel_kind": "studio_evolutionary_supervisor_policy_search_panel",
        "supervisor": "evolutionary_policy_search",
        "search_report_count": len(normalised_reports),
        "dsl_report_count": len(normalised_dsl_reports),
        "example_count": len(normalised_examples),
        "claim_boundary": _EVOLUTIONARY_SEARCH_BOUNDARY,
        "non_actuating": True,
        "execution_disabled": True,
        "operator_review_required": True,
        "hot_patch_permitted": False,
        "live_merge_permitted": False,
        "actuation_permitted": False,
        "search_reports": normalised_reports,
        "dsl_reports": normalised_dsl_reports,
        "example_rows": example_rows,
        "example_domains": tuple(
            sorted({cast("str", example["domain"]) for example in normalised_examples})
        ),
        "best_candidate_rows": tuple(best_rows),
        "candidate_count_range": {
            "minimum": min(candidate_counts),
            "maximum": max(candidate_counts),
        },
        "accepted_candidate_total": accepted_total,
        "rejected_candidate_total": rejected_total,
        "replay_reward_range": {
            "minimum": min(replay_reward_values),
            "maximum": max(replay_reward_values),
        },
        "operator_summary": (
            "evolutionary policy-search review: "
            f"{len(normalised_reports)} replay report(s), "
            f"{len(normalised_dsl_reports)} DSL report(s), "
            f"{len(normalised_examples)} domain example(s), "
            f"{accepted_total} accepted candidate(s)"
        ),
        "operator_action": (
            "render as offline evolutionary review evidence only; compare replay "
            "reward, STL robustness, candidate rejection reasons, and DSL mutation "
            "rows before any separately reviewed policy merge workflow"
        ),
    }


def build_morphogenetic_field_studio_panel(
    svg_artifact: Mapping[str, object],
) -> dict[str, object]:
    """Return a Studio panel payload for morphogenetic field SVG artefacts.

    The helper renders already-computed topology-field evidence only. It
    validates the dependency-free SVG artefact, preserves the snapshot
    statistics and strongest off-diagonal field edges, and keeps actuation
    disabled for operator review.

    Parameters
    ----------
    svg_artifact : Mapping[str, object]
        The morphogenetic field SVG artefact.

    Returns
    -------
    dict[str, object]
        A Studio panel payload for morphogenetic field SVG artefacts.
    """
    record = _normalise_morphogenetic_field_svg_artifact(svg_artifact)
    snapshot = cast("dict[str, object]", record["snapshot"])
    top_edges = cast("tuple[dict[str, object], ...]", snapshot["top_edges"])
    strongest_edge = top_edges[0] if top_edges else {}
    return {
        "panel_kind": "studio_morphogenetic_field_panel",
        "renderer": "morphogenetic_field_svg",
        "format": "svg",
        "width": record["width"],
        "height": record["height"],
        "shape": snapshot["shape"],
        "snapshot": snapshot,
        "top_edge_count": len(top_edges),
        "strongest_edge": strongest_edge,
        "field_energy": {
            "mean": snapshot["mean"],
            "minimum": snapshot["minimum"],
            "maximum": snapshot["maximum"],
            "l2_norm": snapshot["l2_norm"],
        },
        "svg": record["svg"],
        "actuation_permitted": False,
        "operator_action": (
            "render as passive topology-field evidence; review strongest "
            "off-diagonal edges before any downstream policy action"
        ),
    }


def build_hybrid_order_studio_panel(
    records: Sequence[Mapping[str, object]],
    *,
    scenarios: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    """Return a Studio panel for hybrid classical-quantum order evidence.

    The helper renders local simulator evidence only. It preserves the
    explicit quantum co-simulation claim boundary, validates deterministic
    record hashes, summarises statevector/density-matrix simulator backends,
    and folds deterministic scenario fixtures into candidate review rows.
    Nothing in the payload permits live QPU execution or actuation.

    Parameters
    ----------
    records : Sequence[Mapping[str, object]]
        The records to summarise.
    scenarios : Sequence[Mapping[str, object]]
        Scenario records.

    Returns
    -------
    dict[str, object]
        A Studio panel for hybrid classical-quantum order evidence.
    """
    normalised_records = _normalise_hybrid_order_records(records)
    normalised_scenarios, candidate_rows = _normalise_hybrid_order_scenarios(scenarios)
    entropies = [
        cast("float", record["entanglement_entropy"]) for record in normalised_records
    ]
    normalised_entropies = [
        cast("float", record["normalised_entanglement_entropy"])
        for record in normalised_records
    ]
    participation_ratios = [
        cast("float", record["participation_ratio"]) for record in normalised_records
    ]
    strongest = max(
        normalised_records,
        key=lambda record: cast("float", record["entanglement_entropy"]),
    )
    backends = tuple(
        sorted({cast("str", record["backend"]) for record in normalised_records})
    )
    scenario_domains = tuple(
        sorted({cast("str", scenario["domain"]) for scenario in normalised_scenarios})
    )
    return {
        "panel_kind": "studio_hybrid_order_panel",
        "monitor": "hybrid_entanglement_order_parameter",
        "record_count": len(normalised_records),
        "scenario_count": len(normalised_scenarios),
        "claim_boundary": _HYBRID_ORDER_CLAIM_BOUNDARY,
        "non_actuating": True,
        "execution_disabled": True,
        "actuation_permitted": False,
        "qpu_execution_permitted": False,
        "simulator_backends": backends,
        "scenario_domains": scenario_domains,
        "latest": normalised_records[-1],
        "strongest_entanglement": strongest,
        "series": normalised_records,
        "candidate_rows": candidate_rows,
        "entropy_range": {
            "minimum": min(entropies),
            "maximum": max(entropies),
        },
        "normalised_entanglement_range": {
            "minimum": min(normalised_entropies),
            "maximum": max(normalised_entropies),
        },
        "participation_ratio_range": {
            "minimum": min(participation_ratios),
            "maximum": max(participation_ratios),
        },
        "operator_summary": (
            "hybrid order review: "
            f"{len(normalised_records)} monitor records across "
            f"{len(backends)} local simulator backend(s); "
            f"max entropy {max(entropies):.6g}"
        ),
        "operator_action": (
            "render as local quantum co-simulation evidence only; compare "
            "classical R/Psi with entanglement entropy and keep QPU execution, "
            "actuation, and backend promotion behind separate evidence gates"
        ),
    }


def build_multiverse_counterfactual_studio_panel(
    manifest: Mapping[str, object],
    risk_report: Mapping[str, object],
) -> dict[str, object]:
    """Return a Studio panel payload for multiverse branch review evidence.

    The panel joins a non-actuating rollout manifest with a non-actuating risk
    gate report. It validates both audit artefacts before rendering branch
    comparison rows, safest-branch metadata, and coherence ranges for operator
    review. The helper never emits executable actions.

    Parameters
    ----------
    manifest : Mapping[str, object]
        The manifest object.
    risk_report : Mapping[str, object]
        The multiverse risk report mapping.

    Returns
    -------
    dict[str, object]
        A Studio panel payload for multiverse branch review evidence.
    """
    rollout = _normalise_multiverse_manifest(manifest)
    risk = _normalise_multiverse_risk_report(risk_report)
    branch_rows = _join_multiverse_branch_rows(rollout, risk)
    final_values = [cast("float", row["final_R"]) for row in branch_rows]
    mean_values = [cast("float", row["mean_R"]) for row in branch_rows]
    min_values = [cast("float", row["min_R"]) for row in branch_rows]
    max_values = [cast("float", row["max_R"]) for row in branch_rows]
    rejected_ids = [
        cast("str", row["branch_id"])
        for row in branch_rows
        if row["risk_approved"] is not True
    ]
    return {
        "panel_kind": "studio_multiverse_counterfactual_panel",
        "simulator": "multiverse_counterfactual",
        "risk_gate": "multiverse_branch_risk_gate",
        "schema_version": rollout["schema_version"],
        "risk_schema_version": risk["schema_version"],
        "backend": rollout["backend"],
        "horizon": rollout["horizon"],
        "branch_count": rollout["branch_count"],
        "approved_count": risk["approved_count"],
        "rejected_count": risk["rejected_count"],
        "safest_branch_id": risk["safest_branch_id"],
        "safest_branch_hash": risk["safest_branch_hash"],
        "rejected_branch_ids": rejected_ids,
        "rejection_reasons": risk["rejection_reasons"],
        "manifest_hash": rollout["manifest_hash"],
        "risk_report_hash": risk["report_hash"],
        "claim_boundary": _MULTIVERSE_ROLLOUT_CLAIM_BOUNDARY,
        "risk_claim_boundary": _MULTIVERSE_RISK_CLAIM_BOUNDARY,
        "non_actuating": True,
        "execution_disabled": True,
        "actuation_permitted": False,
        "branch_rows": branch_rows,
        "coherence_range": {
            "minimum": min(min_values),
            "maximum": max(max_values),
            "final_minimum": min(final_values),
            "final_maximum": max(final_values),
            "mean_minimum": min(mean_values),
            "mean_maximum": max(mean_values),
        },
        "operator_summary": (
            "multiverse branch review: "
            f"{risk['approved_count']}/{rollout['branch_count']} branches approved"
        ),
        "operator_action": (
            "render as counterfactual review evidence only; no branch action "
            "may be applied without a separate safety-gated control workflow"
        ),
    }


def build_layer_table(spec: BindingSpec) -> tuple[dict[str, object], ...]:
    """Return editable layer rows for the Studio oscillator canvas.

    Parameters
    ----------
    spec : BindingSpec
        The binding specification.

    Returns
    -------
    tuple[dict[str, object], ...]
        Editable layer rows for the Studio oscillator canvas.
    """
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
    """Return oscillator rows suitable for Streamlit data editing.

    Parameters
    ----------
    spec : BindingSpec
        The binding specification.

    Returns
    -------
    tuple[dict[str, object], ...]
        Oscillator rows suitable for Streamlit data editing.
    """
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
    """Return a deterministic layer/coupling graph for Studio canvas review.

    Parameters
    ----------
    spec : BindingSpec
        The binding specification.

    Returns
    -------
    dict[str, object]
        A deterministic layer/coupling graph for Studio canvas review.
    """
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
    """Build a review artefact from edited Studio canvas graph rows.

    Parameters
    ----------
    before_graph : Mapping[str, object]
        The canvas graph before the change.
    after_graph : Mapping[str, object]
        The edited canvas graph after the change.

    Returns
    -------
    ExportManifest
        A review artefact from edited Studio canvas graph rows.
    """
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
    """Build a deterministic canvas layout manifest from node positions.

    Parameters
    ----------
    project_name : str
        Name of the project.
    graph : Mapping[str, object]
        The canvas graph mapping.

    Returns
    -------
    ExportManifest
        A deterministic canvas layout manifest from node positions.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    """
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
    """Build a review patch for persistent Studio topology edits.

    Parameters
    ----------
    project_name : str
        Name of the project.
    before_graph : Mapping[str, object]
        The canvas graph before the change.
    after_graph : Mapping[str, object]
        The edited canvas graph after the change.

    Returns
    -------
    ExportManifest
        A review patch for persistent Studio topology edits.
    """
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
    """Summarise Canvas browser controls for deterministic operator feedback.

    Parameters
    ----------
    canvas_artifact : ExportManifest
        The canvas edit artefact manifest.
    canvas_layout : ExportManifest
        The canvas layout manifest.
    canvas_patch : ExportManifest
        The canvas topology patch manifest.
    canvas_rewrite : Mapping[str, object]
        The canvas binding-rewrite candidate.
    operator_signoff : bool
        Whether the operator has signed off.

    Returns
    -------
    dict[str, object]
        Canvas browser controls for deterministic operator feedback.
    """
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
    """Build validated binding YAML candidate from reviewed canvas edits.

    Parameters
    ----------
    result : StudioReplayResult
        The Studio replay result.
    after_graph : Mapping[str, object]
        The edited canvas graph after the change.

    Returns
    -------
    dict[str, object]
        Validated binding YAML candidate from reviewed canvas edits.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    """
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
        "coupling_count_before": _canvas_graph_count(result, "edge_count"),
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
    """Apply a reviewed canvas binding candidate with hash and validation gates.

    Parameters
    ----------
    candidate : Mapping[str, object]
        The binding rewrite candidate mapping.
    binding_spec_path : str | Path
        Path to the binding-spec file.
    operator_signoff : bool
        Whether the operator has signed off.
    create_backup : bool
        Whether to write a backup before applying.

    Returns
    -------
    dict[str, object]
        A reviewed canvas binding candidate with hash and validation gates.
    """
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
    """Return non-opening connector ownership guidance for Studio.

    Parameters
    ----------
    spec : BindingSpec
        The binding specification.

    Returns
    -------
    dict[str, object]
        Non-opening connector ownership guidance for Studio.
    """
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
    """Return a gated live-connector execution record without opening transport.

    Parameters
    ----------
    connector_plan : Mapping[str, object]
        The live-connector plan mapping.
    transport : str
        Transport identifier.
    payload : Mapping[str, object]
        The payload mapping or bytes.
    dry_run : bool
        Whether to run without opening transport.

    Returns
    -------
    dict[str, object]
        A gated live-connector execution record without opening transport.
    """
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
    """Validate an owned live connector boundary without opening transport.

    Parameters
    ----------
    result : StudioReplayResult
        The Studio replay result.
    transport : str
        Transport identifier.
    owner : str
        Owner of the connector boundary.
    auth_policy : Mapping[str, object]
        The connector authentication policy.
    payload : Mapping[str, object]
        The payload mapping or bytes.
    sequence : int
        Monotonic sequence number.
    capability : str
        The sync capability identifier.
    direction : str
        Sync direction (e.g. ``inbound``/``outbound``).

    Returns
    -------
    dict[str, object]
        An owned live connector boundary without opening transport.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    """
    if not isinstance(result, StudioReplayResult):
        raise ValueError("replay result must be a StudioReplayResult")
    checked_transport = _require_non_empty_text(transport, "transport")
    checked_payload = _normalise_json_mapping(
        cast("Mapping[object, object]", payload),
        "payload",
    )
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
        "queued_count": _mapping_count(adapter_record, "queued_count"),
    }


def build_hardware_target_package(result: StudioReplayResult) -> dict[str, object]:
    """Return a review-only hardware target package for Studio.

    Parameters
    ----------
    result : StudioReplayResult
        The Studio replay result.

    Returns
    -------
    dict[str, object]
        A review-only hardware target package for Studio.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    """
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
    """Return a verified hardware package only when evidence is complete.

    Parameters
    ----------
    result : StudioReplayResult
        The Studio replay result.
    evidence : Mapping[str, object]
        Verification evidence mapping.

    Returns
    -------
    dict[str, object]
        A verified hardware package only when evidence is complete.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    """
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
        "targets": list(_require_sequence(base_package.get("targets"), "targets")),
        "required_evidence": list(
            _require_sequence(
                base_package.get("required_evidence"),
                "required_evidence",
            )
        ),
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
        "export_artifacts": list(
            _require_sequence(
                base_package.get("export_artifacts"),
                "export_artifacts",
            )
        ),
    }


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


def build_export_manifests(
    *,
    project_name: str,
    binding_yaml: str,
    audit_payload: Mapping[str, object],
    validation_errors: Sequence[str],
) -> tuple[ExportManifest, ...]:
    """Build review-only export manifests for Studio.

    Parameters
    ----------
    project_name : str
        Name of the project.
    binding_yaml : str
        The binding spec serialised as YAML.
    audit_payload : Mapping[str, object]
        The audit payload mapping.
    validation_errors : Sequence[str]
        Binding validation error messages.

    Returns
    -------
    tuple[ExportManifest, ...]
        Review-only export manifests for Studio.
    """
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
    """Return target-specific deployment readiness guidance for Studio.

    Parameters
    ----------
    project_state : StudioProjectState
        The Studio project state.

    Returns
    -------
    dict[str, object]
        Target-specific deployment readiness guidance for Studio.
    """
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
    """Return a deterministic deployment package manifest for Studio.

    Parameters
    ----------
    project_state : StudioProjectState
        The Studio project state.

    Returns
    -------
    dict[str, object]
        A deterministic deployment package manifest for Studio.
    """
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
    """Return localhost-only service process packaging for Studio deployment.

    Parameters
    ----------
    project_state : StudioProjectState
        The Studio project state.

    Returns
    -------
    dict[str, object]
        Localhost-only service process packaging for Studio deployment.
    """
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
    """Return ordered, operator-invoked package materialisation commands.

    Parameters
    ----------
    project_state : StudioProjectState
        The Studio project state.

    Returns
    -------
    dict[str, object]
        Ordered, operator-invoked package materialisation commands.
    """
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
        "blocked_targets": list(
            _require_sequence(package.get("blocked_targets"), "blocked_targets")
        ),
        "blocked_reasons": list(
            _require_sequence(package.get("blocked_reasons"), "blocked_reasons")
        ),
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
        "required_artifacts": list(
            _require_sequence(package.get("required_artifacts"), "required_artifacts")
        ),
        "safety_gates": list(
            _require_sequence(package.get("safety_gates"), "safety_gates")
        ),
    }


def build_operator_checklist(
    project_state: StudioProjectState,
) -> tuple[dict[str, object], ...]:
    """Return beginner-friendly ordered deployment steps for Studio.

    Parameters
    ----------
    project_state : StudioProjectState
        The Studio project state.

    Returns
    -------
    tuple[dict[str, object], ...]
        Beginner-friendly ordered deployment steps for Studio.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    """
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
    for target in _require_sequence(readiness.get("targets"), "targets"):
        if not isinstance(target, Mapping):
            raise ValueError("readiness targets must be mappings")
        target_name = _require_non_empty_text(target.get("target"), "target")
        status = _require_non_empty_text(target.get("status"), "status")
        operator_action = _require_non_empty_text(
            target.get("operator_action"),
            "operator_action",
        )
        blocked_detail = "; ".join(
            str(reason)
            for reason in _require_sequence(
                target.get("blocked_reasons", ()),
                "blocked_reasons",
            )
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
    """Return copyable deployment-review commands for ready targets.

    Parameters
    ----------
    project_state : StudioProjectState
        The Studio project state.

    Returns
    -------
    tuple[dict[str, object], ...]
        Copyable deployment-review commands for ready targets.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    """
    readiness = build_deployment_readiness(project_state)
    rows: list[dict[str, object]] = []
    for target in _require_sequence(readiness.get("targets"), "targets"):
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


def build_oscillator_edit_artifact(
    before_rows: Sequence[Mapping[str, object]],
    after_rows: Sequence[Mapping[str, object]],
) -> ExportManifest:
    """Build a review artefact from edited oscillator table rows.

    Parameters
    ----------
    before_rows : Sequence[Mapping[str, object]]
        The table rows before the change.
    after_rows : Sequence[Mapping[str, object]]
        The edited table rows after the change.

    Returns
    -------
    ExportManifest
        A review artefact from edited oscillator table rows.
    """
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
    """Return reasons deploy-like exports must stay review-only.

    Parameters
    ----------
    validation_errors : Sequence[str]
        Binding validation error messages.

    Returns
    -------
    tuple[str, ...]
        Reasons deploy-like exports must stay review-only.
    """
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
    """Run a local binding-spec replay and return Studio-ready payloads.

    Parameters
    ----------
    spec_path : Path
        Filesystem path to the binding-spec file.
    steps : int
        Number of replay steps.
    knobs : StudioKnobState
        The Studio knob state.

    Returns
    -------
    StudioReplayResult
        A local binding-spec replay and return Studio-ready payloads.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    """
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


def _normalise_integrated_information_records(
    records: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    if isinstance(records, str | bytes) or not isinstance(records, Sequence):
        raise ValueError("integrated-information records must be a sequence")
    if not records:
        raise ValueError("integrated-information records must not be empty")
    normalised: list[dict[str, object]] = []
    for index, record in enumerate(records, 1):
        if not isinstance(record, Mapping):
            raise ValueError("integrated-information records must be mappings")
        monitor = record.get("monitor", "integrated_information")
        if monitor != "integrated_information":
            raise ValueError("integrated-information monitor tag is required")
        claim_boundary = _require_non_empty_text(
            record.get("claim_boundary"),
            "claim_boundary",
        )
        if claim_boundary != "engineering_proxy_not_theoretical_iit":
            raise ValueError("integrated-information claim boundary is required")
        n_bins = _positive_int(record.get("n_bins"), "n_bins", minimum=2)
        phi = _bounded_information_scalar(record.get("phi"), "phi", n_bins)
        normalised_phi = _unit_interval_number(
            record.get("normalised_phi"),
            "normalised_phi",
        )
        expected_normalised_phi = min(1.0, phi / float(np.log(n_bins)))
        if abs(normalised_phi - expected_normalised_phi) > 1e-12:
            raise ValueError("normalised_phi must match phi/log(n_bins)")
        total_integration = _bounded_information_scalar(
            record.get("total_integration"),
            "total_integration",
            n_bins,
        )
        if phi > total_integration + 1e-12:
            raise ValueError("phi must not exceed total_integration")
        minimum_partition = _normalise_integrated_information_partition(
            record.get("minimum_partition"),
        )
        pairwise_shape = _integrated_information_pairwise_shape(
            record.get("pairwise_mi"),
            n_bins,
        )
        if pairwise_shape is not None:
            partition_nodes = {node for side in minimum_partition for node in side}
            if partition_nodes != set(range(pairwise_shape[0])):
                raise ValueError("minimum_partition must cover pairwise_mi nodes")
        normalised.append(
            {
                "step": index,
                "phi": phi,
                "normalised_phi": normalised_phi,
                "total_integration": total_integration,
                "minimum_partition": [list(side) for side in minimum_partition],
                "n_bins": n_bins,
                "claim_boundary": claim_boundary,
            }
        )
    return tuple(normalised)


def _normalise_strange_loop_records(
    records: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    if isinstance(records, str | bytes) or not isinstance(records, Sequence):
        raise ValueError("strange-loop records must be a sequence")
    if not records:
        raise ValueError("strange-loop records must not be empty")
    normalised: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("strange-loop records must be mappings")
        expected_trigger = _require_non_empty_text(
            record.get("expected_trigger"),
            "expected_trigger",
        )
        if expected_trigger not in _STRANGE_LOOP_TRIGGERS:
            raise ValueError("expected_trigger is not supported")
        claim_boundary = _require_non_empty_text(
            record.get("claim_boundary"),
            "claim_boundary",
        )
        if claim_boundary != _STRANGE_LOOP_CLAIM_BOUNDARY:
            raise ValueError("strange-loop claim boundary is required")
        if record.get("non_actuating") is not True:
            raise ValueError("non_actuating must be true")
        if record.get("execution_disabled") is not True:
            raise ValueError("execution_disabled must be true")
        passed = record.get("passed_expected_trigger")
        if not isinstance(passed, bool):
            raise ValueError("passed_expected_trigger must be boolean")
        final_knobs = _normalise_text_sequence(
            record.get("final_recommended_knobs", ()),
            "final_recommended_knobs",
        )
        normalised.append(
            {
                "domain": _require_non_empty_text(record.get("domain"), "domain"),
                "scenario_id": _require_non_empty_text(
                    record.get("scenario_id"),
                    "scenario_id",
                ),
                "expected_trigger": expected_trigger,
                "step_count": _positive_int(
                    record.get("step_count"),
                    "step_count",
                    minimum=1,
                ),
                "max_drift_score": _non_negative_float(
                    record.get("max_drift_score"),
                    "max_drift_score",
                ),
                "max_oscillation_score": _non_negative_float(
                    record.get("max_oscillation_score"),
                    "max_oscillation_score",
                ),
                "max_overcontrol_score": _non_negative_float(
                    record.get("max_overcontrol_score"),
                    "max_overcontrol_score",
                ),
                "min_control_coherence": _unit_interval_number(
                    record.get("min_control_coherence"),
                    "min_control_coherence",
                ),
                "triggered_recommendation_count": _non_negative_int(
                    record.get("triggered_recommendation_count"),
                    "triggered_recommendation_count",
                ),
                "final_recommended_knobs": final_knobs,
                "passed_expected_trigger": passed,
                "scenario_hash": _require_sha256_hex(
                    record.get("scenario_hash"),
                    "scenario_hash",
                ),
                "result_hash": _require_sha256_hex(
                    record.get("result_hash"),
                    "result_hash",
                ),
                "non_actuating": True,
                "execution_disabled": True,
                "claim_boundary": claim_boundary,
            }
        )
    return tuple(normalised)


def _normalise_information_geometry_records(
    records: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    if isinstance(records, Mapping) or not isinstance(records, Sequence) or not records:
        raise ValueError("information-geometry records must be a non-empty sequence")
    normalised: list[dict[str, object]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError("information-geometry record must be a mapping")
        label = f"information-geometry record {index}"
        claim_boundary = _require_non_empty_text(
            record.get("claim_boundary"), f"{label} claim_boundary"
        )
        if claim_boundary != _INFORMATION_GEOMETRY_CLAIM_BOUNDARY:
            raise ValueError(f"{label} claim boundary is not review-safe")
        if record.get("non_actuating") is not True:
            raise ValueError(f"{label} non_actuating must be true")
        if record.get("execution_disabled") is not True:
            raise ValueError(f"{label} execution_disabled must be true")
        backend = _require_non_empty_text(record.get("backend"), f"{label} backend")
        if backend not in _INFORMATION_GEOMETRY_BACKENDS:
            raise ValueError(f"{label} backend is not supported")
        state_raw = record.get("state")
        if not isinstance(state_raw, Mapping):
            raise ValueError(f"{label} state must be a mapping")
        simplex = _normalise_simplex_sequence(
            state_raw.get("simplex_coordinates"),
            f"{label} simplex_coordinates",
        )
        target = _normalise_simplex_sequence(
            state_raw.get("target_coordinates"),
            f"{label} target_coordinates",
        )
        if len(simplex) != len(target):
            raise ValueError(f"{label} target_coordinates must match simplex shape")
        metric_tensor = _normalise_square_float_matrix(
            state_raw.get("metric_tensor"),
            f"{label} metric_tensor",
            expected_size=len(simplex),
            positive_diagonal=True,
        )
        tangent_vector = _normalise_float_sequence(
            state_raw.get("tangent_vector"),
            f"{label} tangent_vector",
        )
        if len(tangent_vector) != len(simplex):
            raise ValueError(f"{label} tangent_vector must match simplex shape")
        geodesic_length = _non_negative_float(
            state_raw.get("geodesic_length"),
            f"{label} geodesic_length",
        )
        curvature_proxy = _non_negative_float(
            record.get("curvature_proxy"),
            f"{label} curvature_proxy",
        )
        state_curvature = _non_negative_float(
            state_raw.get("curvature_proxy"),
            f"{label} state.curvature_proxy",
        )
        if abs(state_curvature - curvature_proxy) > 1e-12:
            raise ValueError(f"{label} state curvature_proxy must match proposal")
        fisher_rao = _non_negative_float(
            record.get("fisher_rao_distance"),
            f"{label} fisher_rao_distance",
        )
        if abs(geodesic_length - fisher_rao) > 1e-12:
            raise ValueError(f"{label} geodesic_length must match Fisher-Rao distance")
        normalised.append(
            {
                "step": index + 1,
                "backend": backend,
                "claim_boundary": claim_boundary,
                "non_actuating": True,
                "execution_disabled": True,
                "proposal_hash": _require_sha256_hex(
                    record.get("proposal_hash"),
                    f"{label} proposal_hash",
                ),
                "knob": _single_information_geometry_action(record, label)["knob"],
                "scope": _single_information_geometry_action(record, label)["scope"],
                "action_value": _single_information_geometry_action(record, label)[
                    "value"
                ],
                "ttl_s": _single_information_geometry_action(record, label)["ttl_s"],
                "justification": _single_information_geometry_action(record, label)[
                    "justification"
                ],
                "fisher_rao_distance": fisher_rao,
                "wasserstein_distance": _non_negative_float(
                    record.get("wasserstein_distance"),
                    f"{label} wasserstein_distance",
                ),
                "natural_gradient_norm": _non_negative_float(
                    record.get("natural_gradient_norm"),
                    f"{label} natural_gradient_norm",
                ),
                "curvature_proxy": curvature_proxy,
                "simplex_coordinates": list(simplex),
                "target_coordinates": list(target),
                "metric_tensor": metric_tensor,
                "tangent_vector": list(tangent_vector),
                "geodesic_length": geodesic_length,
            }
        )
    return tuple(normalised)


def _normalise_information_geometry_scenarios(
    scenarios: Sequence[Mapping[str, object]],
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    if isinstance(scenarios, Mapping) or not isinstance(scenarios, Sequence):
        raise ValueError("information-geometry scenarios must be a sequence")
    normalised_scenarios: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, Mapping):
            raise ValueError("information-geometry scenario must be a mapping")
        label = f"information-geometry scenario {index}"
        claim_boundary = _require_non_empty_text(
            scenario.get("claim_boundary"), f"{label} claim_boundary"
        )
        if claim_boundary != _INFORMATION_GEOMETRY_CLAIM_BOUNDARY:
            raise ValueError(f"{label} claim boundary is not review-safe")
        if scenario.get("non_actuating") is not True:
            raise ValueError(f"{label} non_actuating must be true")
        if scenario.get("execution_disabled") is not True:
            raise ValueError(f"{label} execution_disabled must be true")
        current = _normalise_simplex_sequence(
            scenario.get("current_distribution"),
            f"{label} current_distribution",
        )
        target = _normalise_simplex_sequence(
            scenario.get("target_distribution"),
            f"{label} target_distribution",
        )
        if len(current) != len(target):
            raise ValueError(f"{label} target_distribution must match current shape")
        objectives = _normalise_text_sequence(
            scenario.get("objective_labels"),
            f"{label} objective_labels",
        )
        knob_hints = _normalise_text_sequence(
            scenario.get("knob_hints"),
            f"{label} knob_hints",
        )
        control_gradient = _normalise_information_geometry_gradient(
            scenario.get("control_gradient"),
            f"{label} control_gradient",
        )
        scenario_id = _require_non_empty_text(
            scenario.get("scenario_id"),
            f"{label} scenario_id",
        )
        domain = _require_non_empty_text(scenario.get("domain"), f"{label} domain")
        scenario_hash = _require_sha256_hex(
            scenario.get("scenario_hash"),
            f"{label} scenario_hash",
        )
        max_step = _positive_float(scenario.get("max_step"), f"{label} max_step")
        normalised_scenario = {
            "domain": domain,
            "scenario_id": scenario_id,
            "scenario_hash": scenario_hash,
            "claim_boundary": claim_boundary,
            "non_actuating": True,
            "execution_disabled": True,
            "objective_labels": list(objectives),
            "control_gradient": [
                {"knob": knob, "value": value} for knob, value in control_gradient
            ],
            "knob_hints": list(knob_hints),
            "max_step": max_step,
            "dimension": len(current),
        }
        normalised_scenarios.append(normalised_scenario)
        candidate_rows.append(
            {
                "domain": domain,
                "scenario_id": scenario_id,
                "scenario_hash": scenario_hash,
                "dimension": len(current),
                "objective_count": len(objectives),
                "control_knobs": tuple(knob for knob, _ in control_gradient),
                "max_step": max_step,
            }
        )
    return (tuple(normalised_scenarios), tuple(candidate_rows))


def _normalise_sheaf_cohomology_records(
    records: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    if isinstance(records, Mapping) or not isinstance(records, Sequence) or not records:
        raise ValueError("sheaf-cohomology records must be a non-empty sequence")
    normalised: list[dict[str, object]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError("sheaf-cohomology record must be a mapping")
        label = f"sheaf-cohomology record {index}"
        method = _require_non_empty_text(record.get("method"), f"{label} method")
        if method != _SHEAF_RESULT_METHOD:
            raise ValueError(f"{label} method is unsupported")
        laplacian_shape = _normalise_sheaf_shape(
            record.get("laplacian_shape"),
            f"{label} laplacian_shape",
            expected_rank=2,
        )
        residual_shape = _normalise_sheaf_shape(
            record.get("residual_shape"),
            f"{label} residual_shape",
            expected_rank=3,
        )
        if laplacian_shape[0] != laplacian_shape[1]:
            raise ValueError(f"{label} laplacian_shape must be square")
        if residual_shape[0] != residual_shape[1]:
            raise ValueError(f"{label} residual_shape must be node-square")
        if residual_shape[0] * residual_shape[2] != laplacian_shape[0]:
            raise ValueError(f"{label} residual_shape must match laplacian_shape")
        normalised.append(
            {
                "method": _SHEAF_RESULT_METHOD,
                "obstruction_score": _non_negative_float(
                    record.get("obstruction_score"),
                    f"{label} obstruction_score",
                ),
                "consistency_energy": _non_negative_float(
                    record.get("consistency_energy"),
                    f"{label} consistency_energy",
                ),
                "kernel_dimension": _non_negative_int(
                    record.get("kernel_dimension"),
                    f"{label} kernel_dimension",
                ),
                "obstruction_dimension": _non_negative_int(
                    record.get("obstruction_dimension"),
                    f"{label} obstruction_dimension",
                ),
                "edge_count": _non_negative_int(
                    record.get("edge_count"),
                    f"{label} edge_count",
                ),
                "laplacian_shape": laplacian_shape,
                "residual_shape": residual_shape,
                "tolerance": _non_negative_float(
                    record.get("tolerance"),
                    f"{label} tolerance",
                ),
            }
        )
    return tuple(normalised)


def _normalise_sheaf_obstruction_summaries(
    summaries: Sequence[Mapping[str, object]],
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    if (
        isinstance(summaries, Mapping)
        or not isinstance(summaries, Sequence)
        or not summaries
    ):
        raise ValueError("sheaf-obstruction summaries must be a non-empty sequence")
    normalised: list[dict[str, object]] = []
    residual_rows: list[dict[str, object]] = []
    for index, summary in enumerate(summaries):
        if not isinstance(summary, Mapping):
            raise ValueError("sheaf-obstruction summary must be a mapping")
        label = f"sheaf-obstruction summary {index}"
        severity = _require_non_empty_text(summary.get("severity"), f"{label} severity")
        if severity not in {"nominal", "warning", "critical"}:
            raise ValueError(f"{label} severity is unsupported")
        warning_threshold = _non_negative_float(
            summary.get("warning_threshold"),
            f"{label} warning_threshold",
        )
        critical_threshold = _non_negative_float(
            summary.get("critical_threshold"),
            f"{label} critical_threshold",
        )
        if critical_threshold < warning_threshold:
            raise ValueError(f"{label} critical_threshold must be >= warning_threshold")
        top_edges = _normalise_sheaf_residual_rows(
            summary.get("top_residual_edges"),
            label,
        )
        obstruction_score = _non_negative_float(
            summary.get("obstruction_score"),
            f"{label} obstruction_score",
        )
        normalised.append(
            {
                "severity": severity,
                "obstruction_score": obstruction_score,
                "warning_threshold": warning_threshold,
                "critical_threshold": critical_threshold,
                "top_residual_edges": top_edges,
            }
        )
        for row in top_edges:
            residual_rows.append({"summary_index": index, "severity": severity, **row})
    return tuple(normalised), tuple(residual_rows)


def _normalise_sheaf_control_proposals(
    proposals: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    if (
        isinstance(proposals, Mapping)
        or not isinstance(proposals, Sequence)
        or not proposals
    ):
        raise ValueError("sheaf-control proposals must be a non-empty sequence")
    normalised: list[dict[str, object]] = []
    for index, proposal in enumerate(proposals):
        if not isinstance(proposal, Mapping):
            raise ValueError("sheaf-control proposal must be a mapping")
        label = f"sheaf-control proposal {index}"
        method = _require_non_empty_text(proposal.get("method"), f"{label} method")
        if method != _SHEAF_CONTROL_METHOD:
            raise ValueError(f"{label} method is unsupported")
        if proposal.get("non_actuating") is not True:
            raise ValueError(f"{label} non_actuating must be true")
        if proposal.get("execution_disabled") is not True:
            raise ValueError(f"{label} execution_disabled must be true")
        if proposal.get("operator_review_required") is not True:
            raise ValueError(f"{label} operator_review_required must be true")
        accepted = _strict_bool(
            proposal.get("accepted_for_review"),
            f"{label} accepted_for_review",
        )
        baseline_obstruction = _non_negative_float(
            proposal.get("baseline_obstruction_score"),
            f"{label} baseline_obstruction_score",
        )
        projected_obstruction = _non_negative_float(
            proposal.get("projected_obstruction_score"),
            f"{label} projected_obstruction_score",
        )
        baseline_energy = _non_negative_float(
            proposal.get("baseline_consistency_energy"),
            f"{label} baseline_consistency_energy",
        )
        projected_energy = _non_negative_float(
            proposal.get("projected_consistency_energy"),
            f"{label} projected_consistency_energy",
        )
        if accepted and projected_obstruction > baseline_obstruction + 1e-12:
            raise ValueError(f"{label} projected obstruction must be monotone")
        if accepted and projected_energy > baseline_energy + 1e-12:
            raise ValueError(f"{label} projected energy must be monotone")
        update_norm = _non_negative_float(
            proposal.get("update_norm"),
            f"{label} update_norm",
        )
        max_update_norm = _non_negative_float(
            proposal.get("max_update_norm"),
            f"{label} max_update_norm",
        )
        if update_norm > max_update_norm + 1e-12:
            raise ValueError(f"{label} update_norm exceeds max_update_norm")
        blocked_reasons = _normalise_text_sequence(
            proposal.get("blocked_reasons", ()),
            f"{label} blocked_reasons",
        )
        if accepted and blocked_reasons:
            raise ValueError(f"{label} accepted proposal must not be blocked")
        if not accepted and not blocked_reasons:
            raise ValueError(f"{label} rejected proposal requires blocked_reasons")
        normalised.append(
            {
                "method": _SHEAF_CONTROL_METHOD,
                "baseline_obstruction_score": baseline_obstruction,
                "projected_obstruction_score": projected_obstruction,
                "baseline_consistency_energy": baseline_energy,
                "projected_consistency_energy": projected_energy,
                "cohomology_dimensions": _normalise_sheaf_cohomology_dimensions(
                    proposal.get("cohomology_dimensions"),
                    label,
                ),
                "recommended_update_shape": _normalise_sheaf_shape(
                    proposal.get("recommended_update_shape"),
                    f"{label} recommended_update_shape",
                    expected_rank=2,
                ),
                "projected_node_state_shape": _normalise_sheaf_shape(
                    proposal.get("projected_node_state_shape"),
                    f"{label} projected_node_state_shape",
                    expected_rank=2,
                ),
                "update_norm": update_norm,
                "step_size": _positive_float(
                    proposal.get("step_size"),
                    f"{label} step_size",
                ),
                "max_update_norm": max_update_norm,
                "accepted_for_review": accepted,
                "non_actuating": True,
                "execution_disabled": True,
                "operator_review_required": True,
                "blocked_reasons": blocked_reasons,
            }
        )
    return tuple(normalised)


def _normalise_sheaf_residual_rows(
    value: object,
    label: str,
) -> tuple[dict[str, object], ...]:
    if isinstance(value, Mapping) or not isinstance(value, Sequence):
        raise ValueError(f"{label} top_residual_edges must be a sequence")
    rows: list[dict[str, object]] = []
    for index, raw_row in enumerate(value):
        row_label = f"{label} top_residual_edges[{index}]"
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"{row_label} must be a mapping")
        residual = _normalise_float_sequence(
            raw_row.get("residual"),
            f"{row_label} residual",
        )
        if not residual:
            raise ValueError(f"{row_label} residual must not be empty")
        rows.append(
            {
                "target": _non_negative_int(
                    raw_row.get("target"),
                    f"{row_label} target",
                ),
                "source": _non_negative_int(
                    raw_row.get("source"),
                    f"{row_label} source",
                ),
                "norm": _non_negative_float(
                    raw_row.get("norm"),
                    f"{row_label} norm",
                ),
                "residual": residual,
            }
        )
    return tuple(rows)


def _normalise_sheaf_cohomology_dimensions(
    value: object,
    label: str,
) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} cohomology_dimensions must be a mapping")
    return {
        key: _non_negative_int(value.get(key), f"{label} {key}")
        for key in (
            "baseline_kernel_dimension",
            "projected_kernel_dimension",
            "baseline_obstruction_dimension",
            "projected_obstruction_dimension",
        )
    }


def _normalise_sheaf_shape(
    value: object,
    label: str,
    *,
    expected_rank: int,
) -> tuple[int, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a sequence")
    shape = tuple(
        _positive_int(item, f"{label}[{index}]", minimum=1)
        for index, item in enumerate(value)
    )
    if len(shape) != expected_rank:
        raise ValueError(f"{label} must have rank {expected_rank}")
    return shape


def _strict_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _normalise_topos_validation_reports(
    reports: Sequence[Mapping[str, object]],
    *,
    schema_name: str,
    label: str,
) -> tuple[dict[str, object], ...]:
    if schema_name not in _TOPOS_REPORT_SCHEMAS:
        raise ValueError("Topos report schema is not supported")
    if isinstance(reports, Mapping) or not isinstance(reports, Sequence) or not reports:
        raise ValueError(f"{label}s must be a non-empty sequence")
    normalised: list[dict[str, object]] = []
    for index, report in enumerate(reports):
        if not isinstance(report, Mapping):
            raise ValueError(f"{label} must be a mapping")
        item_label = f"{label} {index}"
        if report.get("schema_name") != schema_name:
            raise ValueError(f"{item_label} schema_name must be {schema_name}")
        proof_boundary = _require_non_empty_text(
            report.get("proof_boundary"),
            f"{item_label} proof_boundary",
        )
        if proof_boundary != _TOPOS_PROOF_BOUNDARY:
            raise ValueError(f"{item_label} proof boundary is not review-safe")
        if report.get("non_actuating") is not True:
            raise ValueError(f"{item_label} non_actuating must be true")
        object_count = _positive_int(
            report.get("object_count"),
            f"{item_label} object_count",
            minimum=0,
        )
        morphism_count = _positive_int(
            report.get("morphism_count"),
            f"{item_label} morphism_count",
            minimum=0,
        )
        obligations = _normalise_topos_obligations(
            report.get("obligation_records"),
            f"{item_label} obligation_records",
        )
        objects = _normalise_topos_named_records(
            report.get("objects"),
            f"{item_label} objects",
        )
        morphisms = _normalise_topos_morphisms(
            report.get("morphisms"),
            f"{item_label} morphisms",
        )
        if len(objects) != object_count:
            raise ValueError(f"{item_label} object_count must match objects length")
        if len(morphisms) != morphism_count:
            raise ValueError(f"{item_label} morphism_count must match morphisms length")
        normalised.append(
            {
                "schema_name": schema_name,
                "schema_version": _require_non_empty_text(
                    report.get("schema_version"),
                    f"{item_label} schema_version",
                ),
                "object_count": object_count,
                "morphism_count": morphism_count,
                "obligation_records": obligations,
                "objects": objects,
                "morphisms": morphisms,
                "passed": _required_bool(report.get("passed"), f"{item_label} passed"),
                "report_hash": _require_sha256_hex(
                    report.get("report_hash"),
                    f"{item_label} report_hash",
                ),
                "proof_boundary": _TOPOS_PROOF_BOUNDARY,
                "non_actuating": True,
            }
        )
    return tuple(normalised)


def _normalise_topos_obligations(
    value: object,
    name: str,
) -> tuple[dict[str, object], ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    if not value:
        raise ValueError(f"{name} must not be empty")
    obligations: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError(f"{name} entries must be mappings")
        status = _require_non_empty_text(item.get("status"), f"{name} status")
        if status not in {"passed", "failed"}:
            raise ValueError(f"{name} status must be passed or failed")
        obligations.append(
            {
                "name": _require_non_empty_text(item.get("name"), f"{name} name"),
                "status": status,
                "evidence": _require_non_empty_text(
                    item.get("evidence"),
                    f"{name} evidence",
                ),
            }
        )
    return tuple(obligations)


def _normalise_topos_named_records(
    value: object,
    name: str,
) -> tuple[dict[str, object], ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    records: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError(f"{name} entries must be mappings")
        record: dict[str, object] = {
            "name": _require_non_empty_text(item.get("name"), f"{name} name"),
        }
        if "kind" in item:
            record["kind"] = _require_non_empty_text(item.get("kind"), f"{name} kind")
        if "detail" in item:
            record["detail"] = _require_non_empty_text(
                item.get("detail"),
                f"{name} detail",
            )
        if "regimes" in item:
            record["regimes"] = list(
                _normalise_text_sequence(item.get("regimes"), f"{name} regimes")
            )
        if "action_labels" in item:
            record["action_labels"] = list(
                _normalise_text_sequence(
                    item.get("action_labels"),
                    f"{name} action_labels",
                )
            )
        records.append(record)
    return tuple(records)


def _normalise_topos_morphisms(
    value: object,
    name: str,
) -> tuple[dict[str, object], ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    morphisms: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError(f"{name} entries must be mappings")
        deterministic = _required_bool(
            item.get("deterministic"),
            f"{name} deterministic",
        )
        if deterministic is not True:
            raise ValueError(f"{name} deterministic must be true")
        morphisms.append(
            {
                "source": _require_non_empty_text(
                    item.get("source"),
                    f"{name} source",
                ),
                "target": _require_non_empty_text(
                    item.get("target"),
                    f"{name} target",
                ),
                "label": _require_non_empty_text(item.get("label"), f"{name} label"),
                "deterministic": True,
            }
        )
    return tuple(morphisms)


def _normalise_topos_domain_examples(
    examples: Sequence[Mapping[str, object]],
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    if isinstance(examples, Mapping) or not isinstance(examples, Sequence):
        raise ValueError("Topos examples must be a sequence")
    normalised: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for index, example in enumerate(examples):
        if not isinstance(example, Mapping):
            raise ValueError("Topos example must be a mapping")
        label = f"Topos example {index}"
        proof_boundary = _require_non_empty_text(
            example.get("proof_boundary"),
            f"{label} proof_boundary",
        )
        if proof_boundary != _TOPOS_PROOF_BOUNDARY:
            raise ValueError(f"{label} proof boundary is not review-safe")
        if example.get("non_actuating") is not True:
            raise ValueError(f"{label} non_actuating must be true")
        passed = _required_bool(example.get("passed"), f"{label} passed")
        if passed is not True:
            raise ValueError(f"{label} must be passed")
        obligation_names = _normalise_text_sequence(
            example.get("obligation_names"),
            f"{label} obligation_names",
        )
        domain = _require_non_empty_text(example.get("domain"), f"{label} domain")
        example_hash = _require_sha256_hex(
            example.get("example_hash"),
            f"{label} example_hash",
        )
        binding_count = _positive_int(
            example.get("binding_object_count"),
            f"{label} binding_object_count",
            minimum=1,
        )
        policy_count = _positive_int(
            example.get("policy_object_count"),
            f"{label} policy_object_count",
            minimum=1,
        )
        record = {
            "domain": domain,
            "symbolic_prompt": _require_non_empty_text(
                example.get("symbolic_prompt"),
                f"{label} symbolic_prompt",
            ),
            "binding_object_count": binding_count,
            "policy_object_count": policy_count,
            "obligation_names": list(obligation_names),
            "passed": True,
            "non_actuating": True,
            "proof_boundary": _TOPOS_PROOF_BOUNDARY,
            "example_hash": example_hash,
        }
        normalised.append(record)
        rows.append(
            {
                "domain": domain,
                "example_hash": example_hash,
                "binding_object_count": binding_count,
                "policy_object_count": policy_count,
                "obligation_count": len(obligation_names),
            }
        )
    return (tuple(normalised), tuple(rows))


def _normalise_autopoietic_lineage_manifests(
    manifests: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    rows = _autopoietic_lineage_mapping_sequence(manifests, "lineage manifests")
    normalised: list[dict[str, object]] = []
    seen_lineages: set[str] = set()
    for index, manifest in enumerate(rows):
        name = f"lineage manifest {index}"
        if manifest.get("schema") != _AUTOPOIETIC_LINEAGE_SCHEMA:
            raise ValueError(f"{name} has unsupported schema")
        lineage_sha256 = _autopoietic_lineage_sha(
            manifest.get("lineage_sha256"), f"{name} lineage_sha256"
        )
        if lineage_sha256 in seen_lineages:
            raise ValueError(f"{name} duplicates lineage_sha256")
        seen_lineages.add(lineage_sha256)
        _autopoietic_lineage_bool(manifest, "review_required", True, name)
        _autopoietic_lineage_bool(manifest, "execution_disabled", True, name)
        _autopoietic_lineage_bool(manifest, "live_merge_permitted", False, name)
        _autopoietic_lineage_bool(manifest, "hot_patch_permitted", False, name)
        _autopoietic_lineage_bool(manifest, "actuation_permitted", False, name)

        replay_corpus = _normalise_autopoietic_replay_corpus(
            manifest.get("replay_corpus"), name
        )
        replay_domains = tuple(sorted({str(row["domain"]) for row in replay_corpus}))
        children = _normalise_autopoietic_lineage_children(
            manifest.get("child_candidates"), name
        )
        accepted_count = sum(
            1 for child in children if child["status"] == "accepted_for_review"
        )
        rejected_count = sum(1 for child in children if child["status"] == "rejected")

        child_candidate_count = _autopoietic_lineage_int(
            manifest.get("child_candidate_count"), f"{name} child_candidate_count"
        )
        accepted_child_count = _autopoietic_lineage_int(
            manifest.get("accepted_child_count"), f"{name} accepted_child_count"
        )
        rejected_child_count = _autopoietic_lineage_int(
            manifest.get("rejected_child_count"), f"{name} rejected_child_count"
        )
        replay_corpus_count = _autopoietic_lineage_int(
            manifest.get("replay_corpus_count"), f"{name} replay_corpus_count"
        )
        replay_domain_count = _autopoietic_lineage_int(
            manifest.get("replay_domain_count"), f"{name} replay_domain_count"
        )
        if child_candidate_count != len(children):
            raise ValueError(f"{name} child_candidate_count does not match rows")
        if accepted_child_count != accepted_count:
            raise ValueError(f"{name} accepted_child_count does not match rows")
        if rejected_child_count != rejected_count:
            raise ValueError(f"{name} rejected_child_count does not match rows")
        if replay_corpus_count != len(replay_corpus):
            raise ValueError(f"{name} replay_corpus_count does not match rows")
        if replay_domain_count != len(replay_domains):
            raise ValueError(f"{name} replay_domain_count does not match rows")

        normalised.append(
            {
                "schema": _AUTOPOIETIC_LINEAGE_SCHEMA,
                "lineage_sha256": lineage_sha256,
                "parent_policy_sha256": _autopoietic_lineage_sha(
                    manifest.get("parent_policy_sha256"),
                    f"{name} parent_policy_sha256",
                ),
                "replay_corpus_sha256": _autopoietic_lineage_sha(
                    manifest.get("replay_corpus_sha256"),
                    f"{name} replay_corpus_sha256",
                ),
                "review_required": True,
                "execution_disabled": True,
                "live_merge_permitted": False,
                "hot_patch_permitted": False,
                "actuation_permitted": False,
                "child_candidate_count": child_candidate_count,
                "accepted_child_count": accepted_child_count,
                "rejected_child_count": rejected_child_count,
                "child_candidates": children,
                "replay_corpus_count": replay_corpus_count,
                "replay_domain_count": replay_domain_count,
                "replay_domains": replay_domains,
                "replay_corpus": replay_corpus,
                "lineage_summary": _autopoietic_lineage_text(
                    manifest.get("lineage_summary"), f"{name} lineage_summary"
                )
                if "lineage_summary" in manifest
                else "",
            }
        )
    return tuple(normalised)


def _normalise_intergenerational_inheritance_histories(
    histories: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    rows = _autopoietic_lineage_mapping_sequence(histories, "inheritance histories")
    normalised: list[dict[str, object]] = []
    seen_histories: set[str] = set()
    for index, history in enumerate(rows):
        name = f"inheritance history {index}"
        if history.get("schema") != _INTERGENERATIONAL_HISTORY_SCHEMA:
            raise ValueError(f"{name} has unsupported schema")
        history_sha256 = _autopoietic_lineage_sha(
            history.get("history_sha256"), f"{name} history_sha256"
        )
        if history_sha256 in seen_histories:
            raise ValueError(f"{name} duplicates history_sha256")
        seen_histories.add(history_sha256)
        _intergenerational_history_hash_check(history, name, history_sha256)
        _autopoietic_lineage_bool(history, "hot_patch_review_required", True, name)
        _autopoietic_lineage_bool(history, "direct_hot_patch_permitted", False, name)
        _autopoietic_lineage_bool(history, "actuation_permitted", False, name)
        _autopoietic_lineage_bool(history, "operator_review_required", True, name)
        if history.get("merge_strategy") != "reviewed_hot_patch_only":
            raise ValueError(f"{name} merge_strategy is unsupported")

        child_rows = _normalise_intergenerational_child_rows(
            history.get("child_rows"), name
        )
        history_record_count = _autopoietic_lineage_int(
            history.get("history_record_count"), f"{name} history_record_count"
        )
        signed_metadata_count = _autopoietic_lineage_int(
            history.get("signed_metadata_count"), f"{name} signed_metadata_count"
        )
        replay_domain_count = _autopoietic_lineage_int(
            history.get("replay_domain_count"), f"{name} replay_domain_count"
        )
        replay_domains = _autopoietic_lineage_text_tuple(
            history.get("replay_domains"), f"{name} replay_domains"
        )
        if history_record_count != len(child_rows):
            raise ValueError(f"{name} history_record_count does not match rows")
        if signed_metadata_count != len(child_rows):
            raise ValueError(f"{name} signed_metadata_count does not match rows")
        if replay_domain_count != len(set(replay_domains)):
            raise ValueError(f"{name} replay_domain_count does not match rows")
        fitness_values = tuple(
            float(cast(float, row["fitness_score"])) for row in child_rows
        )
        minimum_fitness = _autopoietic_lineage_float(
            history.get("minimum_fitness_score"), f"{name} minimum_fitness_score"
        )
        maximum_fitness = _autopoietic_lineage_float(
            history.get("maximum_fitness_score"), f"{name} maximum_fitness_score"
        )
        mean_fitness = _autopoietic_lineage_float(
            history.get("mean_fitness_score"), f"{name} mean_fitness_score"
        )
        if not np.isclose(minimum_fitness, min(fitness_values)):
            raise ValueError(f"{name} minimum_fitness_score does not match rows")
        if not np.isclose(maximum_fitness, max(fitness_values)):
            raise ValueError(f"{name} maximum_fitness_score does not match rows")
        if not np.isclose(mean_fitness, float(np.mean(fitness_values))):
            raise ValueError(f"{name} mean_fitness_score does not match rows")

        normalised.append(
            {
                "schema": _INTERGENERATIONAL_HISTORY_SCHEMA,
                "history_sha256": history_sha256,
                "lineage_sha256": _autopoietic_lineage_sha(
                    history.get("lineage_sha256"), f"{name} lineage_sha256"
                ),
                "parent_policy_sha256": _autopoietic_lineage_sha(
                    history.get("parent_policy_sha256"),
                    f"{name} parent_policy_sha256",
                ),
                "history_record_count": history_record_count,
                "signed_metadata_count": signed_metadata_count,
                "replay_domain_count": replay_domain_count,
                "replay_domains": tuple(sorted(replay_domains)),
                "child_rows": child_rows,
                "minimum_fitness_score": minimum_fitness,
                "maximum_fitness_score": maximum_fitness,
                "mean_fitness_score": mean_fitness,
                "hot_patch_review_required": True,
                "direct_hot_patch_permitted": False,
                "merge_strategy": "reviewed_hot_patch_only",
                "actuation_permitted": False,
                "operator_review_required": True,
            }
        )
    return tuple(normalised)


def _normalise_intergenerational_child_rows(
    value: object,
    history_name: str,
) -> tuple[dict[str, object], ...]:
    rows = _autopoietic_lineage_mapping_sequence(value, f"{history_name} child_rows")
    if not rows:
        raise ValueError(f"{history_name} must contain child_rows")
    normalised: list[dict[str, object]] = []
    seen_inheritances: set[str] = set()
    for index, row in enumerate(rows):
        name = f"{history_name} child row {index}"
        inheritance_sha256 = _autopoietic_lineage_sha(
            row.get("inheritance_sha256"), f"{name} inheritance_sha256"
        )
        if inheritance_sha256 in seen_inheritances:
            raise ValueError(f"{name} duplicates inheritance_sha256")
        seen_inheritances.add(inheritance_sha256)
        generation_index = _autopoietic_lineage_int(
            row.get("generation_index"), f"{name} generation_index"
        )
        if generation_index != index:
            raise ValueError(f"{name} generation_index must be contiguous")
        _autopoietic_lineage_bool(row, "hot_patch_review_required", True, name)
        _autopoietic_lineage_bool(row, "direct_hot_patch_permitted", False, name)
        _autopoietic_lineage_bool(row, "actuation_permitted", False, name)
        if row.get("merge_strategy") != "reviewed_hot_patch_only":
            raise ValueError(f"{name} merge_strategy is unsupported")
        normalised.append(
            {
                "generation_index": generation_index,
                "inheritance_sha256": inheritance_sha256,
                "lineage_sha256": _autopoietic_lineage_sha(
                    row.get("lineage_sha256"), f"{name} lineage_sha256"
                ),
                "child_sha256": _autopoietic_lineage_sha(
                    row.get("child_sha256"), f"{name} child_sha256"
                ),
                "signer_id": _autopoietic_lineage_text(
                    row.get("signer_id"), f"{name} signer_id"
                ),
                "signature_sha256": _autopoietic_lineage_sha(
                    row.get("signature_sha256"), f"{name} signature_sha256"
                ),
                "policy_gene_count": _autopoietic_lineage_int(
                    row.get("policy_gene_count"), f"{name} policy_gene_count"
                ),
                "policy_diff_count": _autopoietic_lineage_int(
                    row.get("policy_diff_count"), f"{name} policy_diff_count"
                ),
                "fitness_score": _autopoietic_lineage_float(
                    row.get("fitness_score"), f"{name} fitness_score"
                ),
                "reward_component": _autopoietic_lineage_float(
                    row.get("reward_component"), f"{name} reward_component"
                ),
                "safety_component": _autopoietic_lineage_float(
                    row.get("safety_component"), f"{name} safety_component"
                ),
                "simplicity_component": _autopoietic_lineage_float(
                    row.get("simplicity_component"), f"{name} simplicity_component"
                ),
                "merge_strategy": "reviewed_hot_patch_only",
                "hot_patch_review_required": True,
                "direct_hot_patch_permitted": False,
                "actuation_permitted": False,
            }
        )
    return tuple(normalised)


def _intergenerational_history_hash_check(
    history: Mapping[str, object],
    name: str,
    history_sha256: str,
) -> None:
    body = dict(history)
    body.pop("history_sha256", None)
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    expected = sha256(canonical.encode("utf-8")).hexdigest()
    if expected != history_sha256:
        raise ValueError(f"{name} history_sha256 does not match content")


def _normalise_autopoietic_lineage_children(
    value: object,
    manifest_name: str,
) -> tuple[dict[str, object], ...]:
    children = _autopoietic_lineage_mapping_sequence(
        value, f"{manifest_name} child_candidates"
    )
    if not children:
        raise ValueError(f"{manifest_name} must contain child candidates")
    seen_children: set[str] = set()
    normalised: list[dict[str, object]] = []
    for index, child in enumerate(children):
        name = f"{manifest_name} child {index}"
        child_sha256 = _autopoietic_lineage_sha(
            child.get("child_sha256"), f"{name} child_sha256"
        )
        if child_sha256 in seen_children:
            raise ValueError(f"{name} duplicates child_sha256")
        seen_children.add(child_sha256)
        _autopoietic_lineage_bool(child, "review_required", True, name)
        _autopoietic_lineage_bool(child, "execution_disabled", True, name)
        _autopoietic_lineage_bool(child, "live_merge_permitted", False, name)
        _autopoietic_lineage_bool(child, "hot_patch_permitted", False, name)
        _autopoietic_lineage_bool(child, "actuation_permitted", False, name)
        status = _autopoietic_lineage_text(child.get("status"), f"{name} status")
        if status not in {"accepted_for_review", "rejected"}:
            raise ValueError(f"{name} has unsupported status")
        blocked_reasons = _autopoietic_lineage_text_tuple(
            child.get("blocked_reasons", ()), f"{name} blocked_reasons"
        )
        if status == "accepted_for_review" and blocked_reasons:
            raise ValueError(f"{name} accepted child cannot have blocked reasons")
        if status == "rejected" and not blocked_reasons:
            raise ValueError(f"{name} rejected child must explain blocked reasons")
        normalised.append(
            {
                "child_id": _autopoietic_lineage_text(
                    child.get("child_id"), f"{name} child_id"
                ),
                "child_sha256": child_sha256,
                "status": status,
                "review_required": True,
                "execution_disabled": True,
                "live_merge_permitted": False,
                "hot_patch_permitted": False,
                "actuation_permitted": False,
                "blocked_reasons": blocked_reasons,
                "policy_diff": _normalise_autopoietic_policy_diff(
                    child.get("policy_diff"), name
                ),
            }
        )
    return tuple(normalised)


def _normalise_autopoietic_policy_diff(
    value: object,
    child_name: str,
) -> tuple[dict[str, object], ...]:
    diff_rows = _autopoietic_lineage_mapping_sequence(
        value, f"{child_name} policy_diff"
    )
    if not diff_rows:
        raise ValueError(f"{child_name} must contain policy_diff rows")
    normalised: list[dict[str, object]] = []
    for index, row in enumerate(diff_rows):
        name = f"{child_name} policy_diff {index}"
        normalised.append(
            {
                "knob": _autopoietic_lineage_text(row.get("knob"), f"{name} knob"),
                "parent_value": _autopoietic_lineage_float(
                    row.get("parent_value"), f"{name} parent_value"
                ),
                "child_value": _autopoietic_lineage_float(
                    row.get("child_value"), f"{name} child_value"
                ),
                "delta": _autopoietic_lineage_float(row.get("delta"), f"{name} delta"),
            }
        )
    return tuple(normalised)


def _normalise_autopoietic_replay_corpus(
    value: object,
    manifest_name: str,
) -> tuple[dict[str, object], ...]:
    rows = _autopoietic_lineage_mapping_sequence(
        value, f"{manifest_name} replay_corpus"
    )
    if not rows:
        raise ValueError(f"{manifest_name} must contain replay corpus rows")
    normalised: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        name = f"{manifest_name} replay {index}"
        normalised.append(
            {
                "replay_id": _autopoietic_lineage_text(
                    row.get("replay_id"), f"{name} replay_id"
                ),
                "domain": _autopoietic_lineage_text(
                    row.get("domain"), f"{name} domain"
                ),
                "scenario": _autopoietic_lineage_text(
                    row.get("scenario"), f"{name} scenario"
                ),
                "reward": _autopoietic_lineage_float(
                    row.get("reward"), f"{name} reward"
                ),
                "safety_margin": _autopoietic_lineage_float(
                    row.get("safety_margin"), f"{name} safety_margin"
                ),
                "violation_count": _autopoietic_lineage_int(
                    row.get("violation_count"), f"{name} violation_count"
                ),
            }
        )
    return tuple(normalised)


def _autopoietic_lineage_mapping_sequence(
    value: object,
    name: str,
) -> tuple[Mapping[str, object], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    rows: list[Mapping[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"{name} item {index} must be a mapping")
        rows.append(item)
    return tuple(rows)


def _autopoietic_lineage_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _autopoietic_lineage_text_tuple(value: object, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of strings")
    return tuple(_autopoietic_lineage_text(item, f"{name} item") for item in value)


def _autopoietic_lineage_sha(value: object, name: str) -> str:
    text = _autopoietic_lineage_text(value, name)
    if (
        len(text) != 64
        or text.lower() != text
        or any(char not in "0123456789abcdef" for char in text)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _autopoietic_lineage_bool(
    mapping: Mapping[str, object],
    key: str,
    expected: bool,
    name: str,
) -> None:
    if mapping.get(key) is not expected:
        raise ValueError(f"{name} {key} must be {expected}")


def _autopoietic_lineage_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _autopoietic_lineage_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _normalise_evolutionary_search_reports(
    reports: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    if isinstance(reports, Mapping) or not isinstance(reports, Sequence) or not reports:
        raise ValueError("evolutionary search reports must be a non-empty sequence")
    normalised: list[dict[str, object]] = []
    for index, report in enumerate(reports):
        if not isinstance(report, Mapping):
            raise ValueError("evolutionary search report must be a mapping")
        label = f"evolutionary search report {index}"
        if report.get("schema_name") != _EVOLUTIONARY_SEARCH_SCHEMA:
            raise ValueError(f"{label} schema_name is not supported")
        if report.get("claim_boundary") != _EVOLUTIONARY_SEARCH_BOUNDARY:
            raise ValueError(f"{label} claim_boundary is not review-safe")
        _require_evolutionary_review_gates(report, label, require_actuation=False)
        candidate_count = _positive_int(
            report.get("candidate_count"), f"{label} candidate_count", minimum=1
        )
        accepted_count = _non_negative_int(
            report.get("accepted_count"), f"{label} accepted_count"
        )
        rejected_count = _non_negative_int(
            report.get("rejected_count"), f"{label} rejected_count"
        )
        if accepted_count + rejected_count != candidate_count:
            raise ValueError(
                f"{label} accepted_count and rejected_count must sum to candidate_count"
            )
        candidates = _normalise_evolutionary_candidates(
            report.get("candidates"), f"{label} candidates"
        )
        if len(candidates) != candidate_count:
            raise ValueError(f"{label} candidate_count must match candidates length")
        if (
            sum(1 for candidate in candidates if candidate["accepted"])
            != accepted_count
        ):
            raise ValueError(f"{label} accepted_count must match candidate statuses")
        if (
            sum(1 for candidate in candidates if not candidate["accepted"])
            != rejected_count
        ):
            raise ValueError(f"{label} rejected_count must match candidate statuses")
        best_candidate = _normalise_optional_evolutionary_candidate(
            report.get("best_candidate"), f"{label} best_candidate"
        )
        normalised.append(
            {
                "schema_name": _EVOLUTIONARY_SEARCH_SCHEMA,
                "schema_version": _require_non_empty_text(
                    report.get("schema_version"), f"{label} schema_version"
                ),
                "generation_count": _positive_int(
                    report.get("generation_count"),
                    f"{label} generation_count",
                    minimum=1,
                ),
                "population_size": _positive_int(
                    report.get("population_size"), f"{label} population_size", minimum=1
                ),
                "mutation_step": _positive_float(
                    report.get("mutation_step"), f"{label} mutation_step"
                ),
                "minimum_replay_reward": _finite_number(
                    report.get("minimum_replay_reward"),
                    f"{label} minimum_replay_reward",
                ),
                "minimum_safety_margin": _finite_number(
                    report.get("minimum_safety_margin"),
                    f"{label} minimum_safety_margin",
                ),
                "parent_policy_hash": _require_sha256_hex(
                    report.get("parent_policy_hash"), f"{label} parent_policy_hash"
                ),
                "replay_summary": _normalise_evolutionary_replay_summary(
                    report.get("replay_summary"), f"{label} replay_summary"
                ),
                "stl_spec": _require_non_empty_text(
                    report.get("stl_spec"), f"{label} stl_spec"
                ),
                "stl_monitoring": _normalise_evolutionary_stl_monitoring(
                    report.get("stl_monitoring"), f"{label} stl_monitoring"
                ),
                "candidate_count": candidate_count,
                "accepted_count": accepted_count,
                "rejected_count": rejected_count,
                "best_candidate": best_candidate,
                "candidates": candidates,
                "claim_boundary": _EVOLUTIONARY_SEARCH_BOUNDARY,
                "non_actuating": True,
                "execution_disabled": True,
                "hot_patch_permitted": False,
                "live_merge_permitted": False,
                "operator_review_required": True,
                "report_hash": _require_sha256_hex(
                    report.get("report_hash"), f"{label} report_hash"
                ),
            }
        )
    return tuple(normalised)


def _normalise_evolutionary_candidates(
    value: object, name: str
) -> tuple[dict[str, object], ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    candidates: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for candidate in value:
        candidates.append(_normalise_evolutionary_candidate(candidate, name, seen_ids))
    if not candidates:
        raise ValueError(f"{name} must not be empty")
    return tuple(candidates)


def _normalise_optional_evolutionary_candidate(
    value: object, name: str
) -> dict[str, object] | None:
    if value is None:
        return None
    return _normalise_evolutionary_candidate(value, name, set())


def _normalise_evolutionary_candidate(
    value: object, name: str, seen_ids: set[str]
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} entries must be mappings")
    candidate_id = _require_non_empty_text(value.get("candidate_id"), "candidate_id")
    if candidate_id in seen_ids:
        raise ValueError(f"{name} candidate_id values must be unique")
    seen_ids.add(candidate_id)
    for field, expected in (
        ("review_required", True),
        ("live_merge_permitted", False),
        ("hot_patch_permitted", False),
        ("actuation_permitted", False),
    ):
        if _required_bool(value.get(field), f"{name} {field}") is not expected:
            raise ValueError(f"{name} {field} must be {expected}")
    status = _require_non_empty_text(value.get("status"), f"{name} status")
    if status not in {"accepted_for_review", "rejected"}:
        raise ValueError(f"{name} status is not supported")
    blocked_reasons = _normalise_optional_text_sequence(
        value.get("blocked_reasons"), f"{name} blocked_reasons"
    )
    accepted = status == "accepted_for_review"
    if accepted and blocked_reasons:
        raise ValueError(f"{name} accepted candidates must not have blocked_reasons")
    if not accepted and not blocked_reasons:
        raise ValueError(f"{name} rejected candidates require blocked_reasons")
    return {
        "candidate_id": candidate_id,
        "generation": _positive_int(
            value.get("generation"), f"{name} generation", minimum=1
        ),
        "knob": _require_non_empty_text(value.get("knob"), f"{name} knob"),
        "parent_value": _finite_number(
            value.get("parent_value"), f"{name} parent_value"
        ),
        "candidate_value": _finite_number(
            value.get("candidate_value"), f"{name} candidate_value"
        ),
        "mutation_delta": _finite_number(
            value.get("mutation_delta"), f"{name} mutation_delta"
        ),
        "genome": _normalise_information_geometry_gradient(
            value.get("genome"), f"{name} genome"
        ),
        "replay_fitness": _finite_number(
            value.get("replay_fitness"), f"{name} replay_fitness"
        ),
        "stl_robustness": _finite_number(
            value.get("stl_robustness"), f"{name} stl_robustness"
        ),
        "stl_satisfied": _required_bool(
            value.get("stl_satisfied"), f"{name} stl_satisfied"
        ),
        "replay_violation_count": _non_negative_int(
            value.get("replay_violation_count"), f"{name} replay_violation_count"
        ),
        "blocked_reasons": tuple(blocked_reasons),
        "status": status,
        "accepted": accepted,
        "candidate_hash": _require_sha256_hex(
            value.get("candidate_hash"), f"{name} candidate_hash"
        ),
        "review_required": True,
        "live_merge_permitted": False,
        "hot_patch_permitted": False,
        "actuation_permitted": False,
    }


def _normalise_evolutionary_replay_summary(
    value: object, name: str
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return {
        "replay_count": _positive_int(
            value.get("replay_count"), f"{name} replay_count", minimum=1
        ),
        "mean_reward": _finite_number(value.get("mean_reward"), f"{name} mean_reward"),
        "min_reward": _finite_number(value.get("min_reward"), f"{name} min_reward"),
        "mean_safety_margin": _finite_number(
            value.get("mean_safety_margin"), f"{name} mean_safety_margin"
        ),
        "min_safety_margin": _finite_number(
            value.get("min_safety_margin"), f"{name} min_safety_margin"
        ),
        "violation_count": _non_negative_int(
            value.get("violation_count"), f"{name} violation_count"
        ),
    }


def _normalise_evolutionary_stl_monitoring(
    value: object, name: str
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    result: dict[str, object] = {}
    for key, raw_value in value.items():
        key_text = _require_non_empty_text(key, f"{name} key")
        if isinstance(raw_value, bool):
            result[key_text] = bool(raw_value)
        elif isinstance(raw_value, int | float):
            result[key_text] = _finite_number(raw_value, f"{name} {key_text}")
        elif isinstance(raw_value, str):
            result[key_text] = _require_non_empty_text(raw_value, f"{name} {key_text}")
        elif isinstance(raw_value, Sequence) and not isinstance(raw_value, str | bytes):
            result[key_text] = list(
                _normalise_evolutionary_json_sequence(raw_value, f"{name} {key_text}")
            )
        else:
            raise ValueError(f"{name} values must be JSON-safe scalars or sequences")
    return result


def _normalise_evolutionary_json_sequence(
    values: Sequence[object], name: str
) -> tuple[object, ...]:
    normalised: list[object] = []
    for item in values:
        if isinstance(item, bool):
            normalised.append(bool(item))
        elif isinstance(item, int | float):
            normalised.append(_finite_number(item, name))
        elif isinstance(item, str):
            normalised.append(_require_non_empty_text(item, name))
        else:
            raise ValueError(f"{name} sequence values must be JSON-safe scalars")
    return tuple(normalised)


def _normalise_evolutionary_examples(
    examples: Sequence[Mapping[str, object]],
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    if isinstance(examples, Mapping) or not isinstance(examples, Sequence):
        raise ValueError("evolutionary examples must be a sequence")
    normalised: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for index, example in enumerate(examples):
        if not isinstance(example, Mapping):
            raise ValueError("evolutionary example must be a mapping")
        label = f"evolutionary example {index}"
        if example.get("claim_boundary") != _EVOLUTIONARY_EXAMPLE_BOUNDARY:
            raise ValueError(f"{label} claim_boundary is not review-safe")
        for field, expected in (
            ("operator_review_required", True),
            ("execution_disabled", True),
            ("hot_patch_permitted", False),
            ("live_merge_permitted", False),
            ("actuation_permitted", False),
        ):
            if _required_bool(example.get(field), f"{label} {field}") is not expected:
                raise ValueError(f"{label} {field} must be {expected}")
        scenario_hash = _require_sha256_hex(
            example.get("scenario_hash"), f"{label} scenario_hash"
        )
        domain = _require_non_empty_text(example.get("domain"), f"{label} domain")
        candidate_count = _optional_non_negative_int(
            example.get("candidate_count"), f"{label} candidate_count"
        )
        accepted_count = _optional_non_negative_int(
            example.get("accepted_candidate_count"), f"{label} accepted_candidate_count"
        )
        rejected_count = _optional_non_negative_int(
            example.get("rejected_candidate_count"), f"{label} rejected_candidate_count"
        )
        report_hash = _optional_sha256_hex(
            example.get("report_hash"), f"{label} report_hash"
        )
        if candidate_count is not None:
            if accepted_count is None or rejected_count is None or report_hash is None:
                raise ValueError(
                    f"{label} enriched candidate counts require report_hash"
                )
            if accepted_count + rejected_count != candidate_count:
                raise ValueError(f"{label} accepted/rejected counts must sum")
        normalised.append(
            {
                "domain": domain,
                "scenario_id": _require_non_empty_text(
                    example.get("scenario_id"), f"{label} scenario_id"
                ),
                "scenario_hash": scenario_hash,
                "claim_boundary": _EVOLUTIONARY_EXAMPLE_BOUNDARY,
                "operator_review_required": True,
                "execution_disabled": True,
                "hot_patch_permitted": False,
                "live_merge_permitted": False,
                "actuation_permitted": False,
                "candidate_count": candidate_count,
                "accepted_candidate_count": accepted_count,
                "rejected_candidate_count": rejected_count,
                "report_hash": report_hash,
            }
        )
        rows.append(
            {
                "domain": domain,
                "scenario_id": example["scenario_id"],
                "scenario_hash": scenario_hash,
                "candidate_count": candidate_count,
                "accepted_candidate_count": accepted_count,
                "rejected_candidate_count": rejected_count,
            }
        )
    return (tuple(normalised), tuple(rows))


def _normalise_evolutionary_dsl_reports(
    reports: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    if isinstance(reports, Mapping) or not isinstance(reports, Sequence):
        raise ValueError("evolutionary DSL reports must be a sequence")
    normalised: list[dict[str, object]] = []
    for index, report in enumerate(reports):
        if not isinstance(report, Mapping):
            raise ValueError("evolutionary DSL report must be a mapping")
        label = f"evolutionary DSL report {index}"
        if report.get("schema_name") != _EVOLUTIONARY_DSL_SCHEMA:
            raise ValueError(f"{label} schema_name is not supported")
        _require_evolutionary_review_gates(report, label, require_actuation=True)
        candidate_count = _positive_int(
            report.get("candidate_count"), f"{label} candidate_count", minimum=1
        )
        accepted_count = _non_negative_int(
            report.get("accepted_count"), f"{label} accepted_count"
        )
        rejected_count = _non_negative_int(
            report.get("rejected_count"), f"{label} rejected_count"
        )
        if accepted_count + rejected_count != candidate_count:
            raise ValueError(f"{label} accepted/rejected counts must sum")
        candidates = _normalise_evolutionary_dsl_candidates(
            report.get("candidates"), f"{label} candidates"
        )
        if len(candidates) != candidate_count:
            raise ValueError(f"{label} candidate_count must match candidates length")
        normalised.append(
            {
                "schema_name": _EVOLUTIONARY_DSL_SCHEMA,
                "schema_version": _require_non_empty_text(
                    report.get("schema_version"), f"{label} schema_version"
                ),
                "generation_count": _positive_int(
                    report.get("generation_count"),
                    f"{label} generation_count",
                    minimum=1,
                ),
                "population_size": _positive_int(
                    report.get("population_size"), f"{label} population_size", minimum=1
                ),
                "mutation_step": _positive_float(
                    report.get("mutation_step"), f"{label} mutation_step"
                ),
                "source_policy_hash": _require_sha256_hex(
                    report.get("source_policy_hash"), f"{label} source_policy_hash"
                ),
                "candidate_count": candidate_count,
                "accepted_count": accepted_count,
                "rejected_count": rejected_count,
                "candidates": candidates,
                "execution_disabled": True,
                "hot_patch_permitted": False,
                "live_merge_permitted": False,
                "actuation_permitted": False,
                "operator_review_required": True,
                "non_actuating": True,
                "report_hash": _require_sha256_hex(
                    report.get("report_hash"), f"{label} report_hash"
                ),
            }
        )
    return tuple(normalised)


def _normalise_evolutionary_dsl_candidates(
    value: object, name: str
) -> tuple[dict[str, object], ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    candidates: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for candidate in value:
        if not isinstance(candidate, Mapping):
            raise ValueError(f"{name} entries must be mappings")
        candidate_id = _require_non_empty_text(
            candidate.get("candidate_id"), f"{name} candidate_id"
        )
        if candidate_id in seen_ids:
            raise ValueError(f"{name} candidate_id values must be unique")
        seen_ids.add(candidate_id)
        for field, expected in (
            ("operator_review_required", True),
            ("execution_disabled", True),
            ("live_merge_permitted", False),
            ("hot_patch_permitted", False),
            ("actuation_permitted", False),
        ):
            if _required_bool(candidate.get(field), f"{name} {field}") is not expected:
                raise ValueError(f"{name} {field} must be {expected}")
        status = _require_non_empty_text(candidate.get("status"), f"{name} status")
        if status not in {"accepted", "rejected"}:
            raise ValueError(f"{name} status is not supported")
        candidates.append(
            {
                "candidate_id": candidate_id,
                "generation": _positive_int(
                    candidate.get("generation"), f"{name} generation", minimum=1
                ),
                "mutation_index": _non_negative_int(
                    candidate.get("mutation_index"), f"{name} mutation_index"
                ),
                "source_rule_name": _require_non_empty_text(
                    candidate.get("source_rule_name"), f"{name} source_rule_name"
                ),
                "blocked_reasons": _normalise_optional_text_sequence(
                    candidate.get("blocked_reasons"), f"{name} blocked_reasons"
                ),
                "status": status,
                "accepted": status == "accepted",
                "candidate_hash": _require_sha256_hex(
                    candidate.get("candidate_hash"), f"{name} candidate_hash"
                ),
                "operator_review_required": True,
                "execution_disabled": True,
                "live_merge_permitted": False,
                "hot_patch_permitted": False,
                "actuation_permitted": False,
            }
        )
    if not candidates:
        raise ValueError(f"{name} must not be empty")
    return tuple(candidates)


def _require_evolutionary_review_gates(
    record: Mapping[str, object], label: str, *, require_actuation: bool
) -> None:
    fields = [
        ("non_actuating", True),
        ("execution_disabled", True),
        ("hot_patch_permitted", False),
        ("live_merge_permitted", False),
        ("operator_review_required", True),
    ]
    if require_actuation:
        fields.append(("actuation_permitted", False))
    for field, expected in fields:
        if _required_bool(record.get(field), f"{label} {field}") is not expected:
            raise ValueError(f"{label} {field} must be {expected}")


def _normalise_hybrid_order_records(
    records: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    if isinstance(records, Mapping) or not isinstance(records, Sequence) or not records:
        raise ValueError("hybrid-order records must be a non-empty sequence")
    normalised: list[dict[str, object]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError("hybrid-order record must be a mapping")
        label = f"hybrid-order record {index}"
        claim_boundary = _require_non_empty_text(
            record.get("claim_boundary"), f"{label} claim_boundary"
        )
        if claim_boundary != _HYBRID_ORDER_CLAIM_BOUNDARY:
            raise ValueError(f"{label} claim boundary is not review-safe")
        if record.get("non_actuating") is not True:
            raise ValueError(f"{label} non_actuating must be true")
        if record.get("execution_disabled") is not True:
            raise ValueError(f"{label} execution_disabled must be true")
        backend = _require_non_empty_text(record.get("backend"), f"{label} backend")
        if backend not in _HYBRID_ORDER_BACKENDS:
            raise ValueError(f"{label} backend is not supported")
        qubit_count = _hybrid_positive_int(
            record.get("qubit_count"), f"{label} qubit_count"
        )
        normalised_record = {
            "R": _hybrid_unit_interval(
                _finite_number(record.get("R"), f"{label} R"),
                f"{label} R",
            ),
            "Psi": _finite_number(record.get("Psi"), f"{label} Psi"),
            "entanglement_entropy": _hybrid_non_negative(
                _finite_number(
                    record.get("entanglement_entropy"),
                    f"{label} entanglement_entropy",
                ),
                f"{label} entanglement_entropy",
            ),
            "normalised_entanglement_entropy": _hybrid_unit_interval(
                _finite_number(
                    record.get("normalised_entanglement_entropy"),
                    f"{label} normalised_entanglement_entropy",
                ),
                f"{label} normalised_entanglement_entropy",
            ),
            "participation_ratio": _hybrid_positive_float(
                _finite_number(
                    record.get("participation_ratio"),
                    f"{label} participation_ratio",
                ),
                f"{label} participation_ratio",
            ),
            "qubit_count": qubit_count,
            "bipartition": _normalise_hybrid_bipartition(
                record.get("bipartition"),
                qubit_count=qubit_count,
                label=f"{label} bipartition",
            ),
            "backend": backend,
            "claim_boundary": claim_boundary,
            "non_actuating": True,
            "execution_disabled": True,
            "record_hash": _validated_hybrid_record_hash(record, label),
        }
        normalised.append(normalised_record)
    return tuple(normalised)


def _normalise_hybrid_order_scenarios(
    scenarios: Sequence[Mapping[str, object]],
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    if isinstance(scenarios, Mapping) or not isinstance(scenarios, Sequence):
        raise ValueError("hybrid-order scenarios must be a sequence")
    normalised_scenarios: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    for scenario_index, scenario in enumerate(scenarios):
        if not isinstance(scenario, Mapping):
            raise ValueError("hybrid-order scenario must be a mapping")
        label = f"hybrid-order scenario {scenario_index}"
        claim_boundary = _require_non_empty_text(
            scenario.get("claim_boundary"), f"{label} claim_boundary"
        )
        if claim_boundary != _HYBRID_ORDER_CLAIM_BOUNDARY:
            raise ValueError(f"{label} claim boundary is not review-safe")
        if scenario.get("non_actuating") is not True:
            raise ValueError(f"{label} non_actuating must be true")
        if scenario.get("execution_disabled") is not True:
            raise ValueError(f"{label} execution_disabled must be true")
        domain = _require_non_empty_text(scenario.get("domain"), f"{label} domain")
        scenario_id = _require_non_empty_text(
            scenario.get("scenario_id"), f"{label} scenario_id"
        )
        scenario_hash = _require_sha256_hex(
            scenario.get("scenario_hash"), f"{label} scenario_hash"
        )
        qubit_count = _hybrid_positive_int(
            scenario.get("qubit_count"), f"{label} qubit_count"
        )
        candidates = scenario.get("state_candidates")
        if (
            isinstance(candidates, Mapping)
            or not isinstance(candidates, Sequence)
            or not candidates
        ):
            raise ValueError(f"{label} state_candidates must be a non-empty sequence")
        normalised_scenarios.append(
            {
                "domain": domain,
                "scenario_id": scenario_id,
                "scenario_hash": scenario_hash,
                "qubit_count": qubit_count,
                "claim_boundary": claim_boundary,
                "non_actuating": True,
                "execution_disabled": True,
            }
        )
        for candidate_index, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                raise ValueError(f"{label} candidate must be a mapping")
            candidate_label = f"{label} candidate {candidate_index}"
            candidate_claim = _require_non_empty_text(
                candidate.get("claim_boundary"), f"{candidate_label} claim_boundary"
            )
            if candidate_claim != _HYBRID_ORDER_CLAIM_BOUNDARY:
                raise ValueError(f"{candidate_label} claim boundary is not review-safe")
            if candidate.get("non_actuating") is not True:
                raise ValueError(f"{candidate_label} non_actuating must be true")
            if candidate.get("execution_disabled") is not True:
                raise ValueError(f"{candidate_label} execution_disabled must be true")
            state_type = _require_non_empty_text(
                candidate.get("state_type"), f"{candidate_label} state_type"
            )
            if state_type not in {"product", "entangled"}:
                raise ValueError(f"{candidate_label} state_type is not supported")
            candidate_rows.append(
                {
                    "domain": domain,
                    "scenario_id": scenario_id,
                    "scenario_hash": scenario_hash,
                    "state_id": _require_non_empty_text(
                        candidate.get("state_id"), f"{candidate_label} state_id"
                    ),
                    "state_type": state_type,
                    "entanglement_entropy": _hybrid_non_negative(
                        _finite_number(
                            candidate.get("entanglement_entropy"),
                            f"{candidate_label} entanglement_entropy",
                        ),
                        f"{candidate_label} entanglement_entropy",
                    ),
                    "order_metric_r": _hybrid_unit_interval(
                        _finite_number(
                            candidate.get("order_metric_r"),
                            f"{candidate_label} order_metric_r",
                        ),
                        f"{candidate_label} order_metric_r",
                    ),
                    "order_metric_psi": _finite_number(
                        candidate.get("order_metric_psi"),
                        f"{candidate_label} order_metric_psi",
                    ),
                }
            )
    return tuple(normalised_scenarios), tuple(candidate_rows)


def _validated_hybrid_record_hash(record: Mapping[str, object], label: str) -> str:
    record_hash = _require_sha256_hex(record.get("record_hash"), f"{label} record_hash")
    payload = dict(record)
    payload.pop("record_hash", None)
    expected_hash = sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if expected_hash != record_hash:
        raise ValueError(f"{label} record_hash does not match payload")
    return record_hash


def _normalise_hybrid_bipartition(
    value: object,
    *,
    qubit_count: int,
    label: str,
) -> list[list[int]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a two-group bipartition")
    if len(value) != 2:
        raise ValueError(f"{label} must contain two groups")
    groups: list[list[int]] = []
    merged: list[int] = []
    for group_index, group in enumerate(value):
        if not isinstance(group, Sequence) or isinstance(group, (str, bytes)):
            raise ValueError(f"{label} group {group_index} must be a sequence")
        if not group:
            raise ValueError(f"{label} group {group_index} must be non-empty")
        indices: list[int] = []
        for item in group:
            index = _hybrid_non_bool_int(item, f"{label} index")
            if index < 0 or index >= qubit_count:
                raise ValueError(f"{label} index out of range")
            indices.append(index)
            merged.append(index)
        groups.append(indices)
    if len(set(merged)) != len(merged) or len(merged) != qubit_count:
        raise ValueError(f"{label} must cover each qubit exactly once")
    return groups


def _hybrid_non_bool_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _hybrid_positive_int(value: object, name: str) -> int:
    result = _hybrid_non_bool_int(value, name)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _hybrid_non_negative(value: float, name: str) -> float:
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _hybrid_positive_float(value: float, name: str) -> float:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _hybrid_unit_interval(value: float, name: str) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return value


def _normalise_morphogenetic_field_svg_artifact(
    artifact: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(artifact, Mapping):
        raise ValueError("morphogenetic SVG artifact must be a mapping")
    if artifact.get("format") != "svg":
        raise ValueError("format must be svg")
    width = _positive_int(artifact.get("width"), "width", minimum=1)
    height = _positive_int(artifact.get("height"), "height", minimum=1)
    svg = _require_review_svg(artifact.get("svg"))
    snapshot_raw = artifact.get("snapshot")
    if not isinstance(snapshot_raw, Mapping):
        raise ValueError("snapshot must be a mapping")
    snapshot = _normalise_morphogenetic_field_snapshot(snapshot_raw)
    return {
        "format": "svg",
        "width": width,
        "height": height,
        "snapshot": snapshot,
        "svg": svg,
    }


def _normalise_morphogenetic_field_snapshot(
    snapshot: Mapping[str, object],
) -> dict[str, object]:
    shape = _normalise_matrix_shape(snapshot.get("shape"))
    mean = _unit_interval_number(snapshot.get("mean"), "mean")
    minimum = _unit_interval_number(snapshot.get("minimum"), "minimum")
    maximum = _unit_interval_number(snapshot.get("maximum"), "maximum")
    if minimum > mean + 1e-12 or mean > maximum + 1e-12:
        raise ValueError("snapshot statistics must satisfy minimum <= mean <= maximum")
    heatmap_rows = _normalise_heatmap_rows(
        snapshot.get("heatmap_rows"),
        expected_rows=shape[0],
        expected_columns=shape[1],
    )
    top_edges = _normalise_morphogenetic_top_edges(
        snapshot.get("top_edges"),
        shape=shape,
    )
    return {
        "shape": list(shape),
        "mean": mean,
        "minimum": minimum,
        "maximum": maximum,
        "l2_norm": _non_negative_float(snapshot.get("l2_norm"), "l2_norm"),
        "heatmap_rows": list(heatmap_rows),
        "top_edges": top_edges,
    }


def _normalise_matrix_shape(value: object) -> tuple[int, int]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError("shape must be a two-item integer sequence")
    if len(value) != 2:
        raise ValueError("shape must be a two-item integer sequence")
    rows = _positive_int(value[0], "shape", minimum=1)
    columns = _positive_int(value[1], "shape", minimum=1)
    if rows != columns:
        raise ValueError("shape must be square")
    return (rows, columns)


def _normalise_heatmap_rows(
    value: object,
    *,
    expected_rows: int,
    expected_columns: int,
) -> tuple[str, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError("heatmap_rows must be a sequence of strings")
    if len(value) != expected_rows:
        raise ValueError("heatmap_rows length must match shape")
    rows: list[str] = []
    for row in value:
        if not isinstance(row, str) or row == "":
            raise ValueError("heatmap_rows must contain non-empty strings")
        row_text = row
        if len(row_text) != expected_columns:
            raise ValueError("heatmap row width must match shape")
        rows.append(row_text)
    return tuple(rows)


def _normalise_morphogenetic_top_edges(
    value: object,
    *,
    shape: tuple[int, int],
) -> tuple[dict[str, object], ...]:
    if value is None:
        return ()
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError("top_edges must be a sequence")
    edges: list[dict[str, object]] = []
    previous_weight: float | None = None
    for edge in value:
        if not isinstance(edge, Mapping):
            raise ValueError("top_edges entries must be mappings")
        source = _non_negative_int(edge.get("source"), "top_edges source")
        target = _non_negative_int(edge.get("target"), "top_edges target")
        if source >= shape[0] or target >= shape[1] or source == target:
            raise ValueError("top_edges must reference off-diagonal field edges")
        weight = _unit_interval_number(edge.get("weight"), "top_edges weight")
        if previous_weight is not None and weight > previous_weight + 1e-12:
            raise ValueError("top_edges must be sorted by descending weight")
        previous_weight = weight
        edges.append({"source": source, "target": target, "weight": weight})
    return tuple(edges)


def _normalise_multiverse_manifest(
    manifest: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(manifest, Mapping):
        raise ValueError("multiverse manifest must be a mapping")
    if manifest.get("schema_name") != "multiverse_counterfactual_rollout":
        raise ValueError("schema_name must be multiverse_counterfactual_rollout")
    schema_version = _require_non_empty_text(
        manifest.get("schema_version"),
        "schema_version",
    )
    branch_count = _positive_int(
        manifest.get("branch_count"), "branch_count", minimum=1
    )
    horizon = _positive_int(manifest.get("horizon"), "horizon", minimum=1)
    backend = _require_non_empty_text(manifest.get("backend"), "backend")
    if backend not in _MULTIVERSE_BACKENDS:
        raise ValueError("backend must be a supported multiverse rollout backend")
    if manifest.get("non_actuating") is not True:
        raise ValueError("multiverse manifest must be non_actuating")
    if manifest.get("execution_disabled") is not True:
        raise ValueError("multiverse manifest execution_disabled must be true")
    if manifest.get("claim_boundary") != _MULTIVERSE_ROLLOUT_CLAIM_BOUNDARY:
        raise ValueError("multiverse manifest claim boundary is invalid")
    branch_records = _normalise_multiverse_branch_records(
        manifest.get("branch_records")
    )
    if len(branch_records) != branch_count:
        raise ValueError("branch_count must match branch_records length")
    return {
        "schema_name": "multiverse_counterfactual_rollout",
        "schema_version": schema_version,
        "branch_count": branch_count,
        "horizon": horizon,
        "backend": backend,
        "non_actuating": True,
        "execution_disabled": True,
        "claim_boundary": _MULTIVERSE_ROLLOUT_CLAIM_BOUNDARY,
        "manifest_hash": _require_sha256_hex(
            manifest.get("manifest_hash"),
            "manifest_hash",
        ),
        "branch_records": branch_records,
    }


def _normalise_multiverse_branch_records(
    records: object,
) -> tuple[dict[str, object], ...]:
    if isinstance(records, str | bytes) or not isinstance(records, Sequence):
        raise ValueError("branch_records must be a sequence")
    if not records:
        raise ValueError("branch_records must be non-empty")
    normalised: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError("branch_records entries must be mappings")
        branch_id = _require_non_empty_text(record.get("branch_id"), "branch_id")
        if branch_id in seen_ids:
            raise ValueError("branch_records must have unique branch_id values")
        seen_ids.add(branch_id)
        action_count = _non_negative_int(record.get("action_count"), "action_count")
        action_labels = _normalise_text_sequence(
            record.get("action_labels"),
            "action_labels",
        )
        if len(action_labels) != action_count:
            raise ValueError("action_count must match action_labels length")
        final_r = _unit_interval_number(record.get("final_R"), "final_R")
        mean_r = _unit_interval_number(record.get("mean_R"), "mean_R")
        min_r = _unit_interval_number(record.get("min_R"), "min_R")
        max_r = _unit_interval_number(record.get("max_R"), "max_R")
        if min_r > mean_r + 1e-12 or mean_r > max_r + 1e-12:
            raise ValueError("R interval must satisfy min_R <= mean_R <= max_R")
        if final_r < min_r - 1e-12 or final_r > max_r + 1e-12:
            raise ValueError("R interval must contain final_R")
        normalised.append(
            {
                "branch_index": index,
                "branch_id": branch_id,
                "branch_hash": _require_sha256_hex(
                    record.get("branch_hash"),
                    "branch_hash",
                ),
                "action_count": action_count,
                "action_labels": list(action_labels),
                "topology_edge_count": _non_negative_int(
                    record.get("topology_edge_count"),
                    "topology_edge_count",
                ),
                "topology_scale": _non_negative_float(
                    record.get("topology_scale"),
                    "topology_scale",
                ),
                "final_R": final_r,
                "mean_R": mean_r,
                "min_R": min_r,
                "max_R": max_r,
                "final_psi": _non_negative_float(
                    record.get("final_psi"),
                    "final_psi",
                ),
            }
        )
    return tuple(normalised)


def _normalise_multiverse_risk_report(
    risk_report: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(risk_report, Mapping):
        raise ValueError("multiverse risk report must be a mapping")
    if risk_report.get("schema_name") != "multiverse_branch_risk_gate":
        raise ValueError("schema_name must be multiverse_branch_risk_gate")
    if risk_report.get("non_actuating") is not True:
        raise ValueError("multiverse risk report must be non_actuating")
    if risk_report.get("execution_disabled") is not True:
        raise ValueError("multiverse risk report execution_disabled must be true")
    if risk_report.get("claim_boundary") != _MULTIVERSE_RISK_CLAIM_BOUNDARY:
        raise ValueError("multiverse risk report claim boundary is invalid")
    decisions = _normalise_multiverse_risk_decisions(
        risk_report.get("branch_decisions")
    )
    branch_count = _positive_int(
        risk_report.get("branch_count"),
        "branch_count",
        minimum=1,
    )
    if branch_count != len(decisions):
        raise ValueError("branch_count must match branch_decisions length")
    approved_count = _non_negative_int(
        risk_report.get("approved_count"),
        "approved_count",
    )
    rejected_count = _non_negative_int(
        risk_report.get("rejected_count"),
        "rejected_count",
    )
    if approved_count + rejected_count != branch_count:
        raise ValueError("approved_count and rejected_count must sum to branch_count")
    if approved_count != sum(1 for decision in decisions if decision["approved"]):
        raise ValueError("approved_count must match approved branch decisions")
    if rejected_count != sum(1 for decision in decisions if not decision["approved"]):
        raise ValueError("rejected_count must match rejected branch decisions")
    return {
        "schema_name": "multiverse_branch_risk_gate",
        "schema_version": _require_non_empty_text(
            risk_report.get("schema_version"),
            "schema_version",
        ),
        "branch_count": branch_count,
        "approved_count": approved_count,
        "rejected_count": rejected_count,
        "safest_branch_id": _optional_text(
            risk_report.get("safest_branch_id"),
            "safest_branch_id",
        ),
        "safest_branch_hash": _optional_sha256_hex(
            risk_report.get("safest_branch_hash"),
            "safest_branch_hash",
        ),
        "rejection_reasons": list(
            _normalise_optional_text_sequence(
                risk_report.get("rejection_reasons"),
                "rejection_reasons",
            )
        ),
        "claim_boundary": _MULTIVERSE_RISK_CLAIM_BOUNDARY,
        "non_actuating": True,
        "execution_disabled": True,
        "report_hash": _require_sha256_hex(
            risk_report.get("report_hash"), "report_hash"
        ),
        "branch_decisions": decisions,
    }


def _normalise_multiverse_risk_decisions(
    decisions: object,
) -> tuple[dict[str, object], ...]:
    if isinstance(decisions, str | bytes) or not isinstance(decisions, Sequence):
        raise ValueError("branch_decisions must be a sequence")
    if not decisions:
        raise ValueError("branch_decisions must be non-empty")
    normalised: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for decision in decisions:
        if not isinstance(decision, Mapping):
            raise ValueError("branch_decisions entries must be mappings")
        branch_id = _require_non_empty_text(decision.get("branch_id"), "branch_id")
        if branch_id in seen_ids:
            raise ValueError("branch_decisions must have unique branch_id values")
        seen_ids.add(branch_id)
        final_r = _unit_interval_number(decision.get("final_R"), "final_R")
        mean_r = _unit_interval_number(decision.get("mean_R"), "mean_R")
        min_r = _unit_interval_number(decision.get("min_R"), "min_R")
        max_r = _unit_interval_number(decision.get("max_R"), "max_R")
        if min_r > mean_r + 1e-12 or mean_r > max_r + 1e-12:
            raise ValueError("risk decision R interval is invalid")
        normalised.append(
            {
                "branch_id": branch_id,
                "branch_hash": _require_sha256_hex(
                    decision.get("branch_hash"),
                    "branch_hash",
                ),
                "final_R": final_r,
                "mean_R": mean_r,
                "min_R": min_r,
                "max_R": max_r,
                "action_count": _non_negative_int(
                    decision.get("action_count"),
                    "action_count",
                ),
                "topology_edge_count": _optional_non_negative_int(
                    decision.get("topology_edge_count"),
                    "topology_edge_count",
                ),
                "topology_scale": _optional_non_negative_float(
                    decision.get("topology_scale"),
                    "topology_scale",
                ),
                "approved": _required_bool(decision.get("approved"), "approved"),
                "rejection_reasons": list(
                    _normalise_optional_text_sequence(
                        decision.get("rejection_reasons"),
                        "rejection_reasons",
                    )
                ),
            }
        )
    return tuple(normalised)


def _join_multiverse_branch_rows(
    rollout: Mapping[str, object],
    risk: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    branch_records = cast("tuple[dict[str, object], ...]", rollout["branch_records"])
    decisions = cast("tuple[dict[str, object], ...]", risk["branch_decisions"])
    decision_by_id = {cast("str", item["branch_id"]): item for item in decisions}
    rows: list[dict[str, object]] = []
    for branch in branch_records:
        branch_id = cast("str", branch["branch_id"])
        decision = decision_by_id.get(branch_id)
        if decision is None or decision["branch_hash"] != branch["branch_hash"]:
            raise ValueError("risk decision must match every rollout branch")
        for field_name in ("final_R", "mean_R", "min_R", "max_R", "action_count"):
            if decision[field_name] != branch[field_name]:
                raise ValueError(
                    f"risk decision {field_name} must match rollout branch"
                )
        rows.append(
            {
                "branch_index": branch["branch_index"],
                "branch_id": branch_id,
                "branch_hash": branch["branch_hash"],
                "action_count": branch["action_count"],
                "action_labels": branch["action_labels"],
                "topology_edge_count": branch["topology_edge_count"],
                "topology_scale": branch["topology_scale"],
                "final_R": branch["final_R"],
                "mean_R": branch["mean_R"],
                "min_R": branch["min_R"],
                "max_R": branch["max_R"],
                "final_psi": branch["final_psi"],
                "risk_approved": decision["approved"],
                "risk_rejection_reasons": decision["rejection_reasons"],
            }
        )
    return tuple(rows)


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _require_non_empty_text(value, name)


def _optional_sha256_hex(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _require_sha256_hex(value, name)


def _optional_non_negative_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    return _non_negative_int(value, name)


def _optional_non_negative_float(value: object, name: str) -> float | None:
    if value is None:
        return None
    return _non_negative_float(value, name)


def _normalise_optional_text_sequence(value: object, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    return _normalise_text_sequence(value, name)


def _required_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _require_review_svg(value: object) -> str:
    svg = _require_non_empty_text(value, "svg").strip()
    if not svg.startswith("<svg ") or not svg.endswith("</svg>"):
        raise ValueError("svg must be a complete SVG document")
    if "<script" in svg.lower():
        raise ValueError("svg must not contain script content")
    return svg


def _bounded_information_scalar(
    value: object,
    name: str,
    n_bins: int,
) -> float:
    scalar = _non_negative_float(value, name)
    if scalar > float(np.log(n_bins)) + 1e-12:
        raise ValueError(f"{name} must not exceed log(n_bins)")
    return scalar


def _non_negative_float(value: object, name: str) -> float:
    if isinstance(value, (bool, complex)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite non-negative real value")
    scalar = float(value)
    if not isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be a finite non-negative real value")
    return scalar


def _unit_interval_number(value: object, name: str) -> float:
    scalar = _non_negative_float(value, name)
    if scalar > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return scalar


def _positive_int(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    checked = int(value)
    if checked < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return checked


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    checked = int(value)
    if checked < 0:
        raise ValueError(f"{name} must be non-negative")
    return checked


def _normalise_text_sequence(value: object, name: str) -> tuple[str, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of strings")
    return tuple(_require_non_empty_text(item, name) for item in value)


def _normalise_float_sequence(value: object, name: str) -> tuple[float, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of finite numbers")
    if not value:
        raise ValueError(f"{name} must not be empty")
    return tuple(_finite_number(item, name) for item in value)


def _normalise_simplex_sequence(value: object, name: str) -> tuple[float, ...]:
    values = _normalise_float_sequence(value, name)
    if any(item < 0.0 for item in values):
        raise ValueError(f"{name} must be non-negative")
    mass = sum(values)
    if mass <= 0.0:
        raise ValueError(f"{name} must have positive mass")
    if abs(mass - 1.0) > 1e-9:
        raise ValueError(f"{name} must be normalised to unit mass")
    return values


def _normalise_square_float_matrix(
    value: object,
    name: str,
    *,
    expected_size: int,
    positive_diagonal: bool,
) -> tuple[tuple[float, ...], ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a square matrix")
    if len(value) != expected_size:
        raise ValueError(f"{name} row count must match simplex shape")
    rows: list[tuple[float, ...]] = []
    for row_index, row in enumerate(value):
        row_values = _normalise_float_sequence(row, name)
        if len(row_values) != expected_size:
            raise ValueError(f"{name} column count must match simplex shape")
        if positive_diagonal and row_values[row_index] <= 0.0:
            raise ValueError(f"{name} diagonal must be strictly positive")
        rows.append(row_values)
    matrix = np.asarray(rows, dtype=np.float64)
    if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0.0):
        raise ValueError(f"{name} must be symmetric")
    return tuple(rows)


def _normalise_information_geometry_gradient(
    value: object,
    name: str,
) -> tuple[tuple[str, float], ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of knob/value pairs")
    if not value:
        raise ValueError(f"{name} must not be empty")
    gradient: list[tuple[str, float]] = []
    for item in value:
        if isinstance(item, str | bytes) or not isinstance(item, Sequence):
            raise ValueError(f"{name} entries must be knob/value pairs")
        if len(item) != 2:
            raise ValueError(f"{name} entries must be knob/value pairs")
        knob, raw_value = item
        gradient.append(
            (
                _require_non_empty_text(knob, f"{name} knob"),
                _finite_number(raw_value, f"{name} value"),
            )
        )
    return tuple(gradient)


def _single_information_geometry_action(
    record: Mapping[str, object],
    label: str,
) -> dict[str, object]:
    actions = record.get("action_proposals")
    if isinstance(actions, str | bytes) or not isinstance(actions, Sequence):
        raise ValueError(f"{label} action_proposals must be a sequence")
    if len(actions) != 1:
        raise ValueError(f"{label} action_proposals must contain one review action")
    action = actions[0]
    if not isinstance(action, Mapping):
        raise ValueError(f"{label} action_proposals entries must be mappings")
    return {
        "knob": _require_non_empty_text(action.get("knob"), f"{label} knob"),
        "scope": _require_non_empty_text(action.get("scope"), f"{label} scope"),
        "value": _finite_number(action.get("value"), f"{label} action value"),
        "ttl_s": _positive_float(action.get("ttl_s"), f"{label} ttl_s"),
        "justification": _require_non_empty_text(
            action.get("justification"),
            f"{label} justification",
        ),
    }


def _require_sha256_hex(value: object, name: str) -> str:
    text = _require_non_empty_text(value, name)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return text


def _normalise_integrated_information_partition(
    value: object,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError("minimum_partition must contain two index groups")
    if len(value) != 2:
        raise ValueError("minimum_partition must contain two index groups")
    sides: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for side in value:
        if isinstance(side, str | bytes) or not isinstance(side, Sequence):
            raise ValueError("minimum_partition groups must be index sequences")
        if not side:
            raise ValueError("minimum_partition groups must not be empty")
        normalised_side: list[int] = []
        for node in side:
            if isinstance(node, bool) or not isinstance(node, int):
                raise ValueError("minimum_partition indices must be integers")
            checked = int(node)
            if checked < 0:
                raise ValueError("minimum_partition indices must be non-negative")
            if checked in seen:
                raise ValueError("minimum_partition indices must be unique")
            seen.add(checked)
            normalised_side.append(checked)
        sides.append(tuple(sorted(normalised_side)))
    return (sides[0], sides[1])


def _integrated_information_pairwise_shape(
    value: object,
    n_bins: int,
) -> tuple[int, int] | None:
    if value is None:
        return None
    matrix = np.asarray(value)
    if matrix.dtype == np.bool_ or np.issubdtype(matrix.dtype, np.complexfloating):
        raise ValueError("pairwise_mi must be finite real-valued")
    try:
        checked = matrix.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("pairwise_mi must be finite real-valued") from exc
    if checked.ndim != 2 or checked.shape[0] != checked.shape[1]:
        raise ValueError("pairwise_mi must be a square matrix")
    if checked.shape[0] < 2:
        raise ValueError("pairwise_mi must contain at least two oscillators")
    if not np.all(np.isfinite(checked)):
        raise ValueError("pairwise_mi must be finite real-valued")
    if np.any(checked < -1e-12):
        raise ValueError("pairwise_mi must be non-negative")
    if np.any(checked > float(np.log(n_bins)) + 1e-12):
        raise ValueError("pairwise_mi entries must not exceed log(n_bins)")
    if not np.allclose(checked, checked.T, rtol=1e-12, atol=1e-12):
        raise ValueError("pairwise_mi must be symmetric")
    return (int(checked.shape[0]), int(checked.shape[1]))


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
        "python -m scpn_phase_orchestrator.runtime.cli validate binding_spec.yaml"
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
        rest_response = rest.handle_post(envelope_record, headers=headers)
        return rest_response.to_audit_record(), rest.to_audit_record()
    if transport == "grpc":
        grpc = DigitalTwinSyncGrpcAdapter.for_contract(contract, name="studio-grpc")
        grpc_response = grpc.handle_unary(envelope_record, metadata=headers)
        return grpc_response.to_audit_record(), grpc.to_audit_record()
    if transport == "kafka":
        kafka = DigitalTwinSyncKafkaAdapter.for_contract(contract, name="studio-kafka")
        kafka_response = kafka.handle_message(
            {"topic": kafka.topic, "value": envelope_record},
            headers=headers,
        )
        return kafka_response.to_audit_record(), kafka.to_audit_record()
    if transport == "hardware":
        hardware = DigitalTwinSyncHardwareAdapter.for_contract(
            contract,
            name="studio-hardware",
            device_ids=("studio-review-device",),
        )
        hardware_response = hardware.handle_frame(
            {
                "device_id": "studio-review-device",
                "safety_interlock": True,
                "value": envelope_record,
            },
            headers=headers,
        )
        return hardware_response.to_audit_record(), hardware.to_audit_record()
    raise ValueError(f"connector transport {transport!r} is not a live runtime")


def _mapping_count(mapping: Mapping[str, object], name: str) -> int:
    return _non_negative_int(mapping.get(name), name)


def _canvas_graph_count(result: StudioReplayResult, name: str) -> int:
    return _mapping_count(result.canvas_graph, name)


def _require_sequence(value: object, name: str) -> Sequence[object]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
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
        "coupling_count_before": _canvas_graph_count(result, "edge_count"),
        "coupling_count_after": _canvas_graph_count(result, "edge_count"),
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
        except (BindingLoadError, ValueError, TypeError, OSError) as exc:
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


def _positive_float(value: object, name: str) -> float:
    number = _finite_number(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _require_non_empty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()
