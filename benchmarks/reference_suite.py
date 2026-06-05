# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — v1 reference benchmark suite

from __future__ import annotations

import importlib
import json
import platform
import sys
import tempfile
import time
from collections.abc import Iterable, Mapping
from datetime import date
from hashlib import sha256
from pathlib import Path
from typing import NamedTuple, TypedDict

import numpy as np
from numpy.typing import NDArray

from benchmarks.chimera_benchmark import benchmark_chimera_polyglot_parity_gate
from benchmarks.dimension_benchmark import benchmark_dimension_polyglot_parity_gate
from benchmarks.embedding_benchmark import benchmark_embedding_polyglot_parity_gate
from benchmarks.entropy_prod_benchmark import (
    benchmark_entropy_production_polyglot_parity_gate,
)
from benchmarks.hodge_benchmark import benchmark_hodge_polyglot_parity_gate
from benchmarks.itpc_benchmark import benchmark_itpc_polyglot_parity_gate
from benchmarks.lyapunov_benchmark import benchmark_lyapunov_polyglot_parity_gate
from benchmarks.npe_benchmark import benchmark_npe_polyglot_parity_gate
from benchmarks.order_params_benchmark import (
    benchmark_order_parameter_polyglot_parity_gate,
)
from benchmarks.pha_c_acceptance_benchmark import (
    benchmark_pha_c_acceptance_polyglot_gate,
)
from benchmarks.pha_c_handoff_benchmark import (
    benchmark_pha_c_handoff_polyglot_parity_gate,
)
from benchmarks.pha_c_timeline_benchmark import (
    benchmark_pha_c_timeline_polyglot_parity_gate,
)
from benchmarks.recurrence_benchmark import benchmark_recurrence_polyglot_parity_gate
from benchmarks.spatial_modulator_benchmark import (
    benchmark_spatial_modulator_polyglot_parity_gate,
)
from benchmarks.spectral_benchmark import benchmark_spectral_polyglot_parity_gate
from benchmarks.transfer_entropy_benchmark import (
    benchmark_transfer_entropy_polyglot_parity_gate,
)
from benchmarks.upde_doppler_benchmark import benchmark_upde_doppler_polyglot_gate
from benchmarks.upde_moving_frame_benchmark import (
    benchmark_upde_moving_frame_polyglot_gate,
)
from benchmarks.upde_time_varying_omega_benchmark import (
    benchmark_upde_time_varying_omega_polyglot_gate,
)
from benchmarks.winding_benchmark import benchmark_winding_polyglot_parity_gate
from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.adapters.hybrid_cocompiler import (
    audit_hybrid_target_readiness,
    build_hybrid_cocompiler_manifest,
    build_hybrid_operator_handoff_package,
)
from scpn_phase_orchestrator.adapters.quantum_control_bridge import QuantumControlBridge
from scpn_phase_orchestrator.adapters.snn_bridge import SNNControllerBridge
from scpn_phase_orchestrator.autotune.binding_proposal import (
    propose_binding_from_time_series_csv,
)
from scpn_phase_orchestrator.autotune.learners import (
    generate_hybrid_physics_proposal,
    generate_ppo_like_proposal,
    generate_sac_like_proposal,
)
from scpn_phase_orchestrator.autotune.reward import (
    KnobPolicyCandidate,
    PolicyProposalConfig,
    RewardObservation,
    SafetyConstraintConfig,
)
from scpn_phase_orchestrator.binding.semantic import compile_symbolic_binding
from scpn_phase_orchestrator.exceptions import PolicyError
from scpn_phase_orchestrator.meta.transfer import (
    CrossDomainMetaTransfer,
    MetaPolicyRecord,
)
from scpn_phase_orchestrator.monitor.stl import (
    STLActionProjectionTemplate,
    synthesise_stl_closed_loop_plan,
    synthesise_stl_monitoring_automaton,
)
from scpn_phase_orchestrator.plugins import (
    PluginCapability,
    PluginManifest,
    build_plugin_marketplace_catalog,
    build_rust_plugin_registry,
    build_rust_plugin_runtime_handoff,
)
from scpn_phase_orchestrator.supervisor.alignment import (
    ValueAlignmentPolicy,
    ValueConstraint,
    calibrate_value_alignment_replay_evidence,
)
from scpn_phase_orchestrator.supervisor.causal import (
    build_temporal_causal_hypergraph_experiment,
)
from scpn_phase_orchestrator.supervisor.federated_dp_noise_service import (
    DpNoiseNodePrivacyBudget,
    DpNoiseServiceRequestManifest,
    build_dp_noise_service_deployment_preflight_manifest,
    build_dp_noise_service_manifest,
)
from scpn_phase_orchestrator.supervisor.federated_secure_aggregation import (
    build_federated_secure_aggregation_manifest,
    build_federated_secure_aggregation_preflight_manifest,
)
from scpn_phase_orchestrator.supervisor.federated_transport import (
    build_signed_transport_envelopes,
    build_transport_deployment_preflight_manifest,
    replay_federated_transport_batch,
)
from scpn_phase_orchestrator.supervisor.formal_export import (
    FormalCheckerResult,
    FormalSafetyProperty,
    audit_formal_checker_availability,
    build_formal_verification_package,
    build_runtime_control_certificate,
    export_petri_net_prism,
    export_petri_net_tla,
    export_policy_rules_prism,
    export_policy_rules_tla,
    export_stl_specs_prism,
)
from scpn_phase_orchestrator.supervisor.lineage import (
    build_autopoietic_lineage_replay_corpus,
    build_autopoietic_lineage_sandbox,
    build_intergenerational_policy_inheritance,
    build_intergenerational_policy_inheritance_history,
)
from scpn_phase_orchestrator.supervisor.multiverse import (
    MultiverseBranchSpec,
    simulate_multiverse_counterfactual_branches,
)
from scpn_phase_orchestrator.supervisor.multiverse_examples import (
    build_multiverse_domain_scenarios,
)
from scpn_phase_orchestrator.supervisor.multiverse_risk import (
    MultiverseRiskThresholds,
    evaluate_multiverse_branch_risk,
)
from scpn_phase_orchestrator.supervisor.petri_net import (
    Arc,
    Guard,
    Marking,
    PetriNet,
    Place,
    Transition,
)
from scpn_phase_orchestrator.supervisor.policy_rules import (
    CompoundCondition,
    PolicyAction,
    PolicyCondition,
    PolicyRule,
    PolicySTLSpec,
)
from scpn_phase_orchestrator.supervisor.strange_loop import (
    evaluate_strange_loop_drift_scenarios,
)
from scpn_phase_orchestrator.upde.bayesian import (
    BayesianUPDEConfig,
    audit_bayesian_backend_status,
    bayesian_upde_run,
    fit_gaussian_upde_posterior,
)
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter
from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine
from tools.formal_model_checker_ci import build_domainpack_formal_packages

ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "results" / "reference_suite.json"
BENCHMARK_COMMAND = "PYTHONPATH=.:src python benchmarks/reference_suite.py"
REFERENCE_SUITE_VERSION = "reference_suite_v1"


BenchmarkValue = float | int | str
BenchmarkRecord = dict[str, BenchmarkValue]


class AutoBindingAcceptanceThresholds(NamedTuple):
    min_extractor_coverage: float
    min_expected_edge_recall: float
    max_validation_errors: int
    min_sample_count: int
    max_proposed_edge_multiplier: float


class AutoBindingFixture(NamedTuple):
    domain: str
    csv_text: str
    sample_rate_hz: float | None
    expected_edges: frozenset[tuple[str, str]]
    thresholds: AutoBindingAcceptanceThresholds

    @property
    def sample_count(self) -> int:
        return max(0, len(self.csv_text.splitlines()) - 1)


class ReplayLearnerBenchmarkThresholds(NamedTuple):
    min_acceptance_rate: float
    min_reward_improvement: float
    max_unsafe_acceptances: int
    max_lyapunov_exponent: float
    min_stl_robustness: float
    max_safety_cost: float
    require_non_actuating: bool
    require_safety_evidence: bool


class ReplayLearnerBenchmarkScenario(NamedTuple):
    name: str
    seed_candidate: KnobPolicyCandidate
    min_coherence: float
    min_reward: float
    critical_coupling_estimate: float
    ppo_seed: int
    sac_seed: int
    hybrid_seed: int


class BayesianPosteriorFitThresholds(NamedTuple):
    max_residual_rmse: float
    max_omega_mean_abs_error: float
    max_knm_mean_abs_error: float
    max_credible_interval_width: float
    min_rollout_sample_count: int


class BayesianBackendFailClosedThresholds(NamedTuple):
    min_available_backends: int
    required_fail_closed_backends: frozenset[str]
    max_unexpected_reserved_successes: int


class FormalExportThresholds(NamedTuple):
    min_artifact_count: int
    min_fail_closed_count: int
    min_identifier_map_count: int
    min_package_property_count: int
    min_checker_command_count: int
    min_checker_availability_count: int
    min_missing_checker_count: int
    min_runtime_certificate_count: int
    require_deterministic_hash: bool
    require_checker_execution_disabled: bool
    require_runtime_certificate_verified: bool


class DomainFormalExportThresholds(NamedTuple):
    min_domain_count: int
    min_artifacts_per_domain: int
    min_rules_per_domain: int
    min_stl_specs_per_domain: int
    min_package_property_count: int
    min_checker_command_count: int
    require_deterministic_hash: bool


class DomainFormalExportFixture(NamedTuple):
    domain: str
    rules: tuple[PolicyRule, ...]
    stl_specs: tuple[PolicySTLSpec, ...]
    required_labels: tuple[str, ...]


class STLClosedLoopThresholds(NamedTuple):
    min_plan_count: int
    min_projected_action_count: int
    min_runtime_gate_checked_count: int
    min_runtime_mapped_command_count: int
    min_blocked_reason_count: int
    require_non_actuating: bool
    require_runtime_execution_disabled: bool
    require_deterministic_hash: bool


class HybridCocompilerThresholds(NamedTuple):
    min_target_backend_count: int
    min_quantum_term_count: int
    min_neuromorphic_sample_count: int
    min_blocked_probe_count: int
    require_non_actuating: bool


class QuantumTargetReadinessThresholds(NamedTuple):
    min_ready_count: int
    min_blocked_count: int
    min_blocked_reason_count: int
    min_operator_command_count: int
    require_non_executing: bool
    require_deterministic_hash: bool


class NeuromorphicTargetReadinessThresholds(NamedTuple):
    min_ready_count: int
    min_blocked_count: int
    min_blocked_reason_count: int
    min_operator_command_count: int
    require_non_executing: bool
    require_deterministic_hash: bool


class HybridTargetReadinessThresholds(NamedTuple):
    min_ready_count: int
    min_blocked_count: int
    min_blocked_reason_count: int
    min_operator_command_count: int
    require_non_executing: bool
    require_deterministic_hash: bool
    require_component_hash_linked: bool


class HybridOperatorHandoffThresholds(NamedTuple):
    min_ready_package_count: int
    min_blocked_package_count: int
    min_blocked_reason_count: int
    min_operator_command_count: int
    require_non_executing: bool
    require_deterministic_hash: bool
    require_hash_chain_linked: bool


class HybridEntanglementOrderThresholds(NamedTuple):
    max_product_entropy: float
    min_bell_entropy: float
    min_entropy_gap: float
    min_record_count: int
    require_non_actuating: bool
    require_execution_disabled: bool
    require_claim_boundary: bool
    require_deterministic_hash: bool


class ValueAlignmentReplayCalibrationThresholds(NamedTuple):
    min_replay_case_count: int
    min_approved_case_count: int
    min_blocked_case_count: int
    min_threshold_fallback_case_count: int
    min_fallback_applied_case_count: int
    require_review_only: bool
    require_deterministic_hash: bool


class AutopoieticLineageSandboxThresholds(NamedTuple):
    min_child_candidate_count: int
    min_accepted_child_count: int
    min_rejected_child_count: int
    min_policy_diff_count: int
    min_replay_domain_count: int
    require_review_only: bool
    require_deterministic_hash: bool


class IntergenerationalInheritanceThresholds(NamedTuple):
    min_manifest_count: int
    min_signed_metadata_count: int
    min_policy_gene_count: int
    min_history_record_count: int
    min_replay_domain_count: int
    min_fitness_score: float
    require_review_only: bool
    require_deterministic_hash: bool


class TemporalCausalHypergraphThresholds(NamedTuple):
    min_manifest_count: int
    min_accepted_hyperedge_count: int
    min_baseline_edge_count: int
    min_baseline_family_count: int
    require_research_only: bool
    require_deterministic_hash: bool


class FederatedProductionBoundaryThresholds(NamedTuple):
    min_boundary_surface_count: int
    min_transport_envelope_count: int
    min_secure_accepted_node_count: int
    min_dp_noise_vector_count: int
    require_transport_execution_disabled: bool
    require_secure_execution_disabled: bool
    require_service_execution_disabled: bool
    require_raw_data_export_disabled: bool
    require_operator_review: bool
    require_non_actuating: bool
    require_deterministic_hash: bool


class FederatedDeploymentPreflightThresholds(NamedTuple):
    min_preflight_surface_count: int
    min_transport_preflight_count: int
    min_secure_preflight_count: int
    min_dp_preflight_count: int
    require_transport_execution_disabled: bool
    require_secure_execution_disabled: bool
    require_service_execution_disabled: bool
    require_raw_data_export_disabled: bool
    require_operator_review: bool
    require_non_actuating: bool
    require_deterministic_hash: bool


class MorphogeneticDomainDemoThresholds(NamedTuple):
    min_demo_count: int
    min_total_grown_edges: int
    min_total_shrunk_edges: int
    require_non_actuating: bool
    require_snapshot_rows: bool
    require_deterministic_hash: bool


class IntegratedInformationReplayCorpusThresholds(NamedTuple):
    min_domain_count: int
    min_record_count: int
    min_ordering_evidence_count: int
    require_non_actuating: bool
    require_claim_boundary: bool
    require_deterministic_hash: bool


class ToposSemanticBindingThresholds(NamedTuple):
    min_semantic_report_count: int
    min_policy_object_count: int
    min_domain_example_count: int
    min_obligation_count: int
    require_non_actuating: bool
    require_proof_boundary: bool
    require_deterministic_hash: bool


class SelfModelDigitalTwinThresholds(NamedTuple):
    min_scenario_count: int
    max_breach_count: int
    max_max_observed_error: float
    require_non_actuating: bool
    require_operator_review: bool
    require_execution_disabled: bool
    require_deterministic_hash: bool


class StrangeLoopDriftScenarioThresholds(NamedTuple):
    min_scenario_count: int
    min_long_run_step_count: int
    min_passed_scenario_count: int
    require_non_actuating: bool
    require_execution_disabled: bool
    require_deterministic_hash: bool


class InformationGeometryControlThresholds(NamedTuple):
    min_scenario_count: int
    min_finite_metric_count: int
    min_action_evidence_count: int
    require_non_actuating: bool
    require_execution_disabled: bool
    require_claim_boundary: bool
    require_deterministic_hash: bool
    require_jax_backend_parity: bool


class MultiverseCounterfactualThresholds(NamedTuple):
    min_branch_count: int
    min_domain_scenario_count: int
    min_approved_branch_count: int
    min_rejected_branch_count: int
    require_non_actuating: bool
    require_execution_disabled: bool
    require_deterministic_hash: bool
    require_jax_backend_parity: bool


class PluginEcosystemThresholds(NamedTuple):
    min_plugin_count: int
    min_capability_count: int
    min_handoff_target_hash_count: int
    min_blocked_handoff_count: int
    required_capability_kinds: frozenset[str]
    min_incompatible_count: int
    require_deterministic_hash: bool
    require_loading_disabled: bool


class SemanticRetrievalThresholds(NamedTuple):
    min_evidence_count: int
    min_ranked_record_count: int
    min_feature_complete_count: int
    require_domainpack_top_rank: bool
    require_deterministic_hash: bool


class MetaPackageManifestThresholds(NamedTuple):
    min_record_count: int
    min_domain_count: int
    min_feature_key_count: int
    min_knob_count: int
    require_package_digest_match: bool
    require_execution_disabled: bool
    require_deterministic_hash: bool


class MetaAuditCorpusThresholds(NamedTuple):
    min_record_count: int
    min_domain_count: int
    min_feature_key_count: int
    min_knob_count: int
    min_neighbour_count: int
    min_confidence: float
    required_top_domain: str
    require_deterministic_hash: bool


class EvolutionarySupervisorSearchThresholds(NamedTuple):
    min_scenario_count: int
    min_candidate_count: int
    min_accepted_count: int
    min_rejected_count: int
    min_stl_filter_rejected_count: int
    min_counterfactual_filter_rejected_count: int
    require_non_actuating: bool
    require_execution_disabled: bool
    require_operator_review: bool
    require_live_merge_disabled: bool
    require_hot_patch_disabled: bool
    require_deterministic_hash: bool


class EvolutionaryMutationGrammarThresholds(NamedTuple):
    min_grammar_count: int
    min_candidate_count: int
    min_mutation_kind_count: int
    require_non_actuating: bool
    require_execution_disabled: bool
    require_operator_review: bool
    require_live_merge_disabled: bool
    require_hot_patch_disabled: bool
    require_deterministic_hash: bool


class FederatedMetaOrchestratorThresholds(NamedTuple):
    min_node_count: int
    min_accepted_node_count: int
    min_policy_key_count: int
    max_rejected_node_count: int
    max_privacy_budget_spent: float
    require_non_actuating: bool
    require_execution_disabled: bool
    require_operator_review: bool
    require_live_transport_disabled: bool
    require_raw_data_export_disabled: bool
    require_no_raw_time_series: bool
    require_deterministic_hash: bool


class SheafObstructionBenchmarkThresholds(NamedTuple):
    min_demo_count: int
    min_summary_count: int
    min_top_residual_edge_count: int
    min_critical_count: int
    min_obstruction_delta: float
    min_control_energy_reduction: float
    max_nominal_obstruction_score: float
    require_non_actuating: bool
    require_execution_disabled: bool
    require_operator_review: bool
    require_deterministic_hash: bool


class ReferenceSuiteResult(TypedDict):
    metadata: dict[str, str]
    benchmarks: dict[str, BenchmarkRecord]


def build_benchmark_metadata(*, snapshot_date: str | None = None) -> dict[str, str]:
    return {
        "suite_version": REFERENCE_SUITE_VERSION,
        "snapshot_date": snapshot_date or date.today().isoformat(),
        "command": BENCHMARK_COMMAND,
        "backend": "python_numpy",
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "executable": sys.executable,
        "benchmark_evidence_kind": "local_regression_non_isolated",
        "isolation_method": "none",
        "production_timing_claim": "false",
    }


def benchmark_meta_transfer_package_manifest_quality() -> dict[str, float | int | str]:
    """Benchmark meta-transfer package manifest readiness gates."""
    thresholds = MetaPackageManifestThresholds(
        min_record_count=4,
        min_domain_count=4,
        min_feature_key_count=5,
        min_knob_count=4,
        require_package_digest_match=True,
        require_execution_disabled=True,
        require_deterministic_hash=True,
    )
    records = _meta_transfer_package_records()
    t0 = time.perf_counter()
    model = CrossDomainMetaTransfer.fit(records)
    package_payload = model.to_json_package()
    manifest = model.to_package_manifest()
    repeated_manifest = CrossDomainMetaTransfer.fit(records).to_package_manifest()
    elapsed = time.perf_counter() - t0

    manifest_record = manifest.to_audit_record()
    repeated_record = repeated_manifest.to_audit_record()
    package_sha256 = sha256(package_payload.encode("utf-8")).hexdigest()
    package_digest_matches = int(manifest.package_sha256 == package_sha256)
    execution_disabled = int(manifest.execution_permitted is False)
    manifest_sha256 = sha256(
        json.dumps(manifest_record, sort_keys=True).encode("utf-8")
    ).hexdigest()
    repeated_sha256 = sha256(
        json.dumps(repeated_record, sort_keys=True).encode("utf-8")
    ).hexdigest()
    deterministic_hash = int(manifest_sha256 == repeated_sha256)
    summary = manifest.training_summary
    acceptance_passed = int(
        summary.record_count >= thresholds.min_record_count
        and summary.domain_count >= thresholds.min_domain_count
        and len(summary.feature_keys) >= thresholds.min_feature_key_count
        and len(summary.knob_keys) >= thresholds.min_knob_count
        and package_digest_matches == int(thresholds.require_package_digest_match)
        and execution_disabled == int(thresholds.require_execution_disabled)
        and deterministic_hash == int(thresholds.require_deterministic_hash)
    )

    return {
        "suite": "meta_transfer_package_manifest_quality",
        "record_count": summary.record_count,
        "domain_count": summary.domain_count,
        "feature_key_count": len(summary.feature_keys),
        "knob_count": len(summary.knob_keys),
        "package_bytes": len(package_payload.encode("utf-8")),
        "wall_time_s": elapsed,
        "steps_per_second": summary.record_count / elapsed,
        "manifest_schema": str(manifest_record["schema"]),
        "package_name": manifest.package_name,
        "import_target": manifest.import_target,
        "console_script": manifest.console_script,
        "package_sha256": manifest.package_sha256,
        "manifest_sha256": manifest_sha256,
        "package_digest_matches": package_digest_matches,
        "execution_disabled": execution_disabled,
        "deterministic_hash": deterministic_hash,
        "acceptance_passed": acceptance_passed,
        "acceptance_thresholds_json": json.dumps(
            {
                "min_domain_count": thresholds.min_domain_count,
                "min_feature_key_count": thresholds.min_feature_key_count,
                "min_knob_count": thresholds.min_knob_count,
                "min_record_count": thresholds.min_record_count,
                "require_deterministic_hash": thresholds.require_deterministic_hash,
                "require_execution_disabled": thresholds.require_execution_disabled,
                "require_package_digest_match": thresholds.require_package_digest_match,
            },
            sort_keys=True,
        ),
        "manifest_json": json.dumps(manifest_record, sort_keys=True),
    }


def benchmark_meta_transfer_audit_corpus_quality() -> dict[str, float | int | str]:
    """Benchmark nested audit-history corpus loading and proposal quality."""
    thresholds = MetaAuditCorpusThresholds(
        min_record_count=6,
        min_domain_count=4,
        min_feature_key_count=5,
        min_knob_count=4,
        min_neighbour_count=3,
        min_confidence=0.97,
        required_top_domain="power_grid",
        require_deterministic_hash=True,
    )
    query = {
        "coherence": 0.9,
        "event_rate": 0.07,
        "load_variance": 0.34,
        "phase_spread": 0.1,
        "safety_margin": 0.73,
    }
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="spo-meta-audit-corpus-") as tmp:
        root = Path(tmp)
        _write_meta_transfer_audit_corpus(root)
        model = CrossDomainMetaTransfer.fit_audit_directory(
            root,
            min_records=thresholds.min_record_count,
        )
        proposal = model.propose(query, k_neighbours=thresholds.min_neighbour_count)
        repeated = CrossDomainMetaTransfer.fit_audit_directory(
            root,
            min_records=thresholds.min_record_count,
        ).propose(query, k_neighbours=thresholds.min_neighbour_count)
    elapsed = time.perf_counter() - t0

    proposal_record = proposal.to_audit_record()
    repeated_record = repeated.to_audit_record()
    proposal_sha256 = sha256(
        json.dumps(proposal_record, sort_keys=True).encode("utf-8")
    ).hexdigest()
    deterministic_hash = int(
        proposal_sha256
        == sha256(
            json.dumps(repeated_record, sort_keys=True).encode("utf-8")
        ).hexdigest()
    )
    summary = model.training_summary
    top_domain = proposal.neighbours[0][0] if proposal.neighbours else ""
    acceptance_passed = int(
        summary.record_count >= thresholds.min_record_count
        and summary.domain_count >= thresholds.min_domain_count
        and len(summary.feature_keys) >= thresholds.min_feature_key_count
        and len(summary.knob_keys) >= thresholds.min_knob_count
        and len(proposal.neighbours) >= thresholds.min_neighbour_count
        and proposal.confidence >= thresholds.min_confidence
        and top_domain == thresholds.required_top_domain
        and deterministic_hash == int(thresholds.require_deterministic_hash)
    )

    return {
        "suite": "meta_transfer_audit_corpus_quality",
        "record_count": summary.record_count,
        "domain_count": summary.domain_count,
        "feature_key_count": len(summary.feature_keys),
        "knob_count": len(summary.knob_keys),
        "proposal_knob_count": len(proposal.knobs),
        "neighbour_count": len(proposal.neighbours),
        "wall_time_s": elapsed,
        "steps_per_second": summary.record_count / elapsed,
        "top_neighbour_domain": top_domain,
        "confidence": proposal.confidence,
        "proposal_sha256": proposal_sha256,
        "deterministic_hash": deterministic_hash,
        "acceptance_passed": acceptance_passed,
        "acceptance_thresholds_json": json.dumps(
            {
                "min_confidence": thresholds.min_confidence,
                "min_domain_count": thresholds.min_domain_count,
                "min_feature_key_count": thresholds.min_feature_key_count,
                "min_knob_count": thresholds.min_knob_count,
                "min_neighbour_count": thresholds.min_neighbour_count,
                "min_record_count": thresholds.min_record_count,
                "required_top_domain": thresholds.required_top_domain,
                "require_deterministic_hash": thresholds.require_deterministic_hash,
            },
            sort_keys=True,
        ),
        "proposal_json": json.dumps(proposal_record, sort_keys=True),
        "training_summary_json": json.dumps(
            summary.to_audit_record(),
            sort_keys=True,
        ),
    }


def benchmark_semantic_retrieval_ranking_quality() -> dict[str, float | int | str]:
    """Benchmark symbolic compiler retrieval ranking diagnostics."""
    thresholds = SemanticRetrievalThresholds(
        min_evidence_count=3,
        min_ranked_record_count=3,
        min_feature_complete_count=3,
        require_domainpack_top_rank=True,
        require_deterministic_hash=True,
    )
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="spo-semantic-ranking-") as tmp:
        root = Path(tmp)
        retrieval_root, docs_root = _semantic_retrieval_fixture(root)
        artefacts = compile_symbolic_binding(
            "A 2-layer power grid stability controller with renewable demand",
            name="semantic_retrieval_benchmark",
            oscillators_per_layer=2,
            dry_run_steps=2,
            retrieval_root=retrieval_root,
            docs_root=docs_root,
        )
        repeated = compile_symbolic_binding(
            "A 2-layer power grid stability controller with renewable demand",
            name="semantic_retrieval_benchmark",
            oscillators_per_layer=2,
            dry_run_steps=2,
            retrieval_root=retrieval_root,
            docs_root=docs_root,
        )
    elapsed = time.perf_counter() - t0

    records = artefacts.audit_record["retrieval_evidence"]
    repeated_records = repeated.audit_record["retrieval_evidence"]
    ranked_record_count = sum(
        int(record.get("rank") == index)
        for index, record in enumerate(records, start=1)
    )
    feature_keys = {
        "matched_term_count",
        "name_match_count",
        "phrase_match",
        "prompt_term_count",
        "source_priority",
        "term_density",
    }
    feature_complete_count = sum(
        int(feature_keys <= set(record.get("ranking_features", {})))
        for record in records
    )
    top_record = records[0] if records else {}
    domainpack_top_rank = int(
        top_record.get("source") == "domainpack"
        and top_record.get("domainpack") == "power_grid"
        and top_record.get("rank") == 1
    )
    ranking_projection = _semantic_ranking_projection(records)
    repeated_projection = _semantic_ranking_projection(repeated_records)
    ranking_hash = sha256(
        json.dumps(ranking_projection, sort_keys=True).encode()
    ).hexdigest()
    deterministic_hash = int(
        ranking_hash
        == sha256(json.dumps(repeated_projection, sort_keys=True).encode()).hexdigest()
    )
    acceptance_passed = int(
        len(records) >= thresholds.min_evidence_count
        and ranked_record_count >= thresholds.min_ranked_record_count
        and feature_complete_count >= thresholds.min_feature_complete_count
        and domainpack_top_rank == int(thresholds.require_domainpack_top_rank)
        and deterministic_hash == int(thresholds.require_deterministic_hash)
        and artefacts.audit_record["confidence_factors"]["retrieval_score"] > 0.0
    )

    return {
        "suite": "semantic_retrieval_ranking_quality",
        "evidence_count": len(records),
        "wall_time_s": elapsed,
        "steps_per_second": len(records) / elapsed,
        "ranked_record_count": ranked_record_count,
        "feature_complete_count": feature_complete_count,
        "domainpack_top_rank": domainpack_top_rank,
        "deterministic_hash": deterministic_hash,
        "ranking_sha256": ranking_hash,
        "top_source": str(top_record.get("source", "")),
        "top_domainpack": str(top_record.get("domainpack", "")),
        "retrieval_score": artefacts.audit_record["confidence_factors"][
            "retrieval_score"
        ],
        "acceptance_passed": acceptance_passed,
        "acceptance_thresholds_json": json.dumps(
            {
                "min_evidence_count": thresholds.min_evidence_count,
                "min_feature_complete_count": thresholds.min_feature_complete_count,
                "min_ranked_record_count": thresholds.min_ranked_record_count,
                "require_deterministic_hash": thresholds.require_deterministic_hash,
                "require_domainpack_top_rank": thresholds.require_domainpack_top_rank,
            },
            sort_keys=True,
        ),
        "ranking_projection_json": json.dumps(ranking_projection, sort_keys=True),
    }


def _validate_reference_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{name} must be a non-boolean integer")
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{name} must be positive")
    return parsed


def _validate_reference_positive_float(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (float, int, np.floating, np.integer),
    ):
        raise ValueError(f"{name} must be a positive finite real")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be a positive finite real")
    return parsed


def _phase_delta(angle: float) -> float:
    return float(((angle + np.pi) % (2.0 * np.pi)) - np.pi)


def _run_kuramoto_reference_case(
    *,
    phases: NDArray[np.float64],
    omegas: NDArray[np.float64],
    knm: NDArray[np.float64],
    dt: float,
    n_steps: int,
) -> NDArray[np.float64]:
    alpha = np.zeros_like(knm)
    engine = UPDEEngine(n_oscillators=phases.size, dt=dt, method="rk4")
    state = np.ascontiguousarray(phases, dtype=np.float64)
    for _ in range(n_steps):
        state = engine.step(state, omegas, knm, 0.0, 0.0, alpha)
    return state


def benchmark_kuramoto_reference(
    n_oscillators: int = 64, n_steps: int = 1000, dt: float = 0.01
) -> dict[str, float | int | str]:
    n_oscillators = _validate_reference_positive_int(
        n_oscillators,
        name="n_oscillators",
    )
    if n_oscillators < 2:
        raise ValueError("n_oscillators must be at least 2")
    n_steps = _validate_reference_positive_int(n_steps, name="n_steps")
    dt = _validate_reference_positive_float(dt, name="dt")

    rng = np.random.default_rng(42)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_oscillators)
    omegas = np.zeros(n_oscillators)
    knm = np.full((n_oscillators, n_oscillators), 0.4, dtype=float)
    np.fill_diagonal(knm, 0.0)

    t0 = time.perf_counter()
    phases = _run_kuramoto_reference_case(
        phases=phases,
        omegas=omegas,
        knm=knm,
        dt=dt,
        n_steps=n_steps,
    )
    elapsed = time.perf_counter() - t0
    final_r, _ = compute_order_parameter(phases)

    lock_coupling = 0.35
    lock_omegas = np.array([-0.2, 0.2], dtype=np.float64)
    lock_delta_omega = float(lock_omegas[1] - lock_omegas[0])
    lock_threshold = abs(lock_delta_omega) / 2.0
    lock_steps = max(1000, n_steps)
    lock_phases = _run_kuramoto_reference_case(
        phases=np.array([0.0, 0.1], dtype=np.float64),
        omegas=lock_omegas,
        knm=np.array([[0.0, lock_coupling], [lock_coupling, 0.0]], dtype=np.float64),
        dt=dt,
        n_steps=lock_steps,
    )
    predicted_lock_lag = float(np.arcsin(lock_delta_omega / (2.0 * lock_coupling)))
    observed_lock_lag = _phase_delta(float(lock_phases[1] - lock_phases[0]))
    lock_error = abs(_phase_delta(observed_lock_lag - predicted_lock_lag))
    thresholds = {
        "max_two_oscillator_lock_error_rad": 1.0e-2,
        "min_identical_final_order_parameter": 0.99,
        "require_analytic_lock_condition": True,
        "require_bounded_order_parameter": True,
        "require_zero_self_coupling": True,
    }
    zero_self_coupling = bool(np.allclose(np.diag(knm), 0.0, rtol=0.0, atol=0.0))
    analytic_lock_condition = bool(abs(lock_delta_omega) < 2.0 * lock_coupling)
    identical_coherence_passed = bool(
        float(final_r) >= thresholds["min_identical_final_order_parameter"]
    )
    analytic_lock_passed = bool(
        lock_error <= thresholds["max_two_oscillator_lock_error_rad"]
    )
    bounded_order_parameter = bool(0.0 <= float(final_r) <= 1.0)
    acceptance_passed = int(
        zero_self_coupling
        and analytic_lock_condition
        and identical_coherence_passed
        and analytic_lock_passed
        and bounded_order_parameter
    )
    benchmark_payload = {
        "analytic_lock_condition": analytic_lock_condition,
        "final_order_parameter": float(final_r),
        "lock_error": lock_error,
        "lock_steps": lock_steps,
        "observed_lock_lag": observed_lock_lag,
        "predicted_lock_lag": predicted_lock_lag,
        "thresholds": thresholds,
        "zero_self_coupling": zero_self_coupling,
    }
    benchmark_sha = sha256(
        json.dumps(benchmark_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()

    return {
        "suite": "kuramoto_reference_strogatz_2000",
        "n_oscillators": n_oscillators,
        "n_steps": n_steps,
        "wall_time_s": elapsed,
        "steps_per_second": n_steps / elapsed,
        "final_order_parameter": float(final_r),
        "two_oscillator_coupling": lock_coupling,
        "two_oscillator_delta_omega": lock_delta_omega,
        "two_oscillator_lock_threshold": lock_threshold,
        "two_oscillator_lock_steps": lock_steps,
        "two_oscillator_predicted_lag_rad": predicted_lock_lag,
        "two_oscillator_observed_lag_rad": observed_lock_lag,
        "two_oscillator_lock_error_rad": lock_error,
        "zero_self_coupling": int(zero_self_coupling),
        "analytic_lock_condition": int(analytic_lock_condition),
        "identical_coherence_passed": int(identical_coherence_passed),
        "analytic_lock_passed": int(analytic_lock_passed),
        "bounded_order_parameter": int(bounded_order_parameter),
        "acceptance_passed": acceptance_passed,
        "acceptance_thresholds_json": json.dumps(thresholds, sort_keys=True),
        "benchmark_sha256": benchmark_sha,
    }


def benchmark_stuart_landau_reference(
    n_oscillators: int = 64, n_steps: int = 1000, dt: float = 0.01
) -> dict[str, float | int | str]:
    n_oscillators = _validate_reference_positive_int(
        n_oscillators,
        name="n_oscillators",
    )
    n_steps = _validate_reference_positive_int(n_steps, name="n_steps")
    dt = _validate_reference_positive_float(dt, name="dt")

    rng = np.random.default_rng(7)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_oscillators)
    radius = np.ones(n_oscillators)
    state = np.concatenate((theta, radius))
    omegas = np.full(n_oscillators, 1.0)
    mu = np.full(n_oscillators, 0.5)
    knm = np.full((n_oscillators, n_oscillators), 0.2, dtype=float)
    knm_r = np.full((n_oscillators, n_oscillators), 0.2, dtype=float)
    alpha = np.zeros((n_oscillators, n_oscillators), dtype=float)
    np.fill_diagonal(knm, 0.0)
    np.fill_diagonal(knm_r, 0.0)
    engine = StuartLandauEngine(n_oscillators=n_oscillators, dt=dt, method="rk4")

    t0 = time.perf_counter()
    for _ in range(n_steps):
        state = engine.step(
            state, omegas, mu, knm, knm_r, zeta=0.0, psi=0.0, alpha=alpha, epsilon=1.0
        )
    elapsed = time.perf_counter() - t0
    final_r = float(engine.compute_mean_amplitude(state))

    limit_mu = 0.5
    expected_limit_radius = float(np.sqrt(limit_mu))
    limit_state = np.concatenate(
        (
            np.linspace(0.0, np.pi, 4, dtype=np.float64),
            np.array([0.2, 0.6, 1.2, 1.8], dtype=np.float64),
        )
    )
    limit_engine = StuartLandauEngine(n_oscillators=4, dt=dt, method="rk4")
    zero_4 = np.zeros((4, 4), dtype=np.float64)
    limit_omegas = np.ones(4, dtype=np.float64)
    limit_mu_vec = np.full(4, limit_mu, dtype=np.float64)
    limit_steps = max(1000, n_steps)
    for _ in range(limit_steps):
        limit_state = limit_engine.step(
            limit_state,
            limit_omegas,
            limit_mu_vec,
            zero_4,
            zero_4,
            zeta=0.0,
            psi=0.0,
            alpha=zero_4,
            epsilon=0.0,
        )
    limit_radii = np.asarray(limit_state[4:], dtype=np.float64)
    limit_radius_error = float(np.max(np.abs(limit_radii - expected_limit_radius)))
    limit_phase_domain = bool(
        np.all((limit_state[:4] >= 0.0) & (limit_state[:4] < 2.0 * np.pi))
    )

    decay_state = np.concatenate(
        (
            np.linspace(0.0, np.pi / 2.0, 3, dtype=np.float64),
            np.array([0.4, 0.8, 1.2], dtype=np.float64),
        )
    )
    decay_engine = StuartLandauEngine(n_oscillators=3, dt=dt, method="rk4")
    zero_3 = np.zeros((3, 3), dtype=np.float64)
    decay_mu_vec = np.full(3, -0.25, dtype=np.float64)
    for _ in range(limit_steps):
        decay_state = decay_engine.step(
            decay_state,
            np.ones(3, dtype=np.float64),
            decay_mu_vec,
            zero_3,
            zero_3,
            zeta=0.0,
            psi=0.0,
            alpha=zero_3,
            epsilon=0.0,
        )
    decay_mean_radius = float(np.mean(decay_state[3:]))
    thresholds = {
        "max_limit_cycle_radius_error": 5.0e-3,
        "max_subcritical_mean_radius": 0.10,
        "min_coupled_mean_amplitude": 0.10,
        "require_finite_positive_amplitude": True,
        "require_limit_cycle_contract": True,
        "require_subcritical_decay_contract": True,
        "require_wrapped_phase_domain": True,
        "require_zero_self_coupling": True,
    }
    zero_self_coupling = bool(
        np.allclose(np.diag(knm), 0.0, rtol=0.0, atol=0.0)
        and np.allclose(np.diag(knm_r), 0.0, rtol=0.0, atol=0.0)
    )
    finite_positive_amplitude = bool(np.isfinite(final_r) and final_r > 0.0)
    coupled_mean_amplitude_passed = bool(
        final_r >= thresholds["min_coupled_mean_amplitude"]
    )
    limit_cycle_passed = bool(
        limit_radius_error <= thresholds["max_limit_cycle_radius_error"]
    )
    subcritical_decay_passed = bool(
        decay_mean_radius <= thresholds["max_subcritical_mean_radius"]
    )
    acceptance_passed = int(
        zero_self_coupling
        and finite_positive_amplitude
        and coupled_mean_amplitude_passed
        and limit_cycle_passed
        and subcritical_decay_passed
        and limit_phase_domain
    )
    benchmark_payload = {
        "decay_mean_radius": decay_mean_radius,
        "final_mean_amplitude": final_r,
        "limit_phase_domain": limit_phase_domain,
        "limit_radius_error": limit_radius_error,
        "thresholds": thresholds,
        "zero_self_coupling": zero_self_coupling,
    }
    benchmark_sha = sha256(
        json.dumps(benchmark_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()

    return {
        "suite": "stuart_landau_reference_pikovsky_2001",
        "n_oscillators": n_oscillators,
        "n_steps": n_steps,
        "wall_time_s": elapsed,
        "steps_per_second": n_steps / elapsed,
        "final_mean_amplitude": final_r,
        "expected_limit_cycle_radius": expected_limit_radius,
        "limit_cycle_steps": limit_steps,
        "limit_cycle_max_radius_error": limit_radius_error,
        "subcritical_mean_radius": decay_mean_radius,
        "zero_self_coupling": int(zero_self_coupling),
        "finite_positive_amplitude": int(finite_positive_amplitude),
        "coupled_mean_amplitude_passed": int(coupled_mean_amplitude_passed),
        "limit_cycle_passed": int(limit_cycle_passed),
        "subcritical_decay_passed": int(subcritical_decay_passed),
        "wrapped_phase_domain": int(limit_phase_domain),
        "acceptance_passed": acceptance_passed,
        "acceptance_thresholds_json": json.dumps(thresholds, sort_keys=True),
        "benchmark_sha256": benchmark_sha,
    }


def benchmark_petri_reachability(n_steps: int = 5000) -> dict[str, float | int | str]:
    n_steps = _validate_reference_positive_int(n_steps, name="n_steps")
    net = PetriNet(
        places=[
            Place("nominal"),
            Place("degraded"),
            Place("critical"),
            Place("recovery"),
        ],
        transitions=[
            Transition("n_to_d", inputs=[Arc("nominal")], outputs=[Arc("degraded")]),
            Transition("d_to_c", inputs=[Arc("degraded")], outputs=[Arc("critical")]),
            Transition("c_to_r", inputs=[Arc("critical")], outputs=[Arc("recovery")]),
            Transition("r_to_n", inputs=[Arc("recovery")], outputs=[Arc("nominal")]),
        ],
    )
    marking = Marking(tokens={"nominal": 1})
    visited: set[tuple[tuple[str, int], ...]] = set()
    expected_places = ("nominal", "degraded", "critical", "recovery")
    expected_transition_cycle = ("n_to_d", "d_to_c", "c_to_r", "r_to_n")
    transition_names: list[str] = []
    token_totals: list[int] = []
    observed_places: list[str] = []

    t0 = time.perf_counter()
    for _ in range(n_steps):
        key = tuple(sorted(marking.tokens.items()))
        visited.add(key)
        active_places = tuple(
            place for place, count in marking.tokens.items() if int(count) > 0
        )
        observed_places.extend(active_places)
        token_totals.append(sum(int(count) for count in marking.tokens.values()))
        marking, transition = net.step(marking, {})
        transition_names.append(transition.name if transition is not None else "")
    elapsed = time.perf_counter() - t0

    reachable_markings = len(visited)
    token_conservation = bool(
        token_totals and all(total == 1 for total in token_totals)
    )
    observed_place_set = tuple(sorted(set(observed_places)))
    exact_reachability = observed_place_set == tuple(sorted(expected_places))
    expected_prefix = tuple(
        expected_transition_cycle[index % len(expected_transition_cycle)]
        for index in range(n_steps)
    )
    observed_transition_prefix = tuple(transition_names[:n_steps])
    deterministic_cycle = observed_transition_prefix == expected_prefix
    cycle_period = len(expected_transition_cycle)
    final_expected_place = expected_places[n_steps % cycle_period]
    final_active_places = tuple(
        place for place, count in marking.tokens.items() if int(count) > 0
    )
    final_marking_correct = final_active_places == (final_expected_place,)
    thresholds = {
        "expected_reachable_markings": 4,
        "expected_token_total": 1,
        "expected_transition_cycle": list(expected_transition_cycle),
        "require_deterministic_cycle": True,
        "require_exact_reachability": True,
        "require_final_marking": True,
        "require_token_conservation": True,
    }
    acceptance_passed = int(
        reachable_markings == thresholds["expected_reachable_markings"]
        and token_conservation
        and exact_reachability
        and deterministic_cycle
        and final_marking_correct
    )
    benchmark_payload = {
        "final_active_places": final_active_places,
        "reachable_markings": reachable_markings,
        "thresholds": thresholds,
        "token_totals": token_totals,
        "transition_names": transition_names,
    }
    benchmark_sha = sha256(
        json.dumps(benchmark_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()

    return {
        "suite": "petri_net_reachability",
        "n_steps": n_steps,
        "wall_time_s": elapsed,
        "steps_per_second": n_steps / elapsed,
        "reachable_markings": reachable_markings,
        "expected_reachable_markings": thresholds["expected_reachable_markings"],
        "cycle_period": cycle_period,
        "token_conservation": int(token_conservation),
        "exact_reachability": int(exact_reachability),
        "deterministic_cycle": int(deterministic_cycle),
        "final_marking_correct": int(final_marking_correct),
        "final_active_place": final_active_places[0] if final_active_places else "",
        "expected_final_active_place": final_expected_place,
        "transition_cycle_json": json.dumps(expected_transition_cycle),
        "acceptance_passed": acceptance_passed,
        "acceptance_thresholds_json": json.dumps(thresholds, sort_keys=True),
        "benchmark_sha256": benchmark_sha,
    }


def benchmark_auto_binding_proposal_quality() -> dict[str, float | int | str]:
    fixtures = _auto_binding_quality_fixtures()
    total_channels = 0
    covered_extractors = 0
    validation_error_count = 0
    expected_edge_count = 0
    expected_edge_hits = 0
    proposed_edge_count = 0
    accepted_domain_count = 0
    min_domain_extractor_coverage = 1.0
    min_domain_expected_edge_recall = 1.0
    max_domain_validation_errors = 0
    min_sample_count = min(fixture.sample_count for fixture in fixtures)
    domain_results: list[dict[str, float | int | str | bool]] = []

    t0 = time.perf_counter()
    for fixture in fixtures:
        proposal = propose_binding_from_time_series_csv(
            fixture.csv_text,
            sample_rate_hz=fixture.sample_rate_hz,
            project_name=f"{fixture.domain}_benchmark",
        )
        fixture_validation_errors = len(proposal.binding.validation_errors)
        validation_error_count += fixture_validation_errors
        source_columns = _string_records(proposal.binding.provenance["source_columns"])
        extractor_proposals = proposal.binding.provenance[
            "extractor_parameter_proposals"
        ]
        total_channels += len(source_columns)
        fixture_covered_extractors = _extractor_source_coverage(
            source_columns=source_columns,
            extractor_proposals=_mapping_records(extractor_proposals),
        )
        covered_extractors += fixture_covered_extractors
        proposed_edges = _proposed_source_edges(
            proposal.binding.provenance["initial_coupling_proposal"]
        )
        fixture_expected_hits = len(fixture.expected_edges & proposed_edges)
        fixture_expected_edge_recall = fixture_expected_hits / len(
            fixture.expected_edges
        )
        fixture_extractor_coverage = fixture_covered_extractors / len(source_columns)
        fixture_edge_multiplier = len(proposed_edges) / len(fixture.expected_edges)
        fixture_accepted = _auto_binding_fixture_passes_thresholds(
            fixture=fixture,
            extractor_coverage=fixture_extractor_coverage,
            expected_edge_recall=fixture_expected_edge_recall,
            validation_error_count=fixture_validation_errors,
            proposed_edge_multiplier=fixture_edge_multiplier,
        )
        if fixture_accepted:
            accepted_domain_count += 1
        min_domain_extractor_coverage = min(
            min_domain_extractor_coverage, fixture_extractor_coverage
        )
        min_domain_expected_edge_recall = min(
            min_domain_expected_edge_recall, fixture_expected_edge_recall
        )
        max_domain_validation_errors = max(
            max_domain_validation_errors, fixture_validation_errors
        )
        domain_results.append(
            {
                "domain": fixture.domain,
                "sample_count": fixture.sample_count,
                "source_column_count": len(source_columns),
                "validation_error_count": fixture_validation_errors,
                "extractor_coverage": fixture_extractor_coverage,
                "expected_edge_recall": fixture_expected_edge_recall,
                "proposed_edge_count": len(proposed_edges),
                "proposed_edge_multiplier": fixture_edge_multiplier,
                "accepted": fixture_accepted,
            }
        )
        proposed_edge_count += len(proposed_edges)
        expected_edge_count += len(fixture.expected_edges)
        expected_edge_hits += fixture_expected_hits
    elapsed = time.perf_counter() - t0

    return {
        "suite": "auto_binding_synthetic_quality",
        "fixture_count": len(fixtures),
        "large_fixture_count": sum(fixture.sample_count >= 96 for fixture in fixtures),
        "wall_time_s": elapsed,
        "steps_per_second": len(fixtures) / elapsed,
        "validation_error_count": validation_error_count,
        "extractor_coverage": covered_extractors / total_channels,
        "expected_edge_recall": expected_edge_hits / expected_edge_count,
        "proposed_edge_count": proposed_edge_count,
        "domain_acceptance_passed": int(accepted_domain_count == len(fixtures)),
        "accepted_domain_count": accepted_domain_count,
        "failed_domain_count": len(fixtures) - accepted_domain_count,
        "min_domain_extractor_coverage": min_domain_extractor_coverage,
        "min_domain_expected_edge_recall": min_domain_expected_edge_recall,
        "max_domain_validation_errors": max_domain_validation_errors,
        "min_sample_count": min_sample_count,
        "domain_acceptance_thresholds_json": json.dumps(
            _auto_binding_threshold_summary(fixtures), sort_keys=True
        ),
        "domain_acceptance_results_json": json.dumps(domain_results, sort_keys=True),
    }


def benchmark_replay_policy_candidate_quality() -> dict[str, float | int | str]:
    """Benchmark replay-only learner proposals against deterministic gates."""
    thresholds = ReplayLearnerBenchmarkThresholds(
        min_acceptance_rate=1.0,
        min_reward_improvement=0.03,
        max_unsafe_acceptances=0,
        max_lyapunov_exponent=0.0,
        min_stl_robustness=0.0,
        max_safety_cost=0.08,
        require_non_actuating=True,
        require_safety_evidence=True,
    )
    scenarios = (
        ReplayLearnerBenchmarkScenario(
            name="two_channel_low_coupling",
            seed_candidate=KnobPolicyCandidate(
                K=0.2,
                alpha=0.0,
                zeta=0.05,
                Psi=0.0,
                channel_weights=(0.8, 0.2),
                cross_channel_gains=(0.05,),
            ),
            min_coherence=0.78,
            min_reward=-0.25,
            critical_coupling_estimate=0.72,
            ppo_seed=17,
            sac_seed=23,
            hybrid_seed=31,
        ),
        ReplayLearnerBenchmarkScenario(
            name="three_channel_cross_gain",
            seed_candidate=KnobPolicyCandidate(
                K=0.18,
                alpha=0.01,
                zeta=0.04,
                Psi=0.02,
                channel_weights=(0.55, 0.3, 0.15),
                cross_channel_gains=(0.03, 0.04),
            ),
            min_coherence=0.76,
            min_reward=-0.25,
            critical_coupling_estimate=0.68,
            ppo_seed=41,
            sac_seed=43,
            hybrid_seed=47,
        ),
        ReplayLearnerBenchmarkScenario(
            name="stability_recovery",
            seed_candidate=KnobPolicyCandidate(
                K=0.24,
                alpha=-0.01,
                zeta=0.06,
                Psi=0.01,
                channel_weights=(0.45, 0.35, 0.2),
                cross_channel_gains=(0.02, 0.05),
            ),
            min_coherence=0.79,
            min_reward=-0.25,
            critical_coupling_estimate=0.74,
            ppo_seed=53,
            sac_seed=59,
            hybrid_seed=61,
        ),
    )

    t0 = time.perf_counter()
    learner_results: list[dict[str, float | int | str | bool]] = []
    scenario_results: list[dict[str, float | int | str | bool]] = []
    accepted_count = 0
    unsafe_acceptances = 0
    min_reward_improvement = np.inf
    accepted_scenario_count = 0
    for scenario in scenarios:
        proposal_config = PolicyProposalConfig(
            min_coherence=scenario.min_coherence,
            min_reward=scenario.min_reward,
            max_alternatives=2,
            safety_constraints=SafetyConstraintConfig(
                max_lyapunov_exponent=thresholds.max_lyapunov_exponent,
                min_stl_robustness=thresholds.min_stl_robustness,
                max_safety_cost=thresholds.max_safety_cost,
                require_lyapunov=thresholds.require_safety_evidence,
                require_stl=thresholds.require_safety_evidence,
                require_safety_cost=thresholds.require_safety_evidence,
            ),
        )
        baseline_observation = _deterministic_replay_observation(
            scenario.seed_candidate
        )
        baseline_coherence = baseline_observation.coherence
        learner_proposals = (
            generate_ppo_like_proposal(
                scenario.seed_candidate,
                _deterministic_replay_observation,
                seed_value=scenario.ppo_seed,
                proposal_config=proposal_config,
            ),
            generate_sac_like_proposal(
                scenario.seed_candidate,
                _deterministic_replay_observation,
                seed_value=scenario.sac_seed,
                proposal_config=proposal_config,
            ),
            generate_hybrid_physics_proposal(
                scenario.seed_candidate,
                _deterministic_replay_observation,
                critical_coupling_estimate=scenario.critical_coupling_estimate,
                seed_value=scenario.hybrid_seed,
                proposal_config=proposal_config,
            ),
        )

        scenario_accepted_count = 0
        scenario_unsafe_acceptances = 0
        scenario_min_reward_improvement = np.inf
        scenario_non_actuating = True
        for proposal in learner_proposals:
            policy_proposal = proposal.policy_search.proposal
            selected = policy_proposal.selected
            accepted = policy_proposal.accepted and selected is not None
            accepted_count += int(accepted)
            scenario_accepted_count += int(accepted)
            non_actuating = proposal.actuation_permitted is False
            scenario_non_actuating = scenario_non_actuating and non_actuating
            selected_reward = selected.reward if selected is not None else -np.inf
            selected_coherence = (
                selected.observation.coherence if selected is not None else 0.0
            )
            reward_improvement = selected_coherence - baseline_coherence
            min_reward_improvement = min(min_reward_improvement, reward_improvement)
            scenario_min_reward_improvement = min(
                scenario_min_reward_improvement,
                reward_improvement,
            )
            selected_unsafe = bool(selected.observation.unsafe) if selected else False
            selected_lyapunov = (
                selected.observation.lyapunov_exponent
                if selected is not None
                and selected.observation.lyapunov_exponent is not None
                else np.inf
            )
            selected_stl = (
                selected.observation.stl_robustness
                if selected is not None
                and selected.observation.stl_robustness is not None
                else -np.inf
            )
            selected_safety_cost = (
                selected.observation.safety_cost if selected is not None else np.inf
            )
            selected_safety_evidence = (
                selected is not None
                and selected.observation.lyapunov_exponent is not None
                and selected.observation.stl_robustness is not None
                and selected_safety_cost <= thresholds.max_safety_cost
            )
            unsafe_acceptances += int(accepted and selected_unsafe)
            scenario_unsafe_acceptances += int(accepted and selected_unsafe)
            learner_results.append(
                {
                    "scenario": scenario.name,
                    "learner_kind": proposal.learner_kind,
                    "accepted": accepted,
                    "non_actuating": non_actuating,
                    "selected_reward": selected_reward,
                    "selected_coherence": selected_coherence,
                    "baseline_coherence": baseline_coherence,
                    "coherence_improvement": reward_improvement,
                    "unsafe_selected": selected_unsafe,
                    "selected_lyapunov_exponent": selected_lyapunov,
                    "selected_stl_robustness": selected_stl,
                    "selected_safety_cost": selected_safety_cost,
                    "selected_safety_evidence": selected_safety_evidence,
                    "candidate_count": len(proposal.policy_search.candidates),
                }
            )

        scenario_accepted = (
            scenario_accepted_count == len(learner_proposals)
            and scenario_unsafe_acceptances == 0
            and scenario_non_actuating
            and scenario_min_reward_improvement >= thresholds.min_reward_improvement
            and all(
                result["selected_safety_evidence"] is True
                and result["selected_lyapunov_exponent"]
                <= thresholds.max_lyapunov_exponent
                and result["selected_stl_robustness"] >= thresholds.min_stl_robustness
                and result["selected_safety_cost"] <= thresholds.max_safety_cost
                for result in learner_results
                if result["scenario"] == scenario.name
            )
        )
        accepted_scenario_count += int(scenario_accepted)
        scenario_results.append(
            {
                "scenario": scenario.name,
                "learner_count": len(learner_proposals),
                "accepted_learner_count": scenario_accepted_count,
                "failed_learner_count": len(learner_proposals)
                - scenario_accepted_count,
                "baseline_coherence": baseline_coherence,
                "min_coherence_improvement": scenario_min_reward_improvement,
                "unsafe_acceptance_count": scenario_unsafe_acceptances,
                "non_actuating_proposals": scenario_non_actuating,
                "safety_evidence_count": sum(
                    int(result["selected_safety_evidence"] is True)
                    for result in learner_results
                    if result["scenario"] == scenario.name
                ),
                "accepted": scenario_accepted,
            }
        )
    elapsed = time.perf_counter() - t0

    learner_count = len(learner_results)
    acceptance_rate = accepted_count / learner_count
    threshold_passed = (
        acceptance_rate >= thresholds.min_acceptance_rate
        and min_reward_improvement >= thresholds.min_reward_improvement
        and unsafe_acceptances <= thresholds.max_unsafe_acceptances
        and all(result["non_actuating"] is True for result in learner_results)
        and all(
            result["selected_safety_evidence"] is True for result in learner_results
        )
        and all(
            result["selected_lyapunov_exponent"] <= thresholds.max_lyapunov_exponent
            for result in learner_results
        )
        and all(
            result["selected_stl_robustness"] >= thresholds.min_stl_robustness
            for result in learner_results
        )
        and all(
            result["selected_safety_cost"] <= thresholds.max_safety_cost
            for result in learner_results
        )
        and accepted_scenario_count == len(scenarios)
    )

    return {
        "suite": "replay_policy_candidate_quality",
        "scenario_count": len(scenarios),
        "accepted_scenario_count": accepted_scenario_count,
        "failed_scenario_count": len(scenarios) - accepted_scenario_count,
        "learner_count": learner_count,
        "wall_time_s": elapsed,
        "steps_per_second": learner_count / elapsed,
        "accepted_learner_count": accepted_count,
        "failed_learner_count": learner_count - accepted_count,
        "acceptance_rate": acceptance_rate,
        "min_coherence_improvement": min_reward_improvement,
        "unsafe_acceptance_count": unsafe_acceptances,
        "safety_evidence_count": sum(
            int(result["selected_safety_evidence"] is True)
            for result in learner_results
        ),
        "non_actuating_proposals": int(
            all(result["non_actuating"] is True for result in learner_results)
        ),
        "acceptance_passed": int(threshold_passed),
        "acceptance_thresholds_json": json.dumps(
            {
                "min_acceptance_rate": thresholds.min_acceptance_rate,
                "min_reward_improvement": thresholds.min_reward_improvement,
                "max_unsafe_acceptances": thresholds.max_unsafe_acceptances,
                "max_lyapunov_exponent": thresholds.max_lyapunov_exponent,
                "min_stl_robustness": thresholds.min_stl_robustness,
                "max_safety_cost": thresholds.max_safety_cost,
                "require_non_actuating": thresholds.require_non_actuating,
                "require_safety_evidence": thresholds.require_safety_evidence,
            },
            sort_keys=True,
        ),
        "scenario_results_json": json.dumps(scenario_results, sort_keys=True),
        "learner_results_json": json.dumps(learner_results, sort_keys=True),
    }


def benchmark_self_model_digital_twin() -> dict[str, float | int | str]:
    """Benchmark replay-backed self-model monitoring for digital-twin drift."""
    thresholds = SelfModelDigitalTwinThresholds(
        min_scenario_count=3,
        max_breach_count=1,
        max_max_observed_error=3.0,
        require_non_actuating=True,
        require_operator_review=True,
        require_execution_disabled=True,
        require_deterministic_hash=True,
    )

    from scpn_phase_orchestrator.monitor.self_model import compute_self_model_error
    from scpn_phase_orchestrator.monitor.self_model_examples import (
        build_self_model_reconfiguration_examples,
    )

    t0 = time.perf_counter()
    scenarios = tuple(build_self_model_reconfiguration_examples())
    repeated_scenarios = tuple(build_self_model_reconfiguration_examples())
    if len(scenarios) != len(repeated_scenarios):
        raise RuntimeError(
            "self-model digital-twin benchmark has mismatched replay examples"
        )

    scenario_records: list[dict[str, object]] = []
    repeated_records: list[dict[str, object]] = []
    breach_count = 0
    max_observed_error = 0.0
    non_actuating = 1
    operator_review_required = 1
    execution_disabled = 1
    scenario_hash_matches = 0
    threshold_matches = 0

    for scenario, repeated_scenario in zip(scenarios, repeated_scenarios, strict=True):
        if scenario["scenario_id"] != repeated_scenario["scenario_id"]:
            raise RuntimeError(
                "self-model digital-twin benchmark requires replay scenarios "
                "to be ordered"
            )

        scenario_id = str(scenario["scenario_id"])
        domain = str(scenario["domain"])
        scenario_hash = scenario["scenario_hash"]
        repeated_scenario_hash = repeated_scenario["scenario_hash"]
        if not isinstance(scenario_hash, str) or not isinstance(
            repeated_scenario_hash, str
        ):
            raise ValueError("self-model scenario hashes must be strings")

        threshold = float(scenario["error_threshold"])
        predicted = np.asarray(scenario["predicted_phase"], dtype=np.float64)
        observed = np.asarray(scenario["observed_phase"], dtype=np.float64)

        monitor_record = compute_self_model_error(
            observed_phases=observed,
            predicted_phases=predicted,
            tolerance=threshold,
            max_abs_tolerance=threshold,
            domain=domain,
            scenario_id=scenario_id,
            channel_labels=(f"{domain}_phase",),
        ).to_audit_record()
        repeated_record = compute_self_model_error(
            observed_phases=np.asarray(
                repeated_scenario["observed_phase"], dtype=np.float64
            ),
            predicted_phases=np.asarray(
                repeated_scenario["predicted_phase"], dtype=np.float64
            ),
            tolerance=float(repeated_scenario["error_threshold"]),
            max_abs_tolerance=float(repeated_scenario["error_threshold"]),
            domain=domain,
            scenario_id=scenario_id,
            channel_labels=(f"{domain}_phase",),
        ).to_audit_record()

        breached = bool(monitor_record["breached"])
        breached_counted = int(breached)
        breach_count += breached_counted
        non_actuating &= int(monitor_record["non_actuating"] is True)
        max_observed_error = max(
            max_observed_error,
            float(monitor_record["overall_max_abs_error"]),
        )
        operator_review = int(bool(scenario["operator_review_required"]))
        operator_review_required &= operator_review
        execution_blocked = int(bool(scenario["execution_disabled"]))
        execution_disabled &= execution_blocked
        scenario_hash_match = int(scenario_hash == repeated_scenario_hash)
        scenario_hash_matches += scenario_hash_match
        threshold_match = int(
            bool(scenario["self_model_error"]["within_threshold"]) is (not breached)
        )
        threshold_matches += threshold_match
        record_hash_match = int(
            monitor_record["record_hash"] == repeated_record["record_hash"]
        )

        scenario_records.append(
            {
                "scenario_id": scenario_id,
                "domain": domain,
                "breached": breached,
                "breached_count": breached_counted,
                "scenario_hash": scenario_hash,
                "scenario_hash_match": scenario_hash_match,
                "record_hash_match": record_hash_match,
                "within_threshold_match": threshold_match,
                "non_actuating": non_actuating,
                "operator_review_required": operator_review,
                "execution_disabled": execution_blocked,
                "max_observed_error": float(monitor_record["overall_max_abs_error"]),
                "overall_rmse": float(monitor_record["overall_rmse"]),
                "overall_mae": float(monitor_record["overall_mae"]),
                "claim_boundary": str(monitor_record["claim_boundary"]),
                "record_hash": str(monitor_record["record_hash"]),
            }
        )
        repeated_records.append(
            {
                "scenario_id": scenario_id,
                "breached": bool(repeated_record["breached"]),
                "scenario_hash": str(repeated_scenario_hash),
                "record_hash": str(repeated_record["record_hash"]),
            }
        )

    elapsed = time.perf_counter() - t0
    scenario_count = len(scenario_records)
    scenario_hash_match_count = scenario_hash_matches
    record_hash_match_count = sum(
        int(record["record_hash"] == repeated["record_hash"])
        for record, repeated in zip(scenario_records, repeated_records, strict=True)
    )
    deterministic_hash = int(
        scenario_hash_match_count == scenario_count
        and record_hash_match_count == scenario_count
    )
    acceptance_passed = int(
        scenario_count >= thresholds.min_scenario_count
        and breach_count <= thresholds.max_breach_count
        and max_observed_error <= thresholds.max_max_observed_error
        and non_actuating == int(thresholds.require_non_actuating)
        and operator_review_required == int(thresholds.require_operator_review)
        and execution_disabled == int(thresholds.require_execution_disabled)
        and threshold_matches == scenario_count
        and scenario_hash_match_count == scenario_count
        and deterministic_hash == int(thresholds.require_deterministic_hash)
    )

    return {
        "suite": "self_model_digital_twin",
        "scenario_count": scenario_count,
        "breach_count": breach_count,
        "max_observed_error": max_observed_error,
        "non_actuating": non_actuating,
        "operator_review_required": operator_review_required,
        "execution_disabled": execution_disabled,
        "deterministic_hash": deterministic_hash,
        "scenario_hash_match_count": scenario_hash_match_count,
        "record_hash_match_count": record_hash_match_count,
        "self_model_sha256": _stable_record_hash(scenario_records),
        "wall_time_s": elapsed,
        "steps_per_second": scenario_count / elapsed if elapsed > 0.0 else 0.0,
        "acceptance_passed": acceptance_passed,
        "acceptance_thresholds_json": json.dumps(
            {
                "max_breach_count": thresholds.max_breach_count,
                "max_max_observed_error": thresholds.max_max_observed_error,
                "min_scenario_count": thresholds.min_scenario_count,
                "require_deterministic_hash": thresholds.require_deterministic_hash,
                "require_execution_disabled": thresholds.require_execution_disabled,
                "require_non_actuating": thresholds.require_non_actuating,
                "require_operator_review": thresholds.require_operator_review,
            },
            sort_keys=True,
        ),
        "scenario_results_json": json.dumps(scenario_records, sort_keys=True),
    }


def benchmark_strange_loop_drift_scenario_gate() -> dict[str, float | int | str]:
    """Benchmark long-run strange-loop drift scenarios and review gates."""
    thresholds = StrangeLoopDriftScenarioThresholds(
        min_scenario_count=4,
        min_long_run_step_count=128,
        min_passed_scenario_count=4,
        require_non_actuating=True,
        require_execution_disabled=True,
        require_deterministic_hash=True,
    )

    t0 = time.perf_counter()
    results = evaluate_strange_loop_drift_scenarios()
    repeated_results = evaluate_strange_loop_drift_scenarios()
    elapsed = time.perf_counter() - t0

    records = [result.to_audit_record() for result in results]
    repeated_records = [result.to_audit_record() for result in repeated_results]
    scenario_count = len(records)
    long_run_step_count = sum(int(record["step_count"]) for record in records)
    passed_scenario_count = sum(
        int(record["passed_expected_trigger"] is True) for record in records
    )
    non_actuating = int(all(record["non_actuating"] is True for record in records))
    execution_disabled = int(
        all(record["execution_disabled"] is True for record in records)
    )
    deterministic_hash = int(
        [record["result_hash"] for record in records]
        == [record["result_hash"] for record in repeated_records]
    )
    acceptance_passed = int(
        scenario_count >= thresholds.min_scenario_count
        and long_run_step_count >= thresholds.min_long_run_step_count
        and passed_scenario_count >= thresholds.min_passed_scenario_count
        and non_actuating == int(thresholds.require_non_actuating)
        and execution_disabled == int(thresholds.require_execution_disabled)
        and deterministic_hash == int(thresholds.require_deterministic_hash)
    )

    return {
        "suite": "strange_loop_drift_scenario_gate",
        "scenario_count": scenario_count,
        "long_run_step_count": long_run_step_count,
        "passed_scenario_count": passed_scenario_count,
        "non_actuating": non_actuating,
        "execution_disabled": execution_disabled,
        "deterministic_hash": deterministic_hash,
        "drift_scenario_sha256": _stable_record_hash(records),
        "wall_time_s": elapsed,
        "steps_per_second": long_run_step_count / elapsed if elapsed > 0.0 else 0.0,
        "acceptance_passed": acceptance_passed,
        "acceptance_thresholds_json": json.dumps(
            {
                "min_long_run_step_count": thresholds.min_long_run_step_count,
                "min_passed_scenario_count": thresholds.min_passed_scenario_count,
                "min_scenario_count": thresholds.min_scenario_count,
                "require_deterministic_hash": thresholds.require_deterministic_hash,
                "require_execution_disabled": thresholds.require_execution_disabled,
                "require_non_actuating": thresholds.require_non_actuating,
            },
            sort_keys=True,
        ),
        "scenario_results_json": json.dumps(records, sort_keys=True),
    }


def benchmark_bayesian_posterior_fit_quality() -> dict[str, float | int | str]:
    """Benchmark posterior fitting from observed Kuramoto trajectories."""
    thresholds = BayesianPosteriorFitThresholds(
        max_residual_rmse=2.5e-3,
        max_omega_mean_abs_error=3.0e-2,
        max_knm_mean_abs_error=6.0e-2,
        max_credible_interval_width=1.0e-2,
        min_rollout_sample_count=96,
    )
    phases, omega, knm, alpha, dt = _bayesian_posterior_fit_fixture()

    t0 = time.perf_counter()
    fit = fit_gaussian_upde_posterior(
        phases,
        dt=dt,
        alpha=alpha,
        ridge=1e-8,
        coupling_std_floor=2.5e-3,
        omega_std_floor=2.5e-3,
    )
    result = bayesian_upde_run(
        phases[0],
        omega=fit.omega,
        knm=fit.knm,
        alpha=alpha,
        zeta=0.0,
        psi=0.0,
        config=BayesianUPDEConfig(n_samples=128, seed=41, n_steps=32, dt=dt),
    )
    elapsed = time.perf_counter() - t0

    omega_mean_abs_error = float(np.max(np.abs(fit.omega.mean - omega)))
    knm_mean_abs_error = float(np.max(np.abs(fit.knm.mean - knm)))
    credible_interval_width = float(result.r_upper - result.r_lower)
    audit_record = fit.to_audit_record()
    finite_audit = int(_audit_record_is_finite(audit_record))
    zero_diagonal = int(np.allclose(np.diag(fit.knm.mean), 0.0))
    non_negative_coupling = int(np.all(fit.knm.mean >= 0.0))
    acceptance_passed = int(
        fit.residual_rmse <= thresholds.max_residual_rmse
        and omega_mean_abs_error <= thresholds.max_omega_mean_abs_error
        and knm_mean_abs_error <= thresholds.max_knm_mean_abs_error
        and credible_interval_width <= thresholds.max_credible_interval_width
        and result.sample_count >= thresholds.min_rollout_sample_count
        and finite_audit == 1
        and zero_diagonal == 1
        and non_negative_coupling == 1
    )

    return {
        "suite": "bayesian_posterior_fit_quality",
        "sample_count": len(phases),
        "rollout_sample_count": result.sample_count,
        "wall_time_s": elapsed,
        "steps_per_second": len(phases) / elapsed,
        "residual_rmse": fit.residual_rmse,
        "omega_mean_abs_error": omega_mean_abs_error,
        "knm_mean_abs_error": knm_mean_abs_error,
        "credible_interval_width": credible_interval_width,
        "finite_audit_record": finite_audit,
        "zero_diagonal_coupling": zero_diagonal,
        "non_negative_coupling": non_negative_coupling,
        "acceptance_passed": acceptance_passed,
        "acceptance_thresholds_json": json.dumps(
            {
                "max_credible_interval_width": (thresholds.max_credible_interval_width),
                "max_knm_mean_abs_error": thresholds.max_knm_mean_abs_error,
                "max_omega_mean_abs_error": thresholds.max_omega_mean_abs_error,
                "max_residual_rmse": thresholds.max_residual_rmse,
                "min_rollout_sample_count": thresholds.min_rollout_sample_count,
            },
            sort_keys=True,
        ),
    }


def benchmark_bayesian_backend_fail_closed() -> dict[str, float | int | str]:
    """Benchmark reserved Bayesian sampler names against fail-closed gates."""
    thresholds = BayesianBackendFailClosedThresholds(
        min_available_backends=1,
        required_fail_closed_backends=frozenset({"numpyro", "blackjax"}),
        max_unexpected_reserved_successes=0,
    )
    phases, omega, knm, alpha, _ = _bayesian_backend_audit_fixture()

    t0 = time.perf_counter()
    statuses = audit_bayesian_backend_status(
        phases,
        omega=omega,
        knm=knm,
        alpha=alpha,
        zeta=0.0,
        psi=0.0,
        config=BayesianUPDEConfig(n_samples=16, seed=83, n_steps=3),
    )
    elapsed = time.perf_counter() - t0

    status_by_backend = {status.backend: status for status in statuses}
    available_backends = sum(status.available for status in statuses)
    fail_closed_backends = {
        status.backend
        for status in statuses
        if status.fail_closed and not status.available
    }
    unexpected_reserved_successes = sum(
        status.available
        for status in statuses
        if status.backend in thresholds.required_fail_closed_backends
    )
    acceptance_passed = int(
        available_backends >= thresholds.min_available_backends
        and thresholds.required_fail_closed_backends <= fail_closed_backends
        and unexpected_reserved_successes
        <= thresholds.max_unexpected_reserved_successes
        and status_by_backend["numpy"].sample_count == 16
    )

    return {
        "suite": "bayesian_backend_fail_closed",
        "backend_count": len(statuses),
        "wall_time_s": elapsed,
        "steps_per_second": len(statuses) / elapsed,
        "available_backend_count": available_backends,
        "fail_closed_backend_count": len(fail_closed_backends),
        "unexpected_reserved_success_count": unexpected_reserved_successes,
        "numpy_sample_count": status_by_backend["numpy"].sample_count,
        "acceptance_passed": acceptance_passed,
        "acceptance_thresholds_json": json.dumps(
            {
                "max_unexpected_reserved_successes": (
                    thresholds.max_unexpected_reserved_successes
                ),
                "min_available_backends": thresholds.min_available_backends,
                "required_fail_closed_backends": sorted(
                    thresholds.required_fail_closed_backends
                ),
            },
            sort_keys=True,
        ),
        "backend_results_json": json.dumps(
            [status.to_audit_record() for status in statuses],
            sort_keys=True,
        ),
    }


def benchmark_formal_export_artifact_quality() -> dict[str, float | int | str]:
    """Benchmark formal exporter artefact generation and fail-closed guards."""
    thresholds = FormalExportThresholds(
        min_artifact_count=5,
        min_fail_closed_count=4,
        min_identifier_map_count=12,
        min_package_property_count=3,
        min_checker_command_count=3,
        min_checker_availability_count=3,
        min_missing_checker_count=1,
        min_runtime_certificate_count=1,
        require_deterministic_hash=True,
        require_checker_execution_disabled=True,
        require_runtime_certificate_verified=True,
    )
    net, marking, rules, stl_specs = _formal_export_fixture()

    t0 = time.perf_counter()
    petri_prism = export_petri_net_prism(
        net,
        marking,
        module_name="formal benchmark",
    )
    petri_tla = export_petri_net_tla(
        net,
        marking,
        module_name="FormalBenchmark",
    )
    policy_prism = export_policy_rules_prism(
        rules,
        module_name="policy benchmark",
    )
    policy_tla = export_policy_rules_tla(
        rules,
        module_name="PolicyBenchmark",
    )
    stl_prism = export_stl_specs_prism(
        stl_specs,
        module_name="stl benchmark",
    )
    package = build_formal_verification_package(
        {
            "petri_prism": petri_prism,
            "petri_tla": petri_tla,
            "policy_prism": policy_prism,
        },
        (
            FormalSafetyProperty(
                name="petri_reaches_done",
                artifact_name="petri_prism",
                checker="prism",
                expression='P>=1 [ F "active_done" ]',
                description="Petri net can reach terminal done place.",
            ),
            FormalSafetyProperty(
                name="petri_type_ok",
                artifact_name="petri_tla",
                checker="tlc",
                expression="Safety",
                description="Petri TLA state variables remain bounded.",
            ),
            FormalSafetyProperty(
                name="policy_boost_fires",
                artifact_name="policy_prism",
                checker="prism",
                expression='P>=1 [ F "fires_boost_K" ]',
                description="Policy rule firing remains externally checkable.",
            ),
        ),
        package_name="spo-formal-reference",
    )
    repeated_package = build_formal_verification_package(
        {
            "petri_prism": petri_prism,
            "petri_tla": petri_tla,
            "policy_prism": policy_prism,
        },
        package.properties,
        package_name="spo-formal-reference",
    )
    repeated = export_policy_rules_prism(
        rules,
        module_name="policy benchmark",
    )
    fail_closed_count = _formal_export_fail_closed_count()
    elapsed = time.perf_counter() - t0

    artifact_texts = (
        petri_prism.model,
        petri_tla.module,
        policy_prism.model,
        policy_tla.module,
        stl_prism.model,
    )
    artifact_hash = sha256("\n---\n".join(artifact_texts).encode()).hexdigest()
    repeated_hash = sha256(repeated.model.encode()).hexdigest()
    deterministic_hash = int(
        repeated_hash == sha256(policy_prism.model.encode()).hexdigest()
        and package.package_hash == repeated_package.package_hash
    )
    package_record = package.to_audit_record()
    checker_commands = package_record["checker_commands"]
    if not isinstance(checker_commands, list):
        raise TypeError("formal checker commands must be a list")
    checker_execution_disabled = int(
        all(command.get("execution_permitted") is False for command in checker_commands)
    )
    checker_availability = audit_formal_checker_availability(
        package,
        executable_paths={
            "prism": "/opt/prism/bin/prism",
            "tlc2.TLC": None,
        },
    )
    checker_availability_records = [
        item.to_audit_record() for item in checker_availability
    ]
    certificate_availability = audit_formal_checker_availability(
        package,
        executable_paths={
            "prism": "/opt/prism/bin/prism",
            "tlc2.TLC": "/opt/tlc/tlc2.TLC",
        },
    )
    checker_results = tuple(
        FormalCheckerResult(
            property_name=command.property_name,
            checker=command.checker,
            artifact_name=command.artifact_name,
            package_hash=package.package_hash,
            result_hash=sha256(
                (
                    f"{package.package_hash}:{command.property_name}:reviewed-pass"
                ).encode()
            ).hexdigest(),
            status="passed",
            passed=True,
            detail="reference-suite reviewed checker evidence",
        )
        for command in package.checker_commands
    )
    runtime_certificate = build_runtime_control_certificate(
        package,
        certificate_availability,
        checker_results,
        {
            "R_min": 0.7,
            "max_step_s": 0.05,
            "max_policy_actions": 4.0,
        },
    )
    runtime_certificate_record = runtime_certificate.to_audit_record()
    runtime_certificate_verified = int(
        runtime_certificate.status == "verified_non_actuating"
        and runtime_certificate.required_property_count
        == runtime_certificate.passed_required_count
    )
    runtime_certificate_execution_disabled = int(
        runtime_certificate.actuation_permitted is False
    )
    runtime_certificate_count = 1
    checker_availability_count = len(checker_availability_records)
    checker_available_count = sum(
        int(record["available"]) for record in checker_availability_records
    )
    checker_missing_count = checker_availability_count - checker_available_count
    checker_availability_execution_disabled = int(
        all(
            record.get("execution_permitted") is False
            for record in checker_availability_records
        )
    )
    identifier_map_count = sum(
        len(mapping)
        for mapping in (
            petri_prism.place_names,
            petri_prism.transition_names,
            petri_prism.metric_names,
            policy_prism.rule_names,
            policy_prism.action_names,
            policy_prism.metric_names,
            stl_prism.stl_names,
            stl_prism.metric_names,
        )
    )
    acceptance_passed = int(
        len(artifact_texts) >= thresholds.min_artifact_count
        and fail_closed_count >= thresholds.min_fail_closed_count
        and identifier_map_count >= thresholds.min_identifier_map_count
        and len(package.properties) >= thresholds.min_package_property_count
        and len(package.checker_commands) >= thresholds.min_checker_command_count
        and checker_availability_count >= thresholds.min_checker_availability_count
        and checker_missing_count >= thresholds.min_missing_checker_count
        and deterministic_hash == int(thresholds.require_deterministic_hash)
        and checker_execution_disabled
        == int(thresholds.require_checker_execution_disabled)
        and checker_availability_execution_disabled
        == int(thresholds.require_checker_execution_disabled)
        and runtime_certificate_count >= thresholds.min_runtime_certificate_count
        and runtime_certificate_verified
        == int(thresholds.require_runtime_certificate_verified)
        and runtime_certificate_execution_disabled
        == int(thresholds.require_checker_execution_disabled)
        and "Safety == TypeOK" in petri_tla.module
        and 'label "fires_boost_K"' in policy_prism.model
        and 'label "stl_keep_sync_satisfied"' in stl_prism.model
    )

    return {
        "suite": "formal_export_artifact_quality",
        "artifact_count": len(artifact_texts),
        "wall_time_s": elapsed,
        "steps_per_second": len(artifact_texts) / elapsed,
        "identifier_map_count": identifier_map_count,
        "fail_closed_count": fail_closed_count,
        "package_property_count": len(package.properties),
        "checker_command_count": len(package.checker_commands),
        "checker_availability_count": checker_availability_count,
        "checker_available_count": checker_available_count,
        "checker_missing_count": checker_missing_count,
        "checker_execution_disabled": checker_execution_disabled,
        "checker_availability_execution_disabled": (
            checker_availability_execution_disabled
        ),
        "runtime_certificate_count": runtime_certificate_count,
        "runtime_certificate_verified": runtime_certificate_verified,
        "runtime_certificate_execution_disabled": (
            runtime_certificate_execution_disabled
        ),
        "deterministic_hash": deterministic_hash,
        "artifact_sha256": artifact_hash,
        "package_sha256": package.package_hash,
        "runtime_certificate_sha256": runtime_certificate.certificate_hash,
        "petri_prism_bytes": len(petri_prism.model.encode()),
        "petri_tla_bytes": len(petri_tla.module.encode()),
        "policy_prism_bytes": len(policy_prism.model.encode()),
        "policy_tla_bytes": len(policy_tla.module.encode()),
        "stl_prism_bytes": len(stl_prism.model.encode()),
        "acceptance_passed": acceptance_passed,
        "acceptance_thresholds_json": json.dumps(
            {
                "min_artifact_count": thresholds.min_artifact_count,
                "min_checker_availability_count": (
                    thresholds.min_checker_availability_count
                ),
                "min_checker_command_count": thresholds.min_checker_command_count,
                "min_fail_closed_count": thresholds.min_fail_closed_count,
                "min_identifier_map_count": thresholds.min_identifier_map_count,
                "min_missing_checker_count": thresholds.min_missing_checker_count,
                "min_package_property_count": thresholds.min_package_property_count,
                "min_runtime_certificate_count": (
                    thresholds.min_runtime_certificate_count
                ),
                "require_checker_execution_disabled": (
                    thresholds.require_checker_execution_disabled
                ),
                "require_deterministic_hash": (thresholds.require_deterministic_hash),
                "require_runtime_certificate_verified": (
                    thresholds.require_runtime_certificate_verified
                ),
            },
            sort_keys=True,
        ),
        "checker_commands_json": json.dumps(checker_commands, sort_keys=True),
        "checker_availability_json": json.dumps(
            checker_availability_records,
            sort_keys=True,
        ),
        "runtime_certificate_json": json.dumps(
            runtime_certificate_record,
            sort_keys=True,
        ),
    }


def benchmark_stl_closed_loop_plan_quality() -> dict[str, float | int | str]:
    """Benchmark offline STL closed-loop plan synthesis and fail-closed gates."""
    thresholds = STLClosedLoopThresholds(
        min_plan_count=3,
        min_projected_action_count=1,
        min_runtime_gate_checked_count=3,
        min_runtime_mapped_command_count=1,
        min_blocked_reason_count=3,
        require_non_actuating=True,
        require_runtime_execution_disabled=True,
        require_deterministic_hash=True,
    )
    projection_template = STLActionProjectionTemplate(
        action="raise_coupling",
        knob="K",
        scope="global",
        base_value=0.9,
        step=10.0,
        ttl_s=0.5,
        previous_value=0.9,
        value_bounds=(0.0, 1.0),
        rate_limit=0.05,
    )
    t0 = time.perf_counter()
    projected = synthesise_stl_closed_loop_plan(
        synthesise_stl_monitoring_automaton(
            "eventually (R >= 0.8)",
            {"R": [0.1, 0.2, 0.75]},
        ),
        {"R": [0.1, 0.2, 0.75]},
        (projection_template,),
        horizon_steps=4,
        action_map={"R": "raise_coupling"},
    )
    blocked = synthesise_stl_closed_loop_plan(
        synthesise_stl_monitoring_automaton(
            "eventually (R >= 0.8)",
            {"R": [0.1, 0.2, 0.75]},
        ),
        {"R": [0.1, 0.2, 0.75]},
        (),
        horizon_steps=1,
    )
    satisfied = synthesise_stl_closed_loop_plan(
        synthesise_stl_monitoring_automaton(
            "always (R >= 0.3)",
            {"R": [0.8, 0.9]},
        ),
        {"R": [0.8, 0.9]},
        (),
        horizon_steps=2,
    )
    plans = (projected, blocked, satisfied)
    elapsed = time.perf_counter() - t0

    records = [plan.to_audit_record() for plan in plans]
    repeated_records = [
        plan.to_audit_record()
        for plan in (
            synthesise_stl_closed_loop_plan(
                synthesise_stl_monitoring_automaton(
                    "eventually (R >= 0.8)",
                    {"R": [0.1, 0.2, 0.75]},
                ),
                {"R": [0.1, 0.2, 0.75]},
                (projection_template,),
                horizon_steps=4,
                action_map={"R": "raise_coupling"},
            ),
            synthesise_stl_closed_loop_plan(
                synthesise_stl_monitoring_automaton(
                    "eventually (R >= 0.8)",
                    {"R": [0.1, 0.2, 0.75]},
                ),
                {"R": [0.1, 0.2, 0.75]},
                (),
                horizon_steps=1,
            ),
            synthesise_stl_closed_loop_plan(
                synthesise_stl_monitoring_automaton(
                    "always (R >= 0.3)",
                    {"R": [0.8, 0.9]},
                ),
                {"R": [0.8, 0.9]},
                (),
                horizon_steps=2,
            ),
        )
    ]
    plans_json = json.dumps(records, sort_keys=True)
    deterministic_hash = int(
        sha256(plans_json.encode()).hexdigest()
        == sha256(json.dumps(repeated_records, sort_keys=True).encode()).hexdigest()
    )
    projected_action_count = sum(
        len(plan.projected_plan.approved_actions) for plan in plans
    )
    rejected_candidate_count = sum(
        len(plan.projected_plan.rejected_candidates) for plan in plans
    )
    blocked_reason_count = sum(len(plan.blocked_reasons) for plan in plans)
    runtime_gate_checked_count = sum(
        int(plan.runtime_gate.non_actuating and plan.runtime_gate.execution_disabled)
        for plan in plans
    )
    runtime_mapped_command_count = sum(
        plan.runtime_gate.mapped_command_count for plan in plans
    )
    runtime_execution_disabled = int(
        all(plan.runtime_gate.execution_disabled for plan in plans)
    )
    non_actuating = int(
        all(
            not plan.actuating
            and not plan.synthesis.actuating
            and not plan.projected_plan.actuating
            and plan.runtime_gate.non_actuating
            for plan in plans
        )
    )
    acceptance_passed = int(
        len(plans) >= thresholds.min_plan_count
        and projected_action_count >= thresholds.min_projected_action_count
        and runtime_gate_checked_count >= thresholds.min_runtime_gate_checked_count
        and runtime_mapped_command_count >= thresholds.min_runtime_mapped_command_count
        and blocked_reason_count >= thresholds.min_blocked_reason_count
        and non_actuating == int(thresholds.require_non_actuating)
        and runtime_execution_disabled
        == int(thresholds.require_runtime_execution_disabled)
        and deterministic_hash == int(thresholds.require_deterministic_hash)
        and projected.next_review_start_index == 3
        and projected.next_review_end_index == 6
        and rejected_candidate_count == 1
    )

    return {
        "suite": "stl_closed_loop_plan_quality",
        "plan_count": len(plans),
        "wall_time_s": elapsed,
        "steps_per_second": len(plans) / elapsed,
        "projected_action_count": projected_action_count,
        "rejected_candidate_count": rejected_candidate_count,
        "blocked_reason_count": blocked_reason_count,
        "runtime_gate_checked_count": runtime_gate_checked_count,
        "runtime_mapped_command_count": runtime_mapped_command_count,
        "runtime_execution_disabled": runtime_execution_disabled,
        "non_actuating": non_actuating,
        "deterministic_hash": deterministic_hash,
        "plan_sha256": sha256(plans_json.encode()).hexdigest(),
        "acceptance_passed": acceptance_passed,
        "acceptance_thresholds_json": json.dumps(
            {
                "min_blocked_reason_count": thresholds.min_blocked_reason_count,
                "min_plan_count": thresholds.min_plan_count,
                "min_projected_action_count": (thresholds.min_projected_action_count),
                "min_runtime_gate_checked_count": (
                    thresholds.min_runtime_gate_checked_count
                ),
                "min_runtime_mapped_command_count": (
                    thresholds.min_runtime_mapped_command_count
                ),
                "require_deterministic_hash": thresholds.require_deterministic_hash,
                "require_non_actuating": thresholds.require_non_actuating,
                "require_runtime_execution_disabled": (
                    thresholds.require_runtime_execution_disabled
                ),
            },
            sort_keys=True,
        ),
        "plans_json": plans_json,
    }


def benchmark_domain_formal_safety_exports() -> dict[str, float | int | str]:
    """Benchmark domain-style policy/STL formal safety artefacts."""
    thresholds = DomainFormalExportThresholds(
        min_domain_count=11,
        min_artifacts_per_domain=5,
        min_rules_per_domain=2,
        min_stl_specs_per_domain=2,
        min_package_property_count=2,
        min_checker_command_count=2,
        require_deterministic_hash=True,
    )
    fixtures = _domain_formal_export_fixtures()
    packages = {
        bundle.domainpack: bundle for bundle in build_domainpack_formal_packages()
    }
    repeated_packages = {
        bundle.domainpack: bundle for bundle in build_domainpack_formal_packages()
    }
    domain_results: list[dict[str, float | int | str | bool]] = []
    artifact_texts: list[str] = []

    t0 = time.perf_counter()
    for fixture in fixtures:
        rules = list(fixture.rules)
        stl_specs = list(fixture.stl_specs)
        policy_prism = export_policy_rules_prism(
            rules,
            module_name=f"{fixture.domain} policy",
        )
        policy_tla = export_policy_rules_tla(
            rules,
            module_name=f"{fixture.domain} policy",
        )
        stl_prism = export_stl_specs_prism(
            stl_specs,
            module_name=f"{fixture.domain} stl",
        )
        repeated = export_policy_rules_prism(
            rules,
            module_name=f"{fixture.domain} policy",
        )
        bundle = packages[fixture.domain]
        repeated_bundle = repeated_packages[fixture.domain]
        texts = (
            policy_prism.model,
            policy_tla.module,
            stl_prism.model,
            *bundle.artifacts.values(),
        )
        package_record = bundle.package.to_audit_record()
        checker_commands = package_record["checker_commands"]
        if not isinstance(checker_commands, list):
            raise TypeError("domain formal checker commands must be a list")
        deterministic_hash = int(
            sha256(policy_prism.model.encode()).hexdigest()
            == sha256(repeated.model.encode()).hexdigest()
            and bundle.package.package_hash == repeated_bundle.package.package_hash
        )
        required_labels_present = all(
            label in "\n".join(texts) for label in fixture.required_labels
        )
        checker_execution_disabled = int(
            all(
                command.get("execution_permitted") is False
                for command in checker_commands
            )
        )
        identifier_map_count = sum(
            len(mapping)
            for mapping in (
                policy_prism.rule_names,
                policy_prism.action_names,
                policy_prism.metric_names,
                stl_prism.stl_names,
                stl_prism.metric_names,
            )
        )
        artifact_count = len(texts)
        accepted = bool(
            artifact_count >= thresholds.min_artifacts_per_domain
            and len(rules) >= thresholds.min_rules_per_domain
            and len(stl_specs) >= thresholds.min_stl_specs_per_domain
            and len(bundle.package.properties) >= thresholds.min_package_property_count
            and len(bundle.package.checker_commands)
            >= thresholds.min_checker_command_count
            and deterministic_hash == int(thresholds.require_deterministic_hash)
            and required_labels_present
            and checker_execution_disabled == 1
        )
        domain_results.append(
            {
                "domain": fixture.domain,
                "artifact_count": artifact_count,
                "rule_count": len(rules),
                "stl_spec_count": len(stl_specs),
                "package_property_count": len(bundle.package.properties),
                "checker_command_count": len(bundle.package.checker_commands),
                "checker_execution_disabled": checker_execution_disabled,
                "identifier_map_count": identifier_map_count,
                "deterministic_hash": deterministic_hash,
                "required_labels_present": required_labels_present,
                "accepted": accepted,
            }
        )
        artifact_texts.extend(texts)
    elapsed = time.perf_counter() - t0

    accepted_domain_count = sum(result["accepted"] is True for result in domain_results)
    artifact_hash = sha256("\n---\n".join(artifact_texts).encode()).hexdigest()
    acceptance_passed = int(
        len(fixtures) >= thresholds.min_domain_count
        and accepted_domain_count == len(fixtures)
    )

    return {
        "suite": "domain_formal_safety_exports",
        "domain_count": len(fixtures),
        "artifact_count": len(artifact_texts),
        "wall_time_s": elapsed,
        "steps_per_second": len(artifact_texts) / elapsed,
        "accepted_domain_count": accepted_domain_count,
        "failed_domain_count": len(fixtures) - accepted_domain_count,
        "artifact_sha256": artifact_hash,
        "acceptance_passed": acceptance_passed,
        "acceptance_thresholds_json": json.dumps(
            {
                "min_artifacts_per_domain": thresholds.min_artifacts_per_domain,
                "min_domain_count": thresholds.min_domain_count,
                "min_checker_command_count": thresholds.min_checker_command_count,
                "min_rules_per_domain": thresholds.min_rules_per_domain,
                "min_stl_specs_per_domain": thresholds.min_stl_specs_per_domain,
                "min_package_property_count": (thresholds.min_package_property_count),
                "require_deterministic_hash": (thresholds.require_deterministic_hash),
            },
            sort_keys=True,
        ),
        "domain_results_json": json.dumps(domain_results, sort_keys=True),
    }


def benchmark_hybrid_cocompiler_review_gate() -> dict[str, float | int | str]:
    """Benchmark hybrid quantum/neuromorphic review manifest gates."""
    thresholds = HybridCocompilerThresholds(
        min_target_backend_count=4,
        min_quantum_term_count=3,
        min_neuromorphic_sample_count=2,
        min_blocked_probe_count=2,
        require_non_actuating=True,
    )
    quantum_manifest = _hybrid_quantum_manifest()
    neuromorphic_manifest = _hybrid_neuromorphic_manifest()

    t0 = time.perf_counter()
    manifest = build_hybrid_cocompiler_manifest(
        quantum_manifest,
        neuromorphic_manifest,
        n_channel_semantics=("Q_control", "S_spike", "audit"),
    )
    repeated = build_hybrid_cocompiler_manifest(
        quantum_manifest,
        neuromorphic_manifest,
        n_channel_semantics=("Q_control", "S_spike", "audit"),
    )
    blocked_probe_count = _hybrid_cocompiler_blocked_probe_count(
        quantum_manifest,
        neuromorphic_manifest,
    )
    elapsed = time.perf_counter() - t0

    parity = manifest["co_simulation_parity"]
    target_backends = manifest["target_backends"]
    component_hashes = manifest["component_hashes"]
    non_actuating = (
        manifest["qpu_execution_permitted"] is False
        and manifest["hardware_write_permitted"] is False
        and manifest["actuation_permitted"] is False
    )
    component_hash_count = len(component_hashes)
    deterministic_hash = int(
        manifest["hybrid_manifest_sha256"] == repeated["hybrid_manifest_sha256"]
    )
    acceptance_passed = int(
        manifest["status"] == "co_simulation_parity_passed"
        and len(target_backends) >= thresholds.min_target_backend_count
        and parity["quantum_term_count"] >= thresholds.min_quantum_term_count
        and parity["neuromorphic_sample_count"]
        >= thresholds.min_neuromorphic_sample_count
        and blocked_probe_count >= thresholds.min_blocked_probe_count
        and non_actuating == thresholds.require_non_actuating
        and component_hash_count == 3
        and deterministic_hash == 1
    )

    return {
        "suite": "hybrid_cocompiler_review_gate",
        "manifest_count": 1,
        "wall_time_s": elapsed,
        "steps_per_second": 1.0 / elapsed,
        "target_backend_count": len(target_backends),
        "component_hash_count": component_hash_count,
        "quantum_term_count": parity["quantum_term_count"],
        "neuromorphic_sample_count": parity["neuromorphic_sample_count"],
        "blocked_probe_count": blocked_probe_count,
        "non_actuating": int(non_actuating),
        "deterministic_hash": deterministic_hash,
        "hybrid_manifest_sha256": manifest["hybrid_manifest_sha256"],
        "acceptance_passed": acceptance_passed,
        "acceptance_thresholds_json": json.dumps(
            {
                "min_blocked_probe_count": thresholds.min_blocked_probe_count,
                "min_neuromorphic_sample_count": (
                    thresholds.min_neuromorphic_sample_count
                ),
                "min_quantum_term_count": thresholds.min_quantum_term_count,
                "min_target_backend_count": thresholds.min_target_backend_count,
                "require_non_actuating": thresholds.require_non_actuating,
            },
            sort_keys=True,
        ),
        "target_backends_json": json.dumps(target_backends, sort_keys=True),
    }


def benchmark_quantum_target_readiness_gate() -> dict[str, float | int | str]:
    """Benchmark non-executing QPU target-readiness audit gates."""
    thresholds = QuantumTargetReadinessThresholds(
        min_ready_count=1,
        min_blocked_count=1,
        min_blocked_reason_count=2,
        min_operator_command_count=6,
        require_non_executing=True,
        require_deterministic_hash=True,
    )
    bridge = QuantumControlBridge(n_oscillators=3, trotter_order=2)
    knm = np.array(
        [
            [0.0, 0.1, 0.0],
            [0.1, 0.0, 0.05],
            [0.0, 0.05, 0.0],
        ],
        dtype=np.float64,
    )
    omegas = np.array([0.4, 0.5, 0.6], dtype=np.float64)

    t0 = time.perf_counter()
    manifest = bridge.build_quantum_compiler_manifest(knm, omegas, dt=0.2)
    blocked = bridge.audit_qpu_target_readiness(
        manifest,
        target_backend="qiskit_openqasm3",
        provider="ibm_quantum",
    )
    ready = bridge.audit_qpu_target_readiness(
        manifest,
        target_backend="pennylane_qasm",
        provider="pennylane",
        credentials_configured=True,
        operator_approved=True,
    )
    repeated_ready = bridge.audit_qpu_target_readiness(
        manifest,
        target_backend="pennylane_qasm",
        provider="pennylane",
        credentials_configured=True,
        operator_approved=True,
    )
    elapsed = time.perf_counter() - t0

    records = [blocked, ready]
    ready_count = sum(record["status"] == "ready_not_executed" for record in records)
    blocked_count = sum(record["status"] == "blocked" for record in records)
    blocked_reason_count = sum(
        len(record["blocked_reasons"])
        for record in records
        if isinstance(record["blocked_reasons"], list)
    )
    operator_command_count = sum(
        len(record["operator_commands"])
        for record in records
        if isinstance(record["operator_commands"], list)
    )
    non_executing = all(
        record["qpu_execution_permitted"] is False
        and record["actuation_permitted"] is False
        for record in records
    )
    deterministic_hash = int(
        ready["readiness_sha256"] == repeated_ready["readiness_sha256"]
    )
    acceptance_passed = int(
        ready_count >= thresholds.min_ready_count
        and blocked_count >= thresholds.min_blocked_count
        and blocked_reason_count >= thresholds.min_blocked_reason_count
        and operator_command_count >= thresholds.min_operator_command_count
        and non_executing == thresholds.require_non_executing
        and deterministic_hash == int(thresholds.require_deterministic_hash)
    )

    return {
        "suite": "quantum_target_readiness_gate",
        "record_count": len(records),
        "wall_time_s": elapsed,
        "steps_per_second": len(records) / elapsed,
        "ready_count": ready_count,
        "blocked_count": blocked_count,
        "blocked_reason_count": blocked_reason_count,
        "operator_command_count": operator_command_count,
        "non_executing": int(non_executing),
        "deterministic_hash": deterministic_hash,
        "manifest_sha256": str(manifest["manifest_sha256"]),
        "ready_readiness_sha256": str(ready["readiness_sha256"]),
        "acceptance_passed": acceptance_passed,
        "acceptance_thresholds_json": json.dumps(
            {
                "min_blocked_count": thresholds.min_blocked_count,
                "min_blocked_reason_count": thresholds.min_blocked_reason_count,
                "min_operator_command_count": thresholds.min_operator_command_count,
                "min_ready_count": thresholds.min_ready_count,
                "require_deterministic_hash": thresholds.require_deterministic_hash,
                "require_non_executing": thresholds.require_non_executing,
            },
            sort_keys=True,
        ),
        "target_backends_json": json.dumps(
            [record["target_backend"] for record in records],
            sort_keys=True,
        ),
        "readiness_records_json": json.dumps(records, sort_keys=True),
    }


def benchmark_neuromorphic_target_readiness_gate() -> dict[str, float | int | str]:
    """Benchmark non-executing neuromorphic target-readiness audit gates."""
    thresholds = NeuromorphicTargetReadinessThresholds(
        min_ready_count=1,
        min_blocked_count=1,
        min_blocked_reason_count=3,
        min_operator_command_count=6,
        require_non_executing=True,
        require_deterministic_hash=True,
    )
    bridge = SNNControllerBridge(n_neurons=32)
    state = UPDEState(
        layers=[
            LayerState(R=0.25, psi=0.1),
            LayerState(R=0.75, psi=0.3),
        ],
        cross_layer_alignment=np.array(
            [
                [1.0, 0.4],
                [0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        stability_proxy=0.5,
        regime_id="nominal",
    )

    t0 = time.perf_counter()
    manifest = bridge.build_neuromorphic_schedule_manifest(
        state,
        i_scale=2.0,
        threshold_hz=20.0,
    )
    blocked = bridge.audit_hardware_target_readiness(
        manifest,
        target_backend="lava",
        hardware_site="lab_lava_cluster",
    )
    ready = bridge.audit_hardware_target_readiness(
        manifest,
        target_backend="pynn",
        hardware_site="brainscales_review_lane",
        credentials_configured=True,
        operator_approved=True,
        external_simulator_parity_verified=True,
    )
    repeated_ready = bridge.audit_hardware_target_readiness(
        manifest,
        target_backend="pynn",
        hardware_site="brainscales_review_lane",
        credentials_configured=True,
        operator_approved=True,
        external_simulator_parity_verified=True,
    )
    elapsed = time.perf_counter() - t0

    records = [blocked, ready]
    ready_count = sum(record["status"] == "ready_not_executed" for record in records)
    blocked_count = sum(record["status"] == "blocked" for record in records)
    blocked_reason_count = sum(
        len(record["blocked_reasons"])
        for record in records
        if isinstance(record["blocked_reasons"], list)
    )
    operator_command_count = sum(
        len(record["operator_commands"])
        for record in records
        if isinstance(record["operator_commands"], list)
    )
    non_executing = all(
        record["hardware_write_permitted"] is False
        and record["actuation_permitted"] is False
        for record in records
    )
    deterministic_hash = int(
        ready["readiness_sha256"] == repeated_ready["readiness_sha256"]
    )
    acceptance_passed = int(
        ready_count >= thresholds.min_ready_count
        and blocked_count >= thresholds.min_blocked_count
        and blocked_reason_count >= thresholds.min_blocked_reason_count
        and operator_command_count >= thresholds.min_operator_command_count
        and non_executing == thresholds.require_non_executing
        and deterministic_hash == int(thresholds.require_deterministic_hash)
    )

    return {
        "suite": "neuromorphic_target_readiness_gate",
        "record_count": len(records),
        "wall_time_s": elapsed,
        "steps_per_second": len(records) / elapsed,
        "ready_count": ready_count,
        "blocked_count": blocked_count,
        "blocked_reason_count": blocked_reason_count,
        "operator_command_count": operator_command_count,
        "non_executing": int(non_executing),
        "deterministic_hash": deterministic_hash,
        "manifest_sha256": str(manifest["schedule_sha256"]),
        "ready_readiness_sha256": str(ready["readiness_sha256"]),
        "acceptance_passed": acceptance_passed,
        "acceptance_thresholds_json": json.dumps(
            {
                "min_blocked_count": thresholds.min_blocked_count,
                "min_blocked_reason_count": thresholds.min_blocked_reason_count,
                "min_operator_command_count": thresholds.min_operator_command_count,
                "min_ready_count": thresholds.min_ready_count,
                "require_deterministic_hash": thresholds.require_deterministic_hash,
                "require_non_executing": thresholds.require_non_executing,
            },
            sort_keys=True,
        ),
        "target_backends_json": json.dumps(
            [record["target_backend"] for record in records],
            sort_keys=True,
        ),
        "readiness_records_json": json.dumps(records, sort_keys=True),
    }


def benchmark_hybrid_target_readiness_gate() -> dict[str, float | int | str]:
    """Benchmark non-executing hybrid target-readiness audit gates."""
    thresholds = HybridTargetReadinessThresholds(
        min_ready_count=1,
        min_blocked_count=1,
        min_blocked_reason_count=1,
        min_operator_command_count=6,
        require_non_executing=True,
        require_deterministic_hash=True,
        require_component_hash_linked=True,
    )
    quantum_manifest = _hybrid_quantum_manifest()
    neuromorphic_manifest = _hybrid_neuromorphic_manifest()

    t0 = time.perf_counter()
    hybrid_manifest = build_hybrid_cocompiler_manifest(
        quantum_manifest,
        neuromorphic_manifest,
        n_channel_semantics=("Q_control", "S_spike", "audit"),
    )
    quantum_readiness = _hybrid_quantum_readiness_record(
        manifest_sha256=str(quantum_manifest["manifest_sha256"]),
    )
    neuromorphic_readiness = _hybrid_neuromorphic_readiness_record(
        manifest_sha256=str(neuromorphic_manifest["schedule_sha256"]),
    )
    blocked = audit_hybrid_target_readiness(
        hybrid_manifest,
        quantum_readiness,
        neuromorphic_readiness,
        hybrid_operator_approved=False,
    )
    ready = audit_hybrid_target_readiness(
        hybrid_manifest,
        quantum_readiness,
        neuromorphic_readiness,
        hybrid_operator_approved=True,
    )
    repeated_ready = audit_hybrid_target_readiness(
        hybrid_manifest,
        quantum_readiness,
        neuromorphic_readiness,
        hybrid_operator_approved=True,
    )
    elapsed = time.perf_counter() - t0

    records = [blocked, ready]
    ready_count = sum(record["status"] == "ready_not_executed" for record in records)
    blocked_count = sum(record["status"] == "blocked" for record in records)
    blocked_reason_count = sum(
        len(record["blocked_reasons"])
        for record in records
        if isinstance(record["blocked_reasons"], list)
    )
    operator_command_count = sum(
        len(record["operator_commands"])
        for record in records
        if isinstance(record["operator_commands"], list)
    )
    non_executing = all(
        record["qpu_execution_permitted"] is False
        and record["hardware_write_permitted"] is False
        and record["actuation_permitted"] is False
        for record in records
    )
    deterministic_hash = int(
        ready["readiness_sha256"] == repeated_ready["readiness_sha256"]
    )
    component_hashes = hybrid_manifest["component_hashes"]
    component_hash_linked = int(
        isinstance(component_hashes, dict)
        and quantum_readiness["manifest_sha256"]
        == component_hashes["quantum_manifest_sha256"]
        and neuromorphic_readiness["manifest_sha256"]
        == component_hashes["neuromorphic_schedule_sha256"]
        and ready["quantum_readiness_sha256"] == quantum_readiness["readiness_sha256"]
        and ready["neuromorphic_readiness_sha256"]
        == neuromorphic_readiness["readiness_sha256"]
    )
    acceptance_passed = int(
        ready_count >= thresholds.min_ready_count
        and blocked_count >= thresholds.min_blocked_count
        and blocked_reason_count >= thresholds.min_blocked_reason_count
        and operator_command_count >= thresholds.min_operator_command_count
        and non_executing == thresholds.require_non_executing
        and deterministic_hash == int(thresholds.require_deterministic_hash)
        and component_hash_linked == int(thresholds.require_component_hash_linked)
    )

    return {
        "suite": "hybrid_target_readiness_gate",
        "record_count": len(records),
        "wall_time_s": elapsed,
        "steps_per_second": len(records) / elapsed,
        "ready_count": ready_count,
        "blocked_count": blocked_count,
        "blocked_reason_count": blocked_reason_count,
        "operator_command_count": operator_command_count,
        "non_executing": int(non_executing),
        "deterministic_hash": deterministic_hash,
        "component_hash_linked": component_hash_linked,
        "hybrid_manifest_sha256": str(hybrid_manifest["hybrid_manifest_sha256"]),
        "ready_readiness_sha256": str(ready["readiness_sha256"]),
        "acceptance_passed": acceptance_passed,
        "acceptance_thresholds_json": json.dumps(
            {
                "min_blocked_count": thresholds.min_blocked_count,
                "min_blocked_reason_count": thresholds.min_blocked_reason_count,
                "min_operator_command_count": thresholds.min_operator_command_count,
                "min_ready_count": thresholds.min_ready_count,
                "require_component_hash_linked": (
                    thresholds.require_component_hash_linked
                ),
                "require_deterministic_hash": thresholds.require_deterministic_hash,
                "require_non_executing": thresholds.require_non_executing,
            },
            sort_keys=True,
        ),
        "readiness_records_json": json.dumps(records, sort_keys=True),
    }


def benchmark_hybrid_operator_handoff_package_gate() -> dict[str, float | int | str]:
    """Benchmark non-executing hybrid operator handoff package gates."""
    thresholds = HybridOperatorHandoffThresholds(
        min_ready_package_count=1,
        min_blocked_package_count=1,
        min_blocked_reason_count=1,
        min_operator_command_count=8,
        require_non_executing=True,
        require_deterministic_hash=True,
        require_hash_chain_linked=True,
    )
    quantum_manifest = _hybrid_quantum_manifest()
    neuromorphic_manifest = _hybrid_neuromorphic_manifest()

    t0 = time.perf_counter()
    hybrid_manifest = build_hybrid_cocompiler_manifest(
        quantum_manifest,
        neuromorphic_manifest,
        n_channel_semantics=("Q_control", "S_spike", "audit"),
    )
    quantum_readiness = _hybrid_quantum_readiness_record(
        manifest_sha256=str(quantum_manifest["manifest_sha256"]),
    )
    neuromorphic_readiness = _hybrid_neuromorphic_readiness_record(
        manifest_sha256=str(neuromorphic_manifest["schedule_sha256"]),
    )
    blocked_readiness = audit_hybrid_target_readiness(
        hybrid_manifest,
        quantum_readiness,
        neuromorphic_readiness,
        hybrid_operator_approved=False,
    )
    ready_readiness = audit_hybrid_target_readiness(
        hybrid_manifest,
        quantum_readiness,
        neuromorphic_readiness,
        hybrid_operator_approved=True,
    )
    blocked_package = build_hybrid_operator_handoff_package(
        hybrid_manifest,
        blocked_readiness,
    )
    ready_package = build_hybrid_operator_handoff_package(
        hybrid_manifest,
        ready_readiness,
    )
    repeated_ready_package = build_hybrid_operator_handoff_package(
        hybrid_manifest,
        ready_readiness,
    )
    elapsed = time.perf_counter() - t0

    packages = [blocked_package, ready_package]
    ready_package_count = sum(
        package["status"] == "ready_not_executed" for package in packages
    )
    blocked_package_count = sum(package["status"] == "blocked" for package in packages)
    blocked_reason_count = sum(
        len(package["blocked_reasons"])
        for package in packages
        if isinstance(package["blocked_reasons"], list)
    )
    operator_command_count = sum(
        len(package["operator_commands"])
        for package in packages
        if isinstance(package["operator_commands"], list)
    )
    non_executing = all(
        package["execution_permitted"] is False
        and package["qpu_execution_permitted"] is False
        and package["hardware_write_permitted"] is False
        and package["actuation_permitted"] is False
        for package in packages
    )
    deterministic_hash = int(
        ready_package["package_sha256"] == repeated_ready_package["package_sha256"]
    )
    hash_chain_linked = int(
        ready_package["hybrid_manifest_sha256"]
        == hybrid_manifest["hybrid_manifest_sha256"]
        and ready_package["hybrid_readiness_sha256"]
        == ready_readiness["readiness_sha256"]
    )
    acceptance_passed = int(
        ready_package_count >= thresholds.min_ready_package_count
        and blocked_package_count >= thresholds.min_blocked_package_count
        and blocked_reason_count >= thresholds.min_blocked_reason_count
        and operator_command_count >= thresholds.min_operator_command_count
        and non_executing == thresholds.require_non_executing
        and deterministic_hash == int(thresholds.require_deterministic_hash)
        and hash_chain_linked == int(thresholds.require_hash_chain_linked)
    )

    return {
        "suite": "hybrid_operator_handoff_package_gate",
        "package_count": len(packages),
        "wall_time_s": elapsed,
        "steps_per_second": len(packages) / elapsed,
        "ready_package_count": ready_package_count,
        "blocked_package_count": blocked_package_count,
        "blocked_reason_count": blocked_reason_count,
        "operator_command_count": operator_command_count,
        "non_executing": int(non_executing),
        "deterministic_hash": deterministic_hash,
        "hash_chain_linked": hash_chain_linked,
        "ready_package_sha256": str(ready_package["package_sha256"]),
        "acceptance_passed": acceptance_passed,
        "acceptance_thresholds_json": json.dumps(
            {
                "min_blocked_package_count": thresholds.min_blocked_package_count,
                "min_blocked_reason_count": thresholds.min_blocked_reason_count,
                "min_operator_command_count": thresholds.min_operator_command_count,
                "min_ready_package_count": thresholds.min_ready_package_count,
                "require_deterministic_hash": thresholds.require_deterministic_hash,
                "require_hash_chain_linked": thresholds.require_hash_chain_linked,
                "require_non_executing": thresholds.require_non_executing,
            },
            sort_keys=True,
        ),
        "packages_json": json.dumps(packages, sort_keys=True),
    }


def benchmark_value_alignment_replay_calibration_gate() -> dict[str, float | int | str]:
    """Benchmark deterministic replay calibration for value-alignment guards."""
    thresholds = ValueAlignmentReplayCalibrationThresholds(
        min_replay_case_count=3,
        min_approved_case_count=1,
        min_blocked_case_count=1,
        min_threshold_fallback_case_count=1,
        min_fallback_applied_case_count=2,
        require_review_only=True,
        require_deterministic_hash=True,
    )
    fallback = ControlAction(
        knob="zeta",
        scope="global",
        value=0.0,
        ttl_s=1.0,
        justification="alignment fallback: hold review path",
    )
    policy = ValueAlignmentPolicy(
        constraints=(
            ValueConstraint(
                "bounded-production-review",
                knob="K",
                scope="global",
                max_abs_value=1.0,
            ),
        ),
        fallback_actions=(fallback,),
        minimum_score=0.96,
    )
    replay_cases = {
        "approved_nominal_replay": [
            ControlAction(
                knob="K",
                scope="global",
                value=0.01,
                ttl_s=5.0,
                justification="nominal replay candidate",
            )
        ],
        "blocked_hard_limit_replay": [
            ControlAction(
                knob="K",
                scope="global",
                value=1.2,
                ttl_s=5.0,
                justification="unsafe replay candidate",
            )
        ],
        "fallback_low_margin_replay": [
            ControlAction(
                knob="K",
                scope="global",
                value=0.05,
                ttl_s=5.0,
                justification="low-margin replay candidate",
            )
        ],
    }

    t0 = time.perf_counter()
    calibration = calibrate_value_alignment_replay_evidence(policy, replay_cases)
    repeated = calibrate_value_alignment_replay_evidence(policy, replay_cases)
    elapsed = time.perf_counter() - t0

    review_only = int(calibration["calibration_actuation_permitted"] is False)
    deterministic_hash = int(
        calibration["calibration_sha256"] == repeated["calibration_sha256"]
    )
    replay_case_count = int(calibration["replay_case_count"])
    approved_case_count = int(calibration["approved_case_count"])
    blocked_case_count = int(calibration["blocked_case_count"])
    threshold_fallback_case_count = int(calibration["threshold_fallback_case_count"])
    fallback_applied_case_count = int(calibration["fallback_applied_case_count"])
    acceptance_passed = int(
        replay_case_count >= thresholds.min_replay_case_count
        and approved_case_count >= thresholds.min_approved_case_count
        and blocked_case_count >= thresholds.min_blocked_case_count
        and threshold_fallback_case_count
        >= thresholds.min_threshold_fallback_case_count
        and fallback_applied_case_count >= thresholds.min_fallback_applied_case_count
        and review_only == int(thresholds.require_review_only)
        and deterministic_hash == int(thresholds.require_deterministic_hash)
    )

    return {
        "suite": "value_alignment_replay_calibration_gate",
        "record_count": 1,
        "wall_time_s": elapsed,
        "steps_per_second": replay_case_count / elapsed,
        "replay_case_count": replay_case_count,
        "approved_case_count": approved_case_count,
        "blocked_case_count": blocked_case_count,
        "threshold_fallback_case_count": threshold_fallback_case_count,
        "fallback_applied_case_count": fallback_applied_case_count,
        "review_only": review_only,
        "deterministic_hash": deterministic_hash,
        "calibration_sha256": str(calibration["calibration_sha256"]),
        "acceptance_passed": acceptance_passed,
        "acceptance_thresholds_json": json.dumps(
            {
                "min_approved_case_count": thresholds.min_approved_case_count,
                "min_blocked_case_count": thresholds.min_blocked_case_count,
                "min_fallback_applied_case_count": (
                    thresholds.min_fallback_applied_case_count
                ),
                "min_replay_case_count": thresholds.min_replay_case_count,
                "min_threshold_fallback_case_count": (
                    thresholds.min_threshold_fallback_case_count
                ),
                "require_deterministic_hash": thresholds.require_deterministic_hash,
                "require_review_only": thresholds.require_review_only,
            },
            sort_keys=True,
        ),
        "calibration_records_json": json.dumps(
            calibration["decision_records"],
            sort_keys=True,
        ),
    }


def benchmark_autopoietic_lineage_sandbox_gate() -> dict[str, float | int | str]:
    """Benchmark offline autopoietic child-policy lineage sandbox gates."""
    thresholds = AutopoieticLineageSandboxThresholds(
        min_child_candidate_count=5,
        min_accepted_child_count=3,
        min_rejected_child_count=2,
        min_policy_diff_count=5,
        min_replay_domain_count=4,
        require_review_only=True,
        require_deterministic_hash=True,
    )
    parent_policy = {"K": 0.42, "alpha": 0.18, "zeta": 0.09}
    safe_replays = list(build_autopoietic_lineage_replay_corpus())
    unsafe_replays = [
        {
            "replay_id": "unsafe_grid_replay",
            "domain": "power_grid",
            "scenario": "unsafe_frequency_recovery",
            "reward": 0.3,
            "safety_margin": 0.02,
            "violations": ["stl_margin_breach"],
        }
    ]

    t0 = time.perf_counter()
    safe_manifest = build_autopoietic_lineage_sandbox(
        parent_policy,
        safe_replays,
        child_budget=3,
        mutation_step=0.02,
        minimum_replay_reward=0.7,
        minimum_safety_margin=0.1,
    )
    unsafe_manifest = build_autopoietic_lineage_sandbox(
        parent_policy,
        unsafe_replays,
        child_budget=2,
        mutation_step=0.04,
        minimum_replay_reward=0.7,
        minimum_safety_margin=0.1,
    )
    repeated_safe_manifest = build_autopoietic_lineage_sandbox(
        parent_policy,
        safe_replays,
        child_budget=3,
        mutation_step=0.02,
        minimum_replay_reward=0.7,
        minimum_safety_margin=0.1,
    )
    elapsed = time.perf_counter() - t0

    manifests = [safe_manifest, unsafe_manifest]
    child_candidate_count = sum(
        int(manifest["child_candidate_count"]) for manifest in manifests
    )
    accepted_child_count = sum(
        int(manifest["accepted_child_count"]) for manifest in manifests
    )
    rejected_child_count = sum(
        int(manifest["rejected_child_count"]) for manifest in manifests
    )
    policy_diff_count = sum(
        len(candidate["policy_diff"])
        for manifest in manifests
        for candidate in manifest["child_candidates"]
    )
    replay_domain_count = int(safe_manifest["replay_domain_count"])
    review_only = int(
        all(
            manifest["review_required"] is True
            and manifest["execution_disabled"] is True
            and manifest["live_merge_permitted"] is False
            and manifest["hot_patch_permitted"] is False
            and manifest["actuation_permitted"] is False
            for manifest in manifests
        )
    )
    deterministic_hash = int(
        safe_manifest["lineage_sha256"] == repeated_safe_manifest["lineage_sha256"]
    )
    acceptance_passed = int(
        child_candidate_count >= thresholds.min_child_candidate_count
        and accepted_child_count >= thresholds.min_accepted_child_count
        and rejected_child_count >= thresholds.min_rejected_child_count
        and policy_diff_count >= thresholds.min_policy_diff_count
        and replay_domain_count >= thresholds.min_replay_domain_count
        and review_only == int(thresholds.require_review_only)
        and deterministic_hash == int(thresholds.require_deterministic_hash)
    )

    return {
        "suite": "autopoietic_lineage_sandbox_gate",
        "manifest_count": len(manifests),
        "wall_time_s": elapsed,
        "steps_per_second": child_candidate_count / elapsed,
        "child_candidate_count": child_candidate_count,
        "accepted_child_count": accepted_child_count,
        "rejected_child_count": rejected_child_count,
        "policy_diff_count": policy_diff_count,
        "replay_domain_count": replay_domain_count,
        "review_only": review_only,
        "deterministic_hash": deterministic_hash,
        "safe_lineage_sha256": str(safe_manifest["lineage_sha256"]),
        "acceptance_passed": acceptance_passed,
        "acceptance_thresholds_json": json.dumps(
            {
                "min_accepted_child_count": thresholds.min_accepted_child_count,
                "min_child_candidate_count": thresholds.min_child_candidate_count,
                "min_policy_diff_count": thresholds.min_policy_diff_count,
                "min_replay_domain_count": thresholds.min_replay_domain_count,
                "min_rejected_child_count": thresholds.min_rejected_child_count,
                "require_deterministic_hash": thresholds.require_deterministic_hash,
                "require_review_only": thresholds.require_review_only,
            },
            sort_keys=True,
        ),
        "lineage_manifests_json": json.dumps(manifests, sort_keys=True),
    }


def benchmark_intergenerational_policy_inheritance_gate() -> dict[
    str, float | int | str
]:
    """Benchmark signed review-only intergenerational policy inheritance."""
    thresholds = IntergenerationalInheritanceThresholds(
        min_manifest_count=2,
        min_signed_metadata_count=2,
        min_policy_gene_count=3,
        min_history_record_count=2,
        min_replay_domain_count=4,
        min_fitness_score=0.35,
        require_review_only=True,
        require_deterministic_hash=True,
    )
    parent_policy = {"K": 0.42, "alpha": 0.18, "zeta": 0.09}
    replays = build_autopoietic_lineage_replay_corpus()

    t0 = time.perf_counter()
    lineage = build_autopoietic_lineage_sandbox(
        parent_policy,
        replays,
        child_budget=2,
        mutation_step=0.02,
        minimum_replay_reward=0.7,
        minimum_safety_margin=0.1,
    )
    accepted_children = [
        child
        for child in lineage["child_candidates"]
        if child["status"] == "accepted_for_review"
    ]
    inheritance_manifests = [
        build_intergenerational_policy_inheritance(
            lineage,
            child,
            signer_id="reference-suite-review-key",
            signing_key="reference-suite-local-signing-key",
            objective_weights={"reward": 0.6, "safety": 0.3, "simplicity": 0.1},
        )
        for child in accepted_children
    ]
    repeated = build_intergenerational_policy_inheritance(
        lineage,
        accepted_children[0],
        signer_id="reference-suite-review-key",
        signing_key="reference-suite-local-signing-key",
        objective_weights={"reward": 0.6, "safety": 0.3, "simplicity": 0.1},
    )
    history = build_intergenerational_policy_inheritance_history(
        lineage,
        inheritance_manifests,
    )
    repeated_history = build_intergenerational_policy_inheritance_history(
        lineage,
        inheritance_manifests,
    )
    elapsed = time.perf_counter() - t0

    manifest_count = len(inheritance_manifests)
    history_record_count = int(history["history_record_count"])
    replay_domain_count = int(history["replay_domain_count"])
    signed_metadata_count = sum(
        1 for manifest in inheritance_manifests if manifest["signed_metadata"]
    )
    policy_gene_count = min(
        len(manifest["inherited_policy_genome"]) for manifest in inheritance_manifests
    )
    min_fitness_score = min(
        float(manifest["multi_objective_replay_fitness"]["fitness_score"])
        for manifest in inheritance_manifests
    )
    review_only = int(
        all(
            manifest["hot_patch_review_required"] is True
            and manifest["direct_hot_patch_permitted"] is False
            and manifest["actuation_permitted"] is False
            and manifest["merge_strategy"] == "reviewed_hot_patch_only"
            for manifest in inheritance_manifests
        )
        and history["hot_patch_review_required"] is True
        and history["direct_hot_patch_permitted"] is False
        and history["actuation_permitted"] is False
        and history["merge_strategy"] == "reviewed_hot_patch_only"
    )
    deterministic_hash = int(
        inheritance_manifests[0]["inheritance_sha256"] == repeated["inheritance_sha256"]
        and history["history_sha256"] == repeated_history["history_sha256"]
    )
    acceptance_passed = int(
        manifest_count >= thresholds.min_manifest_count
        and signed_metadata_count >= thresholds.min_signed_metadata_count
        and policy_gene_count >= thresholds.min_policy_gene_count
        and history_record_count >= thresholds.min_history_record_count
        and replay_domain_count >= thresholds.min_replay_domain_count
        and min_fitness_score >= thresholds.min_fitness_score
        and review_only == int(thresholds.require_review_only)
        and deterministic_hash == int(thresholds.require_deterministic_hash)
    )

    return {
        "suite": "intergenerational_policy_inheritance_gate",
        "manifest_count": manifest_count,
        "wall_time_s": elapsed,
        "steps_per_second": manifest_count / elapsed,
        "signed_metadata_count": signed_metadata_count,
        "policy_gene_count": policy_gene_count,
        "history_record_count": history_record_count,
        "replay_domain_count": replay_domain_count,
        "min_fitness_score": min_fitness_score,
        "review_only": review_only,
        "deterministic_hash": deterministic_hash,
        "inheritance_sha256": str(inheritance_manifests[0]["inheritance_sha256"]),
        "history_sha256": str(history["history_sha256"]),
        "acceptance_passed": acceptance_passed,
        "acceptance_thresholds_json": json.dumps(
            {
                "min_fitness_score": thresholds.min_fitness_score,
                "min_history_record_count": thresholds.min_history_record_count,
                "min_manifest_count": thresholds.min_manifest_count,
                "min_policy_gene_count": thresholds.min_policy_gene_count,
                "min_replay_domain_count": thresholds.min_replay_domain_count,
                "min_signed_metadata_count": thresholds.min_signed_metadata_count,
                "require_deterministic_hash": thresholds.require_deterministic_hash,
                "require_review_only": thresholds.require_review_only,
            },
            sort_keys=True,
        ),
        "inheritance_manifests_json": json.dumps(
            inheritance_manifests,
            sort_keys=True,
        ),
        "inheritance_history_json": json.dumps(history, sort_keys=True),
    }


def benchmark_temporal_causal_hypergraph_experiment_gate() -> dict[
    str, float | int | str
]:
    """Benchmark research-only temporal-causal hypergraph baseline gates."""
    thresholds = TemporalCausalHypergraphThresholds(
        min_manifest_count=2,
        min_accepted_hyperedge_count=1,
        min_baseline_edge_count=1,
        min_baseline_family_count=5,
        require_research_only=True,
        require_deterministic_hash=True,
    )
    trace = {
        "driver": [0.0, 1.0, 2.0, 3.0, 4.0],
        "response": [0.0, 0.0, 2.0, 6.0, 12.0],
        "distractor": [1.0, 1.0, 1.0, 1.0, 1.0],
    }
    passing_candidates = [
        {
            "sources": ["driver", "response"],
            "target": "response",
            "time_offsets": [-1, 0],
            "score": 2.6,
        }
    ]
    blocked_candidates = [
        {
            "sources": ["distractor", "driver"],
            "target": "response",
            "time_offsets": [-1, 1],
            "score": 0.1,
        }
    ]

    t0 = time.perf_counter()
    passing = build_temporal_causal_hypergraph_experiment(
        trace,
        passing_candidates,
        lag=1,
        min_abs_weight=0.1,
        required_baseline_margin=0.1,
    )
    blocked = build_temporal_causal_hypergraph_experiment(
        trace,
        blocked_candidates,
        lag=1,
        min_abs_weight=0.1,
        required_baseline_margin=0.1,
    )
    repeated = build_temporal_causal_hypergraph_experiment(
        trace,
        passing_candidates,
        lag=1,
        min_abs_weight=0.1,
        required_baseline_margin=0.1,
    )
    elapsed = time.perf_counter() - t0

    manifests = [passing, blocked]
    accepted_hyperedge_count = sum(
        int(manifest["accepted_hyperedge_count"]) for manifest in manifests
    )
    min_baseline_edge_count = min(
        int(manifest["baseline"]["edge_count"]) for manifest in manifests
    )
    min_baseline_family_count = min(
        len(manifest["baseline"]["baseline_family"]) for manifest in manifests
    )
    research_only = int(
        all(
            manifest["research_only"] is True
            and manifest["production_claim_permitted"] is False
            and manifest["actuation_permitted"] is False
            for manifest in manifests
        )
    )
    deterministic_hash = int(
        passing["experiment_sha256"] == repeated["experiment_sha256"]
    )
    acceptance_passed = int(
        len(manifests) >= thresholds.min_manifest_count
        and accepted_hyperedge_count >= thresholds.min_accepted_hyperedge_count
        and min_baseline_edge_count >= thresholds.min_baseline_edge_count
        and min_baseline_family_count >= thresholds.min_baseline_family_count
        and research_only == int(thresholds.require_research_only)
        and deterministic_hash == int(thresholds.require_deterministic_hash)
    )

    return {
        "suite": "temporal_causal_hypergraph_experiment_gate",
        "manifest_count": len(manifests),
        "wall_time_s": elapsed,
        "steps_per_second": len(manifests) / elapsed,
        "accepted_hyperedge_count": accepted_hyperedge_count,
        "min_baseline_edge_count": min_baseline_edge_count,
        "min_baseline_family_count": min_baseline_family_count,
        "research_only": research_only,
        "deterministic_hash": deterministic_hash,
        "passing_experiment_sha256": str(passing["experiment_sha256"]),
        "acceptance_passed": acceptance_passed,
        "acceptance_thresholds_json": json.dumps(
            {
                "min_accepted_hyperedge_count": (
                    thresholds.min_accepted_hyperedge_count
                ),
                "min_baseline_edge_count": thresholds.min_baseline_edge_count,
                "min_baseline_family_count": thresholds.min_baseline_family_count,
                "min_manifest_count": thresholds.min_manifest_count,
                "require_deterministic_hash": thresholds.require_deterministic_hash,
                "require_research_only": thresholds.require_research_only,
            },
            sort_keys=True,
        ),
        "experiment_manifests_json": json.dumps(manifests, sort_keys=True),
    }


def benchmark_morphogenetic_domain_demo_gate() -> dict[str, float | int | str]:
    """Benchmark additional morphogenetic domainpack demo audit surfaces."""
    thresholds = MorphogeneticDomainDemoThresholds(
        min_demo_count=3,
        min_total_grown_edges=6,
        min_total_shrunk_edges=6,
        require_non_actuating=True,
        require_snapshot_rows=True,
        require_deterministic_hash=True,
    )

    t0 = time.perf_counter()
    demo_runners = _morphogenetic_demo_runners()
    demos = [runner() for runner in demo_runners]
    repeated = [runner() for runner in demo_runners]
    elapsed = time.perf_counter() - t0

    records = [_morphogenetic_demo_record(demo) for demo in demos]
    repeated_records = [_morphogenetic_demo_record(demo) for demo in repeated]
    demo_count = len(records)
    total_grown_edges = sum(int(record["grown_edge_count"]) for record in records)
    total_shrunk_edges = sum(int(record["shrunk_edge_count"]) for record in records)
    non_actuating = int(all(record["actuating"] is False for record in records))
    snapshot_rows = int(
        all(
            int(record["snapshot_heatmap_rows"]) == int(record["field_layers"])
            for record in records
        )
    )
    deterministic_hash = int(records == repeated_records)
    acceptance_passed = int(
        demo_count >= thresholds.min_demo_count
        and total_grown_edges >= thresholds.min_total_grown_edges
        and total_shrunk_edges >= thresholds.min_total_shrunk_edges
        and non_actuating == int(thresholds.require_non_actuating)
        and snapshot_rows == int(thresholds.require_snapshot_rows)
        and deterministic_hash == int(thresholds.require_deterministic_hash)
    )

    return {
        "suite": "morphogenetic_domain_demo_gate",
        "record_count": demo_count,
        "wall_time_s": elapsed,
        "steps_per_second": demo_count / elapsed,
        "total_grown_edges": total_grown_edges,
        "total_shrunk_edges": total_shrunk_edges,
        "non_actuating": non_actuating,
        "snapshot_rows": snapshot_rows,
        "deterministic_hash": deterministic_hash,
        "demo_sha256": _stable_record_hash(records),
        "acceptance_passed": acceptance_passed,
        "acceptance_thresholds_json": json.dumps(
            {
                "min_demo_count": thresholds.min_demo_count,
                "min_total_grown_edges": thresholds.min_total_grown_edges,
                "min_total_shrunk_edges": thresholds.min_total_shrunk_edges,
                "require_deterministic_hash": thresholds.require_deterministic_hash,
                "require_non_actuating": thresholds.require_non_actuating,
                "require_snapshot_rows": thresholds.require_snapshot_rows,
            },
            sort_keys=True,
        ),
        "demo_records_json": json.dumps(records, sort_keys=True),
    }


def benchmark_integrated_information_replay_corpus_gate() -> dict[
    str, float | int | str
]:
    """Benchmark empirical replay corpus coverage for the Phi proxy monitor."""
    thresholds = IntegratedInformationReplayCorpusThresholds(
        min_domain_count=3,
        min_record_count=12,
        min_ordering_evidence_count=6,
        require_non_actuating=True,
        require_claim_boundary=True,
        require_deterministic_hash=True,
    )

    t0 = time.perf_counter()
    corpus_builders = _integrated_information_replay_corpus_builders()
    corpus = [builder() for builder in corpus_builders]
    repeated = [builder() for builder in corpus_builders]
    elapsed = time.perf_counter() - t0

    records = [
        _integrated_information_replay_record(record)
        for domain_records in corpus
        for record in domain_records
    ]
    repeated_records = [
        _integrated_information_replay_record(record)
        for domain_records in repeated
        for record in domain_records
    ]
    domains = sorted({str(record["domain"]) for record in records})
    ordering_evidence_count = sum(
        int(">" in str(record["expected_relationship"])) for record in records
    )
    non_actuating = int(all(record["non_actuating"] is True for record in records))
    claim_boundary = int(
        all(
            record["claim_boundary"] == "engineering_proxy_not_theoretical_iit"
            for record in records
        )
    )
    deterministic_hash = int(records == repeated_records)
    acceptance_passed = int(
        len(domains) >= thresholds.min_domain_count
        and len(records) >= thresholds.min_record_count
        and ordering_evidence_count >= thresholds.min_ordering_evidence_count
        and non_actuating == int(thresholds.require_non_actuating)
        and claim_boundary == int(thresholds.require_claim_boundary)
        and deterministic_hash == int(thresholds.require_deterministic_hash)
    )

    return {
        "suite": "integrated_information_replay_corpus_gate",
        "record_count": len(records),
        "wall_time_s": elapsed,
        "steps_per_second": len(records) / elapsed,
        "domain_count": len(domains),
        "ordering_evidence_count": ordering_evidence_count,
        "non_actuating": non_actuating,
        "claim_boundary": claim_boundary,
        "deterministic_hash": deterministic_hash,
        "corpus_sha256": _stable_record_hash(records),
        "acceptance_passed": acceptance_passed,
        "domains_json": json.dumps(domains, sort_keys=True),
        "acceptance_thresholds_json": json.dumps(
            {
                "min_domain_count": thresholds.min_domain_count,
                "min_ordering_evidence_count": (thresholds.min_ordering_evidence_count),
                "min_record_count": thresholds.min_record_count,
                "require_claim_boundary": thresholds.require_claim_boundary,
                "require_deterministic_hash": thresholds.require_deterministic_hash,
                "require_non_actuating": thresholds.require_non_actuating,
            },
            sort_keys=True,
        ),
        "replay_records_json": json.dumps(records, sort_keys=True),
    }


def _evolutionary_example_payload(payload: object) -> Mapping[str, object]:
    if isinstance(payload, Mapping):
        return payload
    if hasattr(payload, "to_audit_record"):
        audit_record = payload.to_audit_record()
        if isinstance(audit_record, Mapping):
            return audit_record
    if hasattr(payload, "__dict__"):
        raw = payload.__dict__
        if isinstance(raw, Mapping):
            return raw
    raise TypeError("evolutionary supervisor example payload must be mapping-like")


def _evolutionary_candidate_record(payload: Mapping[str, object]) -> dict[str, object]:
    blocked_reasons = payload.get("blocked_reasons", ())
    if isinstance(blocked_reasons, tuple):
        blocked_reason_values = list(blocked_reasons)
    elif isinstance(blocked_reasons, list):
        blocked_reason_values = blocked_reasons
    elif blocked_reasons:
        raise TypeError("candidate blocked_reasons must be a sequence")
    else:
        blocked_reason_values = []

    return {
        "candidate_id": str(payload["candidate_id"]),
        "generation": int(payload["generation"]),
        "knob": str(payload["knob"]),
        "parent_value": float(payload["parent_value"]),
        "candidate_value": float(payload["candidate_value"]),
        "mutation_delta": float(payload["mutation_delta"]),
        "replay_fitness": float(payload["replay_fitness"]),
        "stl_robustness": float(payload["stl_robustness"]),
        "stl_satisfied": bool(payload["stl_satisfied"]),
        "replay_violation_count": int(payload["replay_violation_count"]),
        "blocked_reasons": [str(reason) for reason in blocked_reason_values],
        "review_required": bool(payload["review_required"]),
        "live_merge_permitted": bool(payload["live_merge_permitted"]),
        "hot_patch_permitted": bool(payload["hot_patch_permitted"]),
        "actuation_permitted": bool(payload["actuation_permitted"]),
        "candidate_hash": str(payload["candidate_hash"]),
        "status": str(payload["status"]),
    }


def _evolutionary_example_scalar(
    *, payload: Mapping[str, object], key: str, default: float | int
) -> float:
    value = payload.get(key, default)
    if not isinstance(value, int | float | np.floating):
        raise ValueError(f"evolutionary example key '{key}' must be numeric")
    return float(value)


def benchmark_evolutionary_supervisor_search() -> dict[str, float | int | str]:
    """Benchmark offline evolutionary supervisor search with review and safety gates."""
    thresholds = EvolutionarySupervisorSearchThresholds(
        min_scenario_count=1,
        min_candidate_count=8,
        min_accepted_count=1,
        min_rejected_count=1,
        min_stl_filter_rejected_count=0,
        min_counterfactual_filter_rejected_count=1,
        require_non_actuating=True,
        require_execution_disabled=True,
        require_operator_review=True,
        require_live_merge_disabled=True,
        require_hot_patch_disabled=True,
        require_deterministic_hash=True,
    )

    from scpn_phase_orchestrator.supervisor.evolutionary_examples import (
        build_evolutionary_supervisor_search_examples,
    )
    from scpn_phase_orchestrator.supervisor.evolutionary_search import (
        run_offline_evolutionary_supervisor_search,
    )

    examples = build_evolutionary_supervisor_search_examples()
    if not examples:
        raise RuntimeError("evolutionary search examples list is empty")

    t0 = time.perf_counter()
    scenario_records: list[dict[str, object]] = []
    candidate_records: list[dict[str, object]] = []

    for idx, raw_example in enumerate(examples):
        payload = _evolutionary_example_payload(raw_example)
        scenario_id = str(
            payload.get("scenario_id", payload.get("scenario", f"scenario_{idx}"))
        )
        parent_policy = payload.get("parent_policy")
        if not isinstance(parent_policy, Mapping):
            raise ValueError("evolutionary example parent_policy must be mapping")
        audit_replays = payload.get("audit_replays")
        if not isinstance(audit_replays, Iterable) or isinstance(
            audit_replays, (str, bytes, bytearray)
        ):
            raise ValueError("evolutionary example audit_replays must be sequence")
        trace = payload.get("trace")
        if not isinstance(trace, Mapping):
            raise ValueError("evolutionary example trace must be mapping")
        stl_spec = payload.get("stl_spec")
        if not isinstance(stl_spec, str) or not stl_spec.strip():
            raise ValueError("evolutionary example stl_spec must be non-empty string")

        generation_count = int(payload.get("generation_count", 2))
        population_size = int(payload.get("population_size", 4))
        mutation_step = float(payload.get("mutation_step", 0.05))
        minimum_replay_reward = _evolutionary_example_scalar(
            payload=payload,
            key="minimum_replay_reward",
            default=0.0,
        )
        minimum_safety_margin = _evolutionary_example_scalar(
            payload=payload,
            key="minimum_safety_margin",
            default=0.0,
        )

        kwargs: dict[str, object] = {
            "parent_policy": parent_policy,
            "audit_replays": tuple(audit_replays),
            "stl_spec": stl_spec,
            "trace": trace,
            "generation_count": generation_count,
            "population_size": population_size,
            "mutation_step": mutation_step,
            "minimum_replay_reward": minimum_replay_reward,
            "minimum_safety_margin": minimum_safety_margin,
        }

        report = run_offline_evolutionary_supervisor_search(**kwargs)
        repeated = run_offline_evolutionary_supervisor_search(**kwargs)

        candidate_run_records = [
            _evolutionary_candidate_record(candidate.to_audit_record())
            for candidate in report.candidates
        ]
        repeated_candidate_records = [
            _evolutionary_candidate_record(candidate.to_audit_record())
            for candidate in repeated.candidates
        ]
        candidate_hash_match = int(
            _stable_record_hash(candidate_run_records)
            == _stable_record_hash(repeated_candidate_records)
        )
        report_hash_match = int(report.report_hash == repeated.report_hash)
        stl_filtered_rejections = sum(
            1
            for record in candidate_run_records
            if "stl_spec_not_satisfied" in record["blocked_reasons"]
        )
        counterfactual_filtered_rejections = sum(
            1
            for record in candidate_run_records
            if (
                "counterfactual_safety_delta_exceeds_replay_margin"
                in record["blocked_reasons"]
            )
        )
        for candidate_record in candidate_run_records:
            candidate_record["scenario_id"] = scenario_id
        candidate_records.extend(candidate_run_records)

        scenario_records.append(
            {
                "scenario_id": scenario_id,
                "candidate_count": int(report.candidate_count),
                "accepted_count": int(report.accepted_count),
                "rejected_count": int(report.rejected_count),
                "stl_filter_rejected_count": stl_filtered_rejections,
                "counterfactual_filter_rejected_count": (
                    counterfactual_filtered_rejections
                ),
                "candidate_hash_match": candidate_hash_match,
                "report_hash_match": report_hash_match,
                "report_hash": str(report.report_hash),
                "claim_boundary": str(report.claim_boundary),
                "best_candidate_id": (
                    str(report.best_candidate.candidate_id)
                    if report.best_candidate is not None
                    else ""
                ),
                "non_actuating": bool(report.non_actuating),
                "execution_disabled": bool(report.execution_disabled),
                "operator_review_required": bool(report.operator_review_required),
                "hot_patch_permitted": bool(report.hot_patch_permitted),
                "live_merge_permitted": bool(report.live_merge_permitted),
                "generation_count": generation_count,
                "population_size": population_size,
                "minimum_replay_reward": minimum_replay_reward,
                "minimum_safety_margin": minimum_safety_margin,
                "stl_spec": stl_spec,
            }
        )

    elapsed = time.perf_counter() - t0
    scenario_count = len(scenario_records)
    candidate_count = sum(int(record["candidate_count"]) for record in scenario_records)
    accepted_count = sum(int(record["accepted_count"]) for record in scenario_records)
    rejected_count = sum(int(record["rejected_count"]) for record in scenario_records)
    stl_filter_rejected_count = sum(
        int(record["stl_filter_rejected_count"]) for record in scenario_records
    )
    counterfactual_filter_rejected_count = sum(
        int(record["counterfactual_filter_rejected_count"])
        for record in scenario_records
    )
    deterministic_hash = int(
        all(record["candidate_hash_match"] == 1 for record in scenario_records)
        and all(record["report_hash_match"] == 1 for record in scenario_records)
    )
    non_actuating = int(
        all(record["non_actuating"] is True for record in scenario_records)
        and all(
            candidate["actuation_permitted"] is False for candidate in candidate_records
        )
    )
    execution_disabled = int(
        all(record["execution_disabled"] is True for record in scenario_records)
        and all(
            candidate["actuation_permitted"] is False for candidate in candidate_records
        )
    )
    operator_review_required = int(
        all(record["operator_review_required"] is True for record in scenario_records)
        and all(candidate["review_required"] is True for candidate in candidate_records)
    )
    live_merge_disabled = int(
        all(record["live_merge_permitted"] is False for record in scenario_records)
        and all(
            candidate["live_merge_permitted"] is False
            for candidate in candidate_records
        )
    )
    hot_patch_disabled = int(
        all(record["hot_patch_permitted"] is False for record in scenario_records)
        and all(
            candidate["hot_patch_permitted"] is False for candidate in candidate_records
        )
    )
    claim_boundary_value = str(scenario_records[0]["claim_boundary"])
    claim_boundary = int(
        all(
            str(record["claim_boundary"]) == claim_boundary_value
            and claim_boundary_value
            for record in scenario_records
        )
    )
    acceptance_passed = int(
        scenario_count >= thresholds.min_scenario_count
        and candidate_count >= thresholds.min_candidate_count
        and accepted_count >= thresholds.min_accepted_count
        and rejected_count >= thresholds.min_rejected_count
        and stl_filter_rejected_count >= thresholds.min_stl_filter_rejected_count
        and counterfactual_filter_rejected_count
        >= thresholds.min_counterfactual_filter_rejected_count
        and claim_boundary == 1
        and non_actuating == int(thresholds.require_non_actuating)
        and execution_disabled == int(thresholds.require_execution_disabled)
        and operator_review_required == int(thresholds.require_operator_review)
        and live_merge_disabled == int(thresholds.require_live_merge_disabled)
        and hot_patch_disabled == int(thresholds.require_hot_patch_disabled)
        and deterministic_hash == int(thresholds.require_deterministic_hash)
    )

    return {
        "suite": "evolutionary_supervisor_search",
        "wall_time_s": elapsed,
        "steps_per_second": candidate_count / elapsed if elapsed > 0.0 else 0.0,
        "scenario_count": scenario_count,
        "candidate_count": candidate_count,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "stl_filter_rejected_count": stl_filter_rejected_count,
        "counterfactual_filter_rejected_count": (counterfactual_filter_rejected_count),
        "claim_boundary": claim_boundary,
        "claim_boundary_value": claim_boundary_value,
        "non_actuating": non_actuating,
        "execution_disabled": execution_disabled,
        "operator_review_required": operator_review_required,
        "live_merge_disabled": live_merge_disabled,
        "hot_patch_disabled": hot_patch_disabled,
        "deterministic_hash": deterministic_hash,
        "evolutionary_search_sha256": _stable_record_hash(candidate_records),
        "acceptance_passed": acceptance_passed,
        "acceptance_thresholds_json": json.dumps(
            {
                "min_accepted_count": thresholds.min_accepted_count,
                "min_candidate_count": thresholds.min_candidate_count,
                "min_counterfactual_filter_rejected_count": (
                    thresholds.min_counterfactual_filter_rejected_count
                ),
                "min_rejected_count": thresholds.min_rejected_count,
                "min_scenario_count": thresholds.min_scenario_count,
                "min_stl_filter_rejected_count": (
                    thresholds.min_stl_filter_rejected_count
                ),
                "require_deterministic_hash": thresholds.require_deterministic_hash,
                "require_execution_disabled": thresholds.require_execution_disabled,
                "require_hot_patch_disabled": thresholds.require_hot_patch_disabled,
                "require_live_merge_disabled": thresholds.require_live_merge_disabled,
                "require_non_actuating": thresholds.require_non_actuating,
                "require_operator_review": thresholds.require_operator_review,
            },
            sort_keys=True,
        ),
        "scenario_records_json": json.dumps(scenario_records, sort_keys=True),
        "candidate_records_json": json.dumps(candidate_records, sort_keys=True),
    }


def benchmark_evolutionary_mutation_grammar_gate() -> dict[str, float | int | str]:
    """Benchmark richer offline policy, Petri-net, and topology mutation grammars."""
    thresholds = EvolutionaryMutationGrammarThresholds(
        min_grammar_count=3,
        min_candidate_count=20,
        min_mutation_kind_count=9,
        require_non_actuating=True,
        require_execution_disabled=True,
        require_operator_review=True,
        require_live_merge_disabled=True,
        require_hot_patch_disabled=True,
        require_deterministic_hash=True,
    )

    t0 = time.perf_counter()
    records = _build_evolutionary_mutation_grammar_records()
    repeated_records = _build_evolutionary_mutation_grammar_records()
    elapsed = time.perf_counter() - t0

    candidate_count = sum(int(record["candidate_count"]) for record in records)
    mutation_kinds = sorted(
        {
            str(kind)
            for record in records
            for kind in record["mutation_kinds"]
            if isinstance(record["mutation_kinds"], list)
        }
    )
    non_actuating = int(all(record["non_actuating"] is True for record in records))
    execution_disabled = int(
        all(record["execution_disabled"] is True for record in records)
    )
    operator_review_required = int(
        all(record["operator_review_required"] is True for record in records)
    )
    live_merge_disabled = int(
        all(record["live_merge_permitted"] is False for record in records)
    )
    hot_patch_disabled = int(
        all(record["hot_patch_permitted"] is False for record in records)
    )
    deterministic_hash = int(records == repeated_records)
    acceptance_passed = int(
        len(records) >= thresholds.min_grammar_count
        and candidate_count >= thresholds.min_candidate_count
        and len(mutation_kinds) >= thresholds.min_mutation_kind_count
        and non_actuating == int(thresholds.require_non_actuating)
        and execution_disabled == int(thresholds.require_execution_disabled)
        and operator_review_required == int(thresholds.require_operator_review)
        and live_merge_disabled == int(thresholds.require_live_merge_disabled)
        and hot_patch_disabled == int(thresholds.require_hot_patch_disabled)
        and deterministic_hash == int(thresholds.require_deterministic_hash)
    )

    return {
        "suite": "evolutionary_mutation_grammar_gate",
        "wall_time_s": elapsed,
        "steps_per_second": candidate_count / elapsed if elapsed > 0.0 else 0.0,
        "grammar_count": len(records),
        "candidate_count": candidate_count,
        "mutation_kind_count": len(mutation_kinds),
        "non_actuating": non_actuating,
        "execution_disabled": execution_disabled,
        "operator_review_required": operator_review_required,
        "live_merge_disabled": live_merge_disabled,
        "hot_patch_disabled": hot_patch_disabled,
        "deterministic_hash": deterministic_hash,
        "grammar_sha256": _stable_record_hash(records),
        "acceptance_passed": acceptance_passed,
        "mutation_kinds_json": json.dumps(mutation_kinds, sort_keys=True),
        "acceptance_thresholds_json": json.dumps(
            {
                "min_candidate_count": thresholds.min_candidate_count,
                "min_grammar_count": thresholds.min_grammar_count,
                "min_mutation_kind_count": thresholds.min_mutation_kind_count,
                "require_deterministic_hash": thresholds.require_deterministic_hash,
                "require_execution_disabled": thresholds.require_execution_disabled,
                "require_hot_patch_disabled": thresholds.require_hot_patch_disabled,
                "require_live_merge_disabled": thresholds.require_live_merge_disabled,
                "require_non_actuating": thresholds.require_non_actuating,
                "require_operator_review": thresholds.require_operator_review,
            },
            sort_keys=True,
        ),
        "grammar_records_json": json.dumps(records, sort_keys=True),
    }


def benchmark_federated_meta_orchestrator() -> dict[str, float | int | str]:
    """Benchmark offline federated aggregation and privacy gate evidence."""
    thresholds = FederatedMetaOrchestratorThresholds(
        min_node_count=3,
        min_accepted_node_count=3,
        min_policy_key_count=2,
        max_rejected_node_count=1,
        max_privacy_budget_spent=1.0,
        require_non_actuating=True,
        require_execution_disabled=True,
        require_operator_review=True,
        require_live_transport_disabled=True,
        require_raw_data_export_disabled=True,
        require_no_raw_time_series=True,
        require_deterministic_hash=True,
    )
    from scpn_phase_orchestrator.supervisor.federated import (
        build_federated_meta_orchestrator_manifest,
    )

    updates = (
        {
            "node_id": "site-a",
            "policy_delta": {"K": 0.10, "alpha": -0.02},
            "sample_count": 120,
            "local_loss": 0.21,
            "previous_audit_hash": "a" * 64,
            "privacy_epsilon_spent": 0.8,
        },
        {
            "node_id": "site-b",
            "policy_delta": {"K": 0.04, "alpha": -0.01},
            "sample_count": 80,
            "local_loss": 0.24,
            "previous_audit_hash": "b" * 64,
            "privacy_epsilon_spent": 0.6,
        },
        {
            "node_id": "site-c",
            "policy_delta": {"K": 0.08, "alpha": -0.03},
            "sample_count": 100,
            "local_loss": 0.19,
            "previous_audit_hash": "c" * 64,
            "privacy_epsilon_spent": 0.7,
        },
    )
    t0 = time.perf_counter()
    report = build_federated_meta_orchestrator_manifest(
        updates,
        required_policy_keys=("K", "alpha"),
        clipping_norm=0.2,
        epsilon=1.0,
        delta=1e-6,
        min_node_count=3,
    )
    repeated = build_federated_meta_orchestrator_manifest(
        updates,
        required_policy_keys=("K", "alpha"),
        clipping_norm=0.2,
        epsilon=1.0,
        delta=1e-6,
        min_node_count=3,
    )
    elapsed = time.perf_counter() - t0
    record = report.to_audit_record()
    repeated_record = repeated.to_audit_record()
    node_records = record["node_updates"]
    if not isinstance(node_records, list):
        raise TypeError("federated node_updates must be a list")
    deterministic_hash = int(
        record["report_hash"] == repeated_record["report_hash"]
        and record["aggregate_hash"] == repeated_record["aggregate_hash"]
    )
    raw_field_count = sum(
        int(
            isinstance(node_record, Mapping)
            and any(
                field in node_record
                for field in ("raw_time_series", "time_series", "samples")
            )
        )
        for node_record in node_records
    )
    aggregate_delta = record["aggregate_delta"]
    if not isinstance(aggregate_delta, list):
        raise TypeError("aggregate_delta must be a list")
    policy_key_count = len(aggregate_delta)
    acceptance_passed = int(
        len(node_records) >= thresholds.min_node_count
        and int(record["accepted_node_count"]) >= thresholds.min_accepted_node_count
        and int(record["rejected_node_count"]) <= thresholds.max_rejected_node_count
        and policy_key_count >= thresholds.min_policy_key_count
        and float(record["privacy_budget_spent"]) <= thresholds.max_privacy_budget_spent
        and int(record["non_actuating"] is True)
        == int(thresholds.require_non_actuating)
        and int(record["execution_disabled"] is True)
        == int(thresholds.require_execution_disabled)
        and int(record["operator_review_required"] is True)
        == int(thresholds.require_operator_review)
        and int(record["live_transport_permitted"] is False)
        == int(thresholds.require_live_transport_disabled)
        and int(record["raw_data_export_permitted"] is False)
        == int(thresholds.require_raw_data_export_disabled)
        and int(raw_field_count == 0) == int(thresholds.require_no_raw_time_series)
        and deterministic_hash == int(thresholds.require_deterministic_hash)
    )
    return {
        "suite": "federated_meta_orchestrator",
        "wall_time_s": elapsed,
        "steps_per_second": len(node_records) / elapsed if elapsed > 0.0 else 0.0,
        "node_count": len(node_records),
        "accepted_node_count": int(record["accepted_node_count"]),
        "rejected_node_count": int(record["rejected_node_count"]),
        "policy_key_count": policy_key_count,
        "total_sample_count": int(record["total_sample_count"]),
        "privacy_budget_spent": float(record["privacy_budget_spent"]),
        "privacy_budget_remaining": float(record["privacy_budget_remaining"]),
        "raw_time_series_received": int(record["raw_time_series_received"] is True),
        "raw_field_count": raw_field_count,
        "non_actuating": int(record["non_actuating"] is True),
        "execution_disabled": int(record["execution_disabled"] is True),
        "operator_review_required": int(record["operator_review_required"] is True),
        "live_transport_disabled": int(record["live_transport_permitted"] is False),
        "raw_data_export_disabled": int(record["raw_data_export_permitted"] is False),
        "actuation_disabled": int(record["actuation_permitted"] is False),
        "deterministic_hash": deterministic_hash,
        "claim_boundary": str(record["claim_boundary"]),
        "aggregate_hash": str(record["aggregate_hash"]),
        "report_hash": str(record["report_hash"]),
        "federated_meta_sha256": _stable_record_hash(record),
        "acceptance_passed": acceptance_passed,
        "acceptance_thresholds_json": json.dumps(
            {
                "max_privacy_budget_spent": thresholds.max_privacy_budget_spent,
                "max_rejected_node_count": thresholds.max_rejected_node_count,
                "min_accepted_node_count": thresholds.min_accepted_node_count,
                "min_node_count": thresholds.min_node_count,
                "min_policy_key_count": thresholds.min_policy_key_count,
                "require_deterministic_hash": thresholds.require_deterministic_hash,
                "require_execution_disabled": thresholds.require_execution_disabled,
                "require_live_transport_disabled": (
                    thresholds.require_live_transport_disabled
                ),
                "require_no_raw_time_series": thresholds.require_no_raw_time_series,
                "require_non_actuating": thresholds.require_non_actuating,
                "require_operator_review": thresholds.require_operator_review,
                "require_raw_data_export_disabled": (
                    thresholds.require_raw_data_export_disabled
                ),
            },
            sort_keys=True,
        ),
        "federated_record_json": json.dumps(record, sort_keys=True),
    }


def benchmark_federated_production_boundary_gate() -> dict[str, float | int | str]:
    """Gate offline transport, secure aggregation, and DP-noise service boundaries."""
    thresholds = FederatedProductionBoundaryThresholds(
        min_boundary_surface_count=3,
        min_transport_envelope_count=3,
        min_secure_accepted_node_count=3,
        min_dp_noise_vector_count=2,
        require_transport_execution_disabled=True,
        require_secure_execution_disabled=True,
        require_service_execution_disabled=True,
        require_raw_data_export_disabled=True,
        require_operator_review=True,
        require_non_actuating=True,
        require_deterministic_hash=True,
    )
    policy_keys = ("K", "alpha")
    transport_records = _federated_transport_fixture_records()
    secure_commitments = _federated_secure_commitment_fixture_records()
    dp_request = DpNoiseServiceRequestManifest(
        epsilon=2.5,
        delta=1e-6,
        sensitivity=1.25,
        noise_multiplier=0.8,
        node_count=3,
        seed_hash="f" * 64,
        policy_keys=policy_keys,
        node_budgets=(
            DpNoiseNodePrivacyBudget(node_id="site-a", epsilon_spent=0.5),
            DpNoiseNodePrivacyBudget(node_id="site-b", epsilon_spent=0.6),
            DpNoiseNodePrivacyBudget(node_id="site-c", epsilon_spent=0.4),
        ),
    )

    t0 = time.perf_counter()
    envelopes = build_signed_transport_envelopes(transport_records)
    transport_ledger = replay_federated_transport_batch(envelopes)
    secure_manifest = build_federated_secure_aggregation_manifest(
        secure_commitments,
        required_policy_keys=policy_keys,
        clipping_norm=1.0,
        min_node_count=3,
        epsilon=2.5,
        delta=1e-6,
    )
    dp_manifest = build_dp_noise_service_manifest(dp_request)
    repeated_envelopes = build_signed_transport_envelopes(transport_records)
    repeated_secure_manifest = build_federated_secure_aggregation_manifest(
        secure_commitments,
        required_policy_keys=policy_keys,
        clipping_norm=1.0,
        min_node_count=3,
        epsilon=2.5,
        delta=1e-6,
    )
    repeated_dp_manifest = build_dp_noise_service_manifest(dp_request)
    elapsed = time.perf_counter() - t0

    transport_record = transport_ledger.to_audit_record()
    secure_record = secure_manifest.to_audit_record()
    dp_record = dp_manifest.to_audit_record()
    boundary_record = {
        "transport": transport_record,
        "secure_aggregation": secure_record,
        "dp_noise_service": dp_record,
    }
    boundary_hash = _stable_record_hash(boundary_record)
    deterministic_hash = int(
        tuple(envelope.envelope_hash for envelope in envelopes)
        == tuple(envelope.envelope_hash for envelope in repeated_envelopes)
        and secure_manifest.report_hash == repeated_secure_manifest.report_hash
        and dp_manifest.audit_record_hash == repeated_dp_manifest.audit_record_hash
    )
    transport_execution_disabled = int(
        all(envelope.transport_execution_permitted is False for envelope in envelopes)
    )
    raw_data_export_disabled = int(
        all(envelope.raw_data_export_permitted is False for envelope in envelopes)
        and secure_manifest.raw_data_export_permitted is False
        and dp_manifest.raw_data_export_permitted is False
    )
    operator_review_required = int(
        all(envelope.operator_review_required is True for envelope in envelopes)
        and secure_manifest.operator_review_required is True
        and dp_manifest.operator_review_required is True
    )
    non_actuating = int(
        secure_manifest.non_actuating is True and dp_manifest.non_actuating is True
    )
    secure_execution_disabled = int(
        secure_manifest.secure_aggregation_execution_permitted is False
    )
    service_execution_disabled = int(dp_manifest.service_execution_permitted is False)
    boundary_surface_count = len(boundary_record)
    acceptance_passed = int(
        boundary_surface_count >= thresholds.min_boundary_surface_count
        and transport_ledger.envelope_count >= thresholds.min_transport_envelope_count
        and secure_manifest.accepted_node_count
        >= thresholds.min_secure_accepted_node_count
        and len(dp_manifest.policy_noise_audit_vector)
        >= thresholds.min_dp_noise_vector_count
        and transport_execution_disabled
        == int(thresholds.require_transport_execution_disabled)
        and secure_execution_disabled
        == int(thresholds.require_secure_execution_disabled)
        and service_execution_disabled
        == int(thresholds.require_service_execution_disabled)
        and raw_data_export_disabled == int(thresholds.require_raw_data_export_disabled)
        and operator_review_required == int(thresholds.require_operator_review)
        and non_actuating == int(thresholds.require_non_actuating)
        and deterministic_hash == int(thresholds.require_deterministic_hash)
    )

    return {
        "suite": "federated_production_boundary_gate",
        "wall_time_s": elapsed,
        "steps_per_second": (
            boundary_surface_count / elapsed if elapsed > 0.0 else 0.0
        ),
        "boundary_surface_count": boundary_surface_count,
        "transport_envelope_count": transport_ledger.envelope_count,
        "transport_node_sequence_count": len(transport_ledger.node_last_sequences),
        "secure_accepted_node_count": secure_manifest.accepted_node_count,
        "secure_rejected_node_count": secure_manifest.rejected_node_count,
        "dp_noise_vector_count": len(dp_manifest.policy_noise_audit_vector),
        "dp_privacy_budget_spent": dp_manifest.privacy_budget_spent,
        "dp_privacy_budget_remaining": dp_manifest.privacy_budget_remaining,
        "transport_execution_disabled": transport_execution_disabled,
        "secure_execution_disabled": secure_execution_disabled,
        "service_execution_disabled": service_execution_disabled,
        "raw_data_export_disabled": raw_data_export_disabled,
        "operator_review_required": operator_review_required,
        "non_actuating": non_actuating,
        "deterministic_hash": deterministic_hash,
        "boundary_hash": boundary_hash,
        "acceptance_passed": acceptance_passed,
        "acceptance_thresholds_json": json.dumps(
            {
                "min_boundary_surface_count": (thresholds.min_boundary_surface_count),
                "min_dp_noise_vector_count": thresholds.min_dp_noise_vector_count,
                "min_secure_accepted_node_count": (
                    thresholds.min_secure_accepted_node_count
                ),
                "min_transport_envelope_count": (
                    thresholds.min_transport_envelope_count
                ),
                "require_deterministic_hash": thresholds.require_deterministic_hash,
                "require_non_actuating": thresholds.require_non_actuating,
                "require_operator_review": thresholds.require_operator_review,
                "require_raw_data_export_disabled": (
                    thresholds.require_raw_data_export_disabled
                ),
                "require_secure_execution_disabled": (
                    thresholds.require_secure_execution_disabled
                ),
                "require_service_execution_disabled": (
                    thresholds.require_service_execution_disabled
                ),
                "require_transport_execution_disabled": (
                    thresholds.require_transport_execution_disabled
                ),
            },
            sort_keys=True,
        ),
        "boundary_record_json": json.dumps(boundary_record, sort_keys=True),
    }


def benchmark_federated_deployment_preflight_gate() -> dict[str, float | int | str]:
    """Gate review-only deployment preflights for federated runtime surfaces."""
    thresholds = FederatedDeploymentPreflightThresholds(
        min_preflight_surface_count=3,
        min_transport_preflight_count=1,
        min_secure_preflight_count=1,
        min_dp_preflight_count=1,
        require_transport_execution_disabled=True,
        require_secure_execution_disabled=True,
        require_service_execution_disabled=True,
        require_raw_data_export_disabled=True,
        require_operator_review=True,
        require_non_actuating=True,
        require_deterministic_hash=True,
    )
    policy_keys = ("K", "alpha")
    transport_records = _federated_transport_fixture_records()
    secure_commitments = _federated_secure_commitment_fixture_records()
    dp_request = DpNoiseServiceRequestManifest(
        epsilon=2.5,
        delta=1e-6,
        sensitivity=1.25,
        noise_multiplier=0.8,
        node_count=3,
        seed_hash="f" * 64,
        policy_keys=policy_keys,
        node_budgets=(
            DpNoiseNodePrivacyBudget(node_id="site-a", epsilon_spent=0.5),
            DpNoiseNodePrivacyBudget(node_id="site-b", epsilon_spent=0.6),
            DpNoiseNodePrivacyBudget(node_id="site-c", epsilon_spent=0.4),
        ),
    )

    t0 = time.perf_counter()
    envelopes = build_signed_transport_envelopes(transport_records)
    transport_ledger = replay_federated_transport_batch(envelopes)
    transport_preflight = build_transport_deployment_preflight_manifest(
        {
            "transport": "rest",
            "endpoint": "https://spo-federated-transport.internal/replay",
            "owner": "federated-runtime-owner",
            "auth_policy": "mtls+operator-token",
            "tls": True,
            "replay_supported": True,
            "operator_approved": True,
        },
        replay_ledger=transport_ledger,
    )
    secure_manifest = build_federated_secure_aggregation_manifest(
        secure_commitments,
        required_policy_keys=policy_keys,
        clipping_norm=1.0,
        min_node_count=3,
        epsilon=2.5,
        delta=1e-6,
    )
    secure_preflight = build_federated_secure_aggregation_preflight_manifest(
        secure_manifest,
        quorum_evidence=_federated_secure_quorum_fixture_records(),
        custody_rotation_policy="scheduled",
        custody_records=_federated_secure_custody_fixture_records("scheduled"),
        accepted_node_threshold=3,
        operator_approved=True,
        operator_id="federated-operator",
        service_owner="secure-aggregation-owner",
    )
    dp_manifest = build_dp_noise_service_manifest(dp_request)
    dp_preflight = build_dp_noise_service_deployment_preflight_manifest(
        dp_request,
        dp_manifest,
        mechanism_label="gaussian-mechanism-review-v1",
        privacy_accountant_owner="privacy-accountant-owner",
        seed_custody_label="seed-custody-ledger-federated",
        budget_issuer_label="budget-issuer-federated",
        service_endpoint_label="dp-noise-service-review-endpoint",
        operator_approved=True,
    )

    repeated_transport_preflight = build_transport_deployment_preflight_manifest(
        dict(transport_preflight.transport_audit_record),
        replay_ledger=transport_ledger,
    )
    repeated_secure_preflight = build_federated_secure_aggregation_preflight_manifest(
        secure_manifest,
        quorum_evidence=_federated_secure_quorum_fixture_records(),
        custody_rotation_policy="scheduled",
        custody_records=_federated_secure_custody_fixture_records("scheduled"),
        accepted_node_threshold=3,
        operator_approved=True,
        operator_id="federated-operator",
        service_owner="secure-aggregation-owner",
    )
    repeated_dp_preflight = build_dp_noise_service_deployment_preflight_manifest(
        dp_request,
        dp_manifest,
        mechanism_label="gaussian-mechanism-review-v1",
        privacy_accountant_owner="privacy-accountant-owner",
        seed_custody_label="seed-custody-ledger-federated",
        budget_issuer_label="budget-issuer-federated",
        service_endpoint_label="dp-noise-service-review-endpoint",
        operator_approved=True,
    )
    elapsed = time.perf_counter() - t0

    transport_record = transport_preflight.to_audit_record()
    secure_record = secure_preflight.to_audit_record()
    dp_record = dp_preflight.to_audit_record()
    preflight_record = {
        "transport_preflight": transport_record,
        "secure_aggregation_preflight": secure_record,
        "dp_noise_service_preflight": dp_record,
    }
    preflight_hash = _stable_record_hash(preflight_record)
    deterministic_hash = int(
        transport_preflight.preflight_hash
        == repeated_transport_preflight.preflight_hash
        and secure_preflight.report_hash == repeated_secure_preflight.report_hash
        and dp_preflight.audit_record_hash == repeated_dp_preflight.audit_record_hash
    )
    transport_execution_disabled = int(
        transport_preflight.transport_execution_permitted is False
    )
    secure_execution_disabled = int(
        secure_preflight.secure_aggregation_execution_permitted is False
    )
    service_execution_disabled = int(dp_preflight.service_execution_permitted is False)
    raw_data_export_disabled = int(
        transport_preflight.raw_data_export_permitted is False
        and secure_preflight.raw_data_export_permitted is False
        and dp_preflight.raw_data_export_permitted is False
    )
    operator_review_required = int(
        transport_preflight.operator_review_required is True
        and secure_preflight.operator_review_required is True
        and dp_preflight.operator_review_required is True
    )
    non_actuating = int(
        transport_preflight.non_actuating is True
        and secure_preflight.non_actuating is True
        and dp_preflight.non_actuating is True
    )
    preflight_surface_count = len(preflight_record)
    acceptance_passed = int(
        preflight_surface_count >= thresholds.min_preflight_surface_count
        and int(bool(transport_preflight.preflight_hash))
        >= thresholds.min_transport_preflight_count
        and int(bool(secure_preflight.report_hash))
        >= thresholds.min_secure_preflight_count
        and int(bool(dp_preflight.audit_record_hash))
        >= thresholds.min_dp_preflight_count
        and transport_execution_disabled
        == int(thresholds.require_transport_execution_disabled)
        and secure_execution_disabled
        == int(thresholds.require_secure_execution_disabled)
        and service_execution_disabled
        == int(thresholds.require_service_execution_disabled)
        and raw_data_export_disabled == int(thresholds.require_raw_data_export_disabled)
        and operator_review_required == int(thresholds.require_operator_review)
        and non_actuating == int(thresholds.require_non_actuating)
        and deterministic_hash == int(thresholds.require_deterministic_hash)
    )

    return {
        "suite": "federated_deployment_preflight_gate",
        "wall_time_s": elapsed,
        "steps_per_second": (
            preflight_surface_count / elapsed if elapsed > 0.0 else 0.0
        ),
        "preflight_surface_count": preflight_surface_count,
        "transport_preflight_count": int(bool(transport_preflight.preflight_hash)),
        "secure_preflight_count": int(bool(secure_preflight.report_hash)),
        "dp_preflight_count": int(bool(dp_preflight.audit_record_hash)),
        "transport_execution_disabled": transport_execution_disabled,
        "secure_execution_disabled": secure_execution_disabled,
        "service_execution_disabled": service_execution_disabled,
        "raw_data_export_disabled": raw_data_export_disabled,
        "operator_review_required": operator_review_required,
        "non_actuating": non_actuating,
        "deterministic_hash": deterministic_hash,
        "preflight_hash": preflight_hash,
        "acceptance_passed": acceptance_passed,
        "acceptance_thresholds_json": json.dumps(
            {
                "min_dp_preflight_count": thresholds.min_dp_preflight_count,
                "min_preflight_surface_count": (thresholds.min_preflight_surface_count),
                "min_secure_preflight_count": thresholds.min_secure_preflight_count,
                "min_transport_preflight_count": (
                    thresholds.min_transport_preflight_count
                ),
                "require_deterministic_hash": thresholds.require_deterministic_hash,
                "require_non_actuating": thresholds.require_non_actuating,
                "require_operator_review": thresholds.require_operator_review,
                "require_raw_data_export_disabled": (
                    thresholds.require_raw_data_export_disabled
                ),
                "require_secure_execution_disabled": (
                    thresholds.require_secure_execution_disabled
                ),
                "require_service_execution_disabled": (
                    thresholds.require_service_execution_disabled
                ),
                "require_transport_execution_disabled": (
                    thresholds.require_transport_execution_disabled
                ),
            },
            sort_keys=True,
        ),
        "preflight_record_json": json.dumps(preflight_record, sort_keys=True),
    }


def benchmark_topos_semantic_binding_gate() -> dict[str, float | int | str]:
    """Benchmark categorical proof-obligation surfaces for semantic binding."""
    thresholds = ToposSemanticBindingThresholds(
        min_semantic_report_count=2,
        min_policy_object_count=2,
        min_domain_example_count=3,
        min_obligation_count=12,
        require_non_actuating=True,
        require_proof_boundary=True,
        require_deterministic_hash=True,
    )

    t0 = time.perf_counter()
    semantic_reports = _topos_semantic_validation_reports()
    policy_report = _topos_policy_validation_report()
    domain_examples = _topos_domain_examples()
    repeated_records = _topos_semantic_binding_records()
    elapsed = time.perf_counter() - t0

    semantic_records = [
        _topos_validation_report_record(report.to_audit_record())
        for report in semantic_reports
    ]
    policy_record = _topos_validation_report_record(policy_report.to_audit_record())
    domain_records = [_topos_domain_example_record(item) for item in domain_examples]
    records = [*semantic_records, policy_record, *domain_records]
    obligation_count = sum(
        len(record.get("obligation_names", ())) for record in records
    )
    non_actuating = int(all(record["non_actuating"] is True for record in records))
    proof_boundary = int(
        all(
            record["proof_boundary"]
            == "categorical_validation_prototype_not_formal_topos_proof"
            for record in records
        )
    )
    deterministic_hash = int(records == repeated_records)
    acceptance_passed = int(
        len(semantic_records) >= thresholds.min_semantic_report_count
        and int(policy_record["object_count"]) >= thresholds.min_policy_object_count
        and len(domain_records) >= thresholds.min_domain_example_count
        and obligation_count >= thresholds.min_obligation_count
        and non_actuating == int(thresholds.require_non_actuating)
        and proof_boundary == int(thresholds.require_proof_boundary)
        and deterministic_hash == int(thresholds.require_deterministic_hash)
    )

    return {
        "suite": "topos_semantic_binding_gate",
        "record_count": len(records),
        "wall_time_s": elapsed,
        "steps_per_second": len(records) / elapsed,
        "semantic_report_count": len(semantic_records),
        "policy_object_count": int(policy_record["object_count"]),
        "domain_example_count": len(domain_records),
        "obligation_count": obligation_count,
        "non_actuating": non_actuating,
        "proof_boundary": proof_boundary,
        "deterministic_hash": deterministic_hash,
        "topos_sha256": _stable_record_hash(records),
        "acceptance_passed": acceptance_passed,
        "acceptance_thresholds_json": json.dumps(
            {
                "min_domain_example_count": thresholds.min_domain_example_count,
                "min_obligation_count": thresholds.min_obligation_count,
                "min_policy_object_count": thresholds.min_policy_object_count,
                "min_semantic_report_count": (thresholds.min_semantic_report_count),
                "require_deterministic_hash": thresholds.require_deterministic_hash,
                "require_non_actuating": thresholds.require_non_actuating,
                "require_proof_boundary": thresholds.require_proof_boundary,
            },
            sort_keys=True,
        ),
        "topos_records_json": json.dumps(records, sort_keys=True),
    }


def benchmark_multiverse_counterfactual_gate() -> dict[str, float | int | str]:
    """Benchmark non-actuating multiverse branch simulation and risk review."""
    thresholds = MultiverseCounterfactualThresholds(
        min_branch_count=4,
        min_domain_scenario_count=6,
        min_approved_branch_count=2,
        min_rejected_branch_count=1,
        require_non_actuating=True,
        require_execution_disabled=True,
        require_deterministic_hash=True,
        require_jax_backend_parity=True,
    )

    phases = np.array([0.0, 0.35, 0.85, 1.3, 1.9], dtype=np.float64)
    omegas = np.array([0.03, -0.015, 0.01, -0.005, 0.02], dtype=np.float64)
    baseline_k = np.full((5, 5), 0.14, dtype=np.float64)
    np.fill_diagonal(baseline_k, 0.0)
    baseline_alpha = np.zeros((5, 5), dtype=np.float64)
    sparse_topology = np.array(
        [
            [0.0, 1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    dense_topology = np.ones((5, 5), dtype=np.float64)
    np.fill_diagonal(dense_topology, 0.0)
    branch_specs = (
        MultiverseBranchSpec(branch_id="review_baseline", actions=()),
        MultiverseBranchSpec(
            branch_id="review_safe_coupling",
            actions=(ControlAction("K", "global", 0.04, 1.0, "safe coupling"),),
            topology_mask=sparse_topology,
        ),
        MultiverseBranchSpec(
            branch_id="review_phase_lag",
            actions=(
                ControlAction("alpha", "oscillator_1", 0.02, 1.0, "phase lag"),
                ControlAction("zeta", "global", 0.01, 1.0, "weak drive"),
            ),
            topology_mask=sparse_topology,
        ),
        MultiverseBranchSpec(
            branch_id="review_action_heavy",
            actions=tuple(
                ControlAction("K", "global", 0.02, 1.0, f"stress action {idx}")
                for idx in range(7)
            ),
            topology_mask=dense_topology,
        ),
    )

    t0 = time.perf_counter()
    manifest = simulate_multiverse_counterfactual_branches(
        phases=phases,
        omegas=omegas,
        baseline_k=baseline_k,
        baseline_alpha=baseline_alpha,
        branch_specs=branch_specs,
        horizon=16,
        dt=0.02,
    )
    repeated = simulate_multiverse_counterfactual_branches(
        phases=phases,
        omegas=omegas,
        baseline_k=baseline_k,
        baseline_alpha=baseline_alpha,
        branch_specs=branch_specs,
        horizon=16,
        dt=0.02,
    )
    jax_manifest = simulate_multiverse_counterfactual_branches(
        phases=phases,
        omegas=omegas,
        baseline_k=baseline_k,
        baseline_alpha=baseline_alpha,
        branch_specs=branch_specs,
        horizon=16,
        dt=0.02,
        backend="jax",
    )
    risk_report = evaluate_multiverse_branch_risk(
        manifest.to_audit_record(),
        MultiverseRiskThresholds(
            min_mean_R=0.45,
            min_final_R=0.45,
            max_action_count=2,
            max_topology_edge_count=20,
            max_topology_scale=10.0,
        ),
    )
    domain_scenarios = build_multiverse_domain_scenarios()
    elapsed = time.perf_counter() - t0

    manifest_record = manifest.to_audit_record()
    risk_record = risk_report.to_audit_record()
    branch_records = manifest_record["branch_records"]
    deterministic_hash = int(manifest.manifest_hash == repeated.manifest_hash)
    jax_backend_parity = int(
        manifest.branch_count == jax_manifest.branch_count
        and all(
            numpy_record.branch_id == jax_record.branch_id
            and numpy_record.branch_hash == jax_record.branch_hash
            and numpy_record.action_count == jax_record.action_count
            and numpy_record.action_labels == jax_record.action_labels
            and numpy_record.topology_edge_count == jax_record.topology_edge_count
            and np.isclose(
                numpy_record.topology_scale,
                jax_record.topology_scale,
                atol=1e-12,
            )
            and np.isclose(numpy_record.final_R, jax_record.final_R, atol=1e-10)
            and np.isclose(numpy_record.mean_R, jax_record.mean_R, atol=1e-10)
            and np.isclose(numpy_record.min_R, jax_record.min_R, atol=1e-10)
            and np.isclose(numpy_record.max_R, jax_record.max_R, atol=1e-10)
            and np.isclose(numpy_record.final_psi, jax_record.final_psi, atol=1e-10)
            for numpy_record, jax_record in zip(
                manifest.branch_records,
                jax_manifest.branch_records,
                strict=True,
            )
        )
    )
    non_actuating = int(
        manifest.non_actuating is True
        and risk_report.non_actuating is True
        and all(record["non_actuating"] is True for record in domain_scenarios)
    )
    execution_disabled = int(
        manifest.execution_disabled is True
        and risk_report.execution_disabled is True
        and all(record["execution_disabled"] is True for record in domain_scenarios)
    )
    acceptance_passed = int(
        len(branch_records) >= thresholds.min_branch_count
        and len(domain_scenarios) >= thresholds.min_domain_scenario_count
        and risk_report.approved_count >= thresholds.min_approved_branch_count
        and risk_report.rejected_count >= thresholds.min_rejected_branch_count
        and non_actuating == int(thresholds.require_non_actuating)
        and execution_disabled == int(thresholds.require_execution_disabled)
        and deterministic_hash == int(thresholds.require_deterministic_hash)
        and jax_backend_parity == int(thresholds.require_jax_backend_parity)
    )

    return {
        "suite": "multiverse_counterfactual_gate",
        "branch_count": len(branch_records),
        "domain_scenario_count": len(domain_scenarios),
        "approved_branch_count": risk_report.approved_count,
        "rejected_branch_count": risk_report.rejected_count,
        "wall_time_s": elapsed,
        "steps_per_second": len(branch_records) / elapsed,
        "non_actuating": non_actuating,
        "execution_disabled": execution_disabled,
        "deterministic_hash": deterministic_hash,
        "jax_backend_parity": jax_backend_parity,
        "numpy_backend": manifest.backend,
        "jax_backend": jax_manifest.backend,
        "manifest_sha256": manifest.manifest_hash,
        "risk_report_sha256": risk_report.report_hash,
        "safest_branch_id": risk_report.safest_branch_id or "",
        "acceptance_passed": acceptance_passed,
        "acceptance_thresholds_json": json.dumps(
            {
                "min_approved_branch_count": thresholds.min_approved_branch_count,
                "min_branch_count": thresholds.min_branch_count,
                "min_domain_scenario_count": thresholds.min_domain_scenario_count,
                "min_rejected_branch_count": thresholds.min_rejected_branch_count,
                "require_deterministic_hash": thresholds.require_deterministic_hash,
                "require_execution_disabled": thresholds.require_execution_disabled,
                "require_jax_backend_parity": (thresholds.require_jax_backend_parity),
                "require_non_actuating": thresholds.require_non_actuating,
            },
            sort_keys=True,
        ),
        "branch_records_json": json.dumps(branch_records, sort_keys=True),
        "risk_report_json": json.dumps(risk_record, sort_keys=True),
        "domain_scenarios_json": json.dumps(domain_scenarios, sort_keys=True),
    }


def benchmark_hybrid_entanglement_order_parameter_gate() -> dict[
    str, float | int | str
]:
    """Benchmark hybrid entanglement-aware order parameters on deterministic cases."""
    thresholds = HybridEntanglementOrderThresholds(
        max_product_entropy=0.15,
        min_bell_entropy=0.95,
        min_entropy_gap=0.80,
        min_record_count=2,
        require_non_actuating=True,
        require_execution_disabled=True,
        require_claim_boundary=True,
        require_deterministic_hash=True,
    )

    from scpn_phase_orchestrator.monitor.hybrid_order import (
        compute_hybrid_entanglement_order_parameter,
    )
    from scpn_phase_orchestrator.monitor.hybrid_order_examples import (
        build_hybrid_order_parameter_scenarios,
    )

    def _amplitude_pairs_to_statevector(candidate: Mapping[str, object]) -> np.ndarray:
        amplitudes = candidate.get("amplitudes")
        if not isinstance(amplitudes, list):
            raise ValueError("hybrid order candidate amplitudes must be a list")
        values: list[complex] = []
        for pair in amplitudes:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or not all(isinstance(value, int | float) for value in pair)
            ):
                raise ValueError(
                    "hybrid order candidate amplitudes must be [real, imag] pairs"
                )
            values.append(complex(float(pair[0]), float(pair[1])))
        return np.asarray(values, dtype=np.complex128)

    base_phases = np.array([0.0, 0.82, 1.56, 2.30], dtype=np.float64)
    product_state = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    bell_state = np.array(
        [1 / np.sqrt(2), 0.0, 0.0, 1 / np.sqrt(2)],
        dtype=np.complex128,
    )
    scenario_specs: list[dict[str, object]] = [
        {
            "name": "deterministic_product_state",
            "category": "product",
            "phases": base_phases,
            "quantum_state": product_state,
            "bipartition": ((0,), (1,)),
        },
        {
            "name": "deterministic_bell_like_state",
            "category": "bell_like",
            "phases": base_phases,
            "quantum_state": bell_state,
            "bipartition": ((0,), (1,)),
        },
    ]

    for scenario in build_hybrid_order_parameter_scenarios():
        phases = np.asarray(scenario["phases"], dtype=np.float64)
        bipartition = tuple(
            tuple(int(index) for index in part) for part in scenario["bipartition"]
        )
        for candidate in scenario["state_candidates"]:
            if not isinstance(candidate, Mapping):
                raise ValueError("hybrid order state candidates must be mappings")
            scenario_specs.append(
                {
                    "name": f"{scenario['scenario_id']}:{candidate['state_id']}",
                    "category": str(candidate["state_type"]),
                    "phases": phases,
                    "quantum_state": _amplitude_pairs_to_statevector(candidate),
                    "bipartition": bipartition,
                }
            )

    t0 = time.perf_counter()
    records: list[dict[str, object]] = []
    repeated_records: list[dict[str, object]] = []
    for spec in scenario_specs:
        phases = np.asarray(spec["phases"], dtype=np.float64)
        quantum_state = np.asarray(spec["quantum_state"], dtype=np.complex128)
        bipartition_raw = spec["bipartition"]
        if not isinstance(bipartition_raw, tuple):
            raise ValueError("hybrid order bipartition must be a tuple")
        bipartition = tuple(
            tuple(int(index) for index in part) for part in bipartition_raw
        )
        result = compute_hybrid_entanglement_order_parameter(
            phases=phases,
            quantum_state=quantum_state,
            bipartition=bipartition,
        )
        repeated = compute_hybrid_entanglement_order_parameter(
            phases=phases,
            quantum_state=quantum_state,
            bipartition=bipartition,
        )
        result_record = result.to_audit_record()
        repeated_record = repeated.to_audit_record()
        records.append(
            {
                "scenario": str(spec["name"]),
                "category": str(spec["category"]),
                "R": float(result_record["R"]),
                "Psi": float(result_record["Psi"]),
                "entanglement_entropy": float(result_record["entanglement_entropy"]),
                "normalised_entanglement_entropy": float(
                    result_record["normalised_entanglement_entropy"]
                ),
                "participation_ratio": float(result_record["participation_ratio"]),
                "qubit_count": int(result_record["qubit_count"]),
                "bipartition": result_record["bipartition"],
                "backend": str(result_record["backend"]),
                "claim_boundary": str(result_record["claim_boundary"]),
                "non_actuating": bool(result_record["non_actuating"]),
                "execution_disabled": bool(result_record["execution_disabled"]),
                "record_hash": str(result_record["record_hash"]),
            }
        )
        repeated_records.append(repeated_record)

    elapsed = time.perf_counter() - t0
    product_records = [
        record for record in records if "product" in str(record["category"]).lower()
    ]
    bell_records = [
        record for record in records if "bell" in str(record["category"]).lower()
    ]
    if not product_records or not bell_records:
        raise RuntimeError(
            "Hybrid entanglement benchmark requires product and bell-like scenarios"
        )

    product_entropy = min(
        float(record["entanglement_entropy"]) for record in product_records
    )
    bell_entropy = max(float(record["entanglement_entropy"]) for record in bell_records)
    entanglement_gap = bell_entropy - product_entropy
    deterministic_bundle_hashes = [
        int(record["record_hash"] == repeated_record["record_hash"])
        for record, repeated_record in zip(records, repeated_records, strict=False)
    ]
    deterministic_record_hash = int(
        all(value == 1 for value in deterministic_bundle_hashes)
    )
    deterministic_hash = deterministic_record_hash
    claim_boundary_value = str(records[0]["claim_boundary"])
    claim_boundary = int(
        all(record["claim_boundary"] == claim_boundary_value for record in records)
    )
    non_actuating = int(all(record["non_actuating"] is True for record in records))
    execution_disabled = int(
        all(record["execution_disabled"] is True for record in records)
    )
    acceptance_passed = int(
        len(records) >= thresholds.min_record_count
        and product_entropy <= thresholds.max_product_entropy
        and bell_entropy >= thresholds.min_bell_entropy
        and entanglement_gap >= thresholds.min_entropy_gap
        and non_actuating == int(thresholds.require_non_actuating)
        and execution_disabled == int(thresholds.require_execution_disabled)
        and claim_boundary == int(thresholds.require_claim_boundary)
        and deterministic_hash == int(thresholds.require_deterministic_hash)
        and deterministic_record_hash == int(thresholds.require_deterministic_hash)
    )

    return {
        "suite": "hybrid_entanglement_order_parameter_gate",
        "scenario_count": len(records),
        "wall_time_s": elapsed,
        "steps_per_second": len(records) / elapsed if elapsed > 0.0 else 0.0,
        "product_case_count": len(product_records),
        "bell_case_count": len(bell_records),
        "max_entropy": max(float(record["entanglement_entropy"]) for record in records),
        "min_entropy": min(float(record["entanglement_entropy"]) for record in records),
        "entanglement_gap": entanglement_gap,
        "non_actuating": non_actuating,
        "execution_disabled": execution_disabled,
        "claim_boundary": claim_boundary,
        "deterministic_hash": deterministic_hash,
        "hybrid_sha256": _stable_record_hash(records),
        "acceptance_passed": acceptance_passed,
        "acceptance_thresholds_json": json.dumps(
            {
                "max_product_entropy": thresholds.max_product_entropy,
                "min_bell_entropy": thresholds.min_bell_entropy,
                "min_entropy_gap": thresholds.min_entropy_gap,
                "min_record_count": thresholds.min_record_count,
                "require_claim_boundary": thresholds.require_claim_boundary,
                "require_deterministic_hash": thresholds.require_deterministic_hash,
                "require_execution_disabled": thresholds.require_execution_disabled,
                "require_non_actuating": thresholds.require_non_actuating,
            },
            sort_keys=True,
        ),
        "claim_boundary_value": claim_boundary_value,
        "hybrid_records_json": json.dumps(records, sort_keys=True),
    }


def benchmark_information_geometry_control_gate() -> dict[str, float | int | str]:
    """Benchmark deterministic non-actuating information-geometry control proposals."""
    thresholds = InformationGeometryControlThresholds(
        min_scenario_count=2,
        min_finite_metric_count=8,
        min_action_evidence_count=2,
        require_non_actuating=True,
        require_execution_disabled=True,
        require_claim_boundary=True,
        require_deterministic_hash=True,
        require_jax_backend_parity=True,
    )

    from scpn_phase_orchestrator.supervisor.information_geometry import (
        propose_information_geometry_control,
    )
    from scpn_phase_orchestrator.supervisor.information_geometry_examples import (
        build_information_geometry_control_scenarios,
    )

    def _coerce_distribution(
        *,
        candidate: object,
        fallback_dimension: int,
        rotation: int,
    ) -> np.ndarray:
        values: np.ndarray | None = None
        if isinstance(candidate, Mapping):
            for key in (
                "current_distribution",
                "target_distribution",
                "distribution",
                "simplex",
                "probabilities",
                "p",
                "source_distribution",
                "source_probabilities",
                "state",
            ):
                raw = candidate.get(key)
                if isinstance(raw, Iterable) and not isinstance(raw, (str, bytes)):
                    values = np.asarray(raw, dtype=np.float64)
                    break
        elif isinstance(candidate, Iterable) and not isinstance(
            candidate, (str, bytes)
        ):
            values = np.asarray(candidate, dtype=np.float64)

        if (
            values is None
            or values.ndim != 1
            or len(values) < 2
            or not np.all(np.isfinite(values))
            or np.allclose(values, 0.0)
        ):
            values = np.arange(1.0, float(fallback_dimension) + 1.0) + 0.13 * rotation
            values = np.roll(values, rotation % fallback_dimension).astype(np.float64)

        values = np.clip(values, 1e-12, None)
        if values.ndim != 1 or values.size < 2:
            raise ValueError(
                "information geometry simplex must be a 1D array with length >= 2"
            )
        return values / float(np.sum(values))

    def _to_record(candidate: object) -> Mapping[str, object]:
        if isinstance(candidate, Mapping):
            return candidate
        if hasattr(candidate, "to_audit_record"):
            audit_record = candidate.to_audit_record()
            if isinstance(audit_record, Mapping):
                return audit_record
            raise TypeError("proposal audit record is not a mapping")
        if hasattr(candidate, "__dict__"):
            return dict(candidate.__dict__)
        raise TypeError("proposal does not expose an audit record mapping")

    def _metric(
        candidate: Mapping[str, object],
        keys: tuple[str, ...],
        *,
        state_keys: tuple[str, ...] = (),
    ) -> float:
        for key in keys:
            value = candidate.get(key)
            if isinstance(value, np.ndarray):
                if value.shape == ():
                    value = value.item()
                else:
                    continue
            if isinstance(value, np.bool_):
                continue
            if isinstance(value, int | float | np.floating):
                candidate_value = float(value)
                if np.isfinite(candidate_value):
                    return candidate_value

        state = candidate.get("state")
        if isinstance(state, Mapping):
            for key in state_keys:
                value = state.get(key)
                if isinstance(value, np.ndarray):
                    if value.shape == ():
                        value = value.item()
                    else:
                        continue
                if isinstance(value, np.bool_):
                    continue
                if isinstance(value, int | float | np.floating):
                    candidate_value = float(value)
                    if np.isfinite(candidate_value):
                        return candidate_value

        raise RuntimeError(f"missing or non-finite metric for aliases: {keys}")

    def _as_int_bool(value: object) -> int:
        return int(bool(value))

    def _bool_from_candidate(
        candidate: Mapping[str, object], key: str, default: bool
    ) -> bool:
        if key in candidate:
            return bool(candidate[key])
        return default

    def _evidence_count(candidate: Mapping[str, object]) -> int:
        for key in (
            "actions",
            "action_evidence",
            "proposals",
            "proposed_actions",
            "control_actions",
            "plan",
            "action_proposals",
        ):
            value = candidate.get(key)
            if isinstance(value, Mapping):
                return int(len(value))
            if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
                return len(list(value))
            if value:
                return 1
        return 0

    def _call_proposal(
        current_distribution: np.ndarray,
        target_distribution: np.ndarray,
        fixture: Mapping[str, object],
        *,
        backend: str,
    ) -> object:
        max_step_raw = fixture.get("max_step", 0.05)
        if not isinstance(max_step_raw, (int, float)) or max_step_raw <= 0.0:
            max_step = 0.05
        else:
            max_step = float(max_step_raw)
        knob_name = (
            fixture.get("knob_hints")[0]
            if isinstance(fixture.get("knob_hints"), list) and fixture.get("knob_hints")
            else "K"
        )
        return propose_information_geometry_control(
            current_distribution=current_distribution,
            target_distribution=target_distribution,
            max_step=max_step,
            knob=str(knob_name),
            backend=backend,
        )

    scenario_fixtures = tuple(build_information_geometry_control_scenarios())
    if not scenario_fixtures:
        raise RuntimeError("information geometry scenario fixture is empty")
    t0 = time.perf_counter()
    records: list[dict[str, object]] = []
    finite_metric_total = 0
    proposal_action_evidence_count = 0
    jax_backend_parity_total = 0

    for idx, scenario in enumerate(scenario_fixtures):
        fixture: Mapping[str, object] = (
            scenario if isinstance(scenario, Mapping) else {}
        )
        scenario_name = str(
            fixture.get(
                "scenario_id",
                fixture.get("scenario", f"information_geometry_scenario_{idx}"),
            )
        )
        source_dimension = int(
            float(
                fixture.get(
                    "dimension",
                    fixture.get("size", fixture.get("num_states", 4)),
                )
            )
        )
        source_dimension = max(2, source_dimension)
        source_distribution = _coerce_distribution(
            candidate=(
                fixture.get("source_distribution")
                if isinstance(fixture.get("source_distribution"), Iterable)
                else (
                    fixture.get("source")
                    if fixture.get("source") is not None
                    else fixture
                )
            ),
            fallback_dimension=source_dimension,
            rotation=idx,
        )
        target_distribution = _coerce_distribution(
            candidate=(
                fixture.get("target_distribution")
                if isinstance(fixture.get("target_distribution"), Iterable)
                else (
                    fixture.get("target")
                    if fixture.get("target") is not None
                    else fixture
                )
            ),
            fallback_dimension=source_dimension,
            rotation=idx + 1,
        )
        proposal = _call_proposal(
            source_distribution,
            target_distribution,
            fixture,
            backend="numpy",
        )
        repeated = _call_proposal(
            source_distribution,
            target_distribution,
            fixture,
            backend="numpy",
        )
        jax_proposal = _call_proposal(
            source_distribution,
            target_distribution,
            fixture,
            backend="jax",
        )
        proposal_record = _to_record(proposal)
        repeated_record = _to_record(repeated)
        jax_record = _to_record(jax_proposal)

        proposal_hash = str(
            proposal_record.get(
                "proposal_hash",
                _stable_record_hash(proposal_record),
            )
        )
        repeated_hash = str(
            repeated_record.get(
                "proposal_hash",
                _stable_record_hash(repeated_record),
            )
        )
        proposal_repeat_match = int(proposal_hash == repeated_hash)
        fisher_rao = _metric(
            proposal_record,
            (
                "fisher_rao_distance",
                "fisher_rao",
                "fisher_rao_metric",
            ),
        )
        wasserstein = _metric(
            proposal_record,
            ("wasserstein_distance", "wasserstein", "earth_movers_distance"),
        )
        geodesic = _metric(
            proposal_record,
            ("geodesic_distance", "geodesic", "geodesic_metric"),
            state_keys=("geodesic_length",),
        )
        curvature = _metric(
            proposal_record,
            (
                "curvature",
                "curvature_proxy",
                "riemannian_curvature",
                "information_curvature",
            ),
            state_keys=("curvature_proxy",),
        )
        jax_fisher_rao = _metric(
            jax_record,
            (
                "fisher_rao_distance",
                "fisher_rao",
                "fisher_rao_metric",
            ),
        )
        jax_wasserstein = _metric(
            jax_record,
            ("wasserstein_distance", "wasserstein", "earth_movers_distance"),
        )
        jax_geodesic = _metric(
            jax_record,
            ("geodesic_distance", "geodesic", "geodesic_metric"),
            state_keys=("geodesic_length",),
        )
        jax_curvature = _metric(
            jax_record,
            (
                "curvature",
                "curvature_proxy",
                "riemannian_curvature",
                "information_curvature",
            ),
            state_keys=("curvature_proxy",),
        )
        jax_backend = str(jax_record.get("backend", ""))
        jax_claim_boundary = str(jax_record.get("claim_boundary", ""))
        proposal_claim_boundary = str(proposal_record.get("claim_boundary", ""))
        jax_parity_match = int(
            jax_backend == "jax_native_information_geometry"
            and jax_claim_boundary == proposal_claim_boundary
            and bool(jax_record.get("non_actuating")) is True
            and bool(jax_record.get("execution_disabled")) is True
            and np.isclose(jax_fisher_rao, fisher_rao)
            and np.isclose(jax_wasserstein, wasserstein)
            and np.isclose(jax_geodesic, geodesic)
            and np.isclose(jax_curvature, curvature)
        )
        jax_backend_parity_total += jax_parity_match
        scenario_evidence = _evidence_count(proposal_record)
        proposal_action_evidence_count += int(scenario_evidence > 0)
        finite_metric_total += (
            _as_int_bool(np.isfinite(fisher_rao))
            + _as_int_bool(np.isfinite(wasserstein))
            + _as_int_bool(np.isfinite(geodesic))
            + _as_int_bool(np.isfinite(curvature))
        )

        non_actuating = bool(
            _bool_from_candidate(
                proposal_record,
                "non_actuating",
                default=False,
            )
            if proposal_record.get("non_actuating") is not None
            else bool(proposal_record.get("actuation_permitted") is False)
        )
        execution_disabled = bool(
            _bool_from_candidate(
                proposal_record,
                "execution_disabled",
                default=False,
            )
            if proposal_record.get("execution_disabled") is not None
            else bool(proposal_record.get("actuation_permitted") is False)
        )
        claim_boundary = str(
            proposal_record.get(
                "claim_boundary",
                proposal_record.get("claim", ""),
            )
        )
        records.append(
            {
                "scenario": scenario_name,
                "source_distribution": source_distribution.tolist(),
                "target_distribution": target_distribution.tolist(),
                "fisher_rao_distance": fisher_rao,
                "wasserstein_distance": wasserstein,
                "geodesic_distance": geodesic,
                "curvature": curvature,
                "proposal_hash": proposal_hash,
                "repeat_match": proposal_repeat_match,
                "proposal_action_count": scenario_evidence,
                "non_actuating": non_actuating,
                "execution_disabled": execution_disabled,
                "claim_boundary": claim_boundary,
                "jax_backend": jax_backend,
                "jax_parity_match": jax_parity_match,
            }
        )

    elapsed = time.perf_counter() - t0
    deterministic_hash = int(all(record["repeat_match"] == 1 for record in records))
    non_actuating = int(all(record["non_actuating"] is True for record in records))
    execution_disabled = int(
        all(record["execution_disabled"] is True for record in records)
    )
    claim_boundary_value = str(records[0]["claim_boundary"])
    claim_boundary = int(
        all(
            str(record["claim_boundary"]) == claim_boundary_value
            and claim_boundary_value
            for record in records
        )
    )
    acceptance_passed = int(
        len(records) >= thresholds.min_scenario_count
        and finite_metric_total >= thresholds.min_finite_metric_count
        and proposal_action_evidence_count >= thresholds.min_action_evidence_count
        and non_actuating == int(thresholds.require_non_actuating)
        and execution_disabled == int(thresholds.require_execution_disabled)
        and claim_boundary == int(thresholds.require_claim_boundary)
        and deterministic_hash == int(thresholds.require_deterministic_hash)
        and jax_backend_parity_total
        == len(records) * int(thresholds.require_jax_backend_parity)
    )

    fisher_rao_values = [record["fisher_rao_distance"] for record in records]
    wasserstein_values = [record["wasserstein_distance"] for record in records]
    curvature_values = [record["curvature"] for record in records]

    return {
        "suite": "information_geometry_control_gate",
        "scenario_count": len(records),
        "wall_time_s": elapsed,
        "steps_per_second": len(records) / elapsed if elapsed > 0.0 else 0.0,
        "non_actuating": non_actuating,
        "execution_disabled": execution_disabled,
        "claim_boundary": claim_boundary,
        "claim_boundary_value": claim_boundary_value,
        "proposal_action_evidence_count": proposal_action_evidence_count,
        "finite_metric_count": finite_metric_total,
        "deterministic_hash": deterministic_hash,
        "jax_backend_parity": int(jax_backend_parity_total == len(records)),
        "jax_backend_value": "jax_native_information_geometry",
        "min_fisher_rao_distance": min(fisher_rao_values),
        "max_fisher_rao_distance": max(fisher_rao_values),
        "min_wasserstein_distance": min(wasserstein_values),
        "max_wasserstein_distance": max(wasserstein_values),
        "min_curvature": min(curvature_values),
        "max_curvature": max(curvature_values),
        "information_geometry_sha256": _stable_record_hash(records),
        "acceptance_passed": acceptance_passed,
        "acceptance_thresholds_json": json.dumps(
            {
                "min_action_evidence_count": thresholds.min_action_evidence_count,
                "min_finite_metric_count": thresholds.min_finite_metric_count,
                "min_scenario_count": thresholds.min_scenario_count,
                "require_claim_boundary": thresholds.require_claim_boundary,
                "require_deterministic_hash": thresholds.require_deterministic_hash,
                "require_execution_disabled": thresholds.require_execution_disabled,
                "require_jax_backend_parity": thresholds.require_jax_backend_parity,
                "require_non_actuating": thresholds.require_non_actuating,
            },
            sort_keys=True,
        ),
        "information_geometry_records_json": json.dumps(records, sort_keys=True),
    }


def benchmark_sheaf_obstruction_domain_gate() -> dict[str, float | int | str]:
    """Benchmark heterogeneous sheaf-obstruction demos and residual triage."""
    thresholds = SheafObstructionBenchmarkThresholds(
        min_demo_count=6,
        min_summary_count=6,
        min_top_residual_edge_count=18,
        min_critical_count=5,
        min_obstruction_delta=0.1,
        min_control_energy_reduction=0.1,
        max_nominal_obstruction_score=0.35,
        require_non_actuating=True,
        require_execution_disabled=True,
        require_operator_review=True,
        require_deterministic_hash=True,
    )
    module_paths = (
        "domainpacks.cardiac_rhythm.sheaf_obstruction_demo",
        "domainpacks.edge_consensus_nchannel.sheaf_obstruction_demo",
        "domainpacks.manufacturing_spc.sheaf_obstruction_demo",
        "domainpacks.power_grid.sheaf_obstruction_demo",
        "domainpacks.network_security.sheaf_obstruction_demo",
        "domainpacks.traffic_flow.sheaf_obstruction_demo",
    )

    t0 = time.perf_counter()
    demos = [_load_sheaf_obstruction_demo(module_path) for module_path in module_paths]
    repeated = [
        _load_sheaf_obstruction_demo(module_path) for module_path in module_paths
    ]
    elapsed = time.perf_counter() - t0

    records = [_sheaf_obstruction_demo_record(demo) for demo in demos]
    repeated_records = [_sheaf_obstruction_demo_record(demo) for demo in repeated]
    control_record = _sheaf_obstruction_control_record()
    repeated_control_record = _sheaf_obstruction_control_record()
    summary_count = sum(int(record["summary_present"]) for record in records)
    top_residual_edge_count = sum(
        int(record["top_residual_edge_count"]) for record in records
    )
    critical_count = sum(
        int(record["incident_severity"] == "critical") for record in records
    )
    min_obstruction_delta = min(
        float(record["obstruction_delta"]) for record in records
    )
    max_nominal_obstruction_score = max(
        float(record["nominal_obstruction_score"]) for record in records
    )
    control_energy_reduction = float(control_record["control_energy_reduction"])
    non_actuating = int(
        all(record["actuating"] is False for record in records)
        and control_record["non_actuating"] is True
    )
    execution_disabled = int(control_record["execution_disabled"] is True)
    operator_review_required = int(control_record["operator_review_required"] is True)
    deterministic_hash = int(
        records == repeated_records and control_record == repeated_control_record
    )
    acceptance_passed = int(
        len(records) >= thresholds.min_demo_count
        and summary_count >= thresholds.min_summary_count
        and top_residual_edge_count >= thresholds.min_top_residual_edge_count
        and critical_count >= thresholds.min_critical_count
        and min_obstruction_delta >= thresholds.min_obstruction_delta
        and control_energy_reduction >= thresholds.min_control_energy_reduction
        and max_nominal_obstruction_score <= thresholds.max_nominal_obstruction_score
        and non_actuating == int(thresholds.require_non_actuating)
        and execution_disabled == int(thresholds.require_execution_disabled)
        and operator_review_required == int(thresholds.require_operator_review)
        and deterministic_hash == int(thresholds.require_deterministic_hash)
    )

    return {
        "suite": "sheaf_obstruction_domain_gate",
        "record_count": len(records),
        "wall_time_s": elapsed,
        "steps_per_second": len(records) / elapsed if elapsed > 0.0 else 0.0,
        "summary_count": summary_count,
        "top_residual_edge_count": top_residual_edge_count,
        "critical_count": critical_count,
        "min_obstruction_delta": min_obstruction_delta,
        "control_energy_reduction": control_energy_reduction,
        "max_nominal_obstruction_score": max_nominal_obstruction_score,
        "non_actuating": non_actuating,
        "execution_disabled": execution_disabled,
        "operator_review_required": operator_review_required,
        "deterministic_hash": deterministic_hash,
        "sheaf_obstruction_sha256": _stable_record_hash([*records, control_record]),
        "acceptance_passed": acceptance_passed,
        "acceptance_thresholds_json": json.dumps(
            {
                "max_nominal_obstruction_score": (
                    thresholds.max_nominal_obstruction_score
                ),
                "min_critical_count": thresholds.min_critical_count,
                "min_demo_count": thresholds.min_demo_count,
                "min_control_energy_reduction": (
                    thresholds.min_control_energy_reduction
                ),
                "min_obstruction_delta": thresholds.min_obstruction_delta,
                "min_summary_count": thresholds.min_summary_count,
                "min_top_residual_edge_count": (thresholds.min_top_residual_edge_count),
                "require_deterministic_hash": thresholds.require_deterministic_hash,
                "require_execution_disabled": thresholds.require_execution_disabled,
                "require_non_actuating": thresholds.require_non_actuating,
                "require_operator_review": thresholds.require_operator_review,
            },
            sort_keys=True,
        ),
        "control_record_json": json.dumps(control_record, sort_keys=True),
        "records_json": json.dumps(records, sort_keys=True),
    }


def benchmark_plugin_ecosystem_catalog_quality() -> dict[str, float | int | str]:
    """Benchmark plugin marketplace and Rust registry capability contracts."""
    thresholds = PluginEcosystemThresholds(
        min_plugin_count=2,
        min_capability_count=5,
        min_handoff_target_hash_count=5,
        min_blocked_handoff_count=1,
        required_capability_kinds=frozenset(
            {"extractor", "monitor", "actuator", "bridge"}
        ),
        min_incompatible_count=1,
        require_deterministic_hash=True,
        require_loading_disabled=True,
    )
    manifests = _plugin_ecosystem_manifests()

    t0 = time.perf_counter()
    catalog = build_plugin_marketplace_catalog(manifests)
    full_catalog = build_plugin_marketplace_catalog(
        manifests,
        include_incompatible=True,
    )
    rust_registry = build_rust_plugin_registry(manifests)
    repeated_registry = build_rust_plugin_registry(manifests)
    rust_handoff = build_rust_plugin_runtime_handoff(
        manifests,
        include_incompatible=True,
    )
    repeated_handoff = build_rust_plugin_runtime_handoff(
        manifests,
        include_incompatible=True,
    )
    elapsed = time.perf_counter() - t0

    capabilities = rust_registry["capabilities"]
    if not isinstance(capabilities, list):
        raise TypeError("rust registry capabilities must be a list")
    dispatch_groups = rust_handoff["dispatch_groups"]
    target_hashes = rust_handoff["target_hashes"]
    blocked_capabilities = rust_handoff["blocked_capabilities"]
    if not isinstance(dispatch_groups, dict):
        raise TypeError("rust runtime handoff dispatch groups must be a mapping")
    if not isinstance(target_hashes, dict):
        raise TypeError("rust runtime handoff target hashes must be a mapping")
    if not isinstance(blocked_capabilities, list):
        raise TypeError("rust runtime handoff blocked capabilities must be a list")
    capability_kinds = {str(capability["kind"]) for capability in capabilities}
    registry_hash = sha256(
        json.dumps(rust_registry, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    repeated_hash = sha256(
        json.dumps(repeated_registry, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    handoff_hash = sha256(
        json.dumps(rust_handoff, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    repeated_handoff_hash = sha256(
        json.dumps(
            repeated_handoff,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    deterministic_hash = int(
        registry_hash == repeated_hash and handoff_hash == repeated_handoff_hash
    )
    handoff_loading_disabled = int(rust_handoff["loading_permitted"] is False)
    acceptance_passed = int(
        catalog["plugin_count"] >= thresholds.min_plugin_count
        and rust_registry["capability_count"] >= thresholds.min_capability_count
        and len(target_hashes) >= thresholds.min_handoff_target_hash_count
        and len(blocked_capabilities) >= thresholds.min_blocked_handoff_count
        and thresholds.required_capability_kinds <= capability_kinds
        and full_catalog["incompatible_count"] >= thresholds.min_incompatible_count
        and deterministic_hash == int(thresholds.require_deterministic_hash)
        and handoff_loading_disabled == int(thresholds.require_loading_disabled)
    )

    return {
        "suite": "plugin_ecosystem_catalog_quality",
        "plugin_count": catalog["plugin_count"],
        "full_plugin_count": full_catalog["plugin_count"],
        "compatible_count": catalog["compatible_count"],
        "incompatible_count": full_catalog["incompatible_count"],
        "capability_count": rust_registry["capability_count"],
        "handoff_target_hash_count": len(target_hashes),
        "handoff_blocked_count": len(blocked_capabilities),
        "handoff_loading_disabled": handoff_loading_disabled,
        "wall_time_s": elapsed,
        "steps_per_second": len(manifests) / elapsed,
        "required_kind_count": len(thresholds.required_capability_kinds),
        "observed_kind_count": len(capability_kinds),
        "deterministic_hash": deterministic_hash,
        "registry_sha256": registry_hash,
        "handoff_sha256": handoff_hash,
        "acceptance_passed": acceptance_passed,
        "capability_counts_json": json.dumps(
            rust_registry["capability_counts"],
            sort_keys=True,
        ),
        "handoff_dispatch_groups_json": json.dumps(
            {
                kind: len(records) if isinstance(records, list) else 0
                for kind, records in sorted(dispatch_groups.items())
            },
            sort_keys=True,
        ),
        "acceptance_thresholds_json": json.dumps(
            {
                "min_blocked_handoff_count": thresholds.min_blocked_handoff_count,
                "min_capability_count": thresholds.min_capability_count,
                "min_handoff_target_hash_count": (
                    thresholds.min_handoff_target_hash_count
                ),
                "min_incompatible_count": thresholds.min_incompatible_count,
                "min_plugin_count": thresholds.min_plugin_count,
                "require_deterministic_hash": (thresholds.require_deterministic_hash),
                "require_loading_disabled": thresholds.require_loading_disabled,
                "required_capability_kinds": sorted(
                    thresholds.required_capability_kinds
                ),
            },
            sort_keys=True,
        ),
    }


def _deterministic_replay_observation(
    candidate: KnobPolicyCandidate,
) -> RewardObservation:
    mean_k = float(np.asarray(candidate.K, dtype=np.float64).mean())
    mean_channel_weight = (
        float(np.mean(candidate.channel_weights)) if candidate.channel_weights else 0.0
    )
    coherence = float(
        np.clip(0.68 + 0.44 * mean_k + 0.05 * mean_channel_weight, 0.0, 0.95)
    )
    unsafe = mean_k < 0.0 or abs(mean_k) > 1.2
    lyapunov_exponent = float(-0.02 - 0.05 * max(mean_k, 0.0))
    stl_robustness = float(coherence - 0.7)
    safety_cost = float(
        0.02 * abs(mean_k)
        + 0.01 * abs(float(np.asarray(candidate.zeta, dtype=np.float64).mean()))
        + 0.005 * abs(float(np.asarray(candidate.alpha, dtype=np.float64).mean()))
    )
    return RewardObservation(
        coherence=coherence,
        previous_coherence=0.68,
        unsafe=unsafe,
        regime_changed=False,
        lyapunov_exponent=lyapunov_exponent,
        stl_robustness=stl_robustness,
        safety_cost=safety_cost,
    )


def _bayesian_posterior_fit_fixture() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, float
]:
    dt = 0.02
    omega = np.array([0.92, 1.03, 1.11], dtype=float)
    knm = np.array(
        [
            [0.0, 0.11, 0.04],
            [0.18, 0.0, 0.07],
            [0.09, 0.14, 0.0],
        ],
        dtype=float,
    )
    alpha = np.zeros_like(knm)
    engine = UPDEEngine(n_oscillators=3, dt=dt, method="rk4")
    phase = np.array([0.1, 0.6, 1.4], dtype=float)
    trajectory = [phase.copy()]
    for _ in range(95):
        phase = engine.step(phase, omega, knm, zeta=0.0, psi=0.0, alpha=alpha)
        trajectory.append(phase.copy())
    return np.asarray(trajectory), omega, knm, alpha, dt


def _bayesian_backend_audit_fixture() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, float
]:
    phases = np.array([0.0, 0.4, 1.1, 1.9], dtype=float)
    omega = np.array([0.9, 1.0, 1.08, 1.16], dtype=float)
    knm = np.full((4, 4), 0.18, dtype=float)
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros_like(knm)
    return phases, omega, knm, alpha, 0.01


def _formal_export_fixture() -> tuple[
    PetriNet,
    Marking,
    list[PolicyRule],
    list[PolicySTLSpec],
]:
    net = PetriNet(
        places=[
            Place("warmup"),
            Place("nominal"),
            Place("recovery"),
            Place("critical"),
        ],
        transitions=[
            Transition(
                name="start",
                inputs=[Arc("warmup")],
                outputs=[Arc("nominal")],
                guard=Guard("stability_proxy", ">", 0.6),
            ),
            Transition(
                name="escalate",
                inputs=[Arc("nominal")],
                outputs=[Arc("critical")],
                guard=Guard("R_bad.0", ">=", 0.4),
            ),
            Transition(
                name="recover",
                inputs=[Arc("critical")],
                outputs=[Arc("recovery")],
                guard=Guard("R_good.0", ">=", 0.7),
            ),
        ],
    )
    rules = [
        PolicyRule(
            name="boost K",
            regimes=["DEGRADED", "CRITICAL"],
            condition=PolicyCondition("R_good", 0, "<", 0.6),
            actions=[PolicyAction("K", "global", 0.1, 5.0)],
            max_fires=2,
        ),
        PolicyRule(
            name="damp bad",
            regimes=["CRITICAL"],
            condition=CompoundCondition(
                conditions=[
                    PolicyCondition("R_bad", 0, ">", 0.4),
                    PolicyCondition("stability_proxy", None, "<=", 0.5),
                ],
                logic="AND",
            ),
            actions=[PolicyAction("alpha", "layer_0", -0.05, 3.0)],
        ),
    ]
    stl_specs = [
        PolicySTLSpec(
            name="keep sync",
            spec="always (R >= 0.3 and amplitude_spread < 0.2)",
            severity="hard",
        ),
        PolicySTLSpec(
            name="recover",
            spec="eventually (R_good >= 0.8)",
        ),
    ]
    return net, Marking(tokens={"warmup": 1}), rules, stl_specs


def _formal_export_fail_closed_count() -> int:
    failures = 0
    malformed_rule = PolicyRule(
        name="bad",
        regimes=["DEGRADED"],
        condition=PolicyCondition("R", 0, "!=", 0.1),
        actions=[PolicyAction("K", "global", 0.1, 1.0)],
    )
    probes = (
        lambda: export_petri_net_prism(PetriNet([], []), Marking()),
        lambda: export_petri_net_tla(PetriNet([], []), Marking()),
        lambda: export_policy_rules_prism([malformed_rule]),
        lambda: export_policy_rules_tla([malformed_rule]),
        lambda: export_stl_specs_prism([PolicySTLSpec("bad", "always (R != 0.5)")]),
    )
    for probe in probes:
        try:
            probe()
        except PolicyError:
            failures += 1
    return failures


def _domain_formal_export_fixtures() -> tuple[DomainFormalExportFixture, ...]:
    return (
        DomainFormalExportFixture(
            domain="cardiac_rhythm",
            rules=(
                PolicyRule(
                    name="limit arrhythmia drive",
                    regimes=["DEGRADED", "CRITICAL"],
                    condition=CompoundCondition(
                        conditions=[
                            PolicyCondition("R_bad", 0, ">", 0.25),
                            PolicyCondition("phase_error", None, ">", 0.35),
                        ],
                        logic="AND",
                    ),
                    actions=(PolicyAction("zeta", "pacemaker_guard", -0.02, 1.5),),
                    max_fires=1,
                ),
                PolicyRule(
                    name="restore sinus synchrony",
                    regimes=["RECOVERY"],
                    condition=PolicyCondition("R_good", 0, "<", 0.75),
                    actions=(PolicyAction("K", "atrium_ventricle", 0.03, 2.5),),
                    max_fires=2,
                ),
            ),
            stl_specs=(
                PolicySTLSpec(
                    "bounded arrhythmia proxy",
                    "always (R_bad <= 0.55 and phase_error <= 0.75)",
                    "hard",
                ),
                PolicySTLSpec(
                    "sinus synchrony recovers",
                    "eventually (R_good >= 0.75)",
                ),
            ),
            required_labels=(
                'label "fires_limit_arrhythmia_drive"',
                'label "stl_bounded_arrhythmia_proxy_satisfied"',
            ),
        ),
        DomainFormalExportFixture(
            domain="chemical_reactor",
            rules=(
                PolicyRule(
                    name="quench thermal runaway",
                    regimes=["DEGRADED", "CRITICAL"],
                    condition=CompoundCondition(
                        conditions=[
                            PolicyCondition("temperature_c", None, ">", 420.0),
                            PolicyCondition("pressure_bar", None, ">", 13.5),
                        ],
                        logic="OR",
                    ),
                    actions=(PolicyAction("K", "coolant_loop", 0.07, 2.0),),
                    max_fires=2,
                ),
                PolicyRule(
                    name="restore process margin",
                    regimes=["RECOVERY"],
                    condition=PolicyCondition("R_good", 1, "<", 0.78),
                    actions=(PolicyAction("zeta", "feed_rate", -0.03, 2.5),),
                    max_fires=2,
                ),
            ),
            stl_specs=(
                PolicySTLSpec(
                    "bounded reactor envelope",
                    "always (temperature_c <= 450 and pressure_bar <= 15)",
                    "hard",
                ),
                PolicySTLSpec(
                    "thermal stability recovers",
                    "eventually (R_good >= 0.78)",
                ),
            ),
            required_labels=(
                'label "fires_quench_thermal_runaway"',
                'label "stl_bounded_reactor_envelope_satisfied"',
            ),
        ),
        DomainFormalExportFixture(
            domain="power_grid",
            rules=(
                PolicyRule(
                    name="shed oscillatory tie line",
                    regimes=["DEGRADED", "CRITICAL"],
                    condition=CompoundCondition(
                        conditions=[
                            PolicyCondition("phase_error", None, ">", 0.45),
                            PolicyCondition("R_bad", 0, ">=", 0.3),
                        ],
                        logic="OR",
                    ),
                    actions=(PolicyAction("K", "tie_line", -0.06, 4.0),),
                    max_fires=2,
                ),
                PolicyRule(
                    name="restore generator lock",
                    regimes=["RECOVERY"],
                    condition=PolicyCondition("R_good", 0, "<", 0.8),
                    actions=(PolicyAction("K", "generator_cluster", 0.05, 5.0),),
                    max_fires=3,
                ),
            ),
            stl_specs=(
                PolicySTLSpec(
                    "bounded grid phase error",
                    "always (phase_error <= 0.9 and R_bad <= 0.65)",
                    "hard",
                ),
                PolicySTLSpec(
                    "grid resynchronises",
                    "eventually (R_good >= 0.8)",
                ),
            ),
            required_labels=(
                'label "fires_shed_oscillatory_tie_line"',
                'label "stl_bounded_grid_phase_error_satisfied"',
            ),
        ),
        DomainFormalExportFixture(
            domain="pll_clock",
            rules=(
                PolicyRule(
                    name="limit holdover drift",
                    regimes=["DEGRADED", "CRITICAL"],
                    condition=CompoundCondition(
                        conditions=[
                            PolicyCondition("phase_error_ns", None, ">", 80.0),
                            PolicyCondition("freq_drift_ppm", None, ">", 8.0),
                        ],
                        logic="OR",
                    ),
                    actions=(PolicyAction("zeta", "reference_drive", 0.04, 3.0),),
                    max_fires=2,
                ),
                PolicyRule(
                    name="restore pll lock",
                    regimes=["RECOVERY"],
                    condition=PolicyCondition("R_good", 0, "<", 0.82),
                    actions=(PolicyAction("K", "loop_bandwidth", 0.05, 4.0),),
                    max_fires=2,
                ),
            ),
            stl_specs=(
                PolicySTLSpec(
                    "bounded clock drift",
                    "always (phase_error_ns <= 100 and freq_drift_ppm <= 10)",
                    "hard",
                ),
                PolicySTLSpec(
                    "pll synchrony recovers",
                    "eventually (R_good >= 0.82)",
                ),
            ),
            required_labels=(
                'label "fires_limit_holdover_drift"',
                'label "stl_bounded_clock_drift_satisfied"',
            ),
        ),
        DomainFormalExportFixture(
            domain="autonomous_vehicles",
            rules=(
                PolicyRule(
                    name="limit platoon unsafe gap",
                    regimes=["DEGRADED", "CRITICAL"],
                    condition=CompoundCondition(
                        conditions=[
                            PolicyCondition("gap_distance", None, "<", 0.25),
                            PolicyCondition("brake_reaction", None, ">", 0.25),
                        ],
                        logic="OR",
                    ),
                    actions=(PolicyAction("zeta", "throttle_drive", -0.03, 0.5),),
                    max_fires=1,
                ),
                PolicyRule(
                    name="restore follower synchrony",
                    regimes=["RECOVERY"],
                    condition=PolicyCondition("R_good", 1, "<", 0.76),
                    actions=(PolicyAction("K", "platoon_coupling", 0.04, 2.0),),
                    max_fires=2,
                ),
            ),
            stl_specs=(
                PolicySTLSpec(
                    "bounded vehicle gap",
                    "always (gap_distance >= 0.2 and brake_reaction <= 0.3)",
                    "hard",
                ),
                PolicySTLSpec(
                    "platoon synchrony recovers",
                    "eventually (R_good >= 0.76)",
                ),
            ),
            required_labels=(
                'label "fires_limit_platoon_unsafe_gap"',
                'label "stl_bounded_vehicle_gap_satisfied"',
            ),
        ),
        DomainFormalExportFixture(
            domain="satellite_constellation",
            rules=(
                PolicyRule(
                    name="hold link budget floor",
                    regimes=["DEGRADED", "CRITICAL"],
                    condition=PolicyCondition("link_budget", None, "<", 0.35),
                    actions=(PolicyAction("zeta", "beam_steering", 0.03, 1.0),),
                    max_fires=2,
                ),
                PolicyRule(
                    name="restore beam synchrony",
                    regimes=["RECOVERY"],
                    condition=PolicyCondition("R_good", 2, "<", 0.74),
                    actions=(PolicyAction("K", "global_coupling", 0.04, 2.0),),
                    max_fires=2,
                ),
            ),
            stl_specs=(
                PolicySTLSpec(
                    "bounded satellite link budget",
                    "always (link_budget >= 0.3)",
                    "hard",
                ),
                PolicySTLSpec(
                    "beam synchrony recovers",
                    "eventually (R_good >= 0.74)",
                ),
            ),
            required_labels=(
                'label "fires_hold_link_budget_floor"',
                'label "stl_bounded_satellite_link_budget_satisfied"',
            ),
        ),
        DomainFormalExportFixture(
            domain="power_safety_nchannel",
            rules=(
                PolicyRule(
                    name="protect dispatch lock floor",
                    regimes=["DEGRADED", "CRITICAL"],
                    condition=PolicyCondition("R_2", None, "<", 0.56),
                    actions=(PolicyAction("alpha", "substation_lag", -0.04, 1.0),),
                    max_fires=2,
                ),
                PolicyRule(
                    name="restore feeder synchrony",
                    regimes=["RECOVERY"],
                    condition=PolicyCondition("R_good", 0, "<", 0.78),
                    actions=(PolicyAction("K", "grid_coupling", 0.05, 2.0),),
                    max_fires=2,
                ),
            ),
            stl_specs=(
                PolicySTLSpec(
                    "bounded dispatch lock floor",
                    "always (R_2 >= 0.52)",
                    "hard",
                ),
                PolicySTLSpec(
                    "feeder synchrony recovers",
                    "eventually (R_good >= 0.78)",
                ),
            ),
            required_labels=(
                'label "fires_protect_dispatch_lock_floor"',
                'label "stl_bounded_dispatch_lock_floor_satisfied"',
            ),
        ),
        DomainFormalExportFixture(
            domain="traffic_flow",
            rules=(
                PolicyRule(
                    name="limit queue overflow",
                    regimes=["DEGRADED", "CRITICAL"],
                    condition=PolicyCondition("queue_vehicles", None, ">", 45.0),
                    actions=(PolicyAction("zeta", "offset", -0.03, 1.0),),
                    max_fires=2,
                ),
                PolicyRule(
                    name="restore green wave",
                    regimes=["RECOVERY"],
                    condition=PolicyCondition("R_good", 1, "<", 0.72),
                    actions=(PolicyAction("K", "cycle_length", 0.04, 2.0),),
                    max_fires=2,
                ),
            ),
            stl_specs=(
                PolicySTLSpec(
                    "bounded queue overflow",
                    "always (queue_vehicles <= 50)",
                    "hard",
                ),
                PolicySTLSpec(
                    "green wave recovers",
                    "eventually (R_good >= 0.72)",
                ),
            ),
            required_labels=(
                'label "fires_limit_queue_overflow"',
                'label "stl_bounded_queue_overflow_satisfied"',
            ),
        ),
        DomainFormalExportFixture(
            domain="swarm_robotics",
            rules=(
                PolicyRule(
                    name="limit formation collision",
                    regimes=["DEGRADED", "CRITICAL"],
                    condition=CompoundCondition(
                        conditions=[
                            PolicyCondition("formation_error_m", None, ">", 1.8),
                            PolicyCondition("min_dist_m", None, "<", 0.6),
                        ],
                        logic="OR",
                    ),
                    actions=(PolicyAction("alpha", "obstacle_avoidance", 0.04, 1.0),),
                    max_fires=2,
                ),
                PolicyRule(
                    name="restore flock heading",
                    regimes=["RECOVERY"],
                    condition=PolicyCondition("R_good", 0, "<", 0.75),
                    actions=(PolicyAction("K", "alignment_coupling", 0.04, 2.0),),
                    max_fires=2,
                ),
            ),
            stl_specs=(
                PolicySTLSpec(
                    "bounded swarm formation",
                    "always (formation_error_m <= 2 and min_dist_m >= 0.5)",
                    "hard",
                ),
                PolicySTLSpec(
                    "flock heading recovers",
                    "eventually (R_good >= 0.75)",
                ),
            ),
            required_labels=(
                'label "fires_limit_formation_collision"',
                'label "stl_bounded_swarm_formation_satisfied"',
            ),
        ),
        DomainFormalExportFixture(
            domain="manufacturing_spc",
            rules=(
                PolicyRule(
                    name="hold process envelope",
                    regimes=["DEGRADED", "CRITICAL"],
                    condition=CompoundCondition(
                        conditions=[
                            PolicyCondition("temperature", None, ">", 80.0),
                            PolicyCondition("pressure", None, "<", 2.2),
                        ],
                        logic="OR",
                    ),
                    actions=(PolicyAction("zeta", "damping_global", 0.03, 1.0),),
                    max_fires=2,
                ),
                PolicyRule(
                    name="restore line quality",
                    regimes=["RECOVERY"],
                    condition=PolicyCondition("R_good", 2, "<", 0.77),
                    actions=(PolicyAction("K", "coupling_global", 0.04, 2.0),),
                    max_fires=2,
                ),
            ),
            stl_specs=(
                PolicySTLSpec(
                    "bounded manufacturing envelope",
                    "always (temperature <= 85 and pressure >= 2)",
                    "hard",
                ),
                PolicySTLSpec(
                    "line quality recovers",
                    "eventually (R_good >= 0.77)",
                ),
            ),
            required_labels=(
                'label "fires_hold_process_envelope"',
                'label "stl_bounded_manufacturing_envelope_satisfied"',
            ),
        ),
        DomainFormalExportFixture(
            domain="robotic_cpg",
            rules=(
                PolicyRule(
                    name="limit joint envelope",
                    regimes=["DEGRADED", "CRITICAL"],
                    condition=CompoundCondition(
                        conditions=[
                            PolicyCondition("joint_angle_rad", None, ">", 1.8),
                            PolicyCondition("joint_torque_nm", None, ">", 45.0),
                        ],
                        logic="OR",
                    ),
                    actions=(PolicyAction("zeta", "stride_frequency", -0.03, 1.0),),
                    max_fires=2,
                ),
                PolicyRule(
                    name="restore gait synchrony",
                    regimes=["RECOVERY"],
                    condition=PolicyCondition("R_good", 0, "<", 0.78),
                    actions=(PolicyAction("K", "coupling_global", 0.04, 2.0),),
                    max_fires=2,
                ),
            ),
            stl_specs=(
                PolicySTLSpec(
                    "bounded robotic joint envelope",
                    "always (joint_angle_rad <= 2 and joint_torque_nm <= 50)",
                    "hard",
                ),
                PolicySTLSpec(
                    "gait synchrony recovers",
                    "eventually (R_good >= 0.78)",
                ),
            ),
            required_labels=(
                'label "fires_limit_joint_envelope"',
                'label "stl_bounded_robotic_joint_envelope_satisfied"',
            ),
        ),
    )


def _hybrid_quantum_manifest() -> dict[str, object]:
    return {
        "manifest_kind": "quantum_compiler_manifest",
        "schema_version": 1,
        "status": "co_simulation_parity_passed",
        "target_backends": ["qiskit_openqasm3", "pennylane_qasm"],
        "n_qubits": 2,
        "trotter_order": 2,
        "dt": 0.125,
        "qpu_execution_permitted": False,
        "actuation_permitted": False,
        "frequency_terms": [
            {"qubit": 0, "omega": 1.0, "rz_angle": 0.125},
            {"qubit": 1, "omega": -0.5, "rz_angle": -0.0625},
        ],
        "coupling_terms": [
            {
                "source": 0,
                "target": 1,
                "forward_coupling": 0.25,
                "reverse_coupling": 0.5,
                "symmetric_coupling": 0.375,
                "xx_angle": 0.046875,
                "yy_angle": 0.046875,
            },
        ],
        "openqasm": "OPENQASM 3.0;\nqubit[2] q;\n",
        "qasm_sha256": "a" * 64,
        "co_simulation_parity": {
            "engine": "deterministic_xy_term_reconstruction",
            "max_abs_frequency_error": 0.0,
            "max_abs_coupling_error": 0.0,
            "term_count": 3,
        },
        "operator_commands": [
            "review quantum_compiler_manifest.json",
            "run Qiskit or PennyLane simulator parity before QPU handoff",
        ],
        "manifest_sha256": "b" * 64,
    }


def _hybrid_neuromorphic_manifest() -> dict[str, object]:
    return {
        "manifest_kind": "neuromorphic_schedule_manifest",
        "schema_version": 1,
        "status": "simulator_parity_passed",
        "target_backends": ["lava", "pynn"],
        "n_layers": 2,
        "n_neurons_per_population": 32,
        "tau_rc_s": 0.02,
        "tau_ref_s": 0.002,
        "input_scale": 2.0,
        "threshold_hz": 20.0,
        "actuation_permitted": False,
        "hardware_write_permitted": False,
        "populations": [
            {"layer": 0, "R": 0.25, "psi": 0.1, "estimated_rate_hz": 5.0},
            {"layer": 1, "R": 0.75, "psi": 0.3, "estimated_rate_hz": 15.0},
        ],
        "projections": [
            {"source": 0, "target": 1, "weight": 0.4, "delay_ms": 1.0},
        ],
        "control_actions": [
            {
                "knob": "spike_rate_bias",
                "scope": "layer_1",
                "value": 15.0,
                "ttl_s": 0.125,
                "justification": "deterministic schedule parity",
            },
        ],
        "simulator_parity": {
            "engine": "numpy_lif_rate_estimate",
            "max_abs_rate_error_hz": 0.0,
            "sample_count": 2,
        },
        "operator_commands": [
            "review neuromorphic_schedule_manifest.json",
            "run Lava or PyNN simulator parity before hardware handoff",
        ],
        "schedule_sha256": "c" * 64,
    }


def _hybrid_quantum_readiness_record(*, manifest_sha256: str) -> dict[str, object]:
    record: dict[str, object] = {
        "schema": "scpn_quantum_target_readiness_v1",
        "provider": "pennylane",
        "target_backend": "pennylane_qasm",
        "manifest_sha256": manifest_sha256,
        "status": "ready_not_executed",
        "blocked_reasons": [],
        "credentials_configured": True,
        "operator_approved": True,
        "qpu_execution_permitted": False,
        "actuation_permitted": False,
        "operator_commands": [
            "review quantum_compiler_manifest.json",
            "run simulator parity outside SPO before target handoff",
            "submit QPU job only from an approved external operator workflow",
        ],
    }
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
    record["readiness_sha256"] = sha256(canonical.encode("utf-8")).hexdigest()
    return record


def _hybrid_neuromorphic_readiness_record(
    *,
    manifest_sha256: str,
) -> dict[str, object]:
    record: dict[str, object] = {
        "schema": "scpn_neuromorphic_target_readiness_v1",
        "target_backend": "pynn",
        "hardware_site": "brainscales_review_lane",
        "manifest_sha256": manifest_sha256,
        "status": "ready_not_executed",
        "blocked_reasons": [],
        "credentials_configured": True,
        "operator_approved": True,
        "external_simulator_parity_verified": True,
        "hardware_write_permitted": False,
        "actuation_permitted": False,
        "operator_commands": [
            "review neuromorphic_schedule_manifest.json",
            "run target simulator parity outside SPO before hardware handoff",
            "submit neuromorphic hardware job only from an approved operator workflow",
        ],
    }
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
    record["readiness_sha256"] = sha256(canonical.encode("utf-8")).hexdigest()
    return record


def _hybrid_cocompiler_blocked_probe_count(
    quantum_manifest: dict[str, object],
    neuromorphic_manifest: dict[str, object],
) -> int:
    blocked_count = 0
    broken_quantum = dict(quantum_manifest)
    broken_quantum["status"] = "co_simulation_parity_failed"
    quantum_block = build_hybrid_cocompiler_manifest(
        broken_quantum,
        neuromorphic_manifest,
    )
    blocked_count += int(quantum_block["status"] == "blocked")

    broken_neuromorphic = dict(neuromorphic_manifest)
    broken_neuromorphic["hardware_write_permitted"] = True
    neuromorphic_block = build_hybrid_cocompiler_manifest(
        quantum_manifest,
        broken_neuromorphic,
    )
    blocked_count += int(neuromorphic_block["status"] == "blocked")
    return blocked_count


def _plugin_ecosystem_manifests() -> tuple[PluginManifest, ...]:
    return (
        PluginManifest(
            name="grid_controls_pack",
            version="0.2.0",
            package="grid_controls_pack",
            capabilities=(
                PluginCapability(
                    kind="extractor",
                    name="pmu_phase",
                    target="grid_controls.extractors:PMUPhaseExtractor",
                    channels=("phase", "frequency"),
                ),
                PluginCapability(
                    kind="monitor",
                    name="frequency_drift",
                    target="grid_controls.monitors:FrequencyDriftMonitor",
                    channels=("frequency",),
                ),
                PluginCapability(
                    kind="actuator",
                    name="breaker_guard",
                    target="grid_controls.actuators:BreakerGuard",
                    knobs=("K", "zeta"),
                ),
            ),
        ),
        PluginManifest(
            name="field_bridge_pack",
            version="0.1.0",
            package="field_bridge_pack",
            capabilities=(
                PluginCapability(
                    kind="bridge",
                    name="audit_stream",
                    target="field_bridge.bridges:AuditStreamBridge",
                ),
                PluginCapability(
                    kind="monitor",
                    name="phase_residual",
                    target="field_bridge.monitors:PhaseResidualMonitor",
                    channels=("phase_residual",),
                ),
            ),
        ),
        PluginManifest(
            name="incomplete_monitor_pack",
            version="0.1.0",
            package="incomplete_monitor_pack",
            capabilities=(
                PluginCapability(
                    kind="monitor",
                    name="empty_monitor",
                    target="incomplete.monitors:EmptyMonitor",
                ),
            ),
        ),
    )


def _audit_record_is_finite(value: object) -> bool:
    if isinstance(value, bool | str):
        return True
    if isinstance(value, int | float):
        return bool(np.isfinite(value))
    if isinstance(value, Mapping):
        return all(_audit_record_is_finite(item) for item in value.values())
    if isinstance(value, Iterable):
        return all(_audit_record_is_finite(item) for item in value)
    return False


def _auto_binding_quality_fixtures() -> tuple[AutoBindingFixture, ...]:
    return (
        AutoBindingFixture(
            domain="phase_chain",
            csv_text=_phase_chain_csv(n_samples=128),
            sample_rate_hz=None,
            expected_edges=frozenset({("theta_source", "theta_driven")}),
            thresholds=AutoBindingAcceptanceThresholds(
                min_extractor_coverage=1.0,
                min_expected_edge_recall=1.0,
                max_validation_errors=0,
                min_sample_count=96,
                max_proposed_edge_multiplier=8.0,
            ),
        ),
        AutoBindingFixture(
            domain="industrial_sensor_chain",
            csv_text=_sensor_chain_csv(n_samples=128),
            sample_rate_hz=10.0,
            expected_edges=frozenset({("source", "driven")}),
            thresholds=AutoBindingAcceptanceThresholds(
                min_extractor_coverage=1.0,
                min_expected_edge_recall=1.0,
                max_validation_errors=0,
                min_sample_count=96,
                max_proposed_edge_multiplier=8.0,
            ),
        ),
        AutoBindingFixture(
            domain="cardiac_rhythm_surrogate",
            csv_text=_cardiac_phase_csv(n_samples=160),
            sample_rate_hz=None,
            expected_edges=frozenset(
                {("pacemaker", "atrium"), ("atrium", "ventricle")}
            ),
            thresholds=AutoBindingAcceptanceThresholds(
                min_extractor_coverage=1.0,
                min_expected_edge_recall=1.0,
                max_validation_errors=0,
                min_sample_count=128,
                max_proposed_edge_multiplier=6.0,
            ),
        ),
        AutoBindingFixture(
            domain="power_grid_surrogate",
            csv_text=_power_grid_phase_csv(n_samples=192),
            sample_rate_hz=None,
            expected_edges=frozenset({("generator", "tie_line"), ("tie_line", "load")}),
            thresholds=AutoBindingAcceptanceThresholds(
                min_extractor_coverage=1.0,
                min_expected_edge_recall=1.0,
                max_validation_errors=0,
                min_sample_count=160,
                max_proposed_edge_multiplier=8.0,
            ),
        ),
    )


def _auto_binding_fixture_passes_thresholds(
    *,
    fixture: AutoBindingFixture,
    extractor_coverage: float,
    expected_edge_recall: float,
    validation_error_count: int,
    proposed_edge_multiplier: float,
) -> bool:
    thresholds = fixture.thresholds
    return (
        extractor_coverage >= thresholds.min_extractor_coverage
        and expected_edge_recall >= thresholds.min_expected_edge_recall
        and validation_error_count <= thresholds.max_validation_errors
        and fixture.sample_count >= thresholds.min_sample_count
        and proposed_edge_multiplier <= thresholds.max_proposed_edge_multiplier
    )


def _auto_binding_threshold_summary(
    fixtures: Iterable[AutoBindingFixture],
) -> dict[str, dict[str, float | int]]:
    return {
        fixture.domain: {
            "min_extractor_coverage": fixture.thresholds.min_extractor_coverage,
            "min_expected_edge_recall": fixture.thresholds.min_expected_edge_recall,
            "max_validation_errors": fixture.thresholds.max_validation_errors,
            "min_sample_count": fixture.thresholds.min_sample_count,
            "max_proposed_edge_multiplier": (
                fixture.thresholds.max_proposed_edge_multiplier
            ),
        }
        for fixture in fixtures
    }


def _extractor_source_coverage(
    *,
    source_columns: tuple[str, ...],
    extractor_proposals: Iterable[Mapping[str, object]],
) -> int:
    covered = set()
    source_column_set = set(source_columns)
    for proposal in extractor_proposals:
        parameters = proposal.get("parameters")
        if not isinstance(parameters, Mapping):
            continue
        source_column = parameters.get("source_column")
        if isinstance(source_column, str) and source_column in source_column_set:
            covered.add(source_column)
    return len(covered)


def _proposed_source_edges(
    initial_coupling_proposal: object,
) -> frozenset[tuple[str, str]]:
    if not isinstance(initial_coupling_proposal, Mapping):
        return frozenset()
    edges = initial_coupling_proposal.get("edges")
    if not isinstance(edges, Iterable) or isinstance(edges, str | bytes):
        return frozenset()
    source_edges: set[tuple[str, str]] = set()
    for edge in edges:
        if not isinstance(edge, Mapping):
            continue
        source = edge.get("source")
        target = edge.get("target")
        strength = edge.get("strength")
        if not isinstance(source, str) or not isinstance(target, str):
            continue
        if not isinstance(strength, int | float) or isinstance(strength, bool):
            continue
        if float(strength) <= 0.0:
            continue
        source_edges.add((source, target))
    return frozenset(source_edges)


def _mapping_records(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, Iterable) or isinstance(value, str | bytes):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _string_records(value: object) -> tuple[str, ...]:
    if not isinstance(value, Iterable) or isinstance(value, str | bytes):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _phase_chain_csv(n_samples: int = 32, dt: float = 0.1) -> str:
    rows = ["time,theta_source,theta_driven,theta_independent"]
    for index in range(n_samples):
        time_s = index * dt
        source = 0.21 * index
        driven = 0.15 * index + 0.18 * np.sin(source)
        independent = 1.1 + 0.09 * index
        rows.append(f"{time_s:.12g},{source:.12g},{driven:.12g},{independent:.12g}")
    return "\n".join(rows)


def _semantic_retrieval_fixture(root: Path) -> tuple[Path, Path]:
    domainpack_root = root / "domainpacks"
    power_grid = domainpack_root / "power_grid"
    grid_notes = domainpack_root / "grid_notes"
    power_grid.mkdir(parents=True)
    grid_notes.mkdir()
    (power_grid / "binding_spec.yaml").write_text(
        "name: power_grid\n# power grid renewable demand stability controller\n",
        encoding="utf-8",
    )
    (power_grid / "README.md").write_text(
        "power grid renewable demand stability controller phase coherence",
        encoding="utf-8",
    )
    (grid_notes / "binding_spec.yaml").write_text(
        "name: grid_notes\n# grid stability controller notes\n",
        encoding="utf-8",
    )
    docs_root = root / "docs"
    docs_root.mkdir()
    (docs_root / "power_grid.md").write_text(
        "power grid renewable stability controller deployment notes",
        encoding="utf-8",
    )
    return domainpack_root, docs_root


def _meta_transfer_package_records() -> tuple[MetaPolicyRecord, ...]:
    return (
        MetaPolicyRecord(
            domain="power_grid",
            features={
                "coherence": 0.88,
                "event_rate": 0.08,
                "load_variance": 0.32,
                "phase_spread": 0.11,
                "safety_margin": 0.71,
            },
            knobs={"K": 0.42, "Psi": 0.02, "alpha": 0.01, "zeta": 0.06},
            reward=0.91,
        ),
        MetaPolicyRecord(
            domain="cardiac_rhythm",
            features={
                "coherence": 0.83,
                "event_rate": 0.05,
                "load_variance": 0.12,
                "phase_spread": 0.16,
                "safety_margin": 0.84,
            },
            knobs={"K": 0.35, "Psi": 0.01, "alpha": 0.0, "zeta": 0.08},
            reward=0.94,
        ),
        MetaPolicyRecord(
            domain="traffic_flow",
            features={
                "coherence": 0.74,
                "event_rate": 0.18,
                "load_variance": 0.44,
                "phase_spread": 0.22,
                "safety_margin": 0.62,
            },
            knobs={"K": 0.31, "Psi": 0.04, "alpha": 0.02, "zeta": 0.05},
            reward=0.86,
        ),
        MetaPolicyRecord(
            domain="manufacturing_spc",
            features={
                "coherence": 0.79,
                "event_rate": 0.11,
                "load_variance": 0.26,
                "phase_spread": 0.14,
                "safety_margin": 0.77,
            },
            knobs={"K": 0.38, "Psi": 0.03, "alpha": 0.01, "zeta": 0.07},
            reward=0.89,
        ),
    )


def _write_meta_transfer_audit_corpus(root: Path) -> None:
    records_by_path = {
        root / "grid" / "audit.jsonl": (
            {
                "domain": "power_grid",
                "metrics": {
                    "coherence": 0.89,
                    "event_rate": 0.07,
                    "load_variance": 0.33,
                    "phase_spread": 0.1,
                    "safety_margin": 0.72,
                },
                "actions": [
                    {"knob": "K", "value": 0.43},
                    {"knob": "alpha", "value": 0.01},
                    {"knob": "zeta", "value": 0.06},
                    {"knob": "Psi", "value": 0.02},
                ],
                "reward": 0.93,
            },
            {
                "domain": "power_grid",
                "features": {
                    "coherence": 0.91,
                    "event_rate": 0.06,
                    "load_variance": 0.35,
                    "phase_spread": 0.09,
                    "safety_margin": 0.74,
                },
                "knobs": {"K": 0.44, "Psi": 0.02, "alpha": 0.01, "zeta": 0.06},
                "reward": 0.94,
            },
        ),
        root / "cardiac" / "nested" / "audit.jsonl": (
            {
                "domain": "cardiac_rhythm",
                "features": {
                    "coherence": 0.84,
                    "event_rate": 0.05,
                    "load_variance": 0.13,
                    "phase_spread": 0.16,
                    "safety_margin": 0.85,
                },
                "knobs": {"K": 0.35, "Psi": 0.01, "alpha": 0.0, "zeta": 0.08},
                "reward": 0.91,
            },
        ),
        root / "traffic" / "audit.jsonl": (
            {
                "domain": "traffic_flow",
                "metrics": {
                    "coherence": 0.75,
                    "event_rate": 0.18,
                    "load_variance": 0.44,
                    "phase_spread": 0.22,
                    "safety_margin": 0.62,
                },
                "actions": [
                    {"knob": "K", "value": 0.31},
                    {"knob": "alpha", "value": 0.02},
                    {"knob": "zeta", "value": 0.05},
                    {"knob": "Psi", "value": 0.04},
                ],
                "reward": 0.84,
            },
            {
                "domain": "traffic_flow",
                "features": {
                    "coherence": 0.73,
                    "event_rate": 0.2,
                    "load_variance": 0.46,
                    "phase_spread": 0.24,
                    "safety_margin": 0.6,
                },
                "knobs": {"K": 0.3, "Psi": 0.04, "alpha": 0.02, "zeta": 0.05},
                "reward": 0.82,
            },
        ),
        root / "manufacturing" / "audit.jsonl": (
            {
                "domainpack": "manufacturing_spc",
                "features": {
                    "coherence": 0.79,
                    "event_rate": 0.11,
                    "load_variance": 0.26,
                    "phase_spread": 0.14,
                    "safety_margin": 0.77,
                },
                "knobs": {"K": 0.38, "Psi": 0.03, "alpha": 0.01, "zeta": 0.07},
                "reward": 0.88,
            },
        ),
    }
    for path, records in records_by_path.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
            encoding="utf-8",
        )


def _semantic_ranking_projection(
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            "domainpack": record.get("domainpack"),
            "rank": record.get("rank"),
            "ranking_features": record.get("ranking_features"),
            "score": record.get("score"),
            "source": record.get("source"),
        }
        for record in records
    ]


def _sensor_chain_csv(n_samples: int = 32, dt: float = 0.1) -> str:
    rows = ["time,source,driven,independent"]
    previous_source = 0.0
    for index in range(n_samples):
        time_s = index * dt
        source = np.sin(0.35 * index)
        driven = 0.72 * previous_source + 0.08 * np.cos(0.17 * index)
        independent = np.cos(0.41 * index + 0.3)
        rows.append(f"{time_s:.12g},{source:.12g},{driven:.12g},{independent:.12g}")
        previous_source = source
    return "\n".join(rows)


def _cardiac_phase_csv(n_samples: int = 160, dt: float = 0.02) -> str:
    rows = ["time,pacemaker,atrium,ventricle,artifact"]
    previous_pacemaker = 0.0
    previous_atrium = 0.0
    for index in range(n_samples):
        time_s = index * dt
        pacemaker = 0.19 * index + 0.02 * np.sin(0.07 * index)
        atrium = 0.75 * previous_pacemaker + 0.04 * np.sin(pacemaker)
        ventricle = 0.68 * previous_atrium + 0.03 * np.cos(0.11 * index)
        artifact = np.sin(0.31 * index + 1.7)
        rows.append(
            f"{time_s:.12g},{pacemaker:.12g},{atrium:.12g},"
            f"{ventricle:.12g},{artifact:.12g}"
        )
        previous_pacemaker = pacemaker
        previous_atrium = atrium
    return "\n".join(rows)


def _power_grid_phase_csv(n_samples: int = 192, dt: float = 0.05) -> str:
    rows = ["time,generator,tie_line,load,renewable"]
    previous_generator = 0.0
    previous_tie_line = 0.0
    for index in range(n_samples):
        time_s = index * dt
        generator = np.sin(0.09 * index) + 0.01 * index
        tie_line = 0.62 * previous_generator + 0.05 * np.sin(0.2 * index)
        load = 0.58 * previous_tie_line + 0.05 * np.cos(0.13 * index)
        renewable = np.sin(0.29 * index + 0.4)
        rows.append(
            f"{time_s:.12g},{generator:.12g},{tie_line:.12g},"
            f"{load:.12g},{renewable:.12g}"
        )
        previous_generator = generator
        previous_tie_line = tie_line
    return "\n".join(rows)


def _morphogenetic_demo_record(payload: Mapping[str, object]) -> dict[str, object]:
    audit = payload["audit"]
    snapshot = payload["snapshot"]
    if not isinstance(audit, Mapping) or not isinstance(snapshot, Mapping):
        raise ValueError("morphogenetic demo payload must include audit and snapshot")
    field = audit["field"]
    if not isinstance(field, Mapping):
        raise ValueError("morphogenetic demo audit must include field mapping")
    return {
        "domainpack": str(payload["domainpack"]),
        "scenario": str(payload["scenario"]),
        "actuating": bool(payload["actuating"]),
        "global_coherence": float(audit["global_coherence"]),
        "delta_norm": float(audit["delta_norm"]),
        "field_layers": int(field["shape"][0]),
        "grown_edge_count": len(audit["grown_edges"]),
        "shrunk_edge_count": len(audit["shrunk_edges"]),
        "snapshot_heatmap_rows": len(snapshot["heatmap_rows"]),
        "snapshot_top_edge_count": len(snapshot["top_edges"]),
    }


def _morphogenetic_demo_runners() -> tuple[object, ...]:
    repo_root = ROOT.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    modules = (
        "domainpacks.chemical_reactor.morphogenetic_field_demo",
        "domainpacks.manufacturing_spc.morphogenetic_field_demo",
        "domainpacks.robotic_cpg.morphogenetic_field_demo",
    )
    return tuple(importlib.import_module(module).run_demo for module in modules)


def _integrated_information_replay_record(
    payload: Mapping[str, object],
) -> dict[str, object]:
    required = {
        "case_name",
        "claim_boundary",
        "description",
        "domain",
        "expected_relationship",
        "minimum_partition",
        "n_bins",
        "n_oscillators",
        "n_samples",
        "non_actuating",
        "normalised_phi",
        "phi",
        "total_integration",
    }
    missing = required - set(payload.keys())
    if missing:
        raise ValueError(f"integrated-information replay missing {sorted(missing)}")
    minimum_partition = payload["minimum_partition"]
    if (
        not isinstance(minimum_partition, list)
        or len(minimum_partition) != 2
        or any(not isinstance(part, list) for part in minimum_partition)
    ):
        raise ValueError("integrated-information replay partition must be list pair")
    return {
        "case_name": str(payload["case_name"]),
        "claim_boundary": str(payload["claim_boundary"]),
        "domain": str(payload["domain"]),
        "expected_relationship": str(payload["expected_relationship"]),
        "minimum_partition": minimum_partition,
        "n_bins": int(payload["n_bins"]),
        "n_oscillators": int(payload["n_oscillators"]),
        "n_samples": int(payload["n_samples"]),
        "non_actuating": bool(payload["non_actuating"]),
        "normalised_phi": float(payload["normalised_phi"]),
        "phi": float(payload["phi"]),
        "total_integration": float(payload["total_integration"]),
    }


def _integrated_information_replay_corpus_builders() -> tuple[object, ...]:
    from scpn_phase_orchestrator.monitor.information_replay_cyber_industrial import (
        build_cyber_industrial_integrated_information_replays,
    )
    from scpn_phase_orchestrator.monitor.information_replay_infrastructure import (
        build_infrastructure_integrated_information_replays,
    )
    from scpn_phase_orchestrator.monitor.information_replay_physiology import (
        build_physiology_integrated_information_replays,
    )

    return (
        build_physiology_integrated_information_replays,
        build_infrastructure_integrated_information_replays,
        build_cyber_industrial_integrated_information_replays,
    )


def _topos_semantic_validation_reports() -> tuple[object, ...]:
    from scpn_phase_orchestrator.binding.topos_semantic import (
        validate_symbolic_binding_functor,
    )

    prompts = (
        ("A 2 layer power grid semantic control prompt", "topos_power_grid"),
        ("A 2 layer cardiac rhythm semantic control prompt", "topos_cardiac"),
    )
    return tuple(
        validate_symbolic_binding_functor(
            compile_symbolic_binding(
                prompt,
                name=name,
                oscillators_per_layer=2,
                dry_run_steps=1,
                retrieval_root=None,
                docs_root=None,
            )
        )
        for prompt, name in prompts
    )


def _topos_policy_validation_report() -> object:
    from scpn_phase_orchestrator.supervisor.topos_policy import (
        validate_policy_composition_category,
    )

    rules = (
        PolicyRule(
            name="topos_guard_low_coherence",
            regimes=["NOMINAL", "CRITICAL"],
            condition=PolicyCondition(metric="R", layer=0, op="<", threshold=0.4),
            actions=[PolicyAction(knob="K", scope="layer_0", value=0.05, ttl_s=5.0)],
        ),
        PolicyRule(
            name="topos_guard_stability",
            regimes=["NOMINAL"],
            condition=CompoundCondition(
                logic="AND",
                conditions=[
                    PolicyCondition(
                        metric="stability_proxy",
                        layer=None,
                        op="<",
                        threshold=0.7,
                    ),
                    PolicyCondition(metric="R", layer=1, op=">", threshold=0.2),
                ],
            ),
            actions=[PolicyAction(knob="zeta", scope="global", value=0.1, ttl_s=10.0)],
        ),
    )
    return validate_policy_composition_category(rules)


def _topos_domain_examples() -> tuple[Mapping[str, object], ...]:
    from scpn_phase_orchestrator.binding.topos_examples import (
        build_topos_domain_obligation_examples,
    )

    return build_topos_domain_obligation_examples()


def _topos_validation_report_record(
    payload: Mapping[str, object],
) -> dict[str, object]:
    obligations = payload["obligation_records"]
    if not isinstance(obligations, list):
        raise ValueError("topos validation report obligations must be a list")
    obligation_names = [str(item["name"]) for item in obligations]
    return {
        "kind": str(payload["schema_name"]),
        "object_count": int(payload["object_count"]),
        "morphism_count": int(payload["morphism_count"]),
        "obligation_names": obligation_names,
        "passed": bool(payload["passed"]),
        "non_actuating": bool(payload["non_actuating"]),
        "proof_boundary": str(payload["proof_boundary"]),
        "report_hash": str(payload["report_hash"]),
    }


def _topos_domain_example_record(
    payload: Mapping[str, object],
) -> dict[str, object]:
    obligation_names = payload["obligation_names"]
    if not isinstance(obligation_names, list):
        raise ValueError("topos domain example obligation_names must be a list")
    return {
        "kind": "domain_example",
        "domain": str(payload["domain"]),
        "object_count": int(payload["binding_object_count"])
        + int(payload["policy_object_count"]),
        "morphism_count": len(obligation_names),
        "obligation_names": [str(name) for name in obligation_names],
        "passed": bool(payload["passed"]),
        "non_actuating": bool(payload["non_actuating"]),
        "proof_boundary": str(payload["proof_boundary"]),
        "report_hash": str(payload["example_hash"]),
    }


def _topos_semantic_binding_records() -> list[dict[str, object]]:
    semantic_records = [
        _topos_validation_report_record(report.to_audit_record())
        for report in _topos_semantic_validation_reports()
    ]
    policy_record = _topos_validation_report_record(
        _topos_policy_validation_report().to_audit_record()
    )
    domain_records = [
        _topos_domain_example_record(item) for item in _topos_domain_examples()
    ]
    return [*semantic_records, policy_record, *domain_records]


def _stable_record_hash(records: object) -> str:
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def _federated_transport_fixture_records() -> tuple[dict[str, object], ...]:
    records = (
        {
            "node_id": "site-a",
            "policy_delta": {"K": 0.10, "alpha": -0.02},
            "sample_count": 120,
            "local_loss": 0.21,
            "previous_audit_hash": "a" * 64,
            "privacy_epsilon_spent": 0.5,
            "clipped_l2_norm": 0.25,
            "clip_scale": 1.0,
            "accepted": True,
            "rejection_reasons": [],
        },
        {
            "node_id": "site-b",
            "policy_delta": {"K": 0.04, "alpha": -0.01},
            "sample_count": 80,
            "local_loss": 0.24,
            "previous_audit_hash": "b" * 64,
            "privacy_epsilon_spent": 0.6,
            "clipped_l2_norm": 0.18,
            "clip_scale": 1.0,
            "accepted": True,
            "rejection_reasons": [],
        },
        {
            "node_id": "site-c",
            "policy_delta": {"K": 0.08, "alpha": -0.03},
            "sample_count": 100,
            "local_loss": 0.19,
            "previous_audit_hash": "c" * 64,
            "privacy_epsilon_spent": 0.4,
            "clipped_l2_norm": 0.22,
            "clip_scale": 1.0,
            "accepted": True,
            "rejection_reasons": [],
        },
    )
    return tuple(
        {
            **record,
            "update_hash": _stable_record_hash(
                {
                    "accepted": record["accepted"],
                    "clip_scale": record["clip_scale"],
                    "clipped_l2_norm": record["clipped_l2_norm"],
                    "local_loss": record["local_loss"],
                    "node_id": record["node_id"],
                    "policy_delta": [
                        [key, value]
                        for key, value in sorted(dict(record["policy_delta"]).items())
                    ],
                    "previous_audit_hash": record["previous_audit_hash"],
                    "privacy_epsilon_spent": record["privacy_epsilon_spent"],
                    "rejection_reasons": record["rejection_reasons"],
                    "sample_count": record["sample_count"],
                }
            ),
        }
        for record in records
    )


def _federated_secure_commitment_fixture_records() -> tuple[dict[str, object], ...]:
    records = (
        ("site-a", {"K": 0.10, "alpha": -0.02}, 120),
        ("site-b", {"K": 0.04, "alpha": -0.01}, 80),
        ("site-c", {"K": 0.08, "alpha": -0.03}, 100),
    )
    return tuple(
        {
            "node_id": node_id,
            "masked_policy_delta": dict(sorted(delta.items())),
            "sample_count": sample_count,
            "share_commitment": f"commit-{node_id}",
            "share_commitment_hash": _stable_record_hash(
                {"node_id": node_id, "share_commitment": f"commit-{node_id}"}
            ),
            "share_hash": _stable_record_hash(
                {
                    "node_id": node_id,
                    "masked_policy_delta": [
                        [key, float(value)] for key, value in sorted(delta.items())
                    ],
                }
            ),
        }
        for node_id, delta, sample_count in records
    )


def _federated_secure_quorum_fixture_records() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "node_id": node_id,
            "evidence_hash": _stable_record_hash(
                {"node_id": node_id, "evidence": "federated_quorum"}
            ),
        }
        for node_id in ("site-a", "site-b", "site-c")
    )


def _federated_secure_custody_fixture_records(
    rotation_policy: str,
) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    for node_id in ("site-a", "site-b", "site-c"):
        previous_key = _stable_record_hash(
            {"node_id": node_id, "kind": "key", "label": "previous"}
        )
        previous_share = _stable_record_hash(
            {"node_id": node_id, "kind": "share", "label": "previous"}
        )
        key_label = _stable_record_hash(
            {"node_id": node_id, "kind": "key", "label": "current"}
        )
        share_label = _stable_record_hash(
            {"node_id": node_id, "kind": "share", "label": "current"}
        )
        records.append(
            {
                "node_id": node_id,
                "key_custody_label": key_label,
                "share_custody_label": share_label,
                "previous_key_custody_label": previous_key,
                "previous_share_custody_label": previous_share,
                "key_custody_continuity_hash": _stable_record_hash(
                    {
                        "node_id": node_id,
                        "rotation_policy": rotation_policy,
                        "previous_key_custody_label": previous_key,
                        "key_custody_label": key_label,
                    }
                ),
                "share_custody_continuity_hash": _stable_record_hash(
                    {
                        "node_id": node_id,
                        "rotation_policy": rotation_policy,
                        "previous_share_custody_label": previous_share,
                        "share_custody_label": share_label,
                    }
                ),
            }
        )
    return tuple(records)


def _build_evolutionary_mutation_grammar_records() -> list[dict[str, object]]:
    from scpn_phase_orchestrator.supervisor.evolutionary_petri_grammar import (
        run_offline_evolutionary_petri_mutation_grammar,
    )
    from scpn_phase_orchestrator.supervisor.evolutionary_policy_dsl import (
        run_offline_evolutionary_policy_dsl_search,
    )
    from scpn_phase_orchestrator.supervisor.evolutionary_topology_grammar import (
        run_offline_evolutionary_topology_mutation_search,
    )

    policy_report = run_offline_evolutionary_policy_dsl_search(
        (
            "rule throttle_guard: if R < 0.95 and K > 0.10 then set K += 0.04\n"
            "rule recovery_guard: if R >= 0.20 then set K -= 0.03\n"
            "rule safety_guard: if R < 0.40 then set K = 0.12"
        ),
        generation_count=1,
        population_size=6,
        mutation_step=0.02,
    ).to_audit_record()
    petri_report = run_offline_evolutionary_petri_mutation_grammar(
        {
            "places": [
                {"name": "idle", "token_bound": 2},
                {"name": "nominal", "token_bound": 4},
                {"name": "degraded", "token_bound": 3},
            ],
            "transitions": [
                {"name": "to_nominal", "guard_weights": {"R": 0.8}},
                {"name": "to_degraded", "guard_weights": {"R": 0.2}},
            ],
            "arcs": [
                {
                    "place": "idle",
                    "transition": "to_nominal",
                    "direction": "input",
                    "weight": 1,
                }
            ],
        },
        generation_count=1,
        candidates_per_generation=6,
        mutation_step=0.12,
        max_arc_weight=4,
        max_token_bound=32,
    ).to_audit_record()
    topology_report = run_offline_evolutionary_topology_mutation_search(
        [
            {"node_id": 0, "community": "alpha"},
            {"node_id": 1, "community": "alpha"},
            {"node_id": 2, "community": "beta"},
            {"node_id": 3, "community": "beta"},
        ],
        [
            {"nodes": [0, 1], "weight": 0.28},
            {"nodes": [2, 3], "weight": 0.35},
        ],
        generation_count=1,
        population_size=8,
        mutation_step=0.03,
        max_add_candidates=1,
    ).to_audit_record()
    return [
        _evolutionary_mutation_grammar_record(
            "policy_dsl",
            policy_report,
            candidate_kind_key=("mutation_plan", "component"),
            hash_key="report_hash",
        ),
        _evolutionary_mutation_grammar_record(
            "petri_net",
            petri_report,
            candidate_kind_key=("mutation_type",),
            hash_key="plan_hash",
        ),
        _evolutionary_mutation_grammar_record(
            "topology",
            topology_report,
            candidate_kind_key=("plan", "operation"),
            hash_key="report_hash",
        ),
    ]


def _evolutionary_mutation_grammar_record(
    grammar: str,
    report: Mapping[str, object],
    *,
    candidate_kind_key: tuple[str, ...],
    hash_key: str,
) -> dict[str, object]:
    candidates = report.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"{grammar} grammar report must include candidates")
    mutation_kinds = sorted(
        {
            _nested_str(candidate, candidate_kind_key)
            for candidate in candidates
            if isinstance(candidate, Mapping)
        }
    )
    candidate_hashes = [
        str(candidate.get("candidate_hash", ""))
        for candidate in candidates
        if isinstance(candidate, Mapping)
    ]
    if not all(len(candidate_hash) == 64 for candidate_hash in candidate_hashes):
        raise ValueError(f"{grammar} grammar candidate hashes must be SHA-256 values")
    report_hash = str(report.get(hash_key, ""))
    if len(report_hash) != 64:
        raise ValueError(f"{grammar} grammar report hash must be a SHA-256 value")

    return {
        "grammar": grammar,
        "schema_name": str(report["schema_name"]),
        "schema_version": str(report["schema_version"]),
        "candidate_count": int(report["candidate_count"]),
        "accepted_count": int(report["accepted_count"]),
        "rejected_count": int(report["rejected_count"]),
        "mutation_kinds": mutation_kinds,
        "candidate_hash_count": len(candidate_hashes),
        "report_hash": report_hash,
        "operator_review_required": bool(report["operator_review_required"]),
        "execution_disabled": bool(report["execution_disabled"]),
        "live_merge_permitted": bool(report["live_merge_permitted"]),
        "hot_patch_permitted": bool(report["hot_patch_permitted"]),
        "actuation_permitted": bool(report["actuation_permitted"]),
        "non_actuating": bool(report.get("non_actuating", True)),
    }


def _nested_str(payload: Mapping[str, object], path: tuple[str, ...]) -> str:
    value: object = payload
    for key in path:
        if not isinstance(value, Mapping):
            raise ValueError(f"nested path {path!r} does not resolve to a mapping")
        value = value[key]
    return str(value)


def _load_sheaf_obstruction_demo(module_path: str) -> Mapping[str, object]:
    module = importlib.import_module(module_path)
    payload = module.run_demo()
    if not isinstance(payload, Mapping):
        raise TypeError(f"{module_path}.run_demo() must return a mapping")
    return payload


def _sheaf_obstruction_demo_record(payload: Mapping[str, object]) -> dict[str, object]:
    incident_key = _sheaf_incident_key(payload)
    nominal = _mapping_at(payload, "nominal")
    incident = _mapping_at(payload, incident_key)
    summary_key = f"{incident_key}_summary"
    summary = payload.get(summary_key)
    if not isinstance(summary, Mapping):
        raise ValueError(f"{summary_key} must be present for sheaf obstruction triage")
    top_residual_edges = summary.get("top_residual_edges")
    if not isinstance(top_residual_edges, list):
        raise ValueError(f"{summary_key}.top_residual_edges must be a list")

    nominal_score = float(nominal["obstruction_score"])
    incident_score = float(incident["obstruction_score"])
    obstruction_delta = float(payload["obstruction_delta"])
    if obstruction_delta <= 0.0 or incident_score <= nominal_score:
        raise ValueError("sheaf incident obstruction must exceed nominal obstruction")

    return {
        "domainpack": str(payload["domainpack"]),
        "scenario": str(payload["scenario"]),
        "node_count": len(_sequence_at(payload, "nodes")),
        "channel_count": len(_sequence_at(payload, "channels")),
        "incident_key": incident_key,
        "nominal_obstruction_score": nominal_score,
        "incident_obstruction_score": incident_score,
        "obstruction_delta": obstruction_delta,
        "incident_severity": str(summary["severity"]),
        "top_residual_edge_count": len(top_residual_edges),
        "nominal_edge_count": int(nominal["edge_count"]),
        "incident_edge_count": int(incident["edge_count"]),
        "nominal_kernel_dimension": int(nominal["kernel_dimension"]),
        "incident_kernel_dimension": int(incident["kernel_dimension"]),
        "summary_present": True,
        "actuating": bool(payload["actuating"]),
    }


def _sheaf_obstruction_control_record() -> dict[str, object]:
    from domainpacks.power_grid.sheaf_obstruction_demo import (
        line_fault_power_grid_sheaf_state,
        power_grid_restriction_maps,
    )
    from scpn_phase_orchestrator.supervisor import (
        propose_sheaf_obstruction_control,
    )

    proposal = propose_sheaf_obstruction_control(
        line_fault_power_grid_sheaf_state(),
        power_grid_restriction_maps(),
        step_size=0.25,
        max_update_norm=0.4,
    )
    energy_reduction = (
        proposal.baseline_consistency_energy - proposal.projected_consistency_energy
    )
    return {
        "scenario": "power_grid_line_fault_sheaf_control_review",
        "accepted_for_review": proposal.accepted_for_review,
        "baseline_obstruction_score": proposal.baseline_obstruction_score,
        "projected_obstruction_score": proposal.projected_obstruction_score,
        "baseline_consistency_energy": proposal.baseline_consistency_energy,
        "projected_consistency_energy": proposal.projected_consistency_energy,
        "control_energy_reduction": energy_reduction,
        "baseline_kernel_dimension": proposal.baseline_kernel_dimension,
        "projected_kernel_dimension": proposal.projected_kernel_dimension,
        "baseline_obstruction_dimension": proposal.baseline_obstruction_dimension,
        "projected_obstruction_dimension": proposal.projected_obstruction_dimension,
        "update_norm": proposal.update_norm,
        "non_actuating": proposal.non_actuating,
        "execution_disabled": proposal.execution_disabled,
        "operator_review_required": proposal.operator_review_required,
        "blocked_reason_count": len(proposal.blocked_reasons),
    }


def _sheaf_incident_key(payload: Mapping[str, object]) -> str:
    reserved = {
        "actuating",
        "channels",
        "domainpack",
        "nodes",
        "nominal",
        "nominal_summary",
        "obstruction_delta",
        "scenario",
    }
    incident_keys = [
        key
        for key, value in payload.items()
        if key not in reserved
        and not key.endswith("_summary")
        and isinstance(value, Mapping)
        and "obstruction_score" in value
    ]
    if len(incident_keys) != 1:
        raise ValueError(
            f"expected exactly one sheaf incident record, got {incident_keys!r}"
        )
    return incident_keys[0]


def _mapping_at(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _sequence_at(payload: Mapping[str, object], key: str) -> tuple[object, ...]:
    value = payload.get(key)
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        raise ValueError(f"{key} must be a non-string iterable")
    return tuple(value)


def run_reference_suite(*, snapshot_date: str | None = None) -> ReferenceSuiteResult:
    return {
        "metadata": build_benchmark_metadata(snapshot_date=snapshot_date),
        "benchmarks": {
            "auto_binding": benchmark_auto_binding_proposal_quality(),
            "semantic_retrieval": benchmark_semantic_retrieval_ranking_quality(),
            "replay_policy": benchmark_replay_policy_candidate_quality(),
            "bayesian_posterior": benchmark_bayesian_posterior_fit_quality(),
            "bayesian_backends": benchmark_bayesian_backend_fail_closed(),
            "formal_export": benchmark_formal_export_artifact_quality(),
            "stl_closed_loop": benchmark_stl_closed_loop_plan_quality(),
            "domain_formal_export": benchmark_domain_formal_safety_exports(),
            "hybrid_cocompiler": benchmark_hybrid_cocompiler_review_gate(),
            "quantum_target_readiness": benchmark_quantum_target_readiness_gate(),
            "neuromorphic_target_readiness": (
                benchmark_neuromorphic_target_readiness_gate()
            ),
            "hybrid_target_readiness": benchmark_hybrid_target_readiness_gate(),
            "hybrid_operator_handoff": (
                benchmark_hybrid_operator_handoff_package_gate()
            ),
            "value_alignment_replay_calibration": (
                benchmark_value_alignment_replay_calibration_gate()
            ),
            "autopoietic_lineage": benchmark_autopoietic_lineage_sandbox_gate(),
            "intergenerational_inheritance": (
                benchmark_intergenerational_policy_inheritance_gate()
            ),
            "temporal_causal_hypergraph": (
                benchmark_temporal_causal_hypergraph_experiment_gate()
            ),
            "morphogenetic_domain_demos": benchmark_morphogenetic_domain_demo_gate(),
            "integrated_information_replay_corpus": (
                benchmark_integrated_information_replay_corpus_gate()
            ),
            "evolutionary_supervisor_search": (
                benchmark_evolutionary_supervisor_search()
            ),
            "evolutionary_mutation_grammars": (
                benchmark_evolutionary_mutation_grammar_gate()
            ),
            "federated_meta_orchestrator": benchmark_federated_meta_orchestrator(),
            "federated_production_boundary": (
                benchmark_federated_production_boundary_gate()
            ),
            "federated_deployment_preflight": (
                benchmark_federated_deployment_preflight_gate()
            ),
            "topos_semantic_binding": benchmark_topos_semantic_binding_gate(),
            "multiverse_counterfactual": benchmark_multiverse_counterfactual_gate(),
            "hybrid_entanglement_order": (
                benchmark_hybrid_entanglement_order_parameter_gate()
            ),
            "self_model_digital_twin": benchmark_self_model_digital_twin(),
            "strange_loop_drift_scenarios": (
                benchmark_strange_loop_drift_scenario_gate()
            ),
            "information_geometry_control": (
                benchmark_information_geometry_control_gate()
            ),
            "sheaf_obstruction_domains": benchmark_sheaf_obstruction_domain_gate(),
            "meta_transfer_corpus": benchmark_meta_transfer_audit_corpus_quality(),
            "meta_transfer": benchmark_meta_transfer_package_manifest_quality(),
            "plugin_ecosystem": benchmark_plugin_ecosystem_catalog_quality(),
            "chimera_polyglot": benchmark_chimera_polyglot_parity_gate(),
            "dimension_polyglot": benchmark_dimension_polyglot_parity_gate(),
            "embedding_polyglot": benchmark_embedding_polyglot_parity_gate(),
            "entropy_production_polyglot": (
                benchmark_entropy_production_polyglot_parity_gate()
            ),
            "hodge_polyglot": benchmark_hodge_polyglot_parity_gate(),
            "itpc_polyglot": benchmark_itpc_polyglot_parity_gate(),
            "lyapunov_polyglot": benchmark_lyapunov_polyglot_parity_gate(),
            "npe_polyglot": benchmark_npe_polyglot_parity_gate(),
            "order_parameter_polyglot": (
                benchmark_order_parameter_polyglot_parity_gate()
            ),
            "recurrence_polyglot": benchmark_recurrence_polyglot_parity_gate(),
            "spectral_polyglot": benchmark_spectral_polyglot_parity_gate(),
            "spatial_modulator_polyglot": (
                benchmark_spatial_modulator_polyglot_parity_gate()
            ),
            "upde_doppler_polyglot": benchmark_upde_doppler_polyglot_gate(),
            "upde_moving_frame_polyglot": benchmark_upde_moving_frame_polyglot_gate(),
            "upde_time_varying_omega_polyglot": (
                benchmark_upde_time_varying_omega_polyglot_gate()
            ),
            "pha_c_handoff_polyglot": (
                benchmark_pha_c_handoff_polyglot_parity_gate()
            ),
            "pha_c_timeline_polyglot": (
                benchmark_pha_c_timeline_polyglot_parity_gate()
            ),
            "pha_c_acceptance_polyglot": benchmark_pha_c_acceptance_polyglot_gate(),
            "transfer_entropy_polyglot": (
                benchmark_transfer_entropy_polyglot_parity_gate()
            ),
            "winding_polyglot": benchmark_winding_polyglot_parity_gate(),
            "kuramoto": benchmark_kuramoto_reference(),
            "stuart_landau": benchmark_stuart_landau_reference(),
            "petri_reachability": benchmark_petri_reachability(),
        },
    }


if __name__ == "__main__":
    results = run_reference_suite()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
