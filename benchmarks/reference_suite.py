# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — v1 reference benchmark suite

from __future__ import annotations

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
from scpn_phase_orchestrator.supervisor.formal_export import (
    FormalSafetyProperty,
    audit_formal_checker_availability,
    build_formal_verification_package,
    export_petri_net_prism,
    export_petri_net_tla,
    export_policy_rules_prism,
    export_policy_rules_tla,
    export_stl_specs_prism,
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

ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "results" / "reference_suite.json"
BENCHMARK_COMMAND = "PYTHONPATH=src python benchmarks/reference_suite.py"
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
    require_non_actuating: bool


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
    require_deterministic_hash: bool
    require_checker_execution_disabled: bool


class DomainFormalExportThresholds(NamedTuple):
    min_domain_count: int
    min_artifacts_per_domain: int
    min_rules_per_domain: int
    min_stl_specs_per_domain: int
    require_deterministic_hash: bool


class DomainFormalExportFixture(NamedTuple):
    domain: str
    rules: tuple[PolicyRule, ...]
    stl_specs: tuple[PolicySTLSpec, ...]
    required_labels: tuple[str, ...]


class STLClosedLoopThresholds(NamedTuple):
    min_plan_count: int
    min_projected_action_count: int
    min_blocked_reason_count: int
    require_non_actuating: bool
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


class ValueAlignmentReplayCalibrationThresholds(NamedTuple):
    min_replay_case_count: int
    min_approved_case_count: int
    min_blocked_case_count: int
    min_threshold_fallback_case_count: int
    min_fallback_applied_case_count: int
    require_review_only: bool
    require_deterministic_hash: bool


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


def benchmark_kuramoto_reference(
    n_oscillators: int = 64, n_steps: int = 1000, dt: float = 0.01
) -> dict[str, float | int | str]:
    rng = np.random.default_rng(42)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_oscillators)
    omegas = np.zeros(n_oscillators)
    knm = np.full((n_oscillators, n_oscillators), 0.4, dtype=float)
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros_like(knm)
    engine = UPDEEngine(n_oscillators=n_oscillators, dt=dt, method="rk4")

    t0 = time.perf_counter()
    for _ in range(n_steps):
        phases = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)
    elapsed = time.perf_counter() - t0
    final_r, _ = compute_order_parameter(phases)

    return {
        "suite": "kuramoto_reference_strogatz_2000",
        "n_oscillators": n_oscillators,
        "n_steps": n_steps,
        "wall_time_s": elapsed,
        "steps_per_second": n_steps / elapsed,
        "final_order_parameter": float(final_r),
    }


def benchmark_stuart_landau_reference(
    n_oscillators: int = 64, n_steps: int = 1000, dt: float = 0.01
) -> dict[str, float | int | str]:
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

    return {
        "suite": "stuart_landau_reference_pikovsky_2001",
        "n_oscillators": n_oscillators,
        "n_steps": n_steps,
        "wall_time_s": elapsed,
        "steps_per_second": n_steps / elapsed,
        "final_mean_amplitude": final_r,
    }


def benchmark_petri_reachability(n_steps: int = 5000) -> dict[str, float | int | str]:
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

    t0 = time.perf_counter()
    for _ in range(n_steps):
        key = tuple(sorted(marking.tokens.items()))
        visited.add(key)
        marking, _ = net.step(marking, {})
    elapsed = time.perf_counter() - t0

    return {
        "suite": "petri_net_reachability",
        "n_steps": n_steps,
        "wall_time_s": elapsed,
        "steps_per_second": n_steps / elapsed,
        "reachable_markings": len(visited),
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
        require_non_actuating=True,
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
                    "candidate_count": len(proposal.policy_search.candidates),
                }
            )

        scenario_accepted = (
            scenario_accepted_count == len(learner_proposals)
            and scenario_unsafe_acceptances == 0
            and scenario_non_actuating
            and scenario_min_reward_improvement >= thresholds.min_reward_improvement
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
        "non_actuating_proposals": int(
            all(result["non_actuating"] is True for result in learner_results)
        ),
        "acceptance_passed": int(threshold_passed),
        "acceptance_thresholds_json": json.dumps(
            {
                "min_acceptance_rate": thresholds.min_acceptance_rate,
                "min_reward_improvement": thresholds.min_reward_improvement,
                "max_unsafe_acceptances": thresholds.max_unsafe_acceptances,
                "require_non_actuating": thresholds.require_non_actuating,
            },
            sort_keys=True,
        ),
        "scenario_results_json": json.dumps(scenario_results, sort_keys=True),
        "learner_results_json": json.dumps(learner_results, sort_keys=True),
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
        require_deterministic_hash=True,
        require_checker_execution_disabled=True,
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
        "deterministic_hash": deterministic_hash,
        "artifact_sha256": artifact_hash,
        "package_sha256": package.package_hash,
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
                "require_checker_execution_disabled": (
                    thresholds.require_checker_execution_disabled
                ),
                "require_deterministic_hash": (thresholds.require_deterministic_hash),
            },
            sort_keys=True,
        ),
        "checker_commands_json": json.dumps(checker_commands, sort_keys=True),
        "checker_availability_json": json.dumps(
            checker_availability_records,
            sort_keys=True,
        ),
    }


def benchmark_stl_closed_loop_plan_quality() -> dict[str, float | int | str]:
    """Benchmark offline STL closed-loop plan synthesis and fail-closed gates."""
    thresholds = STLClosedLoopThresholds(
        min_plan_count=3,
        min_projected_action_count=1,
        min_blocked_reason_count=3,
        require_non_actuating=True,
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
    non_actuating = int(
        all(
            not plan.actuating
            and not plan.synthesis.actuating
            and not plan.projected_plan.actuating
            for plan in plans
        )
    )
    acceptance_passed = int(
        len(plans) >= thresholds.min_plan_count
        and projected_action_count >= thresholds.min_projected_action_count
        and blocked_reason_count >= thresholds.min_blocked_reason_count
        and non_actuating == int(thresholds.require_non_actuating)
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
        "non_actuating": non_actuating,
        "deterministic_hash": deterministic_hash,
        "plan_sha256": sha256(plans_json.encode()).hexdigest(),
        "acceptance_passed": acceptance_passed,
        "acceptance_thresholds_json": json.dumps(
            {
                "min_blocked_reason_count": thresholds.min_blocked_reason_count,
                "min_plan_count": thresholds.min_plan_count,
                "min_projected_action_count": (
                    thresholds.min_projected_action_count
                ),
                "require_deterministic_hash": thresholds.require_deterministic_hash,
                "require_non_actuating": thresholds.require_non_actuating,
            },
            sort_keys=True,
        ),
        "plans_json": plans_json,
    }


def benchmark_domain_formal_safety_exports() -> dict[str, float | int | str]:
    """Benchmark domain-style policy/STL formal safety artefacts."""
    thresholds = DomainFormalExportThresholds(
        min_domain_count=3,
        min_artifacts_per_domain=3,
        min_rules_per_domain=2,
        min_stl_specs_per_domain=2,
        require_deterministic_hash=True,
    )
    fixtures = _domain_formal_export_fixtures()
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
        texts = (policy_prism.model, policy_tla.module, stl_prism.model)
        deterministic_hash = int(
            sha256(policy_prism.model.encode()).hexdigest()
            == sha256(repeated.model.encode()).hexdigest()
        )
        required_labels_present = all(
            label in "\n".join(texts) for label in fixture.required_labels
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
        accepted = (
            artifact_count >= thresholds.min_artifacts_per_domain
            and len(rules) >= thresholds.min_rules_per_domain
            and len(stl_specs) >= thresholds.min_stl_specs_per_domain
            and deterministic_hash == int(thresholds.require_deterministic_hash)
            and required_labels_present
        )
        domain_results.append(
            {
                "domain": fixture.domain,
                "artifact_count": artifact_count,
                "rule_count": len(rules),
                "stl_spec_count": len(stl_specs),
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
                "min_rules_per_domain": thresholds.min_rules_per_domain,
                "min_stl_specs_per_domain": thresholds.min_stl_specs_per_domain,
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
        and ready["quantum_readiness_sha256"]
        == quantum_readiness["readiness_sha256"]
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
    return RewardObservation(
        coherence=coherence,
        previous_coherence=0.68,
        unsafe=unsafe,
        regime_changed=False,
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
            domain="plasma_control",
            rules=(
                PolicyRule(
                    name="suppress edge-localised mode",
                    regimes=["DEGRADED", "CRITICAL"],
                    condition=CompoundCondition(
                        conditions=[
                            PolicyCondition("R_bad", 0, ">", 0.35),
                            PolicyCondition("stability_proxy", None, "<=", 0.55),
                        ],
                        logic="AND",
                    ),
                    actions=(PolicyAction("alpha", "edge_mode", -0.04, 2.0),),
                    max_fires=2,
                ),
                PolicyRule(
                    name="recover coherent island",
                    regimes=["RECOVERY"],
                    condition=PolicyCondition("R_good", 0, "<", 0.72),
                    actions=(PolicyAction("K", "island", 0.08, 3.0),),
                    max_fires=3,
                ),
            ),
            stl_specs=(
                PolicySTLSpec(
                    "bounded plasma instability",
                    "always (R_bad <= 0.7 and stability_proxy >= 0.2)",
                    "hard",
                ),
                PolicySTLSpec(
                    "plasma recovery",
                    "eventually (R_good >= 0.72)",
                ),
            ),
            required_labels=(
                'label "fires_suppress_edge_localised_mode"',
                'label "stl_bounded_plasma_instability_satisfied"',
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
            domain="medical_cardiac",
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
            "meta_transfer_corpus": benchmark_meta_transfer_audit_corpus_quality(),
            "meta_transfer": benchmark_meta_transfer_package_manifest_quality(),
            "plugin_ecosystem": benchmark_plugin_ecosystem_catalog_quality(),
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
