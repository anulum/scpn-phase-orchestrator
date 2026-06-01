# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Reference benchmark suite smoke tests

from __future__ import annotations

import json

from benchmarks.reference_suite import (
    BENCHMARK_COMMAND,
    benchmark_auto_binding_proposal_quality,
    benchmark_autopoietic_lineage_sandbox_gate,
    benchmark_bayesian_backend_fail_closed,
    benchmark_bayesian_posterior_fit_quality,
    benchmark_domain_formal_safety_exports,
    benchmark_evolutionary_mutation_grammar_gate,
    benchmark_evolutionary_supervisor_search,
    benchmark_federated_deployment_preflight_gate,
    benchmark_federated_meta_orchestrator,
    benchmark_federated_production_boundary_gate,
    benchmark_formal_export_artifact_quality,
    benchmark_hybrid_cocompiler_review_gate,
    benchmark_hybrid_entanglement_order_parameter_gate,
    benchmark_hybrid_operator_handoff_package_gate,
    benchmark_hybrid_target_readiness_gate,
    benchmark_information_geometry_control_gate,
    benchmark_integrated_information_replay_corpus_gate,
    benchmark_intergenerational_policy_inheritance_gate,
    benchmark_kuramoto_reference,
    benchmark_meta_transfer_audit_corpus_quality,
    benchmark_meta_transfer_package_manifest_quality,
    benchmark_morphogenetic_domain_demo_gate,
    benchmark_multiverse_counterfactual_gate,
    benchmark_neuromorphic_target_readiness_gate,
    benchmark_petri_reachability,
    benchmark_plugin_ecosystem_catalog_quality,
    benchmark_quantum_target_readiness_gate,
    benchmark_replay_policy_candidate_quality,
    benchmark_self_model_digital_twin,
    benchmark_semantic_retrieval_ranking_quality,
    benchmark_sheaf_obstruction_domain_gate,
    benchmark_stl_closed_loop_plan_quality,
    benchmark_strange_loop_drift_scenario_gate,
    benchmark_stuart_landau_reference,
    benchmark_temporal_causal_hypergraph_experiment_gate,
    benchmark_topos_semantic_binding_gate,
    benchmark_value_alignment_replay_calibration_gate,
    build_benchmark_metadata,
    run_reference_suite,
)


def test_kuramoto_reference_benchmark_shape() -> None:
    out = benchmark_kuramoto_reference(n_oscillators=8, n_steps=20, dt=0.01)
    assert out["suite"] == "kuramoto_reference_strogatz_2000"
    assert out["n_oscillators"] == 8
    assert out["n_steps"] == 20
    assert 0.0 <= float(out["final_order_parameter"]) <= 1.0
    assert float(out["steps_per_second"]) > 0.0


def test_stuart_landau_reference_benchmark_shape() -> None:
    out = benchmark_stuart_landau_reference(n_oscillators=8, n_steps=20, dt=0.01)
    assert out["suite"] == "stuart_landau_reference_pikovsky_2001"
    assert out["n_oscillators"] == 8
    assert out["n_steps"] == 20
    assert float(out["final_mean_amplitude"]) > 0.0
    assert float(out["steps_per_second"]) > 0.0


def test_petri_reachability_benchmark_shape() -> None:
    out = benchmark_petri_reachability(n_steps=20)
    assert out["suite"] == "petri_net_reachability"
    assert out["n_steps"] == 20
    assert int(out["reachable_markings"]) >= 2
    assert float(out["steps_per_second"]) > 0.0


def test_auto_binding_proposal_quality_benchmark_shape() -> None:
    out = benchmark_auto_binding_proposal_quality()

    assert out["suite"] == "auto_binding_synthetic_quality"
    assert out["fixture_count"] == 4
    assert out["large_fixture_count"] == 4
    assert out["validation_error_count"] == 0
    assert out["extractor_coverage"] == 1.0
    assert float(out["expected_edge_recall"]) == 1.0
    assert out["domain_acceptance_passed"] == 1
    assert out["accepted_domain_count"] == 4
    assert out["failed_domain_count"] == 0
    assert out["min_domain_extractor_coverage"] == 1.0
    assert out["min_domain_expected_edge_recall"] == 1.0
    assert int(out["max_domain_validation_errors"]) == 0
    assert int(out["min_sample_count"]) >= 96
    assert float(out["steps_per_second"]) > 0.0


def test_auto_binding_proposal_quality_reports_domain_thresholds() -> None:
    out = benchmark_auto_binding_proposal_quality()
    thresholds = json.loads(str(out["domain_acceptance_thresholds_json"]))
    results = json.loads(str(out["domain_acceptance_results_json"]))

    assert set(thresholds) == {
        "phase_chain",
        "industrial_sensor_chain",
        "cardiac_rhythm_surrogate",
        "power_grid_surrogate",
    }
    assert [record["domain"] for record in results] == [
        "phase_chain",
        "industrial_sensor_chain",
        "cardiac_rhythm_surrogate",
        "power_grid_surrogate",
    ]
    assert all(record["accepted"] is True for record in results)
    assert all(record["sample_count"] >= 96 for record in results)
    assert all(record["expected_edge_recall"] >= 1.0 for record in results)
    assert all(record["validation_error_count"] == 0 for record in results)


def test_semantic_retrieval_ranking_quality_benchmark_shape() -> None:
    out = benchmark_semantic_retrieval_ranking_quality()

    assert out["suite"] == "semantic_retrieval_ranking_quality"
    assert out["evidence_count"] >= 3
    assert out["ranked_record_count"] >= 3
    assert out["feature_complete_count"] >= 3
    assert out["domainpack_top_rank"] == 1
    assert out["deterministic_hash"] == 1
    assert out["acceptance_passed"] == 1
    assert out["top_source"] == "domainpack"
    assert out["top_domainpack"] == "power_grid"
    assert float(out["retrieval_score"]) > 0.0
    assert len(str(out["ranking_sha256"])) == 64
    assert float(out["steps_per_second"]) > 0.0


def test_semantic_retrieval_ranking_quality_reports_thresholds() -> None:
    out = benchmark_semantic_retrieval_ranking_quality()
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))
    projection = json.loads(str(out["ranking_projection_json"]))

    assert thresholds == {
        "min_evidence_count": 3,
        "min_feature_complete_count": 3,
        "min_ranked_record_count": 3,
        "require_deterministic_hash": True,
        "require_domainpack_top_rank": True,
    }
    assert [record["rank"] for record in projection] == list(
        range(1, len(projection) + 1)
    )
    assert projection[0]["source"] == "domainpack"
    assert projection[0]["domainpack"] == "power_grid"
    assert projection[0]["ranking_features"]["source_priority"] == 1.0
    assert projection[0]["ranking_features"]["matched_term_count"] >= 1.0


def test_evolutionary_supervisor_search_benchmark_shape() -> None:
    out = benchmark_evolutionary_supervisor_search()

    assert out["suite"] == "evolutionary_supervisor_search"
    assert out["scenario_count"] >= 1
    assert int(out["candidate_count"]) == int(out["accepted_count"]) + int(
        out["rejected_count"]
    )
    assert int(out["candidate_count"]) >= int(out["scenario_count"]) * 4
    assert out["non_actuating"] == 1
    assert out["execution_disabled"] == 1
    assert out["operator_review_required"] == 1
    assert out["live_merge_disabled"] == 1
    assert out["hot_patch_disabled"] == 1
    assert out["claim_boundary"] == 1
    assert out["deterministic_hash"] == 1
    assert len(str(out["evolutionary_search_sha256"])) == 64
    assert float(out["steps_per_second"]) > 0.0


def test_evolutionary_supervisor_search_reports_thresholds_and_records() -> None:
    out = benchmark_evolutionary_supervisor_search()
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))
    scenarios = json.loads(str(out["scenario_records_json"]))
    candidates = json.loads(str(out["candidate_records_json"]))

    assert thresholds == {
        "min_accepted_count": 1,
        "min_candidate_count": 8,
        "min_counterfactual_filter_rejected_count": 1,
        "min_rejected_count": 1,
        "min_scenario_count": 1,
        "min_stl_filter_rejected_count": 0,
        "require_deterministic_hash": True,
        "require_execution_disabled": True,
        "require_hot_patch_disabled": True,
        "require_live_merge_disabled": True,
        "require_non_actuating": True,
        "require_operator_review": True,
    }
    assert len(scenarios) >= thresholds["min_scenario_count"]
    assert int(out["scenario_count"]) == len(scenarios)
    assert int(out["candidate_count"]) == sum(
        int(record["candidate_count"]) for record in scenarios
    )
    assert int(out["accepted_count"]) == sum(
        int(record["accepted_count"]) for record in scenarios
    )
    assert int(out["rejected_count"]) == sum(
        int(record["rejected_count"]) for record in scenarios
    )
    assert (
        int(out["stl_filter_rejected_count"])
        >= thresholds["min_stl_filter_rejected_count"]
    )
    assert (
        int(out["counterfactual_filter_rejected_count"])
        >= thresholds["min_counterfactual_filter_rejected_count"]
    )
    assert all(int(record["report_hash_match"]) == 1 for record in scenarios)
    assert all(int(record["candidate_hash_match"]) == 1 for record in scenarios)
    assert out["deterministic_hash"] == int(thresholds["require_deterministic_hash"])
    assert len(candidates) == int(out["candidate_count"])
    assert {record["status"] for record in candidates} <= {
        "accepted_for_review",
        "rejected",
    }


def test_evolutionary_mutation_grammar_gate_shape() -> None:
    out = benchmark_evolutionary_mutation_grammar_gate()

    assert out["suite"] == "evolutionary_mutation_grammar_gate"
    assert out["grammar_count"] == 3
    assert out["candidate_count"] >= 20
    assert out["mutation_kind_count"] >= 9
    assert out["non_actuating"] == 1
    assert out["execution_disabled"] == 1
    assert out["operator_review_required"] == 1
    assert out["live_merge_disabled"] == 1
    assert out["hot_patch_disabled"] == 1
    assert out["deterministic_hash"] == 1
    assert out["acceptance_passed"] == 1
    assert len(str(out["grammar_sha256"])) == 64
    assert float(out["steps_per_second"]) > 0.0


def test_evolutionary_mutation_grammar_gate_reports_records() -> None:
    out = benchmark_evolutionary_mutation_grammar_gate()
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))
    records = json.loads(str(out["grammar_records_json"]))
    mutation_kinds = json.loads(str(out["mutation_kinds_json"]))

    assert thresholds == {
        "min_candidate_count": 20,
        "min_grammar_count": 3,
        "min_mutation_kind_count": 9,
        "require_deterministic_hash": True,
        "require_execution_disabled": True,
        "require_hot_patch_disabled": True,
        "require_live_merge_disabled": True,
        "require_non_actuating": True,
        "require_operator_review": True,
    }
    assert {record["grammar"] for record in records} == {
        "petri_net",
        "policy_dsl",
        "topology",
    }
    assert {"action", "condition", "add_arc", "token_bound", "community_bridge"} <= set(
        mutation_kinds
    )
    assert all(
        record["candidate_hash_count"] == record["candidate_count"]
        for record in records
    )


def test_federated_meta_orchestrator_benchmark_shape() -> None:
    out = benchmark_federated_meta_orchestrator()

    assert out["suite"] == "federated_meta_orchestrator"
    assert out["node_count"] == 3
    assert out["accepted_node_count"] == 3
    assert out["rejected_node_count"] == 0
    assert out["policy_key_count"] == 2
    assert out["raw_time_series_received"] == 0
    assert out["raw_field_count"] == 0
    assert out["non_actuating"] == 1
    assert out["execution_disabled"] == 1
    assert out["operator_review_required"] == 1
    assert out["live_transport_disabled"] == 1
    assert out["raw_data_export_disabled"] == 1
    assert out["actuation_disabled"] == 1
    assert out["deterministic_hash"] == 1
    assert out["acceptance_passed"] == 1
    assert len(str(out["aggregate_hash"])) == 64
    assert len(str(out["report_hash"])) == 64
    assert len(str(out["federated_meta_sha256"])) == 64
    assert float(out["steps_per_second"]) > 0.0


def test_federated_meta_orchestrator_reports_thresholds_and_records() -> None:
    out = benchmark_federated_meta_orchestrator()
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))
    record = json.loads(str(out["federated_record_json"]))

    assert thresholds == {
        "max_privacy_budget_spent": 1.0,
        "max_rejected_node_count": 1,
        "min_accepted_node_count": 3,
        "min_node_count": 3,
        "min_policy_key_count": 2,
        "require_deterministic_hash": True,
        "require_execution_disabled": True,
        "require_live_transport_disabled": True,
        "require_no_raw_time_series": True,
        "require_non_actuating": True,
        "require_operator_review": True,
        "require_raw_data_export_disabled": True,
    }
    assert record["raw_time_series_received"] is False
    assert record["raw_data_export_permitted"] is False
    assert record["live_transport_permitted"] is False
    assert record["actuation_permitted"] is False
    assert len(record["node_updates"]) == int(out["node_count"])
    assert all("raw_time_series" not in node for node in record["node_updates"])
    assert all("time_series" not in node for node in record["node_updates"])
    assert all("samples" not in node for node in record["node_updates"])
    assert len(record["aggregate_delta"]) == thresholds["min_policy_key_count"]


def test_federated_production_boundary_gate_benchmark_shape() -> None:
    out = benchmark_federated_production_boundary_gate()

    assert out["suite"] == "federated_production_boundary_gate"
    assert out["boundary_surface_count"] == 3
    assert out["transport_envelope_count"] == 3
    assert out["transport_node_sequence_count"] == 3
    assert out["secure_accepted_node_count"] == 3
    assert out["secure_rejected_node_count"] == 0
    assert out["dp_noise_vector_count"] == 2
    assert out["transport_execution_disabled"] == 1
    assert out["secure_execution_disabled"] == 1
    assert out["service_execution_disabled"] == 1
    assert out["raw_data_export_disabled"] == 1
    assert out["operator_review_required"] == 1
    assert out["non_actuating"] == 1
    assert out["deterministic_hash"] == 1
    assert out["acceptance_passed"] == 1
    assert len(str(out["boundary_hash"])) == 64
    assert float(out["steps_per_second"]) > 0.0


def test_federated_production_boundary_gate_reports_thresholds_and_records() -> None:
    out = benchmark_federated_production_boundary_gate()
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))
    record = json.loads(str(out["boundary_record_json"]))

    assert thresholds == {
        "min_boundary_surface_count": 3,
        "min_dp_noise_vector_count": 2,
        "min_secure_accepted_node_count": 3,
        "min_transport_envelope_count": 3,
        "require_deterministic_hash": True,
        "require_non_actuating": True,
        "require_operator_review": True,
        "require_raw_data_export_disabled": True,
        "require_secure_execution_disabled": True,
        "require_service_execution_disabled": True,
        "require_transport_execution_disabled": True,
    }
    assert set(record) == {"dp_noise_service", "secure_aggregation", "transport"}
    assert record["transport"]["envelope_count"] == 3
    assert record["secure_aggregation"]["accepted_node_count"] == 3
    assert record["dp_noise_service"]["service_execution_permitted"] is False
    assert record["dp_noise_service"]["raw_data_export_permitted"] is False
    assert record["dp_noise_service"]["operator_review_required"] is True


def test_federated_deployment_preflight_gate_benchmark_shape() -> None:
    out = benchmark_federated_deployment_preflight_gate()

    assert out["suite"] == "federated_deployment_preflight_gate"
    assert out["preflight_surface_count"] == 3
    assert out["transport_preflight_count"] == 1
    assert out["secure_preflight_count"] == 1
    assert out["dp_preflight_count"] == 1
    assert out["transport_execution_disabled"] == 1
    assert out["secure_execution_disabled"] == 1
    assert out["service_execution_disabled"] == 1
    assert out["raw_data_export_disabled"] == 1
    assert out["operator_review_required"] == 1
    assert out["non_actuating"] == 1
    assert out["deterministic_hash"] == 1
    assert out["acceptance_passed"] == 1
    assert len(str(out["preflight_hash"])) == 64
    assert float(out["steps_per_second"]) > 0.0


def test_federated_deployment_preflight_gate_reports_thresholds_and_records() -> None:
    out = benchmark_federated_deployment_preflight_gate()
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))
    record = json.loads(str(out["preflight_record_json"]))

    assert thresholds == {
        "min_dp_preflight_count": 1,
        "min_preflight_surface_count": 3,
        "min_secure_preflight_count": 1,
        "min_transport_preflight_count": 1,
        "require_deterministic_hash": True,
        "require_non_actuating": True,
        "require_operator_review": True,
        "require_raw_data_export_disabled": True,
        "require_secure_execution_disabled": True,
        "require_service_execution_disabled": True,
        "require_transport_execution_disabled": True,
    }
    assert set(record) == {
        "dp_noise_service_preflight",
        "secure_aggregation_preflight",
        "transport_preflight",
    }
    assert record["transport_preflight"]["transport_execution_permitted"] is False
    assert (
        record["secure_aggregation_preflight"]["secure_aggregation_execution_permitted"]
        is False
    )
    assert record["dp_noise_service_preflight"]["service_execution_permitted"] is False


def test_meta_transfer_package_manifest_quality_benchmark_shape() -> None:
    out = benchmark_meta_transfer_package_manifest_quality()

    assert out["suite"] == "meta_transfer_package_manifest_quality"
    assert out["record_count"] == 4
    assert out["domain_count"] == 4
    assert out["feature_key_count"] == 5
    assert out["knob_count"] == 4
    assert int(out["package_bytes"]) > 0
    assert out["manifest_schema"] == "scpn_meta_package_manifest_v1"
    assert out["package_name"] == "scpn-meta"
    assert out["import_target"] == "scpn_phase_orchestrator.meta"
    assert out["console_script"] == "scpn-meta"
    assert out["package_digest_matches"] == 1
    assert out["execution_disabled"] == 1
    assert out["deterministic_hash"] == 1
    assert out["acceptance_passed"] == 1
    assert len(str(out["package_sha256"])) == 64
    assert len(str(out["manifest_sha256"])) == 64
    assert float(out["steps_per_second"]) > 0.0


def test_meta_transfer_package_manifest_quality_reports_thresholds() -> None:
    out = benchmark_meta_transfer_package_manifest_quality()
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))
    manifest = json.loads(str(out["manifest_json"]))

    assert thresholds == {
        "min_domain_count": 4,
        "min_feature_key_count": 5,
        "min_knob_count": 4,
        "min_record_count": 4,
        "require_deterministic_hash": True,
        "require_execution_disabled": True,
        "require_package_digest_match": True,
    }
    assert manifest["execution_permitted"] is False
    assert manifest["package_sha256"] == out["package_sha256"]
    assert manifest["training_summary"]["record_count"] == 4
    assert manifest["training_summary"]["domain_count"] == 4
    assert manifest["training_summary"]["knob_keys"] == ["K", "Psi", "alpha", "zeta"]


def test_meta_transfer_audit_corpus_quality_benchmark_shape() -> None:
    out = benchmark_meta_transfer_audit_corpus_quality()

    assert out["suite"] == "meta_transfer_audit_corpus_quality"
    assert out["record_count"] == 6
    assert out["domain_count"] == 4
    assert out["feature_key_count"] == 5
    assert out["knob_count"] == 4
    assert out["proposal_knob_count"] == 4
    assert out["neighbour_count"] == 3
    assert out["top_neighbour_domain"] == "power_grid"
    assert float(out["confidence"]) >= 0.97
    assert out["deterministic_hash"] == 1
    assert out["acceptance_passed"] == 1
    assert len(str(out["proposal_sha256"])) == 64
    assert float(out["steps_per_second"]) > 0.0


def test_meta_transfer_audit_corpus_quality_reports_thresholds() -> None:
    out = benchmark_meta_transfer_audit_corpus_quality()
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))
    proposal = json.loads(str(out["proposal_json"]))
    summary = json.loads(str(out["training_summary_json"]))

    assert thresholds == {
        "min_confidence": 0.97,
        "min_domain_count": 4,
        "min_feature_key_count": 5,
        "min_knob_count": 4,
        "min_neighbour_count": 3,
        "min_record_count": 6,
        "required_top_domain": "power_grid",
        "require_deterministic_hash": True,
    }
    assert proposal["method"] == "cosine_nearest_policy_transfer"
    assert proposal["neighbours"][0]["domain"] == "power_grid"
    assert len(proposal["neighbours"]) == 3
    assert set(proposal["knobs"]) == {"K", "Psi", "alpha", "zeta"}
    assert summary["domains"] == [
        "cardiac_rhythm",
        "manufacturing_spc",
        "power_grid",
        "traffic_flow",
    ]
    assert summary["record_count"] == 6


def test_replay_policy_candidate_quality_benchmark_shape() -> None:
    out = benchmark_replay_policy_candidate_quality()

    assert out["suite"] == "replay_policy_candidate_quality"
    assert out["scenario_count"] == 3
    assert out["accepted_scenario_count"] == 3
    assert out["failed_scenario_count"] == 0
    assert out["learner_count"] == 9
    assert out["accepted_learner_count"] == 9
    assert out["failed_learner_count"] == 0
    assert out["acceptance_rate"] == 1.0
    assert out["acceptance_passed"] == 1
    assert out["unsafe_acceptance_count"] == 0
    assert out["non_actuating_proposals"] == 1
    assert float(out["min_coherence_improvement"]) >= 0.03
    assert float(out["steps_per_second"]) > 0.0


def test_replay_policy_candidate_quality_reports_thresholds() -> None:
    out = benchmark_replay_policy_candidate_quality()
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))
    scenario_results = json.loads(str(out["scenario_results_json"]))
    results = json.loads(str(out["learner_results_json"]))

    assert thresholds == {
        "max_unsafe_acceptances": 0,
        "min_acceptance_rate": 1.0,
        "min_reward_improvement": 0.03,
        "require_non_actuating": True,
    }
    assert {record["scenario"] for record in scenario_results} == {
        "stability_recovery",
        "three_channel_cross_gain",
        "two_channel_low_coupling",
    }
    assert all(record["accepted"] is True for record in scenario_results)
    assert all(record["accepted_learner_count"] == 3 for record in scenario_results)
    assert all(record["failed_learner_count"] == 0 for record in scenario_results)
    assert all(record["unsafe_acceptance_count"] == 0 for record in scenario_results)
    assert all(record["non_actuating_proposals"] is True for record in scenario_results)
    assert {record["learner_kind"] for record in results} == {
        "ppo_like_replay",
        "sac_like_replay",
        "hybrid_physics_replay",
    }
    assert {record["scenario"] for record in results} == {
        "stability_recovery",
        "three_channel_cross_gain",
        "two_channel_low_coupling",
    }
    assert all(record["accepted"] is True for record in results)
    assert all(record["non_actuating"] is True for record in results)
    assert all(record["unsafe_selected"] is False for record in results)


def test_self_model_digital_twin_benchmark_shape() -> None:
    out = benchmark_self_model_digital_twin()

    assert out["suite"] == "self_model_digital_twin"
    assert out["scenario_count"] >= 3
    assert int(out["breach_count"]) >= 0
    assert int(out["scenario_hash_match_count"]) == int(out["scenario_count"])
    assert float(out["max_observed_error"]) >= 0.0
    assert out["non_actuating"] == 1
    assert out["operator_review_required"] == 1
    assert out["execution_disabled"] == 1
    assert out["deterministic_hash"] == 1
    assert out["acceptance_passed"] == 1
    assert len(str(out["self_model_sha256"])) == 64
    assert float(out["steps_per_second"]) > 0.0


def test_self_model_digital_twin_benchmark_reports_thresholds_and_records() -> None:
    out = benchmark_self_model_digital_twin()
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))
    scenario_results = json.loads(str(out["scenario_results_json"]))

    assert thresholds == {
        "max_breach_count": 1,
        "max_max_observed_error": 3.0,
        "min_scenario_count": 3,
        "require_deterministic_hash": True,
        "require_execution_disabled": True,
        "require_non_actuating": True,
        "require_operator_review": True,
    }
    assert int(out["scenario_count"]) == len(scenario_results)
    assert int(out["scenario_count"]) >= thresholds["min_scenario_count"]
    assert int(out["breach_count"]) <= thresholds["max_breach_count"]
    assert float(out["max_observed_error"]) <= thresholds["max_max_observed_error"]
    assert int(out["scenario_hash_match_count"]) == int(out["scenario_count"])
    assert out["deterministic_hash"] == int(thresholds["require_deterministic_hash"])
    assert all(record["scenario_hash_match"] == 1 for record in scenario_results)
    assert all(record["record_hash_match"] == 1 for record in scenario_results)
    assert all(record["within_threshold_match"] == 1 for record in scenario_results)
    assert all(record["non_actuating"] == 1 for record in scenario_results)
    assert all(record["operator_review_required"] == 1 for record in scenario_results)
    assert all(record["execution_disabled"] == 1 for record in scenario_results)
    assert all(
        float(record["max_observed_error"]) >= 0.0 for record in scenario_results
    )


def test_bayesian_posterior_fit_quality_benchmark_shape() -> None:
    out = benchmark_bayesian_posterior_fit_quality()

    assert out["suite"] == "bayesian_posterior_fit_quality"
    assert out["sample_count"] >= 96
    assert out["rollout_sample_count"] >= 96
    assert out["acceptance_passed"] == 1
    assert out["finite_audit_record"] == 1
    assert out["zero_diagonal_coupling"] == 1
    assert out["non_negative_coupling"] == 1
    assert float(out["residual_rmse"]) <= 2.5e-3
    assert float(out["omega_mean_abs_error"]) <= 3.0e-2
    assert float(out["knm_mean_abs_error"]) <= 6.0e-2
    assert float(out["credible_interval_width"]) <= 1.0e-2
    assert float(out["steps_per_second"]) > 0.0


def test_bayesian_posterior_fit_quality_reports_thresholds() -> None:
    out = benchmark_bayesian_posterior_fit_quality()
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))

    assert thresholds == {
        "max_credible_interval_width": 1.0e-2,
        "max_knm_mean_abs_error": 6.0e-2,
        "max_omega_mean_abs_error": 3.0e-2,
        "max_residual_rmse": 2.5e-3,
        "min_rollout_sample_count": 96,
    }


def test_bayesian_backend_fail_closed_benchmark_shape() -> None:
    out = benchmark_bayesian_backend_fail_closed()

    assert out["suite"] == "bayesian_backend_fail_closed"
    assert out["backend_count"] == 3
    assert out["available_backend_count"] == 1
    assert out["fail_closed_backend_count"] == 2
    assert out["unexpected_reserved_success_count"] == 0
    assert out["numpy_sample_count"] == 16
    assert out["acceptance_passed"] == 1
    assert float(out["steps_per_second"]) > 0.0


def test_bayesian_backend_fail_closed_reports_thresholds_and_records() -> None:
    out = benchmark_bayesian_backend_fail_closed()
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))
    results = json.loads(str(out["backend_results_json"]))

    assert thresholds == {
        "max_unexpected_reserved_successes": 0,
        "min_available_backends": 1,
        "required_fail_closed_backends": ["blackjax", "numpyro"],
    }
    assert [record["backend"] for record in results] == [
        "numpy",
        "numpyro",
        "blackjax",
    ]
    assert results[0]["available"] is True
    assert results[0]["sample_count"] == 16
    assert all(record["fail_closed"] is True for record in results[1:])
    assert all(record["sample_count"] == 0 for record in results[1:])


def test_formal_export_artifact_quality_benchmark_shape() -> None:
    out = benchmark_formal_export_artifact_quality()

    assert out["suite"] == "formal_export_artifact_quality"
    assert out["artifact_count"] == 5
    assert out["identifier_map_count"] >= 12
    assert out["fail_closed_count"] >= 4
    assert out["package_property_count"] == 3
    assert out["checker_command_count"] == 3
    assert out["checker_availability_count"] == 3
    assert out["checker_available_count"] == 2
    assert out["checker_missing_count"] == 1
    assert out["checker_execution_disabled"] == 1
    assert out["checker_availability_execution_disabled"] == 1
    assert out["deterministic_hash"] == 1
    assert out["acceptance_passed"] == 1
    assert len(str(out["artifact_sha256"])) == 64
    assert len(str(out["package_sha256"])) == 64
    assert int(out["petri_prism_bytes"]) > 0
    assert int(out["petri_tla_bytes"]) > 0
    assert int(out["policy_prism_bytes"]) > 0
    assert int(out["policy_tla_bytes"]) > 0
    assert int(out["stl_prism_bytes"]) > 0
    assert float(out["steps_per_second"]) > 0.0


def test_formal_export_artifact_quality_reports_thresholds() -> None:
    out = benchmark_formal_export_artifact_quality()
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))
    checker_commands = json.loads(str(out["checker_commands_json"]))
    checker_availability = json.loads(str(out["checker_availability_json"]))

    assert thresholds == {
        "min_artifact_count": 5,
        "min_checker_availability_count": 3,
        "min_checker_command_count": 3,
        "min_fail_closed_count": 4,
        "min_identifier_map_count": 12,
        "min_missing_checker_count": 1,
        "min_package_property_count": 3,
        "require_checker_execution_disabled": True,
        "require_deterministic_hash": True,
    }
    assert [command["checker"] for command in checker_commands] == [
        "prism",
        "tlc",
        "prism",
    ]
    assert all(command["execution_permitted"] is False for command in checker_commands)
    assert [command["property_name"] for command in checker_commands] == [
        "petri_reaches_done",
        "petri_type_ok",
        "policy_boost_fires",
    ]
    assert [record["status"] for record in checker_availability] == [
        "ready_not_executed",
        "missing_executable",
        "ready_not_executed",
    ]
    assert [record["executable"] for record in checker_availability] == [
        "prism",
        "tlc2.TLC",
        "prism",
    ]
    assert all(
        record["execution_permitted"] is False for record in checker_availability
    )


def test_stl_closed_loop_plan_quality_benchmark_shape() -> None:
    out = benchmark_stl_closed_loop_plan_quality()

    assert out["suite"] == "stl_closed_loop_plan_quality"
    assert out["plan_count"] == 3
    assert out["projected_action_count"] == 1
    assert out["rejected_candidate_count"] == 1
    assert out["blocked_reason_count"] == 3
    assert out["runtime_gate_checked_count"] == 3
    assert out["runtime_mapped_command_count"] == 1
    assert out["runtime_execution_disabled"] == 1
    assert out["non_actuating"] == 1
    assert out["deterministic_hash"] == 1
    assert out["acceptance_passed"] == 1
    assert len(str(out["plan_sha256"])) == 64
    assert float(out["steps_per_second"]) > 0.0


def test_stl_closed_loop_plan_quality_reports_thresholds_and_plans() -> None:
    out = benchmark_stl_closed_loop_plan_quality()
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))
    plans = json.loads(str(out["plans_json"]))

    assert thresholds == {
        "min_blocked_reason_count": 3,
        "min_plan_count": 3,
        "min_projected_action_count": 1,
        "min_runtime_gate_checked_count": 3,
        "min_runtime_mapped_command_count": 1,
        "require_deterministic_hash": True,
        "require_non_actuating": True,
        "require_runtime_execution_disabled": True,
    }
    assert [plan["actuating"] for plan in plans] == [False, False, False]
    assert [plan["runtime_actuation_gate"]["execution_disabled"] for plan in plans] == [
        True,
        True,
        True,
    ]
    assert plans[0]["runtime_actuation_gate"]["mapped_command_count"] == 1
    assert [plan["next_review_end_index"] for plan in plans] == [6, 3, 3]
    assert plans[0]["projected_action_plan"]["approved_actions"][0]["knob"] == "K"
    assert plans[1]["blocked_reasons"] == [
        "no_projected_actions",
        "unprojected_candidates",
    ]
    assert plans[2]["blocked_reasons"] == ["stl_satisfied_no_control_needed"]


def test_domain_formal_safety_exports_benchmark_shape() -> None:
    out = benchmark_domain_formal_safety_exports()

    assert out["suite"] == "domain_formal_safety_exports"
    assert out["domain_count"] == 4
    assert out["artifact_count"] == 20
    assert out["accepted_domain_count"] == 4
    assert out["failed_domain_count"] == 0
    assert out["acceptance_passed"] == 1
    assert len(str(out["artifact_sha256"])) == 64
    assert float(out["steps_per_second"]) > 0.0


def test_domain_formal_safety_exports_reports_thresholds_and_domains() -> None:
    out = benchmark_domain_formal_safety_exports()
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))
    results = json.loads(str(out["domain_results_json"]))

    assert thresholds == {
        "min_artifacts_per_domain": 5,
        "min_checker_command_count": 2,
        "min_domain_count": 4,
        "min_package_property_count": 2,
        "min_rules_per_domain": 2,
        "min_stl_specs_per_domain": 2,
        "require_deterministic_hash": True,
    }
    assert [record["domain"] for record in results] == [
        "cardiac_rhythm",
        "chemical_reactor",
        "power_grid",
        "pll_clock",
    ]
    assert all(record["accepted"] is True for record in results)
    assert all(record["artifact_count"] == 5 for record in results)
    assert all(record["package_property_count"] == 2 for record in results)
    assert all(record["checker_command_count"] == 2 for record in results)
    assert all(record["checker_execution_disabled"] == 1 for record in results)
    assert all(record["rule_count"] == 2 for record in results)
    assert all(record["stl_spec_count"] == 2 for record in results)
    assert all(record["required_labels_present"] is True for record in results)
    assert all(record["deterministic_hash"] == 1 for record in results)


def test_hybrid_cocompiler_review_gate_benchmark_shape() -> None:
    out = benchmark_hybrid_cocompiler_review_gate()

    assert out["suite"] == "hybrid_cocompiler_review_gate"
    assert out["manifest_count"] == 1
    assert out["target_backend_count"] == 4
    assert out["component_hash_count"] == 3
    assert out["quantum_term_count"] == 3
    assert out["neuromorphic_sample_count"] == 2
    assert out["blocked_probe_count"] == 2
    assert out["non_actuating"] == 1
    assert out["deterministic_hash"] == 1
    assert out["acceptance_passed"] == 1
    assert len(str(out["hybrid_manifest_sha256"])) == 64
    assert float(out["steps_per_second"]) > 0.0


def test_hybrid_cocompiler_review_gate_reports_thresholds_and_backends() -> None:
    out = benchmark_hybrid_cocompiler_review_gate()
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))
    backends = json.loads(str(out["target_backends_json"]))

    assert thresholds == {
        "min_blocked_probe_count": 2,
        "min_neuromorphic_sample_count": 2,
        "min_quantum_term_count": 3,
        "min_target_backend_count": 4,
        "require_non_actuating": True,
    }
    assert backends == ["qiskit_openqasm3", "pennylane_qasm", "lava", "pynn"]


def test_quantum_target_readiness_gate_benchmark_shape() -> None:
    out = benchmark_quantum_target_readiness_gate()

    assert out["suite"] == "quantum_target_readiness_gate"
    assert out["record_count"] == 2
    assert out["ready_count"] == 1
    assert out["blocked_count"] == 1
    assert out["blocked_reason_count"] == 2
    assert out["operator_command_count"] == 6
    assert out["non_executing"] == 1
    assert out["deterministic_hash"] == 1
    assert out["acceptance_passed"] == 1
    assert len(str(out["manifest_sha256"])) == 64
    assert len(str(out["ready_readiness_sha256"])) == 64
    assert float(out["steps_per_second"]) > 0.0


def test_quantum_target_readiness_gate_reports_thresholds_and_records() -> None:
    out = benchmark_quantum_target_readiness_gate()
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))
    backends = json.loads(str(out["target_backends_json"]))
    records = json.loads(str(out["readiness_records_json"]))

    assert thresholds == {
        "min_blocked_count": 1,
        "min_blocked_reason_count": 2,
        "min_operator_command_count": 6,
        "min_ready_count": 1,
        "require_deterministic_hash": True,
        "require_non_executing": True,
    }
    assert backends == ["qiskit_openqasm3", "pennylane_qasm"]
    assert [record["schema"] for record in records] == [
        "scpn_quantum_target_readiness_v1",
        "scpn_quantum_target_readiness_v1",
    ]
    assert [record["status"] for record in records] == [
        "blocked",
        "ready_not_executed",
    ]
    assert records[0]["blocked_reasons"] == [
        "credentials_not_configured",
        "operator_approval_missing",
    ]
    assert records[1]["blocked_reasons"] == []
    assert all(record["qpu_execution_permitted"] is False for record in records)
    assert all(record["actuation_permitted"] is False for record in records)


def test_neuromorphic_target_readiness_gate_benchmark_shape() -> None:
    out = benchmark_neuromorphic_target_readiness_gate()

    assert out["suite"] == "neuromorphic_target_readiness_gate"
    assert out["record_count"] == 2
    assert out["ready_count"] == 1
    assert out["blocked_count"] == 1
    assert out["blocked_reason_count"] == 3
    assert out["operator_command_count"] == 6
    assert out["non_executing"] == 1
    assert out["deterministic_hash"] == 1
    assert out["acceptance_passed"] == 1
    assert len(str(out["manifest_sha256"])) == 64
    assert len(str(out["ready_readiness_sha256"])) == 64
    assert float(out["steps_per_second"]) > 0.0


def test_neuromorphic_target_readiness_gate_reports_thresholds_and_records() -> None:
    out = benchmark_neuromorphic_target_readiness_gate()
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))
    backends = json.loads(str(out["target_backends_json"]))
    records = json.loads(str(out["readiness_records_json"]))

    assert thresholds == {
        "min_blocked_count": 1,
        "min_blocked_reason_count": 3,
        "min_operator_command_count": 6,
        "min_ready_count": 1,
        "require_deterministic_hash": True,
        "require_non_executing": True,
    }
    assert backends == ["lava", "pynn"]
    assert [record["schema"] for record in records] == [
        "scpn_neuromorphic_target_readiness_v1",
        "scpn_neuromorphic_target_readiness_v1",
    ]
    assert [record["status"] for record in records] == [
        "blocked",
        "ready_not_executed",
    ]
    assert records[0]["blocked_reasons"] == [
        "credentials_not_configured",
        "operator_approval_missing",
        "external_simulator_parity_not_verified",
    ]
    assert records[1]["blocked_reasons"] == []
    assert all(record["hardware_write_permitted"] is False for record in records)
    assert all(record["actuation_permitted"] is False for record in records)


def test_hybrid_target_readiness_gate_benchmark_shape() -> None:
    out = benchmark_hybrid_target_readiness_gate()

    assert out["suite"] == "hybrid_target_readiness_gate"
    assert out["record_count"] == 2
    assert out["ready_count"] == 1
    assert out["blocked_count"] == 1
    assert out["blocked_reason_count"] == 1
    assert out["operator_command_count"] == 6
    assert out["non_executing"] == 1
    assert out["deterministic_hash"] == 1
    assert out["component_hash_linked"] == 1
    assert out["acceptance_passed"] == 1
    assert len(str(out["hybrid_manifest_sha256"])) == 64
    assert len(str(out["ready_readiness_sha256"])) == 64
    assert float(out["steps_per_second"]) > 0.0


def test_hybrid_target_readiness_gate_reports_thresholds_and_records() -> None:
    out = benchmark_hybrid_target_readiness_gate()
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))
    records = json.loads(str(out["readiness_records_json"]))

    assert thresholds == {
        "min_blocked_count": 1,
        "min_blocked_reason_count": 1,
        "min_operator_command_count": 6,
        "min_ready_count": 1,
        "require_component_hash_linked": True,
        "require_deterministic_hash": True,
        "require_non_executing": True,
    }
    assert [record["schema"] for record in records] == [
        "scpn_hybrid_target_readiness_v1",
        "scpn_hybrid_target_readiness_v1",
    ]
    assert [record["status"] for record in records] == [
        "blocked",
        "ready_not_executed",
    ]
    assert records[0]["blocked_reasons"] == ["hybrid_operator_approval_missing"]
    assert records[1]["blocked_reasons"] == []
    assert all(record["qpu_execution_permitted"] is False for record in records)
    assert all(record["hardware_write_permitted"] is False for record in records)
    assert all(record["actuation_permitted"] is False for record in records)
    assert records[1]["component_statuses"] == {
        "hybrid": "co_simulation_parity_passed",
        "neuromorphic": "ready_not_executed",
        "quantum": "ready_not_executed",
    }


def test_hybrid_operator_handoff_package_gate_benchmark_shape() -> None:
    out = benchmark_hybrid_operator_handoff_package_gate()

    assert out["suite"] == "hybrid_operator_handoff_package_gate"
    assert out["package_count"] == 2
    assert out["ready_package_count"] == 1
    assert out["blocked_package_count"] == 1
    assert out["blocked_reason_count"] == 1
    assert out["operator_command_count"] == 8
    assert out["non_executing"] == 1
    assert out["deterministic_hash"] == 1
    assert out["hash_chain_linked"] == 1
    assert out["acceptance_passed"] == 1
    assert len(str(out["ready_package_sha256"])) == 64
    assert float(out["steps_per_second"]) > 0.0


def test_hybrid_operator_handoff_package_gate_reports_thresholds_and_records() -> None:
    out = benchmark_hybrid_operator_handoff_package_gate()
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))
    packages = json.loads(str(out["packages_json"]))

    assert thresholds == {
        "min_blocked_package_count": 1,
        "min_blocked_reason_count": 1,
        "min_operator_command_count": 8,
        "min_ready_package_count": 1,
        "require_deterministic_hash": True,
        "require_hash_chain_linked": True,
        "require_non_executing": True,
    }
    assert [package["schema"] for package in packages] == [
        "scpn_hybrid_operator_handoff_package_v1",
        "scpn_hybrid_operator_handoff_package_v1",
    ]
    assert [package["status"] for package in packages] == [
        "blocked",
        "ready_not_executed",
    ]
    assert packages[0]["blocked_reasons"] == ["hybrid_operator_approval_missing"]
    assert packages[1]["blocked_reasons"] == []
    assert all(package["execution_permitted"] is False for package in packages)
    assert all(package["qpu_execution_permitted"] is False for package in packages)
    assert all(package["hardware_write_permitted"] is False for package in packages)
    assert all(package["actuation_permitted"] is False for package in packages)


def test_value_alignment_replay_calibration_gate_benchmark_shape() -> None:
    out = benchmark_value_alignment_replay_calibration_gate()

    assert out["suite"] == "value_alignment_replay_calibration_gate"
    assert out["record_count"] == 1
    assert out["replay_case_count"] == 3
    assert out["approved_case_count"] == 1
    assert out["blocked_case_count"] == 1
    assert out["threshold_fallback_case_count"] == 1
    assert out["fallback_applied_case_count"] == 2
    assert out["review_only"] == 1
    assert out["deterministic_hash"] == 1
    assert out["acceptance_passed"] == 1
    assert len(str(out["calibration_sha256"])) == 64
    assert float(out["steps_per_second"]) > 0.0


def test_value_alignment_replay_calibration_gate_reports_thresholds_and_records() -> (
    None
):
    out = benchmark_value_alignment_replay_calibration_gate()
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))
    records = json.loads(str(out["calibration_records_json"]))

    assert thresholds == {
        "min_approved_case_count": 1,
        "min_blocked_case_count": 1,
        "min_fallback_applied_case_count": 2,
        "min_replay_case_count": 3,
        "min_threshold_fallback_case_count": 1,
        "require_deterministic_hash": True,
        "require_review_only": True,
    }
    assert [record["case_id"] for record in records] == [
        "approved_nominal_replay",
        "blocked_hard_limit_replay",
        "fallback_low_margin_replay",
    ]
    assert [record["satisfied"] for record in records] == [True, False, False]
    assert records[1]["violation_count"] == 1
    assert records[2]["score_counterfactual_count"] == 1


def test_autopoietic_lineage_sandbox_gate_benchmark_shape() -> None:
    out = benchmark_autopoietic_lineage_sandbox_gate()

    assert out["suite"] == "autopoietic_lineage_sandbox_gate"
    assert out["manifest_count"] == 2
    assert out["child_candidate_count"] == 5
    assert out["accepted_child_count"] == 3
    assert out["rejected_child_count"] == 2
    assert out["policy_diff_count"] == 5
    assert out["review_only"] == 1
    assert out["deterministic_hash"] == 1
    assert out["acceptance_passed"] == 1
    assert len(str(out["safe_lineage_sha256"])) == 64
    assert float(out["steps_per_second"]) > 0.0


def test_autopoietic_lineage_sandbox_gate_reports_thresholds_and_records() -> None:
    out = benchmark_autopoietic_lineage_sandbox_gate()
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))
    manifests = json.loads(str(out["lineage_manifests_json"]))

    assert thresholds == {
        "min_accepted_child_count": 3,
        "min_child_candidate_count": 5,
        "min_policy_diff_count": 5,
        "min_rejected_child_count": 2,
        "require_deterministic_hash": True,
        "require_review_only": True,
    }
    assert [manifest["schema"] for manifest in manifests] == [
        "scpn_autopoietic_lineage_sandbox_v1",
        "scpn_autopoietic_lineage_sandbox_v1",
    ]
    assert [manifest["accepted_child_count"] for manifest in manifests] == [3, 0]
    assert [manifest["rejected_child_count"] for manifest in manifests] == [0, 2]
    assert all(manifest["live_merge_permitted"] is False for manifest in manifests)
    assert all(manifest["actuation_permitted"] is False for manifest in manifests)


def test_intergenerational_policy_inheritance_gate_benchmark_shape() -> None:
    out = benchmark_intergenerational_policy_inheritance_gate()

    assert out["suite"] == "intergenerational_policy_inheritance_gate"
    assert out["manifest_count"] == 2
    assert out["signed_metadata_count"] == 2
    assert out["policy_gene_count"] == 3
    assert float(out["min_fitness_score"]) >= 0.35
    assert out["review_only"] == 1
    assert out["deterministic_hash"] == 1
    assert out["acceptance_passed"] == 1
    assert len(str(out["inheritance_sha256"])) == 64
    assert float(out["steps_per_second"]) > 0.0


def test_intergenerational_policy_inheritance_gate_reports_signed_records() -> None:
    out = benchmark_intergenerational_policy_inheritance_gate()
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))
    manifests = json.loads(str(out["inheritance_manifests_json"]))

    assert thresholds == {
        "min_fitness_score": 0.35,
        "min_manifest_count": 2,
        "min_policy_gene_count": 3,
        "min_signed_metadata_count": 2,
        "require_deterministic_hash": True,
        "require_review_only": True,
    }
    assert [manifest["schema"] for manifest in manifests] == [
        "scpn_intergenerational_policy_inheritance_v1",
        "scpn_intergenerational_policy_inheritance_v1",
    ]
    assert all(
        manifest["signed_metadata"]["signature_algorithm"] == "hmac-sha256"
        for manifest in manifests
    )
    assert all(
        manifest["direct_hot_patch_permitted"] is False for manifest in manifests
    )
    assert all(manifest["actuation_permitted"] is False for manifest in manifests)
    assert all(
        manifest["merge_strategy"] == "reviewed_hot_patch_only"
        for manifest in manifests
    )


def test_temporal_causal_hypergraph_experiment_gate_benchmark_shape() -> None:
    out = benchmark_temporal_causal_hypergraph_experiment_gate()

    assert out["suite"] == "temporal_causal_hypergraph_experiment_gate"
    assert out["manifest_count"] == 2
    assert out["accepted_hyperedge_count"] == 1
    assert out["min_baseline_edge_count"] >= 1
    assert out["research_only"] == 1
    assert out["deterministic_hash"] == 1
    assert out["acceptance_passed"] == 1
    assert len(str(out["passing_experiment_sha256"])) == 64
    assert float(out["steps_per_second"]) > 0.0


def test_temporal_causal_hypergraph_experiment_gate_reports_baselines() -> None:
    out = benchmark_temporal_causal_hypergraph_experiment_gate()
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))
    manifests = json.loads(str(out["experiment_manifests_json"]))

    assert thresholds == {
        "min_accepted_hyperedge_count": 1,
        "min_baseline_edge_count": 1,
        "min_manifest_count": 2,
        "require_deterministic_hash": True,
        "require_research_only": True,
    }
    assert [manifest["schema"] for manifest in manifests] == [
        "scpn_temporal_causal_hypergraph_experiment_v1",
        "scpn_temporal_causal_hypergraph_experiment_v1",
    ]
    assert [manifest["baseline_beaten"] for manifest in manifests] == [True, False]
    assert all(manifest["research_only"] is True for manifest in manifests)
    assert all(
        manifest["production_claim_permitted"] is False for manifest in manifests
    )
    assert all(manifest["actuation_permitted"] is False for manifest in manifests)


def test_morphogenetic_domain_demo_gate_benchmark_shape() -> None:
    out = benchmark_morphogenetic_domain_demo_gate()

    assert out["suite"] == "morphogenetic_domain_demo_gate"
    assert out["record_count"] == 3
    assert out["total_grown_edges"] >= 6
    assert out["total_shrunk_edges"] >= 6
    assert out["non_actuating"] == 1
    assert out["snapshot_rows"] == 1
    assert out["deterministic_hash"] == 1
    assert out["acceptance_passed"] == 1
    assert len(str(out["demo_sha256"])) == 64
    assert float(out["steps_per_second"]) > 0.0


def test_morphogenetic_domain_demo_gate_reports_domain_records() -> None:
    out = benchmark_morphogenetic_domain_demo_gate()
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))
    records = json.loads(str(out["demo_records_json"]))

    assert thresholds == {
        "min_demo_count": 3,
        "min_total_grown_edges": 6,
        "min_total_shrunk_edges": 6,
        "require_deterministic_hash": True,
        "require_non_actuating": True,
        "require_snapshot_rows": True,
    }
    assert [record["domainpack"] for record in records] == [
        "chemical_reactor",
        "manufacturing_spc",
        "robotic_cpg",
    ]
    assert all(record["actuating"] is False for record in records)
    assert all(record["delta_norm"] > 0.0 for record in records)
    assert all(record["snapshot_top_edge_count"] >= 6 for record in records)


def test_integrated_information_replay_corpus_gate_benchmark_shape() -> None:
    out = benchmark_integrated_information_replay_corpus_gate()

    assert out["suite"] == "integrated_information_replay_corpus_gate"
    assert out["record_count"] == 12
    assert out["domain_count"] == 3
    assert out["ordering_evidence_count"] >= 6
    assert out["non_actuating"] == 1
    assert out["claim_boundary"] == 1
    assert out["deterministic_hash"] == 1
    assert out["acceptance_passed"] == 1
    assert len(str(out["corpus_sha256"])) == 64
    assert float(out["steps_per_second"]) > 0.0


def test_integrated_information_replay_corpus_gate_reports_domains() -> None:
    out = benchmark_integrated_information_replay_corpus_gate()
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))
    domains = json.loads(str(out["domains_json"]))
    records = json.loads(str(out["replay_records_json"]))

    assert thresholds == {
        "min_domain_count": 3,
        "min_ordering_evidence_count": 6,
        "min_record_count": 12,
        "require_claim_boundary": True,
        "require_deterministic_hash": True,
        "require_non_actuating": True,
    }
    assert domains == ["cyber_industrial", "infrastructure", "physiology"]
    assert all(record["non_actuating"] is True for record in records)
    assert all(
        record["claim_boundary"] == "engineering_proxy_not_theoretical_iit"
        for record in records
    )
    assert all(record["phi"] >= 0.0 for record in records)


def test_topos_semantic_binding_gate_benchmark_shape() -> None:
    out = benchmark_topos_semantic_binding_gate()

    assert out["suite"] == "topos_semantic_binding_gate"
    assert out["record_count"] == 6
    assert out["semantic_report_count"] == 2
    assert out["policy_object_count"] >= 2
    assert out["domain_example_count"] == 3
    assert out["obligation_count"] >= 12
    assert out["non_actuating"] == 1
    assert out["proof_boundary"] == 1
    assert out["deterministic_hash"] == 1
    assert out["acceptance_passed"] == 1
    assert len(str(out["topos_sha256"])) == 64
    assert float(out["steps_per_second"]) > 0.0


def test_topos_semantic_binding_gate_reports_obligations() -> None:
    out = benchmark_topos_semantic_binding_gate()
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))
    records = json.loads(str(out["topos_records_json"]))

    assert thresholds == {
        "min_domain_example_count": 3,
        "min_obligation_count": 12,
        "min_policy_object_count": 2,
        "min_semantic_report_count": 2,
        "require_deterministic_hash": True,
        "require_non_actuating": True,
        "require_proof_boundary": True,
    }
    assert {record["kind"] for record in records} >= {
        "domain_example",
        "policy_composition_category",
        "symbolic_binding_functor",
    }
    assert all(record["passed"] is True for record in records)
    assert all(record["non_actuating"] is True for record in records)
    assert all(record["obligation_names"] for record in records)


def test_multiverse_counterfactual_gate_benchmark_shape() -> None:
    out = benchmark_multiverse_counterfactual_gate()

    assert out["suite"] == "multiverse_counterfactual_gate"
    assert out["branch_count"] >= 4
    assert out["domain_scenario_count"] >= 6
    assert out["approved_branch_count"] >= 2
    assert out["rejected_branch_count"] >= 1
    assert out["non_actuating"] == 1
    assert out["execution_disabled"] == 1
    assert out["deterministic_hash"] == 1
    assert out["acceptance_passed"] == 1
    assert len(str(out["manifest_sha256"])) == 64
    assert len(str(out["risk_report_sha256"])) == 64
    assert float(out["steps_per_second"]) > 0.0


def test_multiverse_counterfactual_gate_reports_records() -> None:
    out = benchmark_multiverse_counterfactual_gate()
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))
    branches = json.loads(str(out["branch_records_json"]))
    risk_report = json.loads(str(out["risk_report_json"]))
    scenarios = json.loads(str(out["domain_scenarios_json"]))

    assert thresholds == {
        "min_approved_branch_count": 2,
        "min_branch_count": 4,
        "min_domain_scenario_count": 6,
        "min_rejected_branch_count": 1,
        "require_deterministic_hash": True,
        "require_execution_disabled": True,
        "require_jax_backend_parity": True,
        "require_non_actuating": True,
    }
    assert {branch["branch_id"] for branch in branches} >= {
        "review_action_heavy",
        "review_safe_coupling",
    }
    assert risk_report["non_actuating"] is True
    assert risk_report["execution_disabled"] is True
    assert risk_report["approved_count"] >= 2
    assert risk_report["rejected_count"] >= 1
    assert {scenario["domain"] for scenario in scenarios} >= {
        "cardiac_rhythm",
        "cyber_industrial",
        "manufacturing_spc",
        "plasma_control",
        "power_grid",
        "traffic_flow",
    }


def test_strange_loop_drift_scenario_gate_benchmark_shape() -> None:
    out = benchmark_strange_loop_drift_scenario_gate()

    assert out["suite"] == "strange_loop_drift_scenario_gate"
    assert out["scenario_count"] >= 4
    assert out["long_run_step_count"] >= 128
    assert out["passed_scenario_count"] == out["scenario_count"]
    assert out["non_actuating"] == 1
    assert out["execution_disabled"] == 1
    assert out["deterministic_hash"] == 1
    assert out["acceptance_passed"] == 1
    assert len(str(out["drift_scenario_sha256"])) == 64
    assert float(out["steps_per_second"]) > 0.0


def test_strange_loop_drift_scenario_gate_reports_modes() -> None:
    out = benchmark_strange_loop_drift_scenario_gate()
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))
    records = json.loads(str(out["scenario_results_json"]))

    assert thresholds == {
        "min_long_run_step_count": 128,
        "min_passed_scenario_count": 4,
        "min_scenario_count": 4,
        "require_deterministic_hash": True,
        "require_execution_disabled": True,
        "require_non_actuating": True,
    }
    assert {record["expected_trigger"] for record in records} >= {
        "stable",
        "policy_drift",
        "control_loop_oscillation",
        "over_control",
    }
    assert all(record["passed_expected_trigger"] is True for record in records)
    assert all(record["non_actuating"] is True for record in records)


def test_hybrid_entanglement_order_parameter_gate_benchmark_shape() -> None:
    out = benchmark_hybrid_entanglement_order_parameter_gate()

    assert out["suite"] == "hybrid_entanglement_order_parameter_gate"
    assert out["scenario_count"] >= 2
    assert int(out["product_case_count"]) >= 1
    assert int(out["bell_case_count"]) >= 1
    assert out["non_actuating"] == 1
    assert out["execution_disabled"] == 1
    assert out["claim_boundary"] == 1
    assert out["deterministic_hash"] == 1
    assert out["entanglement_gap"] > 0.8
    assert out["acceptance_passed"] == 1
    assert float(out["max_entropy"]) >= 0.0
    assert float(out["min_entropy"]) >= 0.0
    assert len(str(out["hybrid_sha256"])) == 64
    assert float(out["steps_per_second"]) > 0.0


def test_hybrid_entanglement_order_parameter_gate_reports_records() -> None:
    out = benchmark_hybrid_entanglement_order_parameter_gate()
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))
    records = json.loads(str(out["hybrid_records_json"]))

    assert thresholds == {
        "max_product_entropy": 0.15,
        "min_bell_entropy": 0.95,
        "min_entropy_gap": 0.80,
        "min_record_count": 2,
        "require_claim_boundary": True,
        "require_deterministic_hash": True,
        "require_execution_disabled": True,
        "require_non_actuating": True,
    }
    assert (
        out["claim_boundary_value"] == "quantum_cosimulation_monitor_not_qpu_execution"
    )
    assert {record["scenario"] for record in records} >= {
        "deterministic_product_state",
        "deterministic_bell_like_state",
    }
    assert any(
        str(record["category"]).lower().startswith("product") for record in records
    )
    assert any(str(record["category"]).lower().startswith("bell") for record in records)
    assert all(
        record["backend"] == "numpy_statevector_density_matrix" for record in records
    )
    assert all(record["non_actuating"] is True for record in records)
    assert all(record["execution_disabled"] is True for record in records)
    assert all(
        record["claim_boundary"] == out["claim_boundary_value"] for record in records
    )
    assert out["entanglement_gap"] >= thresholds["min_entropy_gap"]
    assert out["entanglement_gap"] == max(
        record["entanglement_entropy"]
        for record in records
        if str(record["category"]).lower().startswith("bell")
    ) - min(
        record["entanglement_entropy"]
        for record in records
        if str(record["category"]).lower().startswith("product")
    )
    assert min(
        record["entanglement_entropy"]
        for record in records
        if record["category"] == "product"
    ) < max(
        record["entanglement_entropy"]
        for record in records
        if record["category"] == "bell_like"
    )


def test_information_geometry_control_gate_benchmark_shape() -> None:
    out = benchmark_information_geometry_control_gate()

    assert out["suite"] == "information_geometry_control_gate"
    assert out["scenario_count"] >= 2
    assert out["proposal_action_evidence_count"] >= 2
    assert out["finite_metric_count"] >= 8
    assert out["non_actuating"] == 1
    assert out["execution_disabled"] == 1
    assert out["claim_boundary"] == 1
    assert out["deterministic_hash"] == 1
    assert out["jax_backend_parity"] == 1
    assert out["jax_backend_value"] == "jax_native_information_geometry"
    assert out["acceptance_passed"] == 1
    assert len(str(out["information_geometry_sha256"])) == 64
    assert float(out["min_fisher_rao_distance"]) >= 0.0
    assert float(out["max_curvature"]) >= float(out["min_curvature"])
    assert float(out["steps_per_second"]) > 0.0


def test_information_geometry_control_gate_reports_thresholds_and_records() -> None:
    out = benchmark_information_geometry_control_gate()
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))
    records = json.loads(str(out["information_geometry_records_json"]))

    assert thresholds == {
        "min_action_evidence_count": 2,
        "min_finite_metric_count": 8,
        "min_scenario_count": 2,
        "require_claim_boundary": True,
        "require_deterministic_hash": True,
        "require_execution_disabled": True,
        "require_jax_backend_parity": True,
        "require_non_actuating": True,
    }
    assert len(records) == int(out["scenario_count"])
    assert int(out["scenario_count"]) >= thresholds["min_scenario_count"]
    assert (
        int(out["proposal_action_evidence_count"])
        >= thresholds["min_action_evidence_count"]
    )
    assert int(out["finite_metric_count"]) >= thresholds["min_finite_metric_count"]
    assert {str(record["claim_boundary"]) for record in records} == {
        str(out["claim_boundary_value"])
    }
    assert out["claim_boundary_value"]
    assert all(float(record["fisher_rao_distance"]) >= 0.0 for record in records)
    assert all(float(record["wasserstein_distance"]) >= 0.0 for record in records)
    assert all(float(record["geodesic_distance"]) >= 0.0 for record in records)
    assert all(float(record["curvature"]) >= 0.0 for record in records)
    assert all(record["non_actuating"] is True for record in records)
    assert all(record["execution_disabled"] is True for record in records)
    assert all(record["repeat_match"] == 1 for record in records)
    assert all(record["jax_backend"] == out["jax_backend_value"] for record in records)
    assert all(record["jax_parity_match"] == 1 for record in records)
    assert all(record["proposal_action_count"] >= 1 for record in records)


def test_sheaf_obstruction_domain_gate_shape() -> None:
    out = benchmark_sheaf_obstruction_domain_gate()

    assert out["suite"] == "sheaf_obstruction_domain_gate"
    assert out["record_count"] == 6
    assert out["summary_count"] == 6
    assert out["top_residual_edge_count"] >= 18
    assert out["critical_count"] >= 5
    assert float(out["min_obstruction_delta"]) >= 0.1
    assert float(out["control_energy_reduction"]) >= 0.1
    assert float(out["max_nominal_obstruction_score"]) <= 0.35
    assert out["non_actuating"] == 1
    assert out["execution_disabled"] == 1
    assert out["operator_review_required"] == 1
    assert out["deterministic_hash"] == 1
    assert out["acceptance_passed"] == 1
    assert len(str(out["sheaf_obstruction_sha256"])) == 64
    assert float(out["steps_per_second"]) > 0.0


def test_sheaf_obstruction_domain_gate_reports_records() -> None:
    out = benchmark_sheaf_obstruction_domain_gate()
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))
    control_record = json.loads(str(out["control_record_json"]))
    records = json.loads(str(out["records_json"]))

    assert thresholds == {
        "max_nominal_obstruction_score": 0.35,
        "min_critical_count": 5,
        "min_control_energy_reduction": 0.1,
        "min_demo_count": 6,
        "min_obstruction_delta": 0.1,
        "min_summary_count": 6,
        "min_top_residual_edge_count": 18,
        "require_deterministic_hash": True,
        "require_execution_disabled": True,
        "require_non_actuating": True,
        "require_operator_review": True,
    }
    assert control_record["accepted_for_review"] is True
    assert control_record["execution_disabled"] is True
    assert control_record["operator_review_required"] is True
    assert (
        control_record["projected_consistency_energy"]
        < control_record["baseline_consistency_energy"]
    )
    assert {record["domainpack"] for record in records} == {
        "cardiac_rhythm",
        "edge_consensus_nchannel",
        "manufacturing_spc",
        "network_security",
        "power_grid",
        "traffic_flow",
    }
    assert all(record["summary_present"] is True for record in records)
    assert all(record["top_residual_edge_count"] == 3 for record in records)
    assert all(
        record["incident_obstruction_score"] > record["nominal_obstruction_score"]
        for record in records
    )


def test_plugin_ecosystem_catalog_quality_benchmark_shape() -> None:
    out = benchmark_plugin_ecosystem_catalog_quality()

    assert out["suite"] == "plugin_ecosystem_catalog_quality"
    assert out["plugin_count"] == 2
    assert out["full_plugin_count"] == 3
    assert out["compatible_count"] == 2
    assert out["incompatible_count"] == 1
    assert out["capability_count"] == 5
    assert out["handoff_target_hash_count"] == 6
    assert out["handoff_blocked_count"] == 1
    assert out["handoff_loading_disabled"] == 1
    assert out["required_kind_count"] == 4
    assert out["observed_kind_count"] == 4
    assert out["deterministic_hash"] == 1
    assert out["acceptance_passed"] == 1
    assert len(str(out["registry_sha256"])) == 64
    assert len(str(out["handoff_sha256"])) == 64
    assert float(out["steps_per_second"]) > 0.0


def test_plugin_ecosystem_catalog_quality_reports_thresholds_and_counts() -> None:
    out = benchmark_plugin_ecosystem_catalog_quality()
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))
    capability_counts = json.loads(str(out["capability_counts_json"]))
    dispatch_groups = json.loads(str(out["handoff_dispatch_groups_json"]))

    assert thresholds == {
        "min_blocked_handoff_count": 1,
        "min_capability_count": 5,
        "min_handoff_target_hash_count": 5,
        "min_incompatible_count": 1,
        "min_plugin_count": 2,
        "require_deterministic_hash": True,
        "require_loading_disabled": True,
        "required_capability_kinds": ["actuator", "bridge", "extractor", "monitor"],
    }
    assert capability_counts == {
        "actuator": 1,
        "bridge": 1,
        "domainpack": 0,
        "extractor": 1,
        "monitor": 2,
    }
    assert dispatch_groups == {
        "actuator": 1,
        "bridge": 1,
        "domainpack": 0,
        "extractor": 1,
        "monitor": 2,
    }


def test_reference_suite_aggregates_all_benchmarks() -> None:
    out = run_reference_suite(snapshot_date="2026-05-06")
    assert set(out.keys()) == {"metadata", "benchmarks"}
    assert out["metadata"]["snapshot_date"] == "2026-05-06"
    assert set(out["benchmarks"].keys()) == {
        "auto_binding",
        "evolutionary_supervisor_search",
        "evolutionary_mutation_grammars",
        "federated_deployment_preflight",
        "federated_meta_orchestrator",
        "federated_production_boundary",
        "autopoietic_lineage",
        "bayesian_backends",
        "bayesian_posterior",
        "domain_formal_export",
        "formal_export",
        "hybrid_cocompiler",
        "hybrid_operator_handoff",
        "hybrid_target_readiness",
        "information_geometry_control",
        "integrated_information_replay_corpus",
        "intergenerational_inheritance",
        "hybrid_entanglement_order",
        "meta_transfer",
        "meta_transfer_corpus",
        "morphogenetic_domain_demos",
        "multiverse_counterfactual",
        "neuromorphic_target_readiness",
        "plugin_ecosystem",
        "quantum_target_readiness",
        "replay_policy",
        "self_model_digital_twin",
        "semantic_retrieval",
        "sheaf_obstruction_domains",
        "strange_loop_drift_scenarios",
        "stl_closed_loop",
        "temporal_causal_hypergraph",
        "topos_semantic_binding",
        "value_alignment_replay_calibration",
        "kuramoto",
        "stuart_landau",
        "petri_reachability",
    }


def test_reference_suite_metadata_labels_reproduction_context() -> None:
    metadata = build_benchmark_metadata(snapshot_date="2026-05-06")

    assert metadata["suite_version"] == "reference_suite_v1"
    assert metadata["snapshot_date"] == "2026-05-06"
    assert metadata["command"] == BENCHMARK_COMMAND
    assert metadata["backend"] == "python_numpy"
    assert metadata["python_version"]
    assert metadata["numpy_version"]
    assert metadata["platform"]
