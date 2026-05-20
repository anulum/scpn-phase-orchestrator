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
    benchmark_bayesian_backend_fail_closed,
    benchmark_bayesian_posterior_fit_quality,
    benchmark_domain_formal_safety_exports,
    benchmark_formal_export_artifact_quality,
    benchmark_hybrid_cocompiler_review_gate,
    benchmark_kuramoto_reference,
    benchmark_meta_transfer_audit_corpus_quality,
    benchmark_meta_transfer_package_manifest_quality,
    benchmark_petri_reachability,
    benchmark_plugin_ecosystem_catalog_quality,
    benchmark_replay_policy_candidate_quality,
    benchmark_semantic_retrieval_ranking_quality,
    benchmark_stl_closed_loop_plan_quality,
    benchmark_stuart_landau_reference,
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
        "require_deterministic_hash": True,
        "require_non_actuating": True,
    }
    assert [plan["actuating"] for plan in plans] == [False, False, False]
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
    assert out["domain_count"] == 3
    assert out["artifact_count"] == 9
    assert out["accepted_domain_count"] == 3
    assert out["failed_domain_count"] == 0
    assert out["acceptance_passed"] == 1
    assert len(str(out["artifact_sha256"])) == 64
    assert float(out["steps_per_second"]) > 0.0


def test_domain_formal_safety_exports_reports_thresholds_and_domains() -> None:
    out = benchmark_domain_formal_safety_exports()
    thresholds = json.loads(str(out["acceptance_thresholds_json"]))
    results = json.loads(str(out["domain_results_json"]))

    assert thresholds == {
        "min_artifacts_per_domain": 3,
        "min_domain_count": 3,
        "min_rules_per_domain": 2,
        "min_stl_specs_per_domain": 2,
        "require_deterministic_hash": True,
    }
    assert [record["domain"] for record in results] == [
        "plasma_control",
        "power_grid",
        "medical_cardiac",
    ]
    assert all(record["accepted"] is True for record in results)
    assert all(record["artifact_count"] == 3 for record in results)
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
        "bayesian_backends",
        "bayesian_posterior",
        "domain_formal_export",
        "formal_export",
        "hybrid_cocompiler",
        "meta_transfer",
        "meta_transfer_corpus",
        "plugin_ecosystem",
        "replay_policy",
        "semantic_retrieval",
        "stl_closed_loop",
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
