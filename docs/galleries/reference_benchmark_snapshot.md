<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Reference Benchmark Snapshot -->

# Reference Benchmark Snapshot

This snapshot records the deterministic local reference-suite output used for public roadmap evidence. It is a reproducibility artefact, not a hardware-performance claim.

## Metadata

| Field | Value |
| --- | --- |
| `backend` | `python_numpy` |
| `command` | `PYTHONPATH=src python benchmarks/reference_suite.py` |
| `executable` | `/media/anulum/724AA8E84AA8AA75/aaa_God_of_the_Math_Collection/03_CODE/SCPN-PHASE-ORCHESTRATOR/.venv/bin/python` |
| `numpy_version` | `2.2.6` |
| `platform` | `Linux-6.17.0-23-generic-x86_64-with-glibc2.39` |
| `python_implementation` | `CPython` |
| `python_version` | `3.12.3` |
| `snapshot_date` | `2026-05-20` |
| `suite_version` | `reference_suite_v1` |

## Benchmark summary

| Key | Suite | Acceptance | Steps/s |
| --- | --- | ---: | ---: |
| `auto_binding` | `auto_binding_synthetic_quality` | `n/a` | `79.5511` |
| `autopoietic_lineage` | `autopoietic_lineage_sandbox_gate` | `1` | `14003.6` |
| `bayesian_backends` | `bayesian_backend_fail_closed` | `1` | `9.0716` |
| `bayesian_posterior` | `bayesian_posterior_fit_quality` | `1` | `34.4584` |
| `domain_formal_export` | `domain_formal_safety_exports` | `1` | `22519.3` |
| `formal_export` | `formal_export_artifact_quality` | `1` | `6646.86` |
| `hybrid_cocompiler` | `hybrid_cocompiler_review_gate` | `1` | `8060.03` |
| `hybrid_entanglement_order` | `hybrid_entanglement_order_parameter_gate` | `1` | `2237.8` |
| `hybrid_operator_handoff` | `hybrid_operator_handoff_package_gate` | `1` | `15337.2` |
| `hybrid_target_readiness` | `hybrid_target_readiness_gate` | `1` | `14332.8` |
| `integrated_information_replay_corpus` | `integrated_information_replay_corpus_gate` | `1` | `239.826` |
| `intergenerational_inheritance` | `intergenerational_policy_inheritance_gate` | `1` | `7371.29` |
| `kuramoto` | `kuramoto_reference_strogatz_2000` | `n/a` | `5976.14` |
| `meta_transfer` | `meta_transfer_package_manifest_quality` | `1` | `10366.6` |
| `meta_transfer_corpus` | `meta_transfer_audit_corpus_quality` | `1` | `2895.52` |
| `morphogenetic_domain_demos` | `morphogenetic_domain_demo_gate` | `1` | `57.188` |
| `multiverse_counterfactual` | `multiverse_counterfactual_gate` | `1` | `523.408` |
| `neuromorphic_target_readiness` | `neuromorphic_target_readiness_gate` | `1` | `10779.8` |
| `petri_reachability` | `petri_net_reachability` | `n/a` | `186478` |
| `plugin_ecosystem` | `plugin_ecosystem_catalog_quality` | `1` | `11257.7` |
| `quantum_target_readiness` | `quantum_target_readiness_gate` | `1` | `16186.9` |
| `replay_policy` | `replay_policy_candidate_quality` | `1` | `623.061` |
| `semantic_retrieval` | `semantic_retrieval_ranking_quality` | `1` | `172.837` |
| `stl_closed_loop` | `stl_closed_loop_plan_quality` | `1` | `9988.71` |
| `stuart_landau` | `stuart_landau_reference_pikovsky_2001` | `n/a` | `3084.71` |
| `temporal_causal_hypergraph` | `temporal_causal_hypergraph_experiment_gate` | `1` | `5044.59` |
| `topos_semantic_binding` | `topos_semantic_binding_gate` | `1` | `61.4664` |
| `value_alignment_replay_calibration` | `value_alignment_replay_calibration_gate` | `1` | `16396.4` |

## Benchmark details

### `auto_binding`

```json
{
  "accepted_domain_count": 4,
  "domain_acceptance_passed": 1,
  "domain_acceptance_results_json": "[{\"accepted\": true, \"domain\": \"phase_chain\", \"expected_edge_recall\": 1.0, \"extractor_coverage\": 1.0, \"proposed_edge_count\": 6, \"proposed_edge_multiplier\": 6.0, \"sample_count\": 128, \"source_column_count\": 3, \"validation_error_count\": 0}, {\"accepted\": true, \"domain\": \"industrial_sensor_chain\", \"expected_edge_recall\": 1.0, \"extractor_coverage\": 1.0, \"proposed_edge_count\": 6, \"proposed_edge_multiplier\": 6.0, \"sample_count\": 128, \"source_column_count\": 3, \"validation_error_count\": 0}, {\"accepted\": true, \"domain\": \"cardiac_rhythm_surrogate\", \"expected_edge_recall\": 1.0, \"extractor_coverage\": 1.0, \"proposed_edge_count\": 9, \"proposed_edge_multiplier\": 4.5, \"sample_count\": 160, \"source_column_count\": 4, \"validation_error_count\": 0}, {\"accepted\": true, \"domain\": \"power_grid_surrogate\", \"expected_edge_recall\": 1.0, \"extractor_coverage\": 1.0, \"proposed_edge_count\": 12, \"proposed_edge_multiplier\": 6.0, \"sample_count\": 192, \"source_column_count\": 4, \"validation_error_count\": 0}]",
  "domain_acceptance_thresholds_json": "{\"cardiac_rhythm_surrogate\": {\"max_proposed_edge_multiplier\": 6.0, \"max_validation_errors\": 0, \"min_expected_edge_recall\": 1.0, \"min_extractor_coverage\": 1.0, \"min_sample_count\": 128}, \"industrial_sensor_chain\": {\"max_proposed_edge_multiplier\": 8.0, \"max_validation_errors\": 0, \"min_expected_edge_recall\": 1.0, \"min_extractor_coverage\": 1.0, \"min_sample_count\": 96}, \"phase_chain\": {\"max_proposed_edge_multiplier\": 8.0, \"max_validation_errors\": 0, \"min_expected_edge_recall\": 1.0, \"min_extractor_coverage\": 1.0, \"min_sample_count\": 96}, \"power_grid_surrogate\": {\"max_proposed_edge_multiplier\": 8.0, \"max_validation_errors\": 0, \"min_expected_edge_recall\": 1.0, \"min_extractor_coverage\": 1.0, \"min_sample_count\": 160}}",
  "expected_edge_recall": 1.0,
  "extractor_coverage": 1.0,
  "failed_domain_count": 0,
  "fixture_count": 4,
  "large_fixture_count": 4,
  "max_domain_validation_errors": 0,
  "min_domain_expected_edge_recall": 1.0,
  "min_domain_extractor_coverage": 1.0,
  "min_sample_count": 128,
  "proposed_edge_count": 33,
  "steps_per_second": 79.55111374398388,
  "suite": "auto_binding_synthetic_quality",
  "validation_error_count": 0,
  "wall_time_s": 0.050282137002795935
}
```

### `autopoietic_lineage`

```json
{
  "acceptance_passed": 1,
  "acceptance_thresholds_json": "{\"min_accepted_child_count\": 3, \"min_child_candidate_count\": 5, \"min_policy_diff_count\": 5, \"min_rejected_child_count\": 2, \"require_deterministic_hash\": true, \"require_review_only\": true}",
  "accepted_child_count": 3,
  "child_candidate_count": 5,
  "deterministic_hash": 1,
  "lineage_manifests_json": "[{\"accepted_child_count\": 3, \"actuation_permitted\": false, \"child_budget\": 3, \"child_candidate_count\": 3, \"child_candidates\": [{\"actuation_permitted\": false, \"blocked_reasons\": [], \"child_id\": \"child_001\", \"child_sha256\": \"f74caf8b231b798def8488ba05136354dc6863b964c930cebf65f5bbb52ad209\", \"live_merge_permitted\": false, \"minimum_replay_reward\": 0.7, \"minimum_safety_margin\": 0.1, \"policy_diff\": [{\"child_value\": 0.44, \"delta\": 0.02, \"knob\": \"K\", \"parent_value\": 0.42}], \"replay_reward\": 0.78, \"review_required\": true, \"safety_margin\": 0.18, \"status\": \"accepted_for_review\"}, {\"actuation_permitted\": false, \"blocked_reasons\": [], \"child_id\": \"child_002\", \"child_sha256\": \"7d6987df764895dd182850b39a1aa9bc097cb34c676fab8c5ff37e2e3a380df9\", \"live_merge_permitted\": false, \"minimum_replay_reward\": 0.7, \"minimum_safety_margin\": 0.1, \"policy_diff\": [{\"child_value\": 0.13999999999999999, \"delta\": -0.04, \"knob\": \"alpha\", \"parent_value\": 0.18}], \"replay_reward\": 0.78, \"review_required\": true, \"safety_margin\": 0.18, \"status\": \"accepted_for_review\"}, {\"actuation_permitted\": false, \"blocked_reasons\": [], \"child_id\": \"child_003\", \"child_sha256\": \"5c90d8cd64b38f7fb450a3484bd0f7eee6f5a12fac20352e9ad4293b5ddd73fd\", \"live_merge_permitted\": false, \"minimum_replay_reward\": 0.7, \"minimum_safety_margin\": 0.1, \"policy_diff\": [{\"child_value\": 0.15, \"delta\": 0.06, \"knob\": \"zeta\", \"parent_value\": 0.09}], \"replay_reward\": 0.78, \"review_required\": true, \"safety_margin\": 0.18, \"status\": \"accepted_for_review\"}], \"hot_patch_permitted\": false, \"lineage_sha256\": \"830da8db3a0227d276bb5d8fa97bfe046b10db00c9976338033d64676d1b0ca8\", \"live_merge_permitted\": false, \"minimum_replay_reward\": 0.7, \"minimum_safety_margin\": 0.1, \"mutation_step\": 0.02, \"parent_policy_genome\": {\"K\": 0.42, \"alpha\": 0.18, \"zeta\": 0.09}, \"parent_policy_sha256\": \"a725a1a906e867e6ae8289fc1bdba209cda78b6a19caf97fe2fdbb5c5965f6a7\", \"rejected_child_count\": 0, \"replay_corpus_sha256\": \"a98f5ed7bfac6bcc3e1380cf7127e01c7eb7f59b862c2340f1d872bd11b3ce41\", \"replay_summary\": {\"mean_reward\": 0.78, \"mean_safety_margin\": 0.21, \"min_reward\": 0.74, \"min_safety_margin\": 0.18, \"replay_count\": 2, \"violation_count\": 0}, \"review_required\": true, \"schema\": \"scpn_autopoietic_lineage_sandbox_v1\"}, {\"accepted_child_count\": 0, \"actuation_permitted\": false, \"child_budget\": 2, \"child_candidate_count\": 2, \"child_candidates\": [{\"actuation_permitted\": false, \"blocked_reasons\": [\"replay_reward_below_minimum\", \"safety_margin_below_minimum\", \"replay_violations_present\"], \"child_id\": \"child_001\", \"child_sha256\": \"de8639cf4835a60d11c3cbe618904b3dba703b3b38ec1531610324fc5cf0023d\", \"live_merge_permitted\": false, \"minimum_replay_reward\": 0.7, \"minimum_safety_margin\": 0.1, \"policy_diff\": [{\"child_value\": 0.45999999999999996, \"delta\": 0.04, \"knob\": \"K\", \"parent_value\": 0.42}], \"replay_reward\": 0.3, \"review_required\": true, \"safety_margin\": 0.02, \"status\": \"rejected\"}, {\"actuation_permitted\": false, \"blocked_reasons\": [\"replay_reward_below_minimum\", \"safety_margin_below_minimum\", \"replay_violations_present\"], \"child_id\": \"child_002\", \"child_sha256\": \"bd2556c2cfcbb507f30b5016b446cdfca12a24f5015f0076a31ad0480a801a08\", \"live_merge_permitted\": false, \"minimum_replay_reward\": 0.7, \"minimum_safety_margin\": 0.1, \"policy_diff\": [{\"child_value\": 0.09999999999999999, \"delta\": -0.08, \"knob\": \"alpha\", \"parent_value\": 0.18}], \"replay_reward\": 0.3, \"review_required\": true, \"safety_margin\": 0.02, \"status\": \"rejected\"}], \"hot_patch_permitted\": false, \"lineage_sha256\": \"1f8de5037ca8057434f945387e7d83c91fc5fea255d4a28d737c5c9dfe46a132\", \"live_merge_permitted\": false, \"minimum_replay_reward\": 0.7, \"minimum_safety_margin\": 0.1, \"mutation_step\": 0.04, \"parent_policy_genome\": {\"K\": 0.42, \"alpha\": 0.18, \"zeta\": 0.09}, \"parent_policy_sha256\": \"a725a1a906e867e6ae8289fc1bdba209cda78b6a19caf97fe2fdbb5c5965f6a7\", \"rejected_child_count\": 2, \"replay_corpus_sha256\": \"6451329e3db6ba6957de16795cc4079d65aedcdc8ba22d266d93f3cae7eb0756\", \"replay_summary\": {\"mean_reward\": 0.3, \"mean_safety_margin\": 0.02, \"min_reward\": 0.3, \"min_safety_margin\": 0.02, \"replay_count\": 1, \"violation_count\": 1}, \"review_required\": true, \"schema\": \"scpn_autopoietic_lineage_sandbox_v1\"}]",
  "manifest_count": 2,
  "policy_diff_count": 5,
  "rejected_child_count": 2,
  "review_only": 1,
  "safe_lineage_sha256": "830da8db3a0227d276bb5d8fa97bfe046b10db00c9976338033d64676d1b0ca8",
  "steps_per_second": 14003.64161359697,
  "suite": "autopoietic_lineage_sandbox_gate",
  "wall_time_s": 0.00035704998299479485
}
```

### `bayesian_backends`

```json
{
  "acceptance_passed": 1,
  "acceptance_thresholds_json": "{\"max_unexpected_reserved_successes\": 0, \"min_available_backends\": 1, \"required_fail_closed_backends\": [\"blackjax\", \"numpyro\"]}",
  "available_backend_count": 1,
  "backend_count": 3,
  "backend_results_json": "[{\"available\": true, \"backend\": \"numpy\", \"fail_closed\": false, \"kind\": \"bayesian_backend_status\", \"reason\": \"executed\", \"sample_count\": 16}, {\"available\": false, \"backend\": \"numpyro\", \"fail_closed\": true, \"kind\": \"bayesian_backend_status\", \"reason\": \"numpyro Bayesian UPDE backend is not implemented; use backend='numpy' for reproducible Monte Carlo propagation\", \"sample_count\": 0}, {\"available\": false, \"backend\": \"blackjax\", \"fail_closed\": true, \"kind\": \"bayesian_backend_status\", \"reason\": \"blackjax Bayesian UPDE backend is not implemented; use backend='numpy' for reproducible Monte Carlo propagation\", \"sample_count\": 0}]",
  "fail_closed_backend_count": 2,
  "numpy_sample_count": 16,
  "steps_per_second": 9.071600243768675,
  "suite": "bayesian_backend_fail_closed",
  "unexpected_reserved_success_count": 0,
  "wall_time_s": 0.33070240303641185
}
```

### `bayesian_posterior`

```json
{
  "acceptance_passed": 1,
  "acceptance_thresholds_json": "{\"max_credible_interval_width\": 0.01, \"max_knm_mean_abs_error\": 0.06, \"max_omega_mean_abs_error\": 0.03, \"max_residual_rmse\": 0.0025, \"min_rollout_sample_count\": 96}",
  "credible_interval_width": 0.002121338455159605,
  "finite_audit_record": 1,
  "knm_mean_abs_error": 0.029439030191471344,
  "non_negative_coupling": 1,
  "omega_mean_abs_error": 0.007744271156763904,
  "residual_rmse": 3.904347277377099e-07,
  "rollout_sample_count": 128,
  "sample_count": 96,
  "steps_per_second": 34.45839440734481,
  "suite": "bayesian_posterior_fit_quality",
  "wall_time_s": 2.7859684599679895,
  "zero_diagonal_coupling": 1
}
```

### `domain_formal_export`

```json
{
  "acceptance_passed": 1,
  "acceptance_thresholds_json": "{\"min_artifacts_per_domain\": 3, \"min_domain_count\": 3, \"min_rules_per_domain\": 2, \"min_stl_specs_per_domain\": 2, \"require_deterministic_hash\": true}",
  "accepted_domain_count": 3,
  "artifact_count": 9,
  "artifact_sha256": "ca29f17d051e8206fcd9b7a56063a79dd6e6d16746b7ce800482e4e7297c504b",
  "domain_count": 3,
  "domain_results_json": "[{\"accepted\": true, \"artifact_count\": 3, \"deterministic_hash\": 1, \"domain\": \"plasma_control\", \"identifier_map_count\": 12, \"required_labels_present\": true, \"rule_count\": 2, \"stl_spec_count\": 2}, {\"accepted\": true, \"artifact_count\": 3, \"deterministic_hash\": 1, \"domain\": \"power_grid\", \"identifier_map_count\": 12, \"required_labels_present\": true, \"rule_count\": 2, \"stl_spec_count\": 2}, {\"accepted\": true, \"artifact_count\": 3, \"deterministic_hash\": 1, \"domain\": \"medical_cardiac\", \"identifier_map_count\": 12, \"required_labels_present\": true, \"rule_count\": 2, \"stl_spec_count\": 2}]",
  "failed_domain_count": 0,
  "steps_per_second": 22519.254481456064,
  "suite": "domain_formal_safety_exports",
  "wall_time_s": 0.000399657990783453
}
```

### `formal_export`

```json
{
  "acceptance_passed": 1,
  "acceptance_thresholds_json": "{\"min_artifact_count\": 5, \"min_checker_availability_count\": 3, \"min_checker_command_count\": 3, \"min_fail_closed_count\": 4, \"min_identifier_map_count\": 12, \"min_missing_checker_count\": 1, \"min_package_property_count\": 3, \"require_checker_execution_disabled\": true, \"require_deterministic_hash\": true}",
  "artifact_count": 5,
  "artifact_sha256": "74217fecfc92b3cf0d3d87f7b58c4278d4c758a1309f11c6d99bac429a57e378",
  "checker_availability_count": 3,
  "checker_availability_execution_disabled": 1,
  "checker_availability_json": "[{\"artifact_name\": \"petri_prism\", \"available\": true, \"checker\": \"prism\", \"command\": [\"prism\", \"petri_prism.prism\", \"-pf\", \"P>=1 [ F \\\"active_done\\\" ]\"], \"executable\": \"prism\", \"execution_permitted\": false, \"property_name\": \"petri_reaches_done\", \"resolved_path\": \"/opt/prism/bin/prism\", \"status\": \"ready_not_executed\"}, {\"artifact_name\": \"petri_tla\", \"available\": false, \"checker\": \"tlc\", \"command\": [\"tlc2.TLC\", \"petri_tla.tla\", \"-config\", \"petri_tla.cfg\"], \"executable\": \"tlc2.TLC\", \"execution_permitted\": false, \"property_name\": \"petri_type_ok\", \"resolved_path\": null, \"status\": \"missing_executable\"}, {\"artifact_name\": \"policy_prism\", \"available\": true, \"checker\": \"prism\", \"command\": [\"prism\", \"policy_prism.prism\", \"-pf\", \"P>=1 [ F \\\"fires_boost_K\\\" ]\"], \"executable\": \"prism\", \"execution_permitted\": false, \"property_name\": \"policy_boost_fires\", \"resolved_path\": \"/opt/prism/bin/prism\", \"status\": \"ready_not_executed\"}]",
  "checker_available_count": 2,
  "checker_command_count": 3,
  "checker_commands_json": "[{\"artifact_name\": \"petri_prism\", \"checker\": \"prism\", \"command\": [\"prism\", \"petri_prism.prism\", \"-pf\", \"P>=1 [ F \\\"active_done\\\" ]\"], \"execution_permitted\": false, \"property_name\": \"petri_reaches_done\"}, {\"artifact_name\": \"petri_tla\", \"checker\": \"tlc\", \"command\": [\"tlc2.TLC\", \"petri_tla.tla\", \"-config\", \"petri_tla.cfg\"], \"execution_permitted\": false, \"property_name\": \"petri_type_ok\"}, {\"artifact_name\": \"policy_prism\", \"checker\": \"prism\", \"command\": [\"prism\", \"policy_prism.prism\", \"-pf\", \"P>=1 [ F \\\"fires_boost_K\\\" ]\"], \"execution_permitted\": false, \"property_name\": \"policy_boost_fires\"}]",
  "checker_execution_disabled": 1,
  "checker_missing_count": 1,
  "deterministic_hash": 1,
  "fail_closed_count": 5,
  "identifier_map_count": 22,
  "package_property_count": 3,
  "package_sha256": "b1d5207b71b84ecc674b0d203206371f0861bd8cc03667592dfa060bac171a92",
  "petri_prism_bytes": 1012,
  "petri_tla_bytes": 1281,
  "policy_prism_bytes": 1116,
  "policy_tla_bytes": 1370,
  "steps_per_second": 6646.858973084195,
  "stl_prism_bytes": 808,
  "suite": "formal_export_artifact_quality",
  "wall_time_s": 0.0007522350060753524
}
```

### `hybrid_cocompiler`

```json
{
  "acceptance_passed": 1,
  "acceptance_thresholds_json": "{\"min_blocked_probe_count\": 2, \"min_neuromorphic_sample_count\": 2, \"min_quantum_term_count\": 3, \"min_target_backend_count\": 4, \"require_non_actuating\": true}",
  "blocked_probe_count": 2,
  "component_hash_count": 3,
  "deterministic_hash": 1,
  "hybrid_manifest_sha256": "e5510f11f3339e62ad54b723a53e737835b5c5c4d2a0274f3539533099073fa7",
  "manifest_count": 1,
  "neuromorphic_sample_count": 2,
  "non_actuating": 1,
  "quantum_term_count": 3,
  "steps_per_second": 8060.028047985213,
  "suite": "hybrid_cocompiler_review_gate",
  "target_backend_count": 4,
  "target_backends_json": "[\"qiskit_openqasm3\", \"pennylane_qasm\", \"lava\", \"pynn\"]",
  "wall_time_s": 0.00012406904716044664
}
```

### `hybrid_entanglement_order`

```json
{
  "acceptance_passed": 1,
  "acceptance_thresholds_json": "{\"max_product_entropy\": 0.15, \"min_bell_entropy\": 0.95, \"min_entropy_gap\": 0.8, \"min_record_count\": 2, \"require_claim_boundary\": true, \"require_deterministic_hash\": true, \"require_execution_disabled\": true, \"require_non_actuating\": true}",
  "bell_case_count": 1,
  "claim_boundary": 1,
  "claim_boundary_value": "quantum_cosimulation_monitor_not_qpu_execution",
  "deterministic_hash": 1,
  "entanglement_gap": 1.0,
  "execution_disabled": 1,
  "hybrid_records_json": "[{\"Psi\": 1.17781463781052, \"R\": 0.6702937830710448, \"backend\": \"numpy_statevector_density_matrix\", \"bipartition\": [[0], [1]], \"category\": \"product\", \"claim_boundary\": \"quantum_cosimulation_monitor_not_qpu_execution\", \"entanglement_entropy\": -0.0, \"execution_disabled\": true, \"non_actuating\": true, \"normalised_entanglement_entropy\": -0.0, \"participation_ratio\": 1.0, \"qubit_count\": 2, \"record_hash\": \"a74fbdf46f6315f668201278d1cca6cb5a3232cc5e7e5465d57e63566dd6ba2e\", \"scenario\": \"deterministic_product_state\"}, {\"Psi\": 1.17781463781052, \"R\": 0.6702937830710448, \"backend\": \"numpy_statevector_density_matrix\", \"bipartition\": [[0], [1]], \"category\": \"bell_like\", \"claim_boundary\": \"quantum_cosimulation_monitor_not_qpu_execution\", \"entanglement_entropy\": 1.0, \"execution_disabled\": true, \"non_actuating\": true, \"normalised_entanglement_entropy\": 1.0, \"participation_ratio\": 2.0, \"qubit_count\": 2, \"record_hash\": \"8c3e5eead502fe04ddb9b55ab494cc8e91eabfda1107124417ed2d54fd377b5b\", \"scenario\": \"deterministic_bell_like_state\"}, {\"Psi\": 2.0899424410414196, \"R\": 1.1188630228279524e-16, \"backend\": \"numpy_statevector_density_matrix\", \"bipartition\": [[0], [1]], \"category\": \"product\", \"claim_boundary\": \"quantum_cosimulation_monitor_not_qpu_execution\", \"entanglement_entropy\": -0.0, \"execution_disabled\": true, \"non_actuating\": true, \"normalised_entanglement_entropy\": -0.0, \"participation_ratio\": 1.0, \"qubit_count\": 2, \"record_hash\": \"d331852fd1d30e18bd215d9e96160e7b410b24ea402413cc5916e8b96cfb8d1a\", \"scenario\": \"hybrid_order_quantum_simulation_v1:hybrid_order_quantum_simulation_v1_product_state\"}, {\"Psi\": 2.0899424410414196, \"R\": 1.1188630228279524e-16, \"backend\": \"numpy_statevector_density_matrix\", \"bipartition\": [[0], [1]], \"category\": \"entangled\", \"claim_boundary\": \"quantum_cosimulation_monitor_not_qpu_execution\", \"entanglement_entropy\": 1.0, \"execution_disabled\": true, \"non_actuating\": true, \"normalised_entanglement_entropy\": 1.0, \"participation_ratio\": 2.0, \"qubit_count\": 2, \"record_hash\": \"1e4aac7918d8c340263dbed302b1e0d2841c7c1b454419e3afc7283c4480dfd4\", \"scenario\": \"hybrid_order_quantum_simulation_v1:hybrid_order_quantum_simulation_v1_entangled_state\"}, {\"Psi\": 2.9675474137470546, \"R\": 2.1370841983642773e-16, \"backend\": \"numpy_statevector_density_matrix\", \"bipartition\": [[0], [1, 2]], \"category\": \"product\", \"claim_boundary\": \"quantum_cosimulation_monitor_not_qpu_execution\", \"entanglement_entropy\": -0.0, \"execution_disabled\": true, \"non_actuating\": true, \"normalised_entanglement_entropy\": -0.0, \"participation_ratio\": 1.0, \"qubit_count\": 3, \"record_hash\": \"bc064ed00033640327c053874c927fe4d680b13276ea3eea34e330fcd332c502\", \"scenario\": \"hybrid_order_power_grid_v1:hybrid_order_power_grid_v1_product_state\"}, {\"Psi\": 2.9675474137470546, \"R\": 2.1370841983642773e-16, \"backend\": \"numpy_statevector_density_matrix\", \"bipartition\": [[0], [1, 2]], \"category\": \"entangled\", \"claim_boundary\": \"quantum_cosimulation_monitor_not_qpu_execution\", \"entanglement_entropy\": 1.0, \"execution_disabled\": true, \"non_actuating\": true, \"normalised_entanglement_entropy\": 1.0, \"participation_ratio\": 2.0, \"qubit_count\": 3, \"record_hash\": \"a58f47646f2497eadb745dfa5adc58bea5e7b58943ccfc902c00ba8c9e7c9e70\", \"scenario\": \"hybrid_order_power_grid_v1:hybrid_order_power_grid_v1_entangled_state\"}, {\"Psi\": 4.71238898038469, \"R\": 1.1102230246251565e-16, \"backend\": \"numpy_statevector_density_matrix\", \"bipartition\": [[0, 1], [2, 3]], \"category\": \"product\", \"claim_boundary\": \"quantum_cosimulation_monitor_not_qpu_execution\", \"entanglement_entropy\": -0.0, \"execution_disabled\": true, \"non_actuating\": true, \"normalised_entanglement_entropy\": -0.0, \"participation_ratio\": 1.0, \"qubit_count\": 4, \"record_hash\": \"39d5aa9eca7af737edbc5d8c6e5b6347372fadfbc6f3e9355dd455b27f122646\", \"scenario\": \"hybrid_order_cardiac_rhythm_v1:hybrid_order_cardiac_rhythm_v1_product_state\"}, {\"Psi\": 4.71238898038469, \"R\": 1.1102230246251565e-16, \"backend\": \"numpy_statevector_density_matrix\", \"bipartition\": [[0, 1], [2, 3]], \"category\": \"entangled\", \"claim_boundary\": \"quantum_cosimulation_monitor_not_qpu_execution\", \"entanglement_entropy\": 1.0, \"execution_disabled\": true, \"non_actuating\": true, \"normalised_entanglement_entropy\": 0.5, \"participation_ratio\": 2.0, \"qubit_count\": 4, \"record_hash\": \"b4ceb89b6c27961b531350f45249dfd28c04e63762a7a716abcadca524e81a59\", \"scenario\": \"hybrid_order_cardiac_rhythm_v1:hybrid_order_cardiac_rhythm_v1_entangled_state\"}]",
  "hybrid_sha256": "f2e1b1917871cb7580125161006455ec522bb195f873992b2dee67e1bd7f88ac",
  "max_entropy": 1.0,
  "min_entropy": -0.0,
  "non_actuating": 1,
  "product_case_count": 4,
  "scenario_count": 8,
  "steps_per_second": 2237.7973008481194,
  "suite": "hybrid_entanglement_order_parameter_gate",
  "wall_time_s": 0.0035749439848586917
}
```

### `hybrid_operator_handoff`

```json
{
  "acceptance_passed": 1,
  "acceptance_thresholds_json": "{\"min_blocked_package_count\": 1, \"min_blocked_reason_count\": 1, \"min_operator_command_count\": 8, \"min_ready_package_count\": 1, \"require_deterministic_hash\": true, \"require_hash_chain_linked\": true, \"require_non_executing\": true}",
  "blocked_package_count": 1,
  "blocked_reason_count": 1,
  "deterministic_hash": 1,
  "hash_chain_linked": 1,
  "non_executing": 1,
  "operator_command_count": 8,
  "package_count": 2,
  "packages_json": "[{\"actuation_permitted\": false, \"blocked_reasons\": [\"hybrid_operator_approval_missing\"], \"component_manifest_hashes\": {\"neuromorphic_schedule_sha256\": \"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\", \"quantum_manifest_sha256\": \"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\", \"quantum_qasm_sha256\": \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"}, \"component_statuses\": {\"hybrid\": \"co_simulation_parity_passed\", \"neuromorphic\": \"ready_not_executed\", \"quantum\": \"ready_not_executed\"}, \"execution_permitted\": false, \"hardware_write_permitted\": false, \"hybrid_manifest_sha256\": \"e5510f11f3339e62ad54b723a53e737835b5c5c4d2a0274f3539533099073fa7\", \"hybrid_readiness_sha256\": \"67f184c426a68a3c63a8e7175f0dbdf09e941b6070c93283ea486a3d5fef735d\", \"operator_commands\": [\"review hybrid_neuromorphic_quantum_cocompiler.json\", \"review scpn_hybrid_target_readiness_v1.json\", \"verify package_sha256 before external operator handoff\", \"execute only outside SPO from an approved operator workflow\"], \"package_sha256\": \"ad392e2aae056e4a5e673d00dcf93f166f32f39db53c5c6cbf8e8ab2678f9afd\", \"qpu_execution_permitted\": false, \"schema\": \"scpn_hybrid_operator_handoff_package_v1\", \"status\": \"blocked\", \"target_backends\": [\"qiskit_openqasm3\", \"pennylane_qasm\", \"lava\", \"pynn\"]}, {\"actuation_permitted\": false, \"blocked_reasons\": [], \"component_manifest_hashes\": {\"neuromorphic_schedule_sha256\": \"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\", \"quantum_manifest_sha256\": \"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\", \"quantum_qasm_sha256\": \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"}, \"component_statuses\": {\"hybrid\": \"co_simulation_parity_passed\", \"neuromorphic\": \"ready_not_executed\", \"quantum\": \"ready_not_executed\"}, \"execution_permitted\": false, \"hardware_write_permitted\": false, \"hybrid_manifest_sha256\": \"e5510f11f3339e62ad54b723a53e737835b5c5c4d2a0274f3539533099073fa7\", \"hybrid_readiness_sha256\": \"5dbf280c524594e46047c0fb342383df767713c6ebdd71d43eb5fdc5a0b5cc64\", \"operator_commands\": [\"review hybrid_neuromorphic_quantum_cocompiler.json\", \"review scpn_hybrid_target_readiness_v1.json\", \"verify package_sha256 before external operator handoff\", \"execute only outside SPO from an approved operator workflow\"], \"package_sha256\": \"c742f1c3a2ba7bfad9e1266f743c43120df8cf5c9e12da865dbcf0436b879eb5\", \"qpu_execution_permitted\": false, \"schema\": \"scpn_hybrid_operator_handoff_package_v1\", \"status\": \"ready_not_executed\", \"target_backends\": [\"qiskit_openqasm3\", \"pennylane_qasm\", \"lava\", \"pynn\"]}]",
  "ready_package_count": 1,
  "ready_package_sha256": "c742f1c3a2ba7bfad9e1266f743c43120df8cf5c9e12da865dbcf0436b879eb5",
  "steps_per_second": 15337.190142878888,
  "suite": "hybrid_operator_handoff_package_gate",
  "wall_time_s": 0.00013040198246017098
}
```

### `hybrid_target_readiness`

```json
{
  "acceptance_passed": 1,
  "acceptance_thresholds_json": "{\"min_blocked_count\": 1, \"min_blocked_reason_count\": 1, \"min_operator_command_count\": 6, \"min_ready_count\": 1, \"require_component_hash_linked\": true, \"require_deterministic_hash\": true, \"require_non_executing\": true}",
  "blocked_count": 1,
  "blocked_reason_count": 1,
  "component_hash_linked": 1,
  "deterministic_hash": 1,
  "hybrid_manifest_sha256": "e5510f11f3339e62ad54b723a53e737835b5c5c4d2a0274f3539533099073fa7",
  "non_executing": 1,
  "operator_command_count": 6,
  "readiness_records_json": "[{\"actuation_permitted\": false, \"blocked_reasons\": [\"hybrid_operator_approval_missing\"], \"component_manifest_hashes\": {\"neuromorphic_schedule_sha256\": \"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\", \"quantum_manifest_sha256\": \"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\"}, \"component_statuses\": {\"hybrid\": \"co_simulation_parity_passed\", \"neuromorphic\": \"ready_not_executed\", \"quantum\": \"ready_not_executed\"}, \"hardware_write_permitted\": false, \"hybrid_manifest_sha256\": \"e5510f11f3339e62ad54b723a53e737835b5c5c4d2a0274f3539533099073fa7\", \"hybrid_operator_approved\": false, \"neuromorphic_readiness_sha256\": \"c0aa614538ae1f3e971ea369f399c83d092fae7708aa965dd27bac995e5bfd4c\", \"operator_commands\": [\"review hybrid_neuromorphic_quantum_cocompiler.json\", \"verify quantum and neuromorphic readiness hashes before handoff\", \"submit hybrid execution only from an approved external operator workflow\"], \"qpu_execution_permitted\": false, \"quantum_readiness_sha256\": \"b7dc2e801b17cd78c8b4af4382c4d636dd77c4b958c3cecdde4938933d1f9475\", \"readiness_sha256\": \"67f184c426a68a3c63a8e7175f0dbdf09e941b6070c93283ea486a3d5fef735d\", \"schema\": \"scpn_hybrid_target_readiness_v1\", \"status\": \"blocked\"}, {\"actuation_permitted\": false, \"blocked_reasons\": [], \"component_manifest_hashes\": {\"neuromorphic_schedule_sha256\": \"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\", \"quantum_manifest_sha256\": \"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\"}, \"component_statuses\": {\"hybrid\": \"co_simulation_parity_passed\", \"neuromorphic\": \"ready_not_executed\", \"quantum\": \"ready_not_executed\"}, \"hardware_write_permitted\": false, \"hybrid_manifest_sha256\": \"e5510f11f3339e62ad54b723a53e737835b5c5c4d2a0274f3539533099073fa7\", \"hybrid_operator_approved\": true, \"neuromorphic_readiness_sha256\": \"c0aa614538ae1f3e971ea369f399c83d092fae7708aa965dd27bac995e5bfd4c\", \"operator_commands\": [\"review hybrid_neuromorphic_quantum_cocompiler.json\", \"verify quantum and neuromorphic readiness hashes before handoff\", \"submit hybrid execution only from an approved external operator workflow\"], \"qpu_execution_permitted\": false, \"quantum_readiness_sha256\": \"b7dc2e801b17cd78c8b4af4382c4d636dd77c4b958c3cecdde4938933d1f9475\", \"readiness_sha256\": \"5dbf280c524594e46047c0fb342383df767713c6ebdd71d43eb5fdc5a0b5cc64\", \"schema\": \"scpn_hybrid_target_readiness_v1\", \"status\": \"ready_not_executed\"}]",
  "ready_count": 1,
  "ready_readiness_sha256": "5dbf280c524594e46047c0fb342383df767713c6ebdd71d43eb5fdc5a0b5cc64",
  "record_count": 2,
  "steps_per_second": 14332.807473806763,
  "suite": "hybrid_target_readiness_gate",
  "wall_time_s": 0.00013954000314697623
}
```

### `integrated_information_replay_corpus`

```json
{
  "acceptance_passed": 1,
  "acceptance_thresholds_json": "{\"min_domain_count\": 3, \"min_ordering_evidence_count\": 6, \"min_record_count\": 12, \"require_claim_boundary\": true, \"require_deterministic_hash\": true, \"require_non_actuating\": true}",
  "claim_boundary": 1,
  "corpus_sha256": "004c3939c45cbf6b12a6c9baeb9a792696614f023a0c59bc0e9d97279eaf8b24",
  "deterministic_hash": 1,
  "domain_count": 3,
  "domains_json": "[\"cyber_industrial\", \"infrastructure\", \"physiology\"]",
  "non_actuating": 1,
  "ordering_evidence_count": 6,
  "record_count": 12,
  "replay_records_json": "[{\"case_name\": \"cardiac_respiratory_lock\", \"claim_boundary\": \"engineering_proxy_not_theoretical_iit\", \"domain\": \"physiology\", \"expected_relationship\": \"cardiac_respiratory_lock > cardiac_respiratory_recovery in engineering proxy integration.\", \"minimum_partition\": [[0, 1], [2, 3]], \"n_bins\": 8, \"n_oscillators\": 4, \"n_samples\": 256, \"non_actuating\": true, \"normalised_phi\": 0.21205869656204718, \"phi\": 0.44096366290559985, \"total_integration\": 0.47158338643364467}, {\"case_name\": \"cardiac_respiratory_recovery\", \"claim_boundary\": \"engineering_proxy_not_theoretical_iit\", \"domain\": \"physiology\", \"expected_relationship\": \"cardiac_respiratory_recovery < cardiac_respiratory_lock in engineering proxy integration.\", \"minimum_partition\": [[0, 2], [1, 3]], \"n_bins\": 8, \"n_oscillators\": 4, \"n_samples\": 256, \"non_actuating\": true, \"normalised_phi\": 0.048910957729649485, \"phi\": 0.1017074773463796, \"total_integration\": 0.10884102086330437}, {\"case_name\": \"eeg_sleep_spindle\", \"claim_boundary\": \"engineering_proxy_not_theoretical_iit\", \"domain\": \"physiology\", \"expected_relationship\": \"eeg_sleep_spindle > eeg_sleep_baseline in engineering proxy integration.\", \"minimum_partition\": [[0], [1, 2, 3]], \"n_bins\": 8, \"n_oscillators\": 4, \"n_samples\": 256, \"non_actuating\": true, \"normalised_phi\": 0.43912138463466466, \"phi\": 0.9131272490492912, \"total_integration\": 1.0079023062880308}, {\"case_name\": \"eeg_sleep_baseline\", \"claim_boundary\": \"engineering_proxy_not_theoretical_iit\", \"domain\": \"physiology\", \"expected_relationship\": \"eeg_sleep_baseline < eeg_sleep_spindle in engineering proxy integration.\", \"minimum_partition\": [[0, 3], [1, 2]], \"n_bins\": 8, \"n_oscillators\": 4, \"n_samples\": 256, \"non_actuating\": true, \"normalised_phi\": 0.05592197906196478, \"phi\": 0.11628648635439953, \"total_integration\": 0.2156371949797753}, {\"case_name\": \"power_grid_islanding\", \"claim_boundary\": \"engineering_proxy_not_theoretical_iit\", \"domain\": \"infrastructure\", \"expected_relationship\": \"power_grid_islanding < power_grid_resynchronisation in engineering-information proxy integration\", \"minimum_partition\": [[0], [1, 2, 3, 4, 5]], \"n_bins\": 8, \"n_oscillators\": 6, \"n_samples\": 256, \"non_actuating\": true, \"normalised_phi\": 0.0, \"phi\": 0.0, \"total_integration\": 0.32029448973968233}, {\"case_name\": \"power_grid_resynchronisation\", \"claim_boundary\": \"engineering_proxy_not_theoretical_iit\", \"domain\": \"infrastructure\", \"expected_relationship\": \"power_grid_resynchronisation > power_grid_islanding in engineering-information proxy integration\", \"minimum_partition\": [[0, 1], [2, 3, 4, 5]], \"n_bins\": 8, \"n_oscillators\": 6, \"n_samples\": 256, \"non_actuating\": true, \"normalised_phi\": 0.1712200819054804, \"phi\": 0.35604215108407994, \"total_integration\": 0.47754008544108995}, {\"case_name\": \"traffic_spillback_fragmentation\", \"claim_boundary\": \"engineering_proxy_not_theoretical_iit\", \"domain\": \"infrastructure\", \"expected_relationship\": \"traffic_spillback_fragmentation < traffic_platoon_recovery in engineering-information proxy integration\", \"minimum_partition\": [[0], [1, 2, 3, 4, 5]], \"n_bins\": 8, \"n_oscillators\": 6, \"n_samples\": 256, \"non_actuating\": true, \"normalised_phi\": 0.0, \"phi\": 0.0, \"total_integration\": 0.32032177007594664}, {\"case_name\": \"traffic_platoon_recovery\", \"claim_boundary\": \"engineering_proxy_not_theoretical_iit\", \"domain\": \"infrastructure\", \"expected_relationship\": \"traffic_platoon_recovery > traffic_spillback_fragmentation in engineering-information proxy integration\", \"minimum_partition\": [[0, 1], [2, 3, 4, 5]], \"n_bins\": 8, \"n_oscillators\": 6, \"n_samples\": 256, \"non_actuating\": true, \"normalised_phi\": 0.12903508602483743, \"phi\": 0.26832091821427817, \"total_integration\": 0.48635445391861476}, {\"case_name\": \"cyber_disruption\", \"claim_boundary\": \"engineering_proxy_not_theoretical_iit\", \"domain\": \"cyber_industrial\", \"expected_relationship\": \"cyber_disruption < cyber_recontainment in engineering proxy integration\", \"minimum_partition\": [[0, 4, 5], [1, 2, 3]], \"n_bins\": 8, \"n_oscillators\": 6, \"n_samples\": 256, \"non_actuating\": true, \"normalised_phi\": 0.04625170926710428, \"phi\": 0.09617772562371486, \"total_integration\": 0.10259992143087594}, {\"case_name\": \"cyber_recontainment\", \"claim_boundary\": \"engineering_proxy_not_theoretical_iit\", \"domain\": \"cyber_industrial\", \"expected_relationship\": \"cyber_recontainment > cyber_disruption in engineering proxy integration\", \"minimum_partition\": [[0, 5], [1, 2, 3, 4]], \"n_bins\": 8, \"n_oscillators\": 6, \"n_samples\": 256, \"non_actuating\": true, \"normalised_phi\": 0.1844030700600674, \"phi\": 0.3834554042962013, \"total_integration\": 0.4014770029901035}, {\"case_name\": \"spc_fragmentation\", \"claim_boundary\": \"engineering_proxy_not_theoretical_iit\", \"domain\": \"cyber_industrial\", \"expected_relationship\": \"spc_fragmentation < spc_recovery in engineering proxy integration\", \"minimum_partition\": [[0, 2, 4], [1, 3, 5]], \"n_bins\": 8, \"n_oscillators\": 6, \"n_samples\": 256, \"non_actuating\": true, \"normalised_phi\": 0.044657360796357355, \"phi\": 0.09286237118172999, \"total_integration\": 0.09992210941069486}, {\"case_name\": \"spc_recovery\", \"claim_boundary\": \"engineering_proxy_not_theoretical_iit\", \"domain\": \"cyber_industrial\", \"expected_relationship\": \"spc_recovery > spc_fragmentation in engineering proxy integration\", \"minimum_partition\": [[0, 4, 5], [1, 2, 3]], \"n_bins\": 8, \"n_oscillators\": 6, \"n_samples\": 256, \"non_actuating\": true, \"normalised_phi\": 0.19783580418940888, \"phi\": 0.41138798966309453, \"total_integration\": 0.43156895842705706}]",
  "steps_per_second": 239.82575708039985,
  "suite": "integrated_information_replay_corpus_gate",
  "wall_time_s": 0.050036326982080936
}
```

### `intergenerational_inheritance`

```json
{
  "acceptance_passed": 1,
  "acceptance_thresholds_json": "{\"min_fitness_score\": 0.35, \"min_manifest_count\": 2, \"min_policy_gene_count\": 3, \"min_signed_metadata_count\": 2, \"require_deterministic_hash\": true, \"require_review_only\": true}",
  "deterministic_hash": 1,
  "inheritance_manifests_json": "[{\"actuation_permitted\": false, \"child_sha256\": \"f74caf8b231b798def8488ba05136354dc6863b964c930cebf65f5bbb52ad209\", \"direct_hot_patch_permitted\": false, \"hot_patch_review_required\": true, \"inheritance_sha256\": \"77c1a624ff98d783e167bcbd4762d5df66b45e362cd3e05bb738fb6c3a49eff2\", \"inherited_policy_genome\": {\"K\": 0.44, \"alpha\": 0.18, \"zeta\": 0.09}, \"lineage_sha256\": \"e3fbe6d49daf1949fa3c6f1cea9c4ad86a065ad867aefeb1818d13ec21e2c5c6\", \"merge_strategy\": \"reviewed_hot_patch_only\", \"multi_objective_replay_fitness\": {\"fitness_score\": 0.5720000000000001, \"objective_weights\": {\"reward\": 0.6, \"safety\": 0.3, \"simplicity\": 0.1}, \"reward_component\": 0.78, \"safety_component\": 0.18, \"simplicity_component\": 0.5}, \"parent_policy_sha256\": \"a725a1a906e867e6ae8289fc1bdba209cda78b6a19caf97fe2fdbb5c5965f6a7\", \"policy_diff\": [{\"child_value\": 0.44, \"delta\": 0.02, \"knob\": \"K\", \"parent_value\": 0.42}], \"schema\": \"scpn_intergenerational_policy_inheritance_v1\", \"signed_metadata\": {\"signature_algorithm\": \"hmac-sha256\", \"signature_sha256\": \"713b0a8203340ce321976e551a016e6b8d2e03a81bef414a7395323329d2fdb1\", \"signer_id\": \"reference-suite-review-key\"}}, {\"actuation_permitted\": false, \"child_sha256\": \"7d6987df764895dd182850b39a1aa9bc097cb34c676fab8c5ff37e2e3a380df9\", \"direct_hot_patch_permitted\": false, \"hot_patch_review_required\": true, \"inheritance_sha256\": \"42fa1cdeb427ff0ece034b2d5b70f13f373fbb33ee455d0296d583992cd80064\", \"inherited_policy_genome\": {\"K\": 0.42, \"alpha\": 0.13999999999999999, \"zeta\": 0.09}, \"lineage_sha256\": \"e3fbe6d49daf1949fa3c6f1cea9c4ad86a065ad867aefeb1818d13ec21e2c5c6\", \"merge_strategy\": \"reviewed_hot_patch_only\", \"multi_objective_replay_fitness\": {\"fitness_score\": 0.5720000000000001, \"objective_weights\": {\"reward\": 0.6, \"safety\": 0.3, \"simplicity\": 0.1}, \"reward_component\": 0.78, \"safety_component\": 0.18, \"simplicity_component\": 0.5}, \"parent_policy_sha256\": \"a725a1a906e867e6ae8289fc1bdba209cda78b6a19caf97fe2fdbb5c5965f6a7\", \"policy_diff\": [{\"child_value\": 0.13999999999999999, \"delta\": -0.04, \"knob\": \"alpha\", \"parent_value\": 0.18}], \"schema\": \"scpn_intergenerational_policy_inheritance_v1\", \"signed_metadata\": {\"signature_algorithm\": \"hmac-sha256\", \"signature_sha256\": \"54928d9c550a46b03eef8ef9170a215c72f62732639f0f5d41ddf0185846ab4d\", \"signer_id\": \"reference-suite-review-key\"}}]",
  "inheritance_sha256": "77c1a624ff98d783e167bcbd4762d5df66b45e362cd3e05bb738fb6c3a49eff2",
  "manifest_count": 2,
  "min_fitness_score": 0.5720000000000001,
  "policy_gene_count": 3,
  "review_only": 1,
  "signed_metadata_count": 2,
  "steps_per_second": 7371.287536894261,
  "suite": "intergenerational_policy_inheritance_gate",
  "wall_time_s": 0.00027132302056998014
}
```

### `kuramoto`

```json
{
  "final_order_parameter": 1.0,
  "n_oscillators": 64,
  "n_steps": 1000,
  "steps_per_second": 5976.138486661259,
  "suite": "kuramoto_reference_strogatz_2000",
  "wall_time_s": 0.1673321329872124
}
```

### `meta_transfer`

```json
{
  "acceptance_passed": 1,
  "acceptance_thresholds_json": "{\"min_domain_count\": 4, \"min_feature_key_count\": 5, \"min_knob_count\": 4, \"min_record_count\": 4, \"require_deterministic_hash\": true, \"require_execution_disabled\": true, \"require_package_digest_match\": true}",
  "console_script": "scpn-meta",
  "deterministic_hash": 1,
  "domain_count": 4,
  "execution_disabled": 1,
  "feature_key_count": 5,
  "import_target": "scpn_phase_orchestrator.meta",
  "knob_count": 4,
  "manifest_json": "{\"console_script\": \"scpn-meta\", \"execution_permitted\": false, \"import_target\": \"scpn_phase_orchestrator.meta\", \"package_name\": \"scpn-meta\", \"package_sha256\": \"533acf3b37aa233b7a53da1903c99865a7e34055d3d5bcacef3501c3b9fd273f\", \"schema\": \"scpn_meta_package_manifest_v1\", \"training_summary\": {\"domain_count\": 4, \"domains\": [\"cardiac_rhythm\", \"manufacturing_spc\", \"power_grid\", \"traffic_flow\"], \"feature_keys\": [\"coherence\", \"event_rate\", \"load_variance\", \"phase_spread\", \"safety_margin\"], \"knob_keys\": [\"K\", \"Psi\", \"alpha\", \"zeta\"], \"record_count\": 4, \"reward_max\": 0.94, \"reward_mean\": 0.9, \"reward_min\": 0.86}}",
  "manifest_schema": "scpn_meta_package_manifest_v1",
  "manifest_sha256": "bf551f9836e581eba6469309784c56e3b3fd5cbfee23e55cee6018f0163df6af",
  "package_bytes": 1950,
  "package_digest_matches": 1,
  "package_name": "scpn-meta",
  "package_sha256": "533acf3b37aa233b7a53da1903c99865a7e34055d3d5bcacef3501c3b9fd273f",
  "record_count": 4,
  "steps_per_second": 10366.642319152608,
  "suite": "meta_transfer_package_manifest_quality",
  "wall_time_s": 0.00038585299625992775
}
```

### `meta_transfer_corpus`

```json
{
  "acceptance_passed": 1,
  "acceptance_thresholds_json": "{\"min_confidence\": 0.97, \"min_domain_count\": 4, \"min_feature_key_count\": 5, \"min_knob_count\": 4, \"min_neighbour_count\": 3, \"min_record_count\": 6, \"require_deterministic_hash\": true, \"required_top_domain\": \"power_grid\"}",
  "confidence": 0.9977061617091371,
  "deterministic_hash": 1,
  "domain_count": 4,
  "feature_key_count": 5,
  "knob_count": 4,
  "neighbour_count": 3,
  "proposal_json": "{\"confidence\": 0.9977061617091371, \"feature_keys\": [\"coherence\", \"event_rate\", \"load_variance\", \"phase_spread\", \"safety_margin\"], \"knobs\": {\"K\": 0.41749872361915147, \"Psi\": 0.023185338382920508, \"alpha\": 0.01, \"zeta\": 0.06318533838292051}, \"method\": \"cosine_nearest_policy_transfer\", \"neighbours\": [{\"domain\": \"power_grid\", \"similarity\": 0.9999872402392453}, {\"domain\": \"power_grid\", \"similarity\": 0.9999072410625022}, {\"domain\": \"manufacturing_spc\", \"similarity\": 0.9932240038256637}]}",
  "proposal_knob_count": 4,
  "proposal_sha256": "bfef70f740fbdedc765080f9c9bb0156ec046fd0351dab4f4117c7c259a781fb",
  "record_count": 6,
  "steps_per_second": 2895.5236684597908,
  "suite": "meta_transfer_audit_corpus_quality",
  "top_neighbour_domain": "power_grid",
  "training_summary_json": "{\"domain_count\": 4, \"domains\": [\"cardiac_rhythm\", \"manufacturing_spc\", \"power_grid\", \"traffic_flow\"], \"feature_keys\": [\"coherence\", \"event_rate\", \"load_variance\", \"phase_spread\", \"safety_margin\"], \"knob_keys\": [\"K\", \"Psi\", \"alpha\", \"zeta\"], \"record_count\": 6, \"reward_max\": 0.94, \"reward_mean\": 0.8866666666666667, \"reward_min\": 0.82}",
  "wall_time_s": 0.0020721640321426094
}
```

### `morphogenetic_domain_demos`

```json
{
  "acceptance_passed": 1,
  "acceptance_thresholds_json": "{\"min_demo_count\": 3, \"min_total_grown_edges\": 6, \"min_total_shrunk_edges\": 6, \"require_deterministic_hash\": true, \"require_non_actuating\": true, \"require_snapshot_rows\": true}",
  "demo_records_json": "[{\"actuating\": false, \"delta_norm\": 0.02931981721973743, \"domainpack\": \"chemical_reactor\", \"field_layers\": 4, \"global_coherence\": 0.32690447865207567, \"grown_edge_count\": 4, \"scenario\": \"thermal_stability_stress_with_recovery_replay\", \"shrunk_edge_count\": 8, \"snapshot_heatmap_rows\": 4, \"snapshot_top_edge_count\": 6}, {\"actuating\": false, \"delta_norm\": 0.03036151947000935, \"domainpack\": \"manufacturing_spc\", \"field_layers\": 3, \"global_coherence\": 0.3333333333333333, \"grown_edge_count\": 2, \"scenario\": \"tool_wear_pressure_spike_recovery\", \"shrunk_edge_count\": 4, \"snapshot_heatmap_rows\": 3, \"snapshot_top_edge_count\": 6}, {\"actuating\": false, \"delta_norm\": 0.020326061071851507, \"domainpack\": \"robotic_cpg\", \"field_layers\": 4, \"global_coherence\": 0.7905694150420949, \"grown_edge_count\": 6, \"scenario\": \"quadrupedal_gait_phase_field_replay\", \"shrunk_edge_count\": 6, \"snapshot_heatmap_rows\": 4, \"snapshot_top_edge_count\": 6}]",
  "demo_sha256": "f6ed239f6839b143793c3cc8d9042ed64d7190956375c027d9ede65fef28bff1",
  "deterministic_hash": 1,
  "non_actuating": 1,
  "record_count": 3,
  "snapshot_rows": 1,
  "steps_per_second": 57.188043311145144,
  "suite": "morphogenetic_domain_demo_gate",
  "total_grown_edges": 12,
  "total_shrunk_edges": 18,
  "wall_time_s": 0.05245851801009849
}
```

### `multiverse_counterfactual`

```json
{
  "acceptance_passed": 1,
  "acceptance_thresholds_json": "{\"min_approved_branch_count\": 2, \"min_branch_count\": 4, \"min_domain_scenario_count\": 3, \"min_rejected_branch_count\": 1, \"require_deterministic_hash\": true, \"require_execution_disabled\": true, \"require_non_actuating\": true}",
  "approved_branch_count": 3,
  "branch_count": 4,
  "branch_records_json": "[{\"action_count\": 0, \"action_labels\": [], \"branch_hash\": \"d9aa6a3513c21ea4f20596923dc1d75efb01d590d03ea44689cc5b62a819ee18\", \"branch_id\": \"review_baseline\", \"final_R\": 0.8436346224324295, \"final_psi\": 0.8748161530390772, \"max_R\": 0.8436346224324295, \"mean_R\": 0.8162505885045557, \"min_R\": 0.7872473240249155, \"topology_edge_count\": 20, \"topology_scale\": 2.8000000000000007}, {\"action_count\": 1, \"action_labels\": [\"K:global:0.04\"], \"branch_hash\": \"03df5545bcdaef15940385dd27fa61dadf4a7bd7a0177c3d69afd56f6166d4f1\", \"branch_id\": \"review_safe_coupling\", \"final_R\": 0.8130256299775311, \"final_psi\": 0.8734198973865335, \"max_R\": 0.8130256299775311, \"mean_R\": 0.8002495863599326, \"min_R\": 0.7872473240249155, \"topology_edge_count\": 10, \"topology_scale\": 10.0}, {\"action_count\": 2, \"action_labels\": [\"alpha:oscillator_1:0.02\", \"zeta:global:0.01\"], \"branch_hash\": \"2ef8b780e0532c671d800c50e5183ad5f76c7a6d62a24c460f652ddc0a1a9545\", \"branch_id\": \"review_phase_lag\", \"final_R\": 0.8077674169095258, \"final_psi\": 0.8700768451283023, \"max_R\": 0.8077674169095258, \"mean_R\": 0.7975783754428809, \"min_R\": 0.7872473240249155, \"topology_edge_count\": 10, \"topology_scale\": 10.0}, {\"action_count\": 7, \"action_labels\": [\"K:global:0.02\", \"K:global:0.02\", \"K:global:0.02\", \"K:global:0.02\", \"K:global:0.02\", \"K:global:0.02\", \"K:global:0.02\"], \"branch_hash\": \"cbcadd83906d0bac0022631812e5431e09d107129a5f56e9db874103e1ea542c\", \"branch_id\": \"review_action_heavy\", \"final_R\": 0.8890965556438161, \"final_psi\": 0.8780049632342795, \"max_R\": 0.8890965556438161, \"mean_R\": 0.8415732135009447, \"min_R\": 0.7872473240249155, \"topology_edge_count\": 20, \"topology_scale\": 20.0}]",
  "deterministic_hash": 1,
  "domain_scenario_count": 3,
  "domain_scenarios_json": "[{\"branch_candidates\": [{\"candidate_id\": \"power_grid_load_shed_margin\", \"claim_boundary\": \"counterfactual_branch_rollout_not_live_actuation\", \"execution_disabled\": true, \"knob_variations\": [[\"zeta\", 0.03], [\"K\", 0.12]], \"non_actuating\": true, \"objective_labels\": [\"load_stability\", \"frequency_regulation\"], \"topology_variations\": [\"ring_redundant\", \"mesh_reinforced\"]}, {\"candidate_id\": \"power_grid_regional_islanding\", \"claim_boundary\": \"counterfactual_branch_rollout_not_live_actuation\", \"execution_disabled\": true, \"knob_variations\": [[\"alpha\", 0.04], [\"beta\", 0.02]], \"non_actuating\": true, \"objective_labels\": [\"islanding_resilience\", \"power_flow_safety\"], \"topology_variations\": [\"sector_islands\", \"hierarchical_loop\"]}], \"claim_boundary\": \"counterfactual_branch_rollout_not_live_actuation\", \"domain\": \"power_grid\", \"execution_disabled\": true, \"initial_omegas\": [59.98, 60.02, 59.99, 60.01, 59.95, 60.05], \"initial_omegas_summary\": {\"count\": 6, \"max\": 60.05, \"mean\": 60.0, \"min\": 59.95, \"std\": 0.03162277660168275}, \"initial_phases\": [0.0, 0.33, 0.67, 1.05, 1.58, 2.14], \"initial_phases_summary\": {\"count\": 6, \"max\": 2.14, \"mean\": 0.9616666666666666, \"min\": 0.0, \"std\": 0.7288670813133368}, \"non_actuating\": true, \"objective_labels\": [\"load_stability\", \"frequency_regulation\", \"islanding_resilience\"], \"scenario_hash\": \"453671e219033a3746a8e59fe1ad720894378bf0f8857e86544556ccb61fb805\", \"scenario_id\": \"power_grid_counterfactual_rollout_v1\"}, {\"branch_candidates\": [{\"candidate_id\": \"cardiac_refractory_brake\", \"claim_boundary\": \"counterfactual_branch_rollout_not_live_actuation\", \"execution_disabled\": true, \"knob_variations\": [[\"K\", 0.08], [\"zeta\", 0.04]], \"non_actuating\": true, \"objective_labels\": [\"arrhythmia_suppression\", \"heartbeat_stability\"], \"topology_variations\": [\"node_reconnect_lowpass\", \"dual_loop_bradyzone\"]}, {\"candidate_id\": \"cardiac_autonomic_probe\", \"claim_boundary\": \"counterfactual_branch_rollout_not_live_actuation\", \"execution_disabled\": true, \"knob_variations\": [[\"phi\", -0.03], [\"eta\", 0.02]], \"non_actuating\": true, \"objective_labels\": [\"cycle_variability_reduction\", \"oxygenation_support\"], \"topology_variations\": [\"pacemaker_safe_override\", \"layered_autonomic\"]}], \"claim_boundary\": \"counterfactual_branch_rollout_not_live_actuation\", \"domain\": \"cardiac_rhythm\", \"execution_disabled\": true, \"initial_omegas\": [0.95, 1.02, 0.98, 1.01, 1.05], \"initial_omegas_summary\": {\"count\": 5, \"max\": 1.05, \"mean\": 1.002, \"min\": 0.95, \"std\": 0.034292856398964525}, \"initial_phases\": [0.12, 0.58, 0.94, 1.2, 1.68], \"initial_phases_summary\": {\"count\": 5, \"max\": 1.68, \"mean\": 0.9039999999999999, \"min\": 0.12, \"std\": 0.5311722884337999}, \"non_actuating\": true, \"objective_labels\": [\"arrhythmia_suppression\", \"cardio_stability\", \"oxygenation_support\"], \"scenario_hash\": \"222b7b0d09ff2d0ad1548d554287a5c353d76f884ee274bf329c644e02c95475\", \"scenario_id\": \"cardiac_rhythm_counterfactual_rollout_v1\"}, {\"branch_candidates\": [{\"candidate_id\": \"cyber_isolation_containment\", \"claim_boundary\": \"counterfactual_branch_rollout_not_live_actuation\", \"execution_disabled\": true, \"knob_variations\": [[\"gamma\", 0.06], [\"delta\", 0.03]], \"non_actuating\": true, \"objective_labels\": [\"attack_surface_reduction\", \"service_containment\"], \"topology_variations\": [\"zonal_segmentation\", \"trust_graph_hardening\"]}, {\"candidate_id\": \"cyber_traffic_cushion\", \"claim_boundary\": \"counterfactual_branch_rollout_not_live_actuation\", \"execution_disabled\": true, \"knob_variations\": [[\"K\", 0.05], [\"rho\", -0.01]], \"non_actuating\": true, \"objective_labels\": [\"latency_regulation\", \"availability_guardrail\"], \"topology_variations\": [\"flow_reroute_bypass\", \"priority_queueing\"]}], \"claim_boundary\": \"counterfactual_branch_rollout_not_live_actuation\", \"domain\": \"cyber_industrial\", \"execution_disabled\": true, \"initial_omegas\": [0.98, 1.03, 1.01, 1.0, 1.02, 1.05], \"initial_omegas_summary\": {\"count\": 6, \"max\": 1.05, \"mean\": 1.015, \"min\": 0.98, \"std\": 0.022173557826083472}, \"initial_phases\": [0.21, 0.43, 0.7, 1.02, 1.38, 1.7], \"initial_phases_summary\": {\"count\": 6, \"max\": 1.7, \"mean\": 0.9066666666666666, \"min\": 0.21, \"std\": 0.5198610925579596}, \"non_actuating\": true, \"objective_labels\": [\"attack_surface_reduction\", \"service_containment\", \"latency_regulation\"], \"scenario_hash\": \"93928e5a5d98c4c20a0ad0323405a02ba3910e4cef619d511b106ec71c70ab8e\", \"scenario_id\": \"cyber_industrial_counterfactual_rollout_v1\"}]",
  "execution_disabled": 1,
  "manifest_sha256": "2882c461784d464678178e29f03f6a133f9af51e54ff08196f7caf4c8af8b860",
  "non_actuating": 1,
  "rejected_branch_count": 1,
  "risk_report_json": "{\"approved_count\": 3, \"branch_count\": 4, \"branch_decisions\": [{\"action_count\": 0, \"approved\": true, \"branch_hash\": \"d9aa6a3513c21ea4f20596923dc1d75efb01d590d03ea44689cc5b62a819ee18\", \"branch_id\": \"review_baseline\", \"final_R\": 0.8436346224324295, \"max_R\": 0.8436346224324295, \"mean_R\": 0.8162505885045557, \"min_R\": 0.7872473240249155, \"rejection_reasons\": [], \"topology_edge_count\": 20, \"topology_scale\": 2.8000000000000007}, {\"action_count\": 1, \"approved\": true, \"branch_hash\": \"03df5545bcdaef15940385dd27fa61dadf4a7bd7a0177c3d69afd56f6166d4f1\", \"branch_id\": \"review_safe_coupling\", \"final_R\": 0.8130256299775311, \"max_R\": 0.8130256299775311, \"mean_R\": 0.8002495863599326, \"min_R\": 0.7872473240249155, \"rejection_reasons\": [], \"topology_edge_count\": 10, \"topology_scale\": 10.0}, {\"action_count\": 2, \"approved\": true, \"branch_hash\": \"2ef8b780e0532c671d800c50e5183ad5f76c7a6d62a24c460f652ddc0a1a9545\", \"branch_id\": \"review_phase_lag\", \"final_R\": 0.8077674169095258, \"max_R\": 0.8077674169095258, \"mean_R\": 0.7975783754428809, \"min_R\": 0.7872473240249155, \"rejection_reasons\": [], \"topology_edge_count\": 10, \"topology_scale\": 10.0}, {\"action_count\": 7, \"approved\": false, \"branch_hash\": \"cbcadd83906d0bac0022631812e5431e09d107129a5f56e9db874103e1ea542c\", \"branch_id\": \"review_action_heavy\", \"final_R\": 0.8890965556438161, \"max_R\": 0.8890965556438161, \"mean_R\": 0.8415732135009447, \"min_R\": 0.7872473240249155, \"rejection_reasons\": [\"action_count_exceeds_limit\", \"topology_scale_exceeds_limit\"], \"topology_edge_count\": 20, \"topology_scale\": 20.0}], \"claim_boundary\": \"counterfactual_branch_risk_gate_not_live_actuation\", \"execution_disabled\": true, \"non_actuating\": true, \"rejected_count\": 1, \"rejection_reasons\": [\"action_count_exceeds_limit\", \"topology_scale_exceeds_limit\"], \"report_hash\": \"68f819ccf31e498860daa384573a642b327ad55ec33d7a6de9a648b7a3137b0d\", \"safest_branch_hash\": \"d9aa6a3513c21ea4f20596923dc1d75efb01d590d03ea44689cc5b62a819ee18\", \"safest_branch_id\": \"review_baseline\", \"schema_name\": \"multiverse_branch_risk_gate\", \"schema_version\": \"0.1.0\"}",
  "risk_report_sha256": "68f819ccf31e498860daa384573a642b327ad55ec33d7a6de9a648b7a3137b0d",
  "safest_branch_id": "review_baseline",
  "steps_per_second": 523.4079118308698,
  "suite": "multiverse_counterfactual_gate",
  "wall_time_s": 0.007642223034054041
}
```

### `neuromorphic_target_readiness`

```json
{
  "acceptance_passed": 1,
  "acceptance_thresholds_json": "{\"min_blocked_count\": 1, \"min_blocked_reason_count\": 3, \"min_operator_command_count\": 6, \"min_ready_count\": 1, \"require_deterministic_hash\": true, \"require_non_executing\": true}",
  "blocked_count": 1,
  "blocked_reason_count": 3,
  "deterministic_hash": 1,
  "manifest_sha256": "b6d66744f1488a0711c5b40a7fef273c3ab81e8a8eb030ab873e4e75f831600a",
  "non_executing": 1,
  "operator_command_count": 6,
  "readiness_records_json": "[{\"actuation_permitted\": false, \"blocked_reasons\": [\"credentials_not_configured\", \"operator_approval_missing\", \"external_simulator_parity_not_verified\"], \"credentials_configured\": false, \"external_simulator_parity_verified\": false, \"hardware_site\": \"lab_lava_cluster\", \"hardware_write_permitted\": false, \"manifest_sha256\": \"b6d66744f1488a0711c5b40a7fef273c3ab81e8a8eb030ab873e4e75f831600a\", \"operator_approved\": false, \"operator_commands\": [\"review neuromorphic_schedule_manifest.json\", \"run target simulator parity outside SPO before hardware handoff\", \"submit neuromorphic hardware job only from an approved operator workflow\"], \"readiness_sha256\": \"b766c4400035c3f63fa06d2b5ef34d8aca57daf845eb3c5a201106a03f62aa7f\", \"schema\": \"scpn_neuromorphic_target_readiness_v1\", \"status\": \"blocked\", \"target_backend\": \"lava\"}, {\"actuation_permitted\": false, \"blocked_reasons\": [], \"credentials_configured\": true, \"external_simulator_parity_verified\": true, \"hardware_site\": \"brainscales_review_lane\", \"hardware_write_permitted\": false, \"manifest_sha256\": \"b6d66744f1488a0711c5b40a7fef273c3ab81e8a8eb030ab873e4e75f831600a\", \"operator_approved\": true, \"operator_commands\": [\"review neuromorphic_schedule_manifest.json\", \"run target simulator parity outside SPO before hardware handoff\", \"submit neuromorphic hardware job only from an approved operator workflow\"], \"readiness_sha256\": \"fbff4ea82152b5fb51733f179661b8ec117b1afc076c80972e610af7717368d0\", \"schema\": \"scpn_neuromorphic_target_readiness_v1\", \"status\": \"ready_not_executed\", \"target_backend\": \"pynn\"}]",
  "ready_count": 1,
  "ready_readiness_sha256": "fbff4ea82152b5fb51733f179661b8ec117b1afc076c80972e610af7717368d0",
  "record_count": 2,
  "steps_per_second": 10779.813224195783,
  "suite": "neuromorphic_target_readiness_gate",
  "target_backends_json": "[\"lava\", \"pynn\"]",
  "wall_time_s": 0.00018553197151049972
}
```

### `petri_reachability`

```json
{
  "n_steps": 5000,
  "reachable_markings": 4,
  "steps_per_second": 186477.87066214517,
  "suite": "petri_net_reachability",
  "wall_time_s": 0.026812832977157086
}
```

### `plugin_ecosystem`

```json
{
  "acceptance_passed": 1,
  "acceptance_thresholds_json": "{\"min_blocked_handoff_count\": 1, \"min_capability_count\": 5, \"min_handoff_target_hash_count\": 5, \"min_incompatible_count\": 1, \"min_plugin_count\": 2, \"require_deterministic_hash\": true, \"require_loading_disabled\": true, \"required_capability_kinds\": [\"actuator\", \"bridge\", \"extractor\", \"monitor\"]}",
  "capability_count": 5,
  "capability_counts_json": "{\"actuator\": 1, \"bridge\": 1, \"domainpack\": 0, \"extractor\": 1, \"monitor\": 2}",
  "compatible_count": 2,
  "deterministic_hash": 1,
  "full_plugin_count": 3,
  "handoff_blocked_count": 1,
  "handoff_dispatch_groups_json": "{\"actuator\": 1, \"bridge\": 1, \"domainpack\": 0, \"extractor\": 1, \"monitor\": 2}",
  "handoff_loading_disabled": 1,
  "handoff_sha256": "db0fd80e5a3d3468412f0314558b017f9a2f4473d5d7a9ab768e40d86eaf3f77",
  "handoff_target_hash_count": 6,
  "incompatible_count": 1,
  "observed_kind_count": 4,
  "plugin_count": 2,
  "registry_sha256": "4dc86c339a42dba16bfe99c79fd6197051c87c97c7fbbf2a93dd86c1585ff25b",
  "required_kind_count": 4,
  "steps_per_second": 11257.667908936699,
  "suite": "plugin_ecosystem_catalog_quality",
  "wall_time_s": 0.0002664850326254964
}
```

### `quantum_target_readiness`

```json
{
  "acceptance_passed": 1,
  "acceptance_thresholds_json": "{\"min_blocked_count\": 1, \"min_blocked_reason_count\": 2, \"min_operator_command_count\": 6, \"min_ready_count\": 1, \"require_deterministic_hash\": true, \"require_non_executing\": true}",
  "blocked_count": 1,
  "blocked_reason_count": 2,
  "deterministic_hash": 1,
  "manifest_sha256": "e323283dbcdc138915a6d2a9728fdcce9dfa9600245428298d60c21b3a5ac30d",
  "non_executing": 1,
  "operator_command_count": 6,
  "readiness_records_json": "[{\"actuation_permitted\": false, \"blocked_reasons\": [\"credentials_not_configured\", \"operator_approval_missing\"], \"credentials_configured\": false, \"manifest_sha256\": \"e323283dbcdc138915a6d2a9728fdcce9dfa9600245428298d60c21b3a5ac30d\", \"operator_approved\": false, \"operator_commands\": [\"review quantum_compiler_manifest.json\", \"run simulator parity outside SPO before target handoff\", \"submit QPU job only from an approved external operator workflow\"], \"provider\": \"ibm_quantum\", \"qpu_execution_permitted\": false, \"readiness_sha256\": \"c3b3fed3ff885d7b3738e2d46914be184671614cbb09bf5ba4c52be728fc875d\", \"schema\": \"scpn_quantum_target_readiness_v1\", \"status\": \"blocked\", \"target_backend\": \"qiskit_openqasm3\"}, {\"actuation_permitted\": false, \"blocked_reasons\": [], \"credentials_configured\": true, \"manifest_sha256\": \"e323283dbcdc138915a6d2a9728fdcce9dfa9600245428298d60c21b3a5ac30d\", \"operator_approved\": true, \"operator_commands\": [\"review quantum_compiler_manifest.json\", \"run simulator parity outside SPO before target handoff\", \"submit QPU job only from an approved external operator workflow\"], \"provider\": \"pennylane\", \"qpu_execution_permitted\": false, \"readiness_sha256\": \"aa0f85ce5bbfd35acf04d96e29d3bb64edf7ce5b091193263b13712d98f6134c\", \"schema\": \"scpn_quantum_target_readiness_v1\", \"status\": \"ready_not_executed\", \"target_backend\": \"pennylane_qasm\"}]",
  "ready_count": 1,
  "ready_readiness_sha256": "aa0f85ce5bbfd35acf04d96e29d3bb64edf7ce5b091193263b13712d98f6134c",
  "record_count": 2,
  "steps_per_second": 16186.861862737569,
  "suite": "quantum_target_readiness_gate",
  "target_backends_json": "[\"qiskit_openqasm3\", \"pennylane_qasm\"]",
  "wall_time_s": 0.00012355699436739087
}
```

### `replay_policy`

```json
{
  "acceptance_passed": 1,
  "acceptance_rate": 1.0,
  "acceptance_thresholds_json": "{\"max_unsafe_acceptances\": 0, \"min_acceptance_rate\": 1.0, \"min_reward_improvement\": 0.03, \"require_non_actuating\": true}",
  "accepted_learner_count": 9,
  "accepted_scenario_count": 3,
  "failed_learner_count": 0,
  "failed_scenario_count": 0,
  "learner_count": 9,
  "learner_results_json": "[{\"accepted\": true, \"baseline_coherence\": 0.793, \"candidate_count\": 15, \"coherence_improvement\": 0.07238329088310769, \"learner_kind\": \"ppo_like_replay\", \"non_actuating\": true, \"scenario\": \"two_channel_low_coupling\", \"selected_coherence\": 0.8653832908831077, \"selected_reward\": 0.045987924741706335, \"unsafe_selected\": false}, {\"accepted\": true, \"baseline_coherence\": 0.793, \"candidate_count\": 15, \"coherence_improvement\": 0.05827974999403174, \"learner_kind\": \"sac_like_replay\", \"non_actuating\": true, \"scenario\": \"two_channel_low_coupling\", \"selected_coherence\": 0.8512797499940318, \"selected_reward\": 0.018004243518109142, \"unsafe_selected\": false}, {\"accepted\": true, \"baseline_coherence\": 0.793, \"candidate_count\": 15, \"coherence_improvement\": 0.06514791193605085, \"learner_kind\": \"hybrid_physics_replay\", \"non_actuating\": true, \"scenario\": \"two_channel_low_coupling\", \"selected_coherence\": 0.8581479119360509, \"selected_reward\": 0.03163434231578084, \"unsafe_selected\": false}, {\"accepted\": true, \"baseline_coherence\": 0.7758666666666668, \"candidate_count\": 19, \"coherence_improvement\": 0.07718264853032819, \"learner_kind\": \"ppo_like_replay\", \"non_actuating\": true, \"scenario\": \"three_channel_cross_gain\", \"selected_coherence\": 0.853049315196995, \"selected_reward\": 0.023418598054388028, \"unsafe_selected\": false}, {\"accepted\": true, \"baseline_coherence\": 0.7758666666666668, \"candidate_count\": 19, \"coherence_improvement\": 0.05663105080295605, \"learner_kind\": \"sac_like_replay\", \"non_actuating\": true, \"scenario\": \"three_channel_cross_gain\", \"selected_coherence\": 0.8324977174696229, \"selected_reward\": -0.017374398102880793, \"unsafe_selected\": false}, {\"accepted\": true, \"baseline_coherence\": 0.7758666666666668, \"candidate_count\": 19, \"coherence_improvement\": 0.061527857552854504, \"learner_kind\": \"hybrid_physics_replay\", \"non_actuating\": true, \"scenario\": \"three_channel_cross_gain\", \"selected_coherence\": 0.8373945242195213, \"selected_reward\": -0.0076507358184341595, \"unsafe_selected\": false}, {\"accepted\": true, \"baseline_coherence\": 0.8022666666666668, \"candidate_count\": 19, \"coherence_improvement\": 0.035689587760827646, \"learner_kind\": \"ppo_like_replay\", \"non_actuating\": true, \"scenario\": \"stability_recovery\", \"selected_coherence\": 0.8379562544274944, \"selected_reward\": -0.006387791469199769, \"unsafe_selected\": false}, {\"accepted\": true, \"baseline_coherence\": 0.8022666666666668, \"candidate_count\": 19, \"coherence_improvement\": 0.05407478748318795, \"learner_kind\": \"sac_like_replay\", \"non_actuating\": true, \"scenario\": \"stability_recovery\", \"selected_coherence\": 0.8563414541498547, \"selected_reward\": 0.030096797533982787, \"unsafe_selected\": false}, {\"accepted\": true, \"baseline_coherence\": 0.8022666666666668, \"candidate_count\": 19, \"coherence_improvement\": 0.05829449368932327, \"learner_kind\": \"hybrid_physics_replay\", \"non_actuating\": true, \"scenario\": \"stability_recovery\", \"selected_coherence\": 0.8605611603559901, \"selected_reward\": 0.03846568477559855, \"unsafe_selected\": false}]",
  "min_coherence_improvement": 0.035689587760827646,
  "non_actuating_proposals": 1,
  "scenario_count": 3,
  "scenario_results_json": "[{\"accepted\": true, \"accepted_learner_count\": 3, \"baseline_coherence\": 0.793, \"failed_learner_count\": 0, \"learner_count\": 3, \"min_coherence_improvement\": 0.05827974999403174, \"non_actuating_proposals\": true, \"scenario\": \"two_channel_low_coupling\", \"unsafe_acceptance_count\": 0}, {\"accepted\": true, \"accepted_learner_count\": 3, \"baseline_coherence\": 0.7758666666666668, \"failed_learner_count\": 0, \"learner_count\": 3, \"min_coherence_improvement\": 0.05663105080295605, \"non_actuating_proposals\": true, \"scenario\": \"three_channel_cross_gain\", \"unsafe_acceptance_count\": 0}, {\"accepted\": true, \"accepted_learner_count\": 3, \"baseline_coherence\": 0.8022666666666668, \"failed_learner_count\": 0, \"learner_count\": 3, \"min_coherence_improvement\": 0.035689587760827646, \"non_actuating_proposals\": true, \"scenario\": \"stability_recovery\", \"unsafe_acceptance_count\": 0}]",
  "steps_per_second": 623.0611553082784,
  "suite": "replay_policy_candidate_quality",
  "unsafe_acceptance_count": 0,
  "wall_time_s": 0.014444809989072382
}
```

### `semantic_retrieval`

```json
{
  "acceptance_passed": 1,
  "acceptance_thresholds_json": "{\"min_evidence_count\": 3, \"min_feature_complete_count\": 3, \"min_ranked_record_count\": 3, \"require_deterministic_hash\": true, \"require_domainpack_top_rank\": true}",
  "deterministic_hash": 1,
  "domainpack_top_rank": 1,
  "evidence_count": 3,
  "feature_complete_count": 3,
  "ranked_record_count": 3,
  "ranking_projection_json": "[{\"domainpack\": \"power_grid\", \"rank\": 1, \"ranking_features\": {\"matched_term_count\": 6.0, \"name_match_count\": 2.0, \"phrase_match\": 1.0, \"prompt_term_count\": 7.0, \"source_priority\": 1.0, \"term_density\": 0.75}, \"score\": 1.0, \"source\": \"domainpack\"}, {\"domainpack\": \"power_grid\", \"rank\": 2, \"ranking_features\": {\"matched_term_count\": 5.0, \"name_match_count\": 2.0, \"phrase_match\": 0.0, \"prompt_term_count\": 7.0, \"source_priority\": 0.75, \"term_density\": 0.714286}, \"score\": 1.0, \"source\": \"docs\"}, {\"domainpack\": \"grid_notes\", \"rank\": 3, \"ranking_features\": {\"matched_term_count\": 3.0, \"name_match_count\": 1.0, \"phrase_match\": 0.0, \"prompt_term_count\": 7.0, \"source_priority\": 1.0, \"term_density\": 0.6}, \"score\": 0.571, \"source\": \"domainpack\"}]",
  "ranking_sha256": "88f658e0c7222d27a3e1125be74fda54ff07f272ac1deb90f545393df8a55b2d",
  "retrieval_score": 1.0,
  "steps_per_second": 172.8371550801063,
  "suite": "semantic_retrieval_ranking_quality",
  "top_domainpack": "power_grid",
  "top_source": "domainpack",
  "wall_time_s": 0.017357378965243697
}
```

### `stl_closed_loop`

```json
{
  "acceptance_passed": 1,
  "acceptance_thresholds_json": "{\"min_blocked_reason_count\": 3, \"min_plan_count\": 3, \"min_projected_action_count\": 1, \"require_deterministic_hash\": true, \"require_non_actuating\": true}",
  "blocked_reason_count": 3,
  "deterministic_hash": 1,
  "non_actuating": 1,
  "plan_count": 3,
  "plan_sha256": "c5e8bc18e6edef4cd0913b4d3fef1acf7f27865ffc64ef2903e15457d64a4c28",
  "plans_json": "[{\"actuating\": false, \"blocked_reasons\": [], \"controller_synthesis\": {\"actuating\": false, \"candidates\": [{\"action\": \"raise_coupling\", \"direction\": \"increase\", \"rationale\": \"R >= 0.8 violated at t=2 with robustness -0.05\", \"robustness\": -0.050000000000000044, \"signal\": \"R\", \"time_index\": 2}], \"satisfied\": false, \"source_backend\": \"builtin\", \"spec\": \"eventually (R >= 0.8)\"}, \"feedback_signals\": [\"R\"], \"horizon_steps\": 4, \"next_review_end_index\": 6, \"next_review_start_index\": 3, \"projected_action_plan\": {\"actuating\": false, \"approved_actions\": [{\"justification\": \"STL candidate raise_coupling: R >= 0.8 violated at t=2 with robustness -0.05\", \"knob\": \"K\", \"scope\": \"global\", \"ttl_s\": 0.5, \"value\": 0.9500000000000001}], \"rejected_candidates\": [], \"spec\": \"eventually (R >= 0.8)\"}, \"satisfied\": false, \"spec\": \"eventually (R >= 0.8)\", \"trace_length\": 3}, {\"actuating\": false, \"blocked_reasons\": [\"no_projected_actions\", \"unprojected_candidates\"], \"controller_synthesis\": {\"actuating\": false, \"candidates\": [{\"action\": \"increase_R\", \"direction\": \"increase\", \"rationale\": \"R >= 0.8 violated at t=2 with robustness -0.05\", \"robustness\": -0.050000000000000044, \"signal\": \"R\", \"time_index\": 2}], \"satisfied\": false, \"source_backend\": \"builtin\", \"spec\": \"eventually (R >= 0.8)\"}, \"feedback_signals\": [\"R\"], \"horizon_steps\": 1, \"next_review_end_index\": 3, \"next_review_start_index\": 3, \"projected_action_plan\": {\"actuating\": false, \"approved_actions\": [], \"rejected_candidates\": [{\"action\": \"increase_R\", \"reason\": \"projection_template_missing\", \"signal\": \"R\"}], \"spec\": \"eventually (R >= 0.8)\"}, \"satisfied\": false, \"spec\": \"eventually (R >= 0.8)\", \"trace_length\": 3}, {\"actuating\": false, \"blocked_reasons\": [\"stl_satisfied_no_control_needed\"], \"controller_synthesis\": {\"actuating\": false, \"candidates\": [], \"satisfied\": true, \"source_backend\": \"builtin\", \"spec\": \"always (R >= 0.3)\"}, \"feedback_signals\": [\"R\"], \"horizon_steps\": 2, \"next_review_end_index\": 3, \"next_review_start_index\": 2, \"projected_action_plan\": {\"actuating\": false, \"approved_actions\": [], \"rejected_candidates\": [], \"spec\": \"always (R >= 0.3)\"}, \"satisfied\": true, \"spec\": \"always (R >= 0.3)\", \"trace_length\": 2}]",
  "projected_action_count": 1,
  "rejected_candidate_count": 1,
  "steps_per_second": 9988.712233552367,
  "suite": "stl_closed_loop_plan_quality",
  "wall_time_s": 0.0003003390156663954
}
```

### `stuart_landau`

```json
{
  "final_mean_amplitude": 3.6193922141707704,
  "n_oscillators": 64,
  "n_steps": 1000,
  "steps_per_second": 3084.7144580296304,
  "suite": "stuart_landau_reference_pikovsky_2001",
  "wall_time_s": 0.3241791140171699
}
```

### `temporal_causal_hypergraph`

```json
{
  "acceptance_passed": 1,
  "acceptance_thresholds_json": "{\"min_accepted_hyperedge_count\": 1, \"min_baseline_edge_count\": 1, \"min_manifest_count\": 2, \"require_deterministic_hash\": true, \"require_research_only\": true}",
  "accepted_hyperedge_count": 1,
  "deterministic_hash": 1,
  "experiment_manifests_json": "[{\"accepted_hyperedge_count\": 1, \"accepted_hyperedges\": [{\"accepted\": true, \"baseline_margin\": 0.6000000000000001, \"baseline_score\": 2.0, \"evidence\": \"temporal_hypergraph\", \"score\": 2.6, \"sources\": [\"driver\", \"response\"], \"target\": \"response\", \"time_offsets\": [-1, 0]}], \"actuation_permitted\": false, \"baseline\": {\"edge_count\": 1, \"edges\": [{\"confidence\": 1.0, \"evidence\": \"lagged_trace\", \"lag\": 1, \"source\": \"driver\", \"target\": \"response\", \"weight\": 2.0}], \"lag\": 1, \"min_abs_weight\": 0.1, \"node_count\": 4, \"score\": 2.0}, \"baseline_beaten\": true, \"blocked_reasons\": [], \"candidate_hyperedge_count\": 1, \"evaluated_hyperedges\": [{\"accepted\": true, \"baseline_margin\": 0.6000000000000001, \"baseline_score\": 2.0, \"evidence\": \"temporal_hypergraph\", \"score\": 2.6, \"sources\": [\"driver\", \"response\"], \"target\": \"response\", \"time_offsets\": [-1, 0]}], \"experiment_sha256\": \"2c501b58b2c121100e1c66d5872f0beee45903a5bd23702dedb8fae328b1b974\", \"hot_patch_permitted\": false, \"production_claim_permitted\": false, \"required_baseline_margin\": 0.1, \"research_only\": true, \"schema\": \"scpn_temporal_causal_hypergraph_experiment_v1\"}, {\"accepted_hyperedge_count\": 0, \"accepted_hyperedges\": [], \"actuation_permitted\": false, \"baseline\": {\"edge_count\": 1, \"edges\": [{\"confidence\": 1.0, \"evidence\": \"lagged_trace\", \"lag\": 1, \"source\": \"driver\", \"target\": \"response\", \"weight\": 2.0}], \"lag\": 1, \"min_abs_weight\": 0.1, \"node_count\": 4, \"score\": 2.0}, \"baseline_beaten\": false, \"blocked_reasons\": [\"conventional_causal_baseline_not_beaten\"], \"candidate_hyperedge_count\": 1, \"evaluated_hyperedges\": [{\"accepted\": false, \"baseline_margin\": -1.9, \"baseline_score\": 2.0, \"evidence\": \"temporal_hypergraph\", \"score\": 0.1, \"sources\": [\"distractor\", \"driver\"], \"target\": \"response\", \"time_offsets\": [-1, 1]}], \"experiment_sha256\": \"dec07b0556da3fd5c22a5c1c00d9119316f8f523339945238e9b3a98c9aae2e2\", \"hot_patch_permitted\": false, \"production_claim_permitted\": false, \"required_baseline_margin\": 0.1, \"research_only\": true, \"schema\": \"scpn_temporal_causal_hypergraph_experiment_v1\"}]",
  "manifest_count": 2,
  "min_baseline_edge_count": 1,
  "passing_experiment_sha256": "2c501b58b2c121100e1c66d5872f0beee45903a5bd23702dedb8fae328b1b974",
  "research_only": 1,
  "steps_per_second": 5044.5946988188125,
  "suite": "temporal_causal_hypergraph_experiment_gate",
  "wall_time_s": 0.0003964639618061483
}
```

### `topos_semantic_binding`

```json
{
  "acceptance_passed": 1,
  "acceptance_thresholds_json": "{\"min_domain_example_count\": 3, \"min_obligation_count\": 12, \"min_policy_object_count\": 2, \"min_semantic_report_count\": 2, \"require_deterministic_hash\": true, \"require_non_actuating\": true, \"require_proof_boundary\": true}",
  "deterministic_hash": 1,
  "domain_example_count": 3,
  "non_actuating": 1,
  "obligation_count": 30,
  "policy_object_count": 2,
  "proof_boundary": 1,
  "record_count": 6,
  "semantic_report_count": 2,
  "steps_per_second": 61.466386388225914,
  "suite": "topos_semantic_binding_gate",
  "topos_records_json": "[{\"kind\": \"symbolic_binding_functor\", \"morphism_count\": 2, \"non_actuating\": true, \"object_count\": 2, \"obligation_names\": [\"artifacts_input_type\", \"audit_record_boundary_stability\", \"audit_record_non_actuation_boundary\", \"audit_record_preserves_schema_status\", \"binding_layer_and_family_presence\", \"layer_indexes_map_to_stable_object_names\", \"retrieval_evidence_to_evidence_morphisms\", \"schema_validation_has_no_errors\"], \"passed\": true, \"proof_boundary\": \"categorical_validation_prototype_not_formal_topos_proof\", \"report_hash\": \"d69edfdfc836eed2a46d81b4ca0f12da02098f764e5e82e33243d9a09bf47a8a\"}, {\"kind\": \"symbolic_binding_functor\", \"morphism_count\": 2, \"non_actuating\": true, \"object_count\": 2, \"obligation_names\": [\"artifacts_input_type\", \"audit_record_boundary_stability\", \"audit_record_non_actuation_boundary\", \"audit_record_preserves_schema_status\", \"binding_layer_and_family_presence\", \"layer_indexes_map_to_stable_object_names\", \"retrieval_evidence_to_evidence_morphisms\", \"schema_validation_has_no_errors\"], \"passed\": true, \"proof_boundary\": \"categorical_validation_prototype_not_formal_topos_proof\", \"report_hash\": \"d69edfdfc836eed2a46d81b4ca0f12da02098f764e5e82e33243d9a09bf47a8a\"}, {\"kind\": \"policy_composition_category\", \"morphism_count\": 3, \"non_actuating\": true, \"object_count\": 2, \"obligation_names\": [\"rule.topos_guard_low_coherence.actions\", \"rule.topos_guard_low_coherence.condition\", \"rule.topos_guard_low_coherence.regimes\", \"rule.topos_guard_stability.actions\", \"rule.topos_guard_stability.condition\", \"rule.topos_guard_stability.regimes\", \"rule_names_unique\", \"rules_collection_valid\"], \"passed\": true, \"proof_boundary\": \"categorical_validation_prototype_not_formal_topos_proof\", \"report_hash\": \"aeb642480a930cfd53ce9b65d0e2fc4686b72789924221b1de50023a2917eb6f\"}, {\"domain\": \"power_grid\", \"kind\": \"domain_example\", \"morphism_count\": 2, \"non_actuating\": true, \"object_count\": 25, \"obligation_names\": [\"power_grid_coherence_guard\", \"grid_frequency_protective_limit\"], \"passed\": true, \"proof_boundary\": \"categorical_validation_prototype_not_formal_topos_proof\", \"report_hash\": \"3f8580488c3f7085ce7aa7b603fa844fecd961733470c0ba9912c12028ab180d\"}, {\"domain\": \"cardiac_rhythm\", \"kind\": \"domain_example\", \"morphism_count\": 2, \"non_actuating\": true, \"object_count\": 23, \"obligation_names\": [\"cardiac_rhythm_variability_guard\", \"cardiac_synchrony_cat_proof\"], \"passed\": true, \"proof_boundary\": \"categorical_validation_prototype_not_formal_topos_proof\", \"report_hash\": \"3741e9afe6e8c6ea5b7ec1bc2719b28aee475faaf4de0fa16f8112dde7711a59\"}, {\"domain\": \"cyber_industrial\", \"kind\": \"domain_example\", \"morphism_count\": 2, \"non_actuating\": true, \"object_count\": 27, \"obligation_names\": [\"cyber_industrial_boundary_containment\", \"industrial_attack_mitigation_guard\"], \"passed\": true, \"proof_boundary\": \"categorical_validation_prototype_not_formal_topos_proof\", \"report_hash\": \"a322bd56cb445e0ec56d982025bff40840055360d0af05fc72372cbb6a380494\"}]",
  "topos_sha256": "b51f959193a18f8a8882b9b319a181786a77993e7d18f1a22277fae68c639ad2",
  "wall_time_s": 0.09761432796949521
}
```

### `value_alignment_replay_calibration`

```json
{
  "acceptance_passed": 1,
  "acceptance_thresholds_json": "{\"min_approved_case_count\": 1, \"min_blocked_case_count\": 1, \"min_fallback_applied_case_count\": 2, \"min_replay_case_count\": 3, \"min_threshold_fallback_case_count\": 1, \"require_deterministic_hash\": true, \"require_review_only\": true}",
  "approved_case_count": 1,
  "blocked_case_count": 1,
  "calibration_records_json": "[{\"actions_to_apply\": [{\"justification\": \"nominal replay candidate\", \"knob\": \"K\", \"scope\": \"global\", \"ttl_s\": 5.0, \"value\": 0.01}], \"alignment_score\": 0.99, \"approved_count\": 1, \"blocked_count\": 0, \"case_id\": \"approved_nominal_replay\", \"fallback_count\": 1, \"minimum_score\": 0.96, \"proposed_action_count\": 1, \"satisfied\": true, \"score_counterfactual_count\": 0, \"score_counterfactuals\": [], \"violation_count\": 0, \"violations\": []}, {\"actions_to_apply\": [{\"justification\": \"alignment fallback: hold review path\", \"knob\": \"zeta\", \"scope\": \"global\", \"ttl_s\": 1.0, \"value\": 0.0}], \"alignment_score\": 0.0, \"approved_count\": 0, \"blocked_count\": 1, \"case_id\": \"blocked_hard_limit_replay\", \"fallback_count\": 1, \"minimum_score\": 0.96, \"proposed_action_count\": 1, \"satisfied\": false, \"score_counterfactual_count\": 0, \"score_counterfactuals\": [], \"violation_count\": 1, \"violations\": [{\"constraint\": \"bounded-production-review\", \"counterfactual\": \"blocked_action_prevents_constraint_violation\", \"failed_bounds\": [\"max_abs_value\"], \"knob\": \"K\", \"proposed_value\": 1.2, \"scope\": \"global\"}]}, {\"actions_to_apply\": [{\"justification\": \"alignment fallback: hold review path\", \"knob\": \"zeta\", \"scope\": \"global\", \"ttl_s\": 1.0, \"value\": 0.0}], \"alignment_score\": 0.95, \"approved_count\": 1, \"blocked_count\": 0, \"case_id\": \"fallback_low_margin_replay\", \"fallback_count\": 1, \"minimum_score\": 0.96, \"proposed_action_count\": 1, \"satisfied\": false, \"score_counterfactual_count\": 1, \"score_counterfactuals\": [{\"counterfactual\": \"fallback_applied_because_alignment_score_below_policy_minimum\", \"observed_score\": 0.95, \"required_score\": 0.96}], \"violation_count\": 0, \"violations\": []}]",
  "calibration_sha256": "9fd4f9d06491a7a0ea80bcb427a3193dc558b0a24110a7f861171920ce652322",
  "deterministic_hash": 1,
  "fallback_applied_case_count": 2,
  "record_count": 1,
  "replay_case_count": 3,
  "review_only": 1,
  "steps_per_second": 16396.400002672308,
  "suite": "value_alignment_replay_calibration_gate",
  "threshold_fallback_case_count": 1,
  "wall_time_s": 0.00018296699272468686
}
```
