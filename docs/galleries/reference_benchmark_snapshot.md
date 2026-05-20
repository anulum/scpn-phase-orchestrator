<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Reference Benchmark Snapshot -->

# Reference Benchmark Snapshot

This page publishes a dated benchmark snapshot from the reference suite. Treat
these numbers as historical measurements for the listed environment, not as
fresh validation unless the command is rerun and the JSON artefact is updated.

## Reproduction Metadata

| Field | Value |
|-------|-------|
| Snapshot date | `2026-05-20` |
| Suite version | `reference_suite_v1` |
| Command | `PYTHONPATH=src python benchmarks/reference_suite.py` |
| Backend | `python_numpy` |
| Python | `3.12.3` |
| NumPy | `2.2.6` |
| Platform | `Linux-6.17.0-23-generic-x86_64-with-glibc2.39` |
| Executable | `/media/anulum/724AA8E84AA8AA75/aaa_God_of_the_Math_Collection/03_CODE/SCPN-PHASE-ORCHESTRATOR/.venv/bin/python` |
| JSON artefact | `benchmarks/results/reference_suite.json` |

## Historical Results

| Benchmark key | Suite ID | Record size | Wall time (s) | Steps/s | Acceptance |
|---------------|----------|-------------|---------------|---------|------------|
| `auto_binding` | `auto_binding_synthetic_quality` | 4 | 0.04587485402589664 | 87.19373794065864 | 1 |
| `semantic_retrieval` | `semantic_retrieval_ranking_quality` | n/a | 0.01542349299415946 | 194.50846842126063 | 1 |
| `replay_policy` | `replay_policy_candidate_quality` | n/a | 0.012185700994450599 | 738.5705593874841 | 1 |
| `bayesian_posterior` | `bayesian_posterior_fit_quality` | n/a | 2.2912393439910375 | 41.89872186498886 | 1 |
| `bayesian_backends` | `bayesian_backend_fail_closed` | n/a | 0.2934552769875154 | 10.223022842855984 | 1 |
| `formal_export` | `formal_export_artifact_quality` | n/a | 0.0005464740097522736 | 9149.565964292775 | 1 |
| `stl_closed_loop` | `stl_closed_loop_plan_quality` | n/a | 0.0002731640124693513 | 10982.412993866083 | 1 |
| `domain_formal_export` | `domain_formal_safety_exports` | 3 | 0.000479543989058584 | 18767.82986617836 | 1 |
| `hybrid_cocompiler` | `hybrid_cocompiler_review_gate` | 1 | 0.0001305490150116384 | 7659.958215011046 | 1 |
| `quantum_target_readiness` | `quantum_target_readiness_gate` | 2 | 0.0001195729710161686 | 16726.18805908537 | 1 |
| `neuromorphic_target_readiness` | `neuromorphic_target_readiness_gate` | 2 | 0.00017412198940292 | 11486.200030554328 | 1 |
| `hybrid_target_readiness` | `hybrid_target_readiness_gate` | 2 | 0.00011987402103841305 | 16684.182132833517 | 1 |
| `hybrid_operator_handoff` | `hybrid_operator_handoff_package_gate` | 2 | 0.00012567301746457815 | 15914.315103985742 | 1 |
| `value_alignment_replay_calibration` | `value_alignment_replay_calibration_gate` | 1 | 0.00016536901239305735 | 18141.246395482183 | 1 |
| `autopoietic_lineage` | `autopoietic_lineage_sandbox_gate` | 2 | 0.00033599697053432465 | 14881.086552800369 | 1 |
| `intergenerational_inheritance` | `intergenerational_policy_inheritance_gate` | 2 | 0.0003658159985207021 | 5467.229448924216 | 1 |
| `temporal_causal_hypergraph` | `temporal_causal_hypergraph_experiment_gate` | 2 | 0.0004012970020994544 | 4983.839873053264 | 1 |
| `meta_transfer_corpus` | `meta_transfer_audit_corpus_quality` | 6 | 0.002050206996500492 | 2926.533764757133 | 1 |
| `meta_transfer` | `meta_transfer_package_manifest_quality` | 4 | 0.0004334470140747726 | 9228.348264293205 | 1 |
| `plugin_ecosystem` | `plugin_ecosystem_catalog_quality` | 2 | 0.00028747000033035874 | 10435.871557214383 | 1 |
| `kuramoto` | `kuramoto_reference_strogatz_2000` | 64 | 0.13407328497851267 | 7458.607433690206 | n/a |
| `stuart_landau` | `stuart_landau_reference_pikovsky_2001` | 64 | 0.2681467410293408 | 3729.3013376231165 | n/a |
| `petri_reachability` | `petri_net_reachability` | n/a | 0.01973997103050351 | 253293.17820546284 | n/a |

## Benchmark Record Details

Each record below is copied from the current JSON artefact using stable key ordering. Long JSON-valued fields are preserved as fenced JSON so operators can audit thresholds and evidence without opening the raw artefact first.

### `auto_binding`

- Suite: `auto_binding_synthetic_quality`
- Wall time (s): `0.04587485402589664`
- Steps/s: `87.19373794065864`

| Metric | Value |
|--------|-------|
| `accepted_domain_count` | `4` |
| `domain_acceptance_passed` | `1` |
| `expected_edge_recall` | `1.0` |
| `extractor_coverage` | `1.0` |
| `failed_domain_count` | `0` |
| `fixture_count` | `4` |
| `large_fixture_count` | `4` |
| `max_domain_validation_errors` | `0` |
| `min_domain_expected_edge_recall` | `1.0` |
| `min_domain_extractor_coverage` | `1.0` |
| `min_sample_count` | `128` |
| `proposed_edge_count` | `33` |
| `validation_error_count` | `0` |

`domain_acceptance_results_json`:

```json
[
  {
    "accepted": true,
    "domain": "phase_chain",
    "expected_edge_recall": 1.0,
    "extractor_coverage": 1.0,
    "proposed_edge_count": 6,
    "proposed_edge_multiplier": 6.0,
    "sample_count": 128,
    "source_column_count": 3,
    "validation_error_count": 0
  },
  {
    "accepted": true,
    "domain": "industrial_sensor_chain",
    "expected_edge_recall": 1.0,
    "extractor_coverage": 1.0,
    "proposed_edge_count": 6,
    "proposed_edge_multiplier": 6.0,
    "sample_count": 128,
    "source_column_count": 3,
    "validation_error_count": 0
  },
  {
    "accepted": true,
    "domain": "cardiac_rhythm_surrogate",
    "expected_edge_recall": 1.0,
    "extractor_coverage": 1.0,
    "proposed_edge_count": 9,
    "proposed_edge_multiplier": 4.5,
    "sample_count": 160,
    "source_column_count": 4,
    "validation_error_count": 0
  },
  {
    "accepted": true,
    "domain": "power_grid_surrogate",
    "expected_edge_recall": 1.0,
    "extractor_coverage": 1.0,
    "proposed_edge_count": 12,
    "proposed_edge_multiplier": 6.0,
    "sample_count": 192,
    "source_column_count": 4,
    "validation_error_count": 0
  }
]
```

`domain_acceptance_thresholds_json`:

```json
{
  "cardiac_rhythm_surrogate": {
    "max_proposed_edge_multiplier": 6.0,
    "max_validation_errors": 0,
    "min_expected_edge_recall": 1.0,
    "min_extractor_coverage": 1.0,
    "min_sample_count": 128
  },
  "industrial_sensor_chain": {
    "max_proposed_edge_multiplier": 8.0,
    "max_validation_errors": 0,
    "min_expected_edge_recall": 1.0,
    "min_extractor_coverage": 1.0,
    "min_sample_count": 96
  },
  "phase_chain": {
    "max_proposed_edge_multiplier": 8.0,
    "max_validation_errors": 0,
    "min_expected_edge_recall": 1.0,
    "min_extractor_coverage": 1.0,
    "min_sample_count": 96
  },
  "power_grid_surrogate": {
    "max_proposed_edge_multiplier": 8.0,
    "max_validation_errors": 0,
    "min_expected_edge_recall": 1.0,
    "min_extractor_coverage": 1.0,
    "min_sample_count": 160
  }
}
```

### `semantic_retrieval`

- Suite: `semantic_retrieval_ranking_quality`
- Wall time (s): `0.01542349299415946`
- Steps/s: `194.50846842126063`

| Metric | Value |
|--------|-------|
| `acceptance_passed` | `1` |
| `deterministic_hash` | `1` |
| `domainpack_top_rank` | `1` |
| `evidence_count` | `3` |
| `feature_complete_count` | `3` |
| `ranked_record_count` | `3` |
| `ranking_sha256` | `88f658e0c7222d27a3e1125be74fda54ff07f272ac1deb90f545393df8a55b2d` |
| `retrieval_score` | `1.0` |
| `top_domainpack` | `power_grid` |
| `top_source` | `domainpack` |

`acceptance_thresholds_json`:

```json
{
  "min_evidence_count": 3,
  "min_feature_complete_count": 3,
  "min_ranked_record_count": 3,
  "require_deterministic_hash": true,
  "require_domainpack_top_rank": true
}
```

`ranking_projection_json`:

```json
[
  {
    "domainpack": "power_grid",
    "rank": 1,
    "ranking_features": {
      "matched_term_count": 6.0,
      "name_match_count": 2.0,
      "phrase_match": 1.0,
      "prompt_term_count": 7.0,
      "source_priority": 1.0,
      "term_density": 0.75
    },
    "score": 1.0,
    "source": "domainpack"
  },
  {
    "domainpack": "power_grid",
    "rank": 2,
    "ranking_features": {
      "matched_term_count": 5.0,
      "name_match_count": 2.0,
      "phrase_match": 0.0,
      "prompt_term_count": 7.0,
      "source_priority": 0.75,
      "term_density": 0.714286
    },
    "score": 1.0,
    "source": "docs"
  },
  {
    "domainpack": "grid_notes",
    "rank": 3,
    "ranking_features": {
      "matched_term_count": 3.0,
      "name_match_count": 1.0,
      "phrase_match": 0.0,
      "prompt_term_count": 7.0,
      "source_priority": 1.0,
      "term_density": 0.6
    },
    "score": 0.571,
    "source": "domainpack"
  }
]
```

### `replay_policy`

- Suite: `replay_policy_candidate_quality`
- Wall time (s): `0.012185700994450599`
- Steps/s: `738.5705593874841`

| Metric | Value |
|--------|-------|
| `acceptance_passed` | `1` |
| `acceptance_rate` | `1.0` |
| `accepted_learner_count` | `9` |
| `accepted_scenario_count` | `3` |
| `failed_learner_count` | `0` |
| `failed_scenario_count` | `0` |
| `learner_count` | `9` |
| `min_coherence_improvement` | `0.035689587760827646` |
| `non_actuating_proposals` | `1` |
| `scenario_count` | `3` |
| `unsafe_acceptance_count` | `0` |

`acceptance_thresholds_json`:

```json
{
  "max_unsafe_acceptances": 0,
  "min_acceptance_rate": 1.0,
  "min_reward_improvement": 0.03,
  "require_non_actuating": true
}
```

`learner_results_json`:

```json
[
  {
    "accepted": true,
    "baseline_coherence": 0.793,
    "candidate_count": 15,
    "coherence_improvement": 0.07238329088310769,
    "learner_kind": "ppo_like_replay",
    "non_actuating": true,
    "scenario": "two_channel_low_coupling",
    "selected_coherence": 0.8653832908831077,
    "selected_reward": 0.045987924741706335,
    "unsafe_selected": false
  },
  {
    "accepted": true,
    "baseline_coherence": 0.793,
    "candidate_count": 15,
    "coherence_improvement": 0.05827974999403174,
    "learner_kind": "sac_like_replay",
    "non_actuating": true,
    "scenario": "two_channel_low_coupling",
    "selected_coherence": 0.8512797499940318,
    "selected_reward": 0.018004243518109142,
    "unsafe_selected": false
  },
  {
    "accepted": true,
    "baseline_coherence": 0.793,
    "candidate_count": 15,
    "coherence_improvement": 0.06514791193605085,
    "learner_kind": "hybrid_physics_replay",
    "non_actuating": true,
    "scenario": "two_channel_low_coupling",
    "selected_coherence": 0.8581479119360509,
    "selected_reward": 0.03163434231578084,
    "unsafe_selected": false
  },
  {
    "accepted": true,
    "baseline_coherence": 0.7758666666666668,
    "candidate_count": 19,
    "coherence_improvement": 0.07718264853032819,
    "learner_kind": "ppo_like_replay",
    "non_actuating": true,
    "scenario": "three_channel_cross_gain",
    "selected_coherence": 0.853049315196995,
    "selected_reward": 0.023418598054388028,
    "unsafe_selected": false
  },
  {
    "accepted": true,
    "baseline_coherence": 0.7758666666666668,
    "candidate_count": 19,
    "coherence_improvement": 0.05663105080295605,
    "learner_kind": "sac_like_replay",
    "non_actuating": true,
    "scenario": "three_channel_cross_gain",
    "selected_coherence": 0.8324977174696229,
    "selected_reward": -0.017374398102880793,
    "unsafe_selected": false
  },
  {
    "accepted": true,
    "baseline_coherence": 0.7758666666666668,
    "candidate_count": 19,
    "coherence_improvement": 0.061527857552854504,
    "learner_kind": "hybrid_physics_replay",
    "non_actuating": true,
    "scenario": "three_channel_cross_gain",
    "selected_coherence": 0.8373945242195213,
    "selected_reward": -0.0076507358184341595,
    "unsafe_selected": false
  },
  {
    "accepted": true,
    "baseline_coherence": 0.8022666666666668,
    "candidate_count": 19,
    "coherence_improvement": 0.035689587760827646,
    "learner_kind": "ppo_like_replay",
    "non_actuating": true,
    "scenario": "stability_recovery",
    "selected_coherence": 0.8379562544274944,
    "selected_reward": -0.006387791469199769,
    "unsafe_selected": false
  },
  {
    "accepted": true,
    "baseline_coherence": 0.8022666666666668,
    "candidate_count": 19,
    "coherence_improvement": 0.05407478748318795,
    "learner_kind": "sac_like_replay",
    "non_actuating": true,
    "scenario": "stability_recovery",
    "selected_coherence": 0.8563414541498547,
    "selected_reward": 0.030096797533982787,
    "unsafe_selected": false
  },
  {
    "accepted": true,
    "baseline_coherence": 0.8022666666666668,
    "candidate_count": 19,
    "coherence_improvement": 0.05829449368932327,
    "learner_kind": "hybrid_physics_replay",
    "non_actuating": true,
    "scenario": "stability_recovery",
    "selected_coherence": 0.8605611603559901,
    "selected_reward": 0.03846568477559855,
    "unsafe_selected": false
  }
]
```

`scenario_results_json`:

```json
[
  {
    "accepted": true,
    "accepted_learner_count": 3,
    "baseline_coherence": 0.793,
    "failed_learner_count": 0,
    "learner_count": 3,
    "min_coherence_improvement": 0.05827974999403174,
    "non_actuating_proposals": true,
    "scenario": "two_channel_low_coupling",
    "unsafe_acceptance_count": 0
  },
  {
    "accepted": true,
    "accepted_learner_count": 3,
    "baseline_coherence": 0.7758666666666668,
    "failed_learner_count": 0,
    "learner_count": 3,
    "min_coherence_improvement": 0.05663105080295605,
    "non_actuating_proposals": true,
    "scenario": "three_channel_cross_gain",
    "unsafe_acceptance_count": 0
  },
  {
    "accepted": true,
    "accepted_learner_count": 3,
    "baseline_coherence": 0.8022666666666668,
    "failed_learner_count": 0,
    "learner_count": 3,
    "min_coherence_improvement": 0.035689587760827646,
    "non_actuating_proposals": true,
    "scenario": "stability_recovery",
    "unsafe_acceptance_count": 0
  }
]
```

### `bayesian_posterior`

- Suite: `bayesian_posterior_fit_quality`
- Wall time (s): `2.2912393439910375`
- Steps/s: `41.89872186498886`

| Metric | Value |
|--------|-------|
| `acceptance_passed` | `1` |
| `credible_interval_width` | `0.002121338455159605` |
| `finite_audit_record` | `1` |
| `knm_mean_abs_error` | `0.029439030191471344` |
| `non_negative_coupling` | `1` |
| `omega_mean_abs_error` | `0.007744271156763904` |
| `residual_rmse` | `3.904347277377099e-07` |
| `rollout_sample_count` | `128` |
| `sample_count` | `96` |
| `zero_diagonal_coupling` | `1` |

`acceptance_thresholds_json`:

```json
{
  "max_credible_interval_width": 0.01,
  "max_knm_mean_abs_error": 0.06,
  "max_omega_mean_abs_error": 0.03,
  "max_residual_rmse": 0.0025,
  "min_rollout_sample_count": 96
}
```

### `bayesian_backends`

- Suite: `bayesian_backend_fail_closed`
- Wall time (s): `0.2934552769875154`
- Steps/s: `10.223022842855984`

| Metric | Value |
|--------|-------|
| `acceptance_passed` | `1` |
| `available_backend_count` | `1` |
| `backend_count` | `3` |
| `fail_closed_backend_count` | `2` |
| `numpy_sample_count` | `16` |
| `unexpected_reserved_success_count` | `0` |

`acceptance_thresholds_json`:

```json
{
  "max_unexpected_reserved_successes": 0,
  "min_available_backends": 1,
  "required_fail_closed_backends": [
    "blackjax",
    "numpyro"
  ]
}
```

`backend_results_json`:

```json
[
  {
    "available": true,
    "backend": "numpy",
    "fail_closed": false,
    "kind": "bayesian_backend_status",
    "reason": "executed",
    "sample_count": 16
  },
  {
    "available": false,
    "backend": "numpyro",
    "fail_closed": true,
    "kind": "bayesian_backend_status",
    "reason": "numpyro Bayesian UPDE backend is not implemented; use backend='numpy' for reproducible Monte Carlo propagation",
    "sample_count": 0
  },
  {
    "available": false,
    "backend": "blackjax",
    "fail_closed": true,
    "kind": "bayesian_backend_status",
    "reason": "blackjax Bayesian UPDE backend is not implemented; use backend='numpy' for reproducible Monte Carlo propagation",
    "sample_count": 0
  }
]
```

### `formal_export`

- Suite: `formal_export_artifact_quality`
- Wall time (s): `0.0005464740097522736`
- Steps/s: `9149.565964292775`

| Metric | Value |
|--------|-------|
| `acceptance_passed` | `1` |
| `artifact_count` | `5` |
| `artifact_sha256` | `74217fecfc92b3cf0d3d87f7b58c4278d4c758a1309f11c6d99bac429a57e378` |
| `checker_availability_count` | `3` |
| `checker_availability_execution_disabled` | `1` |
| `checker_available_count` | `2` |
| `checker_command_count` | `3` |
| `checker_execution_disabled` | `1` |
| `checker_missing_count` | `1` |
| `deterministic_hash` | `1` |
| `fail_closed_count` | `5` |
| `identifier_map_count` | `22` |
| `package_property_count` | `3` |
| `package_sha256` | `b1d5207b71b84ecc674b0d203206371f0861bd8cc03667592dfa060bac171a92` |
| `petri_prism_bytes` | `1012` |
| `petri_tla_bytes` | `1281` |
| `policy_prism_bytes` | `1116` |
| `policy_tla_bytes` | `1370` |
| `stl_prism_bytes` | `808` |

`acceptance_thresholds_json`:

```json
{
  "min_artifact_count": 5,
  "min_checker_availability_count": 3,
  "min_checker_command_count": 3,
  "min_fail_closed_count": 4,
  "min_identifier_map_count": 12,
  "min_missing_checker_count": 1,
  "min_package_property_count": 3,
  "require_checker_execution_disabled": true,
  "require_deterministic_hash": true
}
```

`checker_availability_json`:

```json
[
  {
    "artifact_name": "petri_prism",
    "available": true,
    "checker": "prism",
    "command": [
      "prism",
      "petri_prism.prism",
      "-pf",
      "P>=1 [ F \"active_done\" ]"
    ],
    "executable": "prism",
    "execution_permitted": false,
    "property_name": "petri_reaches_done",
    "resolved_path": "/opt/prism/bin/prism",
    "status": "ready_not_executed"
  },
  {
    "artifact_name": "petri_tla",
    "available": false,
    "checker": "tlc",
    "command": [
      "tlc2.TLC",
      "petri_tla.tla",
      "-config",
      "petri_tla.cfg"
    ],
    "executable": "tlc2.TLC",
    "execution_permitted": false,
    "property_name": "petri_type_ok",
    "resolved_path": null,
    "status": "missing_executable"
  },
  {
    "artifact_name": "policy_prism",
    "available": true,
    "checker": "prism",
    "command": [
      "prism",
      "policy_prism.prism",
      "-pf",
      "P>=1 [ F \"fires_boost_K\" ]"
    ],
    "executable": "prism",
    "execution_permitted": false,
    "property_name": "policy_boost_fires",
    "resolved_path": "/opt/prism/bin/prism",
    "status": "ready_not_executed"
  }
]
```

`checker_commands_json`:

```json
[
  {
    "artifact_name": "petri_prism",
    "checker": "prism",
    "command": [
      "prism",
      "petri_prism.prism",
      "-pf",
      "P>=1 [ F \"active_done\" ]"
    ],
    "execution_permitted": false,
    "property_name": "petri_reaches_done"
  },
  {
    "artifact_name": "petri_tla",
    "checker": "tlc",
    "command": [
      "tlc2.TLC",
      "petri_tla.tla",
      "-config",
      "petri_tla.cfg"
    ],
    "execution_permitted": false,
    "property_name": "petri_type_ok"
  },
  {
    "artifact_name": "policy_prism",
    "checker": "prism",
    "command": [
      "prism",
      "policy_prism.prism",
      "-pf",
      "P>=1 [ F \"fires_boost_K\" ]"
    ],
    "execution_permitted": false,
    "property_name": "policy_boost_fires"
  }
]
```

### `stl_closed_loop`

- Suite: `stl_closed_loop_plan_quality`
- Wall time (s): `0.0002731640124693513`
- Steps/s: `10982.412993866083`

| Metric | Value |
|--------|-------|
| `acceptance_passed` | `1` |
| `blocked_reason_count` | `3` |
| `deterministic_hash` | `1` |
| `non_actuating` | `1` |
| `plan_count` | `3` |
| `plan_sha256` | `c5e8bc18e6edef4cd0913b4d3fef1acf7f27865ffc64ef2903e15457d64a4c28` |
| `projected_action_count` | `1` |
| `rejected_candidate_count` | `1` |

`acceptance_thresholds_json`:

```json
{
  "min_blocked_reason_count": 3,
  "min_plan_count": 3,
  "min_projected_action_count": 1,
  "require_deterministic_hash": true,
  "require_non_actuating": true
}
```

`plans_json`:

```json
[
  {
    "actuating": false,
    "blocked_reasons": [],
    "controller_synthesis": {
      "actuating": false,
      "candidates": [
        {
          "action": "raise_coupling",
          "direction": "increase",
          "rationale": "R >= 0.8 violated at t=2 with robustness -0.05",
          "robustness": -0.050000000000000044,
          "signal": "R",
          "time_index": 2
        }
      ],
      "satisfied": false,
      "source_backend": "builtin",
      "spec": "eventually (R >= 0.8)"
    },
    "feedback_signals": [
      "R"
    ],
    "horizon_steps": 4,
    "next_review_end_index": 6,
    "next_review_start_index": 3,
    "projected_action_plan": {
      "actuating": false,
      "approved_actions": [
        {
          "justification": "STL candidate raise_coupling: R >= 0.8 violated at t=2 with robustness -0.05",
          "knob": "K",
          "scope": "global",
          "ttl_s": 0.5,
          "value": 0.9500000000000001
        }
      ],
      "rejected_candidates": [],
      "spec": "eventually (R >= 0.8)"
    },
    "satisfied": false,
    "spec": "eventually (R >= 0.8)",
    "trace_length": 3
  },
  {
    "actuating": false,
    "blocked_reasons": [
      "no_projected_actions",
      "unprojected_candidates"
    ],
    "controller_synthesis": {
      "actuating": false,
      "candidates": [
        {
          "action": "increase_R",
          "direction": "increase",
          "rationale": "R >= 0.8 violated at t=2 with robustness -0.05",
          "robustness": -0.050000000000000044,
          "signal": "R",
          "time_index": 2
        }
      ],
      "satisfied": false,
      "source_backend": "builtin",
      "spec": "eventually (R >= 0.8)"
    },
    "feedback_signals": [
      "R"
    ],
    "horizon_steps": 1,
    "next_review_end_index": 3,
    "next_review_start_index": 3,
    "projected_action_plan": {
      "actuating": false,
      "approved_actions": [],
      "rejected_candidates": [
        {
          "action": "increase_R",
          "reason": "projection_template_missing",
          "signal": "R"
        }
      ],
      "spec": "eventually (R >= 0.8)"
    },
    "satisfied": false,
    "spec": "eventually (R >= 0.8)",
    "trace_length": 3
  },
  {
    "actuating": false,
    "blocked_reasons": [
      "stl_satisfied_no_control_needed"
    ],
    "controller_synthesis": {
      "actuating": false,
      "candidates": [],
      "satisfied": true,
      "source_backend": "builtin",
      "spec": "always (R >= 0.3)"
    },
    "feedback_signals": [
      "R"
    ],
    "horizon_steps": 2,
    "next_review_end_index": 3,
    "next_review_start_index": 2,
    "projected_action_plan": {
      "actuating": false,
      "approved_actions": [],
      "rejected_candidates": [],
      "spec": "always (R >= 0.3)"
    },
    "satisfied": true,
    "spec": "always (R >= 0.3)",
    "trace_length": 2
  }
]
```

### `domain_formal_export`

- Suite: `domain_formal_safety_exports`
- Wall time (s): `0.000479543989058584`
- Steps/s: `18767.82986617836`

| Metric | Value |
|--------|-------|
| `acceptance_passed` | `1` |
| `accepted_domain_count` | `3` |
| `artifact_count` | `9` |
| `artifact_sha256` | `ca29f17d051e8206fcd9b7a56063a79dd6e6d16746b7ce800482e4e7297c504b` |
| `domain_count` | `3` |
| `failed_domain_count` | `0` |

`acceptance_thresholds_json`:

```json
{
  "min_artifacts_per_domain": 3,
  "min_domain_count": 3,
  "min_rules_per_domain": 2,
  "min_stl_specs_per_domain": 2,
  "require_deterministic_hash": true
}
```

`domain_results_json`:

```json
[
  {
    "accepted": true,
    "artifact_count": 3,
    "deterministic_hash": 1,
    "domain": "plasma_control",
    "identifier_map_count": 12,
    "required_labels_present": true,
    "rule_count": 2,
    "stl_spec_count": 2
  },
  {
    "accepted": true,
    "artifact_count": 3,
    "deterministic_hash": 1,
    "domain": "power_grid",
    "identifier_map_count": 12,
    "required_labels_present": true,
    "rule_count": 2,
    "stl_spec_count": 2
  },
  {
    "accepted": true,
    "artifact_count": 3,
    "deterministic_hash": 1,
    "domain": "medical_cardiac",
    "identifier_map_count": 12,
    "required_labels_present": true,
    "rule_count": 2,
    "stl_spec_count": 2
  }
]
```

### `hybrid_cocompiler`

- Suite: `hybrid_cocompiler_review_gate`
- Wall time (s): `0.0001305490150116384`
- Steps/s: `7659.958215011046`

| Metric | Value |
|--------|-------|
| `acceptance_passed` | `1` |
| `blocked_probe_count` | `2` |
| `component_hash_count` | `3` |
| `deterministic_hash` | `1` |
| `hybrid_manifest_sha256` | `e5510f11f3339e62ad54b723a53e737835b5c5c4d2a0274f3539533099073fa7` |
| `manifest_count` | `1` |
| `neuromorphic_sample_count` | `2` |
| `non_actuating` | `1` |
| `quantum_term_count` | `3` |
| `target_backend_count` | `4` |

`acceptance_thresholds_json`:

```json
{
  "min_blocked_probe_count": 2,
  "min_neuromorphic_sample_count": 2,
  "min_quantum_term_count": 3,
  "min_target_backend_count": 4,
  "require_non_actuating": true
}
```

`target_backends_json`:

```json
[
  "qiskit_openqasm3",
  "pennylane_qasm",
  "lava",
  "pynn"
]
```

### `quantum_target_readiness`

- Suite: `quantum_target_readiness_gate`
- Wall time (s): `0.0001195729710161686`
- Steps/s: `16726.18805908537`

| Metric | Value |
|--------|-------|
| `acceptance_passed` | `1` |
| `blocked_count` | `1` |
| `blocked_reason_count` | `2` |
| `deterministic_hash` | `1` |
| `manifest_sha256` | `e323283dbcdc138915a6d2a9728fdcce9dfa9600245428298d60c21b3a5ac30d` |
| `non_executing` | `1` |
| `operator_command_count` | `6` |
| `ready_count` | `1` |
| `ready_readiness_sha256` | `aa0f85ce5bbfd35acf04d96e29d3bb64edf7ce5b091193263b13712d98f6134c` |
| `record_count` | `2` |

`acceptance_thresholds_json`:

```json
{
  "min_blocked_count": 1,
  "min_blocked_reason_count": 2,
  "min_operator_command_count": 6,
  "min_ready_count": 1,
  "require_deterministic_hash": true,
  "require_non_executing": true
}
```

`readiness_records_json`:

```json
[
  {
    "actuation_permitted": false,
    "blocked_reasons": [
      "credentials_not_configured",
      "operator_approval_missing"
    ],
    "credentials_configured": false,
    "manifest_sha256": "e323283dbcdc138915a6d2a9728fdcce9dfa9600245428298d60c21b3a5ac30d",
    "operator_approved": false,
    "operator_commands": [
      "review quantum_compiler_manifest.json",
      "run simulator parity outside SPO before target handoff",
      "submit QPU job only from an approved external operator workflow"
    ],
    "provider": "ibm_quantum",
    "qpu_execution_permitted": false,
    "readiness_sha256": "c3b3fed3ff885d7b3738e2d46914be184671614cbb09bf5ba4c52be728fc875d",
    "schema": "scpn_quantum_target_readiness_v1",
    "status": "blocked",
    "target_backend": "qiskit_openqasm3"
  },
  {
    "actuation_permitted": false,
    "blocked_reasons": [],
    "credentials_configured": true,
    "manifest_sha256": "e323283dbcdc138915a6d2a9728fdcce9dfa9600245428298d60c21b3a5ac30d",
    "operator_approved": true,
    "operator_commands": [
      "review quantum_compiler_manifest.json",
      "run simulator parity outside SPO before target handoff",
      "submit QPU job only from an approved external operator workflow"
    ],
    "provider": "pennylane",
    "qpu_execution_permitted": false,
    "readiness_sha256": "aa0f85ce5bbfd35acf04d96e29d3bb64edf7ce5b091193263b13712d98f6134c",
    "schema": "scpn_quantum_target_readiness_v1",
    "status": "ready_not_executed",
    "target_backend": "pennylane_qasm"
  }
]
```

`target_backends_json`:

```json
[
  "qiskit_openqasm3",
  "pennylane_qasm"
]
```

### `neuromorphic_target_readiness`

- Suite: `neuromorphic_target_readiness_gate`
- Wall time (s): `0.00017412198940292`
- Steps/s: `11486.200030554328`

| Metric | Value |
|--------|-------|
| `acceptance_passed` | `1` |
| `blocked_count` | `1` |
| `blocked_reason_count` | `3` |
| `deterministic_hash` | `1` |
| `manifest_sha256` | `b6d66744f1488a0711c5b40a7fef273c3ab81e8a8eb030ab873e4e75f831600a` |
| `non_executing` | `1` |
| `operator_command_count` | `6` |
| `ready_count` | `1` |
| `ready_readiness_sha256` | `fbff4ea82152b5fb51733f179661b8ec117b1afc076c80972e610af7717368d0` |
| `record_count` | `2` |

`acceptance_thresholds_json`:

```json
{
  "min_blocked_count": 1,
  "min_blocked_reason_count": 3,
  "min_operator_command_count": 6,
  "min_ready_count": 1,
  "require_deterministic_hash": true,
  "require_non_executing": true
}
```

`readiness_records_json`:

```json
[
  {
    "actuation_permitted": false,
    "blocked_reasons": [
      "credentials_not_configured",
      "operator_approval_missing",
      "external_simulator_parity_not_verified"
    ],
    "credentials_configured": false,
    "external_simulator_parity_verified": false,
    "hardware_site": "lab_lava_cluster",
    "hardware_write_permitted": false,
    "manifest_sha256": "b6d66744f1488a0711c5b40a7fef273c3ab81e8a8eb030ab873e4e75f831600a",
    "operator_approved": false,
    "operator_commands": [
      "review neuromorphic_schedule_manifest.json",
      "run target simulator parity outside SPO before hardware handoff",
      "submit neuromorphic hardware job only from an approved operator workflow"
    ],
    "readiness_sha256": "b766c4400035c3f63fa06d2b5ef34d8aca57daf845eb3c5a201106a03f62aa7f",
    "schema": "scpn_neuromorphic_target_readiness_v1",
    "status": "blocked",
    "target_backend": "lava"
  },
  {
    "actuation_permitted": false,
    "blocked_reasons": [],
    "credentials_configured": true,
    "external_simulator_parity_verified": true,
    "hardware_site": "brainscales_review_lane",
    "hardware_write_permitted": false,
    "manifest_sha256": "b6d66744f1488a0711c5b40a7fef273c3ab81e8a8eb030ab873e4e75f831600a",
    "operator_approved": true,
    "operator_commands": [
      "review neuromorphic_schedule_manifest.json",
      "run target simulator parity outside SPO before hardware handoff",
      "submit neuromorphic hardware job only from an approved operator workflow"
    ],
    "readiness_sha256": "fbff4ea82152b5fb51733f179661b8ec117b1afc076c80972e610af7717368d0",
    "schema": "scpn_neuromorphic_target_readiness_v1",
    "status": "ready_not_executed",
    "target_backend": "pynn"
  }
]
```

`target_backends_json`:

```json
[
  "lava",
  "pynn"
]
```

### `hybrid_target_readiness`

- Suite: `hybrid_target_readiness_gate`
- Wall time (s): `0.00011987402103841305`
- Steps/s: `16684.182132833517`

| Metric | Value |
|--------|-------|
| `acceptance_passed` | `1` |
| `blocked_count` | `1` |
| `blocked_reason_count` | `1` |
| `component_hash_linked` | `1` |
| `deterministic_hash` | `1` |
| `hybrid_manifest_sha256` | `e5510f11f3339e62ad54b723a53e737835b5c5c4d2a0274f3539533099073fa7` |
| `non_executing` | `1` |
| `operator_command_count` | `6` |
| `ready_count` | `1` |
| `ready_readiness_sha256` | `5dbf280c524594e46047c0fb342383df767713c6ebdd71d43eb5fdc5a0b5cc64` |
| `record_count` | `2` |

`acceptance_thresholds_json`:

```json
{
  "min_blocked_count": 1,
  "min_blocked_reason_count": 1,
  "min_operator_command_count": 6,
  "min_ready_count": 1,
  "require_component_hash_linked": true,
  "require_deterministic_hash": true,
  "require_non_executing": true
}
```

`readiness_records_json`:

```json
[
  {
    "actuation_permitted": false,
    "blocked_reasons": [
      "hybrid_operator_approval_missing"
    ],
    "component_manifest_hashes": {
      "neuromorphic_schedule_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
      "quantum_manifest_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    },
    "component_statuses": {
      "hybrid": "co_simulation_parity_passed",
      "neuromorphic": "ready_not_executed",
      "quantum": "ready_not_executed"
    },
    "hardware_write_permitted": false,
    "hybrid_manifest_sha256": "e5510f11f3339e62ad54b723a53e737835b5c5c4d2a0274f3539533099073fa7",
    "hybrid_operator_approved": false,
    "neuromorphic_readiness_sha256": "c0aa614538ae1f3e971ea369f399c83d092fae7708aa965dd27bac995e5bfd4c",
    "operator_commands": [
      "review hybrid_neuromorphic_quantum_cocompiler.json",
      "verify quantum and neuromorphic readiness hashes before handoff",
      "submit hybrid execution only from an approved external operator workflow"
    ],
    "qpu_execution_permitted": false,
    "quantum_readiness_sha256": "b7dc2e801b17cd78c8b4af4382c4d636dd77c4b958c3cecdde4938933d1f9475",
    "readiness_sha256": "67f184c426a68a3c63a8e7175f0dbdf09e941b6070c93283ea486a3d5fef735d",
    "schema": "scpn_hybrid_target_readiness_v1",
    "status": "blocked"
  },
  {
    "actuation_permitted": false,
    "blocked_reasons": [],
    "component_manifest_hashes": {
      "neuromorphic_schedule_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
      "quantum_manifest_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    },
    "component_statuses": {
      "hybrid": "co_simulation_parity_passed",
      "neuromorphic": "ready_not_executed",
      "quantum": "ready_not_executed"
    },
    "hardware_write_permitted": false,
    "hybrid_manifest_sha256": "e5510f11f3339e62ad54b723a53e737835b5c5c4d2a0274f3539533099073fa7",
    "hybrid_operator_approved": true,
    "neuromorphic_readiness_sha256": "c0aa614538ae1f3e971ea369f399c83d092fae7708aa965dd27bac995e5bfd4c",
    "operator_commands": [
      "review hybrid_neuromorphic_quantum_cocompiler.json",
      "verify quantum and neuromorphic readiness hashes before handoff",
      "submit hybrid execution only from an approved external operator workflow"
    ],
    "qpu_execution_permitted": false,
    "quantum_readiness_sha256": "b7dc2e801b17cd78c8b4af4382c4d636dd77c4b958c3cecdde4938933d1f9475",
    "readiness_sha256": "5dbf280c524594e46047c0fb342383df767713c6ebdd71d43eb5fdc5a0b5cc64",
    "schema": "scpn_hybrid_target_readiness_v1",
    "status": "ready_not_executed"
  }
]
```

### `hybrid_operator_handoff`

- Suite: `hybrid_operator_handoff_package_gate`
- Wall time (s): `0.00012567301746457815`
- Steps/s: `15914.315103985742`

| Metric | Value |
|--------|-------|
| `acceptance_passed` | `1` |
| `blocked_package_count` | `1` |
| `blocked_reason_count` | `1` |
| `deterministic_hash` | `1` |
| `hash_chain_linked` | `1` |
| `non_executing` | `1` |
| `operator_command_count` | `8` |
| `package_count` | `2` |
| `ready_package_count` | `1` |
| `ready_package_sha256` | `c742f1c3a2ba7bfad9e1266f743c43120df8cf5c9e12da865dbcf0436b879eb5` |

`acceptance_thresholds_json`:

```json
{
  "min_blocked_package_count": 1,
  "min_blocked_reason_count": 1,
  "min_operator_command_count": 8,
  "min_ready_package_count": 1,
  "require_deterministic_hash": true,
  "require_hash_chain_linked": true,
  "require_non_executing": true
}
```

`packages_json`:

```json
[
  {
    "actuation_permitted": false,
    "blocked_reasons": [
      "hybrid_operator_approval_missing"
    ],
    "component_manifest_hashes": {
      "neuromorphic_schedule_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
      "quantum_manifest_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      "quantum_qasm_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    },
    "component_statuses": {
      "hybrid": "co_simulation_parity_passed",
      "neuromorphic": "ready_not_executed",
      "quantum": "ready_not_executed"
    },
    "execution_permitted": false,
    "hardware_write_permitted": false,
    "hybrid_manifest_sha256": "e5510f11f3339e62ad54b723a53e737835b5c5c4d2a0274f3539533099073fa7",
    "hybrid_readiness_sha256": "67f184c426a68a3c63a8e7175f0dbdf09e941b6070c93283ea486a3d5fef735d",
    "operator_commands": [
      "review hybrid_neuromorphic_quantum_cocompiler.json",
      "review scpn_hybrid_target_readiness_v1.json",
      "verify package_sha256 before external operator handoff",
      "execute only outside SPO from an approved operator workflow"
    ],
    "package_sha256": "ad392e2aae056e4a5e673d00dcf93f166f32f39db53c5c6cbf8e8ab2678f9afd",
    "qpu_execution_permitted": false,
    "schema": "scpn_hybrid_operator_handoff_package_v1",
    "status": "blocked",
    "target_backends": [
      "qiskit_openqasm3",
      "pennylane_qasm",
      "lava",
      "pynn"
    ]
  },
  {
    "actuation_permitted": false,
    "blocked_reasons": [],
    "component_manifest_hashes": {
      "neuromorphic_schedule_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
      "quantum_manifest_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      "quantum_qasm_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    },
    "component_statuses": {
      "hybrid": "co_simulation_parity_passed",
      "neuromorphic": "ready_not_executed",
      "quantum": "ready_not_executed"
    },
    "execution_permitted": false,
    "hardware_write_permitted": false,
    "hybrid_manifest_sha256": "e5510f11f3339e62ad54b723a53e737835b5c5c4d2a0274f3539533099073fa7",
    "hybrid_readiness_sha256": "5dbf280c524594e46047c0fb342383df767713c6ebdd71d43eb5fdc5a0b5cc64",
    "operator_commands": [
      "review hybrid_neuromorphic_quantum_cocompiler.json",
      "review scpn_hybrid_target_readiness_v1.json",
      "verify package_sha256 before external operator handoff",
      "execute only outside SPO from an approved operator workflow"
    ],
    "package_sha256": "c742f1c3a2ba7bfad9e1266f743c43120df8cf5c9e12da865dbcf0436b879eb5",
    "qpu_execution_permitted": false,
    "schema": "scpn_hybrid_operator_handoff_package_v1",
    "status": "ready_not_executed",
    "target_backends": [
      "qiskit_openqasm3",
      "pennylane_qasm",
      "lava",
      "pynn"
    ]
  }
]
```

### `value_alignment_replay_calibration`

- Suite: `value_alignment_replay_calibration_gate`
- Wall time (s): `0.00016536901239305735`
- Steps/s: `18141.246395482183`

| Metric | Value |
|--------|-------|
| `acceptance_passed` | `1` |
| `approved_case_count` | `1` |
| `blocked_case_count` | `1` |
| `calibration_sha256` | `9fd4f9d06491a7a0ea80bcb427a3193dc558b0a24110a7f861171920ce652322` |
| `deterministic_hash` | `1` |
| `fallback_applied_case_count` | `2` |
| `record_count` | `1` |
| `replay_case_count` | `3` |
| `review_only` | `1` |
| `threshold_fallback_case_count` | `1` |

`acceptance_thresholds_json`:

```json
{
  "min_approved_case_count": 1,
  "min_blocked_case_count": 1,
  "min_fallback_applied_case_count": 2,
  "min_replay_case_count": 3,
  "min_threshold_fallback_case_count": 1,
  "require_deterministic_hash": true,
  "require_review_only": true
}
```

`calibration_records_json`:

```json
[
  {
    "actions_to_apply": [
      {
        "justification": "nominal replay candidate",
        "knob": "K",
        "scope": "global",
        "ttl_s": 5.0,
        "value": 0.01
      }
    ],
    "alignment_score": 0.99,
    "approved_count": 1,
    "blocked_count": 0,
    "case_id": "approved_nominal_replay",
    "fallback_count": 1,
    "minimum_score": 0.96,
    "proposed_action_count": 1,
    "satisfied": true,
    "score_counterfactual_count": 0,
    "score_counterfactuals": [],
    "violation_count": 0,
    "violations": []
  },
  {
    "actions_to_apply": [
      {
        "justification": "alignment fallback: hold review path",
        "knob": "zeta",
        "scope": "global",
        "ttl_s": 1.0,
        "value": 0.0
      }
    ],
    "alignment_score": 0.0,
    "approved_count": 0,
    "blocked_count": 1,
    "case_id": "blocked_hard_limit_replay",
    "fallback_count": 1,
    "minimum_score": 0.96,
    "proposed_action_count": 1,
    "satisfied": false,
    "score_counterfactual_count": 0,
    "score_counterfactuals": [],
    "violation_count": 1,
    "violations": [
      {
        "constraint": "bounded-production-review",
        "counterfactual": "blocked_action_prevents_constraint_violation",
        "failed_bounds": [
          "max_abs_value"
        ],
        "knob": "K",
        "proposed_value": 1.2,
        "scope": "global"
      }
    ]
  },
  {
    "actions_to_apply": [
      {
        "justification": "alignment fallback: hold review path",
        "knob": "zeta",
        "scope": "global",
        "ttl_s": 1.0,
        "value": 0.0
      }
    ],
    "alignment_score": 0.95,
    "approved_count": 1,
    "blocked_count": 0,
    "case_id": "fallback_low_margin_replay",
    "fallback_count": 1,
    "minimum_score": 0.96,
    "proposed_action_count": 1,
    "satisfied": false,
    "score_counterfactual_count": 1,
    "score_counterfactuals": [
      {
        "counterfactual": "fallback_applied_because_alignment_score_below_policy_minimum",
        "observed_score": 0.95,
        "required_score": 0.96
      }
    ],
    "violation_count": 0,
    "violations": []
  }
]
```

### `autopoietic_lineage`

- Suite: `autopoietic_lineage_sandbox_gate`
- Wall time (s): `0.00033599697053432465`
- Steps/s: `14881.086552800369`

| Metric | Value |
|--------|-------|
| `acceptance_passed` | `1` |
| `accepted_child_count` | `3` |
| `child_candidate_count` | `5` |
| `deterministic_hash` | `1` |
| `manifest_count` | `2` |
| `policy_diff_count` | `5` |
| `rejected_child_count` | `2` |
| `review_only` | `1` |
| `safe_lineage_sha256` | `830da8db3a0227d276bb5d8fa97bfe046b10db00c9976338033d64676d1b0ca8` |

`acceptance_thresholds_json`:

```json
{
  "min_accepted_child_count": 3,
  "min_child_candidate_count": 5,
  "min_policy_diff_count": 5,
  "min_rejected_child_count": 2,
  "require_deterministic_hash": true,
  "require_review_only": true
}
```

`lineage_manifests_json`:

```json
[
  {
    "accepted_child_count": 3,
    "actuation_permitted": false,
    "child_budget": 3,
    "child_candidate_count": 3,
    "child_candidates": [
      {
        "actuation_permitted": false,
        "blocked_reasons": [],
        "child_id": "child_001",
        "child_sha256": "f74caf8b231b798def8488ba05136354dc6863b964c930cebf65f5bbb52ad209",
        "live_merge_permitted": false,
        "minimum_replay_reward": 0.7,
        "minimum_safety_margin": 0.1,
        "policy_diff": [
          {
            "child_value": 0.44,
            "delta": 0.02,
            "knob": "K",
            "parent_value": 0.42
          }
        ],
        "replay_reward": 0.78,
        "review_required": true,
        "safety_margin": 0.18,
        "status": "accepted_for_review"
      },
      {
        "actuation_permitted": false,
        "blocked_reasons": [],
        "child_id": "child_002",
        "child_sha256": "7d6987df764895dd182850b39a1aa9bc097cb34c676fab8c5ff37e2e3a380df9",
        "live_merge_permitted": false,
        "minimum_replay_reward": 0.7,
        "minimum_safety_margin": 0.1,
        "policy_diff": [
          {
            "child_value": 0.13999999999999999,
            "delta": -0.04,
            "knob": "alpha",
            "parent_value": 0.18
          }
        ],
        "replay_reward": 0.78,
        "review_required": true,
        "safety_margin": 0.18,
        "status": "accepted_for_review"
      },
      {
        "actuation_permitted": false,
        "blocked_reasons": [],
        "child_id": "child_003",
        "child_sha256": "5c90d8cd64b38f7fb450a3484bd0f7eee6f5a12fac20352e9ad4293b5ddd73fd",
        "live_merge_permitted": false,
        "minimum_replay_reward": 0.7,
        "minimum_safety_margin": 0.1,
        "policy_diff": [
          {
            "child_value": 0.15,
            "delta": 0.06,
            "knob": "zeta",
            "parent_value": 0.09
          }
        ],
        "replay_reward": 0.78,
        "review_required": true,
        "safety_margin": 0.18,
        "status": "accepted_for_review"
      }
    ],
    "hot_patch_permitted": false,
    "lineage_sha256": "830da8db3a0227d276bb5d8fa97bfe046b10db00c9976338033d64676d1b0ca8",
    "live_merge_permitted": false,
    "minimum_replay_reward": 0.7,
    "minimum_safety_margin": 0.1,
    "mutation_step": 0.02,
    "parent_policy_genome": {
      "K": 0.42,
      "alpha": 0.18,
      "zeta": 0.09
    },
    "parent_policy_sha256": "a725a1a906e867e6ae8289fc1bdba209cda78b6a19caf97fe2fdbb5c5965f6a7",
    "rejected_child_count": 0,
    "replay_corpus_sha256": "a98f5ed7bfac6bcc3e1380cf7127e01c7eb7f59b862c2340f1d872bd11b3ce41",
    "replay_summary": {
      "mean_reward": 0.78,
      "mean_safety_margin": 0.21,
      "min_reward": 0.74,
      "min_safety_margin": 0.18,
      "replay_count": 2,
      "violation_count": 0
    },
    "review_required": true,
    "schema": "scpn_autopoietic_lineage_sandbox_v1"
  },
  {
    "accepted_child_count": 0,
    "actuation_permitted": false,
    "child_budget": 2,
    "child_candidate_count": 2,
    "child_candidates": [
      {
        "actuation_permitted": false,
        "blocked_reasons": [
          "replay_reward_below_minimum",
          "safety_margin_below_minimum",
          "replay_violations_present"
        ],
        "child_id": "child_001",
        "child_sha256": "de8639cf4835a60d11c3cbe618904b3dba703b3b38ec1531610324fc5cf0023d",
        "live_merge_permitted": false,
        "minimum_replay_reward": 0.7,
        "minimum_safety_margin": 0.1,
        "policy_diff": [
          {
            "child_value": 0.45999999999999996,
            "delta": 0.04,
            "knob": "K",
            "parent_value": 0.42
          }
        ],
        "replay_reward": 0.3,
        "review_required": true,
        "safety_margin": 0.02,
        "status": "rejected"
      },
      {
        "actuation_permitted": false,
        "blocked_reasons": [
          "replay_reward_below_minimum",
          "safety_margin_below_minimum",
          "replay_violations_present"
        ],
        "child_id": "child_002",
        "child_sha256": "bd2556c2cfcbb507f30b5016b446cdfca12a24f5015f0076a31ad0480a801a08",
        "live_merge_permitted": false,
        "minimum_replay_reward": 0.7,
        "minimum_safety_margin": 0.1,
        "policy_diff": [
          {
            "child_value": 0.09999999999999999,
            "delta": -0.08,
            "knob": "alpha",
            "parent_value": 0.18
          }
        ],
        "replay_reward": 0.3,
        "review_required": true,
        "safety_margin": 0.02,
        "status": "rejected"
      }
    ],
    "hot_patch_permitted": false,
    "lineage_sha256": "1f8de5037ca8057434f945387e7d83c91fc5fea255d4a28d737c5c9dfe46a132",
    "live_merge_permitted": false,
    "minimum_replay_reward": 0.7,
    "minimum_safety_margin": 0.1,
    "mutation_step": 0.04,
    "parent_policy_genome": {
      "K": 0.42,
      "alpha": 0.18,
      "zeta": 0.09
    },
    "parent_policy_sha256": "a725a1a906e867e6ae8289fc1bdba209cda78b6a19caf97fe2fdbb5c5965f6a7",
    "rejected_child_count": 2,
    "replay_corpus_sha256": "6451329e3db6ba6957de16795cc4079d65aedcdc8ba22d266d93f3cae7eb0756",
    "replay_summary": {
      "mean_reward": 0.3,
      "mean_safety_margin": 0.02,
      "min_reward": 0.3,
      "min_safety_margin": 0.02,
      "replay_count": 1,
      "violation_count": 1
    },
    "review_required": true,
    "schema": "scpn_autopoietic_lineage_sandbox_v1"
  }
]
```

### `intergenerational_inheritance`

- Suite: `intergenerational_policy_inheritance_gate`
- Wall time (s): `0.0003658159985207021`
- Steps/s: `5467.229448924216`

| Metric | Value |
|--------|-------|
| `acceptance_passed` | `1` |
| `deterministic_hash` | `1` |
| `inheritance_sha256` | `77c1a624ff98d783e167bcbd4762d5df66b45e362cd3e05bb738fb6c3a49eff2` |
| `manifest_count` | `2` |
| `min_fitness_score` | `0.5720000000000001` |
| `policy_gene_count` | `3` |
| `review_only` | `1` |
| `signed_metadata_count` | `2` |

`acceptance_thresholds_json`:

```json
{
  "min_fitness_score": 0.35,
  "min_manifest_count": 2,
  "min_policy_gene_count": 3,
  "min_signed_metadata_count": 2,
  "require_deterministic_hash": true,
  "require_review_only": true
}
```

`inheritance_manifests_json`:

```json
[
  {
    "actuation_permitted": false,
    "child_sha256": "f74caf8b231b798def8488ba05136354dc6863b964c930cebf65f5bbb52ad209",
    "direct_hot_patch_permitted": false,
    "hot_patch_review_required": true,
    "inheritance_sha256": "77c1a624ff98d783e167bcbd4762d5df66b45e362cd3e05bb738fb6c3a49eff2",
    "inherited_policy_genome": {
      "K": 0.44,
      "alpha": 0.18,
      "zeta": 0.09
    },
    "lineage_sha256": "e3fbe6d49daf1949fa3c6f1cea9c4ad86a065ad867aefeb1818d13ec21e2c5c6",
    "merge_strategy": "reviewed_hot_patch_only",
    "multi_objective_replay_fitness": {
      "fitness_score": 0.5720000000000001,
      "objective_weights": {
        "reward": 0.6,
        "safety": 0.3,
        "simplicity": 0.1
      },
      "reward_component": 0.78,
      "safety_component": 0.18,
      "simplicity_component": 0.5
    },
    "parent_policy_sha256": "a725a1a906e867e6ae8289fc1bdba209cda78b6a19caf97fe2fdbb5c5965f6a7",
    "policy_diff": [
      {
        "child_value": 0.44,
        "delta": 0.02,
        "knob": "K",
        "parent_value": 0.42
      }
    ],
    "schema": "scpn_intergenerational_policy_inheritance_v1",
    "signed_metadata": {
      "signature_algorithm": "hmac-sha256",
      "signature_sha256": "713b0a8203340ce321976e551a016e6b8d2e03a81bef414a7395323329d2fdb1",
      "signer_id": "reference-suite-review-key"
    }
  },
  {
    "actuation_permitted": false,
    "child_sha256": "7d6987df764895dd182850b39a1aa9bc097cb34c676fab8c5ff37e2e3a380df9",
    "direct_hot_patch_permitted": false,
    "hot_patch_review_required": true,
    "inheritance_sha256": "42fa1cdeb427ff0ece034b2d5b70f13f373fbb33ee455d0296d583992cd80064",
    "inherited_policy_genome": {
      "K": 0.42,
      "alpha": 0.13999999999999999,
      "zeta": 0.09
    },
    "lineage_sha256": "e3fbe6d49daf1949fa3c6f1cea9c4ad86a065ad867aefeb1818d13ec21e2c5c6",
    "merge_strategy": "reviewed_hot_patch_only",
    "multi_objective_replay_fitness": {
      "fitness_score": 0.5720000000000001,
      "objective_weights": {
        "reward": 0.6,
        "safety": 0.3,
        "simplicity": 0.1
      },
      "reward_component": 0.78,
      "safety_component": 0.18,
      "simplicity_component": 0.5
    },
    "parent_policy_sha256": "a725a1a906e867e6ae8289fc1bdba209cda78b6a19caf97fe2fdbb5c5965f6a7",
    "policy_diff": [
      {
        "child_value": 0.13999999999999999,
        "delta": -0.04,
        "knob": "alpha",
        "parent_value": 0.18
      }
    ],
    "schema": "scpn_intergenerational_policy_inheritance_v1",
    "signed_metadata": {
      "signature_algorithm": "hmac-sha256",
      "signature_sha256": "54928d9c550a46b03eef8ef9170a215c72f62732639f0f5d41ddf0185846ab4d",
      "signer_id": "reference-suite-review-key"
    }
  }
]
```

### `temporal_causal_hypergraph`

- Suite: `temporal_causal_hypergraph_experiment_gate`
- Wall time (s): `0.0004012970020994544`
- Steps/s: `4983.839873053264`

| Metric | Value |
|--------|-------|
| `acceptance_passed` | `1` |
| `accepted_hyperedge_count` | `1` |
| `deterministic_hash` | `1` |
| `manifest_count` | `2` |
| `min_baseline_edge_count` | `1` |
| `passing_experiment_sha256` | `2c501b58b2c121100e1c66d5872f0beee45903a5bd23702dedb8fae328b1b974` |
| `research_only` | `1` |

`acceptance_thresholds_json`:

```json
{
  "min_accepted_hyperedge_count": 1,
  "min_baseline_edge_count": 1,
  "min_manifest_count": 2,
  "require_deterministic_hash": true,
  "require_research_only": true
}
```

`experiment_manifests_json`:

```json
[
  {
    "accepted_hyperedge_count": 1,
    "accepted_hyperedges": [
      {
        "accepted": true,
        "baseline_margin": 0.6000000000000001,
        "baseline_score": 2.0,
        "evidence": "temporal_hypergraph",
        "score": 2.6,
        "sources": [
          "driver",
          "response"
        ],
        "target": "response",
        "time_offsets": [
          -1,
          0
        ]
      }
    ],
    "actuation_permitted": false,
    "baseline": {
      "edge_count": 1,
      "edges": [
        {
          "confidence": 1.0,
          "evidence": "lagged_trace",
          "lag": 1,
          "source": "driver",
          "target": "response",
          "weight": 2.0
        }
      ],
      "lag": 1,
      "min_abs_weight": 0.1,
      "node_count": 4,
      "score": 2.0
    },
    "baseline_beaten": true,
    "blocked_reasons": [],
    "candidate_hyperedge_count": 1,
    "evaluated_hyperedges": [
      {
        "accepted": true,
        "baseline_margin": 0.6000000000000001,
        "baseline_score": 2.0,
        "evidence": "temporal_hypergraph",
        "score": 2.6,
        "sources": [
          "driver",
          "response"
        ],
        "target": "response",
        "time_offsets": [
          -1,
          0
        ]
      }
    ],
    "experiment_sha256": "2c501b58b2c121100e1c66d5872f0beee45903a5bd23702dedb8fae328b1b974",
    "hot_patch_permitted": false,
    "production_claim_permitted": false,
    "required_baseline_margin": 0.1,
    "research_only": true,
    "schema": "scpn_temporal_causal_hypergraph_experiment_v1"
  },
  {
    "accepted_hyperedge_count": 0,
    "accepted_hyperedges": [],
    "actuation_permitted": false,
    "baseline": {
      "edge_count": 1,
      "edges": [
        {
          "confidence": 1.0,
          "evidence": "lagged_trace",
          "lag": 1,
          "source": "driver",
          "target": "response",
          "weight": 2.0
        }
      ],
      "lag": 1,
      "min_abs_weight": 0.1,
      "node_count": 4,
      "score": 2.0
    },
    "baseline_beaten": false,
    "blocked_reasons": [
      "conventional_causal_baseline_not_beaten"
    ],
    "candidate_hyperedge_count": 1,
    "evaluated_hyperedges": [
      {
        "accepted": false,
        "baseline_margin": -1.9,
        "baseline_score": 2.0,
        "evidence": "temporal_hypergraph",
        "score": 0.1,
        "sources": [
          "distractor",
          "driver"
        ],
        "target": "response",
        "time_offsets": [
          -1,
          1
        ]
      }
    ],
    "experiment_sha256": "dec07b0556da3fd5c22a5c1c00d9119316f8f523339945238e9b3a98c9aae2e2",
    "hot_patch_permitted": false,
    "production_claim_permitted": false,
    "required_baseline_margin": 0.1,
    "research_only": true,
    "schema": "scpn_temporal_causal_hypergraph_experiment_v1"
  }
]
```

### `meta_transfer_corpus`

- Suite: `meta_transfer_audit_corpus_quality`
- Wall time (s): `0.002050206996500492`
- Steps/s: `2926.533764757133`

| Metric | Value |
|--------|-------|
| `acceptance_passed` | `1` |
| `confidence` | `0.9977061617091371` |
| `deterministic_hash` | `1` |
| `domain_count` | `4` |
| `feature_key_count` | `5` |
| `knob_count` | `4` |
| `neighbour_count` | `3` |
| `proposal_knob_count` | `4` |
| `proposal_sha256` | `bfef70f740fbdedc765080f9c9bb0156ec046fd0351dab4f4117c7c259a781fb` |
| `record_count` | `6` |
| `top_neighbour_domain` | `power_grid` |

`acceptance_thresholds_json`:

```json
{
  "min_confidence": 0.97,
  "min_domain_count": 4,
  "min_feature_key_count": 5,
  "min_knob_count": 4,
  "min_neighbour_count": 3,
  "min_record_count": 6,
  "require_deterministic_hash": true,
  "required_top_domain": "power_grid"
}
```

`proposal_json`:

```json
{
  "confidence": 0.9977061617091371,
  "feature_keys": [
    "coherence",
    "event_rate",
    "load_variance",
    "phase_spread",
    "safety_margin"
  ],
  "knobs": {
    "K": 0.41749872361915147,
    "Psi": 0.023185338382920508,
    "alpha": 0.01,
    "zeta": 0.06318533838292051
  },
  "method": "cosine_nearest_policy_transfer",
  "neighbours": [
    {
      "domain": "power_grid",
      "similarity": 0.9999872402392453
    },
    {
      "domain": "power_grid",
      "similarity": 0.9999072410625022
    },
    {
      "domain": "manufacturing_spc",
      "similarity": 0.9932240038256637
    }
  ]
}
```

`training_summary_json`:

```json
{
  "domain_count": 4,
  "domains": [
    "cardiac_rhythm",
    "manufacturing_spc",
    "power_grid",
    "traffic_flow"
  ],
  "feature_keys": [
    "coherence",
    "event_rate",
    "load_variance",
    "phase_spread",
    "safety_margin"
  ],
  "knob_keys": [
    "K",
    "Psi",
    "alpha",
    "zeta"
  ],
  "record_count": 6,
  "reward_max": 0.94,
  "reward_mean": 0.8866666666666667,
  "reward_min": 0.82
}
```

### `meta_transfer`

- Suite: `meta_transfer_package_manifest_quality`
- Wall time (s): `0.0004334470140747726`
- Steps/s: `9228.348264293205`

| Metric | Value |
|--------|-------|
| `acceptance_passed` | `1` |
| `console_script` | `scpn-meta` |
| `deterministic_hash` | `1` |
| `domain_count` | `4` |
| `execution_disabled` | `1` |
| `feature_key_count` | `5` |
| `import_target` | `scpn_phase_orchestrator.meta` |
| `knob_count` | `4` |
| `manifest_schema` | `scpn_meta_package_manifest_v1` |
| `manifest_sha256` | `bf551f9836e581eba6469309784c56e3b3fd5cbfee23e55cee6018f0163df6af` |
| `package_bytes` | `1950` |
| `package_digest_matches` | `1` |
| `package_name` | `scpn-meta` |
| `package_sha256` | `533acf3b37aa233b7a53da1903c99865a7e34055d3d5bcacef3501c3b9fd273f` |
| `record_count` | `4` |

`acceptance_thresholds_json`:

```json
{
  "min_domain_count": 4,
  "min_feature_key_count": 5,
  "min_knob_count": 4,
  "min_record_count": 4,
  "require_deterministic_hash": true,
  "require_execution_disabled": true,
  "require_package_digest_match": true
}
```

`manifest_json`:

```json
{
  "console_script": "scpn-meta",
  "execution_permitted": false,
  "import_target": "scpn_phase_orchestrator.meta",
  "package_name": "scpn-meta",
  "package_sha256": "533acf3b37aa233b7a53da1903c99865a7e34055d3d5bcacef3501c3b9fd273f",
  "schema": "scpn_meta_package_manifest_v1",
  "training_summary": {
    "domain_count": 4,
    "domains": [
      "cardiac_rhythm",
      "manufacturing_spc",
      "power_grid",
      "traffic_flow"
    ],
    "feature_keys": [
      "coherence",
      "event_rate",
      "load_variance",
      "phase_spread",
      "safety_margin"
    ],
    "knob_keys": [
      "K",
      "Psi",
      "alpha",
      "zeta"
    ],
    "record_count": 4,
    "reward_max": 0.94,
    "reward_mean": 0.9,
    "reward_min": 0.86
  }
}
```

### `plugin_ecosystem`

- Suite: `plugin_ecosystem_catalog_quality`
- Wall time (s): `0.00028747000033035874`
- Steps/s: `10435.871557214383`

| Metric | Value |
|--------|-------|
| `acceptance_passed` | `1` |
| `capability_count` | `5` |
| `compatible_count` | `2` |
| `deterministic_hash` | `1` |
| `full_plugin_count` | `3` |
| `handoff_blocked_count` | `1` |
| `handoff_loading_disabled` | `1` |
| `handoff_sha256` | `db0fd80e5a3d3468412f0314558b017f9a2f4473d5d7a9ab768e40d86eaf3f77` |
| `handoff_target_hash_count` | `6` |
| `incompatible_count` | `1` |
| `observed_kind_count` | `4` |
| `plugin_count` | `2` |
| `registry_sha256` | `4dc86c339a42dba16bfe99c79fd6197051c87c97c7fbbf2a93dd86c1585ff25b` |
| `required_kind_count` | `4` |

`acceptance_thresholds_json`:

```json
{
  "min_blocked_handoff_count": 1,
  "min_capability_count": 5,
  "min_handoff_target_hash_count": 5,
  "min_incompatible_count": 1,
  "min_plugin_count": 2,
  "require_deterministic_hash": true,
  "require_loading_disabled": true,
  "required_capability_kinds": [
    "actuator",
    "bridge",
    "extractor",
    "monitor"
  ]
}
```

`capability_counts_json`:

```json
{
  "actuator": 1,
  "bridge": 1,
  "domainpack": 0,
  "extractor": 1,
  "monitor": 2
}
```

`handoff_dispatch_groups_json`:

```json
{
  "actuator": 1,
  "bridge": 1,
  "domainpack": 0,
  "extractor": 1,
  "monitor": 2
}
```

### `kuramoto`

- Suite: `kuramoto_reference_strogatz_2000`
- Wall time (s): `0.13407328497851267`
- Steps/s: `7458.607433690206`

| Metric | Value |
|--------|-------|
| `final_order_parameter` | `1.0` |
| `n_oscillators` | `64` |
| `n_steps` | `1000` |

### `stuart_landau`

- Suite: `stuart_landau_reference_pikovsky_2001`
- Wall time (s): `0.2681467410293408`
- Steps/s: `3729.3013376231165`

| Metric | Value |
|--------|-------|
| `final_mean_amplitude` | `3.6193922141707704` |
| `n_oscillators` | `64` |
| `n_steps` | `1000` |

### `petri_reachability`

- Suite: `petri_net_reachability`
- Wall time (s): `0.01973997103050351`
- Steps/s: `253293.17820546284`

| Metric | Value |
|--------|-------|
| `n_steps` | `5000` |
| `reachable_markings` | `4` |

