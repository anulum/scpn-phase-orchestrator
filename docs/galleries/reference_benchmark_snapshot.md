# Reference Benchmark Snapshot

This page is generated from `benchmarks/results/reference_suite.json`.
Timing values are local, non-isolated regression evidence unless the metadata states otherwise.

## Metadata

- `suite_version`: `reference_suite_v1`
- `snapshot_date`: `2026-06-05`
- `command`: `PYTHONPATH=.:src python benchmarks/reference_suite.py`
- `backend`: `python_numpy`
- `python_version`: `3.12.3`
- `python_implementation`: `CPython`
- `numpy_version`: `2.4.6`
- `platform`: `Linux-6.17.0-29-generic-x86_64-with-glibc2.39`
- `executable`: `/media/anulum/724AA8E84AA8AA75/aaa_God_of_the_Math_Collection/03_CODE/SCPN-PHASE-ORCHESTRATOR/.venv/bin/python`
- `benchmark_evidence_kind`: `local_regression_non_isolated`
- `isolation_method`: `none`
- `production_timing_claim`: `false`

## Benchmark Records

| Key | Suite | Acceptance passed | n | n_steps | calls | fixture_count | wall time (s) | steps/s | evidence kind |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `auto_binding` | `auto_binding_synthetic_quality` | `n/a` | `n/a` | `n/a` | `n/a` | `4` | `0.0519516181666404` | `76.9947143353566` | `n/a` |
| `semantic_retrieval` | `semantic_retrieval_ranking_quality` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.020590331871062517` | `145.6994485949095` | `n/a` |
| `replay_policy` | `replay_policy_candidate_quality` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.03078699391335249` | `292.3312365387077` | `n/a` |
| `bayesian_posterior` | `bayesian_posterior_fit_quality` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.00855832314118743` | `11217.150651626414` | `n/a` |
| `bayesian_backends` | `bayesian_backend_fail_closed` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0011764559894800186` | `2550.031643194718` | `n/a` |
| `formal_export` | `formal_export_artifact_quality` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.00052488106302917` | `9525.967599486681` | `n/a` |
| `stl_closed_loop` | `stl_closed_loop_plan_quality` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0005248200614005327` | `5716.244901146141` | `n/a` |
| `domain_formal_export` | `domain_formal_safety_exports` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0012209699489176273` | `45046.15371472224` | `n/a` |
| `hybrid_cocompiler` | `hybrid_cocompiler_review_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.00010228296741843224` | `9776.798867294026` | `n/a` |
| `quantum_target_readiness` | `quantum_target_readiness_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.00012050708755850792` | `16596.53419994049` | `n/a` |
| `neuromorphic_target_readiness` | `neuromorphic_target_readiness_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0001696040853857994` | `11792.168776185952` | `n/a` |
| `hybrid_target_readiness` | `hybrid_target_readiness_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `9.675603359937668e-05` | `20670.54555255027` | `n/a` |
| `hybrid_operator_handoff` | `hybrid_operator_handoff_package_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.00011123809963464737` | `17979.4513441783` | `n/a` |
| `value_alignment_replay_calibration` | `value_alignment_replay_calibration_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.00017474801279604435` | `17167.577198725714` | `n/a` |
| `autopoietic_lineage` | `autopoietic_lineage_sandbox_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.00040750415064394474` | `12269.81367453293` | `n/a` |
| `intergenerational_inheritance` | `intergenerational_policy_inheritance_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0006805029697716236` | `2939.0026037229472` | `n/a` |
| `temporal_causal_hypergraph` | `temporal_causal_hypergraph_experiment_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0019752418156713247` | `1012.5342548604666` | `n/a` |
| `morphogenetic_domain_demos` | `morphogenetic_domain_demo_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.045392431085929275` | `66.09031347805336` | `n/a` |
| `integrated_information_replay_corpus` | `integrated_information_replay_corpus_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.04878884111531079` | `245.95788146798574` | `n/a` |
| `evolutionary_supervisor_search` | `evolutionary_supervisor_search` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.004329723073169589` | `4850.194722644669` | `n/a` |
| `evolutionary_mutation_grammars` | `evolutionary_mutation_grammar_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.025701467879116535` | `778.1656710841327` | `n/a` |
| `federated_meta_orchestrator` | `federated_meta_orchestrator` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0004599830135703087` | `6521.979967726456` | `n/a` |
| `federated_production_boundary` | `federated_production_boundary_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.00131798698566854` | `2276.1985001530725` | `n/a` |
| `federated_deployment_preflight` | `federated_deployment_preflight_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.002334353979676962` | `1285.152134645472` | `n/a` |
| `topos_semantic_binding` | `topos_semantic_binding_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.07764111598953605` | `77.27864190938008` | `n/a` |
| `multiverse_counterfactual` | `multiverse_counterfactual_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `1.6499168621376157` | `2.424364579690179` | `n/a` |
| `hybrid_entanglement_order` | `hybrid_entanglement_order_parameter_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.002376115880906582` | `3366.839161458609` | `n/a` |
| `self_model_digital_twin` | `self_model_digital_twin` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.003611583961173892` | `1107.5472820241064` | `n/a` |
| `strange_loop_drift_scenarios` | `strange_loop_drift_scenario_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.023289232049137354` | `6870.127776751939` | `n/a` |
| `information_geometry_control` | `information_geometry_control_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.9426943680737168` | `4.24315678067911` | `n/a` |
| `sheaf_obstruction_domains` | `sheaf_obstruction_domain_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.11304459697566926` | `53.07639781573451` | `n/a` |
| `meta_transfer_corpus` | `meta_transfer_audit_corpus_quality` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.002011456061154604` | `2982.9137786663437` | `n/a` |
| `meta_transfer` | `meta_transfer_package_manifest_quality` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0003853221423923969` | `10380.924322606297` | `n/a` |
| `plugin_ecosystem` | `plugin_ecosystem_catalog_quality` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.00024853204376995564` | `12070.878082734625` | `n/a` |
| `chimera_polyglot` | `chimera_polyglot_parity_gate` | `1` | `32` | `n/a` | `1` | `n/a` | `0.5604445410426706` | `8.921489342545518` | `n/a` |
| `dimension_polyglot` | `dimension_polyglot_parity_gate` | `1` | `n/a` | `n/a` | `1` | `n/a` | `1.2102484051138163` | `4.131383258901946` | `n/a` |
| `embedding_polyglot` | `embedding_polyglot_parity_gate` | `1` | `160` | `n/a` | `1` | `n/a` | `0.6005354581866413` | `1332.1444872142201` | `local_regression_non_isolated` |
| `entropy_production_polyglot` | `entropy_production_polyglot_parity_gate` | `1` | `16` | `n/a` | `1` | `n/a` | `0.09971608500927687` | `802.2777869042629` | `local_regression_non_isolated` |
| `hodge_polyglot` | `hodge_polyglot_parity_gate` | `1` | `10` | `n/a` | `1` | `n/a` | `0.07254752796143293` | `689.2033595766427` | `local_regression_non_isolated` |
| `itpc_polyglot` | `itpc_polyglot_parity_gate` | `1` | `n/a` | `n/a` | `1` | `n/a` | `0.4485286120325327` | `11.147560859812751` | `n/a` |
| `lyapunov_polyglot` | `lyapunov_polyglot_parity_gate` | `1` | `4` | `120` | `1` | `n/a` | `0.044556140433996916` | `112.21797829205455` | `n/a` |
| `npe_polyglot` | `npe_polyglot_parity_gate` | `1` | `20` | `n/a` | `1` | `n/a` | `0.5991481349337846` | `8.345181614481` | `n/a` |
| `order_parameter_polyglot` | `order_parameter_polyglot_parity_gate` | `1` | `64` | `n/a` | `1` | `n/a` | `0.29050597990863025` | `17.211349665065747` | `n/a` |
| `recurrence_polyglot` | `recurrence_polyglot_parity_gate` | `1` | `n/a` | `n/a` | `1` | `n/a` | `0.4387522409670055` | `11.395953189845939` | `n/a` |
| `spectral_polyglot` | `spectral_polyglot_parity_gate` | `1` | `10` | `n/a` | `1` | `n/a` | `1.5436312609817833` | `2.5912924291620736` | `n/a` |
| `spatial_modulator_polyglot` | `spatial_modulator_polyglot_parity_gate` | `1` | `10` | `n/a` | `1` | `n/a` | `0.07088265009224415` | `28.215649349978754` | `local_regression_non_isolated` |
| `upde_doppler_polyglot` | `upde_doppler_polyglot_gate` | `1` | `8` | `8` | `1` | `n/a` | `0.4656392529141158` | `17.180682148108225` | `local_regression_non_isolated` |
| `upde_moving_frame_polyglot` | `upde_moving_frame_polyglot_gate` | `1` | `8` | `8` | `1` | `n/a` | `0.21866202191449702` | `36.5861429888736` | `local_regression_non_isolated` |
| `upde_time_varying_omega_polyglot` | `upde_time_varying_omega_polyglot_gate` | `1` | `8` | `8` | `1` | `n/a` | `0.13142044004052877` | `60.8733314051671` | `local_regression_non_isolated` |
| `pha_c_handoff_polyglot` | `pha_c_handoff_polyglot_parity_gate` | `1` | `8` | `n/a` | `3` | `n/a` | `0.0033463980071246624` | `1494.1438494030685` | `local_regression_non_isolated` |
| `pha_c_timeline_polyglot` | `pha_c_timeline_polyglot_parity_gate` | `1` | `8` | `n/a` | `3` | `n/a` | `0.015391333028674126` | `324.8581517068714` | `local_regression_non_isolated` |
| `pha_c_acceptance_polyglot` | `pha_c_acceptance_polyglot_gate` | `1` | `8` | `n/a` | `2` | `n/a` | `0.1666339859366417` | `30.005883685103242` | `local_regression_non_isolated` |
| `transfer_entropy_polyglot` | `transfer_entropy_polyglot_parity_gate` | `1` | `160` | `n/a` | `1` | `n/a` | `0.7808729710523039` | `1024.4944179869876` | `local_regression_non_isolated` |
| `winding_polyglot` | `winding_polyglot_parity_gate` | `1` | `n/a` | `n/a` | `1` | `n/a` | `0.10203646798618138` | `49.00208816202009` | `n/a` |
| `kuramoto` | `kuramoto_reference_strogatz_2000` | `1` | `n/a` | `1000` | `n/a` | `n/a` | `0.08022092189639807` | `12465.576016334713` | `n/a` |
| `stuart_landau` | `stuart_landau_reference_pikovsky_2001` | `1` | `n/a` | `1000` | `n/a` | `n/a` | `0.10118324495851994` | `9883.05920026532` | `n/a` |
| `petri_reachability` | `petri_net_reachability` | `1` | `n/a` | `5000` | `n/a` | `n/a` | `0.08790427283383906` | `56880.056438795` | `n/a` |
