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
| `auto_binding` | `auto_binding_synthetic_quality` | `n/a` | `n/a` | `n/a` | `n/a` | `4` | `0.07453233585692942` | `53.66798120587968` | `n/a` |
| `semantic_retrieval` | `semantic_retrieval_ranking_quality` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.021175839938223362` | `141.67088572410591` | `n/a` |
| `replay_policy` | `replay_policy_candidate_quality` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.028461748966947198` | `316.21387745537896` | `n/a` |
| `bayesian_posterior` | `bayesian_posterior_fit_quality` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.008546457160264254` | `11232.724648330386` | `n/a` |
| `bayesian_backends` | `bayesian_backend_fail_closed` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0012197690084576607` | `2459.4820652095073` | `n/a` |
| `formal_export` | `formal_export_artifact_quality` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0005628999788314104` | `8882.57272700575` | `n/a` |
| `stl_closed_loop` | `stl_closed_loop_plan_quality` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0005371519364416599` | `5585.011979801045` | `n/a` |
| `domain_formal_export` | `domain_formal_safety_exports` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0014742519706487656` | `37307.0554389671` | `n/a` |
| `hybrid_cocompiler` | `hybrid_cocompiler_review_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.00012679793871939182` | `7886.563536439139` | `n/a` |
| `quantum_target_readiness` | `quantum_target_readiness_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.00015252619050443172` | `13112.502143963853` | `n/a` |
| `neuromorphic_target_readiness` | `neuromorphic_target_readiness_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.00020994804799556732` | `9526.166206804774` | `n/a` |
| `hybrid_target_readiness` | `hybrid_target_readiness_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0001235860399901867` | `16183.057569923021` | `n/a` |
| `hybrid_operator_handoff` | `hybrid_operator_handoff_package_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0001268389169126749` | `15768.031205887266` | `n/a` |
| `value_alignment_replay_calibration` | `value_alignment_replay_calibration_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.00018727383576333523` | `16019.32265536126` | `n/a` |
| `autopoietic_lineage` | `autopoietic_lineage_sandbox_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.00045968499034643173` | `10877.013835565649` | `n/a` |
| `intergenerational_inheritance` | `intergenerational_policy_inheritance_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0005731149576604366` | `3489.701277671023` | `n/a` |
| `temporal_causal_hypergraph` | `temporal_causal_hypergraph_experiment_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0020208389032632113` | `989.6879938180322` | `n/a` |
| `morphogenetic_domain_demos` | `morphogenetic_domain_demo_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.09765231795608997` | `30.721236963867774` | `n/a` |
| `integrated_information_replay_corpus` | `integrated_information_replay_corpus_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.06467426591552794` | `185.54520612067535` | `n/a` |
| `evolutionary_supervisor_search` | `evolutionary_supervisor_search` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0060155862011015415` | `3490.9316063253477` | `n/a` |
| `evolutionary_mutation_grammars` | `evolutionary_mutation_grammar_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.03421116108074784` | `584.6045374722725` | `n/a` |
| `federated_meta_orchestrator` | `federated_meta_orchestrator` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0005353279411792755` | `5604.041502842708` | `n/a` |
| `federated_production_boundary` | `federated_production_boundary_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0016376199200749397` | `1831.9269100382683` | `n/a` |
| `federated_deployment_preflight` | `federated_deployment_preflight_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0019315080717206001` | `1553.1905063837398` | `n/a` |
| `topos_semantic_binding` | `topos_semantic_binding_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.10425653401762247` | `57.550349784175836` | `n/a` |
| `multiverse_counterfactual` | `multiverse_counterfactual_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `2.101559525122866` | `1.9033484192012802` | `n/a` |
| `hybrid_entanglement_order` | `hybrid_entanglement_order_parameter_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0037572828587144613` | `2129.1982267039557` | `n/a` |
| `self_model_digital_twin` | `self_model_digital_twin` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.004320506006479263` | `925.8174838783663` | `n/a` |
| `strange_loop_drift_scenarios` | `strange_loop_drift_scenario_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.024047174025326967` | `6653.588477027894` | `n/a` |
| `information_geometry_control` | `information_geometry_control_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `1.0667757890187204` | `3.7496164059735757` | `n/a` |
| `sheaf_obstruction_domains` | `sheaf_obstruction_domain_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.10946103185415268` | `54.814027406524716` | `n/a` |
| `meta_transfer_corpus` | `meta_transfer_audit_corpus_quality` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0020398299675434828` | `2941.421635856077` | `n/a` |
| `meta_transfer` | `meta_transfer_package_manifest_quality` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0003952109254896641` | `10121.177685166529` | `n/a` |
| `plugin_ecosystem` | `plugin_ecosystem_catalog_quality` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.00027369800955057144` | `10960.985813985933` | `n/a` |
| `chimera_polyglot` | `chimera_polyglot_parity_gate` | `1` | `32` | `n/a` | `1` | `n/a` | `0.5850956980139017` | `8.545610601774074` | `n/a` |
| `dimension_polyglot` | `dimension_polyglot_parity_gate` | `1` | `n/a` | `n/a` | `1` | `n/a` | `1.1177901290357113` | `4.473111606660341` | `n/a` |
| `embedding_polyglot` | `embedding_polyglot_parity_gate` | `1` | `160` | `n/a` | `1` | `n/a` | `0.5867638629861176` | `1363.41047986271` | `local_regression_non_isolated` |
| `entropy_production_polyglot` | `entropy_production_polyglot_parity_gate` | `1` | `16` | `n/a` | `1` | `n/a` | `0.0984102669171989` | `812.9233108097445` | `local_regression_non_isolated` |
| `hodge_polyglot` | `hodge_polyglot_parity_gate` | `1` | `10` | `n/a` | `1` | `n/a` | `0.08044954808428884` | `621.5075310008435` | `local_regression_non_isolated` |
| `itpc_polyglot` | `itpc_polyglot_parity_gate` | `1` | `n/a` | `n/a` | `1` | `n/a` | `0.4212981639429927` | `11.868079255803657` | `n/a` |
| `lyapunov_polyglot` | `lyapunov_polyglot_parity_gate` | `1` | `4` | `120` | `1` | `n/a` | `0.042004742193967104` | `119.03417897225235` | `n/a` |
| `npe_polyglot` | `npe_polyglot_parity_gate` | `1` | `20` | `n/a` | `1` | `n/a` | `0.6951327170245349` | `7.192871055475761` | `n/a` |
| `order_parameter_polyglot` | `order_parameter_polyglot_parity_gate` | `1` | `64` | `n/a` | `1` | `n/a` | `0.4772693330887705` | `10.476264979442995` | `n/a` |
| `recurrence_polyglot` | `recurrence_polyglot_parity_gate` | `1` | `n/a` | `n/a` | `1` | `n/a` | `0.6089165748562664` | `8.211305466894608` | `n/a` |
| `spectral_polyglot` | `spectral_polyglot_parity_gate` | `1` | `10` | `n/a` | `1` | `n/a` | `2.0004202448762953` | `1.999579843408032` | `n/a` |
| `spatial_modulator_polyglot` | `spatial_modulator_polyglot_parity_gate` | `1` | `10` | `n/a` | `1` | `n/a` | `0.07746908697299659` | `25.816749340251555` | `local_regression_non_isolated` |
| `upde_doppler_polyglot` | `upde_doppler_polyglot_gate` | `1` | `8` | `8` | `1` | `n/a` | `0.5369365580845624` | `14.899339371747669` | `local_regression_non_isolated` |
| `upde_moving_frame_polyglot` | `upde_moving_frame_polyglot_gate` | `1` | `8` | `8` | `1` | `n/a` | `0.2713614138774574` | `29.48097846959434` | `local_regression_non_isolated` |
| `upde_time_varying_omega_polyglot` | `upde_time_varying_omega_polyglot_gate` | `1` | `8` | `8` | `1` | `n/a` | `0.1858055149205029` | `43.055772609455694` | `local_regression_non_isolated` |
| `pha_c_handoff_polyglot` | `pha_c_handoff_polyglot_parity_gate` | `1` | `8` | `n/a` | `3` | `n/a` | `0.004308332921937108` | `1160.54169689187` | `local_regression_non_isolated` |
| `pha_c_timeline_polyglot` | `pha_c_timeline_polyglot_parity_gate` | `1` | `8` | `n/a` | `3` | `n/a` | `0.0186280719935894` | `268.4121041469392` | `local_regression_non_isolated` |
| `pha_c_acceptance_polyglot` | `pha_c_acceptance_polyglot_gate` | `1` | `8` | `n/a` | `2` | `n/a` | `0.2115257370751351` | `23.63778549663662` | `local_regression_non_isolated` |
| `transfer_entropy_polyglot` | `transfer_entropy_polyglot_parity_gate` | `1` | `160` | `n/a` | `1` | `n/a` | `0.8976973618846387` | `891.1689328355255` | `local_regression_non_isolated` |
| `winding_polyglot` | `winding_polyglot_parity_gate` | `1` | `n/a` | `n/a` | `1` | `n/a` | `0.11677914089523256` | `42.815865587551365` | `n/a` |
| `kuramoto` | `kuramoto_reference_strogatz_2000` | `1` | `n/a` | `1000` | `n/a` | `n/a` | `0.08986978512257338` | `11127.210314747057` | `n/a` |
| `stuart_landau` | `stuart_landau_reference_pikovsky_2001` | `1` | `n/a` | `1000` | `n/a` | `n/a` | `0.11434625810943544` | `8745.367067831348` | `n/a` |
| `petri_reachability` | `petri_net_reachability` | `1` | `n/a` | `5000` | `n/a` | `n/a` | `0.029142698040232062` | `171569.564118511` | `n/a` |
