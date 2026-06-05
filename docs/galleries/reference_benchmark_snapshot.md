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
| `auto_binding` | `auto_binding_synthetic_quality` | `n/a` | `n/a` | `n/a` | `n/a` | `4` | `0.049979392904788256` | `80.03298494681758` | `n/a` |
| `semantic_retrieval` | `semantic_retrieval_ranking_quality` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.01631022710353136` | `183.9336743110379` | `n/a` |
| `replay_policy` | `replay_policy_candidate_quality` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.029584809904918075` | `304.2101682899058` | `n/a` |
| `bayesian_posterior` | `bayesian_posterior_fit_quality` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.010350008960813284` | `9275.35428843305` | `n/a` |
| `bayesian_backends` | `bayesian_backend_fail_closed` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0011690780520439148` | `2566.1246439064184` | `n/a` |
| `formal_export` | `formal_export_artifact_quality` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0006157089956104755` | `8120.71942369869` | `n/a` |
| `stl_closed_loop` | `stl_closed_loop_plan_quality` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0005827690474689007` | `5147.836888437515` | `n/a` |
| `domain_formal_export` | `domain_formal_safety_exports` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.001244602957740426` | `44190.7996907322` | `n/a` |
| `hybrid_cocompiler` | `hybrid_cocompiler_review_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.00010899198241531849` | `9174.98680030847` | `n/a` |
| `quantum_target_readiness` | `quantum_target_readiness_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.00011515989899635315` | `17367.15660078284` | `n/a` |
| `neuromorphic_target_readiness` | `neuromorphic_target_readiness_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0001817098818719387` | `11006.556051858059` | `n/a` |
| `hybrid_target_readiness` | `hybrid_target_readiness_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.00011048512533307076` | `18101.98426232272` | `n/a` |
| `hybrid_operator_handoff` | `hybrid_operator_handoff_package_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.00012177787721157074` | `16423.34425427125` | `n/a` |
| `value_alignment_replay_calibration` | `value_alignment_replay_calibration_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.00018388614989817142` | `16314.442396348373` | `n/a` |
| `autopoietic_lineage` | `autopoietic_lineage_sandbox_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.00040342495776712894` | `12393.878722015448` | `n/a` |
| `intergenerational_inheritance` | `intergenerational_policy_inheritance_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0004945988766849041` | `4043.6808377026446` | `n/a` |
| `temporal_causal_hypergraph` | `temporal_causal_hypergraph_experiment_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.002182341180741787` | `916.4469871389182` | `n/a` |
| `morphogenetic_domain_demos` | `morphogenetic_domain_demo_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.049451305996626616` | `60.665738538930576` | `n/a` |
| `integrated_information_replay_corpus` | `integrated_information_replay_corpus_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.048684193985536695` | `246.48657023191163` | `n/a` |
| `evolutionary_supervisor_search` | `evolutionary_supervisor_search` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.004346579080447555` | `4831.385696964632` | `n/a` |
| `evolutionary_mutation_grammars` | `evolutionary_mutation_grammar_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.022125127958133817` | `903.9495743412159` | `n/a` |
| `federated_meta_orchestrator` | `federated_meta_orchestrator` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0004204858560115099` | `7134.603832947669` | `n/a` |
| `federated_production_boundary` | `federated_production_boundary_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.001275308895856142` | `2352.3712645210053` | `n/a` |
| `federated_deployment_preflight` | `federated_deployment_preflight_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0015503859613090754` | `1935.0020413413292` | `n/a` |
| `topos_semantic_binding` | `topos_semantic_binding_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.06997447297908366` | `85.74555469383075` | `n/a` |
| `multiverse_counterfactual` | `multiverse_counterfactual_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `1.608962939120829` | `2.4860734220427` | `n/a` |
| `hybrid_entanglement_order` | `hybrid_entanglement_order_parameter_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.002224136143922806` | `3596.9021149443` | `n/a` |
| `self_model_digital_twin` | `self_model_digital_twin` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0031362620647996664` | `1275.4036229607957` | `n/a` |
| `strange_loop_drift_scenarios` | `strange_loop_drift_scenario_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.022664238000288606` | `7059.579942549251` | `n/a` |
| `information_geometry_control` | `information_geometry_control_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `1.2958260490559042` | `3.0868340723002654` | `n/a` |
| `sheaf_obstruction_domains` | `sheaf_obstruction_domain_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.1406453971285373` | `42.660478924287425` | `n/a` |
| `meta_transfer_corpus` | `meta_transfer_audit_corpus_quality` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.003788087982684374` | `1583.9125245840216` | `n/a` |
| `meta_transfer` | `meta_transfer_package_manifest_quality` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0005668280646204948` | `7056.813608334825` | `n/a` |
| `plugin_ecosystem` | `plugin_ecosystem_catalog_quality` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.00036355690099298954` | `8251.80320276151` | `n/a` |
| `chimera_polyglot` | `chimera_polyglot_parity_gate` | `1` | `32` | `n/a` | `1` | `n/a` | `0.9096993911080062` | `5.496321146164605` | `n/a` |
| `dimension_polyglot` | `dimension_polyglot_parity_gate` | `1` | `n/a` | `n/a` | `1` | `n/a` | `1.2499341361690313` | `4.000210775365078` | `n/a` |
| `embedding_polyglot` | `embedding_polyglot_parity_gate` | `1` | `160` | `n/a` | `1` | `n/a` | `0.7245674668811262` | `1104.1069832234814` | `local_regression_non_isolated` |
| `entropy_production_polyglot` | `entropy_production_polyglot_parity_gate` | `1` | `16` | `n/a` | `1` | `n/a` | `0.09934163303114474` | `805.3018413228529` | `local_regression_non_isolated` |
| `hodge_polyglot` | `hodge_polyglot_parity_gate` | `1` | `10` | `n/a` | `1` | `n/a` | `0.11295459792017937` | `442.65573000696315` | `local_regression_non_isolated` |
| `itpc_polyglot` | `itpc_polyglot_parity_gate` | `1` | `n/a` | `n/a` | `1` | `n/a` | `0.5259360510390252` | `9.506859227699135` | `n/a` |
| `lyapunov_polyglot` | `lyapunov_polyglot_parity_gate` | `1` | `4` | `120` | `1` | `n/a` | `0.05733317695558071` | `87.2095401912541` | `n/a` |
| `npe_polyglot` | `npe_polyglot_parity_gate` | `1` | `20` | `n/a` | `1` | `n/a` | `0.6797780599445105` | `7.355341830844238` | `n/a` |
| `order_parameter_polyglot` | `order_parameter_polyglot_parity_gate` | `1` | `64` | `n/a` | `1` | `n/a` | `0.34712134790606797` | `14.404184675363194` | `n/a` |
| `recurrence_polyglot` | `recurrence_polyglot_parity_gate` | `1` | `n/a` | `n/a` | `1` | `n/a` | `0.491965961875394` | `10.163304755759523` | `n/a` |
| `spectral_polyglot` | `spectral_polyglot_parity_gate` | `1` | `10` | `n/a` | `1` | `n/a` | `1.7419724508654326` | `2.2962475658055053` | `n/a` |
| `spatial_modulator_polyglot` | `spatial_modulator_polyglot_parity_gate` | `1` | `10` | `n/a` | `1` | `n/a` | `0.100181297166273` | `19.96380618510617` | `local_regression_non_isolated` |
| `upde_doppler_polyglot` | `upde_doppler_polyglot_gate` | `1` | `8` | `8` | `1` | `n/a` | `0.7919200530741364` | `10.102029831098458` | `local_regression_non_isolated` |
| `upde_moving_frame_polyglot` | `upde_moving_frame_polyglot_gate` | `1` | `8` | `8` | `1` | `n/a` | `0.28075169306248426` | `28.494930565635148` | `local_regression_non_isolated` |
| `upde_time_varying_omega_polyglot` | `upde_time_varying_omega_polyglot_gate` | `1` | `8` | `8` | `1` | `n/a` | `0.20485079404897988` | `39.05281420626174` | `local_regression_non_isolated` |
| `pha_c_handoff_polyglot` | `pha_c_handoff_polyglot_parity_gate` | `1` | `8` | `n/a` | `3` | `n/a` | `0.0037313560023903847` | `1339.9954324371342` | `local_regression_non_isolated` |
| `pha_c_timeline_polyglot` | `pha_c_timeline_polyglot_parity_gate` | `1` | `8` | `n/a` | `3` | `n/a` | `0.015830343822017312` | `315.8491095465565` | `local_regression_non_isolated` |
| `pha_c_acceptance_polyglot` | `pha_c_acceptance_polyglot_gate` | `1` | `8` | `n/a` | `2` | `n/a` | `0.6290342309512198` | `7.948693018564421` | `local_regression_non_isolated` |
| `transfer_entropy_polyglot` | `transfer_entropy_polyglot_parity_gate` | `1` | `160` | `n/a` | `1` | `n/a` | `2.730085711926222` | `293.0310929452677` | `local_regression_non_isolated` |
| `winding_polyglot` | `winding_polyglot_parity_gate` | `1` | `n/a` | `n/a` | `1` | `n/a` | `0.2271770068909973` | `22.009269637041527` | `n/a` |
| `kuramoto` | `kuramoto_reference_strogatz_2000` | `1` | `n/a` | `1000` | `n/a` | `n/a` | `0.10113056912086904` | `9888.206985217515` | `n/a` |
| `stuart_landau` | `stuart_landau_reference_pikovsky_2001` | `1` | `n/a` | `1000` | `n/a` | `n/a` | `0.1291516579221934` | `7742.835176010234` | `n/a` |
| `petri_reachability` | `petri_net_reachability` | `1` | `n/a` | `5000` | `n/a` | `n/a` | `0.1265829720068723` | `39499.7835864412` | `n/a` |
