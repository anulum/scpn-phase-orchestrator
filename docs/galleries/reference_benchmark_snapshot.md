# Reference benchmark snapshot

This page is generated from `benchmarks/results/reference_suite.json`.
Timing fields are local non-isolated regression evidence unless metadata records CPU/core isolation and host-load controls.

## Metadata

- `suite_version`: `reference_suite_v1`
- `snapshot_date`: `2026-06-05`
- `command`: `PYTHONPATH=.:src python benchmarks/reference_suite.py`
- `backend`: `python_numpy`
- `python_version`: `3.12.3`
- `python_implementation`: `CPython`
- `numpy_version`: `2.4.6`
- `platform`: `Linux-6.17.0-29-generic-x86_64-with-glibc2.39`
- `executable`: `/media/anulum/724AA8E84AA8AA75/aaa_God_of_the_Math_Collection/03_CODE/SCPN-PHASE-ORCHESTRATOR/.venv/bin/python3`
- `benchmark_evidence_kind`: `local_regression_non_isolated`
- `isolation_method`: `none`
- `production_timing_claim`: `false`

## Benchmark records

| Key | Suite | Acceptance | N steps | Fixtures | Wall time s | Steps/s | Evidence |
|-----|-------|-----------:|--------:|---------:|------------:|--------:|----------|
| `auto_binding` | `auto_binding_synthetic_quality` | `n/a` | `n/a` | `4` | `0.06049458798952401` | `66.12161737001482` | `n/a` |
| `autopoietic_lineage` | `autopoietic_lineage_sandbox_gate` | `1` | `n/a` | `n/a` | `0.0006294278427958488` | `7943.72231420611` | `n/a` |
| `bayesian_backends` | `bayesian_backend_fail_closed` | `1` | `n/a` | `n/a` | `0.0011847980786114931` | `2532.0770299659894` | `n/a` |
| `bayesian_posterior` | `bayesian_posterior_fit_quality` | `1` | `n/a` | `n/a` | `0.009102588053792715` | `10546.451122766159` | `n/a` |
| `chimera_polyglot` | `chimera_polyglot_parity_gate` | `1` | `n/a` | `n/a` | `0.6457676431164145` | `7.742723026304733` | `n/a` |
| `dimension_polyglot` | `dimension_polyglot_parity_gate` | `1` | `n/a` | `n/a` | `1.2024149729404598` | `4.158298185336707` | `n/a` |
| `domain_formal_export` | `domain_formal_safety_exports` | `1` | `n/a` | `n/a` | `0.001658349996432662` | `33165.495895506094` | `n/a` |
| `embedding_polyglot` | `embedding_polyglot_parity_gate` | `1` | `n/a` | `n/a` | `0.6397198650520295` | `1250.5473781636197` | `local_regression_non_isolated` |
| `entropy_production_polyglot` | `entropy_production_polyglot_parity_gate` | `1` | `n/a` | `n/a` | `0.11442167102359235` | `699.1682544428579` | `local_regression_non_isolated` |
| `evolutionary_mutation_grammars` | `evolutionary_mutation_grammar_gate` | `1` | `n/a` | `n/a` | `0.026115993969142437` | `765.8142371923949` | `n/a` |
| `evolutionary_supervisor_search` | `evolutionary_supervisor_search` | `1` | `n/a` | `n/a` | `0.005169017938897014` | `4062.6672703095837` | `n/a` |
| `federated_deployment_preflight` | `federated_deployment_preflight_gate` | `1` | `n/a` | `n/a` | `0.002008984098210931` | `1493.2920587433234` | `n/a` |
| `federated_meta_orchestrator` | `federated_meta_orchestrator` | `1` | `n/a` | `n/a` | `0.00048491708002984524` | `6186.624731418738` | `n/a` |
| `federated_production_boundary` | `federated_production_boundary_gate` | `1` | `n/a` | `n/a` | `0.0016353859100490808` | `1834.4294038279716` | `n/a` |
| `formal_export` | `formal_export_artifact_quality` | `1` | `n/a` | `n/a` | `0.0005604119505733252` | `8922.008167179141` | `n/a` |
| `hodge_polyglot` | `hodge_polyglot_parity_gate` | `1` | `n/a` | `n/a` | `0.0821873159147799` | `608.3663816426982` | `local_regression_non_isolated` |
| `hybrid_cocompiler` | `hybrid_cocompiler_review_gate` | `1` | `n/a` | `n/a` | `0.00014308909885585308` | `6988.652580776909` | `n/a` |
| `hybrid_entanglement_order` | `hybrid_entanglement_order_parameter_gate` | `1` | `n/a` | `n/a` | `0.00434129498898983` | `1842.7681187961634` | `n/a` |
| `hybrid_operator_handoff` | `hybrid_operator_handoff_package_gate` | `1` | `n/a` | `n/a` | `0.00022578914649784565` | `8857.821693475787` | `n/a` |
| `hybrid_target_readiness` | `hybrid_target_readiness_gate` | `1` | `n/a` | `n/a` | `0.00015108101069927216` | `13237.931032782237` | `n/a` |
| `information_geometry_control` | `information_geometry_control_gate` | `1` | `n/a` | `n/a` | `1.0669131739996374` | `3.7491335728893715` | `n/a` |
| `integrated_information_replay_corpus` | `integrated_information_replay_corpus_gate` | `1` | `n/a` | `n/a` | `0.07033095392398536` | `170.6218859617596` | `n/a` |
| `intergenerational_inheritance` | `intergenerational_policy_inheritance_gate` | `1` | `n/a` | `n/a` | `0.0007373439148068428` | `2712.438469806762` | `n/a` |
| `itpc_polyglot` | `itpc_polyglot_parity_gate` | `1` | `n/a` | `n/a` | `0.4792384079191834` | `10.433220537789571` | `n/a` |
| `kuramoto` | `kuramoto_reference_strogatz_2000` | `1` | `1000` | `n/a` | `0.11899392888881266` | `8403.790086924477` | `n/a` |
| `lyapunov_polyglot` | `lyapunov_polyglot_parity_gate` | `1` | `120` | `n/a` | `0.03599900612607598` | `138.8927233848891` | `n/a` |
| `meta_transfer` | `meta_transfer_package_manifest_quality` | `1` | `n/a` | `n/a` | `0.0005968490149825811` | `6701.862446931807` | `n/a` |
| `meta_transfer_corpus` | `meta_transfer_audit_corpus_quality` | `1` | `n/a` | `n/a` | `0.0037008679937571287` | `1621.241289913934` | `n/a` |
| `morphogenetic_domain_demos` | `morphogenetic_domain_demo_gate` | `1` | `n/a` | `n/a` | `0.07522566709667444` | `39.88000526661495` | `n/a` |
| `multiverse_counterfactual` | `multiverse_counterfactual_gate` | `1` | `n/a` | `n/a` | `1.959374836878851` | `2.04146747458068` | `n/a` |
| `neuromorphic_target_readiness` | `neuromorphic_target_readiness_gate` | `1` | `n/a` | `n/a` | `0.0002727191895246506` | `7333.550688112557` | `n/a` |
| `npe_polyglot` | `npe_polyglot_parity_gate` | `1` | `n/a` | `n/a` | `0.8290427818428725` | `6.031051846185236` | `n/a` |
| `order_parameter_polyglot` | `order_parameter_polyglot_parity_gate` | `1` | `n/a` | `n/a` | `0.31295827589929104` | `15.976570632722249` | `n/a` |
| `petri_reachability` | `petri_net_reachability` | `1` | `5000` | `n/a` | `0.13126136199571192` | `38091.94056788273` | `n/a` |
| `pha_c_acceptance_polyglot` | `pha_c_acceptance_polyglot_gate` | `1` | `n/a` | `n/a` | `0.19418629887513816` | `25.74846953139058` | `local_regression_non_isolated` |
| `pha_c_handoff_polyglot` | `pha_c_handoff_polyglot_parity_gate` | `1` | `n/a` | `n/a` | `0.0023185699246823788` | `2156.5017068376537` | `local_regression_non_isolated` |
| `pha_c_timeline_polyglot` | `pha_c_timeline_polyglot_parity_gate` | `1` | `n/a` | `n/a` | `0.013210603035986423` | `378.48385773001576` | `local_regression_non_isolated` |
| `plugin_ecosystem` | `plugin_ecosystem_catalog_quality` | `1` | `n/a` | `n/a` | `0.00034120306372642517` | `8792.41811968425` | `n/a` |
| `quantum_target_readiness` | `quantum_target_readiness_gate` | `1` | `n/a` | `n/a` | `0.00015224493108689785` | `13136.726364035376` | `n/a` |
| `recurrence_polyglot` | `recurrence_polyglot_parity_gate` | `1` | `n/a` | `n/a` | `0.5089778290130198` | `9.823610607353388` | `n/a` |
| `replay_policy` | `replay_policy_candidate_quality` | `1` | `n/a` | `n/a` | `0.034437590977177024` | `261.34232228858895` | `n/a` |
| `self_model_digital_twin` | `self_model_digital_twin` | `1` | `n/a` | `n/a` | `0.005657376954331994` | `707.0414491184118` | `n/a` |
| `semantic_retrieval` | `semantic_retrieval_ranking_quality` | `1` | `n/a` | `n/a` | `0.017702843993902206` | `169.46429630365373` | `n/a` |
| `sheaf_obstruction_domains` | `sheaf_obstruction_domain_gate` | `1` | `n/a` | `n/a` | `0.1144044860266149` | `52.44549587508455` | `n/a` |
| `spatial_modulator_polyglot` | `spatial_modulator_polyglot_parity_gate` | `1` | `n/a` | `n/a` | `0.07017243304289877` | `28.501220682733525` | `local_regression_non_isolated` |
| `spectral_polyglot` | `spectral_polyglot_parity_gate` | `1` | `n/a` | `n/a` | `1.7495333841070533` | `2.286323905754771` | `n/a` |
| `stl_closed_loop` | `stl_closed_loop_plan_quality` | `1` | `n/a` | `n/a` | `0.0005437710788100958` | `5517.027508275604` | `n/a` |
| `strange_loop_drift_scenarios` | `strange_loop_drift_scenario_gate` | `1` | `n/a` | `n/a` | `0.023375059012323618` | `6844.902505514362` | `n/a` |
| `stuart_landau` | `stuart_landau_reference_pikovsky_2001` | `1` | `1000` | `n/a` | `0.16326062893494964` | `6125.17547263918` | `n/a` |
| `temporal_causal_hypergraph` | `temporal_causal_hypergraph_experiment_gate` | `1` | `n/a` | `n/a` | `0.004001490073278546` | `499.8138101992934` | `n/a` |
| `topos_semantic_binding` | `topos_semantic_binding_gate` | `1` | `n/a` | `n/a` | `0.09571960405446589` | `62.68308419439251` | `n/a` |
| `transfer_entropy_polyglot` | `transfer_entropy_polyglot_parity_gate` | `1` | `n/a` | `n/a` | `0.8808866878971457` | `908.1758312295096` | `local_regression_non_isolated` |
| `upde_doppler_polyglot` | `upde_doppler_polyglot_gate` | `1` | `8` | `n/a` | `0.4901002428960055` | `16.32319125721699` | `local_regression_non_isolated` |
| `upde_moving_frame_polyglot` | `upde_moving_frame_polyglot_gate` | `1` | `8` | `n/a` | `0.22748251096345484` | `35.167538665357895` | `local_regression_non_isolated` |
| `upde_time_varying_omega_polyglot` | `upde_time_varying_omega_polyglot_gate` | `1` | `8` | `n/a` | `0.12767927208915353` | `62.6569988150771` | `local_regression_non_isolated` |
| `value_alignment_replay_calibration` | `value_alignment_replay_calibration_gate` | `1` | `n/a` | `n/a` | `0.00022820592857897282` | `13146.021309265949` | `n/a` |
| `winding_polyglot` | `winding_polyglot_parity_gate` | `1` | `n/a` | `n/a` | `0.11353808920830488` | `44.03808479484494` | `n/a` |
