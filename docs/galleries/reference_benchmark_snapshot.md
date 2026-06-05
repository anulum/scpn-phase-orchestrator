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
| `auto_binding` | `auto_binding_synthetic_quality` | `n/a` | `n/a` | `n/a` | `n/a` | `4` | `0.04596682917326689` | `87.01927176491643` | `n/a` |
| `semantic_retrieval` | `semantic_retrieval_ranking_quality` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0142451380379498` | `210.59816984629012` | `n/a` |
| `replay_policy` | `replay_policy_candidate_quality` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.027051894925534725` | `332.6938842833059` | `n/a` |
| `bayesian_posterior` | `bayesian_posterior_fit_quality` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.009412344079464674` | `10199.372142530088` | `n/a` |
| `bayesian_backends` | `bayesian_backend_fail_closed` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.001217518001794815` | `2464.0292756062113` | `n/a` |
| `formal_export` | `formal_export_artifact_quality` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0005100839771330357` | `9802.307510427727` | `n/a` |
| `stl_closed_loop` | `stl_closed_loop_plan_quality` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0005553790833801031` | `5401.715854586463` | `n/a` |
| `domain_formal_export` | `domain_formal_safety_exports` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0011844448745250702` | `46435.25518404011` | `n/a` |
| `hybrid_cocompiler` | `hybrid_cocompiler_review_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.00011115404777228832` | `8996.523473878464` | `n/a` |
| `quantum_target_readiness` | `quantum_target_readiness_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.00011601601727306843` | `17238.998950399873` | `n/a` |
| `neuromorphic_target_readiness` | `neuromorphic_target_readiness_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.00018099788576364517` | `11049.85282873241` | `n/a` |
| `hybrid_target_readiness` | `hybrid_target_readiness_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.00010498100891709328` | `19051.064765242078` | `n/a` |
| `hybrid_operator_handoff` | `hybrid_operator_handoff_package_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.00012836605310440063` | `15580.443206222068` | `n/a` |
| `value_alignment_replay_calibration` | `value_alignment_replay_calibration_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.00015946500934660435` | `18812.90455061126` | `n/a` |
| `autopoietic_lineage` | `autopoietic_lineage_sandbox_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.00040834397077560425` | `12244.578977137957` | `n/a` |
| `intergenerational_inheritance` | `intergenerational_policy_inheritance_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0005329151172190905` | `3752.942889735601` | `n/a` |
| `temporal_causal_hypergraph` | `temporal_causal_hypergraph_experiment_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0018145288340747356` | `1102.21450463742` | `n/a` |
| `morphogenetic_domain_demos` | `morphogenetic_domain_demo_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.04623365984298289` | `64.88778976590851` | `n/a` |
| `integrated_information_replay_corpus` | `integrated_information_replay_corpus_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.04375159810297191` | `274.27569552447676` | `n/a` |
| `evolutionary_supervisor_search` | `evolutionary_supervisor_search` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.004230612888932228` | `4963.8197942757715` | `n/a` |
| `evolutionary_mutation_grammars` | `evolutionary_mutation_grammar_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.021980252116918564` | `909.9076704677864` | `n/a` |
| `federated_meta_orchestrator` | `federated_meta_orchestrator` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0004304898902773857` | `6968.804768137419` | `n/a` |
| `federated_production_boundary` | `federated_production_boundary_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0012603129725903273` | `2380.3611208048474` | `n/a` |
| `federated_deployment_preflight` | `federated_deployment_preflight_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0014989848714321852` | `2001.3544213649668` | `n/a` |
| `topos_semantic_binding` | `topos_semantic_binding_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.07013033796101809` | `85.55498482461466` | `n/a` |
| `multiverse_counterfactual` | `multiverse_counterfactual_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `1.53445636597462` | `2.6067864089829453` | `n/a` |
| `hybrid_entanglement_order` | `hybrid_entanglement_order_parameter_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.002273036865517497` | `3519.520567995125` | `n/a` |
| `self_model_digital_twin` | `self_model_digital_twin` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.003234222996979952` | `1236.7730993611492` | `n/a` |
| `strange_loop_drift_scenarios` | `strange_loop_drift_scenario_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.026121481088921428` | `6125.226952305502` | `n/a` |
| `information_geometry_control` | `information_geometry_control_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.8921674138400704` | `4.483463459826654` | `n/a` |
| `sheaf_obstruction_domains` | `sheaf_obstruction_domain_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.10838901507668197` | `55.35616312921729` | `n/a` |
| `meta_transfer_corpus` | `meta_transfer_audit_corpus_quality` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0020237998105585575` | `2964.720111493653` | `n/a` |
| `meta_transfer` | `meta_transfer_package_manifest_quality` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.00037324707955121994` | `10716.76168185822` | `n/a` |
| `plugin_ecosystem` | `plugin_ecosystem_catalog_quality` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0002400740049779415` | `12496.146762227114` | `n/a` |
| `chimera_polyglot` | `chimera_polyglot_parity_gate` | `1` | `32` | `n/a` | `1` | `n/a` | `0.5445627230219543` | `9.181678783030513` | `n/a` |
| `dimension_polyglot` | `dimension_polyglot_parity_gate` | `1` | `n/a` | `n/a` | `1` | `n/a` | `1.0391012718901038` | `4.811850524352745` | `n/a` |
| `embedding_polyglot` | `embedding_polyglot_parity_gate` | `1` | `160` | `n/a` | `1` | `n/a` | `0.5484372819773853` | `1458.6900385685083` | `local_regression_non_isolated` |
| `entropy_production_polyglot` | `entropy_production_polyglot_parity_gate` | `1` | `16` | `n/a` | `1` | `n/a` | `0.08443616097792983` | `947.4613610264757` | `local_regression_non_isolated` |
| `hodge_polyglot` | `hodge_polyglot_parity_gate` | `1` | `10` | `n/a` | `1` | `n/a` | `0.0704228829126805` | `709.996494491663` | `local_regression_non_isolated` |
| `itpc_polyglot` | `itpc_polyglot_parity_gate` | `1` | `n/a` | `n/a` | `1` | `n/a` | `0.38883964996784925` | `12.85877096230649` | `n/a` |
| `lyapunov_polyglot` | `lyapunov_polyglot_parity_gate` | `1` | `4` | `120` | `1` | `n/a` | `0.034660781268030405` | `144.2552596069663` | `n/a` |
| `npe_polyglot` | `npe_polyglot_parity_gate` | `1` | `20` | `n/a` | `1` | `n/a` | `0.547312397044152` | `9.135550422397333` | `n/a` |
| `order_parameter_polyglot` | `order_parameter_polyglot_parity_gate` | `1` | `64` | `n/a` | `1` | `n/a` | `0.26220094691962004` | `19.06934379429527` | `n/a` |
| `recurrence_polyglot` | `recurrence_polyglot_parity_gate` | `1` | `n/a` | `n/a` | `1` | `n/a` | `0.43407836998812854` | `11.51865733401262` | `n/a` |
| `spectral_polyglot` | `spectral_polyglot_parity_gate` | `1` | `10` | `n/a` | `1` | `n/a` | `1.3209295908454806` | `3.0281704851806226` | `n/a` |
| `spatial_modulator_polyglot` | `spatial_modulator_polyglot_parity_gate` | `1` | `10` | `n/a` | `1` | `n/a` | `0.05532311508432031` | `36.15125426237686` | `local_regression_non_isolated` |
| `upde_doppler_polyglot` | `upde_doppler_polyglot_gate` | `1` | `8` | `8` | `1` | `n/a` | `0.44557871692813933` | `17.95417890502656` | `local_regression_non_isolated` |
| `upde_moving_frame_polyglot` | `upde_moving_frame_polyglot_gate` | `1` | `8` | `8` | `1` | `n/a` | `0.2045470029115677` | `39.11081505045889` | `local_regression_non_isolated` |
| `upde_time_varying_omega_polyglot` | `upde_time_varying_omega_polyglot_gate` | `1` | `8` | `8` | `1` | `n/a` | `0.125964995007962` | `63.50970759371948` | `local_regression_non_isolated` |
| `pha_c_handoff_polyglot` | `pha_c_handoff_polyglot_parity_gate` | `1` | `8` | `n/a` | `3` | `n/a` | `0.003995686070993543` | `1251.3495582891803` | `local_regression_non_isolated` |
| `pha_c_timeline_polyglot` | `pha_c_timeline_polyglot_parity_gate` | `1` | `8` | `n/a` | `3` | `n/a` | `0.014290111139416695` | `349.89231022902277` | `local_regression_non_isolated` |
| `pha_c_acceptance_polyglot` | `pha_c_acceptance_polyglot_gate` | `1` | `8` | `n/a` | `2` | `n/a` | `0.16979769198223948` | `29.446807795968102` | `local_regression_non_isolated` |
| `transfer_entropy_polyglot` | `transfer_entropy_polyglot_parity_gate` | `1` | `160` | `n/a` | `1` | `n/a` | `0.7241825060918927` | `1104.6939041889623` | `local_regression_non_isolated` |
| `winding_polyglot` | `winding_polyglot_parity_gate` | `1` | `n/a` | `n/a` | `1` | `n/a` | `0.10367423109710217` | `48.22799211616007` | `n/a` |
| `kuramoto` | `kuramoto_reference_strogatz_2000` | `1` | `n/a` | `1000` | `n/a` | `n/a` | `0.07635469106025994` | `13096.772262633993` | `n/a` |
| `stuart_landau` | `stuart_landau_reference_pikovsky_2001` | `1` | `n/a` | `1000` | `n/a` | `n/a` | `0.1007925511803478` | `9921.368080173932` | `n/a` |
| `petri_reachability` | `petri_net_reachability` | `1` | `n/a` | `5000` | `n/a` | `n/a` | `0.025447208900004625` | `196485.20274453718` | `n/a` |
