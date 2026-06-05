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
| `auto_binding` | `auto_binding_synthetic_quality` | `n/a` | `n/a` | `n/a` | `n/a` | `4` | `0.06896619102917612` | `57.999433349999016` | `n/a` |
| `semantic_retrieval` | `semantic_retrieval_ranking_quality` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.024844852974638343` | `120.74935613675814` | `n/a` |
| `replay_policy` | `replay_policy_candidate_quality` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.03808531700633466` | `236.3115422802716` | `n/a` |
| `bayesian_posterior` | `bayesian_posterior_fit_quality` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.01527915894985199` | `6283.068349186194` | `n/a` |
| `bayesian_backends` | `bayesian_backend_fail_closed` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.001762285130098462` | `1702.3351946642053` | `n/a` |
| `formal_export` | `formal_export_artifact_quality` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0010535120964050293` | `4746.029985855728` | `n/a` |
| `stl_closed_loop` | `stl_closed_loop_plan_quality` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0011629220098257065` | `2579.7086774973213` | `n/a` |
| `domain_formal_export` | `domain_formal_safety_exports` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0017013819888234138` | `32326.661714595382` | `n/a` |
| `hybrid_cocompiler` | `hybrid_cocompiler_review_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0001475149765610695` | `6778.972707127209` | `n/a` |
| `quantum_target_readiness` | `quantum_target_readiness_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0001731140073388815` | `11553.080139176` | `n/a` |
| `neuromorphic_target_readiness` | `neuromorphic_target_readiness_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0003237910568714142` | `6176.822853987137` | `n/a` |
| `hybrid_target_readiness` | `hybrid_target_readiness_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.00013274489901959896` | `15066.49230796215` | `n/a` |
| `hybrid_operator_handoff` | `hybrid_operator_handoff_package_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0001261080615222454` | `15859.41434558647` | `n/a` |
| `value_alignment_replay_calibration` | `value_alignment_replay_calibration_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.00023761391639709473` | `12625.523140599524` | `n/a` |
| `autopoietic_lineage` | `autopoietic_lineage_sandbox_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0006408870685845613` | `7801.686514041871` | `n/a` |
| `intergenerational_inheritance` | `intergenerational_policy_inheritance_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.000680247088894248` | `2940.1081351939783` | `n/a` |
| `temporal_causal_hypergraph` | `temporal_causal_hypergraph_experiment_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0029199200216680765` | `684.950267527345` | `n/a` |
| `morphogenetic_domain_demos` | `morphogenetic_domain_demo_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.06033704592846334` | `49.72069735659337` | `n/a` |
| `integrated_information_replay_corpus` | `integrated_information_replay_corpus_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.06080746604129672` | `197.3441878313155` | `n/a` |
| `evolutionary_supervisor_search` | `evolutionary_supervisor_search` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.008351457072421908` | `2514.5312749491313` | `n/a` |
| `evolutionary_mutation_grammars` | `evolutionary_mutation_grammar_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0357975538354367` | `558.6973928984389` | `n/a` |
| `federated_meta_orchestrator` | `federated_meta_orchestrator` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0004890759009867907` | `6134.017222985244` | `n/a` |
| `federated_production_boundary` | `federated_production_boundary_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0012559678871184587` | `2388.5961024710896` | `n/a` |
| `federated_deployment_preflight` | `federated_deployment_preflight_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0015547799412161112` | `1929.533511767249` | `n/a` |
| `topos_semantic_binding` | `topos_semantic_binding_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.09680248401127756` | `61.98188054038981` | `n/a` |
| `multiverse_counterfactual` | `multiverse_counterfactual_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `2.503354180138558` | `1.5978562009865516` | `n/a` |
| `hybrid_entanglement_order` | `hybrid_entanglement_order_parameter_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0034219359513372183` | `2337.8579008393694` | `n/a` |
| `self_model_digital_twin` | `self_model_digital_twin` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.003609985113143921` | `1108.0378103045464` | `n/a` |
| `strange_loop_drift_scenarios` | `strange_loop_drift_scenario_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.025624035159125924` | `6244.137545331789` | `n/a` |
| `information_geometry_control` | `information_geometry_control_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `1.2568303011357784` | `3.182609455218625` | `n/a` |
| `sheaf_obstruction_domains` | `sheaf_obstruction_domain_gate` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.12532611191272736` | `47.875098879459266` | `n/a` |
| `meta_transfer_corpus` | `meta_transfer_audit_corpus_quality` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.0021066078916192055` | `2848.1807287773` | `n/a` |
| `meta_transfer` | `meta_transfer_package_manifest_quality` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.00037694000639021397` | `10611.76827131249` | `n/a` |
| `plugin_ecosystem` | `plugin_ecosystem_catalog_quality` | `1` | `n/a` | `n/a` | `n/a` | `n/a` | `0.00024069682694971561` | `12463.811999593727` | `n/a` |
| `chimera_polyglot` | `chimera_polyglot_parity_gate` | `1` | `32` | `n/a` | `1` | `n/a` | `0.7815421039704233` | `6.397607978634533` | `n/a` |
| `dimension_polyglot` | `dimension_polyglot_parity_gate` | `1` | `n/a` | `n/a` | `1` | `n/a` | `1.4332642061635852` | `3.4885403392466543` | `n/a` |
| `embedding_polyglot` | `embedding_polyglot_parity_gate` | `1` | `160` | `n/a` | `1` | `n/a` | `0.7304364910814911` | `1095.2355335034158` | `local_regression_non_isolated` |
| `entropy_production_polyglot` | `entropy_production_polyglot_parity_gate` | `1` | `16` | `n/a` | `1` | `n/a` | `0.13775190408341587` | `580.754222834959` | `local_regression_non_isolated` |
| `hodge_polyglot` | `hodge_polyglot_parity_gate` | `1` | `10` | `n/a` | `1` | `n/a` | `0.10326262190937996` | `484.2023093688095` | `local_regression_non_isolated` |
| `itpc_polyglot` | `itpc_polyglot_parity_gate` | `1` | `n/a` | `n/a` | `1` | `n/a` | `0.5771946830209345` | `8.662588459462045` | `n/a` |
| `lyapunov_polyglot` | `lyapunov_polyglot_parity_gate` | `1` | `4` | `120` | `1` | `n/a` | `0.04318518005311489` | `115.78045972832193` | `n/a` |
| `npe_polyglot` | `npe_polyglot_parity_gate` | `1` | `20` | `n/a` | `1` | `n/a` | `0.82725935196504` | `6.044053763941372` | `n/a` |
| `order_parameter_polyglot` | `order_parameter_polyglot_parity_gate` | `1` | `64` | `n/a` | `1` | `n/a` | `0.42422167700715363` | `11.786290684800827` | `n/a` |
| `recurrence_polyglot` | `recurrence_polyglot_parity_gate` | `1` | `n/a` | `n/a` | `1` | `n/a` | `0.6641795299947262` | `7.5280850646506705` | `n/a` |
| `spectral_polyglot` | `spectral_polyglot_parity_gate` | `1` | `10` | `n/a` | `1` | `n/a` | `2.3927966139744967` | `1.6716840773842023` | `n/a` |
| `spatial_modulator_polyglot` | `spatial_modulator_polyglot_parity_gate` | `1` | `10` | `n/a` | `1` | `n/a` | `0.08200995111837983` | `24.38728438105075` | `local_regression_non_isolated` |
| `upde_doppler_polyglot` | `upde_doppler_polyglot_gate` | `1` | `8` | `8` | `1` | `n/a` | `0.62219332694076` | `12.857739955738374` | `local_regression_non_isolated` |
| `upde_moving_frame_polyglot` | `upde_moving_frame_polyglot_gate` | `1` | `8` | `8` | `1` | `n/a` | `0.2771857650950551` | `28.86151096993226` | `local_regression_non_isolated` |
| `upde_time_varying_omega_polyglot` | `upde_time_varying_omega_polyglot_gate` | `1` | `8` | `8` | `1` | `n/a` | `0.16711176396347582` | `47.87215340356584` | `local_regression_non_isolated` |
| `pha_c_handoff_polyglot` | `pha_c_handoff_polyglot_parity_gate` | `1` | `8` | `n/a` | `3` | `n/a` | `0.0037325741723179817` | `1339.5581090073633` | `local_regression_non_isolated` |
| `pha_c_timeline_polyglot` | `pha_c_timeline_polyglot_parity_gate` | `1` | `8` | `n/a` | `3` | `n/a` | `0.02561865607276559` | `195.1702691116318` | `local_regression_non_isolated` |
| `pha_c_acceptance_polyglot` | `pha_c_acceptance_polyglot_gate` | `1` | `8` | `n/a` | `2` | `n/a` | `0.2644374230876565` | `18.90806505984815` | `local_regression_non_isolated` |
| `transfer_entropy_polyglot` | `transfer_entropy_polyglot_parity_gate` | `1` | `160` | `n/a` | `1` | `n/a` | `1.049091667169705` | `762.5644403012773` | `local_regression_non_isolated` |
| `winding_polyglot` | `winding_polyglot_parity_gate` | `1` | `n/a` | `n/a` | `1` | `n/a` | `0.13670040597207844` | `36.576336145053354` | `n/a` |
| `kuramoto` | `kuramoto_reference_strogatz_2000` | `1` | `n/a` | `1000` | `n/a` | `n/a` | `0.1205948709975928` | `8292.22662396613` | `n/a` |
| `stuart_landau` | `stuart_landau_reference_pikovsky_2001` | `1` | `n/a` | `1000` | `n/a` | `n/a` | `0.12387648108415306` | `8072.557367210566` | `n/a` |
| `petri_reachability` | `petri_net_reachability` | `1` | `n/a` | `5000` | `n/a` | `n/a` | `0.02606109785847366` | `191856.8445256143` | `n/a` |

## How to read this snapshot

This is an evidence surface for automation trend lines, not a standalone proof
of production readiness. Individual rows show execution-time behavior for configured
benchmarks under a single local environment and single Python backend unless
specifically noted by `evidence kind`.

For release decisions, compare:
- `evidence kind` and isolation method,
- `suite_version` and command metadata,
- and consistency of pass/fail over repeated snapshots.

Because timing and throughput are environment-sensitive, teams should use these
values as directional indicators and pair them with the benchmark provenance
record (command and runtime metadata) before capacity or SLO claims are made.
