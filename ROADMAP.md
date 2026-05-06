# Roadmap

## v0.1 (released)

- Core scaffold with src layout and CLI
- UPDE engine: RK4/RK45 integrator, pre-allocated arrays, Kuramoto coupling
- 3-channel oscillator model (phase, frequency, coherence)
- Coupling matrix builder with decay, cross-hierarchy boosts, geometry constraints
- BoundaryObserver, RegimeManager, SupervisorPolicy pipeline
- PolicyEngine: declarative YAML rules with regime/metric triggers
- ImprintModel: history-dependent coupling modulation (decay, saturation)
- Actuation mapper for domain-agnostic output binding
- 33 domainpacks: neuroscience_eeg, cardiac_rhythm, power_grid, plasma_control, manufacturing_spc, epidemic_sir, traffic_flow, quantum_simulation, chemical_reactor, swarm_robotics, identity_coherence, brain_connectome, sleep_architecture, financial_markets, robotic_cpg, agent_coordination, and 17 more
- Adapter bridges: FusionCoreBridge, PlasmaControlBridge, QuantumControlBridge
- QueueWaves: real-time cascade failure detector with Prometheus ingestion, WebSocket streaming, HTML dashboard, webhook alerts
- Deterministic replay from audit.jsonl with chained phase-vector verification (`spo replay --verify`)
- spo-kernel Rust FFI: UPDE engine, coupling, imprint, order params, lags (112 tests)
- 654 Python tests, 112 Rust tests
- PhaseExtractor base class for signal intake
- PyPI package (trusted publisher OIDC), Zenodo DOI, GitHub Pages docs

## v0.2 (released)

- Extended policy DSL with compound triggers (AND/OR logic) and action chains
- Per-rule cooldown and max-fire rate limiting
- `stability_proxy` metric in policy conditions
- OpenTelemetry trace/metric export for production observability (`OTelExporter`)
- Pre-commit hook for version consistency

## v0.3 (released)

- Petri net regime FSM for multi-phase protocol sequencing (`PetriNet`, `PetriNetAdapter`, `ProtocolNetSpec`)
- SNN controller bridge (`SNNControllerBridge`) with Nengo/Lava optional backends
- Event-driven mode transitions with `EventBus`, `RegimeEvent`, `force_transition()`, `hysteresis_hold_steps`
- Rust `force_transition()` + `transition_log` for FFI parity
- ~90 new tests (total ~800)

## v0.4 (released)

- Stuart-Landau amplitude engine (`StuartLandauEngine`): coupled phase-amplitude ODE with Euler/RK4/RK45
- Phase-amplitude coupling (PAC): Tort et al. modulation index, pac_matrix, pac_gate
- Modulation envelope extraction: sliding-window RMS, modulation depth, EnvelopeState
- `AmplitudeSpec` binding with `amplitude:` YAML block toggling Stuart-Landau mode
- Amplitude coupling (`knm_r`), imprint mu-modulation, amplitude-aware metrics
- CLI amplitude mode with PAC/envelope computation and audit replay support
- ~80 new tests (total ~860)

## v0.4.1 (released)

- Rust `StuartLandauStepper` + FFI `PyStuartLandauStepper` (Euler/RK4/RK45, 12 inline tests)
- Rust `modulation_index` + `pac_matrix` (5 inline tests)
- PAC-driven policy rules (`pac_max`, `mean_amplitude`, `subcritical_fraction`, `amplitude_spread`, `mean_amplitude_layer`)
- Amplitude configs for 6 domainpacks (neuroscience_eeg, cardiac_rhythm, plasma_control, firefly_swarm, rotating_machinery, power_grid)
- `CoherencePlot` matplotlib implementations (R timeline, regime timeline, action audit, amplitude timeline, PAC heatmap)
- Deep audit Phase 1: 12 correctness/safety fixes (hash chain, RK coupling, regime state machine, FFI `run()`, public API expansion)
- 1225 Python tests, 191 Rust tests

## v0.5 (planned)

- ~~Rust RegimeManager: hysteresis, EventBus, downward_streak parity with Python~~ (done)
- ~~Rust LagModel: algorithm alignment with Python adaptive lag pipeline~~ (done)
- ~~FFI improvements: PyStuartLandauStepper numpy array input, PyActionProjector unknown-knob warning~~ (done)
- ~~Expand domainpack channel coverage (target: all 3 channels in 12+ packs)~~ (10/24 done)
- ~~Async test infrastructure: pytest-asyncio + httpx.AsyncClient for QueueWaves~~ (done)
- ~~Tutorial notebooks for binding, audit replay, reporting, adapter bridges~~ (done)
- ~~nn/ module physics validation: 194 tests across 13 phases (183 pass, 10 xfail, 1 skip), 14 findings~~ (done)
- ~~nn/ module documentation: complete API reference (677 lines), updated guide (7 new sections)~~ (done)
- ~~Local GPU validation: GTX 1060, all 9 benchmark suites, JAX 0.9.2~~ (done)
- ~~First automated FIM (strange loop) validation: V75-V86~~ (done)
- ~~Cross-project sync: bidirectional findings exchange with scpn-quantum-control and sc-neurocore~~ (done)
- ~~Petri net + PolicyEngine Rust port (spo-supervisor crate)~~ (done — `spo-kernel/crates/spo-supervisor/src/petri_net.rs`, `spo-kernel/crates/spo-supervisor/src/rule_engine.rs`, FFI bindings in `spo-kernel/crates/spo-ffi/src/lib.rs` (`PyPetriNet`, `PyRuleEngine`), parity coverage in `tests/test_petri_net_parity.py` and `tests/test_rule_engine_parity.py`)

## v1.0

- Production-hardened with fuzz testing (Hypothesis profiles) and fault injection
- Full benchmark suite: Kuramoto reference (Strogatz 2000), Stuart-Landau (Pikovsky 2001), Petri net reachability
- ~~Production hardening slice: SupervisorPolicy Petri fault fallback + Hypothesis fault-injection tests~~ (done — `tests/test_fault_injection_supervisor.py`)
- ~~Benchmark suite slice: unified Kuramoto/Stuart-Landau/Petri reference harness~~ (done — `benchmarks/reference_suite.py`, `tests/test_reference_benchmark_suite.py`)
- ~~RK45 exhausted-retry fallback test coverage~~ (done)
- ~~Complete API documentation with mkdocstrings autodoc for all public modules~~
  (done — maintained public modules under `src/scpn_phase_orchestrator` now
  have API-reference mkdocstrings coverage, guarded by
  `tests/test_api_docs_navigation.py`; generated protobuf stubs remain
  documented through the gRPC facade rather than direct autodoc)
- ~~API docs navigation coverage slice: every `docs/reference/api/*.md` page is
  now wired into `mkdocs.yml` and protected by a regression test~~ (done —
  `tests/test_api_docs_navigation.py`)
- ~~API docs slice: wire missing mkdocstrings API pages into nav/index (autotune, ssgf, visualization)~~ (done — `mkdocs.yml`, `docs/reference/api/index.md`)
- ~~API docs hardening slice: remove broken mkdocstrings import target for non-public ActiveInferenceAgent~~ (done — `docs/reference/api/supervisor.md`)
- ~~API docs slice: add mkdocstrings coverage for core modules (`upde.metrics`, `upde.splitting`, `monitor.npe`, `oscillators.init_phases`)~~ (done — `docs/reference/api/upde.md`, `docs/reference/api/monitor.md`, `docs/reference/api/oscillators.md`)
- ~~Docker multi-stage build with security scanning (Trivy/Grype)~~ (done — `.github/workflows/publish.yml`, `docs/guide/production.md`)
- ~~BoundaryObserver configurable default severity~~ (done — defaults to hard with warning)
- ~~DP tableau deduplication between upde.rs and stuart_landau.rs~~ (done)
- ~~Stable public API freeze discipline slice: top-level manifest, docs, and
  regression test guard `scpn_phase_orchestrator.__all__` drift~~ (done —
  `docs/specs/public_api_manifest.txt`, `tests/test_public_api_manifest.py`)
  - Remaining before v1.0 tag: classify any future public API manifest changes
    in release notes with semantic-versioning impact.

### Deferred track (documented, not current focus)

- Typed-array contract sweep (Python type precision):
  - The latest maintenance sweep shows zero non-parameterized `NDArray` signature sites in `src/` (import-only `NDArray` usage remains).
  - One runtime `np.ndarray` check remains (`visualization/streamer.py`) and is intentionally excluded from this sweep.
  - Track this item as verified, rather than active backlog unless future untyped array annotations are reintroduced.
- Test-hardening maintenance track:
  - Broad superficial-assertion cleanup is registered as long-running quality debt, not an active feature-track blocker.
  - Use `docs/internal/superficial_assertion_audit.md` as the candidate ledger.
  - Remediate weak shape/type/existence-only assertions opportunistically when touching the related module or feature.
  - Prefer behavioural, invariant, error-semantic, parity, or pipeline-effect checks that would fail under a plausible wrong implementation.
  - Do not spend multi-session sweeps on this track unless explicitly prioritised over feature/module roadmap work.

### v1.0 adoption and credibility track

- ~~Ship 5-6 end-to-end tutorial notebooks that start from raw sources and finish with run, visualisation, and deterministic replay~~ (done — `docs/tutorials/05_from_raw_sources_to_run.md`, `docs/tutorials/06_deterministic_replay_for_debugging.md`).
  - CSV sensor stream -> P channel, event log -> I channel, state-machine trace -> S channel, binding spec, engine run, supervisor decisions, actuation output, and `audit.jsonl` replay.
- ~~Add a "minimal viable domainpack in 5 minutes" guide using bundled real sample data, from raw CSV/event/state inputs through scaffold, binding spec, run, visualisation, and replay.~~ (covered by `domainpacks/minimal_domain` and `docs/getting-started/minimal_domainpack_5min.md`)
- ~~Add a high-level "why this knob does what" explainer for K, alpha, zeta, Psi, damping, delay, coupling priors, supervisor thresholds, and actuation limits, aimed at users who do not already know Kuramoto control theory.~~
- ~~Publish short video walkthroughs for first run, binding-spec authoring, policy debugging, audit replay, and deployment profiles.~~ (done — `docs/video_scripts.md`, section “Roadmap Walkthrough Set (v1.0 Adoption Track)”)
- ~~Add a visual binding-spec editor as an optional development extra. First acceptable
  version: load/save `binding_spec.yaml`, validate schema, expose P/I/S channel
  mappings, preview extractor outputs, and produce a minimal reproducible domainpack.~~
  (done — `tools/binding_spec_studio.py`, `docs/guide/interactive_tools.md`)
- ~~Add an interactive supervisor-policy editor and validation loop for the DSL: structured rule builder, trigger/action autocomplete, cooldown/rate-limit previews, schema diagnostics, dry-run evaluation against `audit.jsonl`, and warnings for unreachable or overlapping rules.~~
  (done — `tools/policy_studio.py`, documented in `docs/guide/interactive_tools.md`)
~~Reduce hidden YAML behaviour by documenting every inferred default in generated docs and surfacing resolved runtime configuration in CLI output and audit metadata.~~ (done — `docs/specs/resolved_runtime_defaults.md`, `mkdocs.yml`, `src/scpn_phase_orchestrator/audit/logger.py`, `tests/test_audit_logger.py`)
- ~~Make the mkdocs site the primary entry point: one-page "how the pipeline fires" diagram mapping YAML -> extractors -> engines -> supervisor -> actuation, plus autodoc coverage for every public module.~~ (done — `docs/concepts/pipeline_firing.md`, API nav/index coverage including `reference/api/artifacts.md` and `reference/api/visualization.md`)
~~Publish head-to-head benchmark pages for domainpacks against domain-specific baselines where appropriate, starting with power-grid swing-equation solvers and cardiac rhythm references.~~ (done — `docs/galleries/power_grid_benchmark.md`, `docs/galleries/cardiac_rhythm_benchmark.md`)
- ~~Add reproducible build locks for application and development environments. Evaluate `uv` and `pip-tools`; keep whichever produces maintainable, hash-pinned locks across Linux, macOS, Windows, and CI.~~ (done — standardised on `pip-tools`; documented in `docs/guide/dependency_locks.md`; operational targets in `Makefile` `lock-refresh`/`lock-check`)
- ~~Reduce setup friction with documented install profiles: Python-only, Rust FFI, JAX, Docker, and experimental auxiliary backends. Each profile needs a preflight command that reports missing toolchains, optional dependency status, and expected fallback behaviour.~~ (done — `docs/guide/install_profiles.md`)
- ~~Harden Docker deployment with a documented multi-stage image, explicit production defaults, and CI security scans using Trivy or Grype.~~ (done — `Dockerfile`, `docs/guide/production.md`, `.github/workflows/publish.yml` Trivy scan gate)
~~Keep adapters thin and fuzzed: `hardware_io`, Modbus, OPC-UA, ROS2, Kafka, and related network/file adapters need schema fuzzing, path-scrub tests, and production-default auth/rate-limit examples.~~ (done — `src/scpn_phase_orchestrator/adapters/_schema.py`, `src/scpn_phase_orchestrator/adapters/hardware_io.py`, `src/scpn_phase_orchestrator/adapters/modbus_tls.py`, `src/scpn_phase_orchestrator/adapters/redis_store.py`, `tests/test_adapters_network_validation.py`, docs `guide/adapters.md`, `guide/production.md`)
- ~~Close remaining `nn/` validation xfails/skips before v1.0 unless each has an issue reference, owner, and release-blocking decision.~~ (done — `docs/reference/nn_xfail_skip_register.md` plus pointer in `docs/reference/nn_physics_validation_plan.md`)
- ~~Make N-channel visible in the first-run experience. Ship two or three example domainpacks that use more than P/I/S, including cross-channel coupling, derived channels, and channel groups, with before/after notes showing what the extra channels buy.~~ (done — `digital_twin_nchannel`, `edge_consensus_nchannel`, and `power_safety_nchannel`)
- ~~Add a "minimal viable domainpack in 5 minutes" path to SPO Studio or the CLI: raw sample data, binding scaffold, policy validation, run, visualisation, and replay without requiring users to understand every control-theory detail first.~~ (covered by `domainpacks/minimal_domain` and `docs/getting-started/minimal_domainpack_5min.md`)
- Keep v1.0 focused on N-channel rollout, public benchmarks against standard Kuramoto/Strogatz and Pikovsky references, real hardware examples, and API freeze discipline.

### v1.x architecture focus

- Generalise the current three-channel P/I/S model into a typed N-channel binding architecture. P/I/S remains the default profile, not the ceiling; domainpacks must be able to declare additional named channels with extractor type, units, metric semantics, coupling participation, audit serialisation, replay semantics, and supervisor visibility.
- Add channel algebra for N-channel runs: channel groups, required/optional channels, derived channels, cross-channel coupling policies, and validation rules for missing, delayed, or uncertain channels.
  - Channel algebra summary foundation is in place: `build_channel_algebra_report()` exposes required/optional channels, derived channels, group membership, supervisor visibility, coupling participation, and cross-channel edges for audit/reporting. Remaining scope is runtime execution of delayed/uncertain channel policies.
  - Runtime policy records are in place: delayed channels emit
    `hold_last_runtime_evidence`, uncertain channels emit
    `confidence_weight_runtime_contribution`, and missing required/optional
    channels carry deterministic runtime policies in the audit record.
  - Resolved configuration integration is in place: `resolved_binding_config()` embeds the channel algebra record and CLI formatting surfaces algebra counts plus missing required channel evidence.
  - Audit header contract is covered: `spo run --audit` carries the embedded `channel_algebra` record through `binding_config` and `binding_summary`.
- Update audit, replay, visualisation, and reporting to be channel-count agnostic. Acceptance gate: the same run/replay/report pipeline works for three channels and for at least two domainpacks with more than three declared channels.
  - Report JSON integration is in place: `spo report --json-out` carries audit-header `binding_summary` and exposes `channel_algebra` when present.
  - Report text integration is in place: `spo report` summarises channel-algebra counts and missing required channel evidence when the audit header contains it.
  - Reusable report summary foundation is in place: `reporting.summary.build_audit_report_summary()` exposes the same channel-algebra-aware report payload to notebooks and tools without invoking the CLI.
  - Delayed/uncertain channel classification is in place: `build_channel_algebra_report()` derives delayed and uncertain channel sets from existing channel metadata for audit and reporting.
- Extend optimisation surfaces to include channel weights and cross-channel coupling parameters, not only `K`, `alpha`, `zeta`, and `Psi`.
- Treat Rust and JAX as primary execution paths. Keep Julia, Go, Mojo, and other auxiliary backends experimental unless a maintained production workload shows a 5-10x gain or a capability Rust/JAX cannot provide.
- ~~Document the backend fallback chain in one place, including feature flags, runtime detection, numerical tolerance, benchmark evidence, and deprecation criteria.~~ (done — `docs/guide/backend_fallbacks.md`)
- ~~Add a multi-language backend review gate before each minor release: keep, demote to experimental, or remove based on maintenance cost, CI burden, and measured value.~~ (done — `docs/guide/backend_review_gate.md`, non-destructive default with explicit sign-off for removal)
- Extend visualisation beyond static matplotlib and the current WASM surface: Plotly/Dash dashboards for production operators, real-time streaming plots, and optional 3D views for swarm, traffic, robotics, and spatial domainpacks.

### v1.x differentiators

- ML-driven auto-binding and oscillator discovery: ingest raw multimodal data, propose P/I/S or N-channel extractors, infer an initial coupling graph, and emit a reviewable binding spec.
- Auto domain binding pipeline: turn raw time-series, event logs, and sensor streams into candidate oscillator families, extractor parameters, initial coupling matrices, and a scored `binding_spec.yaml` proposal. Start with SINDy-assisted feature discovery and leave graph-learning coupling inference behind an experimental flag until it has reproducible benchmarks.
- RL and hybrid optimisation layer for knob tuning: leverage the JAX `nn/` backend and `autotune` module to learn policies for `K`, `alpha`, `zeta`, `Psi`, and channel weights from rewards based on coherence metrics minus penalties for `R_bad`, unsafe actuation, and regime churn. Initial scope: replay-trained model-free or hybrid PPO/SAC experiments that emit auditable policy candidates rather than direct production control.
  - Reward-evaluation foundation is in place: `autotune.reward` scores candidate knob policies from replay/simulation observations and emits audit-ready records before any learner can actuate.
  - Replay candidate ranking is in place: `rank_replay_candidates()` orders replay/simulation candidates by reward, filters unsafe rollouts by default, and returns audit-ready reports.
  - Offline policy-search generation is in place: `generate_offline_policy_candidates()` creates deterministic coordinate-search candidates around a seed policy for replay scoring.
  - Replay-trained proposal records are in place: `propose_replay_policy()` applies review gates and serialises accept/reject rationale before any live learner or actuation loop.
  - Replay-only policy search is in place: `search_replay_policy()` binds deterministic candidate generation to a caller-supplied replay/simulation evaluator and returns an audit-ready proposal. Next scope is learner-backed PPO/SAC or hybrid search algorithms behind the same gates.
  - Adaptive replay-only refinement is in place: `search_adaptive_replay_policy()` performs bounded multi-round replay search with decayed coordinate steps and the same proposal gates. Next scope is optional PPO/SAC or hybrid physics learners behind this non-actuating interface.
- Trainable supervisor policies: extend rule-based policy evaluation with reinforcement learning or active-inference loops that optimise long-horizon `R_good` / `R_bad` trade-offs under replayable safety constraints.
- Uncertainty-aware phase estimation: Bayesian or ensemble phase estimates propagated through MPC/OA reduction and supervisor decisions.
- SPO Studio GUI: web-based binding and policy builder that scaffolds, visualises, validates, and replays binding specs, with WASM-backed previews where useful.
- Hierarchical multi-scale orchestration: support nested orchestrators where local/edge supervisors maintain local coherence, exchange reduced phase/coherence summaries, and escalate only bounded regime evidence to a parent supervisor. Reuse Hodge decomposition and transfer-entropy monitors to decide what crosses hierarchy boundaries.
- Distributed edge orchestration: multi-node phase consensus with gossip or local Kuramoto coupling, plus WASM/FPGA deployment paths for decentralised operation.
- Digital-twin binding standard: version `binding_spec.yaml` as an open bidirectional live-sync contract for simulators, services, and hardware twins.
- Formal verification hooks: export Petri-net regimes and policy rules to PRISM, TLA+, SPIN, or equivalent model-checking workflows, with CI artefacts for safety-critical policies.
- Neuromorphic and quantum-native backends: extend existing SNN and quantum-control bridges so the orchestrator can emit Lava/BrainScaleS-style neuromorphic schedules and QPU control schedules directly from validated phase-control plans.
- Extractor and actuator plugin ecosystem: define a stable Python/Rust plugin interface for custom `PhaseExtractor` and `ActuationMapper` implementations, including schema validation, entry-point discovery, audit metadata, compatibility tests, and versioned capability declarations.
  - Python manifest foundation is in place: `plugins.registry` provides entry-point discovery, versioned capability declarations, compatibility reports, audit records, tests, and API documentation.

### Usability moat — finish the job

- One-click SPO Studio web UI for new control engineers: drag/drop oscillators, live `R`/`Psi`/`K` visualisation, real-time knob tuning, and deploy/export paths for Docker, WASM, and FPGA.
- Auto-binding prototype: SINDy-style or graph-learning pipeline from raw time-series, event logs, and graph signals to a proposed `binding_spec.yaml` that stays reviewable by a domain expert.
- RL/autotune layer on the JAX `nn` backend: PPO/SAC or hybrid physics-RL policies that learn `K`, `alpha`, `zeta`, and `Psi` from rewards such as coherence minus penalties for `R_bad`, unsafe actuation, and regime churn.
  - Reward-evaluation, replay ranking, offline candidate generation, proposal records, replay-only policy search, and adaptive replay refinement are in place; next scope is optional PPO/SAC or hybrid physics learners behind the same non-actuating gates.
- Full N-channel and hierarchical orchestration: channel algebra, nested supervisors, and edge/cloud synchronisation protocol for distributed coherence control.
- Formal verification for supervisor: export Petri-net and policy surfaces to PRISM, TLA+, SPIN, or equivalent model-checking workflows for safety properties in critical regimes.
- Plugin ecosystem and marketplace: standard interfaces for domainpacks, extractors, actuators, and bridges so domain experts can publish extensions without forking the core repository.
  - Plugin manifest registry foundation is in place; Rust-side integration remains open.
  - Marketplace catalogue packaging is in place: `build_plugin_marketplace_catalog()` emits deterministic metadata-only catalogue payloads with compatibility records and capability counts.
  - Marketplace example is in place: `examples/plugin_marketplace_catalog.py` builds a validated extractor/actuator manifest and catalogue payload without loading plugin targets.
  - CLI catalogue export is in place: `spo plugins catalog` emits discovered marketplace metadata, with optional incompatible-report inclusion for review jobs.

### Minor polish before v1.0

- Public benchmark suite with reproducible numbers against reference Strogatz/Pikovsky-style implementations, including commands, environment labels, and snapshot dates.
  - Benchmark metadata/snapshot slice is in place: `benchmarks/reference_suite.py`
    emits command, backend, Python/NumPy versions, platform, executable, and
    snapshot date with `benchmarks/results/reference_suite.json`.
  - Public snapshot page is in place:
    `docs/galleries/reference_benchmark_snapshot.md` publishes the dated JSON
    values with reproduction metadata and use-policy warnings.
  - Remaining before claiming complete: expand published comparison coverage
    beyond the current host/backend as needed.
- Windows Rust FFI fully stable; remove the experimental label only after installation, CI, and parity evidence support it.
- One end-to-end hardware example with real FPGA or neuromorphic output, including command, generated artefact, and verification result.

### Long-horizon differentiator backlog

Near-term candidate tracks:

- Dynamic higher-order topology adaptation in the supervisor: edit hyperedges and 2-/3-simplices based on coherence, Lyapunov, transfer-entropy, or policy objectives.
  - Foundation is in place: topology adaptation supervisor support exists.
  - Plasma-control demo and pairwise-support policy hardening are in place:
    `domainpacks/plasma_control/topology_adaptation_demo.py` emits guarded
    topology audit payloads and `TopologyMutationPolicy` can require existing
    pairwise support before creating a new 2-simplex.
  - Traffic-flow transfer-entropy demo is in place:
    `domainpacks/traffic_flow/topology_adaptation_demo.py` derives pairwise
    support from transfer-entropy histories before proposing corridor
    simplices and records Lyapunov before/after energy evidence for the
    proposed mutation.
  - Remaining scope: broader multi-domain Lyapunov policy validation.
- Causal intervention engine: attach counterfactual UPDE/JAX rollouts to material regime transitions and knob changes, maintain a live N-channel causal model, and audit observed plus counterfactual trajectories.
  - Foundation is in place: causal counterfactual rollout support exists.
  - Live causal-model learning foundation is in place:
    `learn_causal_graph()` estimates signed directed edges from lagged monitor
    traces and records explicit `do(knob:scope) -> R` edges from paired
    counterfactual rollouts.
  - Cardiac attribution demo is in place:
    `domainpacks/cardiac_rhythm/causal_attribution_demo.py` compares a
    pacing-drive candidate against a no-action ventricular-disturbance
    baseline and emits trajectory plus attribution audit records.
  - Power-grid attribution demo is in place:
    `domainpacks/power_grid/causal_attribution_demo.py` compares a governor
    droop coupling candidate against a no-action load-step baseline.
  - Remaining scope: larger causal-model learners and additional attribution
    demonstrations for other domain families.
- FEP / predictive-coding supervisor backend: promote ActiveInferenceAgent into a supervisor mode that treats UPDE as the generative process and minimises variational free energy across N-channel hierarchy.
  - Foundation is in place: predictive FEP supervisor support exists.
  - Reusable hierarchy assessment is in place:
    `assess_fep_hierarchy()` runs child FEP supervisors, reduces child
    coherence into a parent phase vector, and emits an audit-ready
    `FEPHierarchyAssessment`.
  - Power-grid hierarchy proof is in place:
    `domainpacks/power_grid/fep_hierarchy_demo.py` runs two child FEP
    supervisors and a parent supervisor over reduced child coherence.
  - Cardiac hierarchy proof is in place:
    `domainpacks/cardiac_rhythm/fep_hierarchy_demo.py` runs pacemaker/atrial
    and ventricular/recovery child axes into a parent cardiac supervisor.
  - Remaining scope: deeper predictive-coding world-model integration.
- STL runtime verification: augment the policy DSL with Signal Temporal Logic formulas, robustness metrics, monitoring automata, audit satisfaction traces, and controller-synthesis linkage. The builtin monitoring automata foundation is implemented; controller synthesis remains open.
  - Foundation is in place: built-in STL robustness monitoring exists.
  - Policy DSL integration is in place: policy YAML can declare
    `stl_monitors`, load them with `load_policy_stl_specs()`, evaluate traces
    with `evaluate_policy_stl_specs()`, and emit audit-ready satisfaction
    records.
  - Model-checker export linkage is in place:
    `spo formal-export --export stl` emits PRISM constants and
    satisfied/violated labels for policy-declared STL monitors.
  - Monitoring automata synthesis foundation is in place:
    `synthesise_stl_monitoring_automaton()` emits audit-ready state and
    transition traces for builtin simple STL monitors.
  - Remaining scope: controller synthesis from STL monitor automata.
- Symbolic-to-binding compiler: generate reviewable `binding_spec.yaml`, policy DSL, and notebook drafts from natural-language domain intent plus local retrieval over docs and domainpacks.
  - Foundation is in place: symbolic binding compiler support exists.
  - Retrieval, confidence, and notebook polish are in place: generation now
    records local domainpack retrieval evidence, confidence factors, and a
    `review_notebook.ipynb` validation notebook.
  - Remaining scope: richer retrieval over long-form docs and generated
    notebook execution evidence.
- Cross-domain meta-transfer: learn a latent policy/binding space from audit histories and propose zero-shot or few-shot initial policies for new domains.
  - Foundation is in place: replay-backed meta-transfer proposals exist.
  - Multi-audit fitting and package export are in place:
    `CrossDomainMetaTransfer.fit_audit_history()` aggregates multiple audit
    JSONL files, exposes an audit-ready training summary, and round-trips
    deterministic JSON packages for proposal jobs.
  - Remaining scope: larger real audit-history training corpora and optional
    `scpn-meta` packaging.
- Quantum-native compiler target: output Qiskit/PennyLane or OpenPulse/QASM fragments for the `quantum_simulation` path with co-simulation validation.
- Neuromorphic compiler target: emit Lava/PyNN or hardware-oriented HDL fragments from binding specs and supervisor policies, with simulator parity evidence.

Speculative research watchlist:

- Self-modelling embodied digital twin with a `self_model_error` monitor and controlled rebinding/reconfiguration regime.
- Evolutionary supervisor policy search over policy DSL, Petri nets, and topology mutations, initially offline over `audit.jsonl` with STL and counterfactual safety filters.
- Information-geometry control layer using Fisher-Rao or Wasserstein metrics as supervisor control primitives.
- Sheaf-cohomology control over N-channel states, with sheaf Laplacian and obstruction metrics.
  - Foundation is in place: sheaf coherence supervisor support exists; remaining scope is heterogeneous-domain demos and obstruction hardening.
- Federated meta-orchestrator with differential-privacy policy-gradient aggregation across edge nodes.
- Byzantine-fault-tolerant meta-orchestrator fabric over signed policy/topology proposals and hash-linked audit evidence.
- Hybrid neuromorphic-quantum co-compiler with shared N-channel audit semantics.
- Value-alignment supervisor guard encoded as binding-spec objectives and Petri-net guard conditions.
  - Foundation is in place: value-alignment guard support exists; remaining scope is binding-spec objective templates and counterfactual violation reporting.
- Autopoietic lineage sandbox for resource-bounded child-policy evolution over audit replays, merging only through reviewable diffs.
- Temporal-causal hypergraph experiments, explicitly gated as research until conventional causal baselines are beaten.
- Intergenerational policy inheritance: signed lineage metadata for child orchestrators, inherited policy genomes, multi-objective replay fitness, and merge-only reviewed hot patches.
- Sheaf-theoretic coherence manifold: obstruction-aware control primitive over a sheaf Laplacian with audit-visible cohomology dimensions.
- Constitutional value-alignment guard: Pareto objective constraints in binding specs, counterfactual violation logs, and forced safe fallback path.
- Strange-loop meta-orchestrator: self-referential supervisor channel that monitors and damps policy drift, over-control, or control-loop oscillation.
  - Foundation is in place: strange-loop supervisor monitor exists; remaining scope is long-run drift scenarios and studio surfacing.
- Morphogenetic field topology: reaction-diffusion-style field over coupling topology with grow/shrink primitives and field snapshot audit records.
  - Foundation is in place: morphogenetic topology field support exists; remaining scope is field snapshot visualisation and domainpack demos.
- Integrated-information monitor: approximate Phi-style global integration metric exposed as a monitor, not as a consciousness claim.
  - Foundation is in place: integrated-information monitor exists; remaining scope is benchmarked approximations.
  - Reporting integration is in place: `build_audit_report_summary()` and `spo report` summarise passive `integrated_information` audit records while preserving the engineering-proxy claim boundary.
- Topos-theoretic semantic binding: categorical validation prototype for binding and policy composition with explicit proof obligations.
- Multiverse counterfactual simulator: vectorised JAX branch rollouts over knob/topology ensembles before committing high-risk actuation.
- Entanglement-aware hybrid order parameters: quantum co-simulation monitor that reports entanglement entropy alongside classical `R` and `Psi`.

Priority order for first implementation tranche:

1. ~~Causal intervention engine foundation.~~
2. ~~STL runtime verification foundation.~~
3. ~~Dynamic higher-order topology adaptation foundation.~~
4. ~~FEP / predictive-coding supervisor backend foundation.~~
5. ~~Symbolic-to-binding compiler foundation.~~

Speculative priority order after first tranche:

1. ~~Strange-loop meta-orchestrator foundation.~~
2. ~~Morphogenetic field topology foundation.~~
3. ~~Integrated-information monitor foundation.~~
4. ~~Sheaf-theoretic coherence manifold foundation.~~
5. ~~Constitutional value-alignment guard foundation.~~
