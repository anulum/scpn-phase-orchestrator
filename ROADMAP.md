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
- Expand domainpack channel coverage (target: all 3 channels in 12+ packs): in progress; 10/24 audited, carry remaining channel-coverage work into the N-channel rollout track.
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
  - Use the local internal audit ledger as the candidate source.
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
  - Channel algebra summary foundation is in place: `build_channel_algebra_report()` exposes required/optional channels, derived channels, group membership, supervisor visibility, coupling participation, and cross-channel edges for audit/reporting.
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
  - Runtime execution is in place for delayed/uncertain channels:
    `ChannelRuntimeExecutor` applies held-previous-tick evidence and
    confidence-weighted layer contributions before supervisor decisions, with
    raw-versus-executed evidence written to audit logs.
- Extend optimisation surfaces to include channel weights and cross-channel coupling parameters, not only `K`, `alpha`, `zeta`, and `Psi`.
- Treat Rust and JAX as primary execution paths. Keep Julia, Go, Mojo, and other auxiliary backends experimental unless a maintained production workload shows a 5-10x gain or a capability Rust/JAX cannot provide.
- ~~Document the backend fallback chain in one place, including feature flags, runtime detection, numerical tolerance, benchmark evidence, and deprecation criteria.~~ (done — `docs/guide/backend_fallbacks.md`)
- ~~Add a multi-language backend review gate before each minor release: keep, demote to experimental, or remove based on maintenance cost, CI burden, and measured value.~~ (done — `docs/guide/backend_review_gate.md`, non-destructive default with explicit sign-off for removal)
- Extend visualisation beyond static matplotlib and the current WASM surface: Plotly/Dash dashboards for production operators, real-time streaming plots, and optional 3D views for swarm, traffic, robotics, and spatial domainpacks.

### v1.x differentiators

- ML-driven auto-binding and oscillator discovery: ingest raw multimodal data, propose P/I/S or N-channel extractors, infer an initial coupling graph, and emit a reviewable binding spec.
- Auto domain binding pipeline: turn raw time-series, event logs, and sensor streams into candidate oscillator families, extractor parameters, initial coupling matrices, and a scored `binding_spec.yaml` proposal. The deterministic review-only foundation is in place; graph-learning and initial-`K` evidence remain non-actuating until larger live-dataset benchmarks justify domain-specific acceptance thresholds.
  - Transfer-entropy causal coupling inference is in place as
    `auto-coupling-estimation`, returning source-to-target `K_nm` estimates,
    support masks, audit records, and UPDE-orientation conversion from phase
    time series.
- RL and hybrid optimisation layer for knob tuning: leverage the JAX `nn/` backend and `autotune` module to learn policies for `K`, `alpha`, `zeta`, `Psi`, and channel weights from rewards based on coherence metrics minus penalties for `R_bad`, unsafe actuation, and regime churn. Initial scope: replay-trained model-free or hybrid PPO/SAC experiments that emit auditable policy candidates rather than direct production control.
  - Reward-evaluation foundation is in place: `autotune.reward` scores candidate knob policies from replay/simulation observations and emits audit-ready records before any learner can actuate.
  - Replay candidate ranking is in place: `rank_replay_candidates()` orders replay/simulation candidates by reward, filters unsafe rollouts by default, and returns audit-ready reports.
  - Offline policy-search generation is in place: `generate_offline_policy_candidates()` creates deterministic coordinate-search candidates around a seed policy for replay scoring.
  - Replay-trained proposal records are in place: `propose_replay_policy()` applies review gates and serialises accept/reject rationale before any live learner or actuation loop.
  - Replay-only policy search is in place: `search_replay_policy()` binds deterministic candidate generation to a caller-supplied replay/simulation evaluator and returns an audit-ready proposal.
  - Adaptive replay-only refinement is in place: `search_adaptive_replay_policy()` performs bounded multi-round replay search with decayed coordinate steps and the same proposal gates.
  - PPO-like, SAC-like, and hybrid-physics learner proposal generators are in
    place behind the same non-actuating replay gates; deterministic
    multi-scenario replay learner benchmark gates are in place, while real
    learner dependencies and benchmarked trained policies remain future work.
  - N-channel optimisation surface is in place for replay-only searches:
    candidate generation, reward scoring, proposal records, and adaptive
    search now include channel weights plus cross-channel coupling gains before
    any learner-backed actuation is allowed.
- Trainable supervisor policies: extend rule-based policy evaluation with reinforcement learning or active-inference loops that optimise long-horizon `R_good` / `R_bad` trade-offs under replayable safety constraints.
- Uncertainty-aware phase estimation: Bayesian or ensemble phase estimates propagated through MPC/OA reduction and supervisor decisions.
  - Bayesian UPDE uncertainty propagation is in place for sampled `omega` and
    `K_nm` distributions, returning posterior-predictive `R ± sigma`,
    credible intervals, and audit records through the existing UPDE kernel.
- SPO Studio GUI: web-based binding and policy builder that scaffolds, visualises, validates, and replays binding specs, with WASM-backed previews where useful.
  - Streamlit operator surface is in place for domainpack loading, raw-source
    import, binding review, beginner-mode guidance, oscillator edit review
    artefacts, ordered beginner walkthroughs, live `R`/`Psi`/`K` metrics,
    replay-only knob tuning, hierarchy monitor, layer/channel canvas review
    artefacts, canvas layout manifests,
    canvas topology patch artefacts, validated binding rewrite candidates,
    signed hash-checked binding apply with backups, deterministic Canvas
    interaction state for browser controls, connector ownership plans,
    owned no-I/O runtime boundary validation for REST/gRPC/Kafka/hardware,
    localhost service-process manifests with health checks, deployment
    readiness, deployment package
    manifests, dry-run connector run records, package materialisation plans,
    hardware target package manifests, verified hardware evidence packages,
    recovery reports, and
    review/deploy export manifests. This is a validated operator prototype, not
    a finished product-grade Studio: real FPGA/neuromorphic evidence remains
    future product work.
- Hierarchical multi-scale orchestration: support nested orchestrators where local/edge supervisors maintain local coherence, exchange reduced phase/coherence summaries, and escalate only bounded regime evidence to a parent supervisor. Reuse Hodge decomposition and transfer-entropy monitors to decide what crosses hierarchy boundaries.
  - Reduced-summary hierarchy foundation is in place: `build_hierarchical_orchestration_plan()` turns child supervisor summaries into a parent `UPDEState` and bounded escalation audit records without exchanging raw child signals.
  - Transport-neutral hierarchy sync envelopes are in place: `build_hierarchy_sync_envelope()` and `ingest_hierarchy_sync_envelopes()` provide deterministic JSON-safe edge/cloud summary exchange with protocol-version and sequence checks.
  - Strict non-socket `HierarchyTransportRuntime` validation and decoded
    JSONL/REST/frame adapter boundaries are in place for reviewable
    live-adapter handoff. Power-grid, cardiac-rhythm, and edge-consensus domainpacks now include `hierarchy_transport_demo.py` transport-demos with JSONL/REST/frame smoke coverage. Remaining scope is owned live transports and deployment paths.
- Distributed edge orchestration: multi-node phase consensus with gossip or local Kuramoto coupling, plus WASM/FPGA deployment paths for decentralised operation.
  - Offline hierarchy sync-envelope ingestion and two domainpack replay demos are in place for reduced summaries.
  - Deterministic offline gossip/local-consensus replay is in place via `simulate_hierarchy_gossip_consensus()`, using accepted sync envelopes and caller-supplied neighbour maps without sockets or live actuation. Live transport remains open.
  - Phase-vector gossip protocol foundation is in place:
    `PhaseGossipNode` and `PhaseSyncMessage` provide canonical JSON wire
    messages, digest verification, sequence watermarks, dimension checks,
    peer timeout handling, bounded circular phase correction, and
    deterministic lossy-network replay for UPDE node phase states. Owned live
    transport adapters and cross-machine deployment demos remain open.
- Digital-twin binding standard: version `binding_spec.yaml` as an open bidirectional live-sync contract for simulators, services, and hardware twins.
  - Digital-twin binding contract foundation is in place: `build_digital_twin_binding_contract()` emits deterministic timing, layer, actuator, N-channel algebra, sync-capability, and contract-hash payloads without opening transport or applying control.
  - Transport-neutral payload validation is in place: digital-twin sync envelopes validate contract hashes, declared capabilities, directions, sequence numbers, and non-empty payloads before a REST/gRPC/Kafka/file/hardware adapter hands data to runtime code.
  - JSONL file replay adapter is in place for deterministic offline adapter smoke tests, separating accepted validations from malformed JSON, invalid envelope shapes, and contract-validation rejections.
  - In-memory reference adapter is in place for runtime-facing tests: accepted envelopes queue in submission order, rejected envelopes return validation reasons, and no disk or network transport is touched.
  - Adapter manifest compatibility checks are in place for concrete transport review: manifests declare transport type, sync capabilities, replay support, and auth posture before adapter code is enabled.
  - Dependency-free REST boundary adapter is in place: framework route handlers can submit parsed JSON and headers through manifest, auth, envelope-shape, contract, direction, and payload validation without this module opening sockets.
  - Dependency-free gRPC boundary adapter is in place: servicers can submit decoded unary request fields and metadata through manifest, auth, envelope-shape, contract, direction, and payload validation without this module opening a gRPC server.
  - Dependency-free Kafka boundary adapter is in place: broker consumers can submit decoded message records and headers through topic, manifest, auth, envelope-shape, contract, direction, and payload validation without this module importing Kafka clients or committing offsets.
  - No-I/O hardware boundary adapter is in place: hardware integrations can submit decoded frames through registered-device, safety-interlock, manifest, auth, envelope-shape, contract, direction, and payload validation while the binding layer keeps `hardware_write_permitted` false.
- Formal verification hooks: export Petri-net regimes and policy rules to PRISM, TLA+, SPIN, or equivalent model-checking workflows, with CI artefacts for safety-critical policies.
- Neuromorphic and quantum-native backends: extend existing SNN and quantum-control bridges so the orchestrator can emit Lava/BrainScaleS-style neuromorphic schedules and QPU control schedules directly from validated phase-control plans.
- Extractor and actuator plugin ecosystem: define a stable Python/Rust plugin interface for custom `PhaseExtractor` and `ActuationMapper` implementations, including schema validation, entry-point discovery, audit metadata, compatibility tests, and versioned capability declarations.
  - Python manifest foundation is in place: `plugins.registry` provides entry-point discovery, versioned capability declarations, compatibility reports, audit records, tests, and API documentation.

### Next high-impact differentiated moves

These are the next moves to keep on the active TODO surface. They overlap with
existing v1.x tracks, but are listed here as operator-visible outcomes so future
sessions do not treat them as abstract research labels.

- Auto-discovery / data-driven binding:
  - Ingest raw streams such as CSV, sensor logs, and events.
  - Auto-propose oscillators, channels, initial `K`, extractor parameters, and a
    reviewable `binding_spec.yaml`.
  - CLI proposal export is in place: `spo auto-bind` turns local time-series
    CSV, event-log JSON, or graph JSON inputs into review-only binding YAML or
    audit JSON without writing files or enabling actuation.
  - Time-series proposal evidence is in place: regular time columns can infer
    sampling rate with no CLI parameter, and provenance records deterministic
    sparse-derivative, phase-aware Kuramoto SINDy, correlation-graph, and
    clustering evidence for review.
  - Multi-library SINDy selection evidence is in place: fitted affine and
    phase-aware SINDy candidates are scored with residual RMSE and a BIC-style
    complexity penalty before a review-only selected library is recorded.
  - Learned graph inference evidence is in place: lagged sparse linear
    prediction emits directed source-to-target graph edges, residual evidence,
    and graph density for review without binding actuation.
  - Auto-coupling estimation is in place: `spo auto-coupling-estimation`
    ingests CSV or `.npy` phase tables and emits transfer-entropy coupling
    matrices with deterministic audit JSON for review before binding use.
  - Extractor-parameter proposals and initial `K` binding are in place:
    time-series auto-binding writes per-family source-column, sampling, and
    statistic configs into the generated YAML, exposes the same records in
    JSON provenance, and emits a validator-accepted `auto_initial_k` template
    plus `cross_channel_couplings` for operator review.
  - Synthetic reference-suite proposal-quality benchmarking is in place:
    `benchmarks/reference_suite.py` now records extractor coverage, validator
    acceptance, expected initial-K support recall, proposed edge count, and
    throughput for deterministic auto-binding fixtures.
  - Larger proposal-quality gates are in place:
    `benchmarks/reference_suite.py` now evaluates four deterministic
    domain-like fixtures (`phase_chain`, `industrial_sensor_chain`,
    `cardiac_rhythm_surrogate`, and `power_grid_surrogate`) with explicit
    domain-specific thresholds for extractor coverage, expected-edge recall,
    validation errors, sample count, and proposed-edge multiplier. The
    benchmark snapshot records pass/fail evidence for all domain gates before
    any automatic runtime use.
  - Remaining work: add private or partner-provided live datasets when they are
    available and preserve the same threshold-gated, review-only acceptance
    contract.
  - Acceptance: the five-minute new-dataset workflow is zero-config except for
    operator review, with validation diagnostics, deterministic replay, and
    benchmarked graph evidence.
- RL / optimisation on knobs:
  - Use JAX `nn/` plus `autotune` to learn replay-only or simulator-backed
    policies for `K`, `alpha`, `zeta`, `Psi`, channel weights, and cross-channel
    gains.
  - Optimise `R_good - penalty(R_bad, safety, regime churn)` and hybridise
    learned proposals with supervisor rules.
  - Benchmark-gated replay learner evidence is in place:
    `benchmarks/reference_suite.py` evaluates PPO-like, SAC-like, and
    hybrid-physics replay proposals across deterministic multi-scenario
    simulator-backed coherence gates. The snapshot records scenario acceptance,
    learner acceptance rate, minimum coherence improvement, unsafe accepted
    candidates, and non-actuating status before any policy can be considered
    for operator review.
  - Acceptance: PPO-like, gradient-based, or hybrid physics learners produce
    benchmarked, auditable policy candidates that remain non-actuating until
    explicit safety gates pass.
- Bayesian / uncertainty-aware control:
  - Propagate uncertainty in natural frequencies and coupling matrices through
    UPDE rollouts before supervisor review.
  - Foundation is in place: `bayesian_upde_run()` samples deterministic or
    Gaussian `omega`/`K_nm` distributions and emits `R ± sigma`, credible
    intervals, sampled final phases, and JSON-safe audit diagnostics.
  - Posterior fitting from observed Kuramoto phase trajectories is now in
    place through `fit_gaussian_upde_posterior()`, including non-negative
    zero-diagonal coupling enforcement, JSON-safe fit diagnostics, and
    reference-suite acceptance gates for residual quality, parameter recovery,
    uncertainty width, and posterior rollout sample count.
  - Backend-name safety is benchmark-gated: `audit_bayesian_backend_status()`
    proves NumPy execution and records deterministic fail-closed audit evidence
    for reserved `numpyro` and `blackjax` sampler names.
  - Remaining scope: implement, validate, and benchmark real NumPyro or
    BlackJAX samplers before changing their fail-closed status.
- Hierarchical and distributed orchestration:
  - Make nested orchestrators explicit: edge supervisors sync locally, report
    reduced aggregates upward, and escalate only bounded evidence.
  - Build on Hodge, transfer entropy, N-channel summaries, and the current
    hierarchy adapter/runtime boundaries.
  - Acceptance: live owned transports and multi-domain distributed demos pass
    replay, sequence, protocol, and safety validation.
- Formal verification export:
  - Export Petri nets, supervisor transitions, policy DSL, and STL monitors to
    PRISM, SPIN, TLA+, SMT, or equivalent proof workflows.
  - PRISM/TLA/STL export evidence is now benchmark-gated:
    `benchmark_formal_export_artifact_quality()` emits Petri, policy, and STL
    artefacts, records deterministic SHA-256 evidence, counts identifier maps,
    and proves malformed nets, policy rules, and STL predicates fail closed
    before text generation.
  - Domain-style formal safety evidence is now benchmark-gated:
    `benchmark_domain_formal_safety_exports()` emits deterministic policy
    PRISM, policy TLA, and STL PRISM artefacts for plasma-control, power-grid,
    and medical/cardiac-style profiles with per-domain acceptance results.
  - Formal verification package manifests are in place:
    `build_formal_verification_package()` records exported artefact hashes,
    named safety properties, and exact external PRISM/TLC checker commands
    without writing files or invoking model checkers locally.
  - Formal package benchmark evidence is in place:
    `benchmark_formal_export_artifact_quality()` now gates package property
    counts, checker command counts, disabled checker execution, and package
    hash determinism alongside PRISM/TLA/STL export artefacts. The same gate
    now records deterministic PRISM/TLC checker readiness status, available and
    missing checker counts, and disabled availability-audit execution.
  - CLI formal package export is in place:
    `spo formal-export --export package` emits the no-execution JSON manifest
    for protocol PRISM/TLA and policy PRISM artefacts. The package export can
    now include non-executing checker-readiness records through
    `--include-checker-readiness`, with deterministic `--checker-path`
    overrides for CI evidence.
  - External-checker readiness audits are in place:
    `audit_formal_checker_availability()` records PRISM/TLC command
    executable availability as deterministic, non-executing audit records with
    ready/missing status, resolved path evidence, and execution disabled.
  - Acceptance: plasma, power, and medical-style policies have reproducible
    safety artefacts suitable for credibility reviews.
- Neuromorphic and quantum tighter integration:
  - Surface partner bridges by emitting Lava/BrainScaleS-style schedules and QPU
    control schedules from validated phase-control plans.
  - Hybrid co-compiler review evidence is now benchmark-gated:
    `benchmark_hybrid_cocompiler_review_gate()` links deterministic quantum
    compiler and neuromorphic schedule manifests, records component hashes,
    proves shared target-backend coverage, and verifies execution/write
    permissions remain disabled under blocked probes.
  - Acceptance: execution stays disabled until real target evidence, hashes,
    parity reports, and operator approval exist.
- Plugin ecosystem:
  - Standardise Python and Rust extension interfaces for custom extractors,
    monitors, actuators, and bridges.
  - Monitor capability declarations are now part of the validated manifest
    surface and must declare channels; marketplace and Rust-registry benchmark
    gates cover extractor, monitor, actuator, and bridge metadata.
  - Acceptance: domain experts can ship schema-validated, capability-declared,
    audit-visible plugins without core forks.
- Observability and digital-twin polish:
  - Promote Prometheus/Grafana live-run telemetry, adapter health, replay
    linkage, twin residuals, and mismatch evidence to first-class operator
    surfaces.
  - Digital-twin operator evidence is in place: live and replay validation
    records reduce to the same accepted/rejected counts, capability and
    direction counts, latest sequence, twin-residual extrema, mismatch reasons,
    adapter health, and operator status payload.
  - Runtime Prometheus exposition is in place for the shared operator evidence
    record, including sync accepted/rejected counts, adapter health, latest
    sequence, residual extrema, status, capability counts, direction counts,
    and mismatch-reason counts.
  - Acceptance: live and replayed runs expose the same digital-twin residual
    channels and operational evidence in dashboards and audit records.

### Usability moat — finish the job

- One-click SPO Studio web UI for new control engineers:
  - Domainpack load, raw-source import, binding review, oscillator edit review
    artefacts, beginner-mode guidance with ordered walkthroughs, layer/channel
    canvas review artefacts, canvas layout manifests, canvas topology patch
    artefacts, live `R`/`Psi`/`K` visualisation, replay-only knob tuning,
    hierarchy monitor, connector ownership plans, deployment
    readiness/package manifests, hardware target package manifests, recovery
    reports, dry-run connector run records, package materialisation plans,
    verified hardware evidence packages, validated binding rewrite candidates,
    signed hash-checked binding apply with backups, owned no-I/O runtime boundary
    validation for REST/gRPC/Kafka/hardware, deterministic Canvas interaction
    state for browser controls, localhost service-process manifests with health
    checks, and Docker/WASM/project export manifests are in place.
    This remains far from a good standalone product: current value is an
    auditable operator workflow and smoke-tested web surface, while true
    one-click product quality still needs real hardware evidence.
- Auto-binding prototype:
  - Deterministic proposal builders from time-series CSV, event-log JSON, and
    graph JSON to reviewable `binding_spec.yaml` records are in place with
    confidence factors and validation diagnostics.
  - CLI proposal export is in place: `spo auto-bind` emits review-only binding
    YAML or audit JSON from local raw sources without writing files or enabling
    actuation. Deeper SINDy/graph-learning inference remains experimental.
- RL/autotune layer on the JAX `nn` backend: PPO/SAC or hybrid physics-RL policies that learn `K`, `alpha`, `zeta`, and `Psi` from rewards such as coherence minus penalties for `R_bad`, unsafe actuation, and regime churn.
  - Reward-evaluation, replay ranking, offline candidate generation, proposal records, replay-only policy search, adaptive replay refinement, PPO-like/SAC-like/hybrid-physics proposal generators, and deterministic multi-scenario replay learner benchmark gates are in place behind non-actuating gates. Real learner dependencies and benchmarked trained policies remain future work.
- Full N-channel and hierarchical orchestration: channel algebra, nested supervisors, and edge/cloud synchronisation protocol for distributed coherence control.
  - N-channel runtime execution and replay-only optimisation surfaces are in place. Hierarchical reduced-summary parent orchestration, offline edge/cloud sync envelopes, strict non-socket runtime validation, decoded JSONL/REST/frame adapter boundaries, power-grid/cardiac replay demos, and deterministic offline gossip replay are in place; owned live transports and broader multi-domain demos remain open.
- Formal verification for supervisor: export Petri-net and policy surfaces to PRISM, TLA+, SPIN, or equivalent model-checking workflows for safety properties in critical regimes.
- Plugin ecosystem and marketplace: standard interfaces for domainpacks, extractors, monitors, actuators, and bridges so domain experts can publish extensions without forking the core repository.
  - Plugin manifest registry foundation is in place; deeper Rust runtime loading remains open.
  - Marketplace catalogue packaging is in place: `build_plugin_marketplace_catalog()` emits deterministic metadata-only catalogue payloads with compatibility records and capability counts.
  - Marketplace example is in place: `examples/plugin_marketplace_catalog.py` builds a validated extractor/actuator manifest and catalogue payload without loading plugin targets.
  - CLI catalogue export is in place: `spo plugins catalog` emits discovered marketplace metadata, with optional incompatible-report inclusion for review jobs.
  - Rust-facing registry export is in place: `build_rust_plugin_registry()` and
    `spo plugins catalog --rust-registry` emit flattened capability JSON for
    Rust-side dispatchers without importing plugin implementation targets.
  - Guarded Rust runtime handoff manifests are in place:
    `build_rust_plugin_runtime_handoff()` groups compatible capabilities for
    Rust dispatch, records deterministic target hashes, carries blocked
    incompatible capabilities for review, and keeps plugin loading disabled by
    default.
  - CLI guarded runtime handoff export is in place:
    `spo plugins catalog --rust-runtime-handoff` emits the same no-load handoff
    JSON and rejects conflicting Rust output modes.
  - Runtime handoff benchmark evidence is in place:
    `benchmark_plugin_ecosystem_catalog_quality()` now gates the guarded
    handoff for deterministic hashes, disabled loading, target hashes, and
    blocked incompatible capability records alongside marketplace and registry
    metadata.
  - Monitor capability support is in place: manifests, marketplace counts,
    compatibility reports, Rust-facing registry records, and the reference
    benchmark gate now require monitor metadata with declared channels.

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
  - Studio now validates pasted hardware evidence bundles for generated
    artefact hash, simulator parity hash/status, toolchain metadata, and
    operator sign-off before emitting a review-ready verified hardware package.
    The actual real-target artefact and parity report remain open.

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
  - Network-security Lyapunov validation proof is in place:
    `domainpacks/network_security/topology_adaptation_demo.py` derives
    pairwise support from transfer-entropy histories before proposing
    traffic/attack/defence simplices and records Lyapunov before/after energy
    evidence for the proposed mutation.
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
  - Traffic-flow attribution demo is in place:
    `domainpacks/traffic_flow/causal_attribution_demo.py` compares a
    signal-cycle coupling candidate against a no-action corridor-spillback
    baseline.
  - Network-security attribution demo is in place:
    `domainpacks/network_security/causal_attribution_demo.py` compares a
    firewall-coupling candidate against a no-action lateral-movement baseline.
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
  - TLA+ export linkage is in place for supervisor protocols and policy rules:
    `spo formal-export --export protocol-tla` and
    `spo formal-export --export policy-tla` emit bounded transition-system
    modules with `Init`, `Next`, `Spec`, and `Safety == TypeOK`.
  - Monitoring automata synthesis foundation is in place:
    `synthesise_stl_monitoring_automaton()` emits audit-ready state and
    transition traces for builtin simple STL monitors.
  - Controller-synthesis linkage foundation is in place:
    `synthesise_stl_controller_candidates()` emits non-actuating,
    audit-ready signal-level candidates from builtin STL monitor automata.
  - Policy-gated projection foundation is in place:
    `project_stl_controller_candidates()` maps candidates through explicit
    projection templates and `ActionProjector` into bounded, non-actuating
    `ControlAction` proposals with rejected-candidate reasons.
  - Offline closed-loop synthesis planning is in place:
    `synthesise_stl_closed_loop_plan()` binds STL automata, feedback signals,
    candidate synthesis, policy-gated projection, fail-closed blockers, and a
    future review horizon without mutating runtime state or enabling actuation.
  - Reference-suite evidence is in place:
    `benchmark_stl_closed_loop_plan_quality()` gates projected non-actuating
    plans, missing-template blockers, satisfied-monitor no-action behaviour,
    deterministic plan hashes, and zero actuation leaks.
  - Remaining scope: runtime integration that passes projected plans through
    the full safety/actuation stack.
- Symbolic-to-binding compiler: generate reviewable `binding_spec.yaml`, policy DSL, and notebook drafts from natural-language domain intent plus local retrieval over docs and domainpacks.
  - Foundation is in place: symbolic binding compiler support exists.
  - LLM-guided scaffold foundation is in place: `spo scaffold --llm` accepts
    natural-language intent, requires a configured provider or offline JSON
    proposal file, normalises strict JSON into deterministic binding YAML, and
    validates the result before writing a domainpack.
  - Retrieval, confidence, and notebook polish are in place: generation now
    records local domainpack plus long-form public-doc retrieval evidence,
    confidence factors, and a `review_notebook.ipynb` validation notebook.
  - Notebook preflight execution evidence is in place: generated audit records
    include compiler-side binding-schema and policy-loader checks matching the
    review notebook.
  - Retrieval ranking diagnostics are in place: every retrieval evidence record
    now carries a deterministic rank plus source-priority, matched-term,
    name/phrase-match, prompt-term, and term-density features for audit review.
  - Remaining scope: optional corpus expansion and larger retrieval-quality
    benchmark gates.
- Cross-domain meta-transfer: learn a latent policy/binding space from audit histories and propose zero-shot or few-shot initial policies for new domains.
  - Foundation is in place: replay-backed meta-transfer proposals exist.
  - Multi-audit fitting and package export are in place:
    `CrossDomainMetaTransfer.fit_audit_history()` aggregates multiple audit
    JSONL files, exposes an audit-ready training summary, and round-trips
    deterministic JSON packages for proposal jobs.
  - Nested audit-directory corpus loading is in place:
    `CrossDomainMetaTransfer.fit_audit_directory()` discovers `**/*.jsonl`
    histories recursively for larger multi-domain replay corpora.
  - Remaining scope: larger real audit-history training corpora and optional
    `scpn-meta` packaging.
- Quantum-native compiler target: deterministic OpenQASM 3 compiler manifests
  are in place for Qiskit/PennyLane handoff, with Z-frequency terms,
  symmetrised XY coupling terms, co-simulation parity evidence, SHA-256 hashes,
  and QPU execution disabled. Real QPU target execution remains open.
- Neuromorphic compiler target: deterministic Lava/PyNN schedule manifests
  from `UPDEState` are in place with population records, projections,
  control-action review records, simulator parity evidence, SHA-256 coverage,
  and disabled hardware writes. Real neuromorphic target execution remains open.

Speculative research watchlist:

- Self-modelling embodied digital twin with a `self_model_error` monitor and controlled rebinding/reconfiguration regime.
- Evolutionary supervisor policy search over policy DSL, Petri nets, and topology mutations, initially offline over `audit.jsonl` with STL and counterfactual safety filters.
- Information-geometry control layer using Fisher-Rao or Wasserstein metrics as supervisor control primitives.
- Sheaf-cohomology control over N-channel states, with sheaf Laplacian and obstruction metrics.
  - Foundation is in place: sheaf coherence supervisor support exists.
  - Heterogeneous edge-consensus demo is in place:
    `domainpacks/edge_consensus_nchannel/sheaf_obstruction_demo.py` compares
    nominal and gateway-stressed six-channel sections across directed
    restriction maps.
  - Power-grid heterogeneous replay is in place:
    `domainpacks/power_grid/sheaf_obstruction_demo.py` compares nominal and
    line-fault grid sections across rotor-angle, frequency, tie-flow, demand,
    and renewable-ramp channels.
  - Network-security heterogeneous replay is in place:
    `domainpacks/network_security/sheaf_obstruction_demo.py` compares nominal
    and lateral-movement sections across traffic-rate, threat-level,
    defence-phase, and trust-score channels.
  - Obstruction hardening foundation is in place:
    `build_sheaf_obstruction_summary()` classifies nominal/warning/critical
    obstruction severity and reports strongest residual edges for audit triage.
  - Remaining scope: additional heterogeneous-domain demos and deeper
    obstruction hardening.
- Federated meta-orchestrator with differential-privacy policy-gradient aggregation across edge nodes.
- Byzantine-fault-tolerant meta-orchestrator:
  offline three-node BFT consensus manifests verify signed policy proposals,
  hash-linked audit parents, quorum winners, rejected nodes, and non-actuating
  review gates. Live distributed transport remains open.
- Hybrid neuromorphic-quantum co-compiler:
  deterministic hybrid manifests combine quantum compiler and neuromorphic
  schedule artefacts under shared N-channel audit semantics, component hashes,
  parity status, and disabled execution gates. Real hybrid target execution
  remains open.
- Value-alignment supervisor guard encoded as binding-spec objectives and Petri-net guard conditions.
  - Foundation is in place: value-alignment guard support exists.
  - Binding-spec templates are in place: `value_alignment` maps can be loaded
    from domainpack binding specs and converted with
    `value_alignment_policy_from_binding_spec()`.
  - Counterfactual reporting is in place: audit records distinguish hard bound
    violations from score-threshold fallbacks.
  - Cardiac rhythm domainpack prior template is in place for review-time
    pacing, coupling, and target-phase actuation guards.
  - Power-grid domainpack prior template is in place for review-time
    governor, AGC-bias, load-shed, and curtailment actuation guards.
  - Autonomous-vehicle domainpack prior template is in place for review-time
    platoon-coupling and throttle-drive actuation guards.
  - Satellite constellation domainpack prior template is in place for
    review-time PLL-coupling and beam-steering actuation guards.
  - Power-safety N-channel domainpack prior template is in place for
    review-time grid-coupling and substation-lag actuation guards.
  - Network-security domainpack prior template is in place for review-time
    firewall-coupling and defence-drive actuation guards.
  - Financial-markets domainpack prior template is in place for review-time
    cross-asset-coupling and rebalance-lag actuation guards.
  - Chemical-reactor domainpack prior template is in place for review-time
    coolant-flow, feed-rate, agitator, and jacket-setpoint actuation guards.
  - Manufacturing SPC domainpack prior template is in place for review-time
    station-coupling, sensor-lag, and damping-drive actuation guards.
  - Robotic CPG domainpack prior template is in place for review-time
    gait-coupling, stride-frequency, and phase-bias actuation guards.
  - Swarm-robotics domainpack prior template is in place for review-time
    alignment-coupling, formation-drive, obstacle-avoidance, and target-heading
    actuation guards.
  - Traffic-flow domainpack prior template is in place for review-time
    cycle-coupling, offset-drive, signal-split, and metering-target actuation
    guards.
  - Plasma-control domainpack prior template is in place for review-time
    transport-coupling, turbulence-lag, and feedback-damping actuation guards.
  - Fusion-equilibrium domainpack prior template is in place for review-time
    equilibrium-coupling and auxiliary-drive actuation guards.
  - Neuroscience EEG domainpack prior template is in place for review-time
    coupling, delta-lag, entrainment-drive, and target-phase actuation guards.
  - Brain-connectome domainpack prior template is in place for review-time
    coupling, neuromodulation-drive, stimulation-phase, and frontoparietal
    target-lag actuation guards.
  - Sleep-architecture domainpack prior template is in place for review-time
    coupling, circadian-drive, and phase-advance actuation guards.
  - Circadian-biology domainpack prior template is in place for review-time
    zeitgeber-drive, meal-phase, inter-clock coupling, and behavioral-lag
    actuation guards.
  - Epidemic SIR domainpack prior template is in place for review-time
    NPI-drive, vaccination-coupling, mobility-restriction, and lockdown-phase
    actuation guards.
  - Agent-coordination domainpack prior template is in place for review-time
    task-redistribution and deadline-drive actuation guards.
  - Quantum-simulation domainpack prior template is in place for review-time
    exchange-coupling and microwave-drive actuation guards.
  - Identity-coherence domainpack prior template is in place for review-time
    context-retrieval and relationship-coupling actuation guards.
  - PLL-clock domainpack prior template is in place for review-time loop
    bandwidth, frequency-trim, reference-drive, and phase-target actuation
    guards.
  - Digital-twin N-channel domainpack prior template is in place for
    review-time twin-coupling and signed line-lag correction guards.
  - Edge-consensus N-channel domainpack prior template is in place for
    review-time edge-coupling and signed gateway-lag correction guards.
  - Firefly-swarm domainpack prior template is in place for review-time
    visual-coupling, ambient-light-drive, and flash-target actuation guards.
  - QueueWaves domainpack prior template is in place for review-time
    retry-lag, service-coupling, and damping-drive actuation guards.
  - Rotating-machinery domainpack prior template is in place for review-time
    speed-setpoint, blade-damper, and bearing-stiffness actuation guards.
  - Laser-array domainpack prior template is in place for review-time
    detuning-offset, phase-lock coupling, injection-current, and feedback-phase
    actuation guards.
  - Bio-stub domainpack prior template is in place for review-time
    inter-scale coupling, entrainment-drive, and reference-phase actuation
    guards.
  - Vortex-shedding domainpack prior template is in place for review-time wake
    coupling, blowing-rate, and splitter-angle actuation guards.
  - Musical-acoustics domainpack prior template is in place for review-time
    harmonic-coupling, tempo-drive, and tuning-offset actuation guards.
  - Gene-oscillator domainpack prior template is in place for review-time
    quorum-coupling and inducer-dose actuation guards.
  - Geometry-walk domainpack prior template is in place for review-time graph
    coupling guards over consensus-control actuation.
  - Metaphysics-demo domainpack prior template is in place for review-time
    signed-coupling, damping-drive, and target-phase actuation guards.
  - Minimal-domain prior template is in place for review-time baseline-coupling
    guards over the minimal regression actuator.
  - Remaining scope: production calibration of domainpack-specific prior
    thresholds against real deployment or replay evidence.
- Autopoietic lineage sandbox for resource-bounded child-policy evolution over audit replays, merging only through reviewable diffs.
- Temporal-causal hypergraph experiments, explicitly gated as research until conventional causal baselines are beaten.
- Intergenerational policy inheritance: signed lineage metadata for child orchestrators, inherited policy genomes, multi-objective replay fitness, and merge-only reviewed hot patches.
- Sheaf-theoretic coherence manifold: obstruction-aware control primitive over a sheaf Laplacian with audit-visible cohomology dimensions.
- Constitutional value-alignment guard: Pareto objective constraints in binding specs, counterfactual violation logs, and forced safe fallback path.
- Strange-loop meta-orchestrator: self-referential supervisor channel that monitors and damps policy drift, over-control, or control-loop oscillation.
  - Foundation is in place: strange-loop supervisor monitor exists; remaining scope is long-run drift scenarios and studio surfacing.
- Morphogenetic field topology: reaction-diffusion-style field over coupling topology with grow/shrink primitives and field snapshot audit records.
  - Foundation is in place: morphogenetic topology field support exists.
  - Field snapshot visualisation foundation is in place:
    `build_morphogenetic_field_snapshot()` emits dependency-free field
    statistics, ASCII heatmap rows, and strongest-edge records for audits,
    reports, or later UI rendering.
  - Passive SVG rendering foundation is in place:
    `render_morphogenetic_field_svg()` emits a dependency-free SVG heatmap and
    labelled top-edge review artefact without mutating policy, coupling, or
    actuation state.
  - Swarm-robotics domainpack demo is in place:
    `domainpacks/swarm_robotics/morphogenetic_field_demo.py` emits a
    deterministic split-flock field audit payload plus snapshot rows without
    live actuation.
  - Power-grid domainpack demo is in place:
    `domainpacks/power_grid/morphogenetic_field_demo.py` emits a deterministic
    stressed-grid replay with grown/shrunk topology-field edges and snapshot
    rows without live actuation.
  - Traffic-flow domainpack demo is in place:
    `domainpacks/traffic_flow/morphogenetic_field_demo.py` emits a deterministic
    corridor-spillback replay with grown/shrunk topology-field edges and
    snapshot rows without live actuation.
  - Plasma-control domainpack demo is in place:
    `domainpacks/plasma_control/morphogenetic_field_demo.py` emits a
    deterministic edge-localised stress replay with grown/shrunk topology-field
    edges and snapshot rows without live actuation.
  - Network-security domainpack demo is in place:
    `domainpacks/network_security/morphogenetic_field_demo.py` emits a
    deterministic lateral-movement replay with grown defence/normal-traffic
    edges and shrunk attack-vector edges without live actuation.
  - Remaining scope: additional domainpack demos and richer Studio UI rendering.
- Integrated-information monitor: approximate Phi-style global integration metric exposed as a monitor, not as a consciousness claim.
  - Foundation is in place: integrated-information monitor exists; remaining scope is benchmarked approximations.
  - Reporting integration is in place: `build_audit_report_summary()` and `spo report` summarise passive `integrated_information` audit records while preserving the engineering-proxy claim boundary.
  - Approximation benchmark foundation is in place:
    `benchmark_integrated_information_approximations()` emits deterministic
    independent/modular/phase-lag/noisy-lock/locked synthetic calibration
    cases with ordering margins, explicitly without hardware performance
    claims.
  - Remaining scope: broader empirical replay benchmark corpus.
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
