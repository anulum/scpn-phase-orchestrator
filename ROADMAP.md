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
- Petri net + PolicyEngine Rust port (spo-supervisor crate)

## v1.0

- Production-hardened with fuzz testing (Hypothesis profiles) and fault injection
- Full benchmark suite: Kuramoto reference (Strogatz 2000), Stuart-Landau (Pikovsky 2001), Petri net reachability
- ~~RK45 exhausted-retry fallback test coverage~~ (done)
- Complete API documentation with mkdocstrings autodoc for all public modules
- Docker multi-stage build with security scanning (Trivy/Grype)
- ~~BoundaryObserver configurable default severity~~ (done — defaults to hard with warning)
- ~~DP tableau deduplication between upde.rs and stuart_landau.rs~~ (done)
- Stable public API freeze with semver guarantees

### Deferred track (documented, not current focus)

- Typed-array contract sweep (Python type precision):
  - Approximately 700 loose `NDArray` signatures remain in `src/` and are still to be parameterised.
  - This remains a tracked maintenance task, but active execution focus is moved to other roadmap items.

### v1.0 adoption and credibility track

- Ship 5-6 end-to-end tutorial notebooks that start from raw sources and finish with run, visualisation, and deterministic replay:
  CSV sensor stream -> P channel, event log -> I channel, state-machine trace -> S channel, binding spec, engine run, supervisor decisions, actuation output, and `audit.jsonl` replay.
- Add a "minimal viable domainpack in 5 minutes" guide using bundled real sample data, from raw CSV/event/state inputs through scaffold, binding spec, run, visualisation, and replay.
- Add a high-level "why this knob does what" explainer for K, alpha, zeta, Psi, damping, delay, coupling priors, supervisor thresholds, and actuation limits, aimed at users who do not already know Kuramoto control theory.
- Publish short video walkthroughs for first run, binding-spec authoring, policy debugging, audit replay, and deployment profiles.
- Add a visual binding-spec editor as an optional development extra. First acceptable version: load/save `binding_spec.yaml`, validate schema, expose P/I/S channel mappings, preview extractor outputs, and produce a minimal reproducible domainpack.
- Add an interactive supervisor-policy editor and validation loop for the DSL: structured rule builder, trigger/action autocomplete, cooldown/rate-limit previews, schema diagnostics, dry-run evaluation against `audit.jsonl`, and warnings for unreachable or overlapping rules.
- Reduce hidden YAML behaviour by documenting every inferred default in generated docs and surfacing resolved runtime configuration in CLI output and audit metadata.
- Make the mkdocs site the primary entry point: one-page "how the pipeline fires" diagram mapping YAML -> extractors -> engines -> supervisor -> actuation, plus autodoc coverage for every public module.
- Publish head-to-head benchmark pages for domainpacks against domain-specific baselines where appropriate, starting with power-grid swing-equation solvers and cardiac rhythm references.
- Add reproducible build locks for application and development environments. Evaluate `uv` and `pip-tools`; keep whichever produces maintainable, hash-pinned locks across Linux, macOS, Windows, and CI.
- ~~Reduce setup friction with documented install profiles: Python-only, Rust FFI, JAX, Docker, and experimental auxiliary backends. Each profile needs a preflight command that reports missing toolchains, optional dependency status, and expected fallback behaviour.~~ (done — `docs/guide/install_profiles.md`)
- ~~Harden Docker deployment with a documented multi-stage image, explicit production defaults, and CI security scans using Trivy or Grype.~~ (done — `Dockerfile`, `docs/guide/production.md`, `.github/workflows/publish.yml` Trivy scan gate)
- Keep adapters thin and fuzzed: `hardware_io`, Modbus, OPC-UA, ROS2, Kafka, and related network/file adapters need schema fuzzing, path-scrub tests, and production-default auth/rate-limit examples.
- Close remaining `nn/` validation xfails/skips before v1.0 unless each has an issue reference, owner, and release-blocking decision.
- ~~Make N-channel visible in the first-run experience. Ship two or three example domainpacks that use more than P/I/S, including cross-channel coupling, derived channels, and channel groups, with before/after notes showing what the extra channels buy.~~ (done — `digital_twin_nchannel`, `edge_consensus_nchannel`, and `power_safety_nchannel`)
- Add a "minimal viable domainpack in 5 minutes" path to SPO Studio or the CLI: raw sample data, binding scaffold, policy validation, run, visualisation, and replay without requiring users to understand every control-theory detail first.
- Keep v1.0 focused on N-channel rollout, public benchmarks against standard Kuramoto/Strogatz and Pikovsky references, real hardware examples, and API freeze discipline.

### v1.x architecture focus

- Generalise the current three-channel P/I/S model into a typed N-channel binding architecture. P/I/S remains the default profile, not the ceiling; domainpacks must be able to declare additional named channels with extractor type, units, metric semantics, coupling participation, audit serialisation, replay semantics, and supervisor visibility.
- Add channel algebra for N-channel runs: channel groups, required/optional channels, derived channels, cross-channel coupling policies, and validation rules for missing, delayed, or uncertain channels.
- Update audit, replay, visualisation, and reporting to be channel-count agnostic. Acceptance gate: the same run/replay/report pipeline works for three channels and for at least two domainpacks with more than three declared channels.
- Extend optimisation surfaces to include channel weights and cross-channel coupling parameters, not only `K`, `alpha`, `zeta`, and `Psi`.
- Treat Rust and JAX as primary execution paths. Keep Julia, Go, Mojo, and other auxiliary backends experimental unless a maintained production workload shows a 5-10x gain or a capability Rust/JAX cannot provide.
- ~~Document the backend fallback chain in one place, including feature flags, runtime detection, numerical tolerance, benchmark evidence, and deprecation criteria.~~ (done — `docs/guide/backend_fallbacks.md`)
- ~~Add a multi-language backend review gate before each minor release: keep, demote to experimental, or remove based on maintenance cost, CI burden, and measured value.~~ (done — `docs/guide/backend_review_gate.md`, non-destructive default with explicit sign-off for removal)
- Extend visualisation beyond static matplotlib and the current WASM surface: Plotly/Dash dashboards for production operators, real-time streaming plots, and optional 3D views for swarm, traffic, robotics, and spatial domainpacks.

### v1.x differentiators

- ML-driven auto-binding and oscillator discovery: ingest raw multimodal data, propose P/I/S or N-channel extractors, infer an initial coupling graph, and emit a reviewable binding spec.
- Auto domain binding pipeline: turn raw time-series, event logs, and sensor streams into candidate oscillator families, extractor parameters, initial coupling matrices, and a scored `binding_spec.yaml` proposal. Start with SINDy-assisted feature discovery and leave graph-learning coupling inference behind an experimental flag until it has reproducible benchmarks.
- RL and hybrid optimisation layer for knob tuning: leverage the JAX `nn/` backend and `autotune` module to learn policies for `K`, `alpha`, `zeta`, `Psi`, and channel weights from rewards based on coherence metrics minus penalties for `R_bad`, unsafe actuation, and regime churn. Initial scope: replay-trained model-free or hybrid PPO/SAC experiments that emit auditable policy candidates rather than direct production control.
- Trainable supervisor policies: extend rule-based policy evaluation with reinforcement learning or active-inference loops that optimise long-horizon `R_good` / `R_bad` trade-offs under replayable safety constraints.
- Uncertainty-aware phase estimation: Bayesian or ensemble phase estimates propagated through MPC/OA reduction and supervisor decisions.
- SPO Studio GUI: web-based binding and policy builder that scaffolds, visualises, validates, and replays binding specs, with WASM-backed previews where useful.
- Hierarchical multi-scale orchestration: support nested orchestrators where local/edge supervisors maintain local coherence, exchange reduced phase/coherence summaries, and escalate only bounded regime evidence to a parent supervisor. Reuse Hodge decomposition and transfer-entropy monitors to decide what crosses hierarchy boundaries.
- Distributed edge orchestration: multi-node phase consensus with gossip or local Kuramoto coupling, plus WASM/FPGA deployment paths for decentralised operation.
- Digital-twin binding standard: version `binding_spec.yaml` as an open bidirectional live-sync contract for simulators, services, and hardware twins.
- Formal verification hooks: export Petri-net regimes and policy rules to PRISM, TLA+, SPIN, or equivalent model-checking workflows, with CI artefacts for safety-critical policies.
- Neuromorphic and quantum-native backends: extend existing SNN and quantum-control bridges so the orchestrator can emit Lava/BrainScaleS-style neuromorphic schedules and QPU control schedules directly from validated phase-control plans.
- Extractor and actuator plugin ecosystem: define a stable Python/Rust plugin interface for custom `PhaseExtractor` and `ActuationMapper` implementations, including schema validation, entry-point discovery, audit metadata, compatibility tests, and versioned capability declarations.
