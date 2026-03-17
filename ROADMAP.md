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
- 21 domainpacks: neuroscience_eeg, cardiac_rhythm, power_grid, plasma_control, manufacturing_spc, epidemic_sir, traffic_flow, quantum_simulation, chemical_reactor, swarm_robotics, and 11 more
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
- 1219 Python tests, 191 Rust tests

## v0.5 (planned)

- ~~Rust RegimeManager: hysteresis, EventBus, downward_streak parity with Python~~ (done)
- ~~Rust LagModel: algorithm alignment with Python adaptive lag pipeline~~ (done)
- ~~FFI improvements: PyStuartLandauStepper numpy array input, PyActionProjector unknown-knob warning~~ (done)
- ~~Expand domainpack channel coverage (target: all 3 channels in 12+ packs)~~ (10/24 done)
- Async test infrastructure: pytest-asyncio + httpx.AsyncClient for QueueWaves
- Tutorial notebooks for binding, audit replay, reporting, adapter bridges
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
