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

## v0.2

- Extended policy DSL with compound triggers and action chains
- OpenTelemetry trace/metric export for production observability

## v0.3

- Petri net regime FSM for multi-phase protocol sequencing
- SNN controller bridge (Nengo/Lava backends)
- Event-driven mode transitions with hysteresis

## v0.4

- Amplitude extension via Stuart-Landau oscillators
- Phase-amplitude coupling (PAC) gating
- Modulation envelope extraction and control

## v1.0

- Production-hardened with fuzz testing and fault injection
- Full benchmark suite against reference Kuramoto implementations
- Complete API documentation and tutorial notebooks
