# Roadmap

## v0.1 (current)

- Core scaffold with src layout and CLI
- UPDE engine: RK4 integrator, pre-allocated arrays, Kuramoto coupling
- 3-channel oscillator model (phase, frequency, coherence)
- Coupling matrix builder with decay, cross-hierarchy boosts, geometry constraints
- BoundaryObserver, RegimeManager, SupervisorPolicy pipeline
- PolicyEngine: declarative YAML rules with regime/metric triggers
- ImprintModel: history-dependent coupling modulation (decay, saturation)
- Actuation mapper for domain-agnostic output binding
- 21 domainpacks across neuroscience, cardiology, power systems, plasma physics, manufacturing, epidemiology, traffic, quantum, biology, chemistry, robotics, telecom, and more
- Adapter bridges: FusionCoreBridge, PlasmaControlBridge, QuantumControlBridge
- spo-kernel Rust FFI: UPDE engine, coupling, imprint, order params, lags (112 tests)
- 113+ Python tests, 112 Rust tests
- PhaseExtractor base class for signal intake
- PyPI package (trusted publisher OIDC), Zenodo DOI, GitHub Pages docs

## v0.2

- Deterministic replay from recorded phase trajectories
- Extended policy DSL with compound triggers and action chains
- Prometheus and OpenTelemetry metric adapters

## v0.3

- Prometheus and OpenTelemetry metric adapters
- Real-time WebSocket streaming of phase state
- Dashboard for live order-parameter monitoring

## v0.4

- Petri net regime FSM for multi-phase protocol sequencing
- SNN controller bridge (Nengo/Lava backends)
- Event-driven mode transitions with hysteresis

## v0.5

- Amplitude extension via Stuart-Landau oscillators
- Phase-amplitude coupling (PAC) gating
- Modulation envelope extraction and control

## v1.0

- Production-hardened with fuzz testing and fault injection
- Full benchmark suite against reference Kuramoto implementations
- Complete API documentation and tutorial notebooks
