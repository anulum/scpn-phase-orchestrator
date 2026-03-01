# Roadmap

## v0.1 (current)

- Core scaffold with src layout and CLI
- UPDE engine: RK4 integrator, pre-allocated arrays, Kuramoto coupling
- 3-channel oscillator model (phase, frequency, coherence)
- Coupling matrix builder with decay and cross-hierarchy boosts
- Supervisor with order-parameter thresholds
- Actuation mapper for domain-agnostic output binding
- 4 domainpacks: audio, neuro, fusion, generic
- PhaseExtractor base class for signal intake

## v0.2

- Deterministic replay from recorded phase trajectories
- Policy DSL for declarative threshold/action rules
- Geometry constraints on coupling topology (ring, lattice, small-world)

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
