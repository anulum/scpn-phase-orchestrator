# Architecture

## Overview

SCPN Phase Orchestrator is a domain-agnostic coherence control compiler.
It transforms hierarchical oscillator systems into phase-locked control
logic via Kuramoto/UPDE dynamics with a Rust-accelerated kernel.

## Pipeline

```
Domain YAML ──► Binding Loader ──► Validator
                                      │
                     ┌────────────────┘
                     ▼
              Oscillator Extractors (P / I / S)
                     │
                     ▼
         ┌── UPDE Engine (RK4/RK45) ──┐
         │   or Stuart-Landau Engine   │
         │   (phase + amplitude)       │
         └────────────┬────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
  Coupling Builder  Imprint Model  Geometry Prior
        │             │             │
        └─────────────┼─────────────┘
                      ▼
              Boundary Observer
                      │
                      ▼
              Regime Manager
              (Petri net FSM)
                      │
                      ▼
              Supervisor Policy
              (compound DSL rules)
                      │
                      ▼
              Actuation Mapper
                      │
                      ▼
              Audit Logger
              (SHA256 chain)
```

## Module Map

| Module | Purpose | Key Types |
|--------|---------|-----------|
| `binding/` | YAML/JSON spec loading, validation | `BindingSpec`, `OscillatorFamily` |
| `oscillators/` | Signal→phase extraction (P/I/S channels) | `PhaseExtractor`, `PhaseState` |
| `coupling/` | K_nm matrix construction with decay/boosts | `CouplingBuilder`, `CouplingState` |
| `upde/` | Phase ODE integration (Kuramoto + Stuart-Landau) | `UPDEEngine`, `StuartLandauEngine` |
| `upde/pac.py` | Phase-amplitude coupling (Tort 2010) | `modulation_index`, `pac_matrix` |
| `upde/envelope.py` | Modulation envelope extraction | `extract_envelope`, `EnvelopeState` |
| `imprint/` | History-dependent coupling modulation | `ImprintModel` |
| `drivers/` | External forcing (P/I/S channels) | `PhysicalDriver`, `SymbolicDriver` |
| `monitor/` | Boundary crossing detection | `BoundaryObserver` |
| `supervisor/` | Regime FSM + policy engine | `RegimeManager`, `SupervisorPolicy` |
| `supervisor/petri_net.py` | Formal Petri net FSM | `PetriNet`, `Marking`, `Guard` |
| `supervisor/events.py` | Event bus for regime transitions | `EventBus`, `RegimeEvent` |
| `actuation/` | Control output mapping | `ActuationMapper` |
| `audit/` | Deterministic audit trail + replay | `AuditLogger`, `ReplayEngine` |
| `reporting/` | Matplotlib visualizations | `CoherencePlot` |
| `adapters/` | Bridge adapters (OTel, SCPN ecosystem) | `OTelExporter`, `FusionCoreBridge` |
| `apps/queuewaves/` | Cascade failure detector application | `QueueWavesConfig`, `PhaseComputePipeline` |

## Rust Kernel (`spo-kernel/`)

| Crate | Purpose |
|-------|---------|
| `spo-types` | Shared config and state types |
| `spo-engine` | UPDE stepper, Stuart-Landau ODE, PAC, coupling |
| `spo-oscillators` | Phase extraction (P/I/S channels) |
| `spo-supervisor` | Regime manager, policy, coherence, projector |
| `spo-ffi` | PyO3 bindings exposing Rust engine to Python |

Python auto-delegates to Rust when `spo_kernel` is importable. Pure-Python
fallback ensures the package works without the native extension.

## Data Flow

1. **Binding**: YAML domainpack parsed → `BindingSpec` validated
2. **Build**: Coupling matrix K_nm, geometry constraints, imprint model constructed
3. **Step loop**: `UPDEEngine.step()` integrates phases via RK4/RK45
4. **Observe**: `BoundaryObserver` checks limit crossings
5. **Decide**: `RegimeManager` transitions regime; `SupervisorPolicy` fires rules
6. **Act**: `ActuationMapper` maps policy actions to knob adjustments
7. **Record**: `AuditLogger` appends step to JSONL with SHA256 chain

## Three-Channel Model

- **P (Physical)**: Continuous waveforms → Hilbert transform → instantaneous phase
- **I (Informational)**: Discrete events → inter-event intervals → phase mapping
- **S (Symbolic)**: State sequences → ring/graph topology → phase assignment

## Extractor Type System

Domainpacks declare `extractor_type` using channel aliases (`physical`,
`informational`, `symbolic`) or algorithm names (`hilbert`, `event`, `ring`,
`graph`, `wavelet`, `zero_crossing`). The loader resolves aliases to their
default algorithm at parse time.

## Deployment Targets

- **CLI**: `spo run`, `spo validate`, `spo replay`, `spo scaffold`
- **Library**: `pip install scpn-phase-orchestrator` (pure Python or with Rust extension)
- **QueueWaves**: `spo queuewaves serve` (FastAPI + WebSocket)
- **Docker**: `docker build -t spo .`
