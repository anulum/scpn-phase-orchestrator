# Architecture

## Overview

SCPN Phase Orchestrator is a domain-agnostic coherence control compiler.
It transforms hierarchical oscillator systems into phase-locked control
logic via Kuramoto/UPDE dynamics with a Rust-accelerated kernel and
optional JAX differentiable backend.

## Pipeline

```
Domain YAML ──► Binding Loader ──► Validator
                                      │
                     ┌────────────────┘
                     ▼
              Oscillator Extractors (P / I / S)
                     │
                     ▼
         ┌── UPDE Engine (9 variants) ────────────┐
         │   Kuramoto, Stuart-Landau, Inertial,   │
         │   Market, Swarmalator, Stochastic,     │
         │   Geometric, Delay, Simplicial         │
         │   + Ott-Antonsen mean-field reduction  │
         └────────────┬───────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
  Coupling Builder  Imprint Model  Geometry Prior
  (Hodge, TE,       (history-dep)  (constraints,
   plasticity)                      spectral)
        │             │             │
        └─────────────┼─────────────┘
                      ▼
              Monitor Array (15 observers)
              ├── Boundary Observer
              ├── Coherence / Order Parameter
              ├── Chimera Detection
              ├── EVS (Entrainment Verification)
              ├── PID (Redundancy/Synergy)
              ├── Lyapunov Exponent
              ├── Entropy Production
              ├── Winding Number
              ├── ITPC, PAC, Transfer Entropy
              ├── Sleep Staging, NPE
              └── STL Runtime Monitor
                      │
                      ▼
              Regime Manager
              (Petri net FSM + hysteresis)
                      │
                      ▼
              Supervisor Policy
              (compound DSL rules + MPC)
                      │
                      ▼
              Actuation Mapper
              (constraint projection)
                      │
                      ▼
              Audit Logger
              (SHA256-chained JSONL)
```

### Parallel Track: Differentiable Backend (nn/)

```
JAX/Equinox ─► KuramotoLayer / StuartLandauLayer
               ├── Simplicial 3-body
               ├── BOLD hemodynamic model
               ├── Reservoir computing
               ├── UDE-Kuramoto (physics + neural residual)
               ├── Inverse pipeline (data → K, ω)
               ├── OIM (graph coloring)
               └── SAF spectral loss
```

All `nn/` functions are JIT-compilable, vmap-compatible, and fully
differentiable. GPU acceleration via `jax[cuda12]`.

## Module Map

### Core Pipeline

| Module | Purpose | Key Types |
|--------|---------|-----------|
| `binding/` | YAML/JSON spec loading, validation | `BindingSpec`, `OscillatorFamily` |
| `oscillators/` | Signal→phase extraction (P/I/S channels) | `PhaseExtractor`, `PhaseState` |
| `coupling/` | K_nm matrix construction, adaptation, analysis | `CouplingBuilder`, `KnmMatrix` |
| `upde/` | Phase ODE integration (10 engine variants) | `UPDEEngine`, `SparseUPDEEngine`, `StuartLandauEngine` |
| `imprint/` | History-dependent coupling modulation | `ImprintState`, `ImprintUpdate` |
| `drivers/` | External forcing (P/I/S channels) | `PhysicalDriver`, `SymbolicDriver` |
| `monitor/` | 15 dynamical observers | `BoundaryObserver`, `ChimeraDetector` |
| `supervisor/` | Regime FSM + policy engine + AI | `RegimeManager`, `ActiveInferenceAgent`, `SupervisorPolicy` |
| `actuation/` | Control output mapping with constraints | `ActuationMapper`, `ConstraintProjection` |
| `audit/` | Deterministic audit trail + replay | `AuditLogger`, `ReplayEngine` |

### UPDE Engines (upde/)

| Engine | Module | Domain |
|--------|--------|--------|
| Standard Kuramoto | `engine.py` | General coupled oscillators |
| Stuart-Landau | `stuart_landau.py` | Phase + amplitude, Hopf bifurcation |
| Inertial (2nd order) | `inertial.py` | Power grid swing equations |
| Market | `market.py` | Financial regime detection |
| Swarmalator | `swarmalator.py` | Spatial + phase coupling |
| Stochastic | `stochastic.py` | Euler-Maruyama, optimal noise D* |
| Geometric | `geometric.py` | Torus-preserving symplectic integrator |
| Delay | `delay.py` | Time-delayed coupling |
| Simplicial | `simplicial.py` | 3-body higher-order interactions |
| Ott-Antonsen | `reduction.py` | O(1) mean-field forward model |
| Variational FEP | `prediction.py` | Free Energy Principle predictor |
| Adjoint | `adjoint.py` | Gradient computation for K optimization |

Supporting: `order_params.py`, `pac.py`, `envelope.py`, `numerics.py`, `metrics.py`, `splitting.py`, `jax_engine.py`

### Coupling Subsystem (coupling/)

| Module | Purpose |
|--------|---------|
| `knm.py` | K_nm matrix construction |
| `geometry_constraints.py` | Spatial coupling constraints |
| `templates.py` | Pre-configured topologies (all-to-all, ring, small-world) |
| `hodge.py` | Hodge decomposition (gradient/curl/harmonic) |
| `plasticity.py` | Three-factor Hebbian adaptation |
| `te_adaptive.py` | Transfer entropy causal coupling |
| `connectome.py` | HCP-inspired brain coupling matrices |
| `lags.py` | Phase lag estimation |
| `spectral.py` | Spectral analysis of coupling |
| `ei_balance.py` | Excitatory/inhibitory balance |
| `prior.py` | Coupling priors |

### Monitor Array (monitor/)

| Monitor | Module | Detects |
|---------|--------|---------|
| Boundary Observer | `boundaries.py` | Safety/performance limit crossings |
| PGBO (Gauged) | `ssgf/pgbo.py` | Phase-Geometry curvature alignment |
| Coherence | `coherence.py` | Order parameter R tracking |
| Chimera Detection | `chimera.py` | Coexistent coherent/incoherent clusters |
| EVS | `evs.py` | Entrainment verification (3-criterion) |
| PID | `pid.py` | Information redundancy/synergy |
| Lyapunov | `lyapunov.py` | Chaos vs stability (λ exponent) |
| Entropy Production | `entropy_prod.py` | Thermodynamic irreversibility |
| Winding Number | `winding.py` | Topological phase wrapping |
| ITPC | `itpc.py` | Inter-trial phase coherence |
| Transfer Entropy | `transfer_entropy.py` | Directed causal information flow |
| Sleep Staging | `sleep_staging.py` | AASM sleep stage classification |
| NPE | `npe.py` | Normalized prediction error |
| Psychedelic Sim | `psychedelic.py` | Entropy surge simulation |
| STL Runtime | `stl.py` | Signal Temporal Logic safety monitor |
| Session Start | `session_start.py` | Startup coherence gate |

### Differentiable Backend (nn/)

| Module | Purpose |
|--------|---------|
| `functional.py` | Pure JAX Kuramoto/Stuart-Landau/simplicial functions |
| `kuramoto_layer.py` | Equinox module, learnable K and ω |
| `stuart_landau_layer.py` | Phase + amplitude equinox module |
| `bold.py` | Balloon-Windkessel BOLD generator |
| `reservoir.py` | Kuramoto reservoir computing |
| `ude.py` | Physics + neural residual (UDE) |
| `inverse.py` | Gradient-based coupling inference |
| `oim.py` | Oscillator Ising machine |

### Extended Modules

| Module | Purpose | Key Types |
|--------|---------|-----------|
| `ssgf/` | Self-Stabilizing Gauge Field framework | `CarrierField`, `FreEnergyMinimizer`, `TCBO`, `PGBO` |
| `autotune/` | Auto-calibration pipeline | `FrequencyID`, `CouplingEstimation` |
| `visualization/` | D3 network graph, Three.js torus | `NetworkGraph`, `TorusViz` |
| `reporting/` | Matplotlib coherence plots | `CoherencePlot` |
| `adapters/` | 12 bridge adapters (OTel, SCPN ecosystem) | `OTelExporter`, `FusionCoreBridge` |
| `apps/queuewaves/` | Cascade failure detector (FastAPI) | `QueueWavesConfig`, `PhaseComputePipeline` |
| `grpc_gen/` | Protocol buffer stubs | gRPC streaming service |

### Top-Level Modules

| Module | Purpose |
|--------|---------|
| `cli.py` | `spo` command entry point |
| `server.py` | FastAPI REST endpoints |
| `server_grpc.py` | Async gRPC streaming service |
| `exceptions.py` | `SPOError` hierarchy (8 subclasses) |
| `_compat.py` | Rust/Python compatibility, version constants |

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
4. **Observe**: Monitor array evaluates all active observers
5. **Decide**: `RegimeManager` transitions regime; `SupervisorPolicy` fires rules
6. **Act**: `ActuationMapper` maps policy actions to knob adjustments
7. **Record**: `AuditLogger` appends step to JSONL with SHA256 chain
8. **Adapt** (optional): Plasticity/TE updates coupling; MPC predicts via OA reduction

## Three-Channel Model

- **P (Physical)**: Continuous waveforms → Hilbert transform → instantaneous phase
- **I (Informational)**: Discrete events → inter-event intervals → phase mapping
- **S (Symbolic)**: State sequences → ring/graph topology → phase assignment

## Extractor Type System

Domainpacks declare `extractor_type` using channel aliases (`physical`,
`informational`, `symbolic`) or algorithm names (`hilbert`, `event`, `ring`,
`graph`, `wavelet`, `zero_crossing`). The loader resolves aliases to their
default algorithm at parse time.

## Internal Class Hierarchies

### Oscillator Extractors

```
PhaseExtractor (ABC)
├── PhysicalExtractor      # Hilbert, wavelet ridge, zero-crossing
├── InformationalExtractor # Inter-event interval, queue depth
└── SymbolicExtractor      # Ring mapping, graph embedding
```

Each extractor produces `list[PhaseState]` from raw signal + sample rate.
Quality scoring is per-extractor (SNR-aware for P, regularity-aware for I,
entropy-aware for S).

### UPDE Engine Family

```
UPDEEngine              # Standard Kuramoto (Euler/RK4/RK45)
├── rust: PyUPDEStepper # Rust-accelerated (auto-delegated when available)
StuartLandauEngine      # Phase + amplitude (Hopf bifurcation)
├── rust: PyStuartLandauStepper
InertialKuramotoEngine  # 2nd-order swing equation (RK4)
SwarmalatorEngine       # Spatial + phase coupling
SimplicialEngine        # Higher-order 3-body interactions
TorusEngine             # Geometric integrator on T^N
DelayEngine             # Time-delayed coupling (ring buffer)
StochasticInjector      # Euler-Maruyama noise layer (composable)
OttAntonsenReduction    # O(1) mean-field (not a stepper, a predictor)
```

All engines share the interface: `step(phases, omegas, knm, ...) → new_phases`.
The Rust kernel provides drop-in replacements for UPDEEngine and StuartLandauEngine.

### Supervisor Stack

```
RegimeManager
├── evaluate(UPDEState, BoundaryState) → Regime
├── transition(Regime) → Regime  [with cooldown + hysteresis]
└── force_transition(Regime)     [bypass safety]

SupervisorPolicy
├── PolicyRule[]                  [condition → action]
└── PolicyEngine.decide()        [priority-ordered evaluation]

PredictiveSupervisor (MPC)
├── predict() → Prediction       [OA forward model]
└── decide() → ControlAction[]   [pre-emptive action]

PetriNetFSM
├── Place[] + Transition[] + Arc[]
├── Marking (token state)
└── fire(transition)              [guarded token flow]
```

### SSGF Closure Loop

```
GeometryCarrier
├── z (latent vector)
├── decode(z) → W (coupling matrix via softplus decoder)
└── update(cost, cost_fn) → SSGFState [finite-difference gradient on z]

CyberneticClosure
├── step(phases) → (W, ClosureState)
│   1. decode current z → W_before
│   2. compute SSGF costs (c1_sync, c2_spectral, c3_sparsity, c4_symmetry)
│   3. gradient descent on z via cost_fn
│   4. decode new z → W_after
└── run(phases, n_steps) → (W_final, history)
```

### Monitor → Supervisor Data Flow

```
                     ┌─── BoundaryState ───┐
                     │                     │
phases, knm ──► Monitor Array             ▼
                │                   RegimeManager
                ├── R, psi              │
                ├── chimera_index       ▼
                ├── Lyapunov λ    SupervisorPolicy
                ├── entropy_prod        │
                ├── winding_num         ▼
                ├── TE matrix     ControlAction[]
                ├── NPE                 │
                ├── ITPC                ▼
                └── EVS           ActuationMapper
                                        │
                    UPDEState ◄──────────┘
                        │          ActionProjector
                        ▼          (rate limit + clamp)
                  Audit Logger            │
                  (SHA256 chain)          ▼
                                    knm_new, zeta_new
                                    (fed back to engine)
```

## Deployment Targets

- **CLI**: `spo run`, `spo validate`, `spo replay`, `spo scaffold`
- **Library**: `pip install scpn-phase-orchestrator` (pure Python or with Rust extension)
- **QueueWaves**: `spo queuewaves serve` (FastAPI + WebSocket)
- **Docker**: `docker build -t spo .` / Helm chart for Kubernetes
- **JAX GPU**: `pip install scpn-phase-orchestrator[nn]` + `jax[cuda12]`
- **FPGA**: Verilog `kuramoto_core.v` targeting Zynq-7020 (16 oscillators, sub-15μs)
- **WebAssembly**: Browser-based Kuramoto visualization via `spo-wasm` crate
- **gRPC**: Async streaming service for real-time phase telemetry
