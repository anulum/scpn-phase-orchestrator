# SCPN Phase Orchestrator

Domain-agnostic coherence control compiler built on Kuramoto/UPDE phase dynamics.

> **Active Development** — SCPN Phase Orchestrator is under intensive development. The core UPDE engine, all 12 integration methods, 3-channel oscillator extraction (P/I/S), supervisor with regime management, and Rust FFI acceleration are fully functional and tested (3 945 Python tests passed, 567 Rust tests, zero functional failures). Rust FFI coverage now spans 53 engine modules across UPDE, coupling, monitor, SSGF, and autotune subsystems with Superior-level documentation (567+ lines, 8 mandatory sections) for every Rust-accelerated module. APIs may evolve as this work progresses.

**Version:** 0.5.0
**Status:** 142 Python Modules | 12 Engine Variants | 19 Monitors | 24 Domainpacks | 53 Rust Engine Modules | 567 Rust Tests | 3 945 Python Tests Passed

[![CI](https://github.com/anulum/scpn-phase-orchestrator/actions/workflows/ci.yml/badge.svg)](https://github.com/anulum/scpn-phase-orchestrator/actions/workflows/ci.yml)
[![CodeQL](https://github.com/anulum/scpn-phase-orchestrator/actions/workflows/codeql.yml/badge.svg)](https://github.com/anulum/scpn-phase-orchestrator/actions/workflows/codeql.yml)
[![OpenSSF Scorecard](https://github.com/anulum/scpn-phase-orchestrator/actions/workflows/scorecard.yml/badge.svg)](https://github.com/anulum/scpn-phase-orchestrator/actions/workflows/scorecard.yml)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/12193/badge)](https://www.bestpractices.dev/projects/12193)
[![PyPI](https://img.shields.io/pypi/v/scpn-phase-orchestrator)](https://pypi.org/project/scpn-phase-orchestrator/)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://anulum.github.io/scpn-phase-orchestrator/)
[![Coverage](https://img.shields.io/badge/coverage-99%25-brightgreen)](https://github.com/anulum/scpn-phase-orchestrator)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-purple)](https://github.com/anulum/scpn-phase-orchestrator/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Rust FFI](https://img.shields.io/badge/Rust-spo--kernel-orange)](https://github.com/anulum/scpn-phase-orchestrator/tree/main/spo-kernel)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/anulum/scpn-phase-orchestrator/blob/main/.pre-commit-config.yaml)
[![REUSE](https://img.shields.io/badge/REUSE-compliant-green)](https://reuse.software/)
[![Polar.sh](https://img.shields.io/badge/Fund-Polar.sh-blue)](https://polar.sh/anulum)

![Synchronization Manifold](docs/assets/synchronization_manifold.png)

## What It Does

Treats Kuramoto phase dynamics as a universal synchrony state-space.
Any hierarchical coupled-cycle system — plasma, cloud infrastructure,
traffic, power grids, factories, biology — maps onto the same engine.

## Core Pipeline

```
Domain Binder → Oscillator Extractors (P/I/S) → UPDE Engine → Supervisor → Actuation Mapper
```

### 3-Channel Oscillator Model

| Channel | Source | Phase Extraction |
|---------|--------|-----------------|
| **Physical (P)** | Continuous waveforms | Hilbert transform, zero-crossing |
| **Informational (I)** | Event/decision streams | Event-phase from message timing |
| **Symbolic (S)** | Discrete state sequences | Ring-phase θ=2πs/N, graph-walk |

### 4 Universal Control Knobs

| Knob | Meaning |
|------|---------|
| **K** | Coupling strength (Knm matrix) |
| **α** | Phase lag (transport/actuator delays) |
| **ζ** | Driver strength (external forcing) |
| **Ψ** | Reference phase (control target) |

### Dual Objective

- **R_good**: Coherence to maintain (actuator ↔ target phase-lock)
- **R_bad**: Coherence to suppress (harmful mode-locking)

## Capabilities

### Differentiable Phase Dynamics (`nn/` module, JAX)

| Module | What it does |
|--------|-------------|
| KuramotoLayer | Phase-only oscillator layer (equinox), learnable K and ω |
| StuartLandauLayer | Phase + amplitude layer, bifurcation parameter μ |
| Simplicial Kuramoto | 3-body higher-order coupling (Gambuzza 2023) |
| BOLD Generator | Balloon-Windkessel hemodynamic model for fMRI |
| Reservoir Computing | Kuramoto network as nonlinear reservoir + ridge readout |
| SAF Spectral Loss | Topology optimization via Laplacian eigenstructure |
| UDE-Kuramoto | Physics backbone sin(Δθ) + learned neural residual |
| Inverse Pipeline | Infer coupling K and frequencies ω from observed data |
| OIM Graph Coloring | Oscillator Ising machine for combinatorial optimization |

All functions are JIT-compilable, vmap-compatible, and differentiable.
Install: `pip install scpn-phase-orchestrator[nn]`

### Advanced Dynamics (`upde/` module, NumPy)

| Module | What it does |
|--------|-------------|
| Inertial Kuramoto | Second-order swing equation for power grid stability |
| Market Kuramoto | Financial regime detection via Hilbert phase + order parameter |
| Swarmalator | Coupled spatial + phase dynamics (O'Keeffe 2017) |
| Simplicial Engine | 3-body coupling with explosive transitions |
| Stuart-Landau Engine | Amplitude dynamics with Hopf bifurcation |
| Stochastic Engine | Euler-Maruyama with optimal noise (D* auto-tuning) |
| Geometric Engine | Torus-preserving symplectic integrator |
| Delay Engine | Time-delayed coupling with circular buffer |
| Ott-Antonsen | Exact mean-field reduction (O(1) prediction) |

### Closed-Loop Control (unique — no other oscillator library has this)

| Module | What it does |
|--------|-------------|
| MPC Supervisor | Predicts R trajectory 10 steps ahead via OA reduction |
| Regime Manager | FSM with hysteresis (NOMINAL/DEGRADED/CRITICAL) |
| Petri Net FSM | Formal state machine with guard conditions |
| Plasticity | Three-factor Hebbian coupling adaptation |
| TE Adaptive | Transfer entropy-based causal coupling updates |
| Audit Trail | SHA256-chained JSONL for deterministic replay |

### Analysis Toolkit (15 monitors)

Order parameter, PLV, PAC (cross-frequency coupling), chimera detection,
EVS (entrainment verification), PID (redundancy/synergy), Lyapunov
exponent, entropy production, winding number, ITPC, coupling estimation
(including non-sinusoidal harmonics), HCP connectome generation.

### Hardware Deployment

| Target | Status |
|--------|--------|
| Rust FFI | 12 PyO3 bindings for native-speed core modules |
| FPGA | 16-oscillator Zynq-7020 kernel, sub-15μs latency |
| WebAssembly | Browser-based Kuramoto visualization, no server needed |
| JAX GPU | Transparent GPU acceleration via XLA |

### Unique Analysis Capabilities

| Module | What it does |
|--------|-------------|
| Hodge Decomposition | Splits coupling K into gradient / curl / harmonic components |
| Transfer Entropy | Directed causal information flow between oscillators |
| Coupling Estimation | Infer K from data (least-squares + higher harmonics) |

## Quickstart

```bash
# Install from PyPI
pip install scpn-phase-orchestrator

# Or with optional extras
pip install scpn-phase-orchestrator[queuewaves]  # FastAPI cascade detector
pip install scpn-phase-orchestrator[plot]         # matplotlib visualisation
pip install scpn-phase-orchestrator[otel]         # OpenTelemetry export

# Scaffold a new domainpack
spo scaffold my_domain

# Validate a domain binding spec
spo validate domainpacks/minimal_domain/binding_spec.yaml

# Run a domain simulation
spo run domainpacks/queuewaves/binding_spec.yaml --steps 1000

# Replay from audit log
spo replay audit.jsonl --output report.json
```

For development, clone the repo and install in editable mode:

```bash
git clone https://github.com/anulum/scpn-phase-orchestrator.git
cd scpn-phase-orchestrator
pip install -e ".[dev]"
```

## Platform Support

| Platform | Python engine | Rust FFI (optional) |
|----------|--------------|---------------------|
| Linux | Full | Full |
| macOS | Full | Full |
| Windows | Full | Experimental (requires MSVC toolchain) |

The PyPI package is pure Python. Rust FFI provides optional acceleration
and is built from source via `maturin develop`.

## Domainpacks

| Pack | Domain | Purpose |
|------|--------|---------|
| `autonomous_vehicles` | Vehicles | Platoon phase-locking, leader-follower sync (3 layers, 8 oscillators) |
| `bio_stub` | Biology | Multi-scale biological oscillators (4 layers, 16 oscillators) |
| `cardiac_rhythm` | Cardiology | Gap-junction coupling, arrhythmia (4 layers, 10 oscillators) |
| `chemical_reactor` | Process control | Hopf bifurcation, Semenov limit (4 layers, 10 oscillators) |
| `circadian_biology` | Chronobiology | SCN clock-gene coupled oscillators (4 layers, 10 oscillators) |
| `epidemic_sir` | Epidemiology | Epidemic wave synchronisation (3 layers, 8 oscillators) |
| `firefly_swarm` | Ecology | Flash synchronisation, Mirollo-Strogatz (2 layers, 8 oscillators) |
| `fusion_equilibrium` | Fusion equilibrium | Grad-Shafranov + FusionCoreBridge (6 layers, 12 oscillators) |
| `geometry_walk` | Graph systems | Random-walk phase coupling (2 layers, 8 oscillators) |
| `laser_array` | Photonics | Semiconductor laser phase-locking (3 layers, 8 oscillators) |
| `manufacturing_spc` | Manufacturing | Statistical process control (3 layers, 9 oscillators) |
| `metaphysics_demo` | P/I/S showcase | Imprint + geometry ablation (3 layers, 7 oscillators) |
| `minimal_domain` | Synthetic | Minimal-but-complete pipeline example (2 layers, 4 oscillators) |
| `network_security` | Cybersecurity | Traffic anomaly detection, DDoS suppression (3 layers, 8 oscillators) |
| `neuroscience_eeg` | Neuroscience | EEG band->phase, seizure detection (6 layers, 14 oscillators) |
| `plasma_control` | Tokamak plasma | MHD/transport multi-scale control (8 layers, 16 oscillators) |
| `pll_clock` | Telecommunications | PLL network clock synchronisation (3 layers, 8 oscillators) |
| `power_grid` | Power systems | Swing equation = Kuramoto (5 layers, 12 oscillators) |
| `quantum_simulation` | Quantum computing | Qubit register phase coupling (3 layers, 8 oscillators) |
| `queuewaves` | Cloud/queues | Retry storm desynchronisation (3 layers, 6 oscillators) |
| `rotating_machinery` | Vibration | Harmonics, ISO 10816 boundaries (4 layers, 10 oscillators) |
| `satellite_constellation` | Aerospace | Orbital slot synchronisation, beam handover (3 layers, 8 oscillators) |
| `swarm_robotics` | Robotics | Vicsek collective motion (3 layers, 8 oscillators) |
| `traffic_flow` | Transportation | Signal coordination = phase sync (4 layers, 10 oscillators) |
| `financial_markets` | Finance | Stock synchronization, crash detection |
| `gene_oscillator` | Synthetic biology | Repressilator quorum coupling |
| `vortex_shedding` | Fluid dynamics | Wake station Stuart-Landau |
| `robotic_cpg` | Robotics | Joint CPG locomotion |
| `sleep_architecture` | Sleep medicine | AASM sleep staging from R |
| `musical_acoustics` | Acoustics | Consonance = R, groove = alpha |
| `brain_connectome` | Neuroscience | HCP-inspired coupling |
| `identity_coherence` | Consciousness | SSGF identity model (6 layers, 30 oscillators) |

### Adding a Domain

1. Create `domainpacks/<name>/binding_spec.yaml` declaring layers,
   oscillator families, coupling, drivers, objectives, and boundaries.
2. Optionally add `policy.yaml` for declarative supervisor rules.
3. Validate: `spo validate domainpacks/<name>/binding_spec.yaml`
4. Run: `spo run domainpacks/<name>/binding_spec.yaml --steps 1000`

See [`metaphysics_demo`](domainpacks/metaphysics_demo/) for a full
example exercising all three channels, imprint modulation, geometry
projection, and policy-driven control.  Spec format reference:
[binding_spec.schema.json](docs/specs/binding_spec.schema.json).

## Development

```bash
pip install -e ".[dev]"
ruff check src/ tests/
ruff format --check src/ tests/
pytest tests/ -v --tb=short
mkdocs build
```

## License

AGPL-3.0-or-later. Commercial licensing available — contact protoscience@anulum.li.

## Citation

See [CITATION.cff](CITATION.cff).

---

<p align="center">
  <a href="https://www.anulum.li">
    <img src="docs/assets/anulum_logo_company.jpg" width="180" alt="ANULUM">
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.anulum.li">
    <img src="docs/assets/fortis_studio_logo.jpg" width="180" alt="Fortis Studio">
  </a>
  <br>
  <em>Developed by <a href="https://www.anulum.li">ANULUM</a> / Fortis Studio</em>
</p>
