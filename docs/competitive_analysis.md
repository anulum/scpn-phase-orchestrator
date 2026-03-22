# Competitive Analysis

SCPN Phase Orchestrator (SPO) is a **domain-agnostic phase-coupling supervisor**, not a
neural simulator. It occupies a different niche from spiking-network simulators (Brian2,
NEST, Nengo) and general ODE libraries (SciPy). This document clarifies the positioning.

## Feature Matrix

| Capability | SPO | Brian2 | NEST | Nengo | SciPy | DynSys.jl |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Kuramoto / phase oscillators | **native** | manual | no | no | manual | manual |
| Stuart-Landau amplitude model | **native** | manual | no | no | manual | manual |
| Supervisory control loop | **yes** | no | no | no | no | no |
| Regime detection & hysteresis | **yes** | no | no | no | no | no |
| Boundary monitoring (soft/hard) | **yes** | no | no | no | no | no |
| Coupling adaptation (imprint) | **yes** | no | no | no | no | no |
| Phase-amplitude coupling (PAC) | **native** | manual | no | no | no | no |
| Deterministic audit replay | **yes** | no | no | no | no | no |
| Domain-agnostic binding spec | **yes** | no | no | no | no | no |
| Spiking neuron models | no | **native** | **native** | **native** | no | no |
| Large-scale spiking (>10^6) | no | yes | **yes** | yes | no | no |
| GPU acceleration | no | partial | partial | yes | no | no |
| Adaptive ODE solvers | RK45 | varies | no | no | **full** | **full** |
| Rust FFI kernel | **yes** | C++ | C++ | no | Fortran | Julia JIT |
| Lyapunov exponents | no | no | no | no | no | **native** |
| Recurrence analysis | no | no | no | no | no | **native** |

## What SPO Is

A **closed-loop phase orchestrator** that:

1. Integrates Kuramoto/Stuart-Landau ODEs (Euler, RK4, RK45)
2. Monitors coherence (R), boundaries, regime transitions
3. Adapts coupling parameters in real time via supervisor policy
4. Supports domain-agnostic binding specs (plug any physical domain)
5. Provides deterministic audit replay for reproducibility

## What SPO Is Not

- Not a general-purpose neural simulator (use Brian2/NEST for spiking networks)
- Not a general ODE solver (use SciPy for arbitrary dynamical systems)
- Not optimized for >10^4 oscillators (compute scales as O(N^2) per step)

## Benchmark Evidence

### Recovery After Perturbation (N=64, dt=0.01)

| Method | Steps to R>0.7 | Wall Time |
|---|---|---|
| SPO passive (Euler) | 364 | 0.045s |
| **SPO supervisor** | **269** | **0.050s** |
| SciPy RK45 | 362 | 0.828s |

Supervisor achieves **26% faster recovery** (fewer steps) and **16x lower wall time**
than SciPy solve_ivp on equivalent dynamics.

### Per-Step Performance (N=64, 1000 steps)

| Backend | us/step |
|---|---|
| Python Euler | 89.5 |
| Rust FFI Euler | 99.2 |
| **Rust batch (1000 steps)** | **52.4** |

Rust batch API eliminates per-call FFI overhead, achieving 1.7x speedup at N=64.

### Batch API Speedup by Oscillator Count

| N | Python | Rust batch | Speedup |
|---|---|---|---|
| 8 | 11.7 us | 0.8 us | **14.6x** |
| 16 | 14.5 us | 2.1 us | **6.9x** |
| 64 | 89.5 us | 52.4 us | **1.7x** |
| 256 | 1237.6 us | 1084.9 us | 1.1x |

### vs DynamicalSystems.jl (Julia)

DynamicalSystems.jl provides Lyapunov exponents, attractor reconstruction, recurrence analysis,
and general dynamical systems tools in Julia. It can model Kuramoto networks but has no domain
binding layer, no supervisory control, and no production deployment story.

**SPO advantage:** Domain-agnostic domainpacks, regime supervision, Rust FFI + JAX acceleration.

**DynamicalSystems.jl advantage:** Julia JIT performance, bifurcation continuation, Lyapunov
spectrum, nonlinear dynamics toolkit.

**Benchmark status:** DynamicalSystems.jl benchmarks require Julia installation and are pending.

## Target Use Cases

1. **Fusion plasma control** — phase-coupled MHD mode suppression with boundary monitoring
2. **Neuromorphic computing** — oscillator-based computing with regime-aware supervision
3. **Queuing systems** — queue-length oscillation detection and adaptive load balancing
4. **Bio-signal entrainment** — brainwave synchronization with real-time coherence feedback
5. **General coupled-oscillator control** — any domain expressible as Kuramoto/Stuart-Landau
