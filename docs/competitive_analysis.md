# Competitive Analysis

SCPN Phase Orchestrator (SPO) is a **domain-agnostic phase-coupling supervisor**, not a
neural simulator. It occupies a different niche from spiking-network simulators (Brian2,
NEST, Nengo), general ODE libraries (SciPy), and recent ML-oriented Kuramoto papers (AKOrN).
This document clarifies the positioning.

## Feature Matrix

| Capability | SPO | Brian2 | NEST | Nengo | SciPy | DynSys.jl | AKOrN | XGI |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Kuramoto / phase oscillators | **native** | manual | no | no | manual | manual | **native** | no |
| Stuart-Landau amplitude model | **native** | manual | no | no | manual | manual | no | no |
| Simplicial (3-body) coupling | **native** | no | no | no | no | no | no | **native** |
| Supervisory control loop | **yes** | no | no | no | no | no | no | no |
| Regime detection & hysteresis | **yes** | no | no | no | no | no | no | no |
| Boundary monitoring (soft/hard) | **yes** | no | no | no | no | no | no | no |
| Coupling adaptation (imprint) | **yes** | no | no | no | no | no | no | no |
| Phase-amplitude coupling (PAC) | **native** | manual | no | no | no | no | no | no |
| Deterministic audit replay | **yes** | no | no | no | no | no | no | no |
| Domain-agnostic binding spec | **yes** | no | no | no | no | no | no | no |
| Differentiable (JAX autodiff) | **yes** | no | no | no | no | no | **yes** | no |
| GPU acceleration (JAX/XLA) | **yes** | partial | partial | yes | no | no | **yes** | no |
| Inverse Kuramoto (data → K) | **analytical+gradient** | no | no | no | no | no | no | no |
| Differentiable 3-body layer | **yes** | no | no | no | no | no | no | no |
| OIM combinatorial solver | **annealing+multi-start** | no | no | no | no | no | no | no |
| Spiking neuron models | **bridge** | **native** | **native** | **native** | no | no | no | no |
| Large-scale spiking (>10^6) | no | yes | **yes** | yes | no | no | no | no |
| Adaptive ODE solvers | RK45 | varies | no | no | **full** | **full** | no | no |
| Rust FFI kernel | **yes** | C++ | C++ | no | Fortran | Julia JIT | no | no |
| Lyapunov exponents | **yes** | no | no | no | no | **native** | no | no |
| Recurrence analysis | **yes** | no | no | no | no | **native** | no | no |
| Hodge decomposition | **yes** | no | no | no | no | no | no | partial |
| Transfer entropy | **yes** | no | no | no | no | no | no | no |

## What SPO Is

A **closed-loop phase orchestrator** that:

1. Integrates Kuramoto/Stuart-Landau ODEs (Euler, RK4, RK45) across 9 engine variants
2. Monitors coherence via 16 dynamical observers (chimera, EVS, Lyapunov, PAC, TE, ...)
3. Adapts coupling parameters in real time via supervisor policy and MPC
4. Supports domain-agnostic binding specs (32 domainpacks across 10+ disciplines)
5. Provides deterministic audit replay for reproducibility
6. Offers a differentiable JAX backend for gradient-based coupling optimization

## What SPO Is Not

- Not a general neuroscience simulator — Brian2/NEST offer morphological neurons, synaptic
  plasticity rules, and scale to >10^6 neurons. SPO's NeurocoreBridge targets a different
  niche: Rust-accelerated LIF ensembles (N=10000, 4ms/100 substeps) coupled directly into
  the phase-oscillator control loop.
- Not a general ODE solver (use SciPy for arbitrary dynamical systems)
- Not optimized for >10^4 oscillators in NumPy mode (compute scales O(N^2) per step;
  JAX GPU mode handles larger N via XLA vectorization)

## Competitor Deep Dives

### AKOrN (ICML 2024)

AKOrN introduced Kuramoto oscillators as attention heads in vision transformers.
SPO differs in three ways:

1. **Amplitude**: AKOrN is phase-only. SPO's Stuart-Landau layer adds amplitude,
   preventing the N>32 degradation AKOrN reports.
2. **Control loop**: AKOrN has no supervisor, regime detection, or policy engine.
   It is a forward-pass-only neural network component.
3. **Domain generality**: AKOrN is vision-specific. SPO's binding spec layer
   handles power grids, finance, neuroscience, robotics, and 28 other domains.

**Shared advantage over standard transformers**: Both exploit oscillatory dynamics
for binding and segmentation without learned position embeddings.

### XGI / HyperGraphX

XGI provides higher-order network simulation (simplicial complexes, hypergraphs).
It can model simplicial Kuramoto but cannot differentiate through the dynamics,
has no supervisor, and no domain binding layer.

**SPO advantage**: Differentiable simplicial Kuramoto via JAX, supervisory control,
32 domainpacks.

**XGI advantage**: Richer hypergraph combinatorics, larger research community for
pure higher-order network science.

### DynamicalSystems.jl (Julia)

DynamicalSystems.jl provides Lyapunov exponents, attractor reconstruction, recurrence analysis,
and general dynamical systems tools in Julia. It can model Kuramoto networks but has no domain
binding layer, no supervisory control, and no production deployment story.

**SPO advantage:** Domain-agnostic domainpacks, regime supervision, Rust FFI + JAX acceleration.
SPO now computes full Lyapunov spectrum, recurrence analysis (RQA+CRQA), delay embedding (Takens),
correlation dimension, Kaplan-Yorke dimension, Poincare sections, basin stability, and
bifurcation continuation natively.

**DynamicalSystems.jl advantage:** Julia JIT performance, pseudo-arclength continuation
for codimension-2 bifurcations, more mature nonlinear dynamics ecosystem.

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

Note: N>64 is dominated by O(N^2) coupling computation where Python's NumPy
BLAS matches Rust. Speedup is largest at small N where FFI overhead dominates.

### JAX GPU (nn/ module) — L40S GPU, JAX 0.9.2 (measured 2026-03-27)

All timings use 5 warmup calls + `block_until_ready` before measurement.
JIT compilation time reported separately from steady-state throughput.

**KuramotoLayer forward pass (100 calls, 50 internal steps each):**

| N | JIT (ms) | Steady (us/call) |
|---|---|---|
| 8 | 115 | 408 |
| 16 | 128 | 444 |
| 32 | 121 | 437 |
| 64 | 152 | 472 |
| 128 | 178 | 514 |
| 256 | 125 | 775 |
| 512 | 404 | 760 |

Per Kuramoto step: 8–15us. Previous L4 results (125ms/call) were dominated
by JIT dispatch overhead — `@eqx.filter_jit` eliminated this.

**SimplicialKuramotoLayer forward pass (100 calls, 50 steps, sigma2=1.0):**

| N | JIT (ms) | Steady (us/call) |
|---|---|---|
| 8 | 157 | 450 |
| 16 | 161 | 456 |
| 32 | 163 | 467 |
| 64 | 198 | 482 |
| 128 | 248 | 756 |
| 256 | 151 | 1015 |

3-body coupling adds ~10% overhead vs pairwise-only KuramotoLayer.
First differentiable 3-body Kuramoto layer in open source.

**JAX GPU vs NumPy CPU (500 Kuramoto steps):**

| N | JAX GPU (ms) | NumPy CPU (ms) | Speedup |
|---|---|---|---|
| 16 | 126 | 2 | 0.02x |
| 64 | 141 | 4 | 0.03x |
| 256 | 118 | 24 | 0.20x |
| 512 | 321 | 89 | 0.28x |
| 1024 | 220 | 348 | **1.6x** |
| 2048 | 142 | 1822 | **12.8x** |

GPU crossover at N=1024. At N=2048, JAX is 12.8x faster than NumPy.

**Guidance:** Use NumPy engines for N<1024 on CPU. Use JAX GPU for
N>=1024 or batched workloads via vmap.

**Batched Kuramoto (vmap, 64 oscillators, 200 steps):**

| Batch | Total (ms) | Per instance (us) | Speedup vs B=1 |
|---|---|---|---|
| 1 | 129 | 128,736 | 1x |
| 4 | 130 | 32,502 | 4x |
| 16 | 171 | 10,706 | 12x |
| 64 | 130 | 2,033 | 63x |
| 256 | 131 | 513 | **251x** |

Batching is where GPU dominates — 256 independent Kuramoto systems
in the same wall time as 1. Use `jax.vmap` for parameter sweeps.

**Analytical vs Gradient Inverse Coupling:**

| N | Analytical corr | Analytical time | Gradient corr | Gradient time | Speedup |
|---|---|---|---|---|---|
| 4 | **1.000** | 0.73s | 0.676 | 81.6s | 112x |
| 8 | **0.959** | 0.55s | 0.564 | 77.6s | 141x |
| 16 | **0.845** | 0.59s | 0.527 | 84.0s | 142x |
| 32 | **0.680** | 0.62s | 0.331 | 84.2s | 136x |

Analytical inverse (Pikovsky 2008): O(N^3) linear regression on sin(Dtheta) basis,
phase-aware finite differences via atan2. Orders of magnitude faster and more
accurate than gradient descent through ODE. The gradient method's correlation
degrades at large N due to vanishing gradients through long ODE chains.

**OIM Solver (annealing + multi-start):**

| Graph | n_colors | Violations | Time |
|---|---|---|---|
| K3 (triangle) | 3 | **0** | 7.6s |
| K_{3,3} (bipartite) | 2 | **0** | 6.8s |

OIM solver with coupling annealing (k: 0.1->10), Langevin noise for
exploration, 10 random restarts, and soft coloring extraction achieves
zero violations on standard test graphs. Previous raw dynamics had
11-22% violation rates.

## Target Use Cases

1. **Fusion plasma control** — phase-coupled MHD mode suppression with boundary monitoring
2. **Power grid stability** — inertial Kuramoto for generator swing dynamics and cascade prediction
3. **Neuromorphic computing** — oscillator-based computing with regime-aware supervision
4. **Financial markets** — Hilbert phase extraction for regime detection and crash early warning
5. **Queuing systems** — queue-length oscillation detection and adaptive load balancing
6. **Bio-signal entrainment** — brainwave synchronization with real-time coherence feedback
7. **Swarm robotics** — swarmalator dynamics for formation control
8. **Machine learning** — differentiable Kuramoto layers, inverse coupling, reservoir computing
9. **Combinatorial optimization** — OIM for graph coloring and max-cut via phase clustering
10. **General coupled-oscillator control** — any domain expressible as Kuramoto/Stuart-Landau
