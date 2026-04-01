# UPDE Engine

The Unified Phase Dynamics Engine (UPDE) is SPO's core integrator subsystem.
It provides 14 ODE engine variants covering standard Kuramoto, amplitude
dynamics (Stuart-Landau), higher-order interactions (simplicial), inertial
systems (power grids), stochastic resonance, geometric integration, time
delays, financial markets, spatial-phase coupling (swarmalators),
hypergraph k-body coupling, mean-field reduction, variational prediction,
adjoint gradients, and bifurcation continuation.

## Pipeline position

```
CouplingBuilder.build() ──→ K_nm, α
                                │
Oscillators.extract() ──→ θ, ω │
                                │
Drivers.compute() ──→ Ψ        │
                                ↓
              ┌─────── UPDEEngine.step(θ, ω, K, ζ, Ψ, α) ───────┐
              │                                                    │
              │    Euler / RK4 / RK45 (adaptive)                   │
              │    Optional: Rust FFI via spo_kernel                │
              │                                                    │
              └────────────── θ_new ∈ [0, 2π)^N ─────────────────┘
                                │
                                ↓
                     compute_order_parameter(θ) → R, ψ
                                │
                                ↓
                     RegimeManager.evaluate() → Regime
```

The engine is the **computational core** of SPO. Every subsystem
feeds into it (coupling, oscillators, drivers) or consumes its
output (order parameters, monitors, supervisor).

### Engine variants

| Engine | State | ODE | Use case |
|--------|-------|-----|----------|
| UPDEEngine | θ ∈ [0,2π)^N | Kuramoto | General synchronisation |
| SparseUPDEEngine | θ ∈ [0,2π)^N | Sparse Kuramoto | High-N scalability ((N \log N)$) |
| SparseUPDEEngine | θ ∈ [0,2π)^N | Sparse Kuramoto | High-N scalability ((N \log N)$) |
| StuartLandauEngine | [θ,r] ∈ R^{2N} | Stuart-Landau | Amplitude dynamics |
| SimplicialEngine | θ ∈ [0,2π)^N | 3-body Kuramoto | Explosive sync |
| InertialEngine | [θ,ω̇] ∈ R^{2N} | Swing equation | Power grids |
| SwarmalatorEngine | [x,θ] ∈ R^{(D+1)N} | Position + phase | Swarm robotics |
| StochasticInjector | θ ∈ [0,2π)^N | Euler-Maruyama | Noise resonance |
| GeometricEngine | z ∈ C^N | SO(2) exponential | Long simulations |
| DelayedEngine | θ + buffer | Delayed Kuramoto | Transport delays |
| MarketEngine | θ from Hilbert | Price → phase | Financial markets |
| SplittingEngine | θ ∈ [0,2π)^N | Symplectic split | Energy-preserving |
| HypergraphEngine | θ ∈ [0,2π)^N | k-body coupling | Mixed-order |
| OttAntosenReduction | z ∈ C | Mean-field ODE | Fast prediction |
| PredictionModel | θ ∈ [0,2π)^N | Error injection | FEP-Kuramoto |
| AdjointGradient | ∂R/∂K | Finite diff / JAX | Optimisation |

### Performance budgets

| Operation | N | Budget | Rust path |
|-----------|---|--------|-----------|
| `UPDEEngine.step()` | 8 | < 50 μs | ~ 30 μs |
| `UPDEEngine.step()` | 64 | < 1 ms | ~ 0.3 ms |
| `UPDEEngine.step()` | 128 | < 5 ms | ~ 1 ms |
| `compute_order_parameter()` | 256 | < 100 μs | ~ 2 μs |
| `StuartLandauEngine.step()` | 32 | < 2 ms | — |
| `SplittingEngine.step()` | 64 | < 1 ms | — |
| `DelayedEngine.step()` | 32 | < 1 ms | — |

## Core Kuramoto Engine

First-order Kuramoto ODE: dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j - θ_i - α_ij) + ζ sin(Ψ - θ_i).
Supports Euler, RK4, and RK45 (adaptive) integration. Optional Rust FFI
acceleration via `spo_kernel.PyUPDEStepper`.

::: scpn_phase_orchestrator.upde.engine

## Stuart-Landau Amplitude Engine

Phase + amplitude dynamics with supercritical/subcritical Hopf bifurcation.
State vector: [θ₁...θₙ | r₁...rₙ]. Amplitude coupling via K_r matrix.
Amplitudes clamped non-negative after each integration step.

::: scpn_phase_orchestrator.upde.stuart_landau

## Simplicial (3-Body) Engine

Higher-order interactions beyond pairwise coupling. The 3-body term
σ₂/N² Σ_{j,k} sin(θ_j + θ_k - 2θ_i) produces explosive (first-order)
synchronization transitions not achievable with pairwise coupling alone.
Vectorized via trig identity: 2·S_i·C_i where S = Σsin(Δθ), C = Σcos(Δθ).

Gambuzza et al. 2023, Nature Physics; Tang et al. 2025.

::: scpn_phase_orchestrator.upde.simplicial

## Second-Order Inertial Engine (Power Grids)

Swing equation: m_i θ̈_i + d_i θ̇_i = P_i + Σ_j K_ij sin(θ_j - θ_i).
Models power grid transient stability where m_i is generator inertia,
d_i is damping, P_i is power injection (positive = generation, negative = load),
and K_ij is transmission line susceptance. RK4 integration.

Includes `frequency_deviation()` (Hz from nominal — >0.5 Hz triggers load
shedding in real grids) and `coherence()` (phase-lock measure).

Filatrella et al. 2008; Dörfler & Bullo 2014.

::: scpn_phase_orchestrator.upde.inertial

## Financial Market Regime Detection

Extracts instantaneous phase from price/return time series via Hilbert
transform, computes Kuramoto order parameter R(t) across assets, classifies
synchronization regimes (desync/transition/synchronised), and detects
crash early warning signals (R crossing threshold from below).

R(t) → 1 preceding market crashes documented for Black Monday 1987 and
the 2008 financial crisis (arXiv:1109.1167).

::: scpn_phase_orchestrator.upde.market

## Swarmalator Dynamics

Agents with both spatial position x_i ∈ R^D and oscillator phase θ_i ∈ S¹.
Phase modulates spatial attraction (J parameter); spatial proximity modulates
phase coupling (K/|x_ij|). D-dimensional (2D, 3D supported).

Five collective states: static sync, static async, static phase wave,
splintered phase wave, active phase wave — depending on J and K signs.

O'Keeffe, Hong, Strogatz, Nature Communications 2017.

::: scpn_phase_orchestrator.upde.swarmalator

## Stochastic Engine

Euler-Maruyama integration with Gaussian noise injection. Includes
automatic optimal noise tuning: D* ≈ K·R_det/2 (Tselios et al. 2025).
Counter-intuitive: noise at D* INCREASES synchronization (stochastic
resonance). Self-consistency solved via modified Bessel equation
(Acebrón et al. 2005).

::: scpn_phase_orchestrator.upde.stochastic

## Geometric (Torus-Preserving) Engine

Symplectic Euler on T^N using SO(2) exponential map: z_i = exp(iθ_i).
Avoids mod 2π discontinuity errors that accumulate in standard integrators
over long simulations. Essential for multi-hour or multi-day simulations
where phase wrapping drift becomes significant.

::: scpn_phase_orchestrator.upde.geometric

## Time-Delayed Coupling Engine

Circular buffer supports arbitrary per-pair time delays τ_ij with automatic
fallback to instantaneous coupling when delay is zero. Time delays generate
"effective higher-order interactions for free" (Ciszak et al. 2025) because
the delayed coupling mixes information across multiple timescales.

::: scpn_phase_orchestrator.upde.delay

## Ott-Antonsen Mean-Field Reduction

Exact analytical reduction for globally-coupled Kuramoto with Lorentzian
frequency distribution. Reduces N-oscillator system to a single complex
ODE: dz/dt = -(Δ + iω₀)z + (K/2)(z - |z|²z).

Critical coupling K_c = 2Δ. Steady-state: R_ss = √(1 - 2Δ/K).
Used by the PredictiveSupervisor as a fast forward model for MPC
(O(1) computation vs O(N) for full simulation).

::: scpn_phase_orchestrator.upde.reduction

## Variational Free Energy Predictor

Implementation of Friston's Free Energy Principle mapped to Kuramoto
dynamics. Precision-weighted prediction error drives coupling updates;
KL divergence provides a complexity penalty. Online precision estimation
from error variance.

Includes `PredictionModel` (forward prediction with error injection)
and `VariationalPredictor` (FEP-Kuramoto correspondence).

::: scpn_phase_orchestrator.upde.prediction

## Adjoint Gradient Computation

Finite-difference and JAX-autodiff gradients of the synchronization cost
(1 - R) with respect to the coupling matrix K_nm. Used for gradient-based
coupling optimization without the overhead of forward-mode differentiation.

::: scpn_phase_orchestrator.upde.adjoint

## Order Parameters & Metrics

Kuramoto order parameter R (global coherence), PLV (pairwise phase-locking
value), and layer coherence (R for oscillator subsets). Optional Rust
acceleration.

::: scpn_phase_orchestrator.upde.order_params

## Phase-Amplitude Coupling (PAC)

Modulation index (MI) via Tort et al. 2010. Bins low-frequency phase,
computes mean high-frequency amplitude per bin, KL divergence from uniform.
Produces N×N PAC matrix: entry [i,j] = MI(phase_i, amplitude_j).
Central to neuroscience cross-frequency coupling analysis.

::: scpn_phase_orchestrator.upde.pac

## Envelope & Numerics

Amplitude envelope extraction and numerical integration utilities
(DP54 coefficients, error estimation, step size control).

::: scpn_phase_orchestrator.upde.envelope

::: scpn_phase_orchestrator.upde.numerics

## Bifurcation Continuation

Traces the synchronization transition R(K) as a function of coupling
strength. The incoherent state (R≈0) bifurcates to partial synchronization
(R>0) at the critical coupling K_c.

**Two interfaces:**

- `trace_sync_transition()`: sweep R(K) over a range of coupling strengths
- `find_critical_coupling()`: binary search for K_c with configurable precision

**Analytical reference:** K_c = 2/(π g(0)) for Lorentzian g(ω) with half-width
Δ gives K_c = 2Δ (Kuramoto 1975, Strogatz 2000).

**Usage:**

```python
from scpn_phase_orchestrator.upde.bifurcation import (
    trace_sync_transition, find_critical_coupling,
)

# Sweep R(K) curve
diagram = trace_sync_transition(omegas, K_range=(0, 5), n_points=50)
print(f"K_c ≈ {diagram.K_critical}")

# Precise K_c via binary search
Kc = find_critical_coupling(omegas, tol=0.05)
```

::: scpn_phase_orchestrator.upde.bifurcation

## Basin Stability

Monte Carlo estimation of the volume of the basin of attraction for
the synchronised state. Basin stability S_B is the probability that a
random initial condition converges to the synchronised attractor.

**Procedure:** Draw n_samples random phase configurations from [0, 2π)^N,
integrate each to steady state, check if R_final > R_threshold. S_B = fraction
that converge.

`multi_basin_stability()` classifies outcomes at multiple R thresholds
to detect multi-stability (chimera states, partial synchronization).

**Usage:**

```python
from scpn_phase_orchestrator.upde.basin_stability import (
    basin_stability, multi_basin_stability,
)

result = basin_stability(omegas, knm, n_samples=1000)
print(f"S_B = {result.S_B:.3f} ({result.n_converged}/{result.n_samples})")

# Multi-threshold detection
results = multi_basin_stability(omegas, knm, R_thresholds=(0.3, 0.6, 0.8))
```

**References:** Menck et al. 2013, Nature Physics 9:89-92.

::: scpn_phase_orchestrator.upde.basin_stability

## Hypergraph (k-Body) Coupling Engine

Generalized k-body Kuramoto interactions via explicit hyperedge lists.
Extends beyond the simplicial engine's fixed 3-body coupling to arbitrary
k-body interactions for any k ≥ 2.

For a k-hyperedge {i₁, ..., iₖ}, the coupling on oscillator iₘ is:
σₖ · sin(Σ_{j≠m} θ_{iⱼ} - (k-1)·θ_{iₘ})

This generalizes:
- k=2: sin(θ_j - θ_i) — standard Kuramoto
- k=3: sin(θ_j + θ_k - 2θ_i) — simplicial
- k=4: sin(θ_j + θ_k + θ_l - 3θ_i) — quartic interaction

Supports mixed-order interactions: some edges pairwise, some 3-body,
some 4-body, in the same network.

**Usage:**

```python
from scpn_phase_orchestrator.upde.hypergraph import HypergraphEngine

eng = HypergraphEngine(n_oscillators=8, dt=0.01)
eng.add_all_to_all(order=3, strength=0.5)  # all 3-body edges
eng.add_edge((0, 1, 2, 3), strength=0.2)   # one 4-body edge

phases = eng.run(phases_init, omegas, n_steps=1000,
                 pairwise_knm=knm)  # combine with standard coupling
```

**References:** Tanaka & Aoyagi 2011, Phys. Rev. Lett. 106:224101;
Bick et al. 2023, Nat. Rev. Physics 5:307-317.

::: scpn_phase_orchestrator.upde.hypergraph

---

## Sparse Engine

The  implements the Kuramoto model using a **CSR (Compressed Sparse Row)** 
coupling matrix. This reduces memory overhead from (N^2)$ to (N + E)$, where $ is the 
number of active edge connections.

It is designed for large-scale simulations (national power grids, social networks) 
where most oscillators are only coupled to local neighbors.

### Features
- **Scalability:** Integrates 0^6$ nodes with 0^7$ edges in sub-second latencies on standard hardware.
- **FFI Parity:** Offloads integration and plasticity to the  Rust backend for zero-overhead performance.
- **In-place Plasticity:** Supports sub-microsecond Hebbian updates to the  array during the integration step.

::: scpn_phase_orchestrator.upde.sparse_engine
