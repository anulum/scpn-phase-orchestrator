# UPDE Engine

The Unified Phase Dynamics Engine (UPDE) is SPO's core integrator subsystem.
It provides 19 ODE engine variants covering standard Kuramoto, Bayesian
uncertainty propagation, amplitude
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
             BayesianUPDE → posterior predictive R ± sigma
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
| BayesianUPDE | θ plus sampled K,ω | Monte Carlo UPDE | Safety-tier uncertainty quantification |
| SparseUPDEEngine | θ ∈ [0,2π)^N | Sparse Kuramoto | High-N scalability ($O(N \log N)$) |
| SheafUPDEEngine | $\vec{\theta} \in \mathbb{R}^{N \times D}$ | Cellular Sheaf | Multi-dimensional block coupling |
| StuartLandauEngine | [θ,r] ∈ R^{2N} | Stuart-Landau | Amplitude dynamics |
| SimplicialEngine | θ ∈ [0,2π)^N | 3-body Kuramoto | Triadic/group synchronization |
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

Direct Go, Julia, and Mojo accelerator entrypoints share the same boundary
contract before optional runtime loading: phase and frequency vectors must be
finite real one-dimensional `float64` arrays with matching length; coupling
and phase-lag matrices must be finite real square matrices (or flattened
square matrices) matching oscillator count; the coupling diagonal must be
exactly zero to exclude self-coupling; `dt`, `atol`, and `rtol` must be
positive finite scalars; `n_steps` must be a non-negative integer; and
`n_substeps` must be a positive integer. A zero-step direct call returns a
copy of the initial phase vector without requiring the optional backend
binary or runtime. Mojo subprocess output must contain exactly one raw stdout
line per oscillator phase; blank, truncated, or overlong output is rejected
before final phase validation.

::: scpn_phase_orchestrator.upde.engine
    options:
      members: false

## Bayesian UPDE Uncertainty Propagation

Samples natural frequencies and coupling matrices from explicit distributions,
runs the existing UPDE kernel for each draw, and reports posterior-predictive
`R ± sigma` with credible intervals and audit diagnostics.

::: scpn_phase_orchestrator.upde.bayesian

## JAX-Accelerated Kuramoto Engine

Optional JAX implementation for GPU-oriented Kuramoto rollouts. It preserves
the same validated inputs and phase wrapping semantics as the NumPy engine.

::: scpn_phase_orchestrator.upde.jax_engine

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

Use this engine when the physical or learned topology contains group effects
that cannot be decomposed into independent pairwise edges. Practical examples
include neural assemblies with co-active triplets, multi-agent coordination
under triangular constraints, reaction loops, power-network group modes, and
simplicial-complex or hypergraph-derived topology where synchronization
thresholds depend on 3-body motifs.

Direct Go, Julia, and Mojo simplicial accelerator entrypoints share the same
validated torus boundary before optional runtime loading: phase and frequency
vectors must be finite real one-dimensional `float64` arrays matching the
oscillator count; flattened pairwise coupling and phase-lag buffers must have
exactly `N*N` values; pairwise self-coupling `K_ii` must be zero because the
pairwise graph represents interactions between distinct oscillators; `zeta`,
`psi`, `sigma2`, `dt`, and `n_steps` must be finite non-boolean controls with
non-negative triadic strength, positive timestep, and non-negative step count.
Zero-step direct calls return a copy of the input phases without loading the
optional runtime. Backend outputs must be finite torus phases in `[0, 2*pi)`.

Gambuzza et al. 2023, Nature Physics; Tang et al. 2025.
**Detailed documentation:** [Simplicial (3-body) — detailed reference](upde_simplicial.md)

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

Direct Go, Julia, and Mojo market accelerator entrypoints share a validated
`float64` boundary before optional runtime loading: flattened phase payloads
must be finite real vectors with exactly `T*N` values; `T`, `N`, and PLV window
controls must be positive non-boolean integers; the PLV window must not exceed
`T`; backend `R(t)` outputs must have length `T` and lie in `[0, 1]`; rolling
PLV outputs must have the expected `(T-window+1)*N*N` cardinality, lie in
`[0, 1]`, preserve unit diagonals, and remain symmetric.

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

Direct Go, Julia, and Mojo swarmalator accelerator entrypoints share a
validated position-phase boundary before optional runtime loading: positions
must be finite real `float64` values with shape `(N, D)` or exactly `N*D`
flattened values; phase and frequency vectors must be finite real
one-dimensional `float64` arrays of length `N`; `N`, `D`, and `dt` must be
positive; and attraction, repulsion, phase-attraction modulation, and
phase-coupling coefficients must be finite real controls. Backend outputs must
return finite positions and torus phases in `[0, 2*pi)`, and Mojo stdout must
contain exactly `N*D + N` scalar lines.

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
**Detailed documentation:** [Geometric (SO(2)) — detailed reference](upde_geometric.md)

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

Direct Go, Julia, and Mojo Ott-Antonsen accelerator entrypoints share the
same scalar boundary before optional runtime loading: the complex order
parameter must lie inside the OA unit disk, Lorentzian width must be
non-negative, timestep and step count must be positive, and all scalar
controls must be finite non-boolean real values. Backend outputs are accepted
only when the returned complex state remains in the OA unit disk, `R` matches
`|z|`, and `psi` matches `atan2(Im(z), Re(z))`, preserving the physical
mean-field state contract across the polyglot chain.

**Detailed documentation:** [Ott-Antonsen Reduction — detailed reference](upde_reduction.md)

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

Direct Go, Julia, and Mojo order-parameter entrypoints share the same typed
boundary before optional runtime loading: phase payloads must be one-dimensional
finite real vectors, PLV inputs must be equal-length phase vectors, and
layer-coherence indices must be unique in-range oscillator indices. Empty
zero-measure calls return the neutral value without loading an optional runtime:
`(0.0, 0.0)` for global order, `0.0` for PLV, and `0.0` for layer coherence.
Backend outputs are accepted only when physical: `R`, PLV, and layer coherence
must be finite values in `[0, 1]`, and mean phase must be finite before being
canonicalised to the public `[0, 2*pi)` convention.

The release benchmark gate for this surface is:

```bash
python benchmarks/order_params_benchmark.py --parity-gate --sizes 64 --calls 1
```

It records Rust/Mojo/Julia/Go/Python status, timing, unavailable-toolchain
reasons, deterministic hashes, and tolerance-bounded parity against the forced
Python reference for global `R`, mean phase, PLV, and layer coherence. The
reference-suite snapshot exposes this gate as `order_parameter_polyglot`.

::: scpn_phase_orchestrator.upde.order_params

::: scpn_phase_orchestrator.upde.metrics

## Phase-Amplitude Coupling (PAC)

Modulation index (MI) via Tort et al. 2010. Bins low-frequency phase,
computes mean high-frequency amplitude per bin, KL divergence from uniform.
Produces N×N PAC matrix: entry [i,j] = MI(phase_i, amplitude_j).
Direct Go, Julia, and Mojo PAC entrypoints now share the same typed boundary
before optional runtime loading: phase and amplitude payloads must be finite
real `float64` vectors, amplitudes must be non-negative, `n_bins` must be a
non-boolean integer of at least two, and matrix calls require exactly `T*N`
flattened phase and amplitude samples. Empty common MI windows return zero
without loading optional runtimes. Backend MI and pairwise PAC outputs are
accepted only when finite, correctly sized, and inside the physical `[0, 1]`
interval. Direct Mojo PAC output must contain exactly one scalar line for
modulation-index calls or exactly N×N scalar lines for matrix calls; blank,
non-finite, truncated, or overlong text output is rejected before public
assembly.
Central to neuroscience cross-frequency coupling analysis.

::: scpn_phase_orchestrator.upde.pac

## Envelope & Numerics

Amplitude envelope extraction and numerical integration utilities
(DP54 coefficients, error estimation, step size control).
**Detailed documentation:** [Envelope (RMS) — detailed reference](upde_envelope.md)

::: scpn_phase_orchestrator.upde.envelope

::: scpn_phase_orchestrator.upde.numerics

## Splitting Engine

Operator-splitting UPDE integrator for stiff regimes and deterministic
phase update decomposition.

::: scpn_phase_orchestrator.upde.splitting

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
    options:
      members:
        - steady_state_r
        - basin_stability
        - multi_basin_stability

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
**Detailed documentation:** [Hypergraph (k-body) — detailed reference](upde_hypergraph.md)

::: scpn_phase_orchestrator.upde.hypergraph

## Strang Splitting Engine

Symmetric operator splitting: A(dt/2) → B(dt) → A(dt/2) where
A is exact rotation (ω·dt) and B is RK4 on coupling. Second-order
accurate, time-reversible, preserves symplectic structure approximately.

Direct Go, Julia, and Mojo Strang-splitting accelerator entrypoints share a
validated torus boundary before optional runtime loading: phase and frequency
vectors must be finite real one-dimensional `float64` arrays matching the
oscillator count; flattened pairwise coupling and phase-lag buffers must have
exactly `N*N` values; pairwise self-coupling `K_ii` must be zero; `zeta` and
`psi` must be finite controls; and direct accelerator `dt` plus `n_steps` must
be positive. Backend outputs must be finite torus phases in `[0, 2*pi)`, and
Mojo stdout must contain exactly one phase line per oscillator. The public
`SplittingEngine` still supports negative `dt` for reversibility checks by
using the Python reference path instead of direct optional accelerators.
**Detailed documentation:** [Strang Splitting — detailed reference](upde_splitting.md)

::: scpn_phase_orchestrator.upde.splitting

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

---

## Cellular Sheaf Engine

The  extends the Kuramoto model from scalar phases to **multi-dimensional phase vectors**. This implements the mathematical framework of Cellular Sheaves for synchronization.

Instead of a single phase $\theta_i$, each oscillator maintains a vector $\vec{\theta}_i \in \mathbb{R}^D$. The scalar coupling {ij}$ is replaced by a restriction map—a block matrix {ij} \in \mathbb{R}^{D \times D}$ that transforms the phase space of node $ into the reference frame of node $.

995781 \dot{\theta}_{i,d} = \omega_{i,d} + \sum_j \sum_k B_{ij}^{dk} \sin(\theta_{j,k} - \theta_{i,d}) + \zeta \sin(\Psi_d - \theta_{i,d}) 995781

### Features
- **Cross-Frequency Coupling:** A dimension $ on node $ (e.g., Theta wave) can directly drive dimension $ on node $ (e.g., Gamma wave) via off-diagonal elements in {ij}$.
- **Complex Topology:** Models opinion dynamics, multi-modal synchronization, and anisotropic structural constraints natively.
- **Rust Kernel:** Fully offloaded to the  via  for real-time multi-dimensional integration.

::: scpn_phase_orchestrator.upde.sheaf_engine

## Time-varying natural frequencies

`UPDEEngine` now accepts configured fixed or callable natural frequencies via
`omega=`. If a call omits the `omegas` argument, the engine resolves the
configured source at the current outer-step time and stores the resolved vector
in `omega_current`. Callable schedules are materialised as finite `(steps, n)`
matrices and dispatched through the Rust, Go, Julia, Mojo, or Python schedule
runner when available.

Use this for drifting oscillators, moving-agent frequency shifts, chirps,
thermal detuning, and Doppler preparation. The detailed contract is documented
in [UPDE — Time-varying omega](upde_time_varying_omega.md).
