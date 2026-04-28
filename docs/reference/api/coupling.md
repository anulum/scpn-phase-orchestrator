# Coupling

The coupling subsystem builds, adapts, and analyses the inter-oscillator
coupling matrix K_nm — the central object in Kuramoto dynamics. K_ij
determines how strongly oscillator j pulls oscillator i toward synchrony.

The subsystem spans 10 modules: construction (knm), geometry constraints,
phase lag estimation, template management, Hodge decomposition, spectral
analysis, plasticity, transfer-entropy adaptation, connectome generation,
E/I balance, and a universal Bayesian prior.

## Pipeline position

```
CouplingBuilder.build() ──→ K_nm, α ──→ UPDEEngine.step()
       ↑                                       │
  UniversalPrior                                ↓
  LagModel.estimate ────→ α            compute_order_parameter()
  connectome loader ─────→ K_nm                 │
  plasticity/TE ←────────────────── phase history
```

CouplingBuilder is the **entry point** of the SPO pipeline. Every engine
variant consumes `(phases, omegas, knm, zeta, psi, alpha)`, so the
coupling matrix and phase-lag matrix are required for any simulation.

---

## K_nm Construction

### CouplingBuilder

Builds coupling matrices from parameters.

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `build` | `(n_layers, base_strength, decay_alpha) → CouplingState` | Exponential-decay K_nm |
| `build_scpn_physics` | `(k_base=0.45, alpha_decay=0.3) → CouplingState` | 16-layer SCPN physics |
| `build_with_amplitude` | `(n, base, decay, amp_str, amp_dec) → CouplingState` | Phase + amplitude K |
| `apply_handshakes` | `(state, path) → CouplingState` | Overlay from JSON spec |
| `switch_template` | `(state, name, templates) → CouplingState` | Runtime topology switch |

### CouplingState (frozen dataclass)

| Field | Type | Description |
|-------|------|-------------|
| `knm` | `NDArray` | Phase coupling matrix K_ij |
| `alpha` | `NDArray` | Phase-lag matrix α_ij |
| `active_template` | `str` | Name of active template |
| `knm_r` | `NDArray \| None` | Amplitude coupling (Stuart-Landau) |

### Coupling equation

For the standard Kuramoto model, the coupling enters as:

```
dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j - θ_i - α_ij) + ζ sin(Ψ - θ_i)
```

K_ij is the (i,j) entry of the coupling matrix. The matrix must satisfy:

1. **Square:** K ∈ R^{N×N}
2. **Symmetric:** K_ij = K_ji (undirected coupling; directed via asymmetric K)
3. **Non-negative:** K_ij ≥ 0
4. **Zero diagonal:** K_ii = 0 (no self-coupling)

### Exponential-decay construction

`CouplingBuilder.build(n, base_strength, decay_alpha)` produces:

```
K_ij = base_strength × exp(-decay_alpha × |i - j|),  K_ii = 0
```

This generates nearest-neighbour-dominant coupling with exponential
fall-off — appropriate for layered systems where adjacent layers interact
more strongly than distant ones.

### SCPN physics construction

`build_scpn_physics(k_base=0.45, alpha_decay=0.3)` produces a 16×16 matrix
using three coupling mechanisms:

1. **Adjacent layers** (|i-j| = 1): timescale matching via
   `SCPN_LAYER_TIMESCALES` (Quantum: 1e-15s to Social: 3.15e7s)
2. **Near-neighbour** (|i-j| ≤ 3): geometric mean of adjacent couplings
3. **Distant** (|i-j| > 3): exponential decay from k_base

The 16 SCPN layers span 22 orders of magnitude in timescale:

| Layer | Name | Timescale |
|-------|------|-----------|
| L1 | Quantum | 1e-15 s |
| L2 | Sub-nuclear | 1e-12 s |
| L3 | Atomic | 1e-10 s |
| L4 | Molecular | 1e-9 s |
| L5 | Cellular | 1e-3 s |
| L6 | Neural | 1e-2 s |
| L7 | Synaptic | 1e-1 s |
| L8 | Circuit | 1 s |
| L9 | Regional | 10 s |
| L10 | Behavioural | 60 s |
| L11 | Cognitive | 600 s |
| L12 | Social | 3600 s |
| L13 | Cultural | 86400 s |
| L14 | Evolutionary | 3.15e6 s |
| L15 | Cosmological | 3.15e7 s |
| L16 | Director (meta) | — |

**Performance:** `build(100)` < 10 ms, `build_scpn_physics()` < 5 ms.

::: scpn_phase_orchestrator.coupling.knm

---

## Geometry Constraints

Enforces structural invariants on K_nm.

### Constraint classes

| Class | `project(knm)` behaviour |
|-------|--------------------------|
| `SymmetryConstraint` | Returns (K + K^T) / 2 |
| `NonNegativeConstraint` | Clamps negative entries to 0 |

### Validation

`validate_knm(knm, atol=1e-12)` checks all four invariants (square,
symmetric, non-negative, zero diagonal). Raises `ValueError` on violation.

`project_knm(knm, constraints)` applies constraints sequentially, then
zeros the diagonal.

::: scpn_phase_orchestrator.coupling.geometry_constraints

---

## Phase Lag Estimation

Estimates inter-oscillator phase lags α_ij from observed time series or
known physical distances.

### From distances

`LagModel.estimate_from_distances(distances, speed)` computes:

```
α_ij = 2π × distances[i,j] / speed
```

Returns an antisymmetric matrix: α_ij = -α_ji. This encodes the fact
that if signal from i reaches j with positive lag, then j reaches i with
negative lag.

### From cross-correlation

`LagModel().estimate_lag(signal_a, signal_b, sample_rate)` finds the
cross-correlation peak lag in seconds between two signals.

### Matrix construction

`build_alpha_matrix(lag_estimates, n_layers, carrier_freq_hz=1.0)` converts
pairwise lag estimates (in seconds) to a phase-offset matrix (in radians):

```
α_ij = 2π × carrier_freq_hz × lag_seconds_ij
```

**Performance:** `estimate_from_distances(64×64)` < 5 ms.

::: scpn_phase_orchestrator.coupling.lags

---

## Coupling Templates

Pre-configured coupling topologies for regime-dependent switching.

### KnmTemplate (frozen dataclass)

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Template identifier |
| `knm` | `NDArray` | Coupling matrix |
| `alpha` | `NDArray` | Phase-lag matrix |
| `description` | `str` | Human-readable description |

### KnmTemplateSet

Registry for named templates:

- `add(template)` — register (overwrites existing with same name)
- `get(name) → KnmTemplate` — retrieve (raises `KeyError` if missing,
  error message lists available names)
- `list_names() → list[str]` — all registered names

**Usage:** The supervisor can switch coupling topology at runtime by calling
`CouplingBuilder.switch_template(state, name, templates)` when a regime
transition occurs (e.g., switching from all-to-all to nearest-neighbour
when entering DEGRADED regime).

::: scpn_phase_orchestrator.coupling.templates

---

## Hodge Decomposition

Decomposes coupling dynamics into three orthogonal components via
Hodge theory (Jiang et al. 2011):

```
total coupling flow = gradient + curl + harmonic
```

### HodgeResult (dataclass)

| Field | Type | Physical meaning |
|-------|------|-----------------|
| `gradient` | `NDArray` | Conservative phase-locking (from potential) |
| `curl` | `NDArray` | Rotational circulation (antisymmetric flow) |
| `harmonic` | `NDArray` | Topological residual (null space of Laplacian) |

### Interpretation

- **Gradient-dominated:** system converges to fixed phase configuration.
  Common in strongly coupled networks.
- **Curl-dominated:** phases cycle through relationships without
  converging. Common in networks with asymmetric coupling.
- **Harmonic component:** topologically invariant under continuous
  deformation of coupling. In the SCPN consciousness model, this
  represents the identity invariant — the part of synchronisation
  that persists across regime changes.

`hodge_decomposition(knm, phases)` computes all three components.

::: scpn_phase_orchestrator.coupling.hodge

---

## Spectral Analysis

Algebraic graph-theoretic properties of the coupling network.

### Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `graph_laplacian(knm)` | `NDArray` | L = D - W (combinatorial Laplacian) |
| `fiedler_value(knm)` | `float` | λ₂(L) — algebraic connectivity |
| `fiedler_vector(knm)` | `NDArray` | Eigenvector of λ₂ |
| `critical_coupling(omegas, knm)` | `float` | K_c = max\|Δω\| / λ₂ |
| `fiedler_partition(knm)` | `(list, list)` | Network bisection via Fiedler sign |
| `spectral_gap(knm)` | `float` | λ₃ - λ₂ (cluster clarity) |
| `sync_convergence_rate(knm, omegas, γ_max)` | `float` | μ = K·λ₂·cos(γ)/N |

### Critical coupling estimate

The Dörfler-Bullo bound gives the minimum coupling strength for
synchronisation:

```
K_c = max_{i,j} |ω_i - ω_j| / λ₂(L)
```

where λ₂ is the Fiedler eigenvalue (algebraic connectivity). Networks
with higher λ₂ synchronise more easily.

::: scpn_phase_orchestrator.coupling.spectral

---

## Three-Factor Hebbian Plasticity

Coupling adaptation rule inspired by biological synaptic plasticity:

```
ΔK_ij = lr × eligibility_ij × modulator × phase_gate
```

### Functions

- `compute_eligibility(phases) → NDArray(n,n)`: pairwise Hebbian trace
  `cos(θ_j - θ_i)` with zero diagonal. In-phase pairs → +1 (strengthen),
  anti-phase → -1 (weaken).

- `three_factor_update(knm, eligibility, modulator, phase_gate, lr=0.01)
  → NDArray`: applies the three-factor rule. Only modifies K when all
  three factors are active.

### Three factors

1. **Eligibility** (local): cos(Δθ) — pairwise Hebbian trace
2. **Modulator** (global): scalar from L16 director layer (dopamine analog)
3. **Phase gate** (global): Boolean from TCBO consciousness boundary

**Reference:** Friston 2005 on free energy and synaptic plasticity.

::: scpn_phase_orchestrator.coupling.plasticity

---

## Transfer Entropy Adaptive Coupling

Directed causal adaptation that breaks symmetry:

```
K_ij(t+1) = (1 - decay) × K_ij(t) + lr × TE(i → j)
```

`te_adapt_coupling(knm, phase_history, lr=0.01, decay=0.0, n_bins=8)`:

- Computes transfer entropy TE(i→j) for all pairs from phase history
- Updates coupling: pairs with causal influence get stronger
- Applies decay to forget old coupling structure
- Clamps K ≥ 0 and zeros diagonal

Unlike Hebbian plasticity (symmetric), TE captures **directed**
information flow — oscillator i can influence j without j influencing i.

**Reference:** Lizier 2012, "Local Information Transfer as
Spatiotemporal Filter."
**Detailed documentation:** [TE Adaptive — detailed reference](coupling_te_adaptive.md)

::: scpn_phase_orchestrator.coupling.te_adaptive

---

## E/I Balance

Computes and adjusts excitatory/inhibitory coupling balance.

### EIBalance (dataclass)

| Field | Type | Description |
|-------|------|-------------|
| `ratio` | `float` | E/I balance ratio |
| `excitatory_strength` | `float` | Mean excitatory coupling |
| `inhibitory_strength` | `float` | Mean inhibitory coupling |
| `is_balanced` | `bool` | True if 0.8 ≤ ratio ≤ 1.2 |

### Functions

- `compute_ei_balance(knm, excitatory_indices, inhibitory_indices)
  → EIBalance`
- `adjust_ei_ratio(knm, excitatory_indices, inhibitory_indices,
  target_ratio=1.0) → NDArray` — scales inhibitory coupling to
  achieve target ratio

::: scpn_phase_orchestrator.coupling.ei_balance

---

## Universal Bayesian Prior

Gaussian prior over coupling parameters, calibrated from the SCPN
experimental programme.

### CouplingPrior (dataclass)

| Field | Type | Default |
|-------|------|---------|
| `K_base` | `float` | 0.47 |
| `decay_alpha` | `float` | 0.25 |
| `K_c_estimate` | `float` | 0.0 |

### UniversalPrior

- `default() → CouplingPrior` — MAP estimate (K_base=0.47, α=0.25)
- `sample(rng=None) → CouplingPrior` — random draw from prior
- `estimate_Kc(omegas, n_layers) → CouplingPrior` — combines prior
  with Dörfler-Bullo K_c for given omegas
- `log_probability(K_base, decay_alpha) → float` — unnormalised
  log-probability under Gaussian prior
**Detailed documentation:** [Universal Prior — detailed reference](coupling_prior.md)

::: scpn_phase_orchestrator.coupling.prior

---

## HCP Connectome Generator

Neuroscience-realistic coupling matrices.

### Synthetic generator

`load_hcp_connectome(n_regions, seed=42)` generates a matrix with:

- Intra-hemispheric: exponential distance decay
- Inter-hemispheric: corpus callosum pattern (homotopic connections)
- Default Mode Network: hub structure with elevated coupling

### Real data bridge

`load_neurolib_hcp(n_regions=80)` loads real HCP structural connectivity
from the neurolib library. Supports n_regions from 2 to 80.

**Performance:** `load_hcp_connectome(80)` < 10 ms (Python), ~48 µs (Rust, 17.6x speedup).
**Detailed documentation:** [HCP Connectome — detailed reference](coupling_connectome.md)

::: scpn_phase_orchestrator.coupling.connectome

---

## Rust FFI acceleration

`spo_kernel.PyCouplingBuilder` provides Rust-accelerated K_nm
construction. The Python implementation is the reference; the Rust path
is selected automatically when `spo_kernel` is importable. Parity is
verified in `tests/test_rust_python_parity_performance.py`.

## Performance summary

| Operation | Budget | Measured |
|-----------|--------|----------|
| `CouplingBuilder.build(100)` | < 10 ms | ~2 ms |
| `build_scpn_physics()` | < 5 ms | ~1 ms |
| `estimate_from_distances(64)` | < 5 ms | ~0.5 ms |
| `load_hcp_connectome(80)` | < 10 ms | ~3 ms |
| `validate_knm(64)` | < 1 ms | ~0.1 ms |
| `graph_laplacian(64)` | < 1 ms | ~0.007 ms |
| `fiedler_value(64)` | < 1 ms | ~0.12 ms |
