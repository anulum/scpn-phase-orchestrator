# Coupling

The coupling subsystem builds, adapts, and analyses the inter-oscillator
coupling matrix K_nm — the central object in Kuramoto dynamics. K_ij
determines how strongly oscillator j pulls oscillator i toward synchrony.

The subsystem spans 27 source files: public API modules for construction
(knm), geometry constraints, phase lag estimation, template management, Hodge
decomposition, spectral analysis, plasticity, transfer-entropy adaptation,
causal inference, connectome generation, E/I balance, attention residuals,
spatial modulation, and a universal Bayesian prior, plus validated backend
bridge files.

## Pipeline position

```
CouplingBuilder.build() ──→ K_nm, α ──→ UPDEEngine.step()
       ↑                                       │
  UniversalPrior                                ↓
  LagModel.estimate ────→ α            compute_order_parameter()
  connectome loader ─────→ K_nm                 │
  auto-coupling-estimation ← raw phase time series
  plasticity/TE ←────────────────── phase history
```

CouplingBuilder is the **entry point** of the SPO pipeline. Every engine
variant consumes `(phases, omegas, knm, zeta, psi, alpha)`, so the
coupling matrix and phase-lag matrix are required for any simulation.
For data-first onboarding, `auto_coupling_estimation()` infers an initial
directed coupling graph from phase time series before review, projection, or
engine execution.
The inference boundary requires finite real phase samples and enforces the
transfer-entropy invariant that directed scores are non-negative with no
self-edge diagonal.
Across the coupling public boundary, boolean aliases mean Python `bool`,
NumPy boolean scalars, and object arrays containing either form; those inputs
are rejected before any float coercion.

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

`apply_handshakes()` parses the JSON specification fail-closed: non-finite
constants, duplicate object keys, non-list `matrix` payloads, self-coupled
entries, and out-of-range layer indices are rejected before any K_nm entries
are modified.

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

`validate_knm(knm, atol=1e-12)` accepts only finite real square matrices and
checks all four invariants: symmetric, non-negative, zero diagonal, and
boolean/complex aliases rejected before numeric projection. Raises
`ValueError` on violation.

`project_knm(knm, constraints)` applies constraints sequentially, then
zeros the diagonal. Built-in and custom constraints are fail-closed: each
constraint must be a `GeometryConstraint`, preserve the matrix shape, and
return finite real square K_nm values before the next projection step.

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

Inputs must be a finite real square physical-distance matrix with
non-negative entries, a zero diagonal, and symmetric pair distances, plus a
finite positive propagation speed. Boolean aliases and complex/object-complex
distance payloads are rejected before numeric coercion because transport
delays are ordered real quantities. Returns an antisymmetric matrix:
α_ij = -α_ji. This encodes the fact that if signal from i reaches j with
positive lag, then j reaches i with negative lag. Directed or asymmetric
empirical delays belong in `build_alpha_matrix`, not in the physical-distance
constructor.

### From cross-correlation

`LagModel().estimate_lag(signal_a, signal_b, sample_rate)` finds the
cross-correlation peak lag in seconds between two signals. Signals must be
finite real one-dimensional arrays with equal non-zero length and non-zero
variance. The sample-rate must be a finite positive real value. Constant,
boolean, complex/object-complex, non-finite, or length-mismatched signals are
rejected before cross-correlation because they do not define a reliable
phase-lag estimate.

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

## Combinatorial Hodge Decomposition

Decomposes the Kuramoto coupling current into three L²-orthogonal
edge-flow components via combinatorial Hodge theory (Jiang, Lim, Yao &
Ye 2011, *Statistical ranking and combinatorial Hodge theory*,
Math. Program. **127** (1):203–244):

```
coupling current  f = gradient ⊕ curl ⊕ harmonic
```

The oscillator network is treated as a simplicial complex `(V, E, T)`:
vertices are oscillators, edges are the pairs `{i, j}` with non-zero
symmetric coupling, and triangles are the 3-cliques of that graph (or an
explicit user-supplied set). The decomposed object is the alternating
edge flow

```
f_ij = ½(K_ij + K_ji) · sin(θ_j − θ_i)
```

— the canonical coupling current, built from the symmetric coupling part
so it satisfies `f_ji = −f_ij`. With node–edge incidence `B1` and
edge–triangle incidence `B2`:

```
gradient = B1ᵀ · L0⁺ · (B1 f)     # curl-free conservative flow
curl     = B2  · L2⁺ · (B2ᵀ f)    # divergence-free rotational flow
harmonic = f − gradient − curl    # ker of the Hodge 1-Laplacian
```

where `L0 = B1 B1ᵀ` and `L2 = B2ᵀ B2`. Because `B1 B2 = 0`, the three
components are mutually L²-orthogonal.

### HodgeResult (dataclass)

| Field | Type | Physical meaning |
|-------|------|-----------------|
| `gradient` | `NDArray (N, N)` | Conservative (curl-free) flow `grad(s)` |
| `curl` | `NDArray (N, N)` | Rotational (divergence-free) flow bounded by triangles |
| `harmonic` | `NDArray (N, N)` | Topological residual in `ker(L1)` (non-zero only on cycles not filled by triangles) |
| `flow` | `NDArray (N, N)` | The input alternating coupling current |
| `potential` | `NDArray (N,)` | Minimum-norm node potential `s` with `gradient = grad(s)` |
| `betti_one` | `int` | First Betti number `β₁` — dimension of the harmonic subspace |

Each flow matrix is antisymmetric (`M[i, j]` is the flow on the oriented
edge `i → j`, `M[j, i] = −M[i, j]`).

### Interpretation

- **Gradient-dominated:** the current is a node-potential difference and
  the system relaxes towards a fixed phase configuration.
- **Curl-dominated:** circulation around filled triangles — local cyclic
  frustration with no global potential.
- **Harmonic component:** flows around topological cycles that no
  triangle bounds; its dimension equals the first Betti number `β₁`. On
  a triangle-free graph carrying a cycle (for example, a 4-cycle), a
  circulating current is *purely harmonic* — the topological content
  that a plain symmetric/antisymmetric matrix split cannot represent. In
  the SCPN consciousness model this is the identity invariant that
  persists across regime changes.

`hodge_decomposition(knm, phases, triangles=None)` computes all three
components; pass an explicit `triangles` list of node triples to override
the default 3-clique fill.

Because the decomposition relies on two least-squares pseudoinverse
solves, exact cross-language parity is not attainable; the dispatcher
validates each accelerated backend against the NumPy reference within
`rtol = 1e-10` / `atol = 1e-12` (matching the spectral solver) and falls
back to NumPy only after the backend has returned a valid Hodge payload.

Direct accelerator boundary contract: the public Python dispatcher, public Rust
wrapper, and the Go, Julia, and Mojo Hodge adapters reject numeric-string
aliases before Python, NumPy, shared-library, Julia, or subprocess coercion.
The public surface applies the boundary to `knm`, `phases`, and explicit
triangle nodes; the direct adapters apply it to counts, flattened coupling,
phase, edge, triangle, backend-output, and Julia raw-return payloads. The
shared typed `float64` path also rejects boolean aliases, complex or non-finite
payloads, malformed flattened `n*n` coupling buffers, phase vectors whose
length does not match `n`, and invalid oscillator counts before optional runtime
loading. After backend execution, the same output validator checks that
`gradient`, `curl`, and `harmonic` are finite real non-boolean `(N, N)` or
flattened `N*N` antisymmetric matrices before publication or parity fallback.
Malformed backend outputs raise immediately; fallback is reserved for validated
numerical parity mismatches. Empty Hodge systems return empty components
without requiring optional runtimes, matching the public Python special case.

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

Direct accelerator boundary contract: Go, Julia, and Mojo spectral adapters use
one shared typed `float64` validation path before loading shared-library, Julia,
or subprocess runtimes. The contract rejects boolean aliases, complex or
non-finite flattened coupling payloads, non-vector inputs, malformed `n*n`
buffer lengths, and invalid oscillator counts. Empty spectral problems return
empty eigenvalue and Fiedler vectors without optional runtime loading.
After backend execution, the same shared output validator is replayed for the
direct Go, Julia, and Mojo adapters and for the public optional primitive path:
returned eigenvalues and the Fiedler vector must be finite real non-boolean
vectors of length `N`, eigenvalues must be non-negative and sorted ascending,
and the Fiedler vector must be non-zero for `N > 1`. Malformed backend physics
payloads raise immediately; fallback remains reserved for loader or runtime
unavailability.
Public spectral helpers enforce the same real-valued boundary on coupling
matrices, frequency vectors, `gamma_max`, optional primitive eigensystem
outputs, and Rust fast-path scalar/vector returns. Boolean aliases are not
coerced into weights or frequencies, and complex-valued aliases are rejected
before NumPy can discard imaginary components.

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
  three factors are active. The boundary enforces the same physical K_nm
  contract consumed by the UPDE engines: `knm` must be finite, real,
  non-negative, square, and zero-diagonal; `eligibility` must be finite, real,
  square, zero-diagonal, and bounded in `[-1, 1]`. Negative modulation can
  depress coupling but is clamped at zero, and the result always keeps a
  zero self-coupling diagonal.

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
- Rejects boolean aliases in both `knm` and `phase_history` before numeric
  coercion

Unlike Hebbian plasticity (symmetric), TE captures **directed**
information flow — oscillator i can influence j without j influencing i.

**Reference:** Lizier 2012, "Local Information Transfer as
Spatiotemporal Filter."
**Detailed documentation:** [TE Adaptive — detailed reference](coupling_te_adaptive.md)

::: scpn_phase_orchestrator.coupling.te_adaptive

---

## E/I Balance

Computes and adjusts excitatory/inhibitory coupling balance. The aggregate
`ratio` summarises overall balance, while the four directed interaction-type
means resolve it into the source→target block strengths that Kuroki &
Mizuseki 2025 (*Neural Computation* **37** (7):1353–1372) identify as the
control parameters of the EI-Kuramoto synchronised / bistable /
desynchronised regimes.

### EIBalance (dataclass)

| Field | Type | Description |
|-------|------|-------------|
| `ratio` | `float` | E/I balance ratio (`excitatory_strength / inhibitory_strength`) |
| `excitatory_strength` | `float` | Mean coupling from excitatory sources over all targets |
| `inhibitory_strength` | `float` | Mean coupling from inhibitory sources over all targets |
| `is_balanced` | `bool` | True if 0.8 ≤ ratio ≤ 1.2 |
| `e_to_e` | `float` | Mean E→E interaction-type coupling |
| `e_to_i` | `float` | Mean E→I interaction-type coupling |
| `i_to_e` | `float` | Mean I→E interaction-type coupling |
| `i_to_i` | `float` | Mean I→I interaction-type coupling |

Each aggregate strength is the count-weighted blend of its two outgoing
interaction-type blocks (e.g. `excitatory_strength` blends `e_to_e` and
`e_to_i` over the target-group sizes).

### Functions

- `compute_ei_balance(knm, excitatory_indices, inhibitory_indices)
  → EIBalance`
- `adjust_ei_ratio(knm, excitatory_indices, inhibitory_indices,
  target_ratio=1.0) → NDArray` — scales inhibitory coupling to
  achieve target ratio

Both helpers reject boolean aliases in `knm` before computing row means or
scaling inhibitory rows.

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
- `sample(rng=None, seed=None) → CouplingPrior` — random draw from prior;
  `seed` must be an integer in the unsigned 64-bit range when provided
- `estimate_Kc(omegas, n_layers) → CouplingPrior` — combines prior
  with Dörfler-Bullo K_c for a finite one-dimensional frequency vector
- `log_probability(K_base, decay_alpha) → float` — unnormalised
  log-probability under Gaussian prior

`estimate_Kc` rejects boolean aliases in `omegas`, including NumPy boolean
scalars carried inside object arrays, before constructing the prior graph.

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

## Spatial coupling modulation

`SpatialCouplingModulator` is the public PHA-C.1 coupling surface for systems where the effective phase coupling must depend on moving geometry instead of static oscillator labels. It turns a zero-diagonal base `K_nm` matrix and a position matrix into a physically constrained modulated coupling matrix.

Use it when spatial proximity, mobile agents, tissue geometry, sensor placement, or edge-node distance changes the strength of phase transfer. The default kernel is `1 / (1 + distance)`, which is bounded, finite at zero separation, symmetric for Euclidean positions, and preserves the zero self-coupling diagonal required by the oscillator engines.

The module also exposes exponential, power-law, and inverse-distance kernels. The inverse-distance form is reserved for Swarmalator compatibility and uses an epsilon-regularised denominator so the historical kernel remains bit-true without introducing singularities.

The reference implementation is NumPy. Rust, Go, Julia, and Mojo adapters are validated as optional accelerators and must reproduce the same invariants before their output is accepted: finite real-valued matrices, exact shape or flat cardinality, non-boolean and non-complex values, non-negative entries, zero diagonal, and symmetry preservation for symmetric inputs. Public positions, base coupling matrices, scalar decay controls, direct accelerator counts/forms/flat buffers, optional backend outputs, and raw Julia returns reject numeric-string aliases before float coercion. The public dispatcher preserves matrix-shaped output for callers after replaying the shared direct output validator; optional backend fallback remains limited to loader or runtime unavailability.

See [Coupling - Spatial Modulator](coupling_spatial_modulator.md) for examples, backend notes, and the benchmark contract.
