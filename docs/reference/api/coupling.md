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
constants, duplicate object keys, non-mapping roots, non-list `matrix` payloads, self-coupled
entries, and out-of-range layer indices are rejected before any K_nm entries
are modified.

Construction coefficients and layer counts reject boolean, text and temporal
aliases before numeric conversion. Dictionary template switching accepts real
numeric object arrays, but refuses text, complex, boolean and temporal values
before changing the state. `KnmTemplateSet` retains its stricter floating-point
dtype contract and returns independent copies of stored matrices.
The installed Rust builder applies the same original scalar-type checks;
its projection checks the original sequence elements, and construction rejects
an overflowing matrix-size product or binary64 byte capacity before allocation.

Construction accepts all finite non-negative binary64 strength/decay values,
including subnormals and the largest finite value. Exponential underflow rounds
to zero; overflowing negative exponents also yield zero. These operations use a
local NumPy error context and restore the caller's warning/error policy. The
phase and amplitude diagonals remain zero. Impossible square binary64 buffers
are refused before native dispatch or NumPy index allocation; representable
capacity does not promise available physical memory.

The SCPN builder preserves the four anchors and existing clipping/boost rules.
It checks all16 exported timescales as finite and positive, evaluates adjacent
mismatch as a difference of logarithms, and scales reciprocal frequencies by a
common factor for the near-neighbour penalty. These are numerically stable forms
of the same equations; timescale values and scientific calibration are unchanged.
Handshake specifications require a JSON mapping with a list-valued `matrix`.
A failed overlay leaves its source unchanged. Template and overlay results copy
phase/lag matrices; the optional amplitude matrix retains its existing shared
reference. A frozen record does not make the contained NumPy arrays immutable.


The [historical construction and projection benchmark](../data/coupling_builder_measurement_types_benchmark_2026-09-26.json)
records actual kernel-absent Python and release-kernel environments, source
hashes and a reproduction script. It verifies public construction and projection
parity at `rtol=atol=1e-12` for 4, 16 and 64 layers, including an observed native
build call from the public API. The public builder currently selects Rust when
available; this snapshot does not establish that Rust construction is faster.
Shared host load and different NumPy versions limit timing comparisons.

The [current construction measurements](../data/coupling_builder_runtime_benchmark_2026-10-02.json)
use identical scalar bits at16,64 and100 layers for actual kernel-absent Python,
public PyO3 dispatch and standalone Rust. Generic and amplitude timings include
public snapshot construction; direct PyO3 timing also includes conversion to
NumPy arrays. SCPN construction is Python-only. Numerical checks run outside
measured intervals after two warm-ups. An existing native or Python caller
profiler is suspended during the complete measurement and restored on success
or refused construction. Individual samples, native call
observations, source/binary pins, interpreter/NumPy versions and shared host
load accompany the data. Reproduce with
`python -m benchmarks.coupling_builder_benchmark --sizes 16 64 100 --calls 100 --repeats 5`
and `cargo bench -p spo-engine --bench coupling_builder_bench -- --noplot` from
`spo-kernel`. Use separate actual native and absent interpreters; timing ratios
across differing NumPy versions or a shared host are not causal speed-up claims.


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

`build_scpn_physics(k_base=0.45, alpha_decay=0.3)` produces a symmetric
16×16 matrix with zero diagonal and phase lags. For one-based layer indices,
construction uses three distance classes:

1. **Adjacent layers** (`|i-j| = 1`): the four declared anchors for L1–L5
   override timescale matching. Other adjacent pairs use
   `k_base / (1 + 0.05 * |log(tau_i) - log(tau_j)|)`, clipped to `[0.1, 0.5]`.
   Adjacency to L16 has the fixed value `0.2`.
2. **Near-neighbours** (`|i-j| = 2`): the geometric mean of the two adjacent
   path couplings, divided by
   `1 + 0.1 * |omega_i - omega_j| / ((omega_i + omega_j) / 2)`, where
   `omega = 1 / tau`. The implementation scales the reciprocal frequencies
   before evaluating this penalty. The penalty is omitted for pairs involving
   L16; the result is clipped to `[0.01, 0.4]`.
3. **Distant layers** (`|i-j| >= 3`):
   `k_base * exp(-alpha_decay * |i-j|)`, clipped to `[0.001, 0.2]`.

After those passes, the symmetric L1–L16 coupling is at least `0.05` and
L5–L7 is at least `0.15`. These boosts take precedence over the distance rules.

The exported `SCPN_LAYER_NAMES` and `SCPN_LAYER_TIMESCALES` define these inputs:

| Layer | Name | Timescale |
|-------|------|-----------|
| L1 | Quantum | 0.1 s |
| L2 | Neural | 0.004 s |
| L3 | Genomic | 3600 s |
| L4 | Tissue | 2 s |
| L5 | Psycho | 1 s |
| L6 | Planetary | 86400 s |
| L7 | Symbolic | 10 s |
| L8 | Cosmic | 31557600 s |
| L9 | Memory | 3.154e9 s |
| L10 | Boundary | 1 s |
| L11 | Noospheric | 86400 s |
| L12 | Gaian | 31557600 s |
| L13 | Source | 0.001 s |
| L14 | Transdim | 1e-20 s |
| L15 | Consilium | 1 s |
| L16 | Meta | 1 s |

These are declared model inputs; this implementation check does not establish
external physical calibration. L2 remains `0.004 s`, although its source comment
says approximately 25 ms and 40 Hz. That discrepancy remains unresolved; no
numerical constant has been changed to reconcile it. L16's timescale is validated
with the complete table, while its adjacency and near-neighbour penalty use the
explicit rules above.

For measured construction timings and their environment boundaries, use the
[current construction measurements](../data/coupling_builder_runtime_benchmark_2026-10-02.json).

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
boolean, complex, text and temporal aliases rejected before numeric projection.
Real numeric object arrays remain supported. Raises
`ValueError` on violation.

`project_knm(knm, constraints)` applies constraints sequentially, then
zeros the diagonal. Built-in and custom constraints are fail-closed: each
constraint must be a `GeometryConstraint`, preserve the matrix shape, and
return finite real square K_nm values before the next projection step.
The empty `(0, 0)` matrix is valid for projection and validation. Constraint
order is significant: averaging signed edges before clipping differs from
clipping each directed edge before averaging.

`SymmetryConstraint` preserves finite pair means when `K_ij + K_ji` would
overflow. It adds before halving for ordinary and subnormal pairs, and halves
first only for overflowing sums. For example, two `float64` maximum entries
remain the maximum finite value; two least positive subnormal entries remain
that subnormal. The standalone constraint preserves the diagonal; the complete
`project_knm` chain zeros it after all constraints.

The direct `spo_kernel.PyCouplingBuilder.project(flat_values, n)` performs
symmetry, then non-negativity, then diagonal zeroing in Rust. It accepts an
empty buffer with `n=0`, validates original source types and exact `n*n`
cardinality, and returns a new flat list without modifying the caller buffer.
Python projection executes NumPy; calling the native method explicitly exercises
Rust. These are equivalent for the two built-in constraints in that order,
not for arbitrary custom constraint stacks.

[Current projection measurements](../data/geometry_projection_runtime_benchmark_2026-10-02.json)
include repeated public NumPy and direct native measurements, actual input and
source hashes, runtime versions and host load. Run
`python -m benchmarks.geometry_projection_benchmark` in both prepared
interpreters. Shared-host values are local regression evidence; they establish
no isolated speed-up or production latency guarantee.

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
finite positive propagation speed. Boolean, complex, textual and temporal
distance payloads are rejected before numeric coercion; real numeric object
matrices remain supported. The direct Rust `PyLagModel.estimate` boundary
checks original flat distance values, counts and speed before extraction. Transport
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
boolean, complex/object-complex, textual, temporal, non-finite, or length-mismatched signals are
rejected before cross-correlation because they do not define a reliable
phase-lag estimate.

### Matrix construction

`build_alpha_matrix(lag_estimates, n_layers, carrier_freq_hz=1.0)` converts
pairwise lag estimates (in seconds) to a phase-offset matrix (in radians):

```
α_ij = 2π × carrier_freq_hz × lag_seconds_ij
```

Counts, lag indices, measured lag scalars, carrier frequency and sample rate
reject temporal aliases before integer or real conversion.

[Python/Rust measurements from 2026-09-26](../data/lag_measurement_types_benchmark_2026-09-26.json)
cover actual distance estimators. Shared-host small fixtures do not establish
production scaling or a controlled before/after speed-up.

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

Registration stores independent, contiguous `float64` matrix copies. Finite
floating-point inputs must also remain finite after precision narrowing;
a conversion that would produce infinity raises `ValueError`, including when
NumPy is configured to raise on overflow. Refusal leaves the existing registry
and caller matrices unchanged. Representable narrower or extended-precision
matrices remain accepted; retrieval returns independent copies.

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
  the SCPN identity-coherence model this is the identity invariant that
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
phase, edge, triangle, backend-output, and Julia raw-return payloads. Direct
Go, Julia, and Mojo edge and triangle indices must be exactly integer-valued
before conversion; fractional, non-finite, temporal, and overflowing indices
are rejected. Exact integer-valued float arrays retain their existing support. The
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
or subprocess runtimes. The contract rejects boolean aliases, numeric-string
aliases, complex or non-finite flattened coupling payloads, non-vector inputs,
malformed `n*n` buffer lengths, and invalid oscillator counts. Empty spectral
problems return empty eigenvalue and Fiedler vectors without optional runtime
loading.
After backend execution, the same shared output validator is replayed for the
direct Go, Julia, and Mojo adapters and for the public optional primitive path:
returned eigenvalues and the Fiedler vector must be finite real non-boolean,
non-numeric-string vectors of length `N`, eigenvalues must be non-negative and
sorted ascending, and the Fiedler vector must be non-zero for `N > 1`.
Malformed backend physics payloads raise immediately; fallback remains reserved
for loader or runtime unavailability.
Public spectral helpers enforce the same real-valued boundary on coupling
matrices, frequency vectors, `gamma_max`, optional primitive eigensystem
outputs, and Rust fast-path scalar/vector returns. Boolean aliases are not
coerced into weights or frequencies, and complex-valued aliases are rejected
before NumPy can discard imaginary components. Numeric-string aliases are
rejected before Python, NumPy, Rust, Julia, Go, or Mojo can widen them into
ordinary floating-point weights, frequencies, scalar controls, or eigensystem
payloads.

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

Phase, coupling and eligibility arrays must contain plain real numbers.
Numeric strings, bytes, booleans, complex values, `datetime64` and `timedelta64`
are rejected before conversion, including those inside object arrays. Object
arrays containing real numeric values remain accepted. Temporal counts are
not phases in radians or coupling strengths; convert units explicitly at the
measurement boundary before calling plasticity functions.

### Three factors

1. **Eligibility** (local): cos(Δθ) — pairwise Hebbian trace
2. **Modulator** (global): scalar from L16 director layer (dopamine analog)
3. **Phase gate** (global): Boolean from topological-integration gate

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

Computes signed arithmetic summaries of caller-declared source groups and
optionally rescales inhibitory rows. These are numerical coupling diagnostics;
the balance flag is not a physiological validation or a universal
synchronisation criterion.

### EIBalance (dataclass)

| Field | Type | Description |
|-------|------|-------------|
| `ratio` | `float` | Signed E/I mean quotient, with the silent-inhibition convention below |
| `excitatory_strength` | `float` | Mean coupling from unique excitatory sources over all targets |
| `inhibitory_strength` | `float` | Mean coupling from unique inhibitory sources over all targets |
| `is_balanced` | `bool` | True if 0.8 ≤ ratio ≤ 1.2 |
| `e_to_e` | `float` | Mean E→E directed block, including diagonal entries |
| `e_to_i` | `float` | Mean E→I directed block |
| `i_to_e` | `float` | Mean I→E directed block |
| `i_to_i` | `float` | Mean I→I directed block, including diagonal entries |

Each group is a set: duplicates count once and non-negative out-of-range
indices are ignored. Negative, boolean and non-integral public indices refuse.
Groups may overlap or leave targets untyped. Each aggregate mean blends its
two directed blocks by target counts **only when the target groups partition
all oscillators**. Empty source or target blocks have zero mean. Scaled,
compensated aggregation prevents overflow of an otherwise representable mean.
Float64 normalisation can discard contributions below its range; arbitrary
exponent cancellation is not guaranteed to produce a correctly rounded mean.

### Functions

- `compute_ei_balance(knm, excitatory_indices, inhibitory_indices)
  → EIBalance`
- `adjust_ei_ratio(knm, excitatory_indices, inhibitory_indices,
  target_ratio=1.0) → NDArray` — scales each unique inhibitory row once

The summary uses signed strengths. If the inhibitory mean has magnitude
below `1e-15`, its ratio is infinity for positive excitation and one
otherwise; other ratios are signed quotients and may overflow to infinity.
Adjustment returns an independent unchanged copy when either source mean
has magnitude below `1e-15` or the ratio is within `1e-10` of target.
Otherwise it scales by `current_ratio / target_ratio`. Target attainment
requires disjoint source groups, adequate float64 precision and an adjusted
inhibitory mean with magnitude at least `1e-15`. Below that threshold the
summary uses its silent convention, even for disjoint groups; an overlapping
row changes both means.

Both public helpers reject boolean, complex, text and temporal coupling
aliases before conversion, while preserving finite real numeric object
matrices. The target must be a finite positive non-boolean real. A non-finite
scale, a scale that rounds to either signed zero, or a non-finite adjusted
element raises `ValueError`; caller bytes remain unchanged
on success and refusal. Public strided and readonly inputs are normalised for
native admission, and every successful adjustment owns independent memory.
The representability requirement applies to the intermediate scale itself:
`[[0, 2], [1e308, 0]]` with E=`[0]`, I=`[1]`, target=`1e308` refuses because
`current_ratio / target_ratio` underflows, even though a mathematically
rescaled individual entry could be represented. A representable non-zero
scale remains admissible; individual products retain float64 rounding.

The installed Rust functions retain their float64 coupling and int64 index
ndarray ABI and require contiguous one-dimensional buffers. They validate
exact `n * n` cardinality, count overflow, finite coupling and non-negative
indices before indexing. Bad buffers/targets/numerics raise `ValueError`;
coercion aliases reject before native scalar extraction. Empty `n=0`
matrices remain valid. Rust core callers now handle `SpoResult<EIBalanceResult>`
and `SpoResult<Vec<f64>>`, including the owning Criterion benchmark.

[Current actual-runtime measurements](../data/ei_balance_runtime_benchmark_2026-10-01.json)
retain three repetitions for summary and adjustment at N=16/64/256 in both
the installed native and genuinely kernel-absent public paths, plus input and
binary provenance. Reproduce each interpreter with
`PYTHONPATH=src python benchmarks/ei_balance_benchmark.py --sizes 16 64 256 --calls 10 --repeats 3`.
The repository diagnostic also exposes `native_binary_provenance(module_name)`
to resolve and hash the actual extension file, supporting both standalone
modules and same-name members of packaging wrappers. Source-only candidates
refuse native attribution; an actually absent module returns no binary.
`validate_ei_measurement(matrix, excitatory_indices, inhibitory_indices,
summary, adjusted, target_ratio)` checks independently obtained real results
against their declared benchmark input and target before any timing loop.
It retains independent matrix means, attained-ratio and adjusted-row checks.
This comparison contract uses moderate positive coupling and a disjoint,
nonempty source partition; it does not widen the numerical target guarantee
for signed, silent or overlapping groups.
The direct Rust core benchmark is
`cd spo-kernel && cargo bench -p spo-engine --bench utility_bench -- compute_ei_balance`.
These are non-isolated shared-host diagnostics; NumPy versions differ.
They establish exercised semantics and local timings, not backend speedup
or production latency. The
[2026-09-26 measurement-type snapshot](../data/ei_balance_measurement_types_benchmark_2026-09-26.json)
is retained as historical ingress evidence.

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

`estimate_Kc` requires plain real frequencies in radians per second. It rejects
text, boolean, complex and temporal aliases, including those carried inside
object arrays, before constructing the prior graph. Numeric-object arrays
containing real values remain supported.

**Detailed documentation:** [Universal Prior — detailed reference](coupling_prior.md)

::: scpn_phase_orchestrator.coupling.prior

---

## HCP Connectome Generator

`load_hcp_connectome(n_regions, seed=42)` generates synthetic intra-half,
callosal and repeated-hub structural weights. It requires a genuine integer
count of at least two, addressable dense float64 storage and an unsigned 64-bit
integer seed. Available RAM is an additional constraint. Python uses PCG64
Gaussian noise; the original Rust builtin uses LCG uniform noise, so seeded
matrices are deterministic within each owner rather than elementwise equal.
A 128-entry cache retains validated matrices; every call publishes a copy.

`load_neurolib_hcp(n_regions=80)` loads the original optional neurolib subject
average in cortical AAL2/LRLR ordering. Counts 2 through 80 return top-left
slices in that ordering. Counts are validated before dataset I/O.

Both paths reject source aliases and invalid structural weights before returning
finite, non-negative symmetric C-contiguous float64 matrices. Synthetic output
must have an exact zero diagonal; the HCP ingress clears the admitted provider
diagonal. Native square/byte overflow and reservation errors become `ValueError`.

[Contracts, equations, actual consumers and current cold/warm measurements](coupling_connectome.md).
Public validation and marshalling costs are included in those public measurements;
uncached Rust-core generation is reported separately without a speed-up claim.

::: scpn_phase_orchestrator.coupling.connectome

---

## Rust FFI acceleration

`spo_kernel.PyCouplingBuilder` provides Rust-accelerated K_nm
construction. The Python implementation is the reference; the Rust path
is selected automatically when `spo_kernel` is importable. Parity is
verified in `tests/test_rust_python_parity_performance.py`.

Rust builder returns are inspected before numeric conversion: boolean,
complex/object-complex, and numeric-string `K_nm` or `alpha` aliases are rejected
and trigger the documented NumPy fallback. Finite real numeric-object matrices
remain compatible; shape, finiteness, non-negativity, symmetry, and zero-
diagonal checks still run before publication.

## Performance summary

| Operation | Budget | Measured |
|-----------|--------|----------|
| `CouplingBuilder.build(100)` | < 10 ms | See current construction measurements above |
| `build_scpn_physics()` | < 5 ms | See current construction measurements above |
| `estimate_from_distances(64)` | < 5 ms | ~0.5 ms |
| `load_hcp_connectome` | [Current cold/warm public calls](coupling_connectome.md#current-measurements) | Original checked native generation, measured separately |
| `validate_knm(64)` | Profile-dependent | Not timed separately in the current projection snapshot |
| `graph_laplacian(64)` | < 1 ms | ~0.007 ms |
| `fiedler_value(64)` | < 1 ms | ~0.12 ms |

## Spatial coupling modulation

`SpatialCouplingModulator` is the public PHA-C.1 coupling surface for systems where the effective phase coupling must depend on moving geometry instead of static oscillator labels. It turns a zero-diagonal base `K_nm` matrix and a position matrix into a physically constrained modulated coupling matrix.

Use it when spatial proximity, mobile agents, tissue geometry, sensor placement, or edge-node distance changes the strength of phase transfer. The default kernel is `1 / (1 + distance)`, which is bounded, finite at zero separation, symmetric for Euclidean positions, and preserves the zero self-coupling diagonal required by the oscillator engines.

The module also exposes exponential, power-law, and inverse-distance kernels. The inverse-distance form is reserved for Swarmalator compatibility and uses an epsilon-regularised denominator so the historical kernel remains bit-true without introducing singularities.

The reference implementation is NumPy. Rust, Go, Julia, and Mojo adapters are validated as optional accelerators and must reproduce the same invariants before their output is accepted: finite real-valued matrices, exact shape or flat cardinality, non-boolean and non-complex values, non-negative entries, zero diagonal, and symmetry preservation for symmetric inputs. Public positions, base coupling matrices, scalar decay controls, direct accelerator counts/forms/flat buffers, optional backend outputs, and raw Julia returns reject numeric-string aliases before float coercion. The public dispatcher preserves matrix-shaped output for callers after replaying the shared direct output validator; optional backend fallback remains limited to loader or runtime unavailability.

See [Coupling - Spatial Modulator](coupling_spatial_modulator.md) for examples, backend notes, and the benchmark contract.

## Measurement source units

Hodge phases, coupling weights and topology indices; spectral weights and
frequencies; spatial coordinates, distances and weights; and inference phase
series require plain real source values before numerical conversion. Text,
boolean, complex and temporal aliases are rejected. Real numeric object arrays
remain compatible. Metadata counts and form codes require plain non-boolean integers. The direct Go/Julia/Mojo bridges share the same array source checks.
The direct Rust spatial boundary also rejects aliases in vectors, controls,
counts and form codes before extracting native values.

[Measured coupling parity records](../data/coupling_measurement_types_benchmark_2026-09-26.json)
record all five available Python/Rust/Go/Julia/Mojo implementations for each
of Hodge (N=6), spectral (N=10) and spatial (N=10, d=2). Each gate used three
calls and passed its declared physics/parity tolerances. These small-fixture
shared-host timings do not establish production-scale performance.

## Original numerical input types

See [numerical source types](numerical_source_types.md) for text, boolean and
temporal refusal, numeric-object compatibility and current language measurements.
