# Hypergraph (k-Body) Kuramoto Engine

## 1. Mathematical Formalism

### Generalised k-Body Coupling

The hypergraph Kuramoto model extends pairwise coupling to arbitrary
higher-order interactions. For a $k$-hyperedge $e = \{i_1, \ldots, i_k\}$
with coupling strength $\sigma_e$, the contribution to oscillator $i_m$
in the hyperedge is:

$$\dot{\theta}_{i_m}\big|_e = \sigma_e \cdot \sin\!\left(\sum_{\substack{j \in e \\ j \neq m}} \theta_{i_j} - (k-1)\,\theta_{i_m}\right)$$

Using the total phase sum $\Phi_e = \sum_{j \in e} \theta_{i_j}$, the
argument simplifies to:

$$\dot{\theta}_{i_m}\big|_e = \sigma_e \cdot \sin(\Phi_e - k\,\theta_{i_m})$$

### Full Dynamics

The complete equation of motion for oscillator $i$ combines natural
frequency, optional pairwise coupling, all hyperedge contributions,
and external drive:

$$\frac{d\theta_i}{dt} = \omega_i + \underbrace{\sum_{j=1}^{N} K_{ij} \sin(\theta_j - \theta_i - \alpha_{ij})}_{\text{pairwise (optional)}} + \underbrace{\sum_{e \ni i} \sigma_e \sin(\Phi_e - |e|\,\theta_i)}_{\text{hypergraph k-body}} + \underbrace{\zeta \sin(\Psi - \theta_i)}_{\text{external drive}}$$

### Special Cases

| Order $k$ | Argument | Model |
|-----------|----------|-------|
| $k = 2$ | $\sin(\theta_j - \theta_i)$ | Standard Kuramoto |
| $k = 3$ | $\sin(\theta_j + \theta_k - 2\theta_i)$ | Simplicial (Tanaka & Aoyagi 2011) |
| $k = 4$ | $\sin(\theta_j + \theta_k + \theta_l - 3\theta_i)$ | 4-body (Bick et al. 2023) |
| Mixed | Combination of above | This engine |

The `HypergraphEngine` supports **mixed-order** interactions: some edges
can be pairwise, some 3-body, some 4-body, in the same system.

### Integration Method

Euler forward integration:

$$\theta_i(t + \Delta t) = \left(\theta_i(t) + \Delta t \cdot \dot{\theta}_i(t)\right) \bmod 2\pi$$

The `run()` method executes the full loop in Rust via `spo_kernel` when
available, with flat-encoded hyperedges for zero-copy FFI transfer.

---

## 2. Theoretical Context

### Why Hypergraph Coupling?

Real complex systems exhibit interactions that cannot be reduced to
pairwise connections:

- **Neural ensembles:** Synaptic transmission involves presynaptic
  calcium dynamics, vesicle release probability, and postsynaptic
  receptor occupancy — a fundamentally multi-party process
  (Petri et al. 2014).
- **Gene regulatory networks:** Transcription factor complexes require
  $k$ proteins to co-localise, creating $k$-body logical gates
  (Battiston et al. 2020).
- **Social contagion:** Adoption of behaviours requires reinforcement
  from multiple peers simultaneously (threshold models), not just
  pairwise influence (Iacopini et al. 2019).
- **Ecological webs:** Pollinator-plant-herbivore triangles exhibit
  dynamics absent from pairwise predator-prey models.

### Theoretical Predictions

Tanaka & Aoyagi (2011) showed that $k$-body coupling with $k \geq 3$
produces **multistable attractors** — multiple distinct synchronised
states coexist at the same coupling strength. This is qualitatively
different from pairwise Kuramoto, which has at most one stable
synchronised state for given parameters.

Skardal & Arenas (2019) proved that higher-order coupling induces
**explosive (first-order) synchronisation transitions**, where the
order parameter $R$ jumps discontinuously. The critical coupling
scales as:

$$K_c^{(k)} \propto \frac{\Delta}{\langle R^{k-2} \rangle}$$

where $\Delta$ is the frequency spread and $\langle R^{k-2} \rangle$
is the $(k-2)$-th moment of the order parameter distribution.

### Topology and Algebraic Structure

A **hypergraph** $H = (V, E)$ consists of a vertex set $V$ and a
collection of hyperedges $E$ where each $e \in E$ is a subset
$e \subseteq V$ with $|e| \geq 2$. Unlike simplicial complexes,
hypergraphs do **not** require the downward closure property: a
3-body edge $\{a, b, c\}$ can exist without the pairwise edges
$\{a,b\}$, $\{a,c\}$, $\{b,c\}$.

The **adjacency tensor** generalises the adjacency matrix:

$$A^{(k)}_{i_1 \ldots i_k} = \begin{cases} \sigma_e & \text{if } \{i_1, \ldots, i_k\} = e \in E \\ 0 & \text{otherwise} \end{cases}$$

For $k = 2$, this reduces to the standard adjacency matrix $A_{ij}$.

### Ott-Antonsen Reduction for Hypergraphs

For all-to-all $k$-body coupling with Lorentzian frequency distribution,
the Ott-Antonsen ansatz yields a reduced equation for the mean-field
$z = R e^{i\psi}$ (Skardal & Arenas 2020):

$$\dot{z} = -(\Delta + i\omega_0) z + \frac{\sigma_k}{2} \left(z^{k-1} \bar{z}^{k-2} - |z|^{2(k-1)} z\right)$$

This generalises the standard OA equation ($k = 2$) and explains why
higher $k$ produces sharper transitions: the effective nonlinearity
increases with $k$.

### Applications in Neuroscience

Recent work by Petri et al. (2014) and Reimann et al. (2017) showed that
brain functional networks exhibit significant higher-order structure
(simplicial complexes of dimension up to 7 in the Blue Brain connectome).
The hypergraph engine enables simulation of these structures with
explicit control over which groups of neurons participate in collective
interactions.

Key parameters for neural hypergraphs:
- **Order distribution:** Which $k$ values are present (typically $k = 2, 3, 4$)
- **Strength hierarchy:** $\sigma_3 < \sigma_2$ (higher-order weaker)
- **Spatial constraint:** Hyperedges limited to local circuit motifs

### Comparison with SimplicialEngine

`SimplicialEngine` is a specialised, optimised version for the $k=3$ case.
It uses the trig factorisation $2 S_i C_i$ to avoid looping over edges.
`HypergraphEngine` is general: it handles any mix of orders but loops
over each hyperedge explicitly. Use `SimplicialEngine` when all higher-order
interactions are exactly 3-body for better performance.

| Feature | SimplicialEngine | HypergraphEngine |
|---------|-----------------|------------------|
| Orders supported | 2 + 3 | Any $k \geq 2$ |
| Edge representation | Implicit (all-to-all 3-body) | Explicit edge list |
| 3-body complexity | $O(N^2)$ via factorisation | $O(|E| \cdot k)$ per step |
| Mixed orders | No | Yes |
| Pairwise via K_nm | Yes | Optional |

---

## 3. Pipeline Position

```
Oscillators.extract() ──→ θ, ω
                               │
CouplingBuilder.build() ──→ K_nm, α (optional pairwise)
                               │
Hyperedge definitions ──→ [(nodes, σ), ...]
                               │
                               ↓
     ┌──── HypergraphEngine(n, dt, hyperedges) ────┐
     │                                              │
     │  Input:  θ, ω, [optional K_nm, α, ζ, Ψ]    │
     │  Edges:  list of Hyperedge(nodes, strength)  │
     │  Method: Euler (Rust FFI or Python)          │
     │  Output: θ_new ∈ [0, 2π)^N                  │
     │                                              │
     └──────────────────────────────────────────────┘
                               │
                               ↓
              compute_order_parameter(θ_new) → R, ψ
```

### FFI Encoding

Hyperedges are serialised for Rust FFI as three flat arrays:
- `edge_nodes: int64[]` — concatenated node indices for all edges
- `edge_offsets: int64[]` — start index in `edge_nodes` for each edge
- `edge_strengths: float64[]` — coupling strength per edge

This avoids Python object overhead and enables zero-copy transfer.

### Input Contracts

| Parameter | Type | Shape | Source |
|-----------|------|-------|--------|
| `phases` | `NDArray[float64]` | `(N,)` | Previous step |
| `omegas` | `NDArray[float64]` | `(N,)` | Oscillators |
| `pairwise_knm` | `NDArray[float64] \| None` | `(N, N)` | CouplingBuilder (optional) |
| `alpha` | `NDArray[float64] \| None` | `(N, N)` | Phase-lag (optional) |
| `zeta` | `float` | scalar | External drive |
| `psi` | `float` | scalar | Drive phase |

---

## 4. Features

- **Arbitrary $k$-body coupling** — any order $k \geq 2$
- **Mixed-order edges** — combine pairwise, 3-body, 4-body in one system
- **Dynamic edge addition** — `add_edge()` and `add_all_to_all(order)`
- **Optional pairwise coupling** via standard $K_{nm}$ matrix
- **External drive** ($\zeta$, $\Psi$) for entrainment
- **Full Rust FFI acceleration** with flat-encoded edge transfer
- **Order parameter computation** — built-in `order_parameter()` method
- **step() and run()** — single-step or batch integration

---

## 5. Usage Examples

### Basic: 3-Body All-to-All

```python
import numpy as np
from scpn_phase_orchestrator.upde.hypergraph import HypergraphEngine, Hyperedge
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

N = 16
engine = HypergraphEngine(N, dt=0.01)
engine.add_all_to_all(order=3, strength=0.5)

print(f"Number of 3-body edges: {engine.n_edges}")  # C(16,3) = 560

phases = np.random.default_rng(42).uniform(0, 2 * np.pi, N)
omegas = np.ones(N)

phases_final = engine.run(phases, omegas, n_steps=1000)
R, _ = compute_order_parameter(phases_final)
print(f"R = {R:.4f}")
```

### Mixed-Order: Pairwise + 3-Body + 4-Body

```python
import numpy as np
from scpn_phase_orchestrator.upde.hypergraph import HypergraphEngine

N = 8
engine = HypergraphEngine(N, dt=0.01)

# Pairwise edges (specific pairs, not K_nm matrix)
engine.add_edge((0, 1), strength=1.0)
engine.add_edge((2, 3), strength=1.0)
engine.add_edge((4, 5), strength=1.0)

# 3-body triangles
engine.add_edge((0, 1, 2), strength=0.5)
engine.add_edge((3, 4, 5), strength=0.5)

# 4-body interaction linking both groups
engine.add_edge((1, 2, 4, 5), strength=0.3)

phases = np.zeros(N)
phases[4:] = np.pi  # Two anti-phase clusters
omegas = np.ones(N)

phases_final = engine.run(phases, omegas, n_steps=2000)
R = engine.order_parameter(phases_final)
print(f"R = {R:.4f}")
```

### Combined with Pairwise K_nm Matrix

```python
import numpy as np
from scpn_phase_orchestrator.upde.hypergraph import HypergraphEngine
from scpn_phase_orchestrator.coupling.knm import CouplingBuilder

N = 12
builder = CouplingBuilder(N, method="exponential", K_base=0.5, decay=0.3)
knm, alpha = builder.build()

engine = HypergraphEngine(N, dt=0.01)
engine.add_all_to_all(order=3, strength=0.2)

phases = np.random.default_rng(42).uniform(0, 2 * np.pi, N)
omegas = np.linspace(0.8, 1.2, N)

# Pairwise via K_nm + 3-body via hyperedges
phases_final = engine.run(
    phases, omegas, n_steps=1000,
    pairwise_knm=knm, alpha=alpha,
)
```

### Measuring Phase Transitions

```python
import numpy as np
from scpn_phase_orchestrator.upde.hypergraph import HypergraphEngine

N = 20
omegas = np.random.default_rng(42).standard_cauchy(N) * 0.05
phases_init = np.random.default_rng(42).uniform(0, 2 * np.pi, N)

sigma_values = np.linspace(0, 2, 40)
R_values = []

for sigma in sigma_values:
    eng = HypergraphEngine(N, dt=0.01)
    eng.add_all_to_all(order=3, strength=sigma)
    p = eng.run(phases_init.copy(), omegas, n_steps=2000)
    R_values.append(eng.order_parameter(p))

# Find critical coupling
for i, (s, r) in enumerate(zip(sigma_values, R_values)):
    if r > 0.5:
        print(f"σ_c ≈ {s:.2f} (R = {r:.3f})")
        break
```

---

## 6. Technical Reference

### Class: HypergraphEngine

::: scpn_phase_orchestrator.upde.hypergraph

### Dataclass: Hyperedge

```python
@dataclass
class Hyperedge:
    nodes: tuple[int, ...]   # oscillator indices in this hyperedge
    strength: float = 1.0    # coupling strength σ_e

    @property
    def order(self) -> int:  # k = number of nodes
        return len(self.nodes)
```

### Rust Engine Function

```rust
pub fn hypergraph_run(
    phases: &[f64],          // length N
    omegas: &[f64],          // length N
    n: usize,
    edges: &[Hyperedge],     // reconstructed from flat encoding
    pairwise_knm: &[f64],   // length N*N or empty
    alpha: &[f64],           // length N*N or empty
    zeta: f64,
    psi: f64,
    dt: f64,
    n_steps: usize,
) -> Vec<f64>                // length N
```

### FFI Flat Encoding

The Python wrapper serialises hyperedges as:

```python
edge_nodes = np.array([0, 1, 2, 1, 2, 3], dtype=np.int64)  # two 3-body edges
edge_offsets = np.array([0, 3], dtype=np.int64)               # starts at 0 and 3
edge_strengths = np.array([0.5, 0.5], dtype=np.float64)
```

Rust reconstructs `Hyperedge` structs from these arrays with zero allocation
overhead per edge beyond the initial `Vec::with_capacity`.

### Auto-Select Logic

Rust path is used only when `_HAS_RUST is True` **and** there are hyperedges
(`self._hyperedges` is non-empty). If only pairwise coupling is used (no
hyperedges), the Python loop is trivially fast and the Rust overhead is
not warranted.

---

## 7. Performance Benchmarks

Measured on Intel Core i5-11600K @ 3.90 GHz, 32 GB DDR4-2400.
200 Euler steps, all-to-all 3-body edges ($C(N,3)$ hyperedges),
$\sigma = 0.5$. Averaged over 3-20 iterations.

| N | Edges $C(N,3)$ | Python (ms) | Rust (ms) | Speedup |
|---|----------------|-------------|-----------|---------|
| 8 | 56 | 52.85 | 0.52 | **101.7x** |
| 16 | 560 | 652.20 | 7.92 | **82.4x** |
| 24 | 2,024 | 2,152.21 | 30.75 | **70.0x** |

### Scaling Analysis

The computation is $O(|E| \cdot k \cdot T)$ where $|E|$ is the number of
hyperedges, $k$ is the average order, and $T$ is the number of steps.
For all-to-all 3-body edges, $|E| = C(N,3) = O(N^3/6)$, making the
total cost $O(N^3 \cdot T)$.

The Rust speedup is dramatic (70-100x) because:
1. Python per-edge loop overhead is eliminated
2. No Python object allocation per edge per step
3. Cache-friendly flat array access in Rust

For large $N$, consider using `SimplicialEngine` instead (all-to-all
3-body via trig factorisation at $O(N^2)$, not $O(N^3)$).

### Memory Usage

Rust: $O(N + |E| \cdot k)$ for edge storage + $O(N)$ working arrays.
Python: Same edge storage + NumPy temporaries for pairwise terms.

### Test Coverage

- **Rust tests:** 9 (hypergraph module in spo-engine)
  - Free rotation, pairwise-only synchronisation, 3-body effect,
    4-body, mixed-order, external drive, zero steps, order parameter
    synchronised, order parameter uniform
- **Python tests:** 11 (`tests/test_hypergraph.py`)
- **Source lines:** 242 (Rust) + 176 (Python) = 418 total

---

## 8. Citations

1. **Tanaka, T. & Aoyagi, T.** (2011).
   "Multistable attractors in a network of phase oscillators with
   three-body interactions."
   *Physical Review Letters* 106:224101.
   DOI: [10.1103/PhysRevLett.106.224101](https://doi.org/10.1103/PhysRevLett.106.224101)

2. **Skardal, P. S. & Arenas, A.** (2019).
   "Abrupt desynchronization and extensive multistability in globally
   coupled oscillator simplexes."
   *Physical Review Letters* 122:248301.
   DOI: [10.1103/PhysRevLett.122.248301](https://doi.org/10.1103/PhysRevLett.122.248301)

3. **Bick, C., Gross, E., Harrington, H. A., & Schaub, M. T.** (2023).
   "What are higher-order networks?"
   *Nature Reviews Physics* 5:307-317.
   DOI: [10.1038/s42254-023-00573-y](https://doi.org/10.1038/s42254-023-00573-y)

4. **Battiston, F., Cencetti, G., Iacopini, I., et al.** (2020).
   "Networks beyond pairwise interactions: Structure and dynamics."
   *Physics Reports* 874:1-92.
   DOI: [10.1016/j.physrep.2020.05.004](https://doi.org/10.1016/j.physrep.2020.05.004)

5. **Iacopini, I., Petri, G., Barrat, A., & Latora, V.** (2019).
   "Simplicial models of social contagion."
   *Nature Communications* 10:2485.
   DOI: [10.1038/s41467-019-10431-6](https://doi.org/10.1038/s41467-019-10431-6)

6. **Petri, G., Expert, P., Turkheimer, F., et al.** (2014).
   "Homological scaffolds of brain functional networks."
   *Journal of the Royal Society Interface* 11:20140873.
   DOI: [10.1098/rsif.2014.0873](https://doi.org/10.1098/rsif.2014.0873)

7. **Gambuzza, L. V., Di Patti, F., Gallo, L., et al.** (2021).
   "Stability of synchronization in simplicial complexes."
   *Nature Communications* 12:1255.
   DOI: [10.1038/s41467-021-21486-9](https://doi.org/10.1038/s41467-021-21486-9)

8. **Reimann, M. W., Nolte, M., Scolamiero, M., et al.** (2017).
   "Cliques of neurons bound into cavities provide a missing link
   between structure and function."
   *Frontiers in Computational Neuroscience* 11:48.
   DOI: [10.3389/fncom.2017.00048](https://doi.org/10.3389/fncom.2017.00048)

9. **Millán, A. P., Torres, J. J., & Bianconi, G.** (2020).
   "Explosive higher-order Kuramoto dynamics on simplicial complexes."
   *Physical Review Letters* 124:218301.
   DOI: [10.1103/PhysRevLett.124.218301](https://doi.org/10.1103/PhysRevLett.124.218301)

9. **Acebrón, J. A., Bonilla, L. L., Pérez Vicente, C. J., Ritort, F.,
   & Spigler, R.** (2005).
   "The Kuramoto model: A simple paradigm for synchronization phenomena."
   *Reviews of Modern Physics* 77:137-185.
   DOI: [10.1103/RevModPhys.77.137](https://doi.org/10.1103/RevModPhys.77.137)

---

## Numerical Considerations

### Euler vs Higher-Order Methods

The `HypergraphEngine` uses Euler integration. For stiff systems (large
$\sigma$ or mixed orders), the time step must satisfy:

$$\Delta t < \frac{1}{\max_i |\dot{\theta}_i|} \approx \frac{1}{\max(\omega) + \sigma_{max} \cdot N}$$

For $N = 50$, $\sigma = 1$: $\Delta t < 0.02$ is typically safe.
For higher accuracy, reduce $\Delta t$ or use `SplittingEngine` for the
pairwise component combined with explicit hyperedge steps.

### Floating-Point Precision

The phase sum $\Phi_e = \sum_{j \in e} \theta_j$ for large $k$ can
accumulate rounding errors. For $k > 10$, the argument to $\sin$ may
lose 1-2 digits of precision. This is rarely an issue in practice since
most physical hyperedges have $k \leq 5$.

### Determinism

Both Python and Rust paths are deterministic given the same inputs.
The Rust path uses the same mathematical operations in the same order,
ensuring bitwise-identical results across runs.

---

## Edge Cases and Limitations

### Empty Hyperedge List

With no hyperedges added, `run()` uses the Python loop (Rust path
requires at least one hyperedge). If pairwise `knm` is provided,
the engine acts as a standard Kuramoto integrator (but without RK4 —
use `UPDEEngine` for higher-order integration).

### Single-Node Hyperedges

A hyperedge with $k = 1$ is mathematically trivial: the argument
$\Phi_e - k\theta_m = \theta_m - \theta_m = 0$, so $\sin(0) = 0$.
No contribution. The engine does not filter these, but they have
zero effect.

### Very Large $|E|$

For $N = 100$ with all-to-all 3-body: $C(100, 3) = 161{,}700$ edges.
Each step requires one $\sin$ evaluation per node per edge, giving
$\sim 16$ million trig calls per step. At $N > 50$, consider whether
`SimplicialEngine` (all-to-all implicit at $O(N^2)$) is more appropriate.

### Duplicate Edges

Adding the same edge twice doubles its effective coupling strength.
The engine does not deduplicate — this is by design, as different
edges between the same nodes may represent distinct physical interactions.

---

## Appendix: Relationship to Other Engines

| If you need... | Use... |
|----------------|--------|
| Only pairwise | `UPDEEngine` |
| Only 3-body (all-to-all) | `SimplicialEngine` (faster) |
| Any $k$-body (explicit edges) | `HypergraphEngine` (this) |
| Mixed pairwise + 3-body | Either (Simplicial is faster) |
| Sparse edge sets | `HypergraphEngine` |
| Dense all-to-all 3-body | `SimplicialEngine` ($O(N^2)$ vs $O(N^3)$) |

---

## Appendix B: Choosing the Right Engine

Decision tree for higher-order Kuramoto integration:

```
Is the coupling purely pairwise?
  ├─ Yes → UPDEEngine (RK4/RK45, fastest for pairwise)
  └─ No → Are ALL higher-order interactions exactly 3-body?
           ├─ Yes → Is it all-to-all (every triplet)?
           │         ├─ Yes → SimplicialEngine (O(N²) factorisation)
           │         └─ No  → HypergraphEngine (explicit edge list)
           └─ No → HypergraphEngine (mixed k-body, this module)
```

For systems with $N > 100$ and dense 3-body coupling, always prefer
`SimplicialEngine` — the $O(N^2)$ vs $O(N^3)$ difference is
significant (e.g. $N=100$: 10,000 vs 166,650 operations per step).

---

## Appendix C: Complexity Summary

| Component | Python | Rust | Notes |
|-----------|--------|------|-------|
| Pairwise coupling | $O(N^2 \cdot T)$ | $O(N^2 \cdot T)$ | Same as UPDEEngine |
| Per-edge coupling | $O(|E| \cdot k \cdot T)$ | $O(|E| \cdot k \cdot T)$ | Rust eliminates Python loop overhead |
| All-to-all $k=3$ | $O(N^3 \cdot T / 6)$ | $O(N^3 \cdot T / 6)$ | Use SimplicialEngine for $O(N^2)$ |
| Edge encoding (FFI) | $O(|E| \cdot k)$ | — | One-time per `run()` call |
| Memory (edges) | $O(|E| \cdot k)$ | $O(|E| \cdot k)$ | Flat arrays |
| Memory (state) | $O(N)$ | $O(N)$ | Phase vector |
