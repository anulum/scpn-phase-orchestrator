# Simplicial (Higher-Order) Kuramoto Engine

## 1. Mathematical Formalism

### Standard Kuramoto with Pairwise + 3-Body Coupling

The simplicial Kuramoto model extends the classical Kuramoto model with
higher-order (3-body) interactions on simplicial complexes. The full
dynamics for oscillator $i$ are:

$$\frac{d\theta_i}{dt} = \omega_i + \underbrace{\sum_{j=1}^{N} K_{ij} \sin(\theta_j - \theta_i - \alpha_{ij})}_{\text{pairwise coupling}} + \underbrace{\frac{\sigma_2}{N^2} \sum_{j,k} \sin(\theta_j + \theta_k - 2\theta_i)}_{\text{3-body simplicial term}} + \underbrace{\zeta \sin(\Psi - \theta_i)}_{\text{external drive}}$$

where:
- $\theta_i \in [0, 2\pi)$ — phase of oscillator $i$
- $\omega_i$ — natural frequency
- $K_{ij}$ — pairwise coupling matrix (from `CouplingBuilder`)
- $\alpha_{ij}$ — phase-lag matrix
- $\sigma_2$ — 3-body coupling strength
- $\zeta$ — external drive amplitude
- $\Psi$ — external drive phase

### 3-Body Term: Trigonometric Identity

The naive triple sum $\sum_{j,k} \sin(\theta_j + \theta_k - 2\theta_i)$
has $O(N^3)$ complexity per oscillator. Using the product-to-sum identity:

$$\sin((\theta_j - \theta_i) + (\theta_k - \theta_i)) = \sin(d_j)\cos(d_k) + \cos(d_j)\sin(d_k)$$

where $d_j = \theta_j - \theta_i$, the full sum factorises:

$$\sum_{j,k} \sin(d_j + d_k) = \left(\sum_j \sin(d_j)\right)\left(\sum_k \cos(d_k)\right) + \left(\sum_j \cos(d_j)\right)\left(\sum_k \sin(d_k)\right) = 2 S_i C_i$$

where $S_i = \sum_j \sin(\theta_j - \theta_i)$ and $C_i = \sum_j \cos(\theta_j - \theta_i)$.

This reduces the 3-body computation from $O(N^3)$ to $O(N)$ per oscillator,
giving $O(N^2)$ total (same as pairwise). The Rust implementation uses this
identity exclusively.

### Integration Method

Euler forward integration with modular arithmetic:

$$\theta_i(t + \Delta t) = \left(\theta_i(t) + \Delta t \cdot \dot{\theta}_i(t)\right) \bmod 2\pi$$

The `run()` method executes `n_steps` Euler steps entirely in Rust when
`spo_kernel` is available.

---

## 2. Theoretical Context

### Why Higher-Order Interactions?

Classical Kuramoto coupling ($\sin(\theta_j - \theta_i)$) models pairwise
interactions: one oscillator influences another. But many real systems
exhibit **group interactions** where the collective state of multiple
agents matters:

- **Neural circuits:** Synaptic triads (feedforward inhibition) involve
  3-neuron motifs where the effect of neuron A on neuron C depends on
  the state of neuron B (Sporns & Kotter 2004).
- **Social dynamics:** Opinion formation depends on group consensus, not
  just pairwise influence (Petri et al. 2014).
- **Ecological networks:** Predator-prey-resource triangles exhibit
  emergent dynamics absent from pairwise analysis (Grilli et al. 2017).

### Explosive Synchronisation

The 3-body term qualitatively changes the synchronisation transition.
With pairwise coupling alone ($\sigma_2 = 0$), the Kuramoto model exhibits
a **continuous (second-order) transition** at critical coupling $K_c$.
Adding $\sigma_2 > 0$ creates a **discontinuous (first-order, explosive)
transition** where the order parameter $R$ jumps abruptly from near-zero
to near-one (Skardal & Arenas 2019).

This explosive transition has profound implications:
- Basins of attraction shrink dramatically (Menck et al. 2013)
- Hysteresis: the desynchronisation threshold $K_c^{down} < K_c^{up}$
- Metastable chimera states become more prevalent

### Simplicial Complexes vs Hypergraphs

The simplicial model restricts to **3-body** interactions (2-simplices).
For arbitrary $k$-body coupling, see `HypergraphEngine`. The simplicial
model is the most studied special case with the richest analytical results
(Gambuzza et al. 2021, 2023).

### Mathematical Properties

**Critical coupling.** For the purely simplicial model ($K_{ij} = 0$,
$\sigma_2 > 0$) with Lorentzian frequency distribution $g(\omega)$ of
half-width $\Delta$, the critical coupling for onset of synchronisation is:

$$\sigma_{2,c} = \frac{2\Delta}{\pi g(\omega_0) |z|^2}$$

where $z$ is the Ott-Antonsen order parameter (Skardal & Arenas 2020).

**Hysteresis width.** The first-order transition exhibits hysteresis with
width proportional to $\sigma_2$. The forward transition (increasing K)
occurs at $K_c^{up}$, while the backward transition (decreasing K) occurs
at $K_c^{down} < K_c^{up}$. The hysteresis region $[K_c^{down}, K_c^{up}]$
harbours bistability between synchronised and incoherent states.

**Basin shrinkage.** Menck et al. (2013) showed that higher-order coupling
reduces the basin of stability of the synchronised state, despite improving
its linear stability. This paradox — more stable but harder to reach —
is a hallmark of explosive synchronisation.

### Historical Development

- **Kuramoto (1975):** Original pairwise model on globally coupled oscillators
- **Tanaka & Aoyagi (2011):** First rigorous 3-body extension with multistability
- **Skardal & Arenas (2019):** Explosive transition proof for simplicial coupling
- **Gambuzza et al. (2021):** Master stability function for simplicial complexes
- **Bick et al. (2023):** Comprehensive review of higher-order network dynamics

---

## 3. Pipeline Position

```
Oscillators.extract() ──→ θ, ω
                               │
CouplingBuilder.build() ──→ K_nm, α
                               │
                               ↓
     ┌──── SimplicialEngine(n, dt, σ₂) ────┐
     │                                      │
     │  Input:  θ, ω, K_nm, ζ, Ψ, α       │
     │  Params: σ₂ (3-body strength)        │
     │  Method: Euler (Rust FFI or Python)  │
     │  Output: θ_new ∈ [0, 2π)^N          │
     │                                      │
     └──────────────────────────────────────┘
                               │
                               ↓
              compute_order_parameter(θ_new) → R, ψ
                               │
                               ↓
              RegimeManager.evaluate() → Regime
```

### Input Contracts

| Parameter | Type | Shape | Units | Source |
|-----------|------|-------|-------|--------|
| `phases` | `NDArray[float64]` | `(N,)` | radians | Previous step or `Oscillators.extract()` |
| `omegas` | `NDArray[float64]` | `(N,)` | rad/s | `Oscillators.extract()` |
| `knm` | `NDArray[float64]` | `(N, N)` | dimensionless | `CouplingBuilder.build()` |
| `alpha` | `NDArray[float64]` | `(N, N)` | radians | `CouplingBuilder` or zeros |
| `zeta` | `float` | scalar | rad/s | `Drivers.compute()` |
| `psi` | `float` | scalar | radians | `Drivers.compute()` |

### Output Contract

| Output | Type | Shape | Units |
|--------|------|-------|-------|
| `phases_new` | `NDArray[float64]` | `(N,)` | radians in $[0, 2\pi)$ |

---

## 4. Features

- **Pairwise + 3-body coupling** in a single integration step
- **Adjustable σ₂** at runtime via `sigma2` property setter
- **Full Rust FFI acceleration** for the `run()` method (entire loop in Rust)
- **Phase-lag matrix α** support for non-zero frustration
- **External drive** (ζ, Ψ) for entrainment studies
- **Modular arithmetic** — phases always in $[0, 2\pi)$
- **Vectorised 3-body term** via trig identity ($O(N^2)$ not $O(N^3)$)
- **Automatic backend selection** — uses Rust when `spo_kernel` is installed,
  falls back to numpy transparently
- **step() and run()** — single-step for fine control, multi-step for batch

---

## 5. Usage Examples

### Basic: 3-Body Explosive Sync

```python
import numpy as np
from scpn_phase_orchestrator.upde.simplicial import SimplicialEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

N = 32
dt = 0.01
sigma2 = 1.5  # 3-body coupling strength

engine = SimplicialEngine(N, dt=dt, sigma2=sigma2)

# Initial conditions: random phases, identical frequencies
rng = np.random.default_rng(42)
phases = rng.uniform(0, 2 * np.pi, N)
omegas = np.ones(N)

# Pairwise coupling: all-to-all with K=0.3
knm = np.full((N, N), 0.3)
np.fill_diagonal(knm, 0.0)
alpha = np.zeros((N, N))

# Run 2000 steps
phases_final = engine.run(phases, omegas, knm, 0.0, 0.0, alpha, 2000)

R, psi = compute_order_parameter(phases_final)
print(f"R = {R:.4f}")  # Expect R > 0.9 with σ₂=1.5
```

### Comparing Pairwise vs Simplicial Transitions

```python
import numpy as np
from scpn_phase_orchestrator.upde.simplicial import SimplicialEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

N = 64
rng = np.random.default_rng(42)
omegas = rng.standard_cauchy(N) * 0.1  # Lorentzian spread
alpha = np.zeros((N, N))
phases_init = rng.uniform(0, 2 * np.pi, N)

K_values = np.linspace(0, 2, 50)

for label, sigma2 in [("pairwise only", 0.0), ("simplicial σ₂=1", 1.0)]:
    R_curve = []
    for K in K_values:
        knm = np.full((N, N), K / N)
        np.fill_diagonal(knm, 0.0)
        eng = SimplicialEngine(N, dt=0.01, sigma2=sigma2)
        p = eng.run(phases_init.copy(), omegas, knm, 0.0, 0.0, alpha, 3000)
        R, _ = compute_order_parameter(p)
        R_curve.append(R)
    print(f"{label}: K_c ≈ {K_values[next(i for i, r in enumerate(R_curve) if r > 0.5)]:.2f}")
```

### Runtime σ₂ Adjustment

```python
engine = SimplicialEngine(16, dt=0.01, sigma2=0.0)
print(engine.sigma2)  # 0.0

# Enable 3-body coupling mid-simulation
engine.sigma2 = 2.0
print(engine.sigma2)  # 2.0
```

### Integration with SSGF Geometry Control

```python
from scpn_phase_orchestrator.upde.simplicial import SimplicialEngine
from scpn_phase_orchestrator.ssgf.costs import compute_ssgf_costs
from scpn_phase_orchestrator.ssgf.carrier import GeometryCarrier
import numpy as np

N = 16
carrier = GeometryCarrier(N, z_dim=6, seed=42)
W = carrier.decode()
engine = SimplicialEngine(N, dt=0.01, sigma2=1.0)

phases = np.random.default_rng(42).uniform(0, 2 * np.pi, N)
omegas = np.ones(N)
alpha = np.zeros((N, N))

# SSGF outer loop: geometry → dynamics → cost → update
for step in range(20):
    phases = engine.run(phases, omegas, W, 0.0, 0.0, alpha, 100)
    costs = compute_ssgf_costs(W, phases)

    def cost_fn(W_trial):
        return compute_ssgf_costs(W_trial, phases).u_total

    state = carrier.update(cost=costs.u_total, cost_fn=cost_fn)
    W = carrier.decode()
    print(f"Step {step}: U_total={costs.u_total:.4f}")
```

### Single-Step Fine Control

```python
import numpy as np
from scpn_phase_orchestrator.upde.simplicial import SimplicialEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

N = 8
engine = SimplicialEngine(N, dt=0.005, sigma2=0.5)
phases = np.random.default_rng(0).uniform(0, 2 * np.pi, N)
omegas = np.ones(N) * 2.0
knm = np.full((N, N), 1.0); np.fill_diagonal(knm, 0.0)
alpha = np.zeros((N, N))

# Step-by-step with monitoring
R_history = []
for t in range(500):
    phases = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)
    R, _ = compute_order_parameter(phases)
    R_history.append(R)

print(f"Final R = {R_history[-1]:.4f}")
print(f"Convergence time ≈ {next(i for i, r in enumerate(R_history) if r > 0.9)} steps")
```

---

## 6. Technical Reference

### Class: SimplicialEngine

::: scpn_phase_orchestrator.upde.simplicial

### Rust Engine Function

The Rust implementation `spo_engine::simplicial::simplicial_run` accepts
flat arrays and performs the entire integration loop without Python
callbacks:

```rust
pub fn simplicial_run(
    phases: &[f64],     // length N
    omegas: &[f64],     // length N
    knm: &[f64],        // length N*N, row-major
    alpha: &[f64],      // length N*N, row-major
    zeta: f64,
    psi: f64,
    sigma2: f64,
    dt: f64,
    n_steps: usize,
) -> Vec<f64>           // length N, final phases
```

### FFI Binding

Python calls `spo_kernel.simplicial_run_rust(phases, omegas, knm, alpha, n, zeta, psi, sigma2, dt, n_steps)`.
Arrays must be contiguous `float64`. The binding validates `len(phases) == n`
and `len(knm) == n*n`.

### Auto-Select Logic

```python
# In simplicial.py
try:
    from spo_kernel import simplicial_run_rust as _rust_simplicial_run
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False
```

When `_HAS_RUST is True`, `SimplicialEngine.run()` delegates entirely to
Rust. The `step()` method always uses Python (single-step overhead makes
FFI call unprofitable).

---

## 7. Performance Benchmarks

Measured on Intel Core i5-11600K @ 3.90 GHz, 32 GB DDR4-2400.
500 Euler steps, σ₂ = 1.0, all-to-all coupling K = 0.5.
Averaged over 10 (Rust) or 3-10 (Python) iterations.

| N | Python (ms) | Rust (ms) | Speedup |
|---|-------------|-----------|---------|
| 8 | 89.21 | 0.82 | **108.6x** |
| 32 | 760.18 | 23.03 | **33.0x** |
| 64 | 875.50 | 74.61 | **11.7x** |
| 128 | 2127.44 | 212.90 | **10.0x** |

### Scaling Analysis

Both implementations are $O(N^2 \cdot T)$ where $T$ is the number of steps.
The pairwise coupling dominates ($O(N^2)$ per step); the 3-body term is
$O(N)$ per oscillator (total $O(N^2)$) thanks to the trig identity.

The Rust speedup is largest at small N (108x at N=8) where Python loop
overhead dominates, and stabilises around 10-11x at larger N where the
actual floating-point computation dominates.

### Memory Allocation

The Rust implementation allocates:
- `Vec<f64>` of length $N$ for the working phases (copied from input)
- `Vec<f64>` of length $N$ for the derivative (per step, stack-reused)
- No heap allocations inside the inner loop

The Python fallback allocates NumPy arrays via broadcasting:
- `(N, N)` for `diff` matrix (pairwise phase differences)
- `(N,)` for `three_body` accumulator
- Multiple temporaries from `np.sin`, `np.sum`

### Numerical Precision

Both Python and Rust use IEEE 754 `float64`. The Euler integrator is
first-order accurate ($O(\Delta t)$). For long simulations where energy
drift matters, use `SplittingEngine` (second-order) or `GeometricEngine`
(symplectic, torus-preserving).

The 3-body factorisation introduces no additional numerical error — it is
an exact algebraic identity, not an approximation.

### Test Coverage

- **Rust tests:** 10 (simplicial module in spo-engine)
  - Identical phases (no drift), zero coupling (free rotation), 3-body effect,
    external drive, synchronisation, phase range, single step, combined
    pairwise + 3-body, zero steps, phase-lag α
- **Python tests:** 7 (`tests/test_simplicial.py`)
  - σ₂=0 reduces to standard Kuramoto, 3-body changes dynamics,
    synchronisation with 3-body, brute-force parity, σ₂ setter, small N,
    run n_steps
- **Source lines:** 295 (Rust) + 130 (Python) = 425 total

---

## 8. Citations

1. **Gambuzza, L. V., Di Patti, F., Gallo, L., et al.** (2021).
   "Stability of synchronization in simplicial complexes."
   *Nature Communications* 12:1255.
   DOI: [10.1038/s41467-021-21486-9](https://doi.org/10.1038/s41467-021-21486-9)

2. **Gambuzza, L. V., Di Patti, F., Gallo, L., et al.** (2023).
   "The master stability function for synchronization in simplicial complexes."
   *Nature Physics* (forthcoming).

3. **Tang, Y., Shi, D., & Lü, L.** (2025).
   "Optimizing higher-order network topology for synchronization."
   *Communications Physics* 5:50.

4. **Skardal, P. S. & Arenas, A.** (2019).
   "Abrupt desynchronization and extensive multistability in globally
   coupled oscillator simplexes."
   *Physical Review Letters* 122:248301.
   DOI: [10.1103/PhysRevLett.122.248301](https://doi.org/10.1103/PhysRevLett.122.248301)

5. **Skardal, P. S. & Arenas, A.** (2020).
   "Higher order interactions in complex networks of phase oscillators
   promote abrupt synchronization switching."
   *Communications Physics* 3:218.

6. **Tanaka, T. & Aoyagi, T.** (2011).
   "Multistable attractors in a network of phase oscillators with
   three-body interactions."
   *Physical Review Letters* 106:224101.
   DOI: [10.1103/PhysRevLett.106.224101](https://doi.org/10.1103/PhysRevLett.106.224101)

7. **Bick, C., Gross, E., Harrington, H. A., & Schaub, M. T.** (2023).
   "What are higher-order networks?"
   *Nature Reviews Physics* 5:307-317.
   DOI: [10.1038/s42254-023-00573-y](https://doi.org/10.1038/s42254-023-00573-y)

8. **Petri, G., Expert, P., Turkheimer, F., et al.** (2014).
   "Homological scaffolds of brain functional networks."
   *Journal of the Royal Society Interface* 11:20140873.
   DOI: [10.1098/rsif.2014.0873](https://doi.org/10.1098/rsif.2014.0873)

9. **Kuramoto, Y.** (1975).
   "Self-entrainment of a population of coupled non-linear oscillators."
   In *International Symposium on Mathematical Problems in Theoretical Physics*,
   Lecture Notes in Physics 39:420-422. Springer.
   DOI: [10.1007/BFb0013365](https://doi.org/10.1007/BFb0013365)

10. **Menck, P. J., Heitzig, J., Marwan, N., & Kurths, J.** (2013).
    "How basin stability complements the linear-stability paradigm."
    *Nature Physics* 9:89-92.
    DOI: [10.1038/nphys2516](https://doi.org/10.1038/nphys2516)

11. **Sporns, O. & Kotter, R.** (2004).
    "Motifs in brain networks."
    *PLoS Biology* 2:e369.
    DOI: [10.1371/journal.pbio.0020369](https://doi.org/10.1371/journal.pbio.0020369)

12. **Acebrón, J. A., Bonilla, L. L., Pérez Vicente, C. J., Ritort, F.,
    & Spigler, R.** (2005).
    "The Kuramoto model: A simple paradigm for synchronization phenomena."
    *Reviews of Modern Physics* 77:137-185.
    DOI: [10.1103/RevModPhys.77.137](https://doi.org/10.1103/RevModPhys.77.137)

---

## Edge Cases and Limitations

### N < 3

When $N < 3$, the 3-body term is identically zero regardless of $\sigma_2$.
The engine reduces to standard pairwise Kuramoto. This is mathematically
correct: a 2-simplex requires at least 3 vertices.

### σ₂ = 0

The engine reduces exactly to the standard Kuramoto model. All 3-body
computation is skipped (the Rust implementation checks `sigma2 != 0.0`
before entering the 3-body loop).

### Large σ₂

Very large $\sigma_2$ values ($> 10$) can cause the 3-body term to
dominate the dynamics, leading to rapid phase clustering and potential
numerical instability with large $\Delta t$. Recommend $\Delta t \leq 0.01$
for $\sigma_2 > 5$.

### Diagonal Coupling

If $K_{ii} \neq 0$, the pairwise self-coupling term $K_{ii} \sin(-\alpha_{ii})$
acts as a frequency shift. The Rust implementation does not zero the diagonal
— this is the caller's responsibility (standard convention: `CouplingBuilder`
always produces zero-diagonal $K_{nm}$).

### Phase Wrapping

The Euler step applies `% 2π` after each step. This is exact for the modular
arithmetic but can cause apparent discontinuities when plotting trajectories.
For smooth visualisation, use `np.unwrap()` on the output.

---

## Appendix: Derivation of the 3-Body Factorisation

Starting from the double sum:

$$T_i = \frac{\sigma_2}{N^2} \sum_{j=1}^{N} \sum_{k=1}^{N} \sin\bigl((\theta_j - \theta_i) + (\theta_k - \theta_i)\bigr)$$

Apply the angle addition formula:

$$\sin(A + B) = \sin A \cos B + \cos A \sin B$$

with $A = \theta_j - \theta_i$, $B = \theta_k - \theta_i$:

$$T_i = \frac{\sigma_2}{N^2} \sum_j \sum_k \bigl[\sin(d_j)\cos(d_k) + \cos(d_j)\sin(d_k)\bigr]$$

The sums factorise because $j$ and $k$ are independent:

$$T_i = \frac{\sigma_2}{N^2} \left[\left(\sum_j \sin(d_j)\right)\left(\sum_k \cos(d_k)\right) + \left(\sum_j \cos(d_j)\right)\left(\sum_k \sin(d_k)\right)\right]$$

$$T_i = \frac{\sigma_2}{N^2} \cdot 2 S_i C_i$$

where $S_i = \sum_j \sin(\theta_j - \theta_i)$ and $C_i = \sum_j \cos(\theta_j - \theta_i)$.

This factorisation is exact (no approximation) and reduces the per-oscillator
cost from $O(N^2)$ to $O(N)$, making the total 3-body computation $O(N^2)$
instead of $O(N^3)$.

### Verification

The factorisation can be verified numerically:

```python
import numpy as np

N = 5
theta = np.random.uniform(0, 2 * np.pi, N)

# Brute force O(N³)
brute = np.zeros(N)
for i in range(N):
    for j in range(N):
        for k in range(N):
            brute[i] += np.sin(theta[j] + theta[k] - 2 * theta[i])

# Factorised O(N²)
fast = np.zeros(N)
for i in range(N):
    d = theta - theta[i]
    S = np.sum(np.sin(d))
    C = np.sum(np.cos(d))
    fast[i] = 2 * S * C

assert np.allclose(brute, fast, atol=1e-10)
```

---

## Appendix B: Relationship to Other Engines

| If you need... | Use... | Why |
|----------------|--------|-----|
| Only pairwise coupling | `UPDEEngine` | Standard Kuramoto, RK4/RK45 |
| 3-body + pairwise | `SimplicialEngine` | This module |
| Arbitrary k-body | `HypergraphEngine` | Generalises to any order |
| Amplitude dynamics | `StuartLandauEngine` | Phase + amplitude |
| Inertial oscillators | `InertialEngine` | Second-order (swing eq.) |
| Energy-preserving | `SplittingEngine` | Symplectic Strang split |
| Long simulations | `GeometricEngine` | Torus-preserving SO(2) |
| Time delays | `DelayedEngine` | VecDeque circular buffer |
| Mean-field prediction | `OttAntonsenReduction` | Complex ODE, fast |
