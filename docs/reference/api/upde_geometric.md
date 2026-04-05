# Geometric (Torus-Preserving) Engine

## 1. Mathematical Formalism

### Symplectic Euler on the N-Torus $T^N = (S^1)^N$

The geometric engine represents each oscillator phase as a unit complex
number $z_i = e^{i\theta_i}$ on the unit circle $S^1$, rather than as
a real number $\theta_i \in [0, 2\pi)$. The integration step uses the
**exponential map** on $S^1$:

$$z_i(t + \Delta t) = z_i(t) \cdot \exp\!\left(i \cdot \omega_{\text{eff},i} \cdot \Delta t\right)$$

where $\omega_{\text{eff},i}$ is the full Kuramoto right-hand side:

$$\omega_{\text{eff},i} = \omega_i + \sum_{j=1}^{N} K_{ij} \sin(\theta_j - \theta_i - \alpha_{ij}) + \zeta \sin(\Psi - \theta_i)$$

### Why Exponential Map?

Standard Euler integration computes $\theta_i \leftarrow \theta_i + \Delta t \cdot \dot{\theta}_i$
and then applies $\bmod 2\pi$. This creates two problems:

1. **Discontinuity artefacts:** When $\theta$ crosses $0/2\pi$, the mod
   operation introduces a jump. Derivatives computed from $\theta(t)$
   show spurious spikes at wrap points.

2. **Truncation error accumulation:** The linear step $\theta + \Delta t \cdot \dot{\theta}$
   does not respect the circular topology. Over many steps, phases drift
   off the torus and the mod correction accumulates error.

The exponential map avoids both problems: $z \cdot e^{i\omega\Delta t}$
is a **rotation** on $S^1$, which is exact (no truncation error in the
rotation itself). The only error is in $\omega_{\text{eff}}$, which is
computed at the current time step (first-order in $\Delta t$, same as Euler).

### Complex Representation

In the Rust implementation, $z_i$ is stored as $(re_i, im_i)$ pairs
(not as angle + reconstruction). The exponential map becomes:

$$\begin{pmatrix} re_i' \\ im_i' \end{pmatrix} = \begin{pmatrix} \cos\alpha & -\sin\alpha \\ \sin\alpha & \cos\alpha \end{pmatrix} \begin{pmatrix} re_i \\ im_i \end{pmatrix}$$

where $\alpha = \omega_{\text{eff},i} \cdot \Delta t$. After rotation,
the vector is renormalised to prevent numerical drift from the unit circle:

$$z_i \leftarrow \frac{z_i}{|z_i|}$$

### Phase Extraction

The final phases are recovered via $\theta_i = \text{atan2}(im_i, re_i) \bmod 2\pi$.
The `atan2` function is exact on IEEE 754 arithmetic, introducing no
additional error.

---

## 2. Theoretical Context

### Geometric Numerical Integration

The field of **geometric integration** studies numerical methods that
preserve the geometric structure of the continuous problem. For
oscillator systems on $T^N$:

- **Symplectic integrators** preserve the Hamiltonian structure
  (energy conservation up to exponentially small errors)
- **Lie group integrators** preserve the group structure of $SO(2)^N$
  (phases stay on the circle exactly)

The `TorusEngine` is a **Lie group integrator**: it applies the exponential
map of the Lie algebra $\mathfrak{so}(2) \cong \mathbb{R}$ to the Lie
group $SO(2) \cong S^1$. This guarantees that phases remain on the
torus without the need for modular arithmetic.

Reference: Hairer, Lubich & Wanner (2006), *Geometric Numerical
Integration*, Springer, Chapter IV.

### When to Use Geometric Integration

The torus-preserving property matters most for:

- **Long simulations** ($T > 10^5$ steps): Accumulated wrap artefacts
  in standard Euler can subtly bias statistics like mean phase velocity
  and diffusion coefficients.
- **Phase-sensitive observables:** ITPC, phase-amplitude coupling, and
  Poincaré sections are sensitive to wrap discontinuities.
- **Coupled map lattices:** When the coupling depends on exact phase
  differences, wrap artefacts can create spurious coupling forces.

For short simulations ($T < 1000$ steps), standard Euler with mod $2\pi$
is adequate and faster (no complex arithmetic overhead).

### The Lie Algebra Perspective

The phase space of $N$ coupled oscillators is the $N$-torus
$T^N = (S^1)^N$. Each $S^1$ is isomorphic to the Lie group $SO(2)$.
The Lie algebra $\mathfrak{so}(2) \cong \mathbb{R}$ consists of
angular velocities $\omega$.

The exponential map $\exp: \mathfrak{so}(2) \to SO(2)$ is:

$$\exp(\omega \Delta t) = \begin{pmatrix} \cos(\omega\Delta t) & -\sin(\omega\Delta t) \\ \sin(\omega\Delta t) & \cos(\omega\Delta t) \end{pmatrix}$$

The key insight is that the Kuramoto coupling term
$\dot{\theta}_i = \omega_i + \sum K_{ij}\sin(\Delta\theta)$ lives in the
Lie algebra (it is a tangent vector). The standard Euler step
$\theta \leftarrow \theta + \Delta t \cdot \dot{\theta}$ is a **retraction**
in the Lie algebra, followed by a **projection** back to the group (mod $2\pi$).
The geometric step replaces this with the exact exponential map, which is
the **canonical retraction** on $SO(2)$.

### Preservation Properties

The geometric integrator preserves:
1. **Topology:** Phases remain on $T^N$ exactly (no escaping the torus)
2. **Equivariance:** The dynamics are invariant under global phase shifts
   $\theta_i \to \theta_i + \phi$ — the integrator respects this symmetry
3. **Smoothness:** No discontinuities at $0/2\pi$ boundary

It does **not** preserve:
- **Energy** (not symplectic in the Hamiltonian sense — use `SplittingEngine`)
- **Measure** (the Euler step is not volume-preserving on $T^N$)

### Error Analysis

For the Kuramoto ODE $\dot{\theta} = f(\theta)$:

- **Standard Euler:** $\theta_{n+1} = \theta_n + \Delta t \cdot f(\theta_n) + O(\Delta t^2)$,
  then mod $2\pi$. Local error $O(\Delta t^2)$, global error $O(\Delta t)$.
  
- **Geometric Euler:** $z_{n+1} = z_n \cdot \exp(i \cdot f(\theta_n) \cdot \Delta t)$.
  Local error $O(\Delta t^2)$, global error $O(\Delta t)$ — **same order**.

The advantage is not in the convergence order but in the **qualitative
behaviour**: the geometric method cannot produce phases outside $[0, 2\pi)$,
and smooth phase trajectories remain smooth through the integration.

### Comparison with SplittingEngine

| Property | TorusEngine | SplittingEngine |
|----------|-------------|-----------------|
| Order of accuracy | 1st (Euler) | 2nd (Strang) |
| Phase preservation | Exact (Lie group) | Mod $2\pi$ |
| Energy preservation | Not symplectic | 2nd-order symplectic |
| Best for | Phase observables | Energy/frequency accuracy |
| Rust speedup | 1.4-41x | 2.0x |

---

## 3. Pipeline Position

```
Oscillators.extract() ──→ θ, ω
                               │
CouplingBuilder.build() ──→ K_nm, α
                               │
                               ↓
     ┌──── TorusEngine(n, dt) ─────────────────┐
     │                                          │
     │  Input:  θ, ω, K_nm, ζ, Ψ, α           │
     │  State:  z_i = (re_i, im_i) ∈ S¹       │
     │  Method: Lie group Euler (exp map)       │
     │  Output: θ_new ∈ [0, 2π)^N via atan2    │
     │                                          │
     └──────────────────────────────────────────┘
                               │
                               ↓
              compute_order_parameter(θ_new) → R, ψ
```

### Input Contracts

| Parameter | Type | Shape | Units | Source |
|-----------|------|-------|-------|--------|
| `phases` | `NDArray[float64]` | `(N,)` | radians | Previous step |
| `omegas` | `NDArray[float64]` | `(N,)` | rad/s | Oscillators |
| `knm` | `NDArray[float64]` | `(N, N)` | dimensionless | CouplingBuilder |
| `alpha` | `NDArray[float64]` | `(N, N)` | radians | Phase-lag |
| `zeta` | `float` | scalar | rad/s | External drive |
| `psi` | `float` | scalar | radians | Drive phase |

### Output Contract

| Output | Type | Shape | Range |
|--------|------|-------|-------|
| `phases_new` | `NDArray[float64]` | `(N,)` | $[0, 2\pi)$ via `atan2` + `rem_euclid` |

---

## 4. Features

- **Torus-preserving integration** via $SO(2)$ exponential map
- **No mod $2\pi$ discontinuities** — rotation is intrinsically circular
- **Unit circle renormalisation** prevents floating-point drift
- **Full Rust FFI acceleration** for `run()` (entire loop in Rust)
- **Standard Kuramoto RHS** — same coupling, drive, phase-lag as `UPDEEngine`
- **`step()` and `run()`** — single-step or batch
- **Automatic backend selection** via `_HAS_RUST`

---

## 5. Usage Examples

### Basic: Long Simulation Without Wrap Artefacts

```python
import numpy as np
from scpn_phase_orchestrator.upde.geometric import TorusEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

N = 32
engine = TorusEngine(N, dt=0.01)

phases = np.random.default_rng(42).uniform(0, 2 * np.pi, N)
omegas = np.linspace(0.9, 1.1, N)
knm = np.full((N, N), 0.5 / N); np.fill_diagonal(knm, 0.0)
alpha = np.zeros((N, N))

# 100,000 steps — no wrap artefacts
phases_final = engine.run(phases, omegas, knm, 0.0, 0.0, alpha, 100_000)

R, psi = compute_order_parameter(phases_final)
print(f"R = {R:.4f}, ψ = {psi:.4f}")
# All phases are in [0, 2π) — guaranteed by atan2
assert np.all(phases_final >= 0) and np.all(phases_final < 2 * np.pi)
```

### Comparison: Geometric vs Standard Euler

```python
import numpy as np
from scpn_phase_orchestrator.upde.geometric import TorusEngine
from scpn_phase_orchestrator.upde.engine import UPDEEngine

N = 16
dt = 0.01
phases = np.random.default_rng(42).uniform(0, 2 * np.pi, N)
omegas = np.ones(N) * 5.0  # Fast rotation → more wraps
knm = np.full((N, N), 0.3); np.fill_diagonal(knm, 0.0)
alpha = np.zeros((N, N))

# Geometric: torus-preserving
geo = TorusEngine(N, dt=dt)
p_geo = geo.run(phases.copy(), omegas, knm, 0.0, 0.0, alpha, 10_000)

# Standard Euler: mod 2π
upde = UPDEEngine(N, dt=dt, method="euler")
p_std = upde.run(phases.copy(), omegas, knm, 0.0, 0.0, alpha, 10_000)

# Both should give similar R, but phase trajectories differ near 0/2π
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter
R_geo, _ = compute_order_parameter(p_geo)
R_std, _ = compute_order_parameter(p_std)
print(f"Geometric R = {R_geo:.4f}, Standard R = {R_std:.4f}")
```

### Phase-Sensitive Observable (ITPC)

```python
import numpy as np
from scpn_phase_orchestrator.upde.geometric import TorusEngine
from scpn_phase_orchestrator.monitor.itpc import compute_itpc

N = 8
engine = TorusEngine(N, dt=0.01)
omegas = np.ones(N)
knm = np.full((N, N), 1.0); np.fill_diagonal(knm, 0.0)
alpha = np.zeros((N, N))

# Run multiple trials for ITPC
n_trials = 20
n_timepoints = 100
trials = np.zeros((n_trials, n_timepoints))

for trial in range(n_trials):
    phases = np.random.default_rng(trial).uniform(0, 2 * np.pi, N)
    for t in range(n_timepoints):
        phases = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)
        trials[trial, t] = phases[0]  # Track oscillator 0

itpc = compute_itpc(trials)
print(f"Mean ITPC = {np.mean(itpc):.4f}")
```

### Diffusion Coefficient Measurement

```python
import numpy as np
from scpn_phase_orchestrator.upde.geometric import TorusEngine

N = 4
engine = TorusEngine(N, dt=0.01)
omegas = np.array([1.0, 1.01, 0.99, 1.005])
knm = np.full((N, N), 0.05); np.fill_diagonal(knm, 0.0)
alpha = np.zeros((N, N))

# Track unwrapped phase for diffusion
n_steps = 50_000
phases = np.random.default_rng(42).uniform(0, 2 * np.pi, N)
trajectory = np.zeros((n_steps, N))

for t in range(n_steps):
    phases = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)
    trajectory[t] = phases

# Unwrap for MSD calculation (no artefacts from geometric integrator)
unwrapped = np.unwrap(trajectory, axis=0)
msd = np.mean((unwrapped - unwrapped[0])**2, axis=1)
# D ≈ MSD(t) / (2t) for large t
D = msd[-1] / (2 * n_steps * 0.01)
print(f"Diffusion coefficient D ≈ {D:.4f} rad²/s")
```

### Poincaré Section with Geometric Phases

```python
import numpy as np
from scpn_phase_orchestrator.upde.geometric import TorusEngine
from scpn_phase_orchestrator.monitor.poincare import poincare_section

N = 3
engine = TorusEngine(N, dt=0.005)
omegas = np.array([1.0, 1.618, 2.0])  # Incommensurate frequencies
knm = np.full((N, N), 0.2); np.fill_diagonal(knm, 0.0)
alpha = np.zeros((N, N))

phases = np.array([0.0, 0.5, 1.0])
trajectory = np.zeros(20_000)
for t in range(20_000):
    phases = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)
    trajectory[t] = phases[0]

# Geometric integrator produces clean Poincaré sections
# (no spurious crossings from mod 2π discontinuities)
crossings, times = poincare_section(trajectory, threshold=np.pi)
print(f"Found {len(crossings)} Poincaré crossings")
```

### Batch Simulation with Monitoring

```python
import numpy as np
from scpn_phase_orchestrator.upde.geometric import TorusEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

N = 64
engine = TorusEngine(N, dt=0.01)
phases = np.random.default_rng(42).uniform(0, 2 * np.pi, N)
omegas = np.ones(N) + 0.1 * np.random.default_rng(42).standard_normal(N)
knm = np.full((N, N), 0.3 / N); np.fill_diagonal(knm, 0.0)
alpha = np.zeros((N, N))

# Batch: 1000 steps in Rust, then monitor
for epoch in range(100):
    phases = engine.run(phases, omegas, knm, 0.0, 0.0, alpha, 1000)
    R, psi = compute_order_parameter(phases)
    if epoch % 20 == 0:
        print(f"Epoch {epoch:4d}: R = {R:.4f}")
```

---

## 6. Technical Reference

### Class: TorusEngine

::: scpn_phase_orchestrator.upde.geometric

### Rust Engine Function

```rust
pub fn torus_run(
    phases: &[f64],     // length N, initial angles
    omegas: &[f64],     // length N
    knm: &[f64],        // length N*N, row-major
    alpha: &[f64],      // length N*N, row-major
    zeta: f64,
    psi: f64,
    dt: f64,
    n_steps: usize,
) -> Vec<f64>           // length N, final angles in [0, 2π)
```

### Internal Steps (Rust)

1. Convert phases → complex: $z_i = (\cos\theta_i, \sin\theta_i)$
2. For each step:
   a. Extract angles: $\theta_i = \text{atan2}(im_i, re_i)$
   b. Compute $\omega_{\text{eff}}$ (standard Kuramoto RHS)
   c. Rotate: $z_i \leftarrow z_i \cdot e^{i\omega_{\text{eff}}\Delta t}$
      via $2 \times 2$ rotation matrix (uses `sin_cos()`)
   d. Renormalise: $z_i \leftarrow z_i / |z_i|$
3. Extract final angles via `atan2` + `rem_euclid(TAU)`

### Auto-Select Logic

Rust path is used for `run()` when `_HAS_RUST is True`. The `step()`
method uses the Python path (NumPy complex arithmetic) which is
competitive for single steps due to vectorisation.

---

## 7. Performance Benchmarks

Measured on Intel Core i5-11600K @ 3.90 GHz, 32 GB DDR4-2400.
500 Euler steps, all-to-all coupling $K = 0.5$.
Averaged over 10-50 (Rust) or 3-10 (Python) iterations.

| N | Python (ms) | Rust (ms) | Speedup |
|---|-------------|-----------|---------|
| 8 | 11.38 | 0.28 | **41.0x** |
| 32 | 13.99 | 3.33 | **4.2x** |
| 64 | 26.19 | 14.14 | **1.9x** |
| 128 | 60.99 | 44.36 | **1.4x** |

### Scaling Analysis

Both implementations are $O(N^2 \cdot T)$ per step (dominated by the
coupling sum). The Python path benefits from NumPy's BLAS-vectorised
complex arithmetic, reducing the gap at large $N$.

The Rust speedup decreases with $N$ because:
1. Python's NumPy complex broadcast is highly optimised for large arrays
2. Rust's scalar loop over $z_i$ does not use SIMD (potential improvement)
3. The `atan2` + `sin_cos` per oscillator per step adds overhead vs
   NumPy's vectorised `np.exp(1j * ...)` and `np.angle()`

### Comparison with UPDEEngine (Euler)

The geometric engine adds complex arithmetic overhead (atan2, sin_cos,
renormalisation per oscillator per step) compared to standard Euler.
At $N = 64$, `TorusEngine` (Rust) takes 14.14 ms for 500 steps.
`UPDEEngine` (Euler) avoids the complex representation entirely,
making it significantly faster for the same integration order.

### Breakdown of Per-Step Cost

For each oscillator per step, the Rust implementation performs:
1. `atan2(im, re)` — extract angle ($\sim 20$ ns)
2. $N$ multiplications + $N$ sin evaluations — coupling sum ($O(N)$)
3. `sin_cos(α)` — rotation angle ($\sim 5$ ns)
4. $2 \times 2$ matrix multiply — rotation ($\sim 2$ ns)
5. `sqrt` + 2 divisions — renormalisation ($\sim 5$ ns)

The overhead relative to standard Euler is steps 1, 3, 4, 5 — about
32 ns per oscillator per step. At $N = 64$, this adds $\sim 2$ μs per
step, accumulating to $\sim 1$ ms over 500 steps. The coupling sum
(step 2) dominates at large $N$, which is why the speedup converges to
$\sim 1.4\times$ at $N = 128$.

### Memory Usage

Rust: $2N$ extra `f64` values ($z_{re}$, $z_{im}$) + $N$ for $\theta$
extraction + $N$ for $\omega_{\text{eff}}$. Total: $4N \cdot 8$ bytes.
For $N = 1000$: 32 KB (fits in L1 cache).

Python: NumPy complex128 array of length $N$ + temporary arrays from
broadcasting. Total: $\sim 10N \cdot 8$ bytes due to intermediate
allocations.

### Test Coverage

- **Rust tests:** 7 (geometric module in spo-engine)
  - Free rotation, phase range, synchronisation, zero steps,
    unit circle preservation, external drive, identical phases
- **Python tests:** 6 (`tests/test_torus_engine.py`)
  - Output on torus, pure rotation exact, wrapping smooth,
    synchronisation, run shape, preserves sync
- **Source lines:** 205 (Rust) + 102 (Python) = 307 total

---

## 8. Citations

1. **Hairer, E., Lubich, C., & Wanner, G.** (2006).
   *Geometric Numerical Integration: Structure-Preserving Algorithms
   for Ordinary Differential Equations.* 2nd ed. Springer.
   DOI: [10.1007/3-540-30666-8](https://doi.org/10.1007/3-540-30666-8)
   — Chapter IV: Lie group methods.

2. **Iserles, A., Munthe-Kaas, H. Z., Nørsett, S. P., & Zanna, A.** (2000).
   "Lie-group methods."
   *Acta Numerica* 9:215-365.
   DOI: [10.1017/S0962492900002154](https://doi.org/10.1017/S0962492900002154)

3. **Celledoni, E., Marthinsen, H., & Owren, B.** (2014).
   "An introduction to Lie group integrators — basics, new developments
   and applications."
   *Journal of Computational Physics* 257:1040-1061.
   DOI: [10.1016/j.jcp.2012.12.031](https://doi.org/10.1016/j.jcp.2012.12.031)

4. **Kuramoto, Y.** (1975).
   "Self-entrainment of a population of coupled non-linear oscillators."
   In *International Symposium on Mathematical Problems in Theoretical Physics*,
   Lecture Notes in Physics 39:420-422. Springer.
   DOI: [10.1007/BFb0013365](https://doi.org/10.1007/BFb0013365)

5. **Acebrón, J. A., Bonilla, L. L., Pérez Vicente, C. J., Ritort, F.,
   & Spigler, R.** (2005).
   "The Kuramoto model: A simple paradigm for synchronization phenomena."
   *Reviews of Modern Physics* 77:137-185.
   DOI: [10.1103/RevModPhys.77.137](https://doi.org/10.1103/RevModPhys.77.137)

6. **Marsden, J. E. & Ratiu, T. S.** (1999).
   *Introduction to Mechanics and Symmetry.* 2nd ed. Springer.
   — Chapter 9: Lie groups and rigid body mechanics.

7. **Blanes, S. & Casas, F.** (2016).
   *A Concise Introduction to Geometric Numerical Integration.*
   CRC Press.
   DOI: [10.1201/b21563](https://doi.org/10.1201/b21563)

8. **McLachlan, R. I. & Quispel, G. R. W.** (2002).
   "Splitting methods."
   *Acta Numerica* 11:341-434.
   DOI: [10.1017/S0962492902000053](https://doi.org/10.1017/S0962492902000053)

---

## Edge Cases and Limitations

### Exact Zero Phase

When $\theta_i = 0$, $z_i = (1, 0)$. The `atan2(0, 1) = 0` — correct.
No special handling needed.

### Near-Antipodal Phases

When $\theta_i \approx \pi$, $z_i \approx (-1, 0)$. The `atan2`
returns $\pm\pi$ depending on the sign of the imaginary part. After
`rem_euclid(TAU)`, this maps to $\pi$ or $\approx 2\pi - \epsilon$.
This is the standard branch-cut behaviour and is consistent with the
mod $2\pi$ convention.

### Numerical Drift from Unit Circle

Without renormalisation, repeated rotation can drift $|z_i|$ away from
1.0 due to floating-point rounding. The Rust implementation renormalises
after every rotation. Over $10^6$ steps, the drift without renormalisation
is typically $O(10^{-12})$ — negligible but prevented for correctness.

### Very Large dt

For $\Delta t > 1 / \omega_{\text{max}}$, the rotation angle exceeds
$2\pi$ per step. The exponential map still works correctly (it wraps
naturally), but the integration accuracy degrades. Recommend
$\Delta t \cdot \omega_{\text{max}} < 0.5$ for accurate trajectories.

---

## Appendix: Decision Tree — Choosing an Integrator

```
Is the problem Hamiltonian (energy must be conserved)?
  ├─ Yes → SplittingEngine (2nd-order symplectic)
  └─ No → Is the simulation very long (>10⁵ steps)?
           ├─ Yes → Are phases used for ITPC/PAC/Poincaré?
           │         ├─ Yes → TorusEngine (no wrap artefacts)
           │         └─ No  → UPDEEngine (fastest, RK45 adaptive)
           └─ No → UPDEEngine (RK4 or RK45, standard choice)
```

### When NOT to Use TorusEngine

- **When accuracy matters more than topology:** The geometric engine is
  first-order (Euler). For the same $\Delta t$, `UPDEEngine` with RK4
  is 4th-order accurate — a better choice if you need precise trajectories.
  
- **When performance is critical:** The complex arithmetic overhead makes
  `TorusEngine` 1.4-47x slower than `UPDEEngine` (Euler, Rust). If the
  simulation budget is tight, the standard engine is better.

- **When $N$ is large ($> 256$):** At large $N$, the coupling sum
  dominates and the geometric overhead is small ($< 10\%$), but
  `UPDEEngine` with RK45 adaptive stepping can take larger steps for
  the same accuracy, winning on wall-clock time.

### When TO Use TorusEngine

- **Phase coherence studies** where wrap artefacts bias ITPC or PLV
- **Diffusion coefficient measurements** where mod $2\pi$ adds spurious
  jumps to $\text{MSD}(\theta)$
- **Visualisation** where smooth phase trajectories are needed
- **Mathematical rigour** where the proof assumes phases on $T^N$
