# Strang Operator Splitting Integrator

## 1. Mathematical Formalism

### The Kuramoto ODE and Its Decomposition

The general phase dynamics in SCPN read:

$$\frac{d\theta_i}{dt} = \omega_i + \sum_j K_{ij} \sin(\theta_j - \theta_i - \alpha_{ij}) + \zeta \sin(\psi - \theta_i)$$

This is a sum of two structurally different contributions:

$$\frac{d\theta_i}{dt} = \underbrace{\omega_i}_{\text{A: linear rotation}} + \underbrace{\sum_j K_{ij} \sin(\theta_j - \theta_i - \alpha_{ij}) + \zeta \sin(\psi - \theta_i)}_{\text{B: nonlinear coupling}}$$

**Operator A** is the free rotation: $\dot{\theta}_i = \omega_i$.
Its exact solution is $\theta_i(t+h) = \theta_i(t) + \omega_i h$.

**Operator B** is the coupling-only nonlinear part:
$\dot{\theta}_i = \sum_j K_{ij} \sin(\theta_j - \theta_i - \alpha_{ij}) + \zeta \sin(\psi - \theta_i)$.
This has no closed-form solution and requires numerical integration.

### Strang Splitting Scheme

The Strang splitting (Strang, 1968) composes the two operators
symmetrically to achieve second-order accuracy:

$$\Phi_h^{\text{Strang}} = \Phi_{h/2}^A \circ \Phi_h^B \circ \Phi_{h/2}^A$$

One full step advances the solution by $h = \Delta t$:

1. **A(dt/2):** $\theta_i \leftarrow (\theta_i + \frac{\Delta t}{2} \omega_i) \bmod 2\pi$ — exact
2. **B(dt):** RK4 substep on the coupling-only derivative — 4th-order
3. **A(dt/2):** $\theta_i \leftarrow (\theta_i + \frac{\Delta t}{2} \omega_i) \bmod 2\pi$ — exact

### Order of Accuracy

Let $\Phi_h^X$ denote the exact flow of operator $X$ for time $h$.
The Lie-Trotter splitting $\Phi_h^A \circ \Phi_h^B$ is first-order:
the local truncation error is $O(h^2)$.

The Strang (symmetric) splitting achieves second order because all
odd-order error terms cancel by symmetry:

$$\Phi_{h/2}^A \circ \Phi_h^B \circ \Phi_{h/2}^A = \Phi_h^{A+B} + O(h^3)$$

The global error after $N$ steps ($T = Nh$) is therefore $O(h^2)$.

### Why Not Split As A→B Instead of A/2→B→A/2?

The asymmetric Lie-Trotter splitting $\Phi_h^A \circ \Phi_h^B$ is
only first-order accurate. The symmetric Strang version gains one
order "for free" through palindromic composition. This is the same
principle that makes Verlet/leapfrog symplectic integrators
second-order despite using only first-order building blocks.

### RK4 on the Coupling Substep

The nonlinear operator B is integrated with classical Runge-Kutta:

$$k_1 = f_B(\theta^n)$$
$$k_2 = f_B\big((\theta^n + \tfrac{h}{2} k_1) \bmod 2\pi\big)$$
$$k_3 = f_B\big((\theta^n + \tfrac{h}{2} k_2) \bmod 2\pi\big)$$
$$k_4 = f_B\big((\theta^n + h\, k_3) \bmod 2\pi\big)$$
$$\theta^{n+1} = \big(\theta^n + \tfrac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)\big) \bmod 2\pi$$

where $f_B(\theta)_i = \sum_j K_{ij}\sin(\theta_j - \theta_i - \alpha_{ij}) + \zeta\sin(\psi - \theta_i)$.

The $\bmod 2\pi$ wrapping after each stage prevents phase accumulation
beyond $[0, 2\pi)$.

### Coupling Derivative

The coupling-only derivative for oscillator $i$:

$$f_B(\theta)_i = \sum_{j=1}^{N} K_{ij} \sin(\theta_j - \theta_i - \alpha_{ij}) + \zeta \sin(\psi - \theta_i)$$

The first term encodes pairwise coupling with phase frustration
$\alpha_{ij}$. The second term is an external driving force pulling
all oscillators toward global phase $\psi$ with strength $\zeta$.

---

## 2. Theoretical Context

### Why Operator Splitting for Phase Dynamics?

Monolithic integrators (e.g., adaptive RK45) treat the entire
right-hand side as a single nonlinear function. This works, but
it has a fundamental inefficiency: the linear rotation $\omega_i$
is integrated numerically despite having an exact solution.

Over long integrations ($T > 100$ oscillation periods), the numerical
error in the rotation term accumulates. For high-frequency oscillators
($\omega_i \gg K_{ij}$), this rotation error dominates. Splitting
eliminates it entirely: the A-operator is solved exactly, and
numerical error only arises from the coupling term.

### Geometric Properties

The Strang splitting inherits favourable geometric properties:

1. **Time-reversibility:** The palindromic structure
   $A(h/2) \to B(h) \to A(h/2)$ is its own adjoint. Running
   forward then backward with $-\Delta t$ returns to the starting
   point (up to floating-point errors).

2. **Phase-space volume preservation:** For Hamiltonian-like
   systems, symmetric splittings preserve the symplectic structure
   approximately. While the Kuramoto model is dissipative (not
   Hamiltonian), the energy-like invariant
   $E = -\sum_{ij} K_{ij} \cos(\theta_j - \theta_i)$
   drifts much more slowly under splitting than under non-symplectic
   methods.

3. **No secular energy drift:** Monolithic RK4 can introduce
   systematic energy drift proportional to $h^5 T$. The Strang
   splitting has bounded energy error oscillating around zero.

### Historical Context

- **Strang, G. (1968):** "On the construction and comparison of
  difference schemes." Introduced the symmetric splitting.
- **Marchuk, G. I. (1968):** Independent discovery of the same
  scheme in the Soviet literature (Marchuk splitting).
- **Hairer, Lubich & Wanner (2006):** *Geometric Numerical
  Integration*, §II.5 — comprehensive treatment of splitting
  methods for ODEs and PDEs.
- **McLachlan & Quispel (2002):** Survey of splitting methods
  with emphasis on geometric properties.
- **Blanes & Casas (2016):** *A Concise Introduction to Geometric
  Numerical Integration* — modern reference.

### Comparison with Other Integrators in SPO

| Integrator | Order | Rotation exact? | Symplectic? | Cost per step |
|------------|-------|----------------|-------------|---------------|
| Euler | 1 | No | No | $O(N^2)$ |
| RK4 (monolithic) | 4 | No | No | $4 \times O(N^2)$ |
| **Strang split** | **2** | **Yes** | **Approx.** | **$4 \times O(N^2)$** |
| Geometric (SO(2)) | 1 | Yes | Yes | $O(N^2)$ |

The Strang splitting occupies a middle ground: it is cheaper per
step than monolithic RK4 in practice (the A-substeps are $O(N)$
additions, not $O(N^2)$ sine evaluations), and it preserves the
rotation exactly. The trade-off is second-order global accuracy
versus fourth-order for monolithic RK4.

For the SCPN pipeline, the Strang splitting is preferred when:
- Integration spans many oscillation periods ($T > 100/\omega_{\max}$)
- The coupling is moderate ($K \sim \omega$)
- Long-term stability matters more than per-step accuracy

---

## 3. Pipeline Position

```
CouplingBuilder.build() ──→ CouplingState(knm, alpha)
                                       │
Oscillators.extract() ──→ ω₁, ..., ωₙ │
                               │       │
                               ↓       ↓
   ┌── SplittingEngine(n, dt) ────────────────┐
   │                                           │
   │  Input:  phases, omegas, knm, ζ, ψ, α   │
   │  Method: A(dt/2) → B_RK4(dt) → A(dt/2)  │
   │  Output: phases (updated, in [0, 2π))     │
   │                                           │
   │  step():  single Strang step              │
   │  run():   n_steps × step (Rust if avail.) │
   │                                           │
   └───────────────────────────────────────────┘
                               │
                               ↓
              compute_order_parameter(phases) → (R, ψ)
                               │
                               ↓
              RegimeManager.evaluate() → regime
```

### Input Contracts

| Parameter | Type | Shape | Range | Source |
|-----------|------|-------|-------|--------|
| `phases` | `NDArray[float64]` | `(N,)` | $[0, 2\pi)$ | Previous step or initial conditions |
| `omegas` | `NDArray[float64]` | `(N,)` | any | Natural frequencies from `Oscillators` |
| `knm` | `NDArray[float64]` | `(N, N)` | any | Coupling matrix from `CouplingBuilder` |
| `zeta` | `float` | scalar | $\geq 0$ | External drive strength |
| `psi` | `float` | scalar | $[0, 2\pi)$ | External drive phase |
| `alpha` | `NDArray[float64]` | `(N, N)` | any | Phase frustration matrix |
| `n_steps` | `int` | scalar | $\geq 0$ | Number of Strang steps (for `run()`) |

### Output Contract

| Field | Type | Shape | Range | Meaning |
|-------|------|-------|-------|---------|
| (return) | `NDArray[float64]` | `(N,)` | $[0, 2\pi)$ | Updated phases |

All output phases are wrapped to $[0, 2\pi)$ via `rem_euclid(TAU)`
(Rust) or `% (2 * np.pi)` (Python).

### Interchangeability with UPDEEngine

`SplittingEngine` and `UPDEEngine(method="rk4")` share the same
interface: `step(phases, omegas, knm, zeta, psi, alpha)` and
`run(..., n_steps)`. They can be swapped without changing
downstream pipeline code. The choice depends on the integration
regime (see §2, Comparison table).

---

## 4. Features

- **Exact linear rotation** — the $\omega_i$ part is solved analytically,
  eliminating truncation error in the dominant oscillatory term
- **Second-order symmetric splitting** — Strang composition, palindromic,
  all odd error terms cancel
- **RK4 coupling substep** — 4th-order accurate on the nonlinear part
- **Time-reversible** — forward + backward returns to initial state
  (verified in tests: error < 0.05 after 50+50 steps)
- **Phase-wrapped outputs** — all phases in $[0, 2\pi)$ at every stage
- **External drive** — supports global $\zeta \sin(\psi - \theta_i)$ forcing
- **Phase frustration** — supports full $\alpha_{ij}$ matrix
- **Rust FFI acceleration** — 1.2-2.5x speedup for $N \geq 32$
- **Pre-allocated scratch arrays** — `_phase_diff`, `_sin_diff`, `_scratch`
  avoid per-step heap allocation in the Python path
- **Same interface as UPDEEngine** — drop-in replacement for the
  standard monolithic integrator

---

## 5. Usage Examples

### Basic: Single Step

```python
import numpy as np
from scpn_phase_orchestrator.upde.splitting import SplittingEngine

N = 16
eng = SplittingEngine(N, dt=0.01)

rng = np.random.default_rng(42)
phases = rng.uniform(0, 2 * np.pi, N)
omegas = np.ones(N) * 2.0
knm = np.full((N, N), 0.5); np.fill_diagonal(knm, 0.0)
alpha = np.zeros((N, N))

new_phases = eng.step(phases, omegas, knm, zeta=0.0, psi=0.0, alpha=alpha)
assert np.all(new_phases >= 0) and np.all(new_phases < 2 * np.pi)
```

### Batch: Run Many Steps

```python
from scpn_phase_orchestrator.upde.splitting import SplittingEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter
import numpy as np

N = 32
eng = SplittingEngine(N, dt=0.01)
rng = np.random.default_rng(42)
phases = rng.uniform(0, 2 * np.pi, N)
omegas = np.ones(N)
knm = np.full((N, N), 1.0); np.fill_diagonal(knm, 0.0)
alpha = np.zeros((N, N))

# 500 Strang steps (uses Rust if available)
phases = eng.run(phases, omegas, knm, 0.0, 0.0, alpha, n_steps=500)
R, psi = compute_order_parameter(phases)
print(f"R = {R:.4f}, ψ = {psi:.4f}")
# Expect R > 0.9 for strong coupling
```

### With External Drive

```python
import numpy as np
from scpn_phase_orchestrator.upde.splitting import SplittingEngine

N = 8
eng = SplittingEngine(N, dt=0.01)
phases = np.zeros(N)
omegas = np.zeros(N)
knm = np.zeros((N, N))
alpha = np.zeros((N, N))

# External drive pulls all phases toward ψ = π/2
phases = eng.run(phases, omegas, knm, zeta=1.0, psi=np.pi/2, alpha=alpha, n_steps=200)
# Phases should cluster near π/2
print(f"Mean phase: {np.mean(phases):.4f}")
```

### Comparison with Monolithic RK4

```python
import numpy as np
from scpn_phase_orchestrator.upde.splitting import SplittingEngine
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

N = 16
dt = 0.005
rng = np.random.default_rng(42)
phases0 = rng.uniform(0, 2 * np.pi, N)
omegas = np.ones(N) * 1.5
knm = np.full((N, N), 0.3); np.fill_diagonal(knm, 0.0)
alpha = np.zeros((N, N))

split = SplittingEngine(N, dt=dt)
mono = UPDEEngine(N, dt=dt, method="rk4")

ps = split.run(phases0.copy(), omegas, knm, 0.0, 0.0, alpha, n_steps=400)
pm = mono.run(phases0.copy(), omegas, knm, 0.0, 0.0, alpha, n_steps=400)

R_split, _ = compute_order_parameter(ps)
R_mono, _ = compute_order_parameter(pm)
print(f"R_split = {R_split:.4f}, R_mono = {R_mono:.4f}")
# Should agree within < 0.1
```

### Full Pipeline: CouplingBuilder → Splitting → Regime

```python
import numpy as np
from scpn_phase_orchestrator.coupling.knm import CouplingBuilder
from scpn_phase_orchestrator.upde.splitting import SplittingEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter
from scpn_phase_orchestrator.supervisor.regimes import RegimeManager
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState
from scpn_phase_orchestrator.monitor.boundaries import BoundaryState

N = 16
cb = CouplingBuilder()
cs = cb.build(n_layers=N, base_strength=0.5, decay_alpha=0.2)

eng = SplittingEngine(N, dt=0.01)
rng = np.random.default_rng(42)
phases = rng.uniform(0, 2 * np.pi, N)
omegas = np.ones(N)

phases = eng.run(phases, omegas, cs.knm, 0.0, 0.0, cs.alpha, n_steps=300)
R, psi = compute_order_parameter(phases)

layer = LayerState(R=R, psi=psi)
state = UPDEState(
    layers=[layer],
    cross_layer_alignment=np.array([R]),
    stability_proxy=R,
    regime_id="nominal",
)
rm = RegimeManager(hysteresis=0.05)
regime = rm.evaluate(state, BoundaryState())
print(f"R = {R:.4f}, Regime: {regime.name}")
```

---

## 6. Technical Reference

### Class: SplittingEngine

::: scpn_phase_orchestrator.upde.splitting

### Constructor

```python
SplittingEngine(n_oscillators: int, dt: float)
```

Pre-allocates three scratch arrays:
- `_phase_diff`: `(N, N)` — pairwise phase differences
- `_sin_diff`: `(N, N)` — $\sin(\theta_j - \theta_i - \alpha_{ij})$
- `_scratch`: `(N,)` — coupling derivative accumulator

### Methods

| Method | Signature | Returns |
|--------|-----------|---------|
| `step` | `(phases, omegas, knm, zeta, psi, alpha)` | `NDArray[float64]` shape `(N,)` |
| `run` | `(phases, omegas, knm, zeta, psi, alpha, n_steps)` | `NDArray[float64]` shape `(N,)` |

`step()` always uses the Python path (for single-step interactive use).
`run()` dispatches to Rust when `_HAS_RUST` is `True`.

### Rust Engine Function

```rust
pub fn splitting_run(
    phases: &[f64],     // N phases in [0, 2π)
    omegas: &[f64],     // N natural frequencies
    knm: &[f64],        // N×N coupling matrix (row-major flat)
    alpha: &[f64],      // N×N frustration matrix (row-major flat)
    zeta: f64,          // external drive strength
    psi: f64,           // external drive phase
    dt: f64,            // time step
    n_steps: usize,     // number of Strang steps
) -> Vec<f64>           // N phases in [0, 2π)
```

The Rust implementation uses `rem_euclid(TAU)` for wrapping (always
non-negative, unlike `%` which can return negative values in some
languages).

### Internal Functions (Rust)

- `rk4_coupling(p, knm, alpha, n, zeta, psi, dt)` — one RK4 step
  on the coupling-only derivative, modifies `p` in place
- `coupling_deriv(theta, knm, alpha, n, zeta, psi) -> Vec<f64>` —
  computes $f_B(\theta)$ for all $N$ oscillators

### Auto-Select Logic

```python
try:
    from spo_kernel import splitting_run_rust as _rust_splitting_run
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False
```

The `run()` method dispatches to Rust when available. The Python
path flattens `knm` and `alpha` to contiguous row-major arrays
before the FFI call.

---

## 7. Performance Benchmarks

Measured on Intel Core i5-11600K @ 3.90 GHz, 32 GB DDR4-2400.
100 Strang steps, random phases/omegas/coupling, median of 20-50 runs.

### Varying N (100 steps)

| N | Python (ms) | Rust (ms) | Speedup |
|---|-------------|-----------|---------|
| 32 | 16.338 | 6.646 | **2.5x** |
| 128 | 147.620 | 120.388 | **1.2x** |
| 256 | 1301.959 | 575.504 | **2.3x** |

### Varying Steps (N=64)

| Steps | Python (ms) | Rust (ms) | Speedup |
|-------|-------------|-----------|---------|
| 10 | 3.753 | 2.693 | **1.4x** |
| 100 | 45.039 | 27.373 | **1.6x** |
| 500 | 222.750 | 131.050 | **1.7x** |
| 1,000 | 445.273 | 266.008 | **1.7x** |

### Why Only 1.2-2.5x Speedup?

Unlike scalar ODE modules (e.g., OA reduction at 38-96x), the
splitting engine is dominated by $O(N^2)$ `sin()` evaluations in the
coupling derivative. The Python path uses NumPy broadcasting:

```python
np.subtract(theta[np.newaxis, :], theta[:, np.newaxis], out=self._phase_diff)
np.sin(self._phase_diff, out=self._sin_diff)
np.sum(knm * self._sin_diff, axis=1, out=self._scratch)
```

NumPy's `sin()` calls GLIBC's vectorised `libm` (or Intel MKL on
some systems), which is already SIMD-optimised. The Rust path uses
scalar `f64::sin()` in a double loop. The Rust advantage comes from
avoiding Python object overhead and the 4 intermediate NumPy
allocations per RK4 stage.

For maximum speedup, the Rust path would need SIMD-vectorised sine
(e.g., `sleef-rs` or `packed_simd`). This is on the roadmap.

### Memory Usage

| Component | Python | Rust |
|-----------|--------|------|
| Scratch arrays | 3 × $N^2$ floats (pre-allocated) | 4 × $N$ Vecs (per step) |
| Total for N=256 | ~1.5 MB | ~8 KB |

The Python path pre-allocates large scratch arrays to avoid per-step
allocation. The Rust path allocates small per-step vectors for RK4
stages (k1-k4).

### Test Coverage

- **Rust tests:** 7 (splitting module in spo-engine)
  - Free rotation (exact), synchronisation, phases in range, zero steps,
    external drive, second-order accuracy verification, identical phases
- **Python tests:** 13 (`tests/test_splitting.py`)
  - Output range, zero coupling rotation, synchronisation, agreement
    with monolithic RK4, run n_steps, external drive, preserves sync,
    reversibility, finite stability (2000 steps), pipeline end-to-end
    (CouplingBuilder → Splitting → Regime), R convergence vs monolithic,
    performance budget (step(64) < 3ms)
- **Source lines:** 260 (Rust) + 122 (Python) = 382 total

---

## 8. Citations

1. **Strang, G.** (1968).
   "On the construction and comparison of difference schemes."
   *SIAM Journal on Numerical Analysis* 5(3):506-517.
   DOI: [10.1137/0705041](https://doi.org/10.1137/0705041)

2. **Hairer, E., Lubich, C., & Wanner, G.** (2006).
   *Geometric Numerical Integration: Structure-Preserving Algorithms
   for Ordinary Differential Equations.* 2nd ed., Springer.
   ISBN: 978-3-540-30663-4. §II.5 (Splitting Methods).

3. **McLachlan, R. I. & Quispel, G. R. W.** (2002).
   "Splitting methods."
   *Acta Numerica* 11:341-434.
   DOI: [10.1017/S0962492902000053](https://doi.org/10.1017/S0962492902000053)

4. **Blanes, S. & Casas, F.** (2016).
   *A Concise Introduction to Geometric Numerical Integration.*
   CRC Press. ISBN: 978-1-4822-6341-1.

5. **Marchuk, G. I.** (1968).
   "Some application of splitting-up methods to the solution of
   mathematical physics problems."
   *Aplikace Matematiky* 13(2):103-132.

6. **Kuramoto, Y.** (1984).
   *Chemical Oscillations, Waves, and Turbulence.*
   Springer. ISBN: 978-3-642-69691-6.

7. **Acebrón, J. A., Bonilla, L. L., Pérez Vicente, C. J., Ritort, F.,
   & Spigler, R.** (2005).
   "The Kuramoto model: A simple paradigm for synchronization phenomena."
   *Reviews of Modern Physics* 77:137-185.
   DOI: [10.1103/RevModPhys.77.137](https://doi.org/10.1103/RevModPhys.77.137)

8. **Yoshida, H.** (1990).
   "Construction of higher order symplectic integrators."
   *Physics Letters A* 150(5-7):262-268.
   DOI: [10.1016/0375-9601(90)90092-3](https://doi.org/10.1016/0375-9601(90)90092-3)

---

## Edge Cases and Limitations

### Negative dt (Backward Integration)

The `SplittingEngine` supports negative `dt` for backward integration.
The test suite verifies time-reversibility: 50 forward steps followed
by 50 backward steps recovers the initial state to within 0.05 radians.
This is a direct consequence of the palindromic Strang structure.

### Very Strong Coupling ($K \gg \omega$)

When coupling dominates ($K > 10 \omega_{\max}$), the B-substep
becomes stiff. The RK4 integrator may lose accuracy or stability.
In this regime, either reduce $\Delta t$ or consider using the
monolithic `UPDEEngine` with an adaptive stepper.

Stability criterion for the RK4 substep: $K \cdot N \cdot \Delta t < 2.8$.
For $N = 256$ and $K = 1/N$, this gives $\Delta t < 2.8$ — well
above typical values.

### All Identical Phases

When all oscillators start with the same phase and the same frequency,
the coupling derivative is zero (since $\sin(0) = 0$). The splitting
engine correctly reduces to pure rotation: $\theta(t) = \theta_0 + \omega t$.
This is verified in the Rust test suite.

### Zero Steps

`run(phases, ..., n_steps=0)` returns a copy of the input phases
without modification. Both Python and Rust paths handle this correctly.

### Phase Wrapping Precision

The Python `% (2 * np.pi)` and Rust `rem_euclid(TAU)` produce
slightly different results at machine precision. The maximum
discrepancy is bounded by $\varepsilon_{\text{machine}} \approx 2.2 \times 10^{-16}$.

---

## Integration with Other SPO Modules

### With SSGF Geometry Control

The splitting engine integrates naturally with the SSGF two-timescale
loop:

```python
# Fast timescale: splitting integrates phases
phases = split_eng.step(phases, omegas, W, 0.0, 0.0, alpha)

# Slow timescale: SSGF updates geometry W
snapshot = pgbo.observe(phases, W)
W = ssgf_step(W, snapshot, learning_rate=0.001)
```

The exact rotation in the A-substep is particularly valuable here:
when $W$ changes slowly, the dominant contribution to $\dot{\theta}_i$
is the rotation $\omega_i$, which is handled without numerical error.

### With OttAntonsenReduction

For all-to-all coupling with Lorentzian $g(\omega)$, the OA reduction
predicts $R_{ss}$. The splitting engine can validate this prediction:

```python
oa = OttAntonsenReduction(omega_0=0, delta=0.5, K=3.0)
R_predicted = oa.steady_state_R()

# Full simulation with splitting
phases = split_eng.run(phases, omegas, knm, 0.0, 0.0, alpha, n_steps=5000)
R_measured, _ = compute_order_parameter(phases)

# Should agree within finite-N fluctuations
assert abs(R_predicted - R_measured) < 0.1
```

### With GeometricEngine

The `GeometricEngine` (SO(2) exponential map) and `SplittingEngine`
are complementary:
- **GeometricEngine:** preserves the torus geometry exactly (unit complex
  numbers), first-order accurate
- **SplittingEngine:** preserves the rotation exactly, second-order
  accurate

For problems where both geometric fidelity and rotation accuracy
matter, alternating between the two or composing them is possible.

---

## Troubleshooting

### Issue: Splitting Diverges but Monolithic RK4 Does Not

**Diagnosis:** The coupling is stiff ($K \cdot N \gg 1/\Delta t$).
The Strang splitting is only second-order, so it requires smaller
$\Delta t$ than fourth-order RK4 for the same accuracy.

**Solution:** Reduce $\Delta t$ by factor 4 (to match the accuracy
of RK4 at the original step size), or switch to monolithic RK4.

### Issue: R Oscillates Instead of Converging

**Diagnosis:** This may be correct physics — the Kuramoto model with
frustration ($\alpha \neq 0$) or heterogeneous frequencies can exhibit
persistent oscillations in $R$. Check whether monolithic RK4 shows
the same behaviour.

If only splitting oscillates: the step size is too large for the
coupling strength. The per-step splitting error is $O(h^3)$ per step,
accumulating as $O(h^2)$ globally.

### Issue: Phases Cluster at 0 or $2\pi$

**Diagnosis:** This is a wrapping artefact. Phases near 0 and near
$2\pi$ represent the same physical state on $S^1$. Use circular
statistics (e.g., `compute_order_parameter`) rather than linear
mean/variance.
