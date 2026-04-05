# Phase-SINDy Symbolic Discovery

## 1. Mathematical Formalism

### Sparse Identification of Nonlinear Dynamics (SINDy)

SINDy discovers governing equations from data by solving a sparse
regression problem. For coupled phase oscillators, the dynamics
of oscillator $i$ are:

$$\frac{d\theta_i}{dt} = \omega_i + \sum_{j \neq i} K_{ij} \sin(\theta_j - \theta_i)$$

### Library Construction

For each oscillator $i$, construct a library matrix
$\Theta \in \mathbb{R}^{(T-1) \times (1 + N-1)}$:

$$\Theta = \begin{bmatrix} 1 & \sin(\theta_1 - \theta_i) & \sin(\theta_2 - \theta_i) & \cdots & \sin(\theta_N - \theta_i) \end{bmatrix}$$

The first column is a constant (captures $\omega_i$); the remaining
columns are pairwise sinusoidal interaction terms.

### Target Vector

The time derivative is estimated from the phase trajectory:

$$\dot{\theta}_i(t) \approx \frac{\theta_i(t + \Delta t) - \theta_i(t)}{\Delta t}$$

with phase unwrapping to handle $2\pi$ wraparounds.

### STLSQ (Sequential Thresholded Least Squares)

SINDy solves the regression $\dot{\theta}_i = \Theta \cdot \xi_i$
using iterative hard thresholding:

1. **Initialise:** $\xi_i = (\Theta^T \Theta)^{-1} \Theta^T \dot{\theta}_i$
   (least squares)
2. **Threshold:** Set $|\xi_{i,k}| < \lambda$ to zero
3. **Re-solve:** Least squares on remaining non-zero coefficients
4. **Repeat** for `max_iter` iterations

The threshold $\lambda$ controls sparsity: higher $\lambda$ produces
sparser equations (fewer coupling terms), lower $\lambda$ retains
more structure.

### Coefficient Interpretation

The coefficient vector $\xi_i$ decodes as:

| Index | Feature | Coefficient | Physical Meaning |
|-------|---------|-------------|------------------|
| 0 | 1 | $\omega_i$ | Natural frequency |
| 1 | $\sin(\theta_1 - \theta_i)$ | $K_{i1}$ | Coupling $1 \to i$ |
| 2 | $\sin(\theta_2 - \theta_i)$ | $K_{i2}$ | Coupling $2 \to i$ |
| ... | ... | ... | ... |

### Normal Equations (Rust)

The Rust implementation solves the normal equations directly:

$$\xi = (X^T X)^{-1} X^T y$$

using Gaussian elimination with partial pivoting, rather than the
SVD-based lstsq used in Python (via LAPACK dgelsd).

---

## 2. Theoretical Context

### Why SINDy for Oscillator Networks?

Traditional system identification methods assume a model structure
(e.g., linear state-space). SINDy is **data-driven**: it discovers
the governing equations from observed dynamics without assuming
a specific model.

For coupled oscillators, this is powerful because:
1. The coupling topology $K_{ij}$ is often unknown
2. The natural frequencies $\omega_i$ may be uncertain
3. The interaction function (assumed sinusoidal) can be verified
   by inspecting the residuals

### Relation to Compressed Sensing

STLSQ is a greedy sparse recovery algorithm, related to:
- **LASSO** ($L_1$ penalty): convex relaxation of sparsity
- **Orthogonal Matching Pursuit (OMP):** greedy selection
- **Iterative Hard Thresholding:** closest to STLSQ

STLSQ is preferred because it preserves the least-squares fit
quality while enforcing sparsity through hard thresholding.

### Phase-SINDy Specifics

The standard SINDy library includes polynomials, trigonometric
functions, and their products. Phase-SINDy specialises the library
to the Kuramoto interaction terms:
- Constant term (natural frequency)
- Pairwise $\sin(\theta_j - \theta_i)$ (first harmonic coupling)

Higher harmonics ($\sin(2(\theta_j - \theta_i))$, etc.) could be
added but are not included in the current implementation.

### Historical Context

- **Brunton, S. L., Proctor, J. L., & Kutz, J. N.** (2016):
  "Discovering governing equations from data by sparse identification
  of nonlinear dynamical systems." The original SINDy paper.
  *PNAS* 113(15):3932-3937.
- **Champion, K. et al.** (2019): "Data-driven discovery of
  coordinates and governing equations." Extended SINDy with
  autoencoder coordinates.
- **Stankovski, T. et al.** (2012): "Inference of time-varying
  Kuramoto coupling functions." Phase-specific coupling discovery.
- **Kralemann, B. et al.** (2008): "Phase dynamics of coupled
  oscillators reconstructed from data." Bayesian approach to
  coupling function estimation.

### Limitations

1. **Assumes sinusoidal coupling:** If the true interaction is
   non-sinusoidal, SINDy will produce biased estimates
2. **Requires sufficient data:** $T \gg N$ for reliable regression
3. **Noise sensitivity:** Finite-difference derivatives amplify noise
4. **Global coupling only:** Cannot detect time-varying or
   state-dependent coupling

### Identifiability

For the Kuramoto model, coupling coefficients are identifiable
from phase data if and only if:
1. **The system is not fully synchronised** ($R < 1$): at $R = 1$,
   all $\sin(\theta_j - \theta_i) = 0$ and the coupling matrix
   disappears from the regression
2. **There is sufficient excitation**: oscillators must explore
   a range of phase differences for the regression to distinguish
   different couplings
3. **The library is correctly specified**: if the true coupling
   includes higher harmonics or phase frustration, the pure
   $\sin(\Delta\theta)$ library is misspecified

### Comparison with Other System Identification Methods

| Method | Linearity | Sparsity | Noise | Complexity |
|--------|-----------|----------|-------|------------|
| Phase-SINDy (STLSQ) | Nonlinear | Yes | Moderate | $O(TN^2)$ |
| Granger causality | Linear | No | Good | $O(TN^2)$ |
| Transfer entropy | Nonlinear | No | Good | $O(T N^2 B)$ |
| Dynamic Mode Decomposition | Linear | No | Good | $O(TN^2)$ |
| Bayesian coupling inference | Nonlinear | No | Excellent | $O(T N^3)$ |

Phase-SINDy is unique in combining nonlinear discovery with
built-in sparsity promotion. It is the natural choice when the
coupling topology is expected to be sparse (most $K_{ij} = 0$).

### STLSQ Convergence

The STLSQ algorithm is not guaranteed to converge to a global
minimum. It is a greedy heuristic that works well in practice
for sparse problems. Typical convergence requires 5-20 iterations.
The `max_iter` parameter defaults to 10, which is sufficient for
most oscillator networks.

The algorithm can oscillate if two features have nearly identical
contributions (aliasing). In this case, increasing the threshold
or adding a small $L_2$ regularisation can help.

---

## 3. Pipeline Position

```
 UPDEEngine.run() ──→ phases(t), shape (T, N)
                           │
                           ↓
 ┌── PhaseSINDy.fit(phases, dt) ─────────────────┐
 │                                                 │
 │  Step 1: Compute θ̇ via finite differences     │
 │  Step 2: Build library Θ for each node         │
 │  Step 3: STLSQ regression → ξ_i               │
 │  Step 4: Extract ω_i (diagonal), K_ij (off)    │
 │                                                 │
 │  Output: list of coefficient vectors            │
 │          [ω_i, K_i1, K_i2, ...] per oscillator │
 └──────────────────┬──────────────────────────────┘
                    │
                    ↓
 PhaseSINDy.get_equations() → symbolic equations
                    │
                    ↓
 CouplingBuilder.from_discovered(ξ) → updated K_nm
```

### Input Contracts

| Parameter | Type | Shape | Range | Meaning |
|-----------|------|-------|-------|---------|
| `phases` | `NDArray[float64]` | `(T, N)` | $[0, 2\pi)$ | Phase trajectory |
| `dt` | `float` | scalar | $> 0$ | Time step |

### Output Contract

| Field | Type | Shape | Meaning |
|-------|------|-------|---------|
| `coefficients` | `list[NDArray]` | $N \times [1+N-1]$ | Per-node coefficient vectors |
| `feature_names` | `list[list[str]]` | $N \times [1+N-1]$ | Human-readable feature labels |

---

## 4. Features

- **Data-driven equation discovery** — no assumed coupling topology
- **STLSQ sparse regression** — identifies active couplings,
  prunes spurious ones
- **Per-node regression** — handles independent coupling per oscillator
- **Symbolic equations** — `get_equations()` returns human-readable
  discovered dynamics
- **Rust FFI acceleration** — 7.8-15x speedup for small N (≤ 8)
- **Coefficient remapping** — Rust diagonal=ω, off-diagonal=K_ij
  correctly mapped to Python layout
- **Configurable threshold** — controls sparsity/accuracy trade-off
- **Configurable max iterations** — controls STLSQ convergence

---

## 5. Usage Examples

### Basic: Discover Equations

```python
import numpy as np
from scpn_phase_orchestrator.autotune.sindy import PhaseSINDy
from scpn_phase_orchestrator.upde.engine import UPDEEngine

N = 4
dt = 0.01
eng = UPDEEngine(N, dt=dt)
rng = np.random.default_rng(42)
phases = rng.uniform(0, 2 * np.pi, N)
omegas = np.array([1.0, 1.5, 2.0, 0.5])
knm = np.array([
    [0.0, 0.3, 0.0, 0.0],
    [0.3, 0.0, 0.5, 0.0],
    [0.0, 0.5, 0.0, 0.2],
    [0.0, 0.0, 0.2, 0.0],
])
alpha = np.zeros((N, N))

# Collect trajectory
trajectory = [phases.copy()]
for _ in range(500):
    phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
    trajectory.append(phases.copy())
traj = np.array(trajectory)  # (T, N)

# Discover equations
sindy = PhaseSINDy(threshold=0.05, max_iter=10)
sindy.fit(traj, dt)
for eq in sindy.get_equations():
    print(eq)
```

### Compare Discovered vs True Coupling

```python
import numpy as np
from scpn_phase_orchestrator.autotune.sindy import PhaseSINDy

# True: K_01 = 0.3, K_12 = 0.5
# SINDy should recover these
sindy = PhaseSINDy(threshold=0.01)
coeffs = sindy.fit(trajectory, dt=0.01)

# coeffs[0] = [ω_0, K_01, K_02, K_03]
print(f"True ω_0 = 1.0, Discovered: {coeffs[0][0]:.3f}")
print(f"True K_01 = 0.3, Discovered: {coeffs[0][1]:.3f}")
```

### Threshold Sensitivity

```python
import numpy as np
from scpn_phase_orchestrator.autotune.sindy import PhaseSINDy

for threshold in [0.01, 0.05, 0.1, 0.5]:
    sindy = PhaseSINDy(threshold=threshold)
    sindy.fit(trajectory, dt=0.01)
    n_nonzero = sum(np.count_nonzero(c) for c in sindy.coefficients)
    print(f"λ={threshold:.2f}: {n_nonzero} non-zero coefficients")
```

### Pipeline: Discover → Validate → Deploy

```python
import numpy as np
from scpn_phase_orchestrator.autotune.sindy import PhaseSINDy
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

N = 8
dt = 0.01
eng = UPDEEngine(N, dt=dt)
rng = np.random.default_rng(42)

# True parameters
true_omegas = rng.uniform(0.5, 2.0, N)
true_knm = np.zeros((N, N))
for i in range(N - 1):
    true_knm[i, i+1] = 0.3
    true_knm[i+1, i] = 0.3

# Generate trajectory
phases = rng.uniform(0, 2 * np.pi, N)
trajectory = [phases.copy()]
for _ in range(1000):
    phases = eng.step(phases, true_omegas, true_knm, 0.0, 0.0, np.zeros((N, N)))
    trajectory.append(phases.copy())
traj = np.array(trajectory)

# Discover
sindy = PhaseSINDy(threshold=0.05)
sindy.fit(traj, dt)

# Extract discovered coupling
K_disc = np.zeros((N, N))
for i, xi in enumerate(sindy.coefficients):
    j_idx = 0
    for j in range(N):
        if j != i:
            K_disc[i, j] = xi[1 + j_idx]
            j_idx += 1

# Validate: simulate with discovered K
phases2 = rng.uniform(0, 2 * np.pi, N)
for _ in range(500):
    phases2 = eng.step(phases2, true_omegas, K_disc, 0.0, 0.0, np.zeros((N, N)))
R, _ = compute_order_parameter(phases2)
print(f"R with discovered K: {R:.4f}")
```

### Visualise Discovered Equations

```python
from scpn_phase_orchestrator.autotune.sindy import PhaseSINDy

sindy = PhaseSINDy(threshold=0.05)
sindy.fit(trajectory, dt=0.01)
equations = sindy.get_equations()
for eq in equations:
    print(eq)
# Example output:
# d(theta_0)/dt = 1.0021 * 1 + 0.2987 * sin(theta_1 - theta_0)
# d(theta_1)/dt = 1.5003 * 1 + 0.3012 * sin(theta_0 - theta_1)
```

---

## 6. Technical Reference

### Class: PhaseSINDy

::: scpn_phase_orchestrator.autotune.sindy

### Constructor

```python
PhaseSINDy(threshold: float = 0.05, max_iter: int = 10)
```

### Rust Engine Function

```rust
pub fn sindy_fit(
    phases: &[f64],   // row-major (T × N) phase trajectory
    n_osc: usize,
    n_time: usize,
    dt: f64,
    threshold: f64,
    max_iter: usize,
) -> Vec<f64>         // N×N: [i][i]=ω_i, [i][j]=K_ij for j≠i
```

Internal helpers:
- `compute_theta_dot` — finite-difference derivative with unwrapping
- `build_library` — construct Θ for one node
- `stlsq_node` — STLSQ regression for one node

### Coefficient Layout

| Backend | Layout |
|---------|--------|
| Rust | $N \times N$ matrix: diagonal=ω, off-diagonal=K |
| Python | List of vectors: $[\omega_i, K_{j_1}, K_{j_2}, \ldots]$ per node |

The Python wrapper remaps the Rust layout to match the Python
convention.

---

## 7. Performance Benchmarks

Measured on Intel Core i5-11600K @ 3.90 GHz, 32 GB DDR4-2400.
T = 500 timesteps, threshold = 0.05, max_iter = 10.

| N | Python (ms) | Rust (ms) | Speedup |
|---|-------------|-----------|---------|
| 4 | 4.800 | 0.321 | **15.0x** |
| 8 | 9.917 | 1.274 | **7.8x** |
| 16 | 26.316 | 28.722 | **0.9x** |

### Why Does Rust Slow Down at N=16?

The bottleneck shifts from Python overhead (dominant at small N)
to the linear algebra solver:
- **Python:** LAPACK lstsq (dgelsd) via SciPy, $O(TN^2)$ with
  optimised BLAS
- **Rust:** Normal equations with Gaussian elimination, $O(TN^2 + N^3)$
  with naive loops

At N=16, the LAPACK solver's SIMD advantage dominates.

### Memory Usage

- Library Θ: $(T-1) \times N$ per node = $N \times (T-1) \times N$ total
- Working storage: $N \times N$ for normal equations per node
- Total for N=8, T=500: ~160 KB

### Test Coverage

- **Rust tests:** 5 (sindy module in spo-engine)
- **Python tests:** 1 (`tests/test_sindy.py`)
- **Source lines:** 269 (Rust) + 113 (Python) = 382 total

---

## 8. Citations

1. **Brunton, S. L., Proctor, J. L., & Kutz, J. N.** (2016).
   "Discovering governing equations from data by sparse identification
   of nonlinear dynamical systems."
   *PNAS* 113(15):3932-3937.
   DOI: [10.1073/pnas.1517384113](https://doi.org/10.1073/pnas.1517384113)

2. **Champion, K., Lusch, B., Kutz, J. N., & Brunton, S. L.** (2019).
   "Data-driven discovery of coordinates and governing equations."
   *PNAS* 116(45):22445-22451.
   DOI: [10.1073/pnas.1906995116](https://doi.org/10.1073/pnas.1906995116)

3. **Stankovski, T., Duggento, A., McClintock, P. V. E., &
   Stefanovska, A.** (2012).
   "Inference of time-evolving coupled dynamical systems in the
   presence of noise."
   *Physical Review Letters* 109(2):024101.
   DOI: [10.1103/PhysRevLett.109.024101](https://doi.org/10.1103/PhysRevLett.109.024101)

4. **Kralemann, B., Cimponeriu, L., Rosenblum, M., Pikovsky, A., &
   Mrowka, R.** (2008).
   "Phase dynamics of coupled oscillators reconstructed from data."
   *Physical Review E* 77(6):066205.
   DOI: [10.1103/PhysRevE.77.066205](https://doi.org/10.1103/PhysRevE.77.066205)

5. **Tibshirani, R.** (1996).
   "Regression shrinkage and selection via the lasso."
   *Journal of the Royal Statistical Society B* 58(1):267-288.

6. **Donoho, D. L.** (2006).
   "Compressed sensing."
   *IEEE Transactions on Information Theory* 52(4):1289-1306.
   DOI: [10.1109/TIT.2006.871582](https://doi.org/10.1109/TIT.2006.871582)

7. **Kuramoto, Y.** (1984).
   *Chemical Oscillations, Waves, and Turbulence.*
   Springer. ISBN: 978-3-642-69691-6.

8. **Rudy, S. H., Brunton, S. L., Proctor, J. L., & Kutz, J. N.**
   (2017).
   "Data-driven discovery of partial differential equations."
   *Science Advances* 3(4):e1602614.
   DOI: [10.1126/sciadv.1602614](https://doi.org/10.1126/sciadv.1602614)

---

## Edge Cases and Limitations

### Too Few Timesteps

For $T < N + 2$, the regression is underdetermined. The function
requires $T \geq 3$ for finite-difference derivatives. Recommended:
$T \geq 10 N$.

### Noisy Phase Data

Finite-difference derivatives amplify noise by factor $1/\Delta t$.
For noisy data, consider smoothing the trajectory before fitting
or using a Savitzky-Golay filter for derivatives.

### All Coefficients Thresholded to Zero

If the threshold $\lambda$ is too high, all coefficients may be
zeroed out. The discovered equation becomes $\dot{\theta}_i = 0$.
Reduce $\lambda$ or increase the trajectory length.

### Phase Wrapping

The Python path uses `np.unwrap` to handle $2\pi$ discontinuities.
The Rust path implements inline unwrapping. Both produce identical
results for smooth trajectories but may differ for highly noisy data
where unwrapping ambiguity exists.

---

## Troubleshooting

### Issue: Discovered ω Differs from True ω

**Diagnosis:** The finite-difference derivative estimates
$\dot{\theta}$ with first-order accuracy. For large $\Delta t$,
this introduces systematic bias.

**Solution:** Use smaller $\Delta t$ or a higher-order derivative
estimate (central differences, Savitzky-Golay).

### Issue: Spurious Coupling Detected

**Diagnosis:** The threshold $\lambda$ is too low, allowing noise
to appear as coupling.

**Solution:** Increase $\lambda$. A good heuristic: $\lambda \approx
2 \sigma_{\text{noise}} / \sqrt{T}$ where $\sigma_{\text{noise}}$
is the phase noise standard deviation.

---

## Integration with Other SPO Modules

### With CouplingBuilder

Discovered coupling coefficients can initialise or validate the
coupling matrix:

```python
sindy = PhaseSINDy(threshold=0.05)
sindy.fit(trajectory, dt)
# Extract K matrix from coefficients
K_discovered = np.zeros((N, N))
for i, xi in enumerate(sindy.coefficients):
    j_idx = 0
    for j in range(N):
        if j != i:
            K_discovered[i, j] = xi[1 + j_idx]
            j_idx += 1
```

### With UniversalPrior

The discovered $\omega_i$ and $K_{ij}$ can be compared to the
universal prior's expectations. Large deviations suggest the domain
is unusual and may benefit from custom prior parameters.

### With TE-Directed Adaptation

SINDy and TE-directed adaptation are complementary:
- **SINDy:** Discovers the static coupling structure from a single
  trajectory window (offline)
- **TE-directed:** Continuously adapts coupling based on ongoing
  causal information flow (online)

A typical workflow: use SINDy for initial discovery, then hand off
to TE-directed adaptation for online refinement.
