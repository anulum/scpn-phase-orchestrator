# Coupling Estimation from Phase Data

## 1. Mathematical Formalism

### Inverse Problem: Recover K from θ(t)

Given observed phase trajectories $\theta_i(t)$ and known natural
frequencies $\omega_i$, estimate the coupling matrix $K_{ij}$ by
inverting the Kuramoto equation:

$$\frac{d\theta_i}{dt} - \omega_i = \sum_{j=1}^{N} K_{ij} \sin(\theta_j - \theta_i)$$

### Least-Squares Formulation

For each oscillator $i$, define:

- **Target:** $y_i(t) = \dot{\theta}_i(t) - \omega_i$
- **Regressors:** $X_{ij}(t) = \sin(\theta_j(t) - \theta_i(t))$

The regression problem is:

$$y_i = X_i \cdot K_{i,:}^T$$

where $X_i \in \mathbb{R}^{T \times N}$ and $K_{i,:}$ is the $i$-th
row of the coupling matrix.

### Solution via Pseudoinverse

$$K_{i,:} = (X_i^T X_i)^{-1} X_i^T y_i$$

The Python implementation uses `numpy.linalg.lstsq` (LAPACK dgelsd,
SVD-based) which is numerically stable and handles rank-deficient
systems gracefully.

### Phase Derivative Estimation

The time derivative is computed via first-order finite differences
on the unwrapped phase:

$$\dot{\theta}_i(t) \approx \frac{\text{unwrap}(\theta_i(t+1)) - \text{unwrap}(\theta_i(t))}{\Delta t}$$

Phase unwrapping removes $2\pi$ discontinuities before differentiation.

### Higher-Harmonic Extension

The `estimate_coupling_harmonics` function extends the library to
include higher Fourier harmonics:

$$\dot{\theta}_i - \omega_i = \sum_{j} \sum_{k=1}^{K_h} \left[ a_{ijk} \sin(k \Delta\theta_{ij}) + b_{ijk} \cos(k \Delta\theta_{ij}) \right]$$

This captures non-sinusoidal coupling functions that appear in
biological oscillators (Stankovski et al. 2017).

### Identifiability Conditions

The coupling matrix is identifiable when:
1. $T \gg N$: sufficient data points relative to unknowns
2. The phases explore diverse phase differences (not locked at $\Delta\theta = 0$)
3. The coupling function is correctly specified (sinusoidal for first harmonic)

### Fisher Information and Cramér-Rao Bound

The minimum variance of the coupling estimate is bounded by the
Cramér-Rao inequality:

$$\text{Var}(\hat{K}_{ij}) \geq \frac{\sigma^2_{\text{noise}}}{T \cdot \text{Var}(\sin(\theta_j - \theta_i))}$$

where $\sigma^2_{\text{noise}}$ is the phase noise variance. This
shows that:
- Longer trajectories ($T$) reduce estimation variance as $1/T$
- Higher phase variability (more diverse $\Delta\theta$) improves estimates
- Phase noise directly degrades accuracy

### SVD vs Normal Equations

The pseudoinverse can be computed two ways:

**SVD (Python/LAPACK):**
$$K_i = V \Sigma^{-1} U^T y_i$$
where $X_i = U \Sigma V^T$ is the thin SVD. Numerically stable even
for ill-conditioned problems ($\kappa(X) > 10^{10}$). Cost: $O(TN^2)$.

**Normal equations (Rust):**
$$K_i = (X_i^T X_i)^{-1} X_i^T y_i$$
Requires solving an $N \times N$ system. Numerically unstable when
$X_i^T X_i$ is near-singular ($\kappa$ doubles). Cost: $O(TN^2 + N^3)$.

For well-conditioned problems ($\kappa < 10^6$), both give identical
results. For ill-conditioned problems (e.g., highly synchronised
systems), the SVD path is more reliable.

### Fourier Coupling Functions

The general coupling function between oscillators $i$ and $j$ can
be expanded in Fourier series:

$$\Gamma_{ij}(\Delta\theta) = \sum_{k=1}^{\infty} \left[ a_k \sin(k \Delta\theta) + b_k \cos(k \Delta\theta) \right]$$

The standard Kuramoto model uses only the first harmonic ($k=1$,
$b_1 = 0$). Real biological oscillators (cardiac, neural, circadian)
often have significant second and third harmonics. The
`estimate_coupling_harmonics` function fits up to $K_h$ harmonics.

### Rust Path (Disabled)

The Rust implementation uses normal equations with Gaussian
elimination. Benchmarking showed it is **3x slower** than Python's
LAPACK lstsq for this problem, so the Rust auto-select is disabled.
The Rust module exists for correctness testing and environments
without LAPACK.

---

## 2. Theoretical Context

### Coupling Estimation in Oscillator Networks

Recovering the coupling structure from observed dynamics is a
fundamental problem in nonlinear dynamics. Unlike linear system
identification (where correlation suffices), coupled oscillator
networks require nonlinear regression on trigonometric basis functions.

### Comparison with Other Methods

| Method | Assumption | Complexity | Noise Robustness |
|--------|-----------|------------|------------------|
| **Least squares (this)** | Sinusoidal coupling | $O(TN^2)$ | Moderate |
| Phase-SINDy (STLSQ) | Sparse sinusoidal | $O(TN^2)$ | Moderate |
| Transfer entropy | Model-free | $O(TN^2 B)$ | Good |
| Granger causality | Linear | $O(TN^2)$ | Good |
| Maximum likelihood | Any parametric | $O(TN^3)$ | Excellent |
| Bayesian (MCMC) | Any parametric | $O(TN^3 \cdot S)$ | Excellent |

Least-squares coupling estimation is the simplest and fastest
approach. It is appropriate when:
- The coupling is approximately sinusoidal
- The data is relatively clean (SNR > 10)
- Speed matters more than statistical optimality

### Relation to Phase-SINDy

The `estimate_coupling` function and `PhaseSINDy` solve the same
mathematical problem. The differences:
- `estimate_coupling`: Dense solution (all K_ij estimated, including
  near-zero values)
- `PhaseSINDy`: Sparse solution (small K_ij thresholded to zero)

Use `estimate_coupling` when the coupling topology is expected to
be dense. Use `PhaseSINDy` when sparsity is expected.

### Historical Context

- **Pikovsky, A., Rosenblum, M., & Kurths, J.** (2001):
  *Synchronization.* Phase dynamics reconstruction from data.
- **Stankovski, T. et al.** (2017): "Coupling functions: Universal
  insights." Higher-harmonic coupling estimation.
- **Kralemann, B. et al.** (2008): Phase dynamics reconstruction
  from noisy oscillator data.
- **Tokuda, I. T. et al.** (2007): "Inferring phase equations from
  multivariate time series." Bayesian approach.
- **Ota, K. & Aoyagi, T.** (2014): Direct estimation of coupling
  from short time series.

### Why LAPACK Beats Rust

The LAPACK lstsq solver uses:
1. **SVD decomposition** (divide-and-conquer dgesdd)
2. **BLAS Level 3** matrix-matrix operations (cache-aware)
3. **SIMD vectorisation** (SSE2/AVX2 on x86)

The Rust implementation uses:
1. **Normal equations** $(X^T X)^{-1} X^T y$ — numerically less stable
2. **Gaussian elimination** — scalar loops without SIMD
3. **No BLAS dependency** — portable but slower

For $N = 16$ with $T = 200$, the $O(TN^2 + N^3)$ cost is dominated
by the matrix operations where BLAS excels. The 3x slowdown is
consistent with the BLAS advantage for medium-sized dense linear
algebra.

---

## 3. Pipeline Position

```
 UPDEEngine.run() ──→ phases(n, T), omegas(n)
                           │
                           ↓
 ┌── estimate_coupling(phases, omegas, dt) ──────┐
 │                                                │
 │  Per oscillator i:                             │
 │    y_i = dθ_i/dt - ω_i                        │
 │    X_i = [sin(θ_j - θ_i)] for all j           │
 │    K_i = lstsq(X_i, y_i)                      │
 │                                                │
 │  Output: K_ij (n × n), zero diagonal           │
 └──────────────────┬─────────────────────────────┘
                    │
                    ↓
 CouplingBuilder / Validation against known topology
```

### Input Contracts

| Parameter | Type | Shape | Range | Meaning |
|-----------|------|-------|-------|---------|
| `phases` | `NDArray[float64]` | `(N, T)` | $[0, 2\pi)$ | Phase trajectories |
| `omegas` | `NDArray[float64]` | `(N,)` | any | Known natural frequencies |
| `dt` | `float` | scalar | $> 0$ | Time step |

### Output Contract

| Field | Type | Shape | Constraints |
|-------|------|-------|-------------|
| (return) | `NDArray[float64]` | `(N, N)` | Diagonal = 0 |

---

## 4. Features

- **Dense coupling estimation** — estimates all $N^2$ coupling
  coefficients simultaneously
- **Pseudoinverse solver** — LAPACK dgelsd via numpy.linalg.lstsq
- **Phase unwrapping** — handles $2\pi$ discontinuities
- **Per-node regression** — independent fit for each oscillator
- **Higher harmonics** — `estimate_coupling_harmonics` for
  non-sinusoidal coupling functions
- **Robust to singular matrices** — `contextlib.suppress(LinAlgError)`
  returns zero row for degenerate nodes
- **Rust engine available** — disabled by default (3x slower than LAPACK)
- **Zero diagonal** — self-coupling forced to zero
- **Minimum data check** — raises ValueError for $T < 3$

---

## 5. Usage Examples

### Basic: Estimate Coupling

```python
import numpy as np
from scpn_phase_orchestrator.autotune.coupling_est import estimate_coupling
from scpn_phase_orchestrator.upde.engine import UPDEEngine

N = 8
dt = 0.01
eng = UPDEEngine(N, dt=dt)
rng = np.random.default_rng(42)
phases = rng.uniform(0, 2 * np.pi, N)
omegas = rng.uniform(0.5, 2.0, N)
knm_true = np.zeros((N, N))
for i in range(N - 1):
    knm_true[i, i+1] = 0.3
    knm_true[i+1, i] = 0.3

# Generate trajectory
trajectory = [phases.copy()]
for _ in range(500):
    phases = eng.step(phases, omegas, knm_true, 0.0, 0.0, np.zeros((N, N)))
    trajectory.append(phases.copy())
traj = np.array(trajectory).T  # (N, T)

# Estimate
K_est = estimate_coupling(traj, omegas, dt)
print(f"True K[0,1] = {knm_true[0,1]:.3f}")
print(f"Est  K[0,1] = {K_est[0,1]:.3f}")
print(f"RMSE = {np.sqrt(np.mean((K_est - knm_true)**2)):.4f}")
```

### Higher Harmonics

```python
import numpy as np
from scpn_phase_orchestrator.autotune.coupling_est import estimate_coupling_harmonics

# With non-sinusoidal coupling (biological oscillators)
result = estimate_coupling_harmonics(traj, omegas, dt, n_harmonics=3)
print(f"1st harmonic sin: {result['sin_1'][0, 1]:.4f}")
print(f"2nd harmonic sin: {result['sin_2'][0, 1]:.4f}")
print(f"3rd harmonic sin: {result['sin_3'][0, 1]:.4f}")
```

### Validation: Estimated vs True Coupling

```python
import numpy as np
from scpn_phase_orchestrator.autotune.coupling_est import estimate_coupling

K_est = estimate_coupling(traj, omegas, dt)

# Correlation between true and estimated
mask = ~np.eye(N, dtype=bool)
corr = np.corrcoef(knm_true[mask], K_est[mask])[0, 1]
print(f"Pearson correlation: {corr:.4f}")
```

### Noise Robustness Test

```python
import numpy as np
from scpn_phase_orchestrator.autotune.coupling_est import estimate_coupling

for noise_std in [0.0, 0.01, 0.05, 0.1]:
    noisy_traj = traj + np.random.default_rng(42).normal(0, noise_std, traj.shape)
    K_est = estimate_coupling(noisy_traj, omegas, dt)
    rmse = np.sqrt(np.mean((K_est - knm_true)**2))
    print(f"Noise σ={noise_std:.2f}: RMSE = {rmse:.4f}")
```

### Trajectory Length Sensitivity

```python
import numpy as np
from scpn_phase_orchestrator.autotune.coupling_est import estimate_coupling

for T_use in [50, 100, 200, 500, 1000]:
    K_est = estimate_coupling(traj[:, :T_use], omegas, dt)
    rmse = np.sqrt(np.mean((K_est - knm_true)**2))
    print(f"T={T_use:4d}: RMSE = {rmse:.4f}")
# Expect RMSE decreases as T increases
```

### Complete Pipeline: Generate → Estimate → Validate → Simulate

```python
import numpy as np
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter
from scpn_phase_orchestrator.autotune.coupling_est import estimate_coupling

N = 8
dt = 0.01
eng = UPDEEngine(N, dt=dt)
rng = np.random.default_rng(42)
omegas = rng.uniform(0.5, 2.0, N)
knm_true = np.zeros((N, N))
for i in range(N - 1):
    knm_true[i, i+1] = 0.4
    knm_true[i+1, i] = 0.4

# 1. Generate training trajectory
phases = rng.uniform(0, 2 * np.pi, N)
traj = [phases.copy()]
for _ in range(1000):
    phases = eng.step(phases, omegas, knm_true, 0.0, 0.0, np.zeros((N, N)))
    traj.append(phases.copy())
traj = np.array(traj).T

# 2. Estimate coupling
K_est = estimate_coupling(traj, omegas, dt)

# 3. Validate with R comparison
phases_true = rng.uniform(0, 2 * np.pi, N)
phases_est = phases_true.copy()
for _ in range(500):
    phases_true = eng.step(phases_true, omegas, knm_true, 0.0, 0.0, np.zeros((N, N)))
    phases_est = eng.step(phases_est, omegas, K_est, 0.0, 0.0, np.zeros((N, N)))
R_true, _ = compute_order_parameter(phases_true)
R_est, _ = compute_order_parameter(phases_est)
print(f"R (true K): {R_true:.4f}, R (est K): {R_est:.4f}")
```

---

## 6. Technical Reference

### Function: estimate_coupling

::: scpn_phase_orchestrator.autotune.coupling_est

### Function: estimate_coupling_harmonics

```python
def estimate_coupling_harmonics(
    phases: NDArray,      # (N, T) phase trajectories
    omegas: NDArray,      # (N,) natural frequencies
    dt: float,            # time step
    n_harmonics: int = 2, # number of Fourier harmonics
) -> dict[str, NDArray]   # {"sin_1": (N,N), "cos_1": (N,N), ...}
```

### Rust Engine Functions (Disabled)

```rust
pub fn coupling_est_fit(
    phases: &[f64],   // row-major (N × T)
    omegas: &[f64],
    n: usize, t: usize, dt: f64,
) -> Vec<f64>         // N×N coupling matrix
```

Internal helpers: `unwrapped_deriv`, `solve_node`,
`forward_elimination`, `back_substitution`.

### Auto-Select: DISABLED

```python
# Rust path exists but is disabled:
# _HAS_RUST = False  (hardcoded after benchmarking showed 3x slowdown)
```

---

## 7. Performance Benchmarks

Measured on Intel Core i5-11600K @ 3.90 GHz, 32 GB DDR4-2400.
T = 200 timesteps, Python (LAPACK lstsq) only.

| N | Python (ms) | Rust (ms) | Speedup |
|---|-------------|-----------|---------|
| 4 | 0.493 | 1.5* | **0.3x** |
| 8 | 1.155 | 3.5* | **0.3x** |
| 16 | 4.579 | 14* | **0.3x** |

*Rust benchmarks from prior session (before auto-select was disabled).

### Why Rust is 3x Slower

See §2 "Why LAPACK Beats Rust" — BLAS-accelerated SVD vs naive
Gaussian elimination.

### Scaling

The cost is $O(T N^2)$ for the regression (dominated by the matrix
construction) plus $O(N^3)$ for the pseudoinverse per node, giving
$O(N^4)$ total. For $N = 16$, $T = 200$: ~4.6 ms.

### Memory Usage

- Regressors: $N \times T$ per node = $N^2 T$ total
- Total for N=8, T=200: ~100 KB

### Test Coverage

- **Rust tests:** 7 (coupling_est module in spo-engine)
- **Python tests:** 11 (`tests/test_coupling_estimation.py`)
- **Source lines:** 238 (Rust) + 124 (Python) = 362 total

---

## 8. Citations

1. **Pikovsky, A., Rosenblum, M., & Kurths, J.** (2001).
   *Synchronization: A Universal Concept in Nonlinear Sciences.*
   Cambridge University Press. ISBN: 978-0-521-59285-7.

2. **Stankovski, T., Pereira, T., McClintock, P. V. E., &
   Stefanovska, A.** (2017).
   "Coupling functions: Universal insights into dynamical interaction
   mechanisms."
   *Reviews of Modern Physics* 89(4):045001.
   DOI: [10.1103/RevModPhys.89.045001](https://doi.org/10.1103/RevModPhys.89.045001)

3. **Kralemann, B., Cimponeriu, L., Rosenblum, M., Pikovsky, A., &
   Mrowka, R.** (2008).
   "Phase dynamics of coupled oscillators reconstructed from data."
   *Physical Review E* 77(6):066205.
   DOI: [10.1103/PhysRevE.77.066205](https://doi.org/10.1103/PhysRevE.77.066205)

4. **Tokuda, I. T., Jain, S., Kiss, I. Z., & Hudson, J. L.** (2007).
   "Inferring phase equations from multivariate time series."
   *Physical Review Letters* 99(6):064101.
   DOI: [10.1103/PhysRevLett.99.064101](https://doi.org/10.1103/PhysRevLett.99.064101)

5. **Ota, K. & Aoyagi, T.** (2014).
   "Direct extraction of phase dynamics from fluctuating rhythmic
   data."
   *Frontiers in Computational Neuroscience* 8:98.
   DOI: [10.3389/fncom.2014.00098](https://doi.org/10.3389/fncom.2014.00098)

6. **Dörfler, F. & Bullo, F.** (2014).
   "Synchronization in complex networks of phase oscillators."
   *Automatica* 50(6):1539-1564.
   DOI: [10.1016/j.automatica.2014.04.012](https://doi.org/10.1016/j.automatica.2014.04.012)

7. **Golub, G. H. & Van Loan, C. F.** (2013).
   *Matrix Computations.* 4th ed. Johns Hopkins University Press.
   ISBN: 978-1-4214-0794-4.
   (Chapters on least squares and SVD algorithms.)

8. **Kuramoto, Y.** (1984).
   *Chemical Oscillations, Waves, and Turbulence.*
   Springer. ISBN: 978-3-642-69691-6.

---

## Edge Cases and Limitations

### T < 3

Raises `ValueError`. At least 3 timesteps are needed for the
finite-difference derivative.

### Fully Synchronised Phases (R = 1)

When all $\sin(\theta_j - \theta_i) = 0$, the regressor matrix is
rank-zero. The lstsq returns zero coupling. This is correct: at
perfect synchronisation, the coupling is unidentifiable from the
dynamics.

### Negative Estimated Coupling

The least-squares solution can return negative $K_{ij}$, indicating
inhibitory (repulsive) coupling. This is physically meaningful in
some systems (e.g., phase-repulsive oscillators) but should be
validated against the known physics.

### omegas Mismatch

If the provided $\omega_i$ differ from the true natural frequencies,
the residual $\dot{\theta}_i - \omega_i$ contains a systematic
error that biases the coupling estimates. Use accurate frequency
estimates (e.g., from `identify_frequencies`).

---

## Troubleshooting

### Issue: All Coupling Estimates Near Zero

**Diagnosis:** Either (a) the system is fully synchronised, (b)
the trajectory is too short ($T < 10N$), or (c) the coupling is
genuinely near zero.

**Solution:** Run the simulation longer or add perturbations
to excite diverse phase differences.

### Issue: Large Asymmetric Coupling

**Diagnosis:** Asymmetric estimates ($K_{ij} \neq K_{ji}$) arise
from finite-sample noise or non-sinusoidal coupling.

**Solution:** Symmetrise the result: $K \leftarrow (K + K^T) / 2$.
Or use `estimate_coupling_harmonics` to capture non-sinusoidal terms.

---

## Integration with Other SPO Modules

### With Phase-SINDy

Use `estimate_coupling` for the dense estimate and `PhaseSINDy`
for the sparse estimate. Compare the two to assess sparsity:

```python
K_dense = estimate_coupling(traj, omegas, dt)
sindy = PhaseSINDy(threshold=0.05)
sindy.fit(traj.T, dt)
# Compare: sindy should have sparser K
```

### With UniversalPrior

The estimated coupling can validate the universal prior:

```python
K_est = estimate_coupling(traj, omegas, dt)
K_base_est = K_est[K_est > 0].mean()
prior = UniversalPrior()
lp = prior.log_probability(K_base_est, 0.25)
print(f"Estimated K_base = {K_base_est:.3f}, log p = {lp:.3f}")
```

### With OttAntonsenReduction

The estimated effective coupling $K_{\text{eff}} = \frac{1}{N} \sum_{ij} K_{ij}$
can be validated against the OA steady-state prediction:

```python
K_eff = K_est.sum() / N
oa = OttAntonsenReduction(omega_0=np.median(omegas),
                          delta=np.std(omegas), K=K_eff)
R_predicted = oa.steady_state_R()
```

### With SSGF Geometry Control

The estimated coupling provides a data-driven initialisation for
the SSGF geometry carrier, replacing the random projection:

```python
carrier = GeometryCarrier(N, z_dim=8)
# Use estimated K as target for the SSGF cost function
target_K = estimate_coupling(traj, omegas, dt)
```

### With TE-Directed Adaptation

Compare the static coupling estimate (this module) with the
TE-directed adaptive coupling:

```python
K_static = estimate_coupling(traj, omegas, dt)
K_te = te_adapt_coupling(K_init, traj, lr=0.01)
# If K_static ≈ K_te → TE and regression agree on structure
corr = np.corrcoef(K_static.ravel(), K_te.ravel())[0, 1]
print(f"Structural agreement: r = {corr:.4f}")
```
