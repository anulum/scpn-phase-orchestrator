# Basin Stability — Attractor Volume Estimation

The `basin_stability` module estimates the probability that a random
initial condition converges to the synchronised attractor — a Monte Carlo
measure of dynamical robustness introduced by Menck et al. (2013). Basin
stability $S_B \in [0, 1]$ answers: "how likely is this system to
synchronise from a random start?"

Unlike local stability analysis (eigenvalues), basin stability is a
**global** measure. A system can be locally stable ($\lambda_{\max} < 0$)
yet have low basin stability ($S_B \ll 1$) if the basin of attraction is
small relative to phase space.

---

## 1. Mathematical Formalism

### 1.1 Definition

For a dynamical system $\dot{\theta} = f(\theta)$ with an attractor $A$
(the synchronised state), the basin of attraction $\mathcal{B}(A)$ is the
set of all initial conditions that converge to $A$:

$$
\mathcal{B}(A) = \{\theta_0 \in \Omega : \lim_{t \to \infty} \phi_t(\theta_0) \in A\}
$$

where $\phi_t$ is the flow and $\Omega = [0, 2\pi)^N$ is the full phase space.

Basin stability is the normalised volume:

$$
S_B(A) = \frac{\text{Vol}(\mathcal{B}(A))}{\text{Vol}(\Omega)}
$$

Since $\Omega = [0, 2\pi)^N$ is a torus, $\text{Vol}(\Omega) = (2\pi)^N$.

### 1.2 Monte Carlo Estimation

Direct computation of $\text{Vol}(\mathcal{B})$ is intractable for $N > 3$.
Menck et al. (2013) proposed Monte Carlo estimation:

1. Draw $M$ initial conditions $\theta_0^{(k)} \sim \text{Uniform}([0, 2\pi)^N)$
2. For each, integrate $\dot{\theta} = f(\theta)$ to steady state
3. Compute $R_{\text{final}}^{(k)}$ — the time-averaged order parameter
4. Count $N_{\text{sync}} = \#\{k : R_{\text{final}}^{(k)} \geq R_{\text{thresh}}\}$

$$
\hat{S}_B = \frac{N_{\text{sync}}}{M}
$$

The estimator $\hat{S}_B$ is unbiased with standard error:

$$
\text{SE}(\hat{S}_B) = \sqrt{\frac{S_B(1 - S_B)}{M}}
$$

For $M = 100$ (default): $\text{SE} \leq 0.05$ (5% maximum standard error).
For $M = 1000$: $\text{SE} \leq 0.016$.

### 1.3 Convergence Criterion

A trial is classified as "synchronised" if $\bar{R} \geq R_{\text{thresh}}$,
where $\bar{R}$ is the time-averaged order parameter over the measurement
window:

$$
\bar{R} = \frac{1}{T_m} \sum_{t=T_0}^{T_0 + T_m} R(t)
$$

with $T_0 = n_{\text{transient}}$ (discard transient) and
$T_m = n_{\text{measure}}$ (averaging window).

Default threshold $R_{\text{thresh}} = 0.8$ — chosen to distinguish genuine
synchronisation from partial clustering. Lower thresholds detect partial sync;
higher thresholds require near-perfect locking.

### 1.4 Multi-Basin Analysis

For systems with multiple attractors (e.g., synchronised, chimera, desync),
`multi_basin_stability` classifies outcomes at multiple thresholds:

| Regime | Criterion |
|--------|-----------|
| Desynchronised | $\bar{R} < R_1$ (e.g., $R_1 = 0.3$) |
| Partial sync | $R_1 \leq \bar{R} < R_3$ (e.g., between 0.3 and 0.8) |
| Synchronised | $\bar{R} \geq R_3$ (e.g., $R_3 = 0.8$) |

This detects multistability: if both $S_B(\text{sync})$ and
$S_B(\text{desync})$ are positive, the system has coexisting attractors.

### 1.5 Relationship to K_c

At the critical coupling $K = K_c$:
- $S_B \approx 0$ (for $K < K_c$) — random ICs almost never synchronise
- $S_B$ jumps discontinuously to $\sim 0.5$ at $K_c$ (for finite $N$)
- $S_B \to 1$ for $K \gg K_c$ — sync basin fills phase space

The $S_B(K)$ curve provides complementary information to the $R(K)$
bifurcation diagram: $R(K)$ shows the attractor value, $S_B(K)$ shows
its robustness.

### 1.6 Statistical Properties

For $M$ independent uniform samples:

| Quantity | Formula |
|----------|---------|
| Estimator | $\hat{S}_B = N_{\text{sync}} / M$ |
| Variance | $\text{Var}[\hat{S}_B] = S_B(1-S_B)/M$ |
| 95% CI | $\hat{S}_B \pm 1.96\sqrt{\hat{S}_B(1-\hat{S}_B)/M}$ |
| Required $M$ for error $\epsilon$ | $M \geq S_B(1-S_B)/\epsilon^2$ |

Worst case ($S_B = 0.5$): need $M = 2500$ for 1% precision.

---

## 2. Theoretical Context

### 2.1 Historical Background

Basin stability was introduced by Menck, Heitzig, Marwan & Kurths (2013)
in *Nature Physics* as a nonlinear complement to linear stability analysis.
The motivation: conventional Lyapunov exponents and eigenvalue analysis
only probe infinitesimal perturbations, while real systems face
finite-amplitude disturbances.

Ji et al. (2014) extended the concept to multiple basins and applied it
to power grid stability, showing that network topology strongly affects
$S_B$ even when all nodes are locally stable.

### 2.2 Role in SCPN

Basin stability serves three purposes in the SCPN framework:

1. **Robustness assessment** — $S_B$ quantifies how "safe" the current
   operating regime is. A coherent state with $S_B = 0.95$ is far more
   robust than one with $S_B = 0.55$.

2. **Coupling calibration validation** — after setting $K_{ij}$, basin
   stability confirms that the synchronised state has adequate robustness
   ($S_B > 0.8$ is the SCPN target).

3. **Chimera detection** — multi-basin analysis reveals if chimera states
   coexist with synchronised states (both $S_B(\text{sync})$ and
   $S_B(\text{partial})$ positive).

### 2.3 Comparison with Other Stability Measures

| Measure | Type | Perturbation size | Cost |
|---------|------|-------------------|------|
| Eigenvalue (λ_max) | Local, linear | Infinitesimal | $O(N^3)$ one-shot |
| Lyapunov exponent | Local, nonlinear | Infinitesimal | $O(T \cdot N^2)$ |
| Basin stability | Global, nonlinear | Finite | $O(M \cdot T \cdot N^2)$ |

Basin stability is the most expensive but the most physically meaningful
for finite perturbations.

### 2.4 Applications Beyond Oscillators

Basin stability has found wide application:

- **Power grids** (Menck et al. 2014) — $S_B$ of each generator node
  predicts vulnerability to cascading failures. Dead-end nodes (degree 1)
  have systematically lower $S_B$.
- **Neural networks** — $S_B$ of attractor states in Hopfield networks
  measures memory capacity.
- **Climate tipping points** — $S_B$ of climate states quantifies resilience
  to perturbations.
- **SCPN layers** — each layer's coherent state has its own $S_B$. Low
  $S_B$ layers are candidates for additional coupling or control.

### 2.5 Limitations

- **Curse of dimensionality:** uniform sampling in $[0, 2\pi)^N$ becomes
  increasingly inefficient for large $N$. For $N > 100$, importance sampling
  or adaptive methods are preferred (not currently implemented).
- **Threshold dependence:** $S_B$ depends on $R_{\text{thresh}}$ choice.
  Use `multi_basin_stability` to explore sensitivity.
- **Transient length:** near $K_c$, convergence is slow ($O(1/|K-K_c|)$
  time). Insufficient `n_transient` underestimates $S_B$.
- **Ergodicity assumption:** the method assumes uniform sampling is
  representative. For systems with very narrow basins embedded in
  high-dimensional space, importance sampling would be more efficient.
- **No trajectory information:** $S_B$ tells you how large the basin is,
  but not where it is or what its geometry looks like. For that, combine
  with Lyapunov exponent analysis.

### 2.6 Interpretation Guide

| $S_B$ value | Interpretation | SCPN action |
|-------------|----------------|-------------|
| $S_B > 0.95$ | Extremely robust — almost all ICs synchronise | Safe operating regime |
| $0.8 < S_B < 0.95$ | Robust — occasional outlier ICs fail | Acceptable for production |
| $0.5 < S_B < 0.8$ | Moderately robust — significant fraction fails | Increase coupling or add drive |
| $0.2 < S_B < 0.5$ | Weak — most ICs do not synchronise | Near K_c, regime unstable |
| $S_B < 0.2$ | Very weak or no synchronisation | Below K_c or frustrated coupling |

### 2.7 Numerical Considerations

**Burn-in sensitivity.** If `n_transient` is too short, transient dynamics
inflate $\bar{R}$ for trials that would eventually desynchronise. Rule of
thumb: `n_transient` should be at least $10 / \Delta t = 1000$ for dt=0.01.

**Measurement noise.** Time-averaging over `n_measure` steps reduces
fluctuations but doesn't eliminate them. For small $N$ (< 16), $R$ fluctuates
by $\sim 1/\sqrt{N}$ even in steady state. Set threshold conservatively:
$R_{\text{thresh}} > 1/\sqrt{N} + 3\sigma$ where $\sigma$ is the expected
fluctuation.

**Parallel reproducibility.** With Rust backend, trial ordering in Rayon may
differ across runs (non-deterministic scheduling). However, each trial's
result is deterministic given its seed-derived initial condition. The
aggregate $S_B$ is identical across runs.

---

## 3. Pipeline Position

```
┌──────────────┐     ┌────────────────────┐     ┌──────────────┐
│ coupling/    │────→│ basin_stability()  │────→│ BasinStab    │
│ knm.py       │     │                    │     │ Result       │
│ (K_ij)       │     │ M Monte Carlo runs │     │  .S_B        │
└──────────────┘     │ Each: init → run → │     │  .R_final    │
                     │ measure R          │     │  .n_converged│
┌──────────────┐     │                    │     └──────┬───────┘
│ oscillators/ │────→│ omegas             │            │
│ base.py      │     └────────────────────┘     ┌──────▼───────┐
└──────────────┘                                │ supervisor/  │
                     ┌────────────────────┐     │ regimes.py   │
                     │ multi_basin_       │     │ (robustness  │
                     │ stability()        │────→│  assessment) │
                     │ Multiple thresholds│     └──────────────┘
                     └────────────────────┘
```

**Inputs:**
- `omegas` (N,) — natural frequencies
- `knm` (N, N) — coupling matrix
- `alpha` (N, N) — phase-lag matrix (optional)
- `n_samples` — Monte Carlo sample count $M$
- `R_threshold` — sync classification threshold

**Outputs:**
- `BasinStabilityResult` with $S_B$, per-trial $R$ values, counts

---

## 4. Features

### 4.1 Single and Multi-Threshold

| Function | Purpose | Output |
|----------|---------|--------|
| `basin_stability()` | Single threshold | One `BasinStabilityResult` |
| `multi_basin_stability()` | Multiple thresholds | Dict of results per threshold |

### 4.2 Rust Acceleration

Rust backend (`basin_stability_rust`) parallelises Monte Carlo trials
with Rayon. Each trial is independent — embarrassingly parallel. Speedup
scales linearly with core count up to `n_samples`.

### 4.3 Reproducibility

All functions accept `seed` parameter. Initial conditions are generated
from `np.random.default_rng(seed)`, ensuring identical results across runs.

### 4.4 Per-Trial R Values

`BasinStabilityResult.R_final` returns the full $(M,)$ array of final
order parameters — not just the aggregate $S_B$. This enables:
- Histogramming $R$ distribution
- Detecting bimodality (multiple attractors)
- Computing confidence intervals

---

## 5. Usage Examples

### 5.1 Basic Basin Stability

```python
import numpy as np
from scpn_phase_orchestrator.upde.basin_stability import basin_stability

N = 32
rng = np.random.default_rng(0)
omegas = rng.standard_cauchy(N) * 0.3
knm = np.full((N, N), 3.0 / N)
np.fill_diagonal(knm, 0.0)

result = basin_stability(omegas, knm, n_samples=200, R_threshold=0.8)
print(f"S_B = {result.S_B:.3f} ({result.n_converged}/{result.n_samples})")
# Above K_c: expect S_B > 0.8
```

### 5.2 Multi-Basin Analysis

```python
from scpn_phase_orchestrator.upde.basin_stability import multi_basin_stability

results = multi_basin_stability(
    omegas, knm,
    R_thresholds=(0.3, 0.6, 0.8),
    n_samples=500,
)
for label, res in results.items():
    print(f"{label}: S_B = {res.S_B:.3f}")
# If S_B(0.3) >> S_B(0.8): partial sync is more common than full sync
```

### 5.3 S_B vs K Sweep

```python
from scpn_phase_orchestrator.upde.basin_stability import basin_stability

K_values = np.linspace(0, 5, 20)
S_B_values = []
for K in K_values:
    knm_scaled = np.full((N, N), K / N)
    np.fill_diagonal(knm_scaled, 0.0)
    res = basin_stability(omegas, knm_scaled, n_samples=100)
    S_B_values.append(res.S_B)

# S_B should jump from ~0 to ~1 near K_c
```

### 5.4 Confidence Interval

```python
result = basin_stability(omegas, knm, n_samples=1000)
se = np.sqrt(result.S_B * (1 - result.S_B) / result.n_samples)
print(f"S_B = {result.S_B:.3f} ± {1.96*se:.3f} (95% CI)")
```

### 5.5 R Distribution Histogram

```python
result = basin_stability(omegas, knm, n_samples=500)
# result.R_final is (500,) array of final R values

import matplotlib.pyplot as plt
plt.hist(result.R_final, bins=30, edgecolor="black")
plt.axvline(result.R_threshold, color="r", ls="--", label="threshold")
plt.xlabel("Final R")
plt.ylabel("Count")
plt.title(f"Basin Stability: S_B = {result.S_B:.2f}")
plt.legend()
```

### 5.6 Frustrated Coupling

```python
alpha = np.full((N, N), np.pi / 4)
np.fill_diagonal(alpha, 0.0)

result = basin_stability(omegas, knm, alpha=alpha, n_samples=200)
print(f"Frustrated S_B = {result.S_B:.3f}")
# Frustration reduces S_B (smaller basin of attraction)
```

### 5.7 Bimodality Detection

```python
result = basin_stability(omegas, knm, n_samples=500, R_threshold=0.5)

# Check for bimodal R distribution (two attractors)
R_low = result.R_final[result.R_final < 0.3]
R_high = result.R_final[result.R_final > 0.7]
R_mid = result.R_final[(result.R_final >= 0.3) & (result.R_final <= 0.7)]

print(f"Low basin: {len(R_low)/result.n_samples:.1%}")
print(f"Mid basin: {len(R_mid)/result.n_samples:.1%}")
print(f"High basin: {len(R_high)/result.n_samples:.1%}")
# If both Low and High are substantial: multistability detected
```

### 5.8 Comparing Topologies

```python
# Compare all-to-all vs ring coupling
knm_aa = np.full((N, N), 3.0 / N)
np.fill_diagonal(knm_aa, 0.0)

knm_ring = np.zeros((N, N))
for i in range(N):
    knm_ring[i, (i+1) % N] = 3.0
    knm_ring[i, (i-1) % N] = 3.0

res_aa = basin_stability(omegas, knm_aa, n_samples=200)
res_ring = basin_stability(omegas, knm_ring, n_samples=200)
print(f"All-to-all S_B = {res_aa.S_B:.3f}")
print(f"Ring S_B = {res_ring.S_B:.3f}")
# All-to-all typically has higher S_B (stronger global coupling)
```

### 5.9 Per-Layer SCPN Analysis

```python
# Assess each SCPN layer's basin stability independently
from scpn_phase_orchestrator.coupling.knm import build_knm

N_per_layer = 4
N_total = 16  # 4 layers × 4 oscillators
knm_full = build_knm(N_total, template="scpn_default")

for layer_idx in range(4):
    start = layer_idx * N_per_layer
    end = start + N_per_layer
    knm_layer = knm_full[start:end, start:end]
    omegas_layer = omegas[start:end]
    res = basin_stability(omegas_layer, knm_layer, n_samples=100)
    print(f"Layer {layer_idx}: S_B = {res.S_B:.3f}")
```

---

## 6. Technical Reference

### 6.1 Module API

::: scpn_phase_orchestrator.upde.basin_stability
    options:
        show_root_heading: true
        members_order: source

### 6.2 Function Signatures

**`basin_stability(omegas, knm, alpha, dt, n_transient, n_measure, n_samples, R_threshold, seed) → BasinStabilityResult`**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `omegas` | `NDArray` (N,) | — | Natural frequencies |
| `knm` | `NDArray` (N,N) | — | Coupling matrix |
| `alpha` | `NDArray` (N,N) or None | zeros | Phase-lag matrix |
| `dt` | `float` | `0.01` | Integration timestep |
| `n_transient` | `int` | `500` | Transient steps |
| `n_measure` | `int` | `200` | Measurement steps |
| `n_samples` | `int` | `100` | Monte Carlo samples $M$ |
| `R_threshold` | `float` | `0.8` | Sync classification threshold |
| `seed` | `int` | `42` | RNG seed |

**`multi_basin_stability(omegas, knm, alpha, dt, n_transient, n_measure, n_samples, R_thresholds, seed) → dict[str, BasinStabilityResult]`**

Same parameters plus `R_thresholds: tuple[float, ...]` (default `(0.3, 0.6, 0.8)`).
Returns dict with keys like `"R>=0.30"`, `"R>=0.60"`, `"R>=0.80"`.

### 6.3 Data Class

**`BasinStabilityResult`**

| Field | Type | Description |
|-------|------|-------------|
| `S_B` | `float` | Basin stability $\in [0, 1]$ |
| `n_samples` | `int` | Total Monte Carlo samples |
| `n_converged` | `int` | Samples with $\bar{R} \geq R_{\text{thresh}}$ |
| `R_final` | `NDArray` (M,) | Per-trial final order parameter |
| `R_threshold` | `float` | Threshold used |

---

## 7. Performance Benchmarks

### 7.1 Cost Model

Total cost: $M \times (n_t + n_m) \times O(N^2)$ FLOPs.

| N | M | n_transient | n_measure | Python | Rust | Speedup |
|---|---|-------------|-----------|--------|------|---------|
| 16 | 100 | 500 | 200 | ~1.5s | ~0.15s | ~10x |
| 32 | 100 | 500 | 200 | ~5s | ~0.5s | ~10x |
| 64 | 100 | 500 | 200 | ~20s | ~2s | ~10x |
| 64 | 1000 | 500 | 200 | ~200s | ~18s | ~11x |

Rust speedup is primarily from Rayon parallelisation over $M$ trials
(each trial is independent).

### 7.2 Accuracy vs Samples

| M (n_samples) | SE(S_B) worst case | Time (N=32, Rust) |
|---------------|--------------------|--------------------|
| 50 | 0.071 | 0.08s |
| 100 | 0.050 | 0.15s |
| 500 | 0.022 | 0.7s |
| 1000 | 0.016 | 1.4s |
| 5000 | 0.007 | 7s |

For SCPN calibration, $M = 100$ provides 5% precision — sufficient for
go/no-go decisions. For publication, use $M = 1000+$.

### 7.3 Complexity

| Function | Time | Space |
|----------|------|-------|
| `basin_stability` | $O(M \cdot (n_t+n_m) \cdot N^2)$ | $O(M + N^2)$ |
| `multi_basin_stability` | same (single run, multiple thresholds) | $O(M + N^2)$ |

### 7.4 Scaling with N

The per-trial cost is $O(N^2)$ per step (coupling computation). For
large networks:

| N | M=100 time (Rust) | Per-trial (ms) | Bottleneck |
|---|-------------------|----------------|------------|
| 16 | 0.15s | 1.5 | RNG + loop overhead |
| 64 | 2.0s | 20 | Coupling $O(N^2)$ |
| 256 | 35s | 350 | Coupling $O(N^2)$ |
| 1024 | ~500s | 5000 | Memory bandwidth |

For $N > 256$, consider reducing `n_transient` or using sparse coupling
(`SparseEngine`-compatible topology) to reduce per-step cost.

### 7.5 Memory

| N | M | R_final array | Scratch per trial | Peak |
|---|---|---------------|-------------------|------|
| 16 | 100 | 0.8 KB | 2 KB | ~3 KB |
| 64 | 100 | 0.8 KB | 32 KB | ~33 KB |
| 256 | 1000 | 8 KB | 512 KB | ~520 KB |
| 1024 | 100 | 0.8 KB | 8 MB | ~8 MB |

Memory is dominated by the $N \times N$ scratch arrays during integration,
not by the $M$-element result array.

### 7.6 Recommended Settings

| Use case | M | n_transient | n_measure | Estimated time (N=32, Rust) |
|----------|---|-------------|-----------|-------------------------------|
| Quick check | 50 | 300 | 100 | 0.04s |
| SCPN calibration | 100 | 500 | 200 | 0.15s |
| Publication | 1000 | 2000 | 500 | 4s |
| High precision | 5000 | 5000 | 1000 | 40s |

---

## 8. Citations

1. **Menck P.J., Heitzig J., Marwan N., Kurths J.** (2013). How basin
   stability complements the linear-stability paradigm. *Nature Physics*
   **9**(2):89–92. doi:10.1038/nphys2516

2. **Ji P., Peron T.K.D., Menck P.J., Rodrigues F.A., Kurths J.** (2014).
   Cluster explosive synchronization in complex networks. *Scientific
   Reports* **4**:4783. doi:10.1038/srep04783

3. **Kuramoto Y.** (1984). *Chemical Oscillations, Waves, and Turbulence*.
   Springer-Verlag. doi:10.1007/978-3-642-69689-3

4. **Menck P.J., Heitzig J., Kurths J., Schellnhuber H.J.** (2014).
   How dead ends undermine power grid stability. *Nature Communications*
   **5**:3969. doi:10.1038/ncomms4969

5. **Schultz P., Heitzig J., Kurths J.** (2014). Detours around basin
   stability in power networks. *New Journal of Physics*
   **16**(12):125001. doi:10.1088/1367-2630/16/12/125001

---

## Test Coverage

- `tests/test_basin_stability.py` — 7 tests: S_B bounds [0,1], strong
  coupling S_B→1, weak coupling S_B→0, multi_basin keys, R_final shape,
  seed reproducibility, custom threshold
- `tests/test_prop_basin_stability.py` — 17 property tests (Hypothesis):
  S_B always in [0,1], n_converged ≤ n_samples, R_final shape matches
  n_samples, deterministic with same seed

Total: **24 tests**.

---

## Source

- Python: `src/scpn_phase_orchestrator/upde/basin_stability.py` (257 lines)
- Rust: `spo-kernel/crates/spo-engine/src/basin_stability.rs`
- FFI: `spo-kernel/crates/spo-ffi/src/lib.rs` (basin_stability_rust)
