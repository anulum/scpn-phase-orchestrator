# Bifurcation Analysis — Synchronisation Phase Transitions

The `bifurcation` module traces the Kuramoto synchronisation transition:
how the order parameter $R$ changes as coupling strength $K$ increases from
zero (incoherent) through the critical point $K_c$ into partial or full
synchronisation. It provides numerical continuation of steady-state $R(K)$
and binary-search estimation of $K_c$.

This is the primary tool for characterising the dynamics of a Kuramoto
network — answering "at what coupling does this system synchronise?"

---

## 1. Mathematical Formalism

### 1.1 The Synchronisation Transition

For the Kuramoto model with natural frequency distribution $g(\omega)$,
the system exhibits a second-order phase transition at critical coupling
$K_c$. Below $K_c$, the only stable state is incoherence ($R = 0$).
Above $K_c$, a partially synchronised state ($R > 0$) emerges.

**Analytical result (Kuramoto 1975):** For a symmetric unimodal distribution
$g(\omega)$ with all-to-all coupling $K_{ij} = K/N$:

$$
K_c = \frac{2}{\pi g(0)}
$$

For a Lorentzian distribution $g(\omega) = \frac{\Delta/\pi}{\omega^2 + \Delta^2}$
with half-width $\Delta$:

$$
K_c = 2\Delta
$$

Above $K_c$, the order parameter grows as:

$$
R = \sqrt{1 - \frac{K_c}{K}} \quad \text{for } K > K_c
$$

This square-root scaling is characteristic of a supercritical pitchfork
bifurcation — the same universality class as the ferromagnetic transition
in the Ising model.

### 1.2 Steady-State R Computation

The module computes $R(K)$ by direct numerical simulation. For each coupling
value $K$:

1. Initialise $N$ oscillators with random phases $\theta_i \sim U(0, 2\pi)$
2. Integrate the UPDE for $n_{\text{transient}}$ steps (discard transient)
3. Continue for $n_{\text{measure}}$ steps, computing $R$ at each step
4. Return $\bar{R} = \frac{1}{n_{\text{measure}}} \sum_{t} R(t)$

The Euler integrator is used:

$$
\theta_i^{n+1} = \theta_i^n + \Delta t \left(
\omega_i + K \sum_j \frac{1}{N} \sin(\theta_j^n - \theta_i^n - \alpha_{ij})
\right)
$$

### 1.3 Critical Coupling Detection

Two methods are provided:

**Sweep method** (`trace_sync_transition`): Evaluates $R(K)$ at $n_{\text{points}}$
uniformly spaced values in $[K_{\min}, K_{\max}]$. Detects $K_c$ as the
interpolated crossing of the $R = 0.1$ threshold:

$$
K_c \approx K_i + \frac{0.1 - R_i}{R_{i+1} - R_i}(K_{i+1} - K_i)
$$

where $R_i < 0.1 \leq R_{i+1}$ is the first threshold crossing.

**Binary search** (`find_critical_coupling`): Bisects $[0, 20]$ to find
$K_c$ to tolerance $\delta$. At each step, evaluates $R(K_{\text{mid}})$
and narrows the interval. Converges in $\log_2(20/\delta)$ iterations.

### 1.4 Hysteresis and Multistability

For finite $N$, the transition can exhibit hysteresis — $R$ at a given $K$
depends on whether $K$ is increasing or decreasing. The module always sweeps
from the incoherent side ($K$ increasing) using fixed initial conditions
(seeded RNG), making results reproducible but potentially missing the
upper branch of a hysteretic transition.

For frustrated coupling ($\alpha \neq 0$), multiple stable states can coexist.
The sweep captures the basin that the random initial condition falls into.

---

## 2. Theoretical Context

### 2.1 Historical Background

The Kuramoto synchronisation transition was first predicted analytically by
Kuramoto (1975) using a self-consistency argument. The rigorous derivation
of $K_c$ and the bifurcation structure was given by Strogatz (2000) via
the Ott-Antonsen ansatz. Crawford (1994) provided a centre-manifold analysis
showing the pitchfork nature of the bifurcation.

Keller (1977) developed the pseudo-arclength continuation method for
tracking solution branches through bifurcation points. While SPO currently
uses simpler sweep/bisection methods, the module's docstring references
Keller for future extension to full numerical continuation.

### 2.2 Role in SCPN

Bifurcation analysis serves two purposes in the SCPN framework:

1. **Calibration** — Determining $K_c$ for a given frequency distribution
   validates that the coupling matrix $K_{ij}$ is correctly scaled. If the
   numerical $K_c$ deviates from the analytical prediction by more than 5%,
   the coupling calibration is suspect.

2. **Regime mapping** — The $R(K)$ curve defines operating regimes:
   - $K < K_c$: incoherent (regime "desync")
   - $K_c < K < 2K_c$: partial sync (regime "partial")
   - $K > 2K_c$: strong sync (regime "coherent")

   The supervisor uses these thresholds for automatic regime classification.

### 2.3 Finite-Size Effects

For finite $N$, the transition is smoothed:
- $R(K=0) \sim 1/\sqrt{N}$ (random fluctuations), not exactly 0
- The threshold $R = 0.1$ is chosen empirically as a compromise between
  sensitivity and finite-size noise
- For $N = 16$ (SCPN layers), the transition is broad and $K_c$ estimates
  have ~10% uncertainty

The binary search method (`find_critical_coupling`) provides better
precision than the sweep by using 30 bisection iterations, achieving
$\delta K \approx 20/2^{30} \approx 2 \times 10^{-8}$ theoretical
precision (limited in practice by simulation noise).

### 2.4 Frequency Distributions and K_c

The analytical $K_c$ depends on the frequency distribution:

| Distribution | $g(\omega)$ | $K_c$ formula | Example ($\Delta=0.5$) |
|-------------|-------------|---------------|------------------------|
| Lorentzian | $\frac{\Delta/\pi}{\omega^2+\Delta^2}$ | $2\Delta$ | 1.0 |
| Gaussian | $\frac{1}{\sigma\sqrt{2\pi}}e^{-\omega^2/2\sigma^2}$ | $\frac{2}{\pi g(0)} = \sigma\sqrt{2\pi}$ | 1.25 |
| Uniform $[-a,a]$ | $\frac{1}{2a}$ | $\frac{4a}{\pi}$ | 0.64 |
| Identical ($\sigma=0$) | $\delta(\omega-\omega_0)$ | 0 (any $K>0$ syncs) | 0 |
| Bimodal | two peaks | No closed form | (numerical only) |

The Lorentzian is special: the Ott-Antonsen (2008) ansatz reduces the
infinite-dimensional system to a single ODE for $z(t)$, making the
bifurcation analytically tractable.

### 2.5 Beyond the Pitchfork

For heterogeneous coupling topologies ($K_{ij}$ not uniform):
- Scale-free networks have $K_c \propto \langle k \rangle / \langle k^2 \rangle$
  (Ichinomiya 2004, Restrepo et al. 2005)
- Small-world networks interpolate between lattice ($K_c$ large) and
  all-to-all ($K_c$ small)
- The SCPN block-diagonal structure creates a hierarchy of transitions:
  intra-layer sync first ($K_c^{\text{intra}}$), then inter-layer
  ($K_c^{\text{inter}} > K_c^{\text{intra}}$)

### 2.6 Numerical Considerations

**Transient length.** The system needs $O(1/|K - K_c|)$ steps to reach
steady state near the critical point. Default `n_transient=2000` with
`dt=0.01` gives 20 time units — sufficient for $|K - K_c| > 0.05$.
For precision near $K_c$, increase to 10,000+.

**Measurement averaging.** Finite-time fluctuations in $R$ scale as
$O(1/\sqrt{n_{\text{measure}}})$. Default `n_measure=500` gives ~4%
standard error. For publication-quality diagrams, use 5000+.

**Euler vs RK4.** The internal integrator uses Euler for speed.
For stiff systems or very large $K$, Euler may require smaller $dt$
than the default 0.01. The user cannot change the internal method —
the engine is hardcoded to Euler for the sweep.

**Initial condition sensitivity.** Near $K_c$, the system is sensitive
to initial conditions. The fixed seed (default 42) ensures
reproducibility, but different seeds may give different $K_c$ estimates
for small $N$. For robust $K_c$: average over multiple seeds.

---

## 3. Pipeline Position

```
┌──────────────┐     ┌────────────────────┐     ┌──────────────┐
│ coupling/    │────→│ trace_sync_        │────→│ Bifurcation  │
│ knm.py       │     │ transition()       │     │ Diagram      │
│ (K template) │     │                    │     │  .K_critical  │
└──────────────┘     │ Sweeps K in range  │     │  .R_values    │
                     │ Runs UPDE at each K│     │  .K_values    │
┌──────────────┐     │                    │     └──────┬───────┘
│ oscillators/ │────→│ omegas             │            │
│ base.py      │     └────────────────────┘     ┌──────▼───────┐
│ (frequencies)│                                │ supervisor/  │
└──────────────┘     ┌────────────────────┐     │ regimes.py   │
                     │ find_critical_     │     │ (calibration)│
                     │ coupling()         │     └──────────────┘
                     │ Binary search K_c  │
                     └────────────────────┘
```

**Inputs:**
- `omegas` (N,) — natural frequency distribution
- `knm_template` (N, N) — coupling topology (scaled by K during sweep)
- `alpha` (N, N) — phase-lag matrix (optional, default zeros)
- `K_range` — sweep bounds `(K_min, K_max)`

**Outputs:**
- `BifurcationDiagram` with list of `BifurcationPoint(K, R, stable)`
- `K_critical` — estimated critical coupling

---

## 4. Features

### 4.1 Sweep and Bisection

| Function | Purpose | Precision | Cost |
|----------|---------|-----------|------|
| `trace_sync_transition` | Full $R(K)$ curve | ~$\Delta K / 2$ | $n_{\text{points}} \times (n_t + n_m)$ steps |
| `find_critical_coupling` | $K_c$ only | ~$\delta$ (tolerance) | ~$30 \times (n_t + n_m)$ steps |

### 4.2 Rust Acceleration

Both functions delegate to Rust when available:
- `steady_state_r_rust` — single $(K, R)$ evaluation
- `trace_sync_transition_rust` — full sweep with parallel $K$ evaluation
- `find_critical_coupling_bif_rust` — binary search

The Rust sweep uses `par_iter` over $K$ values (Rayon), providing
near-linear speedup with core count for $n_{\text{points}} \geq 8$.

### 4.3 Reproducibility

All functions accept a `seed` parameter for the RNG. Default seed is 42.
Initial phases are generated once and reused for all $K$ values in a sweep,
ensuring that variations in $R(K)$ are due to coupling strength, not
initial conditions.

### 4.4 Data Classes

**`BifurcationPoint`**: `(K: float, R: float, stable: bool)`

**`BifurcationDiagram`**:
- `.points` — list of `BifurcationPoint`
- `.K_critical` — estimated $K_c$ (or `None` if no transition detected)
- `.K_values` — property returning `NDArray` of all K values
- `.R_values` — property returning `NDArray` of all R values

---

## 5. Usage Examples

### 5.1 Basic Bifurcation Diagram

```python
import numpy as np
from scpn_phase_orchestrator.upde.bifurcation import trace_sync_transition

# 64 oscillators with Lorentzian frequencies (Δ = 0.5)
rng = np.random.default_rng(0)
omegas = rng.standard_cauchy(64) * 0.5  # half-width Δ = 0.5

diagram = trace_sync_transition(
    omegas,
    K_range=(0.0, 5.0),
    n_points=50,
    n_transient=3000,
    n_measure=1000,
)

print(f"K_c = {diagram.K_critical:.3f}")
# Analytical: K_c = 2Δ = 1.0
# Numerical: K_c ≈ 1.0 ± 0.1 (finite-size)
```

### 5.2 Precise K_c via Binary Search

```python
from scpn_phase_orchestrator.upde.bifurcation import find_critical_coupling

K_c = find_critical_coupling(
    omegas,
    tol=0.01,
    n_transient=5000,
    n_measure=2000,
)
print(f"K_c = {K_c:.4f}")  # precision ~ 0.01
```

### 5.3 Custom Coupling Topology

```python
# Ring coupling instead of all-to-all
N = 64
knm_ring = np.zeros((N, N))
for i in range(N):
    knm_ring[i, (i + 1) % N] = 1.0
    knm_ring[i, (i - 1) % N] = 1.0

diagram = trace_sync_transition(
    omegas,
    knm_template=knm_ring,
    K_range=(0.0, 10.0),
    n_points=100,
)
print(f"Ring K_c = {diagram.K_critical:.3f}")
# Ring coupling requires larger K than all-to-all
```

### 5.4 Frustrated Coupling (Sakaguchi)

```python
# Phase-lag α = π/6 shifts and raises K_c
alpha = np.full((N, N), np.pi / 6)
np.fill_diagonal(alpha, 0.0)

diagram = trace_sync_transition(
    omegas,
    alpha=alpha,
    K_range=(0.0, 8.0),
    n_points=80,
)
print(f"Frustrated K_c = {diagram.K_critical:.3f}")
```

### 5.5 Plotting the Diagram

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(diagram.K_values, diagram.R_values, "b-", lw=2)
if diagram.K_critical is not None:
    ax.axvline(diagram.K_critical, color="r", ls="--", label=f"$K_c$ = {diagram.K_critical:.2f}")
ax.set_xlabel("Coupling strength K")
ax.set_ylabel("Order parameter R")
ax.set_title("Kuramoto Synchronisation Transition")
ax.legend()
plt.savefig("bifurcation_diagram.png", dpi=150)
```

### 5.6 Validating Against Analytical K_c

```python
# Lorentzian with known Δ
Delta = 0.5
omegas_lorentzian = rng.standard_cauchy(256) * Delta
K_c_analytical = 2 * Delta

K_c_numerical = find_critical_coupling(omegas_lorentzian, tol=0.01)
error_pct = abs(K_c_numerical - K_c_analytical) / K_c_analytical * 100
print(f"Analytical K_c = {K_c_analytical:.3f}")
print(f"Numerical  K_c = {K_c_numerical:.3f}")
print(f"Error: {error_pct:.1f}%")
# Expect < 10% for N = 256
```

### 5.7 Multi-Seed Robust K_c Estimation

```python
# Average K_c over 10 seeds for robust estimate
K_c_values = []
for s in range(10):
    kc = find_critical_coupling(omegas, seed=s, tol=0.05)
    if not np.isnan(kc):
        K_c_values.append(kc)

K_c_mean = np.mean(K_c_values)
K_c_std = np.std(K_c_values)
print(f"K_c = {K_c_mean:.3f} ± {K_c_std:.3f} (N_seeds={len(K_c_values)})")
```

### 5.8 Interpreting Results

**$K_c$ much lower than expected:** Coupling matrix may be too dense.
Check that `knm_template` is correctly normalised (typically $K_{ij} = 1/N$
for all-to-all).

**$K_c$ much higher than expected:** Frequency spread may be wider than
assumed. Check `np.std(omegas)`.

**$K_c = \text{NaN}$:** No synchronisation transition detected in $[0, 20]$.
Possible causes: frequencies too spread ($K_c > 20$), coupling topology
too sparse, or frustration ($\alpha$) too large.

**$R(K)$ oscillates instead of monotone increase:** Possible chimera state
or metastability. Increase `n_transient` to let the system settle.

**Noisy $R(K)$ curve:** Increase `n_measure` (averaging window). For
$N < 32$, finite-size noise is inherent and cannot be eliminated.

---

## 6. Technical Reference

### 6.1 Module API

::: scpn_phase_orchestrator.upde.bifurcation
    options:
        show_root_heading: true
        members_order: source

### 6.2 Function Signatures

**`trace_sync_transition(omegas, knm_template, alpha, K_range, n_points, dt, n_transient, n_measure, seed) → BifurcationDiagram`**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `omegas` | `NDArray` (N,) | — | Natural frequencies |
| `knm_template` | `NDArray` (N,N) or None | all-to-all/N | Coupling topology |
| `alpha` | `NDArray` (N,N) or None | zeros | Phase-lag matrix |
| `K_range` | `tuple[float,float]` | `(0.0, 5.0)` | Sweep bounds |
| `n_points` | `int` | `50` | Number of K samples |
| `dt` | `float` | `0.01` | Integration timestep |
| `n_transient` | `int` | `2000` | Transient steps to discard |
| `n_measure` | `int` | `500` | Measurement steps for averaging |
| `seed` | `int` | `42` | RNG seed |

**`find_critical_coupling(omegas, knm_template, dt, n_transient, n_measure, tol, seed) → float`**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `omegas` | `NDArray` (N,) | — | Natural frequencies |
| `knm_template` | `NDArray` (N,N) or None | all-to-all/N | Coupling topology |
| `dt` | `float` | `0.01` | Integration timestep |
| `n_transient` | `int` | `3000` | Transient steps |
| `n_measure` | `int` | `1000` | Measurement steps |
| `tol` | `float` | `0.05` | Bisection tolerance on K |
| `seed` | `int` | `42` | RNG seed |
| **Returns** | `float` | | $K_c$ estimate. `NaN` if no transition in [0, 20] |

### 6.3 Data Classes

**`BifurcationPoint`**

| Field | Type | Description |
|-------|------|-------------|
| `K` | `float` | Coupling strength |
| `R` | `float` | Steady-state order parameter |
| `stable` | `bool` | Stability flag (always True in current impl) |

**`BifurcationDiagram`**

| Field/Property | Type | Description |
|----------------|------|-------------|
| `points` | `list[BifurcationPoint]` | All (K, R) data points |
| `K_critical` | `float or None` | Estimated critical coupling |
| `K_values` | `NDArray` (property) | Array of K values |
| `R_values` | `NDArray` (property) | Array of R values |

---

## 7. Performance Benchmarks

### 7.1 Sweep Cost

Each $R(K)$ evaluation requires $(n_t + n_m) \times O(N^2)$ floating-point
operations. Total cost for a full sweep:

$$
\text{Cost} = n_{\text{points}} \times (n_t + n_m) \times O(N^2)
$$

| N | n_points | n_transient | n_measure | Python time | Rust time | Speedup |
|---|----------|-------------|-----------|-------------|-----------|---------|
| 16 | 50 | 2000 | 500 | ~2.5s | ~0.2s | ~12x |
| 64 | 50 | 2000 | 500 | ~40s | ~3.5s | ~11x |
| 256 | 50 | 2000 | 500 | ~850s | ~60s | ~14x |

Rust speedup comes from:
- Sin/cos precomputation ($O(N)$ instead of $O(N^2)$)
- Rayon `par_iter` over K values (parallel sweep)
- Cache-friendly memory layout

### 7.2 Binary Search Cost

Binary search requires ~30 iterations × $(n_t + n_m)$ steps each.
For N = 64: Python ~25s, Rust ~2s.

### 7.3 Accuracy vs Cost Trade-off

| Setting | K_c precision | Time (N=64, Rust) |
|---------|---------------|-------------------|
| `n_t=1000, n_m=200, tol=0.5` | ~0.5 | 0.3s |
| `n_t=2000, n_m=500, tol=0.1` | ~0.1 | 1.5s |
| `n_t=5000, n_m=2000, tol=0.01` | ~0.01 | 8s |
| `n_t=10000, n_m=5000, tol=0.001` | ~0.005 | 30s |

For SCPN calibration, `tol=0.1` is sufficient.

### 7.4 Complexity

| Function | Time | Space |
|----------|------|-------|
| `_steady_state_R` | $O((n_t + n_m) \cdot N^2)$ | $O(N^2)$ |
| `trace_sync_transition` | $O(n_p \cdot (n_t + n_m) \cdot N^2)$ | $O(n_p + N^2)$ |
| `find_critical_coupling` | $O(30 \cdot (n_t + n_m) \cdot N^2)$ | $O(N^2)$ |

---

## 8. Citations

1. **Kuramoto Y.** (1975). Self-entrainment of a population of coupled
   non-linear oscillators. *International Symposium on Mathematical Problems
   in Theoretical Physics*, Lecture Notes in Physics **39**:420–422.

2. **Kuramoto Y.** (1984). *Chemical Oscillations, Waves, and Turbulence*.
   Springer-Verlag, Berlin. doi:10.1007/978-3-642-69689-3

3. **Strogatz S.H.** (2000). From Kuramoto to Crawford: exploring the
   onset of synchronization. *Physica D* **143**(1–4):1–20.
   doi:10.1016/S0167-2789(00)00094-4

4. **Ott E., Antonsen T.M.** (2008). Low dimensional behavior of large
   systems of globally coupled oscillators. *Chaos* **18**(3):037113.
   doi:10.1063/1.2930766

5. **Crawford J.D.** (1994). Amplitude expansions for instabilities in
   populations of globally-coupled oscillators. *Journal of Statistical
   Physics* **74**(5–6):1047–1084. doi:10.1007/BF02188217

6. **Keller H.B.** (1977). Numerical solution of bifurcation and nonlinear
   eigenvalue problems. In: *Applications of Bifurcation Theory*,
   pp. 359–384. Academic Press.

7. **Acebrón J.A. et al.** (2005). The Kuramoto model: A simple paradigm
   for synchronization phenomena. *Rev. Mod. Phys.* **77**(1):137–185.
   doi:10.1103/RevModPhys.77.137

8. **Ichinomiya T.** (2004). Frequency synchronization in a random
   oscillator network. *Physical Review E* **70**(2):026116.
   doi:10.1103/PhysRevE.70.026116

9. **Restrepo J.G., Ott E., Hunt B.R.** (2005). Onset of synchronization
   in large networks of coupled oscillators. *Physical Review E*
   **71**(3):036151. doi:10.1103/PhysRevE.71.036151

---

## Test Coverage

- `tests/test_bifurcation.py` — 16 tests: trace with identical omegas
  (R→1 for K>0), Lorentzian K_c within 20% of analytical, empty diagram,
  K_range boundaries, seed reproducibility, find_critical_coupling
  precision, custom knm_template, BifurcationPoint fields,
  BifurcationDiagram properties

Total: **16 tests**.

---

## Source

- Python: `src/scpn_phase_orchestrator/upde/bifurcation.py` (311 lines)
- Rust: `spo-kernel/crates/spo-engine/src/bifurcation.rs`
- FFI: `spo-kernel/crates/spo-ffi/src/lib.rs` (steady_state_r_rust,
  trace_sync_transition_rust, find_critical_coupling_bif_rust)
