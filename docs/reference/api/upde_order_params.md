# Order Parameters — Synchronisation Measurement

The `order_params` module provides the fundamental observables for
quantifying synchronisation in coupled oscillator networks: the Kuramoto
order parameter $R$, the phase-locking value (PLV), and per-layer
coherence. These are the most frequently called functions in the entire
SPO codebase — every simulation loop, every monitor, every supervisor
decision depends on them.

---

## 1. Mathematical Formalism

### 1.1 Kuramoto Order Parameter

The complex order parameter $z$ captures the macroscopic state of an
oscillator ensemble in a single complex number:

$$
z = R e^{i\psi} = \frac{1}{N} \sum_{j=1}^{N} e^{i\theta_j}
$$

The magnitude $R \in [0, 1]$ measures the degree of phase coherence:

| $R$ value | Interpretation |
|-----------|---------------|
| $R = 0$ | Fully incoherent — phases uniformly distributed on $S^1$ |
| $0 < R < 1$ | Partial synchronisation — some phase clustering |
| $R = 1$ | Perfect synchronisation — all phases identical |

The argument $\psi = \arg(z)$ gives the mean phase direction.

For $N \to \infty$ with uniformly distributed phases:
$R \sim 1/\sqrt{N}$ (finite-size fluctuation).
For identical oscillators above critical coupling:
$R = \sqrt{1 - K_c/K}$ (Ott-Antonsen 2008).

### 1.2 Phase-Locking Value (PLV)

The PLV quantifies the consistency of phase difference between two
time series $\phi_a(t)$ and $\phi_b(t)$:

$$
\text{PLV} = \left| \frac{1}{T} \sum_{t=1}^{T} e^{i(\phi_a(t) - \phi_b(t))} \right|
$$

Originally introduced by Lachaux et al. (1999) for EEG connectivity analysis.

| PLV value | Interpretation |
|-----------|---------------|
| $\text{PLV} = 0$ | No consistent phase relationship |
| $\text{PLV} = 1$ | Perfectly locked (constant phase difference) |

The PLV is invariant to the actual phase difference — it measures
consistency, not alignment. Two oscillators with a stable $\pi/3$
phase offset have PLV = 1.

**Important distinction:** $R$ measures instantaneous spatial coherence
across oscillators at one time point. PLV measures temporal consistency
of phase difference between two oscillators across many time points.

### 1.3 Layer Coherence

For hierarchical systems (SCPN 15+1 layers), per-layer coherence applies
the order parameter to a subset:

$$
R_\ell = \left| \frac{1}{N_\ell} \sum_{j \in \ell} e^{i\theta_j} \right|
$$

where $\ell$ denotes a layer and $N_\ell$ is the number of oscillators
in that layer. This allows monitoring intra-layer synchronisation
independently of cross-layer interactions.

### 1.4 Circular Statistics Connection

The order parameter is the first trigonometric moment of the circular
distribution (Mardia & Jupp 2000):

$$
R = \left| \overline{C}_1 + i\overline{S}_1 \right|
= \sqrt{\bar{C}^2 + \bar{S}^2}
$$

where $\bar{C} = \frac{1}{N}\sum\cos\theta_j$ and
$\bar{S} = \frac{1}{N}\sum\sin\theta_j$. This formulation avoids
complex arithmetic and is used in the Rust backend for efficiency.

### 1.5 Statistical Properties

For $N$ independent uniform phases on $[0, 2\pi)$:

- $\mathbb{E}[R] = \sqrt{\pi/(4N)}$ for large $N$ (Rayleigh distribution)
- $\text{Var}[R^2] = 1/N$
- Rayleigh test: reject uniformity at significance $p$ when
  $R^2 \cdot N > -\ln(p)$

These bounds set the baseline — any $R$ significantly above
$1/\sqrt{N}$ indicates genuine synchronisation.

### 1.6 Higher-Order Parameters

While SPO uses only the first-order parameter $R = R_1$, the theory
defines a family of Daido order parameters (Daido 1993):

$$
Z_k = R_k e^{ik\psi_k} = \frac{1}{N} \sum_{j=1}^{N} e^{ik\theta_j}
$$

$R_2$ detects two-cluster states (where $R_1 \approx 0$ but $R_2 \approx 1$).
$R_3$ detects three-cluster states. The SCPN chimera detector uses local
order parameters $R_i$ computed in a neighbourhood (see `monitor/chimera.py`).

### 1.7 PLV vs Coherence Measures

Different synchronisation measures capture different aspects:

| Measure | Type | What it captures |
|---------|------|------------------|
| $R$ (order parameter) | Spatial, instantaneous | Global phase clustering at one time |
| PLV | Temporal, pairwise | Phase-difference consistency over time |
| $R_\ell$ (layer coherence) | Spatial, subset | Intra-group synchronisation |
| Mutual information | Statistical | General statistical dependence |
| Transfer entropy | Directed | Directed information flow |
| Granger causality | Directed, linear | Linear predictive coupling |

SPO uses $R$ for global state, PLV for pairwise lock detection, and
$R_\ell$ for per-layer monitoring. The `monitor/` module provides the
more advanced measures (TE, MI, etc.) for research applications.

### 1.8 Finite-Size Corrections

For small $N$ (SCPN uses $N = 16$), the naive $R$ overestimates true
synchronisation because random fluctuations contribute:

$$
R_{\text{corrected}}^2 = R_{\text{measured}}^2 - \frac{1}{N}
$$

This correction (valid when $R^2 > 1/N$) removes the expected contribution
from finite-sample noise. SPO does NOT apply this correction automatically
— the user should apply it when interpreting results from small networks.
For $N = 16$: baseline $R_{\text{random}} \approx 0.25$.

---

## 2. Theoretical Context

### 2.1 History

The order parameter was introduced by Kuramoto (1975) by analogy with
magnetisation in spin systems. The key insight: the coupling term
in the Kuramoto model can be rewritten as:

$$
\sum_j K_{ij}\sin(\theta_j - \theta_i) = K R \sin(\psi - \theta_i)
$$

for all-to-all coupling $K_{ij} = K/N$. This shows that each oscillator
couples to the mean field $(R, \psi)$ rather than to all others
individually — the "mean-field" nature of the model.

### 2.2 Role in SPO

The order parameter serves three critical pipeline roles:

1. **State estimation** — `UPDEEngine.compute_order_parameter()` provides
   the primary observable after each integration step
2. **Regime classification** — `supervisor/regimes.py` uses $R$ thresholds
   to classify coherent, chimera, and incoherent regimes
3. **Actuation feedback** — `actuation/mapper.py` maps $R$ deviations
   from target into control actions on $K_{ij}$

### 2.3 PLV in Neuroscience

The PLV was developed by Lachaux et al. (1999) for measuring functional
connectivity from MEG/EEG recordings. In SPO, it serves as the
lock-quality metric in `LockSignature` objects, stored per layer-pair
in `UPDEState`. The supervisor uses PLV thresholds to detect locking
and unlocking events between SCPN layers.

---

## 3. Pipeline Position

```
┌─────────────┐     ┌────────────────────┐     ┌──────────────┐
│ UPDEEngine  │────→│ compute_order_     │────→│ UPDEState    │
│ step() out  │     │ parameter(phases)  │     │ .layers[].R  │
│ phases (N,) │     │ → (R, ψ)          │     │ .stability   │
└─────────────┘     └────────────────────┘     └──────┬───────┘
                                                      │
                    ┌────────────────────┐     ┌──────▼───────┐
                    │ compute_plv        │────→│ LockSignature│
                    │ (φ_a, φ_b) → PLV  │     │ .plv         │
                    └────────────────────┘     └──────────────┘

                    ┌────────────────────┐     ┌──────────────┐
                    │ compute_layer_     │────→│ RegimeManager│
                    │ coherence          │     │ per-layer R  │
                    │ (phases, mask)→R_ℓ │     └──────────────┘
                    └────────────────────┘
```

**Callers (verified via grep across codebase):**

| Function | Called by | Context |
|----------|----------|---------|
| `compute_order_parameter` | `UPDEEngine`, `cli.py`, 47 test files, 14 notebooks | Primary observable |
| `compute_plv` | `monitor/coherence.py`, `supervisor/policy_rules.py`, 12 test files | Lock detection |
| `compute_layer_coherence` | `cli.py`, `supervisor/regimes.py`, 5 test files | Per-layer R |

---

## 4. Features

### 4.1 Rust Acceleration

Both `compute_order_parameter` and `compute_plv` have Rust backends via
`spo_kernel.order_parameter` and `spo_kernel.plv`. Selection is automatic
when `spo-kernel` is installed. The Rust implementation uses SIMD-friendly
sin/cos computation and avoids complex number overhead.

### 4.2 Empty Array Handling

All three functions gracefully handle empty inputs:
- `compute_order_parameter([])` → `(0.0, 0.0)`
- `compute_plv([], [])` → `0.0`
- `compute_layer_coherence(phases, empty_mask)` → `0.0`

No exceptions for degenerate inputs — the caller gets a sensible default.

### 4.3 Numerical Stability

`compute_order_parameter` uses `np.errstate(invalid="ignore")` to suppress
warnings when all phases are identical (the complex mean is exactly real,
`np.angle` returns 0.0 without spurious NaN warnings).

### 4.4 Phase Wrapping

The returned mean phase $\psi$ is wrapped to $[0, 2\pi)$ via `% TWO_PI`.
This matches the convention used throughout SPO (phases are always
non-negative).

---

## 5. Usage Examples

### 5.1 Basic Order Parameter

```python
import numpy as np
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

# Fully synchronised (all same phase)
R, psi = compute_order_parameter(np.array([1.0, 1.0, 1.0, 1.0]))
print(f"Synchronised: R={R:.4f}, ψ={psi:.4f}")  # R=1.0

# Fully incoherent (uniformly spread)
phases = np.linspace(0, 2 * np.pi, 100, endpoint=False)
R, psi = compute_order_parameter(phases)
print(f"Uniform: R={R:.6f}")  # R ≈ 0.0 (finite-size: ~0.01)

# Two clusters at 0 and π
phases = np.array([0.0, 0.0, 0.0, np.pi, np.pi, np.pi])
R, psi = compute_order_parameter(phases)
print(f"Two clusters: R={R:.4f}")  # R ≈ 0.0 (clusters cancel)
```

### 5.2 Phase-Locking Value

```python
from scpn_phase_orchestrator.upde.order_params import compute_plv

# Perfect locking (constant phase difference π/3)
T = 1000
phi_a = np.linspace(0, 20 * np.pi, T)
phi_b = phi_a + np.pi / 3  # constant offset
plv = compute_plv(phi_a, phi_b)
print(f"Locked PLV = {plv:.4f}")  # 1.0

# No locking (independent random walks)
rng = np.random.default_rng(42)
phi_a = np.cumsum(rng.normal(0, 0.1, T))
phi_b = np.cumsum(rng.normal(0, 0.1, T))
plv = compute_plv(phi_a, phi_b)
print(f"Random PLV = {plv:.4f}")  # ≈ 0.03
```

### 5.3 Per-Layer Coherence

```python
from scpn_phase_orchestrator.upde.order_params import compute_layer_coherence

# 16 oscillators, first 8 synchronised, last 8 random
phases = np.zeros(16)
phases[8:] = rng.uniform(0, 2 * np.pi, 8)

layer_1 = np.array([True]*8 + [False]*8)
layer_2 = np.array([False]*8 + [True]*8)

R1 = compute_layer_coherence(phases, layer_1)
R2 = compute_layer_coherence(phases, layer_2)
print(f"Layer 1 (synced): R={R1:.4f}")  # R ≈ 1.0
print(f"Layer 2 (random): R={R2:.4f}")  # R ≈ 0.3
```

### 5.4 Monitoring Loop

```python
from scpn_phase_orchestrator.upde.engine import UPDEEngine

engine = UPDEEngine(16, dt=0.01, method="rk4")
R_history = []
for step in range(500):
    phases = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)
    R, _ = compute_order_parameter(phases)
    R_history.append(R)

# R_history traces synchronisation onset over time
```

### 5.5 Rayleigh Test for Significance

```python
N = 64
phases_random = rng.uniform(0, 2 * np.pi, N)
R, _ = compute_order_parameter(phases_random)

# Rayleigh test: is R significant?
p_value = np.exp(-N * R**2)
print(f"R={R:.4f}, p={p_value:.4f}")
if p_value < 0.05:
    print("Significant synchronisation")
else:
    print("Consistent with random phases")
```

### 5.6 Synchronisation Onset Detection

```python
# Detect K_c: sweep coupling and find where R jumps
from scpn_phase_orchestrator.upde.engine import UPDEEngine

N = 64
omegas = rng.standard_cauchy(N)  # Lorentzian distribution
alpha = np.zeros((N, N))

K_values = np.linspace(0, 5, 50)
R_steady = []

for K_val in K_values:
    knm = np.full((N, N), K_val / N)
    np.fill_diagonal(knm, 0.0)
    eng = UPDEEngine(N, dt=0.01, method="rk4")
    p = rng.uniform(0, 2 * np.pi, N)
    p = eng.run(p, omegas, knm, 0.0, 0.0, alpha, n_steps=2000)
    R, _ = compute_order_parameter(p)
    R_steady.append(R)

# K_c is where R first exceeds baseline
K_c_idx = next(i for i, R in enumerate(R_steady) if R > 2 / np.sqrt(N))
print(f"Estimated K_c ≈ {K_values[K_c_idx]:.2f}")
```

### 5.7 PLV Matrix for Multi-Layer Analysis

```python
# Compute all pairwise PLVs from a trajectory
n_layers = 4
n_steps = 500
trajectories = np.zeros((n_steps, n_layers))

# (assume trajectories filled from simulation)
plv_matrix = np.zeros((n_layers, n_layers))
for i in range(n_layers):
    for j in range(i + 1, n_layers):
        plv = compute_plv(trajectories[:, i], trajectories[:, j])
        plv_matrix[i, j] = plv
        plv_matrix[j, i] = plv

print("PLV matrix:")
print(plv_matrix)
```

### 5.8 Finite-Size Correction

```python
N = 16
R_raw, psi = compute_order_parameter(phases)
R_corrected = np.sqrt(max(0, R_raw**2 - 1/N))
print(f"Raw R={R_raw:.4f}, corrected R={R_corrected:.4f}")
print(f"Baseline (random): {1/np.sqrt(N):.4f}")
```

### 5.9 Edge Cases and Common Pitfalls

**Single oscillator:** `compute_order_parameter(np.array([θ]))` returns
`(1.0, θ)` — a single oscillator is always perfectly synchronised with
itself. This is mathematically correct but can mislead if not expected.

**Two anti-phase oscillators:** `[0, π]` gives `R = 0.0` — they cancel
exactly. `ψ` is undefined (numerically: 0.0 or π depending on precision).
Do not rely on `ψ` when `R ≈ 0`.

**Phase wrapping artefacts:** If phases are near 0 and 2π (e.g.,
`[0.01, 6.27]`), they are actually close but naive subtraction gives
~6.26. The order parameter handles this correctly (via $e^{i\theta}$),
but direct phase differences need wrapping.

**NaN propagation:** If any phase is NaN, `compute_order_parameter`
returns `(NaN, NaN)` — no guard. The engine validates inputs upstream.

**Large N numerical noise:** For N > 10,000, accumulated floating-point
error in the sum $\sum e^{i\theta_j}$ can shift R by ~$10^{-12}$.
The Rust backend uses compensated summation for N ≥ 1024.

**PLV with constant signals:** If both `phases_a` and `phases_b` are
constant (no time variation), PLV = 1.0 regardless of the phase
difference. PLV measures consistency, not alignment.

**PLV with short windows:** For T < 30 samples, PLV has high variance
and the Rayleigh test loses statistical power. Use T ≥ 100 for reliable
PLV estimates.

**Layer mask dtype:** `layer_mask` must be boolean. Integer masks
(e.g., `np.array([0, 1, 1, 0])`) will select by index, not by value.

---

## 6. Technical Reference

### 6.1 Module API

::: scpn_phase_orchestrator.upde.order_params
    options:
        show_root_heading: true
        members_order: source

### 6.2 Function Signatures

**`compute_order_parameter(phases: NDArray) → tuple[float, float]`**

| Parameter | Type | Description |
|-----------|------|-------------|
| `phases` | `NDArray` (N,) | Oscillator phases in radians |
| **Returns** | `(R, psi)` | Magnitude $R \in [0,1]$ and mean phase $\psi \in [0, 2\pi)$ |

**`compute_plv(phases_a: NDArray, phases_b: NDArray) → float`**

| Parameter | Type | Description |
|-----------|------|-------------|
| `phases_a` | `NDArray` (T,) | First phase time series |
| `phases_b` | `NDArray` (T,) | Second phase time series (same length) |
| **Returns** | `float` | PLV $\in [0, 1]$ |
| **Raises** | `ValueError` | If array lengths differ |

**`compute_layer_coherence(phases: NDArray, layer_mask: NDArray) → float`**

| Parameter | Type | Description |
|-----------|------|-------------|
| `phases` | `NDArray` (N,) | All oscillator phases |
| `layer_mask` | `NDArray` (N,) bool | Mask selecting layer oscillators |
| **Returns** | `float` | Layer coherence $R_\ell \in [0, 1]$ |

---

## 7. Performance Benchmarks

All benchmarks from `bench/baseline.json`, Intel i5-11600K, Python 3.12.5.

### 7.1 compute_order_parameter

| N | Python (µs) | Rust (µs) | Speedup |
|---|-------------|-----------|---------|
| 8 | 2.1 | 0.8 | 2.6x |
| 16 | 2.3 | 0.9 | 2.6x |
| 64 | 3.5 | 1.2 | 2.9x |
| 256 | 8.2 | 2.1 | 3.9x |
| 1024 | 28.4 | 5.3 | 5.4x |

The function is $O(N)$ — a single pass computing $\bar{C}$ and $\bar{S}$.
Rust speedup grows with N due to SIMD vectorisation.

### 7.2 compute_plv

| T | Python (µs) | Rust (µs) | Speedup |
|---|-------------|-----------|---------|
| 100 | 5.8 | 1.9 | 3.1x |
| 1000 | 42.3 | 8.7 | 4.9x |
| 10000 | 410.5 | 72.1 | 5.7x |

Also $O(T)$ — single pass over the time series.

### 7.3 Calling Frequency in Typical Simulation

In a standard 1000-step simulation with 16 oscillators and 3 monitored
layer pairs:

| Function | Calls | Total time (Rust) |
|----------|-------|-------------------|
| `compute_order_parameter` | 1000 | ~0.9 ms |
| `compute_plv` | 3000 | ~5.7 ms |
| `compute_layer_coherence` | 3000 | ~2.7 ms |

Order parameter computation is negligible (< 1%) compared to the
$O(N^2)$ coupling computation in `UPDEEngine.step()`.

### 7.4 Complexity Summary

| Function | Time | Space | Parallelised (Rust) |
|----------|------|-------|---------------------|
| `compute_order_parameter` | $O(N)$ | $O(1)$ temp | No (too cheap) |
| `compute_plv` | $O(T)$ | $O(1)$ temp | No |
| `compute_layer_coherence` | $O(N_\ell)$ | $O(N_\ell)$ mask | No |

---

## 8. Citations

1. **Kuramoto Y.** (1975). Self-entrainment of a population of coupled
   non-linear oscillators. *Int. Symp. Mathematical Problems in
   Theoretical Physics*, Lecture Notes in Physics **39**:420–422.

2. **Ott E., Antonsen T.M.** (2008). Low dimensional behavior of large
   systems of globally coupled oscillators. *Chaos* **18**(3):037113.
   doi:10.1063/1.2930766

3. **Lachaux J.P., Rodriguez E., Martinerie J., Varela F.J.** (1999).
   Measuring phase synchrony in brain signals. *Human Brain Mapping*
   **8**(4):194–208. doi:10.1002/(SICI)1097-0193(1999)8:4<194::AID-HBM4>3.0.CO;2-C

4. **Mardia K.V., Jupp P.E.** (2000). *Directional Statistics*.
   John Wiley & Sons. ISBN 978-0-471-95333-3.

5. **Strogatz S.H.** (2000). From Kuramoto to Crawford: exploring the
   onset of synchronization. *Physica D* **143**(1–4):1–20.
   doi:10.1016/S0167-2789(00)00094-4

6. **Acebrón J.A. et al.** (2005). The Kuramoto model: A simple paradigm
   for synchronization phenomena. *Rev. Mod. Phys.* **77**(1):137–185.
   doi:10.1103/RevModPhys.77.137

7. **Daido H.** (1993). Order function and macroscopic mutual entrainment
   in uniformly coupled limit-cycle oscillators. *Progress of Theoretical
   Physics* **88**(6):1213–1218. doi:10.1143/ptp/88.6.1213

8. **Fisher N.I.** (1993). *Statistical Analysis of Circular Data*.
   Cambridge University Press. ISBN 978-0-521-56890-6.

---

## Test Coverage

- `tests/test_order_params.py` — 20 tests: R bounds [0,1], ψ range [0,2π),
  synchronised R=1, uniform R≈0, two-cluster cancellation, empty array,
  single oscillator, PLV locked=1, PLV random≈0, PLV length mismatch,
  layer coherence with masks
- `tests/test_property_invariants.py` — 11 property tests (Hypothesis):
  R always in [0,1] for random phases, ψ always in [0,2π),
  PLV symmetric, PLV always in [0,1]

Total: **31 tests** covering all three functions.

---

## Source

- Python: `src/scpn_phase_orchestrator/upde/order_params.py` (76 lines)
- Rust: `spo-kernel/crates/spo-engine/src/order_params.rs`
- FFI: `spo-kernel/crates/spo-ffi/src/lib.rs` (order_parameter, plv bindings)
