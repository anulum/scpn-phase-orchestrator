# Transfer Entropy Directed Adaptive Coupling

## 1. Mathematical Formalism

### Transfer Entropy

Transfer entropy (TE) quantifies the directed information flow
between two time series. For discrete-time series $X$ (source) and
$Y$ (target), the transfer entropy from $X$ to $Y$ is:

$$TE(X \to Y) = H(Y_{t+1} | Y_t) - H(Y_{t+1} | Y_t, X_t)$$

where $H(A|B)$ denotes the conditional Shannon entropy. Equivalently:

$$TE(X \to Y) = \sum_{y_{t+1}, y_t, x_t} p(y_{t+1}, y_t, x_t) \log \frac{p(y_{t+1} | y_t, x_t)}{p(y_{t+1} | y_t)}$$

If knowing $X_t$ reduces the uncertainty about $Y_{t+1}$ beyond
what $Y_t$ already provides, then $TE(X \to Y) > 0$: $X$ causally
influences $Y$.

### Key Properties

1. **Non-negative:** $TE(X \to Y) \geq 0$ by the data processing
   inequality.
2. **Asymmetric:** $TE(X \to Y) \neq TE(Y \to X)$ in general —
   directional causality.
3. **Model-free:** Does not assume linear dynamics, Gaussian noise,
   or any parametric form.
4. **Equivalent to conditional mutual information:**
   $TE(X \to Y) = I(X_t; Y_{t+1} | Y_t)$.

### Histogram-Based Estimation

The SPO implementation discretises phase trajectories into $B$ equal
bins on $[0, 2\pi)$ and estimates joint probabilities from bin
co-occurrence counts:

$$\hat{p}(y_{t+1}, y_t, x_t) = \frac{N(y_{t+1}, y_t, x_t)}{T-1}$$

The conditional entropy is computed per conditioning value:

$$H(Y_{t+1} | C) = -\sum_c P(C=c) \sum_{y} P(Y_{t+1}=y | C=c) \log P(Y_{t+1}=y | C=c)$$

For the TE computation, the conditioning variable is $Y_t$ alone
(first term) and $(Y_t, X_t)$ jointly (second term).

### TE-Directed Coupling Update Rule

The coupling matrix $K_{ij}$ is updated using TE as a learning signal:

$$K_{ij}(t+1) = (1 - \lambda) \cdot K_{ij}(t) + \eta \cdot TE(i \to j)$$

with the constraints:
- **Diagonal zero:** $K_{ii} = 0$ (no self-coupling)
- **Non-negative:** $K_{ij} \geq 0$ (excitatory only)

Where:
- $\eta$ is the learning rate (how fast TE drives coupling adaptation)
- $\lambda$ is the decay rate (how fast old couplings forget)

### Interpretation

This update rule implements a form of **Hebbian-like plasticity** based
on information-theoretic causality rather than correlation:

- **High $TE(i \to j)$:** Oscillator $i$ drives oscillator $j$ →
  strengthen $K_{ij}$
- **Low $TE(i \to j)$:** No causal influence → coupling decays toward zero
- **Asymmetric TE:** $TE(i \to j) \neq TE(j \to i)$ →
  coupling matrix becomes asymmetric, reflecting directed causality

This is fundamentally different from standard Hebbian plasticity
($\Delta K_{ij} \propto \cos(\theta_j - \theta_i)$), which only
captures instantaneous phase alignment, not temporal causality.

---

## 2. Theoretical Context

### Why Transfer Entropy for Coupling Adaptation?

Traditional coupling adaptation in Kuramoto-type models uses
Hebbian-like rules:

$$\Delta K_{ij} = \epsilon \cdot \sin(\theta_j - \theta_i)$$

This strengthens coupling between already-synchronised oscillators.
The problem: it captures correlation, not causation. Two oscillators
may be synchronised because of a common driver (confound) rather
than direct influence.

Transfer entropy resolves this by measuring the **directional
predictive information flow**. It is a nonlinear generalisation
of Granger causality (Schreiber, 2000):

| Property | Hebbian | Granger causality | Transfer entropy |
|----------|---------|-------------------|-----------------|
| Linearity | No (uses sin) | Linear only | Model-free |
| Directionality | Symmetric | Directed | Directed |
| Confound robustness | No | Partial | Yes (conditioned) |
| Computational cost | $O(N^2)$ | $O(N^2 T^2)$ | $O(N^2 T B)$ |

### Historical Context

- **Schreiber, T. (2000):** Introduced transfer entropy as an
  information-theoretic measure of directed coupling. The original
  paper demonstrated that TE detects coupling directionality that
  linear cross-correlation misses.
- **Lizier, J. T. (2012):** Extended TE to "local information
  transfer" — a spatiotemporal filter that decomposes global TE into
  per-timestep contributions. This enables detection of transient
  causal events.
- **Vicente, R. et al. (2011):** Applied TE to neural spike trains,
  showing it outperforms Granger causality for nonlinear neural
  dynamics.
- **Wibral, M. et al. (2014):** Comprehensive review of TE in
  neuroscience with practical estimation guidelines.

### TE in Coupled Oscillator Networks

For Kuramoto-type oscillators, TE naturally captures the causal
structure:
- If $K_{ij} > 0$ and $K_{ji} = 0$, then $TE(i \to j) > 0$
  and $TE(j \to i) \approx 0$.
- The magnitude of TE scales with coupling strength and phase
  coherence.
- At full synchronisation ($R = 1$), TE approaches zero because
  knowing $Y_t$ already perfectly predicts $Y_{t+1}$ — no
  additional information from $X_t$.

This means TE-directed adaptation is most active during partial
synchronisation, automatically reducing its influence as the system
approaches full coherence.

### Relation to Active Inference

In the SCPN framework, TE-directed coupling adaptation can be
viewed as the network's **sensory inference**: the system infers
its own causal structure from observed dynamics, then adapts its
connectivity to align with the inferred structure. This is a form
of structure learning that complements the SSGF geometry control
(which operates on a slower timescale).

---

## 3. Pipeline Position

```
UPDEEngine.step() ──→ phases(t) ──→ trajectory buffer
                                          │
                                          ↓ (n, T) phase history
         ┌── te_adapt_coupling() ─────────────────┐
         │                                         │
         │  Step 1: transfer_entropy_matrix()      │
         │    TE(i→j) for all pairs (Python/Rust)  │
         │                                         │
         │  Step 2: coupling update (Rust if avail.)│
         │    K_new = (1-λ)K + η·TE                │
         │    diag(K_new) = 0, K_new ≥ 0           │
         │                                         │
         │  Output: updated (n, n) coupling matrix │
         └─────────────────────────────────────────┘
                                          │
                                          ↓
         UPDEEngine.step(phases, omegas, K_new, ...)
```

### Input Contracts

| Parameter | Type | Shape | Range | Source |
|-----------|------|-------|-------|--------|
| `knm` | `NDArray[float64]` | `(N, N)` | $\geq 0$ | Current coupling matrix |
| `phase_history` | `NDArray[float64]` | `(N, T)` | $[0, 2\pi)$ | Recent phase trajectories |
| `lr` | `float` | scalar | $\geq 0$ | Learning rate (default 0.01) |
| `decay` | `float` | scalar | $[0, 1]$ | Decay rate (default 0.0) |
| `n_bins` | `int` | scalar | $\geq 2$ | Histogram bins (default 8) |

### Output Contract

| Field | Type | Shape | Constraints |
|-------|------|-------|-------------|
| (return) | `NDArray[float64]` | `(N, N)` | $\geq 0$, diagonal = 0 |

### Typical Update Frequency

The TE matrix computation is $O(N^2 \cdot T \cdot B)$ where $T$ is
the trajectory length and $B$ the number of bins. For $N = 32$,
$T = 200$, $B = 8$, this is ~50k operations — fast enough to run
every ~100 integration steps.

Recommended cadence: every 50-200 integration steps, collect a
trajectory window, compute TE, update coupling. This amortises the
TE cost over many cheap integration steps.

---

## 4. Features

- **Directed causality detection** — TE captures asymmetric information
  flow, unlike correlation-based methods
- **Model-free** — no linearity or Gaussianity assumptions
- **Hebbian-like update** — strengthens causal connections, weakens
  spurious ones
- **Decay mechanism** — old coupling fades with rate $\lambda$, enabling
  tracking of time-varying causal structure
- **Guaranteed non-negative** — coupling clamped to $\geq 0$
- **Zero self-coupling** — diagonal always forced to zero
- **Rust FFI for update step** — the coupling update (not TE
  computation) dispatches to Rust
- **Configurable histogram resolution** — `n_bins` trades off
  statistical precision vs. bias
- **Composable** — output is a standard coupling matrix compatible
  with all SPO engines

---

## 5. Usage Examples

### Basic: One Adaptation Step

```python
import numpy as np
from scpn_phase_orchestrator.coupling.te_adaptive import te_adapt_coupling

N = 8
rng = np.random.default_rng(42)
knm = np.full((N, N), 0.3)
np.fill_diagonal(knm, 0.0)

# Simulate 200 timesteps of phase data
phase_history = rng.uniform(0, 2 * np.pi, (N, 200))

# Adapt coupling using transfer entropy
knm_new = te_adapt_coupling(knm, phase_history, lr=0.01, decay=0.01)
print(f"Sum of coupling: {knm.sum():.2f} → {knm_new.sum():.2f}")
print(f"Max asymmetry: {np.max(np.abs(knm_new - knm_new.T)):.4f}")
```

### Closed-Loop Adaptation

```python
import numpy as np
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter
from scpn_phase_orchestrator.coupling.te_adaptive import te_adapt_coupling

N = 16
eng = UPDEEngine(N, dt=0.01)
rng = np.random.default_rng(42)
phases = rng.uniform(0, 2 * np.pi, N)
omegas = rng.standard_normal(N)
knm = np.full((N, N), 0.2)
np.fill_diagonal(knm, 0.0)
alpha = np.zeros((N, N))

for cycle in range(10):
    # Collect trajectory
    trajectory = []
    for _ in range(200):
        phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        trajectory.append(phases.copy())
    traj = np.array(trajectory).T  # (N, T)

    # Adapt coupling
    knm = te_adapt_coupling(knm, traj, lr=0.05, decay=0.02)

    R, _ = compute_order_parameter(phases)
    print(f"Cycle {cycle}: R = {R:.4f}, K_sum = {knm.sum():.2f}")
```

### With SplittingEngine

```python
import numpy as np
from scpn_phase_orchestrator.upde.splitting import SplittingEngine
from scpn_phase_orchestrator.coupling.te_adaptive import te_adapt_coupling

N = 8
eng = SplittingEngine(N, dt=0.01)
rng = np.random.default_rng(42)
phases = rng.uniform(0, 2 * np.pi, N)
omegas = np.ones(N) * 2.0
knm = np.full((N, N), 0.3); np.fill_diagonal(knm, 0.0)
alpha = np.zeros((N, N))

# Run trajectory
traj = []
for _ in range(200):
    phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
    traj.append(phases.copy())

# Adapt
knm = te_adapt_coupling(knm, np.array(traj).T, lr=0.01)
```

### Comparing TE Directionality

```python
import numpy as np
from scpn_phase_orchestrator.monitor.transfer_entropy import (
    transfer_entropy_matrix,
)

N = 4
rng = np.random.default_rng(42)
# Create directional coupling: 0 → 1 → 2 → 3
phases = np.zeros((N, 300))
phases[0] = rng.uniform(0, 2 * np.pi, 300)
for i in range(1, N):
    phases[i, 1:] = 0.8 * phases[i-1, :-1] + 0.2 * rng.uniform(0, 2*np.pi, 299)

te = transfer_entropy_matrix(phases, n_bins=8)
print("TE matrix (TE[i,j] = TE(i→j)):")
print(te.round(4))
# Expect: te[0,1] > te[1,0], te[1,2] > te[2,1], etc.
```

---

## 6. Technical Reference

### Function: te_adapt_coupling

::: scpn_phase_orchestrator.coupling.te_adaptive

### Function Signature

```python
def te_adapt_coupling(
    knm: NDArray,           # (N, N) current coupling matrix
    phase_history: NDArray,  # (N, T) recent phase trajectories
    lr: float = 0.01,       # learning rate
    decay: float = 0.0,     # coupling decay rate
    n_bins: int = 8,         # histogram bins for TE estimation
) -> NDArray:               # (N, N) updated coupling matrix
```

### Two-Phase Computation

The function executes in two phases:

1. **TE computation** (`transfer_entropy_matrix`): Always runs in
   Python (or Rust if `transfer_entropy.py` has `_HAS_RUST`).
   This is the $O(N^2 \cdot T \cdot B)$ bottleneck.

2. **Coupling update** (`te_adapt_coupling` core): Dispatches to
   Rust when `_HAS_RUST` is `True`. This is $O(N^2)$ — negligible
   compared to phase 1.

### Rust Engine Function

```rust
pub fn te_adapt_coupling(
    knm: &[f64],    // N×N coupling matrix (row-major flat)
    te: &[f64],     // N×N transfer entropy matrix (row-major flat)
    n: usize,       // number of oscillators
    lr: f64,        // learning rate
    decay: f64,     // coupling decay rate
) -> Vec<f64>       // N×N updated coupling matrix
```

The Rust function receives the already-computed TE matrix and applies
the update rule element-wise with diagonal zeroing and non-negative
clamping.

### Auto-Select Logic

```python
try:
    from spo_kernel import te_adapt_coupling_rust as _rust_te_adapt
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False
```

The Rust path accelerates only the coupling update step. The TE
matrix computation remains in the Python/Rust path of
`transfer_entropy.py`.

### Conditional Entropy Estimation

The `_conditional_entropy` helper computes $H(Y|C)$ by iterating
over all conditioning values $c$ and computing the entropy of $Y$
within each conditioned bin. A smoothing constant $10^{-30}$ prevents
$\log(0)$.

---

## 7. Performance Benchmarks

Measured on Intel Core i5-11600K @ 3.90 GHz, 32 GB DDR4-2400.
Phase history length $T = 200$, $B = 8$ bins, median of 10-20 runs.

### End-to-End (TE Computation + Update)

| N | Python (ms) | Rust (ms) | Speedup |
|---|-------------|-----------|---------|
| 16 | 3.191 | 3.986 | **0.8x** |
| 32 | 13.676 | 12.989 | **1.1x** |
| 64 | 60.312 | 61.156 | **1.0x** |

### Why ~1x Speedup?

The Rust path only accelerates the coupling update ($O(N^2)$
element-wise operations). The dominant cost is the TE matrix
computation ($O(N^2 \cdot T \cdot B)$), which involves:
- $N^2$ calls to `phase_transfer_entropy`
- Each call: binning $T$ samples, computing conditional entropy
  over $B^2$ joint bins

This TE computation runs in Python regardless of the Rust flag.
The coupling update itself is ~microseconds for $N \leq 64$,
invisible in the total cost.

To achieve meaningful speedup, the TE matrix computation itself
needs Rust acceleration (already available via `transfer_entropy.py`
`_HAS_RUST`). When both TE and update run in Rust, the full pipeline
benefits from native speed.

### Cost Breakdown (N=32, T=200)

| Phase | Time (ms) | Fraction |
|-------|-----------|----------|
| TE matrix computation | ~13.5 | ~99% |
| Coupling update | ~0.02 | ~1% |
| Total | ~13.5 | 100% |

### Memory Usage

- TE matrix: $N^2$ floats (~8 KB for $N = 32$)
- Coupling matrix: $N^2$ floats
- Phase history: $N \times T$ floats (~50 KB for $N = 32$, $T = 200$)
- Binned arrays: $3 \times T$ ints (per pair, temporary)

### Test Coverage

- **Rust tests:** 6 (te_adaptive module in spo-engine)
  - Diagonal always zero, no-decay adds TE, full decay reduces,
    clamp non-negative, preserves asymmetry, zero LR no change
- **Python tests:** 7 (`tests/test_te_adaptive.py`)
  - Output shape, zero diagonal, non-negative, coupling increases,
    decay reduces, pipeline wiring (engine → TE → adapt → engine)
- **Source lines:** 128 (Rust) + 64 (Python) = 192 total

---

## 8. Citations

1. **Schreiber, T.** (2000).
   "Measuring information transfer."
   *Physical Review Letters* 85(2):461-464.
   DOI: [10.1103/PhysRevLett.85.461](https://doi.org/10.1103/PhysRevLett.85.461)

2. **Lizier, J. T.** (2012).
   "Local information transfer as a spatiotemporal filter for
   complex systems."
   *Physical Review E* 77(2):026110.
   DOI: [10.1103/PhysRevE.77.026110](https://doi.org/10.1103/PhysRevE.77.026110)

3. **Vicente, R., Wibral, M., Lindner, M., & Pipa, G.** (2011).
   "Transfer entropy — a model-free measure of effective
   connectivity for the neurosciences."
   *Journal of Computational Neuroscience* 30(1):45-67.
   DOI: [10.1007/s10827-010-0262-3](https://doi.org/10.1007/s10827-010-0262-3)

4. **Wibral, M., Vicente, R., & Lizier, J. T.** (eds.) (2014).
   *Directed Information Measures in Neuroscience.*
   Springer. ISBN: 978-3-642-54474-3.

5. **Staniek, M. & Lehnertz, K.** (2008).
   "Symbolic transfer entropy."
   *Physical Review Letters* 100(15):158101.
   DOI: [10.1103/PhysRevLett.100.158101](https://doi.org/10.1103/PhysRevLett.100.158101)

6. **Granger, C. W. J.** (1969).
   "Investigating causal relations by econometric models and
   cross-spectral methods."
   *Econometrica* 37(3):424-438.
   DOI: [10.2307/1912791](https://doi.org/10.2307/1912791)

7. **Barnett, L., Barrett, A. B., & Seth, A. K.** (2009).
   "Granger causality and transfer entropy are equivalent for
   Gaussian variables."
   *Physical Review Letters* 103(23):238701.
   DOI: [10.1103/PhysRevLett.103.238701](https://doi.org/10.1103/PhysRevLett.103.238701)

8. **Kuramoto, Y.** (1984).
   *Chemical Oscillations, Waves, and Turbulence.*
   Springer. ISBN: 978-3-642-69691-6.

---

## Edge Cases and Limitations

### Short Trajectories ($T < 20$)

With few timesteps, the histogram-based TE estimate has high
variance. The conditional bins contain too few samples for reliable
entropy estimation. The function returns $TE = 0$ when $T < 3$
(minimum for the temporal structure $X_t, Y_t, Y_{t+1}$).

**Recommendation:** Use $T \geq 100$ for reliable estimates.
The bias-variance trade-off is:
- Few bins ($B = 4$): low variance, high bias (cannot resolve
  fine structure)
- Many bins ($B = 32$): low bias, high variance (bins sparsely
  populated)
- Default $B = 8$ balances these for typical phase dynamics.

### All Oscillators Synchronised ($R \approx 1$)

When all phases are nearly identical, $Y_t$ perfectly predicts
$Y_{t+1}$ → adding $X_t$ provides no additional information →
$TE \approx 0$ for all pairs. The coupling matrix decays toward
zero (if $\lambda > 0$) or stays constant (if $\lambda = 0$).

This is correct behaviour: at full synchronisation, the causal
structure becomes undetectable from the dynamics.

### Negative TE Before Clamping

The histogram-based TE estimator can produce slightly negative
values due to estimation noise (finite-sample bias). The clamping
$K_{ij} \geq 0$ prevents negative coupling. If the TE estimate is
strongly negative (possible with adversarial or pathological data),
the coupling for that edge is clamped to zero.

### Decay Rate $\lambda = 1$

With full decay, all coupling is erased each step and replaced
entirely by $\eta \cdot TE$. This is aggressive and can cause
oscillatory coupling behaviour. Typical values: $\lambda \in [0, 0.1]$.

---

## Integration with Other SPO Modules

### With SSGF Geometry Control

The TE-directed adaptation operates on a **faster timescale** than
SSGF geometry optimisation:

| Mechanism | Timescale | Drives |
|-----------|-----------|--------|
| SSGF (free energy) | Slow ($\eta_{SSGF} \sim 10^{-3}$) | Global topology |
| TE-directed adaptation | Medium ($\eta_{TE} \sim 10^{-2}$) | Local causality |
| Hebbian plasticity | Fast ($\eta_{Hebb} \sim 10^{-1}$) | Pairwise alignment |

In a three-timescale architecture:
1. Hebbian plasticity adjusts coupling every step (fast)
2. TE adaptation corrects causal structure every ~100 steps (medium)
3. SSGF optimises global topology every ~1000 steps (slow)

### With CouplingBuilder

The `te_adapt_coupling` function modifies an existing coupling matrix.
The initial matrix typically comes from `CouplingBuilder`:

```python
cs = CouplingBuilder().build(n_layers=N, base_strength=0.5)
knm = cs.knm  # Initial topology
# ... run dynamics, collect trajectory ...
knm = te_adapt_coupling(knm, trajectory, lr=0.01)
# knm now reflects both structural connectivity (CouplingBuilder)
# and functional connectivity (TE-directed)
```

### With RegimeManager

The TE matrix itself can serve as a diagnostic: if
$\max_{ij} TE(i \to j) \to 0$, the system has reached a dynamically
trivial state (full sync or full incoherence). The `RegimeManager`
can use this as an additional signal for regime transitions.

---

## Troubleshooting

### Issue: Coupling Grows Unbounded

**Diagnosis:** $\eta$ (learning rate) is too high relative to $\lambda$
(decay). Without decay, coupling can only increase (TE is non-negative).

**Solution:** Set $\lambda > 0$ (e.g., $\lambda = 0.01$) or reduce $\eta$.
For stable long-term adaptation: $\eta \cdot \max(TE) \lesssim \lambda \cdot \max(K)$.

### Issue: TE Matrix is All Zeros

**Diagnosis:** Either (a) trajectory is too short ($T < 20$), (b) all
oscillators are fully synchronised, or (c) `n_bins` is too large for
the available data.

**Solution:** Increase $T$, reduce $n_bins$, or check if the system
is in a trivial dynamical state.

### Issue: Coupling Becomes Fully Asymmetric

**Diagnosis:** This is expected behaviour — TE is asymmetric by design.
If symmetric coupling is desired, symmetrise after update:
$K_{ij} \leftarrow (K_{ij} + K_{ji}) / 2$.
