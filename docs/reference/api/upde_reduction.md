# Ott-Antonsen Mean-Field Reduction

## 1. Mathematical Formalism

### The Ott-Antonsen Ansatz

For an infinite population of globally coupled Kuramoto oscillators
with Lorentzian (Cauchy) natural frequency distribution:

$$g(\omega) = \frac{\Delta / \pi}{(\omega - \omega_0)^2 + \Delta^2}$$

where $\omega_0$ is the centre frequency and $\Delta$ is the half-width
at half-maximum, the exact low-dimensional dynamics of the Kuramoto
order parameter $z = R e^{i\psi}$ are given by:

$$\frac{dz}{dt} = -(\Delta + i\omega_0)\,z + \frac{K}{2}\left(z - |z|^2 z\right)$$

This is the **Ott-Antonsen reduction** — a single complex ODE that
captures the exact macroscopic dynamics of the infinite-$N$ Kuramoto
model.

### Decomposition of the ODE

Separating real and imaginary parts $z = x + iy$:

$$\dot{x} = -\Delta x + \omega_0 y + \frac{K}{2}(1 - x^2 - y^2)\,x$$

$$\dot{y} = -\Delta y - \omega_0 x + \frac{K}{2}(1 - x^2 - y^2)\,y$$

The three terms have clear physical meanings:
1. **$-\Delta z$**: Damping from frequency spread (wider distribution → faster decay)
2. **$-i\omega_0 z$**: Rotation at the mean frequency
3. **$\frac{K}{2}(z - |z|^2 z)$**: Cubic self-coupling from the Lorentzian
   closure (Ott & Antonsen 2008, Eq. 9)

### Steady-State Solution

Setting $\dot{R} = 0$ in the polar representation:

$$R_{ss} = \begin{cases} 0 & \text{if } K \leq K_c = 2\Delta \\ \sqrt{1 - \frac{2\Delta}{K}} & \text{if } K > K_c \end{cases}$$

This is exact for the Lorentzian distribution. The transition at $K_c = 2\Delta$
is continuous (second-order), with the universal Kuramoto scaling
$R \propto \sqrt{K - K_c}$ near the transition.

### Critical Coupling

$$K_c = 2\Delta$$

This is the **Dörfler-Bullo criterion** for globally coupled oscillators
with Lorentzian spread. Larger frequency spread (larger $\Delta$)
requires stronger coupling to achieve synchronisation.

### Integration Method

RK4 (Runge-Kutta 4th order) on the complex ODE:

$$z_{n+1} = z_n + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

where $k_i$ are the standard RK4 stages applied to $f(z) = -(\Delta + i\omega_0)z + \frac{K}{2}(z - |z|^2 z)$.

---

## 2. Theoretical Context

### Why Mean-Field Reduction?

Simulating $N$ oscillators costs $O(N^2)$ per step (all-to-all coupling).
For $N = 10{,}000$ with 1000 steps, that is $10^{11}$ operations. The
Ott-Antonsen reduction replaces this with a **single complex ODE** —
constant cost per step, independent of $N$.

This is not an approximation: for Lorentzian $g(\omega)$ and $N \to \infty$,
the reduction is **exact**. For finite $N$, it provides a mean-field
prediction that converges as $1/\sqrt{N}$.

### Domain of Validity

The Ott-Antonsen ansatz applies when:
1. **Coupling is global** (all-to-all): $K_{ij} = K/N$ for all $i, j$
2. **Frequency distribution is Lorentzian** (or can be well approximated)
3. **No noise** (deterministic dynamics)
4. **No phase frustration** ($\alpha_{ij} = 0$)

For non-Lorentzian distributions, the ansatz is no longer exact but
remains a useful approximation. For Gaussian $g(\omega)$, the error
is typically $< 5\%$ in $R_{ss}$ for moderate coupling.

### Historical Context

- **Kuramoto (1975):** Original model, heuristic self-consistency for $R$
- **Strogatz (2000):** Mean-field theory for finite $N$
- **Ott & Antonsen (2008):** Exact low-dimensional reduction for Lorentzian $g(\omega)$
- **Ott & Antonsen (2009):** Extension to multivariate coupling
- **Marvel, Mirollo & Strogatz (2009):** Identified the OA manifold as
  the Möbius group (Watanabe-Strogatz theory)
- **Pikovsky & Rosenblum (2008):** Analogous reduction via circular
  cumulants

The Ott-Antonsen result is considered one of the most significant
theoretical advances in coupled oscillator theory since Kuramoto's
original work.

### Applications in SPO

The `OttAntonsenReduction` serves as a **fast diagnostic**:
- Before running expensive N-oscillator simulations, predict whether
  the given $(K, \Delta)$ will produce synchronisation
- Estimate the steady-state $R$ for parameter sweeps in seconds rather
  than hours
- Validate N-oscillator results against the exact mean-field solution
- Provide initial conditions for the full simulation (seed phases near
  the predicted $R_{ss}$)

### Lorentzian Fitting

The `predict_from_oscillators()` method automatically estimates
$(\omega_0, \Delta)$ from a set of measured frequencies:
- $\omega_0 = \text{median}(\omega)$ (robust central tendency)
- $\Delta = \text{IQR}(\omega) / 2$ (the IQR of a Lorentzian equals $2\Delta$)

This enables one-call diagnostics: pass frequencies, get predicted $R$.

---

## 3. Pipeline Position

```
Oscillators.extract() ──→ ω₁, ω₂, ..., ωₙ
                                │
                                ↓
     ┌── OttAntonsenReduction(ω₀, Δ, K, dt) ──┐
     │                                          │
     │  Input:  z₀ (seed), n_steps             │
     │  Param:  ω₀, Δ (from g(ω)), K, dt      │
     │  Method: RK4 on complex ODE (Rust/Py)   │
     │  Output: OAState(z, R, ψ, K_c)          │
     │                                          │
     │  OR: predict_from_oscillators(ω, K)      │
     │      → fit Lorentzian → run → OAState    │
     │                                          │
     └──────────────────────────────────────────┘
                                │
                                ↓
              R_predicted vs R_measured (from UPDEEngine)
              → validation / parameter guidance
```

### Input Contracts

| Parameter | Type | Range | Source |
|-----------|------|-------|--------|
| `omega_0` | `float` | any | Median of natural frequencies |
| `delta` | `float` | $\geq 0$ | Half-width of Lorentzian fit |
| `K` | `float` | $\geq 0$ | Coupling strength |
| `dt` | `float` | $> 0$ | Integration time step (default 0.01) |
| `z0` | `complex` | $|z_0| < 1$ | Seed (typically $0.01 + 0i$) |
| `n_steps` | `int` | $\geq 0$ | Number of RK4 steps |

### Output Contract

| Field | Type | Range | Meaning |
|-------|------|-------|---------|
| `z` | `complex` | $|z| \leq 1$ | Mean-field order parameter |
| `R` | `float` | $[0, 1]$ | $|z|$, coherence |
| `psi` | `float` | $(-\pi, \pi]$ | $\arg(z)$, mean phase |
| `K_c` | `float` | $\geq 0$ | Critical coupling $2\Delta$ |

---

## 4. Features

- **Exact reduction** for Lorentzian $g(\omega)$ — not an approximation
- **RK4 integration** — 4th-order accurate on the complex ODE
- **Rust FFI acceleration** — 38-96x speedup over Python loop
- **Analytical steady-state** — `steady_state_R()` returns exact $R_{ss}$
- **Automatic Lorentzian fitting** — `predict_from_oscillators()` fits
  $(\omega_0, \Delta)$ from measured frequencies
- **Critical coupling** — `K_c` property returns $2\Delta$
- **Single-step and batch** — `step()` for interactive, `run()` for batch
- **Immutable state** — returns `OAState` dataclass, engine is reusable

---

## 5. Usage Examples

### Basic: Predict R for Given Parameters

```python
from scpn_phase_orchestrator.upde.reduction import OttAntonsenReduction

oa = OttAntonsenReduction(omega_0=0.0, delta=0.5, K=3.0, dt=0.01)

print(f"K_c = {oa.K_c:.2f}")           # 1.0
print(f"R_ss = {oa.steady_state_R():.4f}")  # sqrt(1 - 2*0.5/3.0) = 0.8165

state = oa.run(z0=complex(0.01, 0.0), n_steps=2000)
print(f"R (numerical) = {state.R:.4f}")  # Should match R_ss
```

### Fast Diagnostic from Oscillator Frequencies

```python
import numpy as np
from scpn_phase_orchestrator.upde.reduction import OttAntonsenReduction

# Given N oscillator frequencies (e.g. from EEG channels)
omegas = np.random.default_rng(42).standard_cauchy(100) * 0.3 + 1.0

oa = OttAntonsenReduction(omega_0=0, delta=0.1, K=1.0, dt=0.01)
prediction = oa.predict_from_oscillators(omegas, K=2.0)

print(f"Predicted R = {prediction.R:.4f}")
print(f"K_c = {prediction.K_c:.4f}")
print(f"Supercritical: {2.0 > prediction.K_c}")
```

### Parameter Sweep

```python
import numpy as np
from scpn_phase_orchestrator.upde.reduction import OttAntonsenReduction

delta = 0.5
K_values = np.linspace(0, 5, 100)

for K in K_values:
    oa = OttAntonsenReduction(omega_0=0, delta=delta, K=K, dt=0.01)
    R_analytical = oa.steady_state_R()
    state = oa.run(complex(0.01, 0.0), 3000)
    print(f"K={K:.2f}: R_analytical={R_analytical:.4f}, R_numerical={state.R:.4f}")
```

### Validation Against Full Simulation

```python
import numpy as np
from scpn_phase_orchestrator.upde.reduction import OttAntonsenReduction
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

N = 1000
K = 3.0
delta = 0.5

# OA prediction (milliseconds)
oa = OttAntonsenReduction(omega_0=0, delta=delta, K=K, dt=0.01)
R_oa = oa.steady_state_R()

# Full simulation (seconds)
omegas = np.random.default_rng(42).standard_cauchy(N) * delta
knm = np.full((N, N), K / N); np.fill_diagonal(knm, 0.0)
phases = np.random.default_rng(42).uniform(0, 2 * np.pi, N)

engine = UPDEEngine(N, dt=0.01)
for _ in range(5000):
    phases = engine.step(phases, omegas, knm, 0.0, 0.0, np.zeros((N, N)))
R_sim, _ = compute_order_parameter(phases)

print(f"OA prediction: R = {R_oa:.4f}")
print(f"Full sim (N={N}): R = {R_sim:.4f}")
# Error: typically < 0.05 for N >= 100
```

---

## 6. Technical Reference

### Class: OttAntonsenReduction

::: scpn_phase_orchestrator.upde.reduction

### Dataclass: OAState

```python
@dataclass
class OAState:
    z: complex   # Mean-field z = R·exp(iψ)
    R: float     # |z|, order parameter
    psi: float   # arg(z), mean phase
    K_c: float   # Critical coupling 2Δ
```

### Rust Engine Functions

```rust
// RK4 integration of OA ODE, returns (z_re, z_im, R, psi)
pub fn oa_run(z_re, z_im, omega_0, delta, k_coupling, dt, n_steps) -> (f64, f64, f64, f64)

// Analytical steady-state R
pub fn steady_state_r_oa(delta, k_coupling) -> f64

// Fit Lorentzian (omega_0, delta) from frequency array
pub fn fit_lorentzian(omegas: &[f64]) -> (f64, f64)
```

### Auto-Select Logic

All three functions (`run`, `steady_state_R`, `predict_from_oscillators`)
use Rust when available. The RK4 loop in Rust is 38-96x faster because
the Python path executes a Python-level loop with complex arithmetic
per step.

---

## 7. Performance Benchmarks

Measured on Intel Core i5-11600K @ 3.90 GHz, 32 GB DDR4-2400.
ω₀ = 0, Δ = 1, K = 4, z₀ = 0.01. Averaged over 100-1000 iterations.

| Steps | Python (ms) | Rust (ms) | Speedup |
|-------|-------------|-----------|---------|
| 100 | 0.203 | 0.0053 | **38.1x** |
| 500 | 1.836 | 0.0192 | **95.8x** |
| 2,000 | 7.183 | 0.0763 | **94.2x** |
| 10,000 | 20.811 | 0.3641 | **57.2x** |

### Why 38-96x Speedup?

The OA reduction is a **scalar** complex ODE. Each RK4 step requires
4 evaluations of $f(z)$, each involving:
- 1 complex multiply ($|z|^2$)
- 2 complex multiply-adds
- ~10 floating-point operations total

In Python, each step crosses the Python interpreter boundary for every
arithmetic operation (complex + float overhead). In Rust, the entire
loop runs in native code with no interpreter overhead. The 38-96x
range reflects the interplay between Python interpreter overhead
(dominant at high step counts → ~96x) and FFI call overhead
(dominant at low step counts → ~38x). For the typical use case
(500-2000 steps), the speedup is ~95x.

### Memory Usage

Negligible: 2 floats (re, im) for state, 8 floats for RK4 stages.
No heap allocation in the inner loop.

### Test Coverage

- **Rust tests:** 9 (reduction module in spo-engine)
  - Steady-state subcritical/supercritical, convergence, subcritical
    decay, rotation with ω₀, fit_lorentzian, zero steps, psi angle,
    critical coupling
- **Python tests:** 12 (`tests/test_ott_antonsen.py`)
  - Critical coupling, below/above/at K_c, step finite, run convergence,
    run decay, negative delta raises, predict_from_oscillators,
    predict identical omegas, OAState fields, pipeline wiring
- **Source lines:** 208 (Rust) + 99 (Python) = 307 total

---

## 8. Citations

1. **Ott, E. & Antonsen, T. M.** (2008).
   "Low dimensional behavior of large systems of globally coupled
   oscillators."
   *Chaos* 18(3):037113.
   DOI: [10.1063/1.2930766](https://doi.org/10.1063/1.2930766)

2. **Ott, E. & Antonsen, T. M.** (2009).
   "Long time evolution of phase oscillator systems."
   *Chaos* 19(2):023117.
   DOI: [10.1063/1.3136851](https://doi.org/10.1063/1.3136851)

3. **Marvel, S. A., Mirollo, R. E., & Strogatz, S. H.** (2009).
   "Identical phase oscillators with global sinusoidal coupling evolve
   by Möbius group action."
   *Chaos* 19(4):043104.
   DOI: [10.1063/1.3247089](https://doi.org/10.1063/1.3247089)

4. **Pikovsky, A. & Rosenblum, M.** (2008).
   "Partially integrable dynamics of hierarchical populations of
   coupled oscillators."
   *Physical Review Letters* 101(26):264103.
   DOI: [10.1103/PhysRevLett.101.264103](https://doi.org/10.1103/PhysRevLett.101.264103)

5. **Strogatz, S. H.** (2000).
   "From Kuramoto to Crawford: exploring the onset of synchronization
   in populations of coupled oscillators."
   *Physica D* 143(1-4):1-20.
   DOI: [10.1016/S0167-2789(00)00094-4](https://doi.org/10.1016/S0167-2789(00)00094-4)

6. **Dörfler, F. & Bullo, F.** (2014).
   "Synchronization in complex networks of phase oscillators:
   A survey."
   *Automatica* 50(6):1539-1564.
   DOI: [10.1016/j.automatica.2014.04.012](https://doi.org/10.1016/j.automatica.2014.04.012)

7. **Kuramoto, Y.** (1975).
   "Self-entrainment of a population of coupled non-linear oscillators."
   In *Int. Symp. Mathematical Problems in Theoretical Physics*,
   Lecture Notes in Physics 39:420-422. Springer.

8. **Acebrón, J. A., Bonilla, L. L., Pérez Vicente, C. J., Ritort, F.,
   & Spigler, R.** (2005).
   "The Kuramoto model: A simple paradigm for synchronization phenomena."
   *Reviews of Modern Physics* 77:137-185.
   DOI: [10.1103/RevModPhys.77.137](https://doi.org/10.1103/RevModPhys.77.137)

---

## Edge Cases and Limitations

### |z₀| ≥ 1

The OA manifold requires $|z| \leq 1$. If seeded with $|z_0| > 1$,
the cubic term $-|z|^2 z$ drives $|z|$ back below 1, but the transient
may be non-physical. Always seed with $|z_0| \ll 1$ (e.g., 0.01).

### Δ = 0 (Identical Oscillators)

When $\Delta = 0$, $K_c = 0$ — any positive coupling produces
synchronisation. The steady-state $R_{ss} = \sqrt{1 - 0} = 1$.
The OA ODE becomes $\dot{z} = -i\omega_0 z + \frac{K}{2}(z - |z|^2 z)$
— pure rotation plus cubic growth. Starting from $|z_0| < 1$,
$|z| \to 1$ exponentially with rate $K/2$.

### Non-Lorentzian Distributions

For Gaussian $g(\omega) = \frac{1}{\sigma\sqrt{2\pi}} e^{-(\omega-\omega_0)^2/2\sigma^2}$:
- The OA ansatz is **not** exact
- Using $\Delta = \sigma \cdot \sqrt{2\ln 2}$ (FWHM matching) gives
  $R_{ss}$ accurate to $\sim 5\%$ for moderate coupling
- For bimodal distributions, the OA ansatz does not apply

### Numerical Stability

The RK4 integrator is unconditionally stable for $K \Delta t < 4$.
For typical parameters ($K \leq 10$, $\Delta t = 0.01$), this is
always satisfied. If $K > 100$, reduce $\Delta t$ accordingly.

### Finite-$N$ Corrections

For $N$ oscillators (not $N \to \infty$):
- $R$ fluctuates around $R_{ss}$ with variance $\sim 1/N$
- The transition at $K_c$ is rounded: $R > 0$ for all $K > 0$
  (finite-size precursors)
- The `predict_from_oscillators` method accounts for the Lorentzian
  fitting uncertainty but not the finite-$N$ fluctuations

---

## Derivation of the Ott-Antonsen Ansatz

### Step 1: Continuity Equation

For $N \to \infty$ oscillators with density $f(\theta, \omega, t)$
on $S^1$, the continuity equation reads:

$$\frac{\partial f}{\partial t} + \frac{\partial}{\partial \theta}\big(f \cdot v(\theta, \omega, t)\big) = 0$$

where $v = \omega + K R \sin(\psi - \theta)$ is the velocity field
from the Kuramoto model with mean-field coupling.

### Step 2: Fourier Expansion

The density admits a Fourier series in $\theta$:

$$f(\theta, \omega, t) = \frac{1}{2\pi}\left[1 + \sum_{n=1}^{\infty} \hat{f}_n(\omega, t) e^{in\theta} + \text{c.c.}\right]$$

Substituting into the continuity equation yields an infinite hierarchy
of ODEs for the Fourier coefficients $\hat{f}_n$.

### Step 3: The Ansatz

Ott and Antonsen (2008) observed that if we restrict to the manifold:

$$\hat{f}_n(\omega, t) = \alpha(\omega, t)^n$$

then the infinite hierarchy **collapses** to a single ODE for $\alpha$:

$$\dot{\alpha} = -i\omega\alpha + \frac{K}{2}(\bar{z}\alpha^2 - z)$$

where $z = R e^{i\psi}$ is the Kuramoto order parameter, defined by:

$$z(t) = \int_{-\infty}^{\infty} \int_0^{2\pi} e^{i\theta} f(\theta, \omega, t)\,d\theta\,d\omega$$

### Step 4: Lorentzian Closure

For $g(\omega) = \frac{\Delta/\pi}{(\omega - \omega_0)^2 + \Delta^2}$,
the integral over $\omega$ can be evaluated by closing the contour in
the lower half-plane, picking up the single pole at $\omega = \omega_0 - i\Delta$:

$$z(t) = \overline{\alpha(\omega_0 - i\Delta, t)}$$

This yields the closed ODE:

$$\dot{z} = -(\Delta + i\omega_0)z + \frac{K}{2}(z - |z|^2 z)$$

The key insight is that the Lorentzian distribution has a single
pole in the lower half-plane, which makes the contour integral exact.
Distributions with multiple poles (e.g., bimodal) or no poles
(e.g., Gaussian, which has an essential singularity) break the closure.

---

## Rust Implementation Details

### oa_deriv — Inner Loop Kernel

The derivative function `oa_deriv` computes both real and imaginary
parts of $f(z)$ without heap allocation:

```rust
fn oa_deriv(re: f64, im: f64, omega_0: f64, delta: f64, half_k: f64) -> (f64, f64) {
    let abs_sq = re * re + im * im;
    let lin_re = -delta * re + omega_0 * im;
    let lin_im = -delta * im - omega_0 * re;
    let cubic_factor = half_k * (1.0 - abs_sq);
    (lin_re + cubic_factor * re, lin_im + cubic_factor * im)
}
```

Each RK4 step calls `oa_deriv` 4 times (k1-k4), for a total of
~40 floating-point operations per step. The Rust compiler inlines
this function and uses SSE2/AVX for the multiply-adds.

### fit_lorentzian — Robust Estimation

The Lorentzian fitting uses order statistics rather than
maximum-likelihood estimation:
- **Median** for $\omega_0$: breakdown point 50%, robust to outliers
- **IQR/2** for $\Delta$: exploits the exact relationship $\text{IQR} = 2\Delta$
  for a Cauchy distribution

The sort is $O(n \log n)$, but $n$ is the number of oscillators
(typically $\leq 10{,}000$), so the cost is negligible compared to
the integration.

### FFI Transfer

The `oa_run` function returns a tuple `(f64, f64, f64, f64)`, which
PyO3 converts to a Python tuple with zero-copy semantics (scalar
values only, no array allocation). The `fit_lorentzian` function
accepts a `PyReadonlyArray1<f64>` reference, avoiding any data copy
from NumPy to Rust.

---

## Troubleshooting

### Issue: R Does Not Converge

**Symptom:** After many steps, $R$ oscillates or drifts instead of
settling to $R_{ss}$.

**Diagnosis:** Check whether $\omega_0 \neq 0$. In the rotating frame,
$R$ converges while $\psi$ rotates at frequency $\omega_0$. The complex
$z$ traces a circle of radius $R_{ss}$, so $R = |z|$ converges even
though $z$ itself does not settle to a fixed point.

If $R$ truly oscillates, verify that $K \Delta t < 4$ (RK4 stability
bound). For $K = 100$ and $\Delta t = 0.01$, $K \Delta t = 1.0$ is safe.

### Issue: predict_from_oscillators Returns R ≈ 0 for Clearly Coupled System

**Diagnosis:** The Lorentzian fit via IQR assumes unimodal,
heavy-tailed $g(\omega)$. For bimodal or Gaussian distributions,
the estimated $\Delta$ may be too large, yielding $K_c > K$.

**Solution:** Manually set $(\omega_0, \Delta)$ from known distribution
parameters rather than using the automatic fitting.

### Issue: R_ss = 0 Despite Supercritical K

**Diagnosis:** The `steady_state_R` function requires `K > 2Δ`.
If you pass `K` as the total coupling but the per-pair coupling
is $K/N$, you need to use the per-pair value multiplied by $N$
(i.e., the effective global coupling strength).

### Issue: Mismatch Between OA and Full Simulation

**Diagnosis:** Common causes:
1. Non-Lorentzian frequency distribution (OA is approximate)
2. Finite-$N$ fluctuations ($R$ deviates by $\sim 1/\sqrt{N}$)
3. Phase frustration ($\alpha_{ij} \neq 0$) — OA assumes zero frustration
4. Non-global coupling (sparse $K_{ij}$) — OA assumes all-to-all

For cases 1-2, the mismatch should decrease with larger $N$ and
Lorentzian-like distributions. For cases 3-4, the OA reduction is
not applicable.

---

## Integration with Other SPO Modules

### With UPDEEngine

The `OttAntonsenReduction` validates full simulations:

```python
# Quick check: will this parameter set synchronise?
oa = OttAntonsenReduction(omega_0=0, delta=0.5, K=3.0)
if oa.steady_state_R() > 0.5:
    # Worth running the expensive N-oscillator simulation
    engine = UPDEEngine(n_oscillators=256, dt=0.01)
    # ... run full sim ...
```

### With SSGF Costs

The predicted $R_{ss}$ can initialise the SSGF sync deficit cost:

$$C_1 = 1 - R \approx 1 - R_{ss}$$

This provides a baseline expectation for the geometry optimisation.
If the full simulation yields $R < R_{ss}$, the geometry is
suboptimal and the SSGF engine should increase the learning rate.

### With RegimeManager

The OA reduction enables fast regime classification:

| $K / K_c$ | Regime | $R_{ss}$ |
|------------|--------|----------|
| $< 0.5$ | Incoherent | 0 |
| $0.5 - 1.0$ | Subcritical | 0 |
| $1.0 - 1.5$ | Onset | $0 - 0.58$ |
| $1.5 - 3.0$ | Partial sync | $0.58 - 0.82$ |
| $> 3.0$ | Full sync | $> 0.82$ |

The supervisor can use these thresholds to decide whether to engage
geometry control or whether the current coupling is sufficient.
