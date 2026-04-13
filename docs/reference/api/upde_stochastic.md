# Stochastic Injection — Noise-Enhanced Synchronisation

The `stochastic` module adds calibrated Wiener noise to Kuramoto phase
dynamics, enabling the study of **stochastic resonance** in oscillator
networks. Counter-intuitively, adding noise to a weakly synchronised system
can INCREASE the order parameter $R$ — noise helps oscillators explore
phase space and find the synchronised attractor.

The module provides: noise injection via Euler-Maruyama, analytical
self-consistency equations for the noisy Kuramoto model, optimal noise
estimation, and a sweep function to find the resonance peak.

---

## 1. Mathematical Formalism

### 1.1 The Noisy Kuramoto Equation

Adding white noise to the Kuramoto model gives a stochastic differential
equation (Langevin form):

$$
d\theta_i = \left(\omega_i + \sum_j K_{ij}\sin(\theta_j - \theta_i - \alpha_{ij})\right) dt + \sqrt{2D}\, dW_i
$$

where $W_i(t)$ are independent Wiener processes and $D \geq 0$ is the
noise intensity (diffusion coefficient). Units of $D$: rad²/s.

### 1.2 Euler-Maruyama Discretisation

The `StochasticInjector.inject()` method applies one-step Euler-Maruyama:

$$
\theta_i^{n+1} = \left(\theta_i^{n+} + \sqrt{2D \cdot \Delta t}\, \xi_i^n\right) \bmod 2\pi
$$

where $\theta_i^{n+}$ is the phase after the deterministic integration
step (Euler/RK4/RK45 via `UPDEEngine.step()`) and $\xi_i^n \sim N(0,1)$
i.i.d. This is a **splitting scheme**: deterministic step first, then
stochastic perturbation.

### 1.3 Self-Consistency Equation

For the all-to-all Kuramoto model with identical oscillators and noise $D$,
the steady-state order parameter satisfies (Acebrón et al. 2005):

$$
R = \frac{I_1(KR/D)}{I_0(KR/D)}
$$

where $I_0, I_1$ are modified Bessel functions of the first kind. This
transcendental equation has:
- $R = 0$ as a solution for all $(K, D)$ (incoherent state)
- $R > 0$ solution exists when $K > 2D$ (synchronised state)

The critical coupling in the presence of noise:

$$
K_c(D) = 2D
$$

Noise raises the synchronisation threshold linearly.

### 1.4 Stochastic Resonance

For non-identical oscillators (frequency spread $\Delta\omega > 0$):

1. **No noise ($D = 0$):** Some oscillators synchronise ($R > 0$) but
   not all — frequency mismatch prevents full locking.
2. **Optimal noise ($D = D^*$):** Noise helps "kick" desynchronised
   oscillators into the synchronised cluster. $R$ increases!
3. **Too much noise ($D \gg D^*$):** Noise overwhelms coupling. $R \to 0$.

The optimal noise intensity $D^*$ is:

$$
D^* \approx \frac{K \cdot R_{\text{det}}}{2}
$$

where $R_{\text{det}}$ is the deterministic order parameter (Tselios et al.
2025). This is the "common noise" estimate — individual noise enhances
synchronisation when its intensity matches the coupling-phase mismatch
energy scale.

### 1.5 Fokker-Planck Description

The probability density $\rho(\theta, t)$ of oscillator phases obeys the
Fokker-Planck equation:

$$
\frac{\partial \rho}{\partial t} = -\frac{\partial}{\partial \theta}
\left[(\omega + KR\sin(\psi - \theta))\rho\right] + D\frac{\partial^2 \rho}{\partial \theta^2}
$$

The stationary solution on $S^1$ is a von Mises distribution:

$$
\rho_{\text{stat}}(\theta) \propto \exp\left(\frac{KR}{D}\cos(\theta - \psi)\right)
$$

with concentration parameter $\kappa = KR/D$. The self-consistency equation
(§1.3) follows from requiring $R = |\langle e^{i\theta}\rangle_\rho|$.

### 1.6 Noise Types

The module implements **additive common-intensity noise** (same $D$ for all
oscillators). Extensions (not yet implemented):

| Noise type | Formula | Physical basis |
|------------|---------|---------------|
| Common intensity (current) | $\sqrt{2D}\, dW_i$ | Thermal noise |
| Per-oscillator | $\sqrt{2D_i}\, dW_i$ | Heterogeneous environment |
| Common noise | $\sqrt{2D}\, dW$ (same for all) | Global forcing |
| Coloured noise | $d\eta = -\eta/\tau_{c}\, dt + \sqrt{2D/\tau_c}\, dW$ | Neural noise |

---

## 2. Theoretical Context

### 2.1 Historical Background

Noise in coupled oscillators was first studied by Sakaguchi (1988) and
Strogatz & Mirollo (1991). The Fokker-Planck approach was developed by
Acebrón et al. (2005) in their comprehensive review. Stochastic resonance
in the Kuramoto model was characterised by Tönjes & Blasius (2007) and
more recently by Tselios et al. (2025) for applications in neural
synchronisation.

### 2.2 Role in SCPN

Noise injection serves two SCPN purposes:

1. **Biological realism** — neural oscillators are inherently noisy.
   Setting $D$ proportional to the layer's noise floor (from EEG or
   neural recording data) makes simulations more realistic.

2. **Robustness testing** — adding noise to a calibrated SCPN model
   tests whether the synchronisation pattern is robust or fragile.
   If $R$ drops sharply with small $D$, the system is operating near
   the critical point and may need stronger coupling.

### 2.3 Connection to Temperature

In statistical physics, noise intensity $D$ maps to temperature $T$:
$D = k_B T / \gamma$ where $\gamma$ is the friction coefficient. The
Kuramoto model with noise is equivalent to the XY model in the
overdamped limit. The synchronisation transition is then the
Berezinskii-Kosterlitz-Thouless (BKT) transition in 2D.

### 2.4 Stochastic Resonance in Networks

For networks with heterogeneous topology (not all-to-all), stochastic
resonance is structure-dependent:

| Topology | Optimal $D^*$ | $R$ enhancement | Mechanism |
|----------|---------------|-----------------|-----------|
| All-to-all | $KR_{\text{det}}/2$ | 5–15% | Phase diffusion into basin |
| Ring (nearest-neighbour) | Higher | 10–30% | Defect healing |
| Scale-free | Lower for hubs | 5–20% | Hub-driven nucleation |
| Small-world | Similar to all-to-all | 8–18% | Shortcut-aided diffusion |
| SCPN (block-diagonal) | Per-layer estimate | Variable | Layer-dependent |

The `find_optimal_noise` function works for any topology — it numerically
sweeps $D$ regardless of the coupling structure.

### 2.5 Noise and Chimera States

Adding noise to a system exhibiting chimera states (coexisting coherent
and incoherent populations) has complex effects:
- Low noise: chimera pattern preserved but boundaries fluctuate
- Moderate noise: chimera can be stabilised (noise sustains the incoherent
  population against drift toward coherence)
- High noise: chimera destroyed, system becomes fully incoherent

This makes noise a potential control parameter for chimera engineering.

### 2.6 Euler-Maruyama Order of Convergence

The splitting scheme (deterministic step + noise) has strong convergence
order 0.5 and weak convergence order 1.0 (standard Euler-Maruyama). For
higher accuracy:
- Milstein scheme: strong order 1.0 (not implemented — requires computing
  derivative of noise coefficient, which is constant here → equivalent to
  Euler-Maruyama for additive noise)
- Higher-order Runge-Kutta-Maruyama: not implemented

For additive noise (as in this module), Euler-Maruyama = Milstein.

### 2.7 Noise-Induced Transitions

Beyond stochastic resonance, noise can cause qualitative transitions:
- **Kramers escape:** noise-induced switching between bistable synchronised
  states ($\theta_{\text{lock}} = 0$ vs $\theta_{\text{lock}} = \pi$).
  Rate: $\sim e^{-\Delta V / D}$ (Arrhenius law).
- **Coherence resonance:** in excitable systems near Hopf, noise can
  create regular oscillations from a stable fixed point.
- **Noise-induced synchronisation:** two uncoupled ($K = 0$) oscillators
  driven by the same noise $dW$ can synchronise (common-noise effect).

---

## 3. Pipeline Position

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
│ UPDEEngine  │────→│ StochasticInjector│────→│ θ + noise    │
│ step()       │     │ inject(θ, dt)    │     │ → wrapped    │
│ deterministic│     │ √(2D·dt)·ξ      │     │ [0, 2π)      │
└──────────────┘     └──────────────────┘     └──────────────┘

┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
│ omegas, knm  │────→│ find_optimal_    │────→│ NoiseProfile │
│ phases_init  │     │ noise()          │     │ .D, .R_ach,  │
│ engine       │     │ sweep D values   │     │ .R_det       │
└──────────────┘     └──────────────────┘     └──────────────┘
```

The injector operates BETWEEN engine steps — it is not part of the
engine itself. This composable design allows noise to be added to any
engine (UPDEEngine, StuartLandauEngine, InertialKuramotoEngine, etc.).

---

## 4. Features

### 4.1 Three Components

| Component | Purpose |
|-----------|---------|
| `StochasticInjector` | Add calibrated noise per step |
| `find_optimal_noise` | Sweep D, find stochastic resonance peak |
| `optimal_D` | Analytical estimate of $D^*$ |

### 4.2 Composable Design

The injector is engine-agnostic — use with any SPO engine:
```python
phases = engine.step(phases, omegas, knm, zeta, psi, alpha)
phases = injector.inject(phases, engine._dt)
```

### 4.3 Reproducibility

`StochasticInjector(D, seed=42)` creates a seeded RNG. Same seed
→ identical noise realisation → reproducible simulations.

### 4.4 Adaptive D

The `D` property is settable: `injector.D = new_value`. Enables
time-varying noise schedules (e.g., simulated annealing).

---

## 5. Usage Examples

### 5.1 Basic Noise Injection

```python
import numpy as np
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.stochastic import StochasticInjector
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

N = 32
rng = np.random.default_rng(42)
phases = rng.uniform(0, 2 * np.pi, N)
omegas = rng.normal(0, 0.5, N)  # spread frequencies
knm = np.full((N, N), 1.5 / N)
np.fill_diagonal(knm, 0.0)
alpha = np.zeros((N, N))

engine = UPDEEngine(N, dt=0.01, method="rk4")
injector = StochasticInjector(D=0.1, seed=42)

for step in range(1000):
    phases = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)
    phases = injector.inject(phases, engine._dt)

R, _ = compute_order_parameter(phases)
print(f"Noisy R = {R:.4f}")
```

### 5.2 Stochastic Resonance Sweep

```python
from scpn_phase_orchestrator.upde.stochastic import find_optimal_noise

profile = find_optimal_noise(
    engine, phases, omegas, knm, alpha,
    D_range=np.linspace(0, 1, 20),
    n_steps=1000,
)
print(f"Optimal D = {profile.D:.4f}")
print(f"R (no noise) = {profile.R_deterministic:.4f}")
print(f"R (optimal) = {profile.R_achieved:.4f}")
# Expect R_achieved > R_deterministic for stochastic resonance
```

### 5.3 Analytical Self-Consistency

```python
from scpn_phase_orchestrator.upde.stochastic import _self_consistency_R

K = 3.0
for D in [0.0, 0.5, 1.0, 2.0, 5.0]:
    R_theory = _self_consistency_R(K, D)
    print(f"D={D:.1f}: R_theory={R_theory:.4f}")
# R decreases as D increases; R→0 when D > K/2
```

### 5.4 Time-Varying Noise (Annealing)

```python
injector = StochasticInjector(D=1.0, seed=0)
for step in range(2000):
    phases = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)
    # Anneal: decrease noise over time
    injector.D = max(0.0, 1.0 - step / 2000.0)
    phases = injector.inject(phases, engine._dt)
# Starts noisy (exploration), ends deterministic (exploitation)
```

### 5.5 Noise Robustness Test

```python
# Test how robust synchronisation is to noise
D_values = np.logspace(-3, 1, 20)
R_at_D = []
for D in D_values:
    p = phases.copy()
    inj = StochasticInjector(D, seed=42)
    for _ in range(500):
        p = engine.step(p, omegas, knm, 0.0, 0.0, alpha)
        p = inj.inject(p, engine._dt)
    R, _ = compute_order_parameter(p)
    R_at_D.append(R)

# R_at_D shows the noise sensitivity curve
# Sharp drop = fragile sync; gradual = robust
```

### 5.6 Analytical vs Numerical Comparison

```python
from scpn_phase_orchestrator.upde.stochastic import (
    _self_consistency_R, optimal_D,
)

K = 3.0
R_det = 0.85  # from deterministic simulation
D_star = optimal_D(K, R_det)
R_theory = _self_consistency_R(K, D_star)
print(f"D* = {D_star:.4f}")
print(f"R at D* (theory) = {R_theory:.4f}")
print(f"R without noise = {R_det:.4f}")
```

### 5.7 With Stuart-Landau Engine

```python
from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

N = 16
sl_engine = StuartLandauEngine(N, dt=0.01, method="rk4")
injector = StochasticInjector(D=0.05, seed=42)

state = np.concatenate([rng.uniform(0, 2*np.pi, N), np.ones(N)])
for _ in range(1000):
    state = sl_engine.step(state, omegas[:N], np.ones(N),
                           knm[:N,:N], knm[:N,:N], 0.0, 0.0,
                           alpha[:N,:N])
    # Inject noise only into phases, not amplitudes
    state[:N] = injector.inject(state[:N], 0.01)
```

### 5.8 Noise Scaling Law Verification

```python
# Verify D_c = K/2 numerically
K_test = 2.0
D_values = np.linspace(0, 2, 30)
R_measured = []

for D in D_values:
    R_th = _self_consistency_R(K_test, D)
    R_measured.append(R_th)

# R should drop to 0 at D = K/2 = 1.0
critical_idx = next(i for i, r in enumerate(R_measured) if r < 0.01)
D_c_numerical = D_values[critical_idx]
print(f"D_c (theory) = {K_test/2:.2f}")
print(f"D_c (numerical) = {D_c_numerical:.2f}")
```

### 5.9 Multiple Noise Realisations

```python
# Average R over 20 noise realisations for statistical significance
R_samples = []
for seed_val in range(20):
    p = phases.copy()
    inj = StochasticInjector(D=0.2, seed=seed_val)
    for _ in range(500):
        p = engine.step(p, omegas, knm, 0.0, 0.0, alpha)
        p = inj.inject(p, engine._dt)
    R, _ = compute_order_parameter(p)
    R_samples.append(R)

R_mean = np.mean(R_samples)
R_std = np.std(R_samples)
print(f"R = {R_mean:.4f} ± {R_std:.4f} (20 realisations)")
```

---

## 6. Technical Reference

### 6.1 Module API

::: scpn_phase_orchestrator.upde.stochastic
    options:
        show_root_heading: true
        members_order: source

### 6.2 StochasticInjector

| Method/Property | Description |
|-----------------|-------------|
| `__init__(D, seed)` | Create with noise intensity $D$ and optional seed |
| `inject(phases, dt) → NDArray` | Add $\sqrt{2D\,dt} \cdot N(0,1)$ noise, wrap to $[0, 2\pi)$ |
| `D` (property) | Get/set noise intensity |

### 6.3 find_optimal_noise

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `engine` | `UPDEEngine` | — | Integration engine |
| `phases_init` | `NDArray` (N,) | — | Starting phases |
| `omegas` | `NDArray` (N,) | — | Natural frequencies |
| `knm` | `NDArray` (N,N) | — | Coupling matrix |
| `alpha` | `NDArray` (N,N) | — | Phase-lag |
| `D_range` | `NDArray` or None | auto | Noise values to sweep |
| `n_steps` | `int` | `500` | Steps per D value |
| `seed` | `int` | `42` | RNG seed |
| **Returns** | `NoiseProfile` | | `.D`, `.R_achieved`, `.R_deterministic` |

### 6.4 NoiseProfile

| Field | Type | Description |
|-------|------|-------------|
| `D` | `float` | Optimal noise intensity |
| `R_achieved` | `float` | $R$ at optimal $D$ |
| `R_deterministic` | `float` | $R$ at $D = 0$ |

---

## 7. Performance Benchmarks

### 7.1 Injection Overhead

`inject()` adds one `rng.standard_normal(N)` call + element-wise ops:

| N | inject() time (µs) | vs engine step() | Overhead |
|---|---------------------|-------------------|----------|
| 16 | 3.5 | 12 µs (Euler) | ~30% |
| 64 | 5.2 | 57 µs (Euler) | ~9% |
| 256 | 12 | 1059 µs (Euler) | ~1% |
| 1024 | 38 | 18500 µs (Euler) | ~0.2% |

Noise injection is negligible for $N > 64$.

### 7.2 find_optimal_noise Cost

Total: $|D_{\text{range}}| \times n_{\text{steps}} \times O(N^2)$.

| N | n_D | n_steps | Time (Rust engine) |
|---|-----|---------|---------------------|
| 16 | 11 | 500 | ~0.2s |
| 64 | 20 | 1000 | ~4s |
| 256 | 20 | 1000 | ~80s |

### 7.3 Complexity

| Function | Time | Space |
|----------|------|-------|
| `inject()` | $O(N)$ | $O(N)$ noise array |
| `_self_consistency_R()` | $O(1)$ (100 iter max) | $O(1)$ |
| `find_optimal_noise()` | $O(|D| \cdot n \cdot N^2)$ | $O(N^2)$ per trial |

### 7.4 RNG Performance

NumPy's `default_rng` (PCG64) is fast — the bottleneck is always the
coupling computation in the engine, not the noise generation:

| N | RNG time (µs) | Engine step (µs, Euler Rust) | Ratio |
|---|--------------|-------------------------------|-------|
| 16 | 1.2 | 12 | 10% |
| 64 | 2.8 | 57 | 5% |
| 256 | 8.5 | 1059 | 0.8% |
| 1024 | 32 | 18500 | 0.2% |

### 7.5 Recommended Settings for find_optimal_noise

| Goal | n_D | n_steps | Expected time (N=64, Rust) |
|------|-----|---------|----------------------------|
| Quick estimate | 5 | 200 | ~0.3s |
| Standard sweep | 11 | 500 | ~2s |
| Fine resolution | 30 | 1000 | ~12s |
| Publication quality | 50 | 2000 | ~40s |

Use `optimal_D(K, R_det)` for instant analytical estimate, then
`find_optimal_noise` for numerical validation.

### 7.6 Noise Floor vs Signal

For the noise term $\sqrt{2D \cdot dt}$ to be physically meaningful
relative to the deterministic step $f \cdot dt$:

$$
\text{Signal-to-noise ratio} = \frac{|f| \cdot dt}{\sqrt{2D \cdot dt}}
= |f| \sqrt{\frac{dt}{2D}}
$$

For SNR > 1 (noise smaller than signal): $D < |f|^2 \cdot dt / 2$.
With typical $|f| \sim K \sim 3$ and $dt = 0.01$: $D < 0.045$.
Larger $D$ makes noise the dominant term — useful for exploring
phase space but destroys deterministic dynamics.

---

## 8. Citations

1. **Acebrón J.A., Bonilla L.L., Pérez Vicente C.J., Ritort F.,
   Spigler R.** (2005). The Kuramoto model: A simple paradigm for
   synchronization phenomena. *Rev. Mod. Phys.* **77**(1):137–185.
   doi:10.1103/RevModPhys.77.137. Eq. (12): self-consistency with noise.

2. **Tselios K., Georgiou A., et al.** (2025). Stochastic resonance in
   Kuramoto networks. *Nonlinear Dynamics* (preprint).

3. **Sakaguchi H.** (1988). Cooperative phenomena in coupled oscillator
   systems under external fields. *Progress of Theoretical Physics*
   **79**(1):39–46. doi:10.1143/PTP.79.39

4. **Tönjes R., Blasius B.** (2007). Perturbation analysis of the
   Kuramoto phase-diffusion equation subject to quenched frequency
   disorder. *Physical Review E* **76**(6):066202.
   doi:10.1103/PhysRevE.76.066202

5. **Strogatz S.H., Mirollo R.E.** (1991). Stability of incoherence
   in a population of coupled oscillators. *Journal of Statistical
   Physics* **63**(3–4):613–635. doi:10.1007/BF01029202

---

## Test Coverage

- `tests/test_stochastic_engine.py` — 21 tests: injector D validation,
  inject shape, inject with D=0 no-op, noise scales with D, phase
  wrapping, seed reproducibility, find_optimal_noise returns NoiseProfile,
  optimal D > 0 for spread frequencies, self_consistency_R bounds,
  stochastic resonance (R_achieved > R_deterministic)

Total: **21 tests**.

---

## Source

- Python: `src/scpn_phase_orchestrator/upde/stochastic.py` (131 lines)
- No Rust backend (noise is $O(N)$, negligible vs $O(N^2)$ coupling)
