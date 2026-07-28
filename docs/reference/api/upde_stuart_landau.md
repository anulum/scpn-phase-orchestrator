# Stuart-Landau Engine — Phase-Amplitude Dynamics

The `StuartLandauEngine` extends the Kuramoto model from pure phase dynamics
to coupled **phase-amplitude** oscillators. Each oscillator has both a phase
$\theta_i$ and an amplitude $r_i$, governed by the Stuart-Landau normal form.
This enables modelling of amplitude death, oscillation quenching, amplitude
chimeras, and the Hopf bifurcation — phenomena impossible in pure-phase models.

The Stuart-Landau oscillator is the generic normal form for any system near
a supercritical Hopf bifurcation (Strogatz 2015). It is the simplest model
that captures both phase synchronisation AND amplitude dynamics.

---

## 1. Mathematical Formalism

### 1.1 The Coupled Stuart-Landau System

Each oscillator $i$ has complex amplitude $W_i = r_i e^{i\theta_i}$ governed by:

$$
\dot{W}_i = (\mu_i + i\omega_i - |W_i|^2) W_i + \text{coupling}
$$

In polar coordinates $(r_i, \theta_i)$:

**Phase equation:**
$$
\dot{\theta}_i = \omega_i + \sum_{j=1}^{N} K_{ij} \sin(\theta_j - \theta_i - \alpha_{ij}) + \zeta\sin(\Psi - \theta_i)
$$

**Amplitude equation:**
$$
\dot{r}_i = (\mu_i - r_i^2) r_i + \epsilon \sum_{j=1}^{N} K^r_{ij} r_j \cos(\theta_j - \theta_i - \alpha_{ij})
$$

where:

| Symbol | Description | Range |
|--------|-------------|-------|
| $\theta_i$ | Phase | $[0, 2\pi)$ |
| $r_i$ | Amplitude | $[0, \infty)$, clamped to $\geq 0$ |
| $\omega_i$ | Natural frequency | rad/s |
| $\mu_i$ | Hopf bifurcation parameter | $\mu > 0$: oscillating, $\mu < 0$: damped |
| $K_{ij}$ | Phase coupling matrix | dimensionless |
| $K^r_{ij}$ | Amplitude coupling matrix | dimensionless |
| $\alpha_{ij}$ | Sakaguchi phase-lag | radians |
| $\epsilon$ | Amplitude coupling strength | dimensionless |
| $\zeta, \Psi$ | External drive | amplitude, phase |

### 1.2 The Hopf Bifurcation Parameter $\mu$

The term $(\mu_i - r_i^2) r_i$ is the Stuart-Landau normal form:

- $\mu_i > 0$: **Supercritical Hopf** — oscillator has a stable limit cycle
  at $r_i = \sqrt{\mu_i}$. Perturbations decay back to the limit cycle.
- $\mu_i = 0$: **Bifurcation point** — oscillator is at the onset of oscillation.
- $\mu_i < 0$: **Subcritical** — oscillator decays to $r_i = 0$ (fixed point).
  The oscillation is dead.

This enables modelling **amplitude death**: when coupling drives $r_i \to 0$
for some oscillators, they stop oscillating. This is a coupling-induced
phenomenon not captured by pure-phase Kuramoto.

### 1.3 Amplitude Coupling

The cos term $K^r_{ij} r_j \cos(\theta_j - \theta_i - \alpha_{ij})$ models
amplitude interaction:

- **In-phase neighbours** ($\theta_j \approx \theta_i$): $\cos \approx 1$ →
  amplitudes mutually reinforce
- **Anti-phase neighbours** ($\theta_j \approx \theta_i + \pi$): $\cos \approx -1$ →
  amplitudes mutually suppress

The scale factor $\epsilon$ controls the relative strength of amplitude
coupling vs the intrinsic $(\mu - r^2)r$ dynamics.

### 1.4 State Vector Layout

The engine stores both phase and amplitude in a single $(2N,)$ array:

$$
\text{state} = [\theta_1, \ldots, \theta_N, r_1, \ldots, r_N]
$$

This layout enables a unified RK4/RK45 integrator — both variables are
advanced simultaneously through each Runge-Kutta stage.

### 1.5 Post-Step Clamping

After each integration step:
1. $\theta_i \gets \theta_i \bmod 2\pi$ (phase wrapping on $S^1$)
2. $r_i \gets \max(r_i, 0)$ (amplitude non-negativity)

Amplitude clamping is necessary because intermediate RK stages can
produce negative $r$, which is physically meaningless. The clamping
preserves the RK4 order of accuracy for smooth solutions where $r > 0$
throughout.

### 1.6 Order Parameter

The amplitude-weighted Kuramoto order parameter:

$$
Z = \frac{1}{N} \sum_{i=1}^{N} r_i e^{i\theta_i}
$$

$R = |Z|$ combines phase coherence with amplitude weighting — oscillators
with larger amplitude contribute more to the collective field.
`compute_order_parameter(state)` returns $(R, \psi)$.

---

## 2. Theoretical Context

### 2.1 Historical Background

The Stuart-Landau equation (Stuart 1960, Landau 1944) describes the
universal dynamics near a supercritical Hopf bifurcation. It was
connected to coupled oscillator networks by Matthews, Mirollo & Strogatz
(1991) who studied amplitude death in globally coupled oscillators.

Daido (1993) and Nakagawa & Kuramoto (1993) showed that amplitude
dynamics qualitatively change the synchronisation landscape: new phenomena
like oscillation death, amplitude chimeras, and nontrivial amplitude
patterns emerge that have no counterpart in the pure-phase model.

### 2.2 Amplitude Death vs Oscillation Death

| Phenomenon | Mechanism | $r_i$ | Phases |
|------------|-----------|--------|--------|
| Amplitude death | Coupling stabilises $r=0$ fixed point | All $r_i \to 0$ | Undefined |
| Oscillation death | Coupling creates new non-zero fixed points | $r_i \to r^* \neq \sqrt{\mu}$ | Locked at fixed values |
| Partial death | Some oscillators die, others continue | Mixed | Mixed |

Amplitude death requires frequency mismatch + coupling above a threshold:
$K > K_{\text{AD}}(\Delta\omega)$. It is easier to achieve with
Sakaguchi phase-lag $\alpha \neq 0$.

### 2.3 Role in SCPN

The Stuart-Landau engine models SCPN layers where oscillation amplitude
is physically meaningful:

- **Layer 2 (Neurochemical):** neurotransmitter concentration oscillations
  with amplitude-dependent effects
- **Layer 5 (Psychoemotional):** emotional intensity (amplitude) coupled
  to emotional phase
- **Layer 13 (Source Field):** field strength (amplitude) modulating
  phase coupling

### 2.4 Amplitude Death Conditions

For two oscillators with frequency mismatch $\Delta\omega = |\omega_1 - \omega_2|$
and all-to-all coupling $K$, amplitude death occurs when (Reddy, Sen &
Johnston 1998):

$$
K > K_{\text{AD}} = \frac{\Delta\omega}{2} \cdot \frac{1}{\sin(\alpha)}
$$

for Sakaguchi phase-lag $0 < \alpha < \pi/2$. Without phase-lag ($\alpha = 0$),
amplitude death requires infinite coupling — the Sakaguchi term is essential.

For $N$ oscillators with distributed frequencies $g(\omega)$:
- Frequency mismatch increases the critical coupling for synchronisation
- But DECREASES the threshold for amplitude death
- Result: there exists a "death island" in $(K, \Delta\omega)$ space where
  oscillations are quenched

### 2.5 Normal Form Theory

The Stuart-Landau equation $\dot{W} = (\mu + i\omega)W - |W|^2 W$ is the
**universal** normal form near a supercritical Hopf bifurcation. Any
smooth dynamical system with a pair of complex-conjugate eigenvalues crossing
the imaginary axis can be reduced to this form via centre-manifold reduction
+ normal form transformation (Guckenheimer & Holmes 1983).

This means the SPO Stuart-Landau engine can model:
- Neural oscillators (Morris-Lecar, Fitzhugh-Nagumo near Hopf)
- Chemical oscillators (Belousov-Zhabotinsky near onset)
- Laser arrays (coupled mode theory)
- Any system with oscillatory instability

The $\mu$ parameter maps to the distance from the bifurcation point in
the original system's parameter space.

### 2.6 Comparison with UPDEEngine

| Feature | `UPDEEngine` | `StuartLandauEngine` |
|---------|-------------|---------------------|
| State dimension | $N$ (phase only) | $2N$ (phase + amplitude) |
| Can model amplitude death | No | Yes |
| Can model oscillation quenching | No | Yes |
| Computational cost | $O(N^2)$ per step | $O(N^2)$ per step (same) |
| Number of parameters | 4 (ω, K, α, ζ) | 7 (ω, μ, K, K_r, α, ζ, ε) |

Use `StuartLandauEngine` when amplitude effects matter. Use `UPDEEngine`
when amplitude is constant (i.e., all oscillators on their limit cycle).

---

## 3. Pipeline Position

```
┌──────────────┐     ┌──────────────────────┐     ┌──────────────┐
│ coupling/    │────→│ StuartLandauEngine   │────→│ monitor/     │
│ knm.py       │     │                      │     │ order_params │
│ (K, K_r)     │     │ State: [θ₁..θ_N,    │     │ chimera      │
└──────────────┘     │         r₁..r_N]    │     └──────────────┘
                     │ Euler/RK4/RK45      │
┌──────────────┐     │                      │     ┌──────────────┐
│ oscillators/ │────→│ ω_i, μ_i             │     │ nn/          │
│ base.py      │     └──────────────────────┘     │ stuart_landau│
│ (ω, μ)       │                                  │ _layer.py    │
└──────────────┘                                  └──────────────┘
```

**Inputs:**
- `state` (2N,) — $[\theta_1, \ldots, \theta_N, r_1, \ldots, r_N]$
- `omegas` (N,) — natural frequencies
- `mu` (N,) — Hopf bifurcation parameters
- `knm` (N,N) — phase coupling
- `knm_r` (N,N) — amplitude coupling
- `alpha` (N,N) — phase-lag
- `zeta`, `psi` — external drive
- `epsilon` — amplitude coupling scale

**Outputs:**
- `state` (2N,) — updated $[\theta, r]$ vector

---

## 4. Features

### 4.1 Three Integration Methods

Same as `UPDEEngine`: Euler, RK4, Dormand-Prince RK45 (adaptive).
Applied to the combined $(2N)$-dimensional system.

### 4.2 Amplitude-Weighted Order Parameter

`compute_order_parameter(state)` returns the amplitude-weighted $R$,
not just the phase-only $R$. Dead oscillators ($r_i = 0$) contribute
nothing to the order parameter.

### 4.3 Mean Amplitude Diagnostic

`compute_mean_amplitude(state)` returns $\bar{r} = \frac{1}{N}\sum r_i$.
Tracking $\bar{r}(t)$ detects amplitude death ($\bar{r} \to 0$).

### 4.4 Full Input Validation

All 8 input arrays/scalars validated for shape and finiteness.

### 4.5 Rust Acceleration

`PyStuartLandauStepper` (spo-kernel) accelerates `step()`. Speedup ~10.8x.

---

## 5. Usage Examples

### 5.1 Basic Coupled Oscillators

```python
import numpy as np
from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

N = 8
rng = np.random.default_rng(42)

# Initial state: random phases, amplitudes near limit cycle
theta0 = rng.uniform(0, 2 * np.pi, N)
r0 = np.ones(N) * 0.8
state = np.concatenate([theta0, r0])

omegas = np.ones(N)
mu = np.ones(N) * 1.0  # μ=1 → limit cycle at r=1

knm = np.full((N, N), 1.0 / N)
np.fill_diagonal(knm, 0.0)
knm_r = knm.copy()
alpha = np.zeros((N, N))

engine = StuartLandauEngine(N, dt=0.01, method="rk4")

for step in range(500):
    state = engine.step(state, omegas, mu, knm, knm_r, 0.0, 0.0, alpha)

R, psi = engine.compute_order_parameter(state)
r_mean = engine.compute_mean_amplitude(state)
print(f"R = {R:.4f}, mean amplitude = {r_mean:.4f}")
```

### 5.2 Amplitude Death

```python
# Frequency mismatch + strong coupling → amplitude death
omegas_mixed = np.array([1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0])
mu_death = np.ones(N) * 0.5
knm_strong = np.full((N, N), 5.0 / N)
np.fill_diagonal(knm_strong, 0.0)

state = np.concatenate([rng.uniform(0, 2*np.pi, N), np.ones(N)])
engine = StuartLandauEngine(N, dt=0.01, method="rk4")

for step in range(2000):
    state = engine.step(state, omegas_mixed, mu_death, knm_strong,
                        knm_strong, 0.0, 0.0, alpha)

r_mean = engine.compute_mean_amplitude(state)
print(f"Amplitude death: mean r = {r_mean:.4f}")
# Expect r_mean ≈ 0 (oscillators quenched)
```

### 5.3 Hopf Bifurcation Sweep

```python
# Sweep μ from -1 to +2, track steady-state amplitude
mu_values = np.linspace(-1, 2, 30)
r_steady = []

for mu_val in mu_values:
    mu_arr = np.ones(N) * mu_val
    state = np.concatenate([np.zeros(N), np.ones(N) * 0.1])
    eng = StuartLandauEngine(N, dt=0.01, method="rk4")
    for _ in range(1000):
        state = eng.step(state, omegas, mu_arr, knm, knm_r, 0.0, 0.0, alpha)
    r_steady.append(engine.compute_mean_amplitude(state))

# r ≈ 0 for μ < 0, r ≈ √μ for μ > 0 (Hopf bifurcation)
```

### 5.4 Amplitude Chimera

```python
# Non-local coupling can create amplitude chimeras:
# some oscillators at full amplitude, others dead
N_chim = 64
knm_nonlocal = np.zeros((N_chim, N_chim))
for i in range(N_chim):
    for j in range(N_chim):
        dist = min(abs(i-j), N_chim - abs(i-j))
        if 0 < dist <= 10:
            knm_nonlocal[i, j] = 2.0 / 10

state = np.concatenate([
    rng.uniform(0, 2*np.pi, N_chim),
    np.ones(N_chim) + rng.normal(0, 0.1, N_chim),
])
eng = StuartLandauEngine(N_chim, dt=0.01, method="rk4")
mu_arr = np.ones(N_chim) * 1.0

for _ in range(3000):
    state = eng.step(state, np.ones(N_chim), mu_arr,
                     knm_nonlocal, knm_nonlocal, 0.0, 0.0,
                     np.zeros((N_chim, N_chim)))

amplitudes = state[N_chim:]
# Plot amplitudes — expect coherent + incoherent regions (chimera)
```

### 5.5 Adaptive RK45

```python
engine = StuartLandauEngine(N, dt=0.01, method="rk45", atol=1e-8, rtol=1e-5)
state = engine.step(state, omegas, mu, knm, knm_r, 0.0, 0.0, alpha)
print(f"Adaptive dt: {engine.last_dt:.6f}")
```

### 5.6 JAX Neural Network Layer

```python
# StuartLandauEngine feeds into nn/stuart_landau_layer.py for
# differentiable phase-amplitude dynamics in JAX
# See: docs/reference/api/nn.md for JAX integration
```

### 5.7 Amplitude Monitoring Over Time

```python
# Track amplitude evolution to detect death onset
r_history = []
state = np.concatenate([rng.uniform(0, 2*np.pi, N), np.ones(N)])
for step in range(2000):
    state = engine.step(state, omegas_mixed, mu_death,
                        knm_strong, knm_strong, 0.0, 0.0, alpha)
    r_history.append(engine.compute_mean_amplitude(state))

# Plot r_history — should show exponential decay if amplitude death
r_arr = np.array(r_history)
death_time = np.argmax(r_arr < 0.01) if np.any(r_arr < 0.01) else None
print(f"Amplitude death at step: {death_time}")
```

### 5.8 Phase-Amplitude Correlation

```python
# Measure whether amplitude and phase are correlated
state_final = state.copy()
phases = state_final[:N]
amplitudes = state_final[N:]
correlation = np.corrcoef(phases, amplitudes)[0, 1]
print(f"Phase-amplitude correlation: {correlation:.4f}")
# In amplitude chimera: expect strong negative correlation
# (high amplitude ↔ coherent phase region)
```

---

## 6. Technical Reference

### 6.1 Module API

::: scpn_phase_orchestrator.upde.stuart_landau
    options:
        show_root_heading: true
        members_order: source

### 6.2 Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_oscillators` | `int` | — | Number of oscillators $N$ |
| `dt` | `float` | — | Timestep |
| `method` | `str` | `"euler"` | `"euler"`, `"rk4"`, `"rk45"` |
| `atol` | `float` | `1e-6` | Absolute tolerance (RK45) |
| `rtol` | `float` | `1e-3` | Relative tolerance (RK45) |

### 6.3 step() Parameters

Every array must be a finite real numeric payload with its declared shape.
Boolean, complex, and numeric-string aliases are rejected before conversion,
including state passed to both diagnostic methods. Optional Rust state returns
replay the same source-type, shape, and finite contract before publication;
Rust `last_dt` must be a strictly positive finite real scalar.

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `state` | `(2N,)` | $[\theta_1, \ldots, \theta_N, r_1, \ldots, r_N]$ |
| `omegas` | `(N,)` | Natural frequencies |
| `mu` | `(N,)` | Hopf parameters |
| `knm` | `(N,N)` | Phase coupling |
| `knm_r` | `(N,N)` | Amplitude coupling |
| `zeta` | `float` | External drive strength |
| `psi` | `float` | External drive phase |
| `alpha` | `(N,N)` | Phase-lag matrix |
| `epsilon` | `float` | Amplitude coupling scale (default 1.0) |

### 6.4 Diagnostic Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `compute_order_parameter(state)` | `(R, ψ)` | Amplitude-weighted order parameter |
| `compute_mean_amplitude(state)` | `float` | $\bar{r} = \frac{1}{N}\sum r_i$ |
| `last_dt` | `float` | Last accepted dt (RK45) |

### 6.5 Exceptions

All `ValueError` — same patterns as `UPDEEngine` (shape, finiteness).

---

## 7. Performance Benchmarks

### 7.1 Rust Speedup

| N | Method | Python (µs/step) | Rust (µs/step) | Speedup |
|---|--------|-------------------|----------------|---------|
| 8 | euler | 45 | 4.2 | 10.7x |
| 8 | rk4 | 180 | 16.7 | 10.8x |
| 16 | euler | 85 | 8.1 | 10.5x |
| 64 | euler | 680 | 65 | 10.5x |
| 256 | euler | 11000 | 1050 | 10.5x |

### 7.2 vs UPDEEngine

State dimension is $2N$ instead of $N$, but the coupling computation
is still $O(N^2)$ (both sin and cos terms computed from the same
phase difference). Overhead vs UPDEEngine: ~1.5x per step (extra
amplitude coupling + cos evaluation).

### 7.3 Complexity

| Operation | Time | Space |
|-----------|------|-------|
| `_derivative` | $O(N^2)$ | $O(N^2)$ scratch + $O(N)$ state |
| `_euler_step` | $O(N^2)$ | $O(N)$ result |
| `_rk4_step` | $4 \times O(N^2)$ | $4 \times O(2N)$ stage copies |
| `_rk45_step` | $7 \times O(N^2)$ | $7 \times O(2N)$ stages |

### 7.4 Memory

Pre-allocated: `_phase_diff` $(N^2)$, `_sin_diff` $(N^2)$, `_cos_diff` $(N^2)$,
`_scratch_dtheta` $(N)$, `_scratch_dr` $(N)$, `_scratch_deriv` $(2N)$.
Total: $3N^2 + 4N$ floats. For RK45: add $7 \times 2N + 2N$ stage buffers.

| N | Scratch memory | RK45 stages | Total |
|---|---------------|-------------|-------|
| 8 | 1.5 KB | 0.9 KB | ~2.4 KB |
| 16 | 6 KB | 1.8 KB | ~7.8 KB |
| 64 | 96 KB | 7.2 KB | ~103 KB |
| 256 | 1.5 MB | 28 KB | ~1.5 MB |

### 7.5 Recommended Settings

| Use case | Method | dt | Notes |
|----------|--------|-----|-------|
| Quick exploration | euler | 0.01 | Fast but $O(\Delta t)$ error |
| Production SCPN | rk4 | 0.01 | Balanced accuracy/speed |
| Amplitude chimera research | rk45 | auto | Handles stiffness near death |
| JAX training loop | euler | 0.01 | Differentiable, fast forward |

### 7.6 Amplitude Death Detection Speed

Amplitude death typically requires $O(1/\mu)$ time units to manifest.
For $\mu = 0.5$ and $dt = 0.01$: ~200 steps. For $\mu = 0.1$: ~1000 steps.
Budget simulations accordingly.

### 7.7 Numerical Stability of Amplitude Clamping

The `r ≥ 0` clamp is applied AFTER each full step. During intermediate
RK4 stages, $r$ can go negative. The derivative evaluation uses
`r_clamped = max(r, 0)` for the coupling term, preventing sign flips
in the amplitude coupling. This is a pragmatic fix — for rigorous
treatment, switch to RK45 which adapts dt to keep $r$ non-negative
throughout.

---

## 8. Citations

1. **Stuart J.T.** (1960). On the non-linear mechanics of wave disturbances
   in stable and unstable parallel flows. *Journal of Fluid Mechanics*
   **9**(3):353–370. doi:10.1017/S002211206000116X

2. **Landau L.D.** (1944). On the problem of turbulence.
   *Doklady Akademii Nauk SSSR* **44**:339–342.

3. **Matthews P.C., Mirollo R.E., Strogatz S.H.** (1991). Dynamics of a
   large system of coupled nonlinear oscillators. *Physica D*
   **52**(2–3):293–331. doi:10.1016/0167-2789(91)90129-W

4. **Nakagawa N., Kuramoto Y.** (1993). Collective chaos in a population
   of globally coupled oscillators. *Progress of Theoretical Physics*
   **89**(2):313–323. doi:10.1143/ptp/89.2.313

5. **Strogatz S.H.** (2015). *Nonlinear Dynamics and Chaos: With Applications
   to Physics, Biology, Chemistry, and Engineering*. 2nd ed., Westview Press.
   Chapter 8: Bifurcations Revisited.

6. **Acebrón J.A. et al.** (2005). The Kuramoto model: A simple paradigm
   for synchronization phenomena. *Rev. Mod. Phys.* **77**(1):137–185.
   doi:10.1103/RevModPhys.77.137

7. **Reddy D.V.R., Sen A., Johnston G.L.** (1998). Time delay induced
   death in coupled limit cycle oscillators. *Physical Review Letters*
   **80**(23):5109–5112. doi:10.1103/PhysRevLett.80.5109

8. **Guckenheimer J., Holmes P.** (1983). *Nonlinear Oscillations,
   Dynamical Systems, and Bifurcations of Vector Fields*. Springer-Verlag.
   doi:10.1007/978-1-4612-1140-2

9. **Daido H.** (1993). A solvable model of coupled limit-cycle oscillators
   exhibiting partial perfect synchrony and novel frequency spectra.
   *Physica D* **69**(3–4):394–403. doi:10.1016/0167-2789(93)90102-7

---

## Test Coverage

- `tests/test_stuart_landau_engine.py` — 26 tests: step shape, euler/rk4/rk45,
  amplitude clamping, NaN rejection, order parameter, mean amplitude,
  μ>0 limit cycle convergence, μ<0 decay, amplitude death
- `tests/test_stuart_landau_coupling.py` — 12 tests: phase coupling, amplitude
  coupling, epsilon scaling, knm_r symmetry
- `tests/test_stuart_landau_parity.py` — 7 tests: Rust vs Python numerical parity
- `tests/test_stuart_landau_validation.py` — public scalar/array ingress,
  observable-state, optional Rust state-output, and adaptive-timestep contracts
- `tests/test_stuart_landau_nn.py` — 24 tests: JAX layer integration
- `tests/test_stuart_landau_binding.py` — 15 tests: YAML binding
- `tests/test_stuart_landau_cli.py` — 15 tests: CLI integration
- `tests/test_stuart_landau_imprint.py` — 7 tests: imprint feedback

Focused engine contract slice: **92 collected, 85 passed, 7 expected optional
Rust skips**, with exact **100% line and branch coverage** for
`upde/stuart_landau.py`.

---

## Source

- Python: `src/scpn_phase_orchestrator/upde/stuart_landau.py` (523 lines)
- Rust: `spo-kernel/crates/spo-engine/src/stuart_landau.rs`
- FFI: `spo-kernel/crates/spo-ffi/src/lib.rs` (PyStuartLandauStepper)
