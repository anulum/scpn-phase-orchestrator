# UPDE Engine — Core Phase Integration

The `UPDEEngine` is the central integration kernel of the SCPN Phase Orchestrator.
It numerically solves the **Universal Phase Dynamics Equation (UPDE)**, a
generalised Kuramoto model with Sakaguchi phase-lag coupling and external drive.

Every simulation in SPO — from 16-oscillator SCPN layer models to 1000-node
networks — passes through this engine. It is the computational heart of the
system.

---

## 1. Mathematical Formalism

### 1.1 The UPDE

The Universal Phase Dynamics Equation governs the evolution of N coupled
oscillators on the circle $S^1$:

$$
\frac{d\theta_i}{dt} = \omega_i
+ \sum_{j=1}^{N} K_{ij} \sin(\theta_j - \theta_i - \alpha_{ij})
+ \zeta \sin(\Psi - \theta_i)
$$

where:

| Symbol | Description | Units |
|--------|-------------|-------|
| $\theta_i$ | Phase of oscillator $i$ | radians, $[0, 2\pi)$ |
| $\omega_i$ | Natural frequency of oscillator $i$ | rad/s |
| $K_{ij}$ | Coupling strength from oscillator $j$ to $i$ | dimensionless |
| $\alpha_{ij}$ | Phase lag (Sakaguchi parameter) from $j$ to $i$ | radians |
| $\zeta$ | External drive amplitude | dimensionless |
| $\Psi$ | External drive target phase | radians |
| $N$ | Number of oscillators | integer |

The three terms represent:

1. **Intrinsic dynamics** $\omega_i$ — each oscillator's natural tendency
2. **Pairwise coupling** $K_{ij}\sin(\theta_j - \theta_i - \alpha_{ij})$ —
   Sakaguchi-Kuramoto interaction (Sakaguchi & Kuramoto 1986)
3. **External drive** $\zeta\sin(\Psi - \theta_i)$ — global forcing field

### 1.2 Integration Methods

The engine implements three integration schemes:

#### Euler (first-order)

$$
\theta_i^{n+1} = \left(\theta_i^n + \Delta t \cdot f(\theta^n)\right) \bmod 2\pi
$$

Simplest, fastest per step, but requires small $\Delta t$ for accuracy.
Error $O(\Delta t)$.

#### Classical RK4 (fourth-order)

$$
\theta^{n+1} = \theta^n + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4) \bmod 2\pi
$$

where $k_1 = f(\theta^n)$, $k_2 = f(\theta^n + \frac{\Delta t}{2}k_1)$, etc.
Error $O(\Delta t^4)$. Four derivative evaluations per step.

#### Dormand-Prince RK45 (adaptive fifth-order)

7-stage method with embedded 4th-order error estimator (Dormand & Prince 1980).
Coefficients from Hairer, Nørsett & Wanner, *Solving Ordinary Differential
Equations I*, Table 5.2.

Adaptive step-size control via PI controller:

$$
h_{new} = 0.9 \cdot h \cdot \left(\frac{1}{\text{err\_norm}}\right)^{1/5}
$$

with mixed tolerance scaling:

$$
\text{err\_norm} = \max_i \frac{|y_5^i - y_4^i|}{\text{atol} + \text{rtol} \cdot \max(|\theta^n_i|, |y_5^i|)}
$$

Step accepted when $\text{err\_norm} \leq 1$. Rejected steps shrink $h$ with
exponent $-1/4$ and safety factor 0.9.

### 1.3 Phase Wrapping

All methods apply $\theta \bmod 2\pi$ after each step, keeping phases on the
circle topology $S^1$. This is mathematically exact for the Kuramoto model
since the coupling function $\sin(\cdot)$ is $2\pi$-periodic.

---

## 2. Theoretical Context

### 2.1 Historical Background

The Kuramoto model (Kuramoto 1975, 1984) is the canonical mean-field model
for coupled oscillator synchronisation. Originally formulated with uniform
all-to-all coupling $K/N$, it was generalised to arbitrary coupling
topologies $K_{ij}$ by Strogatz (2000).

The critical coupling strength $K_c$ separates incoherent ($K < K_c$,
$R \approx 0$) from partially synchronised ($K > K_c$, $R > 0$) regimes.
For identical oscillators with all-to-all coupling, $K_c = 0$ and any
positive coupling leads to full synchronisation ($R = 1$). For Lorentzian
frequency distributions $g(\omega) = \frac{\gamma/\pi}{\omega^2 + \gamma^2}$,
the Ott-Antonsen (2008) reduction gives $K_c = 2\gamma$.

The Sakaguchi extension (Sakaguchi & Kuramoto 1986) introduced the phase-lag
parameter $\alpha_{ij}$, which models asymmetric or frustrated coupling —
essential for biological neural networks where synaptic delays create
effective phase shifts. When $\alpha_{ij} = \alpha$ (uniform), the coupling
becomes $K\sin(\theta_j - \theta_i - \alpha)$, which shifts the synchronisation
manifold: oscillators lock with phase difference $\alpha$ rather than zero.
For $|\alpha| > \pi/4$, synchronisation becomes impossible regardless of $K$
(frustration threshold).

The external drive term $\zeta\sin(\Psi - \theta_i)$ enables entrainment
to an external periodic signal. The Arnold tongue of entrainment has width
proportional to $\zeta$: for identical oscillators with detuning $\Delta\omega$,
entrainment occurs when $\zeta > |\Delta\omega|$. Applications include:

- **Brain-computer interfaces** — closed-loop EEG entrainment via LSLBCIBridge
- **Power grid stabilisation** — frequency regulation via InertialKuramotoEngine
- **Gaian mesh coupling** — distributed inter-instance synchronisation (Layer 12)
- **Manufacturing SPC** — process variable coherence monitoring

### 2.2 Order Parameter

The Kuramoto order parameter quantifies macroscopic synchronisation:

$$
R e^{i\psi} = \frac{1}{N} \sum_{j=1}^{N} e^{i\theta_j}
$$

where $R \in [0, 1]$ measures coherence (0 = incoherent, 1 = fully locked)
and $\psi$ is the mean phase. The engine delegates this computation to
`order_params.compute_order_parameter()`.

### 2.3 Numerical Considerations

**Stiffness.** The coupling term $K_{ij}\sin(\cdot)$ is non-stiff for moderate
$K$, making explicit Runge-Kutta methods efficient. However, for very large
$K/N$ ratios (strong coupling), the system becomes stiff and RK45 adaptive
stepping automatically reduces $\Delta t$.

**Phase wrapping.** The $\bmod 2\pi$ operation is applied AFTER the full
integration step (not after each stage), preserving the integrator's order
of accuracy. This is correct because $\sin(\cdot)$ is periodic.

**Floating-point.** At extreme $N$ (>10,000), the $O(N^2)$ coupling sum
can accumulate rounding error. The Rust backend uses compensated summation
(Kahan algorithm) for N ≥ 1024.

### 2.2 Why UPDE?

The "Universal" in UPDE reflects that this single equation subsumes:
- Standard Kuramoto (set $\alpha = 0$, $\zeta = 0$)
- Sakaguchi-Kuramoto (set $\zeta = 0$)
- Externally driven Kuramoto (set $\alpha = 0$)
- SCPN 15+1 layer model (full $K_{ij}$, $\alpha_{ij}$, $\zeta$, $\Psi$)

The engine doesn't know about "layers" — it sees N oscillators with an
$N \times N$ coupling matrix. Layer structure is encoded in $K_{ij}$
(block-diagonal for intra-layer, off-diagonal for inter-layer coupling).

### 2.3 Relation to Other SPO Engines

| Engine | Extends UPDE with | Use case |
|--------|-------------------|----------|
| `UPDEEngine` | — (base) | Standard Kuramoto networks |
| `StuartLandauEngine` | Amplitude dynamics $\dot{r}_i$ | Amplitude death, oscillation quenching |
| `InertialKuramotoEngine` | Second-order $\ddot{\theta}_i$ | Power grids (swing equation) |
| `HypergraphEngine` | Higher-order interactions | Simplicial/hypergraph coupling |
| `SplittingEngine` | Strang operator splitting | Mixed stiff/non-stiff systems |
| `DelayedEngine` | Time-delayed coupling | Neural circuits with axonal delays |
| `SwarmalatorEngine` | Spatial position + phase | Mobile coupled oscillators |
| `StochasticEngine` | Wiener noise $\sigma dW_i$ | Noisy biological systems |
| `TorusEngine` | Geometric manifold | Phase dynamics on torus topology |
| `SheafEngine` | Sheaf-theoretic coupling | Category-theory formulation |

All specialised engines share the same `step()` → `run()` API pattern.

---

## 3. Pipeline Position

```
┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│ binding/      │────→│ UPDEEngine  │────→│ monitor/     │
│ loader.py     │     │ step()/run()│     │ order_params │
│ (YAML config) │     │             │     │ lyapunov     │
└──────────────┘     │  ↓ phases   │     │ chimera      │
                     │  ↓ R, ψ     │     └──────┬───────┘
┌──────────────┐     │             │            │
│ coupling/    │────→│  K_nm       │     ┌──────▼───────┐
│ knm.py       │     │  α_ij      │     │ supervisor/  │
│ plasticity   │     │  ω_i       │     │ policy.py    │
└──────────────┘     └─────────────┘     │ regimes.py   │
                                         └──────┬───────┘
┌──────────────┐                                │
│ oscillators/ │────→ ω_i (natural freq)  ┌─────▼────────┐
│ base.py      │                          │ actuation/   │
└──────────────┘                          │ mapper.py    │
                                          │ constraints  │
┌──────────────┐                          └──────────────┘
│ adapters/    │
│ (external    │───→ zeta, Psi (drive)
│  sensors)    │
└──────────────┘
```

**Inputs:**
- `phases` (N,) — current oscillator phases in $[0, 2\pi)$
- `omegas` (N,) — natural frequencies
- `knm` (N, N) — coupling matrix $K_{ij}$
- `alpha` (N, N) — phase-lag matrix $\alpha_{ij}$
- `zeta` (float) — external drive amplitude
- `psi` (float) — external drive phase

**Outputs:**
- `phases` (N,) — updated phases after integration step

---

## 4. Features

### 4.1 Integration Methods

| Method | Order | Evaluations/step | Adaptive | Best for |
|--------|-------|-------------------|----------|----------|
| `euler` | 1 | 1 | No | Fast prototyping, large N |
| `rk4` | 4 | 4 | No | Production accuracy |
| `rk45` | 5(4) | 7 | Yes | Variable-stiffness systems |

### 4.2 Input Validation

Every call to `step()` validates:
- Phase, omega, knm, alpha shapes match `n_oscillators`
- No NaN or Inf in any input array
- `zeta` and `psi` are finite scalars

This prevents silent corruption from propagating through the pipeline.

### 4.3 Rust Acceleration

When `spo-kernel` is installed, the engine automatically delegates to
`PyUPDEStepper` (Rust + Rayon). The Python implementation serves as
fallback and reference. Selection is automatic via `_HAS_RUST` flag.

### 4.4 Pre-allocated Scratch Arrays

The engine pre-allocates intermediate arrays (`_phase_diff`, `_sin_diff`,
`_scratch_dtheta`) at construction time to avoid per-step allocation.
For RK45, seven stage buffers (`_ks`) and an error buffer are also
pre-allocated.

### 4.5 Batch Execution

`run(phases, omegas, knm, zeta, psi, alpha, n_steps)` executes multiple
steps. With Rust backend, this avoids N round-trips across the FFI boundary.

---

## 5. Usage Examples

### 5.1 Basic Synchronisation

```python
import numpy as np
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

# 16 oscillators, random initial phases, identical frequencies
N = 16
rng = np.random.default_rng(42)
phases = rng.uniform(0, 2 * np.pi, N)
omegas = np.ones(N)  # identical natural frequencies

# All-to-all coupling at K = 2.0 (above critical coupling)
knm = np.full((N, N), 2.0 / N)
np.fill_diagonal(knm, 0.0)
alpha = np.zeros((N, N))  # no phase lag

engine = UPDEEngine(N, dt=0.01, method="rk4")
phases = engine.run(phases, omegas, knm, zeta=0.0, psi=0.0, alpha=alpha, n_steps=500)

R, psi = compute_order_parameter(phases)
print(f"Order parameter R = {R:.4f}")  # expect R ≈ 1.0 (synchronised)
```

### 5.2 Adaptive Integration

```python
engine = UPDEEngine(N, dt=0.01, method="rk45", atol=1e-8, rtol=1e-5)
phases = engine.step(phases, omegas, knm, zeta=0.0, psi=0.0, alpha=alpha)
print(f"Adaptive dt used: {engine.last_dt:.6f}")
```

### 5.3 External Drive (Entrainment)

```python
# Drive all oscillators towards phase π/2 with strength 0.5
phases = engine.run(
    phases, omegas, knm,
    zeta=0.5, psi=np.pi / 2,
    alpha=alpha, n_steps=1000,
)
```

### 5.4 Sakaguchi Phase-Lag

```python
# Frustration: α_ij = π/4 breaks perfect synchronisation
alpha = np.full((N, N), np.pi / 4)
np.fill_diagonal(alpha, 0.0)
phases = engine.run(phases, omegas, knm, 0.0, 0.0, alpha, n_steps=500)
R, _ = compute_order_parameter(phases)
print(f"Frustrated R = {R:.4f}")  # R < 1 due to phase lag
```

### 5.5 Full Pipeline Wiring

```python
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter
from scpn_phase_orchestrator.coupling.knm import build_knm
from scpn_phase_orchestrator.monitor.boundaries import BoundaryObserver
from scpn_phase_orchestrator.supervisor.policy import PolicyEngine

# Build from SCPN layer structure
N = 16
knm = build_knm(N, template="scpn_default")
alpha = np.zeros((N, N))
omegas = np.ones(N)
phases = rng.uniform(0, 2 * np.pi, N)

engine = UPDEEngine(N, dt=0.01, method="rk4")
observer = BoundaryObserver(thresholds={"R_min": 0.3, "R_max": 0.95})

# Simulation loop
for step in range(1000):
    phases = engine.step(phases, omegas, knm, zeta=0.0, psi=0.0, alpha=alpha)
    R, psi = compute_order_parameter(phases)

    violations = observer.observe(R=R, psi=psi, step=step)
    if violations:
        print(f"Step {step}: boundary violation — {violations}")
```

### 5.6 Comparing Methods

```python
import time

for method in ("euler", "rk4", "rk45"):
    eng = UPDEEngine(N, dt=0.01, method=method)
    p = phases.copy()
    t0 = time.perf_counter()
    p = eng.run(p, omegas, knm, 0.0, 0.0, alpha, n_steps=1000)
    elapsed = time.perf_counter() - t0
    R, _ = compute_order_parameter(p)
    print(f"{method:5s}: {elapsed:.4f}s, R={R:.4f}")
```

---

## 6. Technical Reference

### 6.1 Class API

::: scpn_phase_orchestrator.upde.engine.UPDEEngine
    options:
        show_root_heading: true
        members_order: source

### 6.2 Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_oscillators` | `int` | — | Number of oscillators N |
| `dt` | `float` | — | Integration timestep $\Delta t$ |
| `method` | `str` | `"euler"` | `"euler"`, `"rk4"`, or `"rk45"` |
| `atol` | `float` | `1e-6` | Absolute tolerance (RK45 only) |
| `rtol` | `float` | `1e-3` | Relative tolerance (RK45 only) |

### 6.3 Method Signatures

**`step(phases, omegas, knm, zeta, psi, alpha) → NDArray`**

Advance phases by one timestep. Returns new phases in $[0, 2\pi)$.

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `phases` | `(N,)` | Current phases |
| `omegas` | `(N,)` | Natural frequencies |
| `knm` | `(N, N)` | Coupling matrix |
| `zeta` | `float` | External drive amplitude |
| `psi` | `float` | External drive phase |
| `alpha` | `(N, N)` | Phase-lag matrix |

**`run(phases, omegas, knm, zeta, psi, alpha, n_steps) → NDArray`**

Execute `n_steps` integration steps. Returns final phases.

**`compute_order_parameter(phases) → tuple[float, float]`**

Delegates to `order_params.compute_order_parameter`. Returns $(R, \psi)$.

**`last_dt → float`**

Property: actual $\Delta t$ used on last accepted step (relevant for RK45
adaptive stepping).

### 6.4 Exceptions

| Exception | Condition |
|-----------|-----------|
| `ValueError` | Unknown method name |
| `ValueError` | Shape mismatch (phases, omegas, knm, alpha) |
| `ValueError` | NaN or Inf in input arrays |
| `ValueError` | Non-finite zeta or psi |

---

## 7. Performance Benchmarks

All benchmarks from `bench/baseline.json`, measured on Windows 11,
Python 3.12.5, NumPy 2.2.6, spo-kernel (Rust) backend.

### 7.1 Rust Backend (spo-kernel)

| N | Method | µs/step | steps/s | R_final |
|---|--------|---------|---------|---------|
| 8 | euler | 7.5 | 133,333 | 1.000 |
| 8 | rk4 | 5.6 | 178,571 | 1.000 |
| 8 | rk45 | 13.9 | 71,942 | 0.981 |
| 16 | euler | 11.6 | 86,207 | 1.000 |
| 16 | rk4 | 24.4 | 40,984 | 1.000 |
| 16 | rk45 | 37.9 | 26,385 | 0.827 |
| 64 | euler | 57.0 | 17,544 | 0.908 |
| 64 | rk4 | 257.3 | 3,887 | 0.913 |
| 256 | euler | 1,058.9 | 944 | 0.264 |
| 256 | rk4 | 3,142.3 | 318 | 0.267 |
| 1024 | euler | 18,494.4 | 54 | 0.186 |

### 7.2 Python Fallback (scaling_results.json)

| N | steps/s | ms/step | Memory (MB) |
|---|---------|---------|-------------|
| 16 | 29,156 | 0.034 | 0.0 |
| 64 | 10,799 | 0.093 | 0.07 |
| 256 | 143 | 7.003 | 1.05 |
| 1000 | 21 | 46.746 | 16.01 |

### 7.3 Speedup (Rust vs Python)

At N=256: Rust Euler 1.06ms vs Python 7.0ms → **~6.6x speedup**.
At N=1024: Rust Euler 18.5ms vs Python ~200ms (extrapolated) → **~12x**.

The Rust backend uses sin/cos precomputation and Rayon parallelisation
for N ≥ 256, achieving near-linear scaling with core count.

### 7.4 Complexity

| Operation | Time complexity | Space complexity |
|-----------|----------------|------------------|
| `_derivative()` | $O(N^2)$ | $O(N^2)$ scratch |
| `_euler_step()` | $O(N^2)$ | $O(N)$ result |
| `_rk4_step()` | $4 \times O(N^2)$ | $O(N^2) + O(N)$ copies |
| `_rk45_step()` | $7 \times O(N^2)$ | $O(N^2) + 7 \times O(N)$ stages |

The coupling sum $\sum_j K_{ij}\sin(\theta_j - \theta_i - \alpha_{ij})$
dominates — it is an $O(N^2)$ matrix-vector operation. For sparse coupling
graphs, use `SparseEngine` instead.

### 7.5 Method Selection Guide

| Scenario | Recommended | Reason |
|----------|-------------|--------|
| Real-time (< 1ms budget) | `euler` + Rust | Minimal overhead per step |
| Research accuracy | `rk4` | 4th-order, predictable cost |
| Unknown stiffness | `rk45` | Auto-adapts dt |
| Large N (> 1000) | `euler` + Rust | O(N²) per eval, fewer evals |
| Lyapunov spectrum | `rk4` | Fixed dt needed for Jacobian |
| Long integration (> 10⁴ steps) | `rk45` | Error accumulation control |

### 7.6 Memory Footprint

| N | Scratch arrays | RK45 stages | Total overhead |
|---|---------------|-------------|----------------|
| 16 | 2 KB | 1 KB | ~3 KB |
| 64 | 32 KB | 4 KB | ~36 KB |
| 256 | 512 KB | 14 KB | ~526 KB |
| 1024 | 8 MB | 56 KB | ~8 MB |
| 4096 | 128 MB | 224 KB | ~128 MB |

The dominant cost is the $N \times N$ scratch arrays (`_phase_diff`,
`_sin_diff`). For N > 2000, consider `SparseEngine` if the coupling
graph has density < 50%.

### 7.7 Profiling Tips

```python
import time
engine = UPDEEngine(N, dt=0.01, method="rk4")
t0 = time.perf_counter()
phases = engine.run(phases, omegas, knm, 0.0, 0.0, alpha, n_steps=100)
elapsed = time.perf_counter() - t0
print(f"{elapsed/100*1e6:.1f} µs/step")
```

To compare Rust vs Python, set `_HAS_RUST = False` in `_compat.py`
temporarily (or uninstall `spo-kernel`).

---

## 8. Citations

1. **Kuramoto Y.** (1975). Self-entrainment of a population of coupled
   non-linear oscillators. *International Symposium on Mathematical Problems
   in Theoretical Physics*, Lecture Notes in Physics **39**:420–422.
   Springer.

2. **Kuramoto Y.** (1984). *Chemical Oscillations, Waves, and Turbulence*.
   Springer-Verlag, Berlin. doi:10.1007/978-3-642-69689-3

3. **Sakaguchi H., Kuramoto Y.** (1986). A soluble active rotater model
   showing phase transitions via mutual entertainment.
   *Progress of Theoretical Physics* **76**(3):576–581.
   doi:10.1143/PTP.76.576

4. **Strogatz S.H.** (2000). From Kuramoto to Crawford: exploring the
   onset of synchronization in populations of coupled oscillators.
   *Physica D* **143**(1–4):1–20. doi:10.1016/S0167-2789(00)00094-4

5. **Dormand J.R., Prince P.J.** (1980). A family of embedded Runge-Kutta
   formulae. *Journal of Computational and Applied Mathematics*
   **6**(1):19–26. doi:10.1016/0771-050X(80)90013-3

6. **Hairer E., Nørsett S.P., Wanner G.** (1993). *Solving Ordinary
   Differential Equations I: Nonstiff Problems*. 2nd ed., Springer-Verlag.
   Table 5.2 (Dormand-Prince coefficients).

7. **Acebrón J.A., Bonilla L.L., Pérez Vicente C.J., Ritort F.,
   Spigler R.** (2005). The Kuramoto model: A simple paradigm for
   synchronization phenomena. *Reviews of Modern Physics*
   **77**(1):137–185. doi:10.1103/RevModPhys.77.137

---

## Test Coverage

- `tests/test_upde_engine.py` — 12 tests: method dispatch, euler/rk4/rk45
  correctness, shape validation, NaN rejection, run() multi-step
- `tests/test_rust_python_parity_performance.py` — 9 tests: Rust vs Python
  numerical parity within 1e-10, performance speedup verification
- `tests/test_nan_inf_edges.py` — 32 tests: degenerate inputs, zero coupling,
  single oscillator, large N boundary conditions

Total: **53 tests** covering the core engine.

---

## Source

- Python: `src/scpn_phase_orchestrator/upde/engine.py` (278 lines)
- Rust: `spo-kernel/crates/spo-engine/src/upde.rs` (~400 lines)
- FFI: `spo-kernel/crates/spo-ffi/src/lib.rs` (PyUPDEStepper binding)
