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

**Phase wrapping.** Dense output is projected after stage evaluation.
Producers use the floating period of their actual precision and map only a
remainder rounded to that upper endpoint, or signed zero, to positive zero.
`nextafter(period, 0)` remains an interior phase; no tolerance clips it.
Positive-step output is finite and half-open; zero-step runs return independent
unwrapped copies. Nonfinite computed output still refuses. Dense/CSR stateful
nonfinite phase-computation refusal preserves caller phases and the adaptive proposal for valid retry;
earlier accepted steps of a native batch are not rewound by a later refusal.

**Floating-point.** At extreme $N$ (>10,000), the $O(N^2)$ coupling sum
can accumulate rounding error. The Rust backend parallelises row blocks for
N ≥ 256. With zero phase lag, each sine/cosine sum uses eight independent
accumulators; nonzero lag uses scalar row reductions. These ordinary
floating-point reductions do not provide compensated-summation guarantees.

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

Writable coupling and lag may share storage, including identical matrices and overlapping
views. The dense native binding snapshots readonly input values before borrowing
writable coupling; enabled native plasticity still updates the original array.
Its borrow/writeability faults raise `ValueError` before advancing the solver.
See [shared NumPy storage](upde.md#shared-numpy-storage) for the snapshot,
readonly moving-frame and sparse-scope boundaries.
Public stateful calls inherit Rust's writable-coupling requirement when its
kernel is installed; selected stateless Rust calls do too. Readonly coupling
raises `ValueError` there but is accepted by the other CPU paths. Use a writable
copy for backend-independent admission.

### 4.3 CPU fallback chain and WebGPU host dispatch

`upde.engine` now exposes a stateless batched kernel `upde_run` that
dispatches across Rust → WebGPU → Mojo → Julia → Go → Python. The first
available backend becomes `ACTIVE_BACKEND` on first computation or explicit
status access; importing the module does not probe optional toolchains. The
others are available as overrides for tests and benchmarks. Stateful fixed-
frequency `UPDEEngine` selects its own Rust/NumPy solver; callable-frequency
batch runs use the schedule dispatcher. WebGPU dispatch requires a host-provided
`SPO_WEBGPU_DISPATCH_BRIDGE=module:function` adapter, implements Euler only, and
has no schedule runner. See [host dispatch setup](../../guide/webgpu_backend.md).

| Backend | Probe                                                         | Artefact                         |
| ------- | ------------------------------------------------------------- | -------------------------------- |
| Rust    | `from spo_kernel import PyUPDEStepper`                        | `spo_kernel` wheel via maturin.  |
| WebGPU  | Host adapter via `SPO_WEBGPU_DISPATCH_BRIDGE`                    | Generated float32 WGSL/ES module; Euler only. |
| Mojo    | `mojo/upde_engine_mojo` executable                            | `mojo build mojo/upde_engine.mojo`. |
| Julia   | `juliacall` + `julia/upde_engine.jl`                          | Julia 1.11.                      |
| Go      | `go/libupde_engine.so`                                        | `go build -buildmode=c-shared`.  |
| Python  | Pure NumPy                                                    | Always available.                |

Parity tolerances against the Python reference:

| Backend    | Tolerance |
| ---------- | --------- |
| Rust       | `1e-12`   |
| Julia      | `1e-12`   |
| Go         | `1e-12`   |
| Mojo       | `1e-6`    |
| Python     | exact     |

These are finite ordinary-trajectory parity tolerances, not guarantees of
bitwise equivalence for arbitrary large phases. Real phase-cut/runtime tests
exercise all three CPU integrators, both JAX precisions and actual browser
Euler. JAX uses its configured float32/float64 period. WebGPU transcendental
accuracy follows WGSL and does not promise binary64 NumPy parity. Its driver
refuses controls that overflow/underflow a positive binary32 substep and rejects
invalid actual readback; it does not repair arbitrary malformed backend output.
Each invocation destroys its seven temporary buffers when dispatch or readback
exits, including a failed mapping. Device loss still propagates a browser
exception; subsequent execution requires a new device.
See [the runtime diagnostics](../data/upde_phase_wrapping_benchmark_2026-10-01.json).

### 4.4 Pre-allocated Scratch Arrays

The engine pre-allocates intermediate arrays (`_phase_diff`, `_sin_diff`,
`_scratch_dtheta`) at construction time to reuse derivative storage.
For RK45, seven stage buffers (`_ks`) and an error buffer are also
pre-allocated. Candidate/output arrays and RK4 stage copies still allocate.
Rust dense/CSR integration snapshots three O(N) vectors per step: phases and
the sine/cosine order-parameter caches. Computed-divergence refusals restore
these vectors and the step proposal before plasticity updates.

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
print(f"Configured or proposed dt: {engine.last_dt:.6f}")
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

Property: configured timestep for the native dense backend and fixed methods,
or the most recently retained NumPy RK45 `step()` proposal. Batch `run()` calls
leave this diagnostic unchanged, and the dense wrapper does not synchronise
the native RK45 proposal into it. Neither value is the elapsed time of an
accepted adaptive step.

### 6.4 Exceptions

| Exception | Condition |
|-----------|-----------|
| `ValueError` | Unknown method name |
| `ValueError` | Shape mismatch (phases, omegas, knm, alpha) |
| `ValueError` | NaN or Inf in input arrays |
| `ValueError` | Non-finite zeta or psi |
| `ValueError` | Non-finite computed phases after integration |

---

## 7. Performance Benchmarks

### 7.0 Five-backend comparison (2026-10-01)

Observed mean milliseconds per stateless call on the shared Linux x86-64 host,
CPython 3.12.3, NumPy 2.5.3, Rust 1.98.1 release build and embedded Julia 1.11.9:
`dt=0.01`, 500 steps, one warmup and three measured calls. Reproduce with:

```bash
.venv/bin/python -m benchmarks.upde_engine_benchmark --output results.json
```

| N | method | rust (ms) | mojo (ms) | julia (ms) | go (ms) | python (ms) |
|---|--------|----------:|----------:|-----------:|--------:|------------:|
| 8 | euler | 0.8275 | 57.1444 | 0.7806 | 1.0739 | 11.6304 |
| 8 | rk4 | 2.0923 | 49.4179 | 2.4365 | 2.2390 | 54.6946 |
| 8 | rk45 | 2.5307 | 54.4482 | 5.7154 | 6.5171 | 102.4653 |
| 32 | euler | 5.9014 | 65.3821 | 8.5493 | 18.3068 | 19.6329 |
| 32 | rk4 | 25.6531 | 76.6047 | 27.0021 | 48.4175 | 75.8595 |
| 32 | rk45 | 45.3127 | 255.9066 | 74.6279 | 80.3099 | 200.8557 |
| 64 | euler | 21.8070 | 142.5860 | 28.8938 | 51.2086 | 43.6911 |
| 64 | rk4 | 140.1609 | 161.2979 | 114.6480 | 331.2789 | 144.5777 |
| 64 | rk45 | 144.5623 | 296.6771 | 233.4334 | 546.6765 | 326.0371 |

The [raw runtime record](../data/upde_phase_wrapping_benchmark_2026-10-01.json)
contains exact source/artefact hashes, all five CPU comparisons, the scheduled,
Doppler and moving-frame acceptance results, CSR repetitions, and actual
JAX/browser phase-cut readbacks. These non-isolated measurements do not establish
a backend ranking, a causal speedup or a deployment deadline. The maintained
stateless CLI emits means, not individual timing samples.

Public `ACTIVE_BACKEND="python"` now selects the actual NumPy runner;
runtime-profiled fixed/scheduled regressions verify each available CPU backend
is exercised, with no silent reference fallback for selected accelerators.
The host-managed WebGPU Euler bridge has no schedule loader and is not covered
by this five-language comparison. Earlier forced-Python parity results must be
rerun: the old selector could have executed an accelerator instead of NumPy.

### 7.1 Stateful boundary

The dense native class now snapshots readonly inputs before borrowing mutable
coupling. Its stateful wrapper cost is measured separately from the stateless
comparison above; current real NumPy-only and native stateful timing diagnostics are
in [Rust FFI acceleration](../../guide/rust_ffi.md#benchmark-comparison).
The NumPy environments differ, so no causal speedup is claimed. Batch runs
amortise entry-time snapshot allocation across their inner timesteps.

### 7.2 Complexity

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

Compare actual separate environments, one with the installed extension and one
without it. Do not modify private import flags or uninstall a shared environment's
kernel merely to obtain a reference timing.

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

- `tests/test_upde_engine.py`: integration methods, synchronisation, phase
  wrapping and validation.
- `tests/test_upde_engine_matrix_aliases.py`: actual public shared and overlapping
  matrices, analytical zero-coupling trajectory and input/state preservation.
- `native-tests/test_upde_stepper_aliases.py`: installed dense FFI snapshots,
  mutable plasticity writeback, contiguous/writable buffer refusals and the
  distinct readonly moving-frame contract.
- `tests/test_upde_time_varying_omega.py`, `tests/test_upde_doppler.py` and
  `tests/test_upde_moving_frame.py`: scheduled and kinematic integration.

These are owning regression surfaces, not a claim of whole-project coverage.

---

## Source

- Python: `src/scpn_phase_orchestrator/upde/engine.py`
- Rust: `spo-kernel/crates/spo-engine/src/upde.rs`
- FFI: `spo-kernel/crates/spo-ffi/src/upde_stepper.rs` (dense buffer ownership),
  registered by `spo-kernel/crates/spo-ffi/src/lib.rs`

## Time-varying omega support

The stateful engine accepts `omega=` at construction time. The value may be a
fixed vector or a callable `omega(t)` returning a finite real vector with shape
`(n_oscillators,)`. Existing explicit `step(phases, omegas, ...)` and
`run(phases, omegas, ...)` calls retain precedence over configured omega.

For callable sources, `run()` resolves one frequency vector per outer step and
calls `upde_run_omega_schedule`, which is implemented across the Rust, Go,
Julia, Mojo, and Python UPDE backend surfaces. See
[UPDE — Time-varying omega](upde_time_varying_omega.md).

## Doppler-corrected schedule path

`DopplerEngine` builds on the time-varying omega path by resolving one velocity
vector per outer step, computing the graph-weighted Doppler correction, and
passing `omega_eff = omega + doppler_term` into the same UPDE integrators. The
public contract is documented in [UPDE — Doppler Engine](upde_doppler.md).
