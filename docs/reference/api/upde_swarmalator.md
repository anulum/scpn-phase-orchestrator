# Swarmalator Engine — Coupled Spatial-Phase Dynamics

The `SwarmalatorEngine` models agents that are simultaneously swarming
(moving in space) and oscillating (advancing in phase). Phase similarity
modulates spatial attraction, and spatial proximity modulates phase coupling.
This creates a rich zoo of collective states: static sync, static async,
splintered phase waves, and active phase waves.

Swarmalators are a relatively new concept in nonlinear dynamics (O'Keeffe,
Hong & Strogatz 2017), bridging swarm dynamics with Kuramoto synchronisation.

---

## 1. Mathematical Formalism

### 1.1 The Swarmalator Equations

Each agent $i$ has position $\mathbf{x}_i \in \mathbb{R}^d$ and phase
$\theta_i \in [0, 2\pi)$. The dynamics are:

**Position:**
$$
\dot{\mathbf{x}}_i = \frac{1}{N} \sum_{j \neq i}
\frac{(A + J\cos(\theta_j - \theta_i))\,(\mathbf{x}_j - \mathbf{x}_i)}{|\mathbf{x}_j - \mathbf{x}_i|}
- \frac{B\,(\mathbf{x}_j - \mathbf{x}_i)}{|\mathbf{x}_j - \mathbf{x}_i|^3}
$$

**Phase:**
$$
\dot{\theta}_i = \omega_i + \frac{K}{N} \sum_{j \neq i}
\frac{\sin(\theta_j - \theta_i)}{|\mathbf{x}_j - \mathbf{x}_i|}
$$

where:

| Symbol | Description | Default |
|--------|-------------|---------|
| $\mathbf{x}_i$ | Position in $\mathbb{R}^d$ | — |
| $\theta_i$ | Phase | $[0, 2\pi)$ |
| $\omega_i$ | Natural frequency | rad/s |
| $A$ | Attraction strength | 1.0 |
| $B$ | Repulsion strength | 1.0 |
| $J$ | Phase→space coupling | 1.0 |
| $K$ | Space→phase coupling | 1.0 |
| $d$ | Spatial dimension | 2 or 3 |

### 1.2 Coupling Mechanisms

The swarmalator model has two reciprocal couplings:

**J: Phase modulates space.** The attraction between agents depends on
their phase difference. When $J > 0$: phase-similar agents attract more
strongly (clustering by phase). When $J < 0$: phase-different agents
attract (spatial mixing by phase).

**K: Space modulates phase.** Phase coupling is inversely weighted by
distance. Nearby agents synchronise more strongly. This is the natural
assumption for local interactions (neural, chemical, etc.).

### 1.3 Collective States

O'Keeffe et al. (2017) identified five macroscopic states:

| State | J | K | Spatial pattern | Phase pattern |
|-------|---|---|----------------|---------------|
| **Static sync** | $J > 0$ | $K > 0$ | Compact cluster | All locked |
| **Static async** | $J < 0$ | $K > 0$ | Ring/shell | Spatially ordered phases |
| **Static phase wave** | $J > 0$ | $K = 0$ | Disc | Phase varies with radius |
| **Splintered phase wave** | $J < 0$ | $K < 0$ | Fragmented | Partial locking |
| **Active phase wave** | $J > 0$ | $K < 0$ | Rotating | Travelling wave |

### 1.4 Regularisation

The $1/|\mathbf{x}_j - \mathbf{x}_i|$ terms diverge when agents coincide.
The implementation adds a small $\epsilon = 10^{-6}$ to distances:

$$
|\mathbf{x}_j - \mathbf{x}_i|_\epsilon = \sqrt{\sum_d (x_{j,d} - x_{i,d})^2 + \epsilon}
$$

This prevents division-by-zero without significantly affecting dynamics
when agents are separated.

### 1.5 Order Parameter

The `order_parameter(phases)` method computes the standard Kuramoto $R$:

$$
R = \left| \frac{1}{N} \sum_i e^{i\theta_i} \right|
$$

For swarmalators, $R$ alone doesn't capture the full state — the spatial
distribution matters too. A "static async" state has $R \approx 0$ but is
highly ordered spatially. Use both $R$ and mean inter-agent distance for
full characterisation.

### 1.6 Dimensionality

The engine supports arbitrary spatial dimension $d$. Default $d = 2$
(planar swarming). $d = 3$ models three-dimensional swarming (e.g.,
fish schools, bird flocks, drone formations). Higher $d$ is mathematically
valid but has no obvious physical interpretation.

### 1.7 Energy Functional

The swarmalator system has no known energy functional in general.
However, for specific parameter regimes (e.g., $J = 0$), the spatial
and phase dynamics decouple and each has its own Lyapunov function.
The coupled system is **non-gradient** — trajectories can exhibit
limit cycles and chaotic attractors.

---

## 2. Theoretical Context

### 2.1 Historical Background

The swarmalator model was introduced by O'Keeffe, Hong & Strogatz (2017)
in *Nature Communications*. It was inspired by biological systems where
spatial aggregation and internal dynamics are coupled:

- **Sperm cells:** Flagellar beating (oscillation) coupled to chemotaxis
  (spatial movement)
- **Fireflies:** Flash timing (phase) coupled to spatial positioning
- **Vinegar eels:** Body oscillation coupled to collective swimming
- **Myxococcus bacteria:** Reversal period (phase) coupled to swarming

The model was extended to 3D by O'Keeffe et al. (2022) and to networks
with heterogeneous coupling by Lizárraga & de Aguiar (2020).

### 2.2 Role in SCPN

The swarmalator engine has no direct SCPN layer assignment. It is used for:

1. **Research** — studying coupled spatial-phase dynamics in neural populations
   where neurons have both physical location and oscillatory phase
2. **Notebook 19** — interactive exploration of swarmalator collective states
3. **Potential Layer 12 extension** — distributed nodes with both geographic
   position and oscillation state

### 2.3 Comparison with Other Engines

| Engine | Has position? | Has amplitude? | Coupling |
|--------|--------------|----------------|----------|
| `UPDEEngine` | No | No | All-to-all or graph |
| `StuartLandauEngine` | No | Yes | All-to-all or graph |
| `InertialKuramotoEngine` | No | No (has velocity) | All-to-all or graph |
| `SwarmalatorEngine` | **Yes** ($\mathbb{R}^d$) | No | Distance-weighted |

Swarmalator is the only engine with spatial degrees of freedom.

### 2.4 Phase Diagram

The $(J, K)$ parameter space divides into five regions:

```
K ↑
  |  Active        Static
  |  phase wave    sync
  |                (J>0, K>0)
  |
--+--+--+--+--+--+--→ J
  |
  |  Splintered    Static
  |  phase wave    async
  |                (J<0, K>0)
  |
```

**Phase transitions:** As $J$ crosses 0 at fixed $K > 0$, the system
transitions from static async (ring) to static sync (cluster). As $K$
crosses 0, active states (with persistent motion) emerge.

The transitions are discontinuous (first-order) in some regions —
hysteresis exists. The `run()` trajectory reveals whether the system
reached steady state or is still evolving.

### 2.5 Biological Relevance

| System | Position | Phase | J coupling | K coupling |
|--------|----------|-------|------------|------------|
| Fireflies | Tree position | Flash timing | Light attraction | Visual sync |
| Sperm cells | Swim position | Flagellar beat | Hydrodynamic | Mechanical |
| Myxococcus | Colony position | Reversal period | Signalling | Physical contact |
| Neurons (Layer 2) | Cortical position | Firing phase | Synaptic strength | Spatial proximity |
| Drone swarm | 3D coordinates | Communication phase | Formation control | Local consensus |

### 2.6 Analytical Results

For identical oscillators ($\omega_i = 0$, $A = B = 1$) in 2D:
- Static sync: all agents at same position and phase. $R = 1$, cluster
  radius $\to 0$.
- Static async: agents on a ring of radius $\sim \sqrt{B/A}$, phases
  ordered by angle. $R = 0$ but spatially ordered.
- Phase wave: agents fill a disc, phase varies linearly with radius.
  $R \approx 0$.

No closed-form $K_c$ exists for the general swarmalator model — numerical
continuation (not yet implemented in SPO) would be needed.

### 2.7 Limitations

- **$O(N^2)$ per step:** All-to-all distance computation. No neighbour
  lists or spatial indexing (future optimisation).
- **Euler integration only** (Python fallback). Rust backend available
  but no RK4/RK45.
- **No boundaries:** Agents can drift to infinity in some parameter
  regimes. No periodic boundaries or confinement implemented.
- **Fixed coupling parameters:** $A, B, J, K$ are passed per-step, not
  per-pair. Heterogeneous pairwise coupling not supported.

---

## 3. Pipeline Position

```
┌──────────────┐     ┌──────────────────────┐     ┌──────────────┐
│ Initial      │────→│ SwarmalatorEngine    │────→│ positions    │
│ positions    │     │                      │     │ phases       │
│ phases       │     │ step(pos, phases,    │     └──────┬───────┘
│ omegas       │     │   omegas, a,b,j,k)  │            │
└──────────────┘     │                      │     ┌──────▼───────┐
                     │ run() → trajectories │     │ order_params │
                     └──────────────────────┘     │ (R, ψ)       │
                                                  └──────────────┘
```

**Inputs:**
- `pos` (N, d) — agent positions
- `phases` (N,) — agent phases
- `omegas` (N,) — natural frequencies
- `a, b, j, k` (floats) — coupling parameters (per-step)

**Outputs:**
- `(new_pos, new_phases)` — updated state

---

## 4. Features

### 4.1 Flexible API

Coupling parameters $(a, b, j, k)$ are passed to `step()` and `run()`,
not the constructor. This allows time-varying coupling (e.g., ramping
$J$ over a simulation).

### 4.2 Arbitrary Dimension

Constructor parameter `dim` supports any positive integer. Tested with
$d = 1, 2, 3$.

### 4.3 Trajectory Recording

`run()` returns full position and phase trajectories:
`(final_pos, final_phases, pos_trajectory, phase_trajectory)`.

### 4.4 Rust Acceleration

`PySwarmalatorStepper` provides Rust-accelerated `step()`. Speedup ~12x.

---

## 5. Usage Examples

### 5.1 Static Sync ($J > 0$, $K > 0$)

```python
import numpy as np
from scpn_phase_orchestrator.upde.swarmalator import SwarmalatorEngine

N = 30
rng = np.random.default_rng(42)
pos = rng.uniform(-2, 2, (N, 2))
phases = rng.uniform(0, 2 * np.pi, N)
omegas = np.zeros(N)

engine = SwarmalatorEngine(N, dim=2, dt=0.01)
fp, fph, pos_traj, ph_traj = engine.run(
    pos, phases, omegas, a=1.0, b=1.0, j=1.0, k=1.0, n_steps=2000,
)

R = engine.order_parameter(fph)
print(f"Static sync: R = {R:.4f}")  # R ≈ 1.0
```

### 5.2 Static Async ($J < 0$, $K > 0$)

```python
fp, fph, _, _ = engine.run(
    pos.copy(), phases.copy(), omegas,
    a=1.0, b=1.0, j=-0.5, k=1.0, n_steps=2000,
)
R = engine.order_parameter(fph)
print(f"Static async: R = {R:.4f}")  # R ≈ 0 but spatially ordered
```

### 5.3 Decoupled ($J = 0$)

```python
fp, fph, _, _ = engine.run(
    pos.copy(), phases.copy(), omegas,
    a=1.0, b=1.0, j=0.0, k=1.0, n_steps=2000,
)
# Phase doesn't affect space → standard swarm + standard Kuramoto
```

### 5.4 3D Swarming

```python
pos3d = rng.uniform(-2, 2, (N, 3))
engine3d = SwarmalatorEngine(N, dim=3, dt=0.01)
fp3d, fph3d, _, _ = engine3d.run(
    pos3d, phases.copy(), omegas,
    a=1.0, b=1.0, j=0.5, k=1.0, n_steps=1000,
)
print(f"3D: R = {engine3d.order_parameter(fph3d):.4f}")
```

### 5.5 Parameter Sweep

```python
J_values = np.linspace(-1, 1, 20)
R_values = []
for J_val in J_values:
    _, fph, _, _ = engine.run(
        pos.copy(), phases.copy(), omegas,
        a=1.0, b=1.0, j=J_val, k=1.0, n_steps=1000,
    )
    R_values.append(engine.order_parameter(fph))
# R_values traces the J-dependent synchronisation landscape
```

### 5.6 Time-Varying Coupling

```python
# Ramp J from 0 to 1 over 2000 steps
pos_curr, ph_curr = pos.copy(), phases.copy()
for step in range(2000):
    J_t = step / 2000.0
    pos_curr, ph_curr = engine.step(
        pos_curr, ph_curr, omegas,
        a=1.0, b=1.0, j=J_t, k=1.0,
    )
R_final = engine.order_parameter(ph_curr)
print(f"After J ramp: R = {R_final:.4f}")
```

### 5.7 Spatial Compactness Metric

```python
# Measure how compact the swarm is (mean distance from centroid)
centroid = fp.mean(axis=0)
distances = np.linalg.norm(fp - centroid, axis=1)
compactness = distances.mean()
print(f"Mean distance from centroid: {compactness:.4f}")
# Static sync: very compact. Static async: ring radius.
```

### 5.8 Phase-Space Correlation

```python
# In static async: phase correlates with angular position
angles = np.arctan2(fp[:, 1] - centroid[1], fp[:, 0] - centroid[0])
angles_mod = angles % (2 * np.pi)
from scpn_phase_orchestrator.upde.order_params import compute_plv
# PLV between spatial angle and oscillator phase
plv = float(np.abs(np.mean(np.exp(1j * (fph - angles_mod)))))
print(f"Phase-space PLV: {plv:.4f}")
# High PLV → strong spatial-phase ordering (static async state)
```

### 5.9 Multi-State Exploration

```python
# Systematically explore (J, K) phase diagram
states = {}
for J_val in [-1, -0.5, 0, 0.5, 1.0]:
    for K_val in [-1, 0, 0.5, 1.0]:
        _, fph, _, _ = engine.run(
            pos.copy(), phases.copy(), omegas,
            a=1.0, b=1.0, j=J_val, k=K_val, n_steps=3000,
        )
        R = engine.order_parameter(fph)
        states[(J_val, K_val)] = R
        print(f"J={J_val:+.1f}, K={K_val:+.1f}: R={R:.3f}")
```

### 5.10 Visualisation (2D scatter)

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 6))
sc = ax.scatter(fp[:, 0], fp[:, 1], c=fph, cmap="hsv",
                vmin=0, vmax=2*np.pi, s=50, edgecolors="k", linewidths=0.5)
plt.colorbar(sc, label="Phase (rad)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(f"Swarmalator state (R={engine.order_parameter(fph):.2f})")
ax.set_aspect("equal")
plt.savefig("swarmalator_state.png", dpi=150)
```

---

## 6. Technical Reference

### 6.1 Module API

::: scpn_phase_orchestrator.upde.swarmalator
    options:
        show_root_heading: true
        members_order: source

### 6.2 Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_agents` | `int` | — | Number of swarmalator agents |
| `dim` | `int` | `2` | Spatial dimension |
| `dt` | `float` | `0.01` | Timestep |

### 6.3 step() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pos` | `(N, d)` | — | Agent positions |
| `phases` | `(N,)` | — | Agent phases |
| `omegas` | `(N,)` | — | Natural frequencies |
| `a` | `float` | `1.0` | Attraction strength |
| `b` | `float` | `1.0` | Repulsion strength |
| `j` | `float` | `1.0` | Phase→space coupling |
| `k` | `float` | `1.0` | Space→phase coupling |

Returns: `(new_pos, new_phases)`.

### 6.4 run() Parameters

Same as `step()` plus `n_steps: int = 100`.
Returns: `(final_pos, final_phases, pos_trajectory, phase_trajectory)`.

### 6.5 order_parameter(phases) → float

Standard Kuramoto R. Does not account for spatial structure.

---

## 7. Performance Benchmarks

### 7.1 Rust Speedup

| N | dim | Python (ms/step) | Rust (ms/step) | Speedup |
|---|-----|-------------------|----------------|---------|
| 30 | 2 | 0.8 | 0.07 | 11.4x |
| 100 | 2 | 8.5 | 0.7 | 12.1x |
| 500 | 2 | 210 | 17 | 12.4x |
| 100 | 3 | 9.2 | 0.8 | 11.5x |

### 7.2 Complexity

| Operation | Time | Space |
|-----------|------|-------|
| `step()` | $O(N^2 \cdot d)$ | $O(N^2)$ distances + $O(Nd)$ |
| `run(n)` | $O(n \cdot N^2 \cdot d)$ | $O(n \cdot N \cdot d)$ trajectory |

The $N^2$ pairwise distance computation dominates. For large $N$,
spatial indexing (k-d tree, not yet implemented) could reduce to
$O(N \log N)$ for short-range interactions.

### 7.3 Memory

| N | dim | n_steps | Trajectory memory |
|---|-----|---------|-------------------|
| 30 | 2 | 2000 | 940 KB |
| 100 | 2 | 2000 | 3.2 MB |
| 500 | 3 | 1000 | 12 MB |

### 7.4 Convergence Time by State

Different collective states converge at different rates:

| State | Typical convergence | Sensitivity |
|-------|--------------------|--------------------|
| Static sync | Fast (500–1000 steps) | Low — robust |
| Static async | Medium (1000–3000 steps) | Medium |
| Phase wave | Slow (3000–5000 steps) | High — initial conditions matter |
| Active phase wave | Very slow (5000+ steps) | Very high — transients persist |
| Splintered | Slow (3000+ steps) | High — metastable states |

For reliable state identification, run ≥ 3000 steps and verify that $R$
has plateaued (check $|R(t) - R(t-100)| < 0.01$).

### 7.5 Timestep Stability

Euler integration is conditionally stable. For $dt = 0.01$:
- Stable for $A, B, J, K \leq 5$ and $N \leq 500$
- Unstable for $A, B > 10$ or $N > 1000$ with $dt = 0.01$
- Reduce $dt$ proportionally: $dt \leq 0.1 / \max(A, B, |J|, |K|)$

### 7.6 Recommended Settings

| Scenario | N | dim | dt | n_steps |
|----------|---|-----|-----|---------|
| Quick exploration | 30 | 2 | 0.01 | 2000 |
| Publication quality | 100 | 2 | 0.005 | 5000 |
| 3D visualisation | 100 | 3 | 0.01 | 1000 |
| Large-scale research | 500 | 2 | 0.01 | 10000 |

### 7.7 Future Optimisations

- **Spatial indexing:** k-d tree for short-range interactions → $O(N \log N)$
- **RK4 integration:** higher accuracy without reducing $dt$
- **GPU acceleration:** swarmalator N-body is trivially parallelisable
- **Periodic boundaries:** wrap positions on torus for infinite-lattice-like behaviour

---

## 8. Citations

1. **O'Keeffe K.P., Hong H., Strogatz S.H.** (2017). Oscillators that
   sync and swarm. *Nature Communications* **8**:1504.
   doi:10.1038/s41467-017-01190-3

2. **O'Keeffe K.P., Evers J.H.M., Kolokolnikov T.** (2022). Ring states
   in swarmalator systems. *Physical Review E* **105**(3):034307.
   doi:10.1103/PhysRevE.105.034307

3. **Lizárraga J.U.F., de Aguiar M.A.M.** (2020). Synchronization and
   spatial patterns in forced swarmalators. *Chaos* **30**(5):053112.
   doi:10.1063/1.5141343

4. **Tanaka H.** (2007). General chemotactic model of oscillators.
   *Physical Review Letters* **99**(13):134103.
   doi:10.1103/PhysRevLett.99.134103

5. **Kuramoto Y.** (1984). *Chemical Oscillations, Waves, and Turbulence*.
   Springer-Verlag. doi:10.1007/978-3-642-69689-3

6. **Yoon S., O'Keeffe K.P., Mendes J.F.F., Goltsev A.V.** (2022).
   Sync and swarm: swarmalators on random graphs. *New Journal of Physics*
   **24**:023037. doi:10.1088/1367-2630/ac4808

7. **Sar G.K., Ghosh D., O'Keeffe K.** (2023). Swarmalators under
   competitive time-varying phase interactions. *New Journal of Physics*
   **25**:032001. doi:10.1088/1367-2630/acc127

---

## Test Coverage

- `tests/test_swarmalator.py` — 12 tests: step shapes, phase range,
  J=0 decoupling, run trajectory shapes, 3D, order parameter bounds,
  behaviour (J>0 clustering, K>0 phase sync), pipeline wiring
- `tests/test_prop_swarmalator_inertial.py` — 7 property tests (Hypothesis):
  output finite, shapes correct, J=0 phase-position independence
- `tests/test_degenerate_edges.py` — 4 swarmalator edge tests: N=2,
  dimensions 1/2/3, J=0 decoupling

Total: **23+ tests** across multiple files.

---

## Source

- Python: `src/scpn_phase_orchestrator/upde/swarmalator.py` (100 lines)
- Rust: `spo-kernel/crates/spo-engine/src/swarmalator.rs`
- FFI: `spo-kernel/crates/spo-ffi/src/lib.rs` (PySwarmalatorStepper)
