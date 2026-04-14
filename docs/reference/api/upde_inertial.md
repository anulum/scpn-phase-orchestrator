# Inertial Kuramoto — Second-Order Swing Equation

The `InertialKuramotoEngine` extends the Kuramoto model with rotational
inertia and damping — the **swing equation** from power systems engineering.
Each oscillator has both phase $\theta_i$ and frequency deviation
$\dot{\omega}_i$ as state variables, making the system second-order.

This is the canonical model for:
- **Power grid stability** — synchronous generators with mechanical inertia
- **Rotating machinery** — coupled mechanical oscillators
- **Neural oscillators with adaptation** — slow frequency adaptation

---

## 1. Mathematical Formalism

### 1.1 The Swing Equation (Second-Order Kuramoto)

$$
M_i \ddot{\theta}_i = P_i - D_i \dot{\theta}_i + \sum_{j=1}^{N} K_{ij} \sin(\theta_j - \theta_i)
$$

Rewritten as a first-order system:

$$
\dot{\theta}_i = \dot{\omega}_i
$$

$$
M_i \dot{\dot{\omega}}_i = P_i - D_i \dot{\omega}_i + \sum_{j=1}^{N} K_{ij} \sin(\theta_j - \theta_i)
$$

where:

| Symbol | Description | Units | Power grid analogue |
|--------|-------------|-------|---------------------|
| $\theta_i$ | Rotor angle | radians | Generator electrical angle |
| $\dot{\omega}_i$ | Angular velocity deviation | rad/s | Frequency deviation from nominal |
| $M_i$ | Inertia constant | s² | $H_i / (\pi f_0)$ (generator H-constant) |
| $D_i$ | Damping coefficient | s⁻¹ | Governor droop characteristic |
| $P_i$ | Mechanical power input | dimensionless | $P_{\text{mech}} - P_{\text{elec}}$ mismatch |
| $K_{ij}$ | Coupling strength | dimensionless | Susceptance $B_{ij} V_i V_j$ |

### 1.2 RK4 Integration

The engine uses classical fourth-order Runge-Kutta for both $\theta$ and
$\dot{\omega}$ simultaneously. Each stage evaluates:

$$
\dot{\theta} = \dot{\omega}, \quad
\ddot{\omega} = \frac{P + K\sin(\cdot) - D\dot{\omega}}{M}
$$

Four stages with weights $(1/6, 1/3, 1/3, 1/6)$, advancing both state
variables in lockstep.

### 1.3 Relationship to Standard Kuramoto

Setting $M_i = 0$ (zero inertia) eliminates the acceleration term, and
the system reduces to:

$$
D_i \dot{\theta}_i = P_i + \sum_j K_{ij} \sin(\theta_j - \theta_i)
$$

which is the standard (overdamped) Kuramoto model with $\omega_i = P_i / D_i$.
The inertial model adds oscillatory transients and the possibility of
desynchronisation through oscillatory instability (not just saddle-node
bifurcation as in first-order).

### 1.4 Critical Coupling for Inertial Systems

For identical inertia $M$ and damping $D$ with all-to-all coupling:

$$
K_c = \frac{2\Delta}{\cos(\arctan(M \cdot \Omega_{\text{nat}} / D))}
$$

where $\Delta$ is the frequency spread and $\Omega_{\text{nat}}$ is the
natural oscillation frequency $\sqrt{K/M}$. Key insight: inertia raises
$K_c$ — heavier oscillators are harder to synchronise.

### 1.5 Frequency Deviation and Grid Stability

In power grids, the frequency deviation $\dot{\omega}_i$ directly maps
to the grid frequency error:

$$
f_i(t) = f_0 + \frac{\dot{\omega}_i(t)}{2\pi}
$$

where $f_0 = 50$ Hz (Europe) or 60 Hz (Americas). The `frequency_deviation()`
method returns $\max_i |\dot{\omega}_i| / 2\pi$ — the worst-case frequency
error in Hz.

Grid stability requires $\max_i |\Delta f_i| < 0.5$ Hz for normal operation
and $< 2.5$ Hz before load shedding triggers.

### 1.6 Oscillatory Instability

Unlike first-order Kuramoto (which only has monotone convergence or
divergence), the inertial model exhibits **oscillatory instability**:
frequency oscillations grow in amplitude before the system desynchronises.
This is the dominant failure mode in power grids — "inter-area oscillations"
between generator clusters.

The damping ratio $\zeta_d = D / (2\sqrt{KM})$ determines the character:
- $\zeta_d > 1$: overdamped (first-order-like, no oscillations)
- $0 < \zeta_d < 1$: underdamped (oscillatory approach to sync)
- $\zeta_d = 0$: undamped (perpetual oscillations)

---

## 2. Theoretical Context

### 2.1 Historical Background

The swing equation was developed by Park (1929) and Concordia (1951) for
analysing synchronous generator transient stability. Filatrella, Nielsen &
Pedersen (2008) connected it to the Kuramoto model, showing that a network
of coupled generators is mathematically equivalent to a second-order
Kuramoto model with inertia.

Dörfler & Bullo (2012, 2014) provided rigorous synchronisation conditions
for the swing equation on general graphs, establishing the relationship
between graph structure, inertia distribution, and critical coupling.

Rohden et al. (2012) showed that decentralised renewable energy sources
(with lower inertia than conventional generators) can destabilise power
grids by reducing the effective inertia constant.

### 2.2 Role in SCPN

The inertial engine models SCPN Layer 6 (Planetary/geophysical oscillations)
and Layer 10 (Boundary Control), where oscillators have physical mass or
momentum that resists instantaneous phase changes. It is also used for
power grid digital twin simulations via the `plasma_control_bridge` adapter.

### 2.3 Modal Analysis

For small perturbations around the synchronised state $\theta_i = \theta_0$,
$\dot{\omega}_i = 0$, linearisation gives:

$$
M_i \ddot{\delta}_i + D_i \dot{\delta}_i + \sum_j L_{ij} \delta_j = 0
$$

where $L_{ij} = K_{ij} \cos(\theta_j^0 - \theta_i^0)$ is the linearised
coupling (Laplacian-like). This is a damped harmonic system with $N$
coupled modes.

The eigenvalues of the linearised system $\lambda_k$ determine stability:
- All $\text{Re}(\lambda_k) < 0$: stable (all modes damped)
- Any $\text{Re}(\lambda_k) > 0$: oscillatory instability
- $\text{Re}(\lambda_k) = 0$: marginal (Hopf bifurcation possible)

The slowest-decaying mode (smallest $|\text{Re}(\lambda)|$) is the
**critical mode** — it determines the transient settling time and is
typically an inter-area oscillation between generator clusters.

### 2.4 Energy Function

For uniform damping $D$, the system has a Lyapunov-like energy function:

$$
V = \sum_i \frac{1}{2} M_i \dot{\omega}_i^2
  - \sum_{i<j} K_{ij} \cos(\theta_j - \theta_i)
  + \sum_i P_i (\theta_i - \theta_i^*)
$$

where $\theta^*$ is the stable equilibrium. $\dot{V} \leq 0$ (energy
dissipated by damping), proving asymptotic stability when $V$ has a
minimum at the sync state. The `run()` trajectory can be used to
verify $V(t) \to V_{\min}$ numerically.

### 2.5 Applications

| Application | Typical N | Inertia M | Damping D | Power P |
|-------------|-----------|-----------|-----------|---------|
| Power grid (EU) | 10–1000 | 3–8 s | 0.5–2.0 s⁻¹ | ±0.1–1.0 |
| Mechanical rotors | 2–20 | 0.1–10 kg·m² | 0.01–1.0 N·m·s | torque |
| Neural adaptation | 16–256 | 0.01–0.1 | 1.0–10.0 | firing rate |
| SCPN Layer 6 | 16 | 1.0 | 0.5 | 0.0 |

---

## 3. Pipeline Position

```
┌──────────────┐     ┌──────────────────────┐     ┌──────────────┐
│ coupling/    │────→│ InertialKuramoto    │────→│ monitor/     │
│ knm.py       │     │ Engine              │     │ order_params │
│ (K_ij)       │     │                     │     │ lyapunov     │
└──────────────┘     │ State: (θ, ω̇)      │     └──────────────┘
                     │ RK4 integration     │
┌──────────────┐     │                     │     ┌──────────────┐
│ oscillators/ │────→│ power P_i           │     │ supervisor/  │
│ (P, M, D)    │     │ inertia M_i         │────→│ regimes.py   │
└──────────────┘     │ damping D_i         │     │ (freq check) │
                     └──────────────────────┘     └──────────────┘
```

**Inputs:**
- `theta` (N,) — rotor angles in $[0, 2\pi)$
- `omega_dot` (N,) — angular velocity deviations
- `power` (N,) — mechanical power input $P_i$
- `knm` (N,N) — coupling matrix
- `inertia` (N,) — inertia constants $M_i$
- `damping` (N,) — damping coefficients $D_i$

**Outputs:**
- `(theta, omega_dot)` — updated state after RK4 step
- `run()` also returns trajectory arrays

---

## 4. Features

### 4.1 Second-Order State

Unlike `UPDEEngine` (first-order, phase only), this engine tracks both
phase $\theta_i$ and frequency deviation $\dot{\omega}_i$. The state
space is $2N$-dimensional.

### 4.2 RK4 Integration

Classical fourth-order Runge-Kutta applied to the coupled system. Both
$\theta$ and $\dot{\omega}$ are advanced simultaneously within each stage.

### 4.3 Per-Oscillator Parameters

Each oscillator has individual inertia $M_i$ and damping $D_i$, enabling
heterogeneous networks (e.g., mix of large and small generators).

### 4.4 Trajectory Recording

`run()` returns full trajectories `(theta_traj, omega_traj)` in addition
to final state — essential for transient stability analysis.

### 4.5 Diagnostic Methods

- `frequency_deviation(omega_dot)` — max frequency error in Hz
- `coherence(theta)` — Kuramoto order parameter R

### 4.6 Rust Acceleration

`PyInertialStepper` (spo-kernel) provides Rust-accelerated `step()`.
`run()` currently loops over `step()` in Python (Rust batch not yet
implemented). Speedup ~60x per step.

---

## 5. Usage Examples

### 5.1 Basic Power Grid Simulation

```python
import numpy as np
from scpn_phase_orchestrator.upde.inertial import InertialKuramotoEngine

N = 10  # 10 generators
rng = np.random.default_rng(42)

theta = rng.uniform(0, 2 * np.pi, N)
omega_dot = np.zeros(N)  # start at nominal frequency
power = rng.normal(0, 0.1, N)  # small power imbalances
power -= power.mean()  # ensure net zero

# All-to-all coupling (K = 3.0)
knm = np.full((N, N), 3.0 / N)
np.fill_diagonal(knm, 0.0)

inertia = np.ones(N) * 5.0  # H = 5 seconds
damping = np.ones(N) * 1.0  # D = 1.0

engine = InertialKuramotoEngine(N, dt=0.01)
th_final, od_final, th_traj, od_traj = engine.run(
    theta, omega_dot, power, knm, inertia, damping, n_steps=1000
)

print(f"R = {engine.coherence(th_final):.4f}")
print(f"Max freq deviation: {engine.frequency_deviation(od_final):.4f} Hz")
```

### 5.2 Generator Trip (Large Perturbation)

```python
# Simulate loss of generator (sudden power imbalance)
power_trip = power.copy()
power_trip[0] = 0.0  # generator 0 trips offline
power_trip -= power_trip.mean()  # redistribute

th, od = th_final.copy(), od_final.copy()
th_post, od_post, _, od_traj_post = engine.run(
    th, od, power_trip, knm, inertia, damping, n_steps=5000
)

max_dev = np.max(np.abs(od_traj_post)) / (2 * np.pi)
print(f"Max transient frequency deviation: {max_dev:.4f} Hz")
# If max_dev > 2.5 Hz: system unstable → load shedding needed
```

### 5.3 Heterogeneous Inertia

```python
# Mix of conventional (high M) and renewable (low M) generators
inertia_mixed = np.array([8, 8, 8, 8, 8, 0.5, 0.5, 0.5, 0.5, 0.5])
# Renewables have low inertia → faster frequency swings

engine_mixed = InertialKuramotoEngine(N, dt=0.005)
_, _, _, od_traj_mixed = engine_mixed.run(
    theta, np.zeros(N), power, knm, inertia_mixed, damping, n_steps=2000
)

# Renewable generators (last 5) will show larger frequency oscillations
renewable_max = np.max(np.abs(od_traj_mixed[:, 5:])) / (2 * np.pi)
conventional_max = np.max(np.abs(od_traj_mixed[:, :5])) / (2 * np.pi)
print(f"Renewable max Δf: {renewable_max:.4f} Hz")
print(f"Conventional max Δf: {conventional_max:.4f} Hz")
```

### 5.4 Damping Ratio Analysis

```python
K_eff = 3.0  # effective coupling
M = 5.0      # inertia
D_values = [0.1, 1.0, 5.0, 20.0]

for D in D_values:
    zeta = D / (2 * np.sqrt(K_eff * M))
    regime = "overdamped" if zeta > 1 else "underdamped"
    eng = InertialKuramotoEngine(N, dt=0.01)
    _, od_f, _, _ = eng.run(
        rng.uniform(0, 2*np.pi, N), np.zeros(N), power,
        knm, np.ones(N)*M, np.ones(N)*D, n_steps=2000,
    )
    max_f = engine.frequency_deviation(od_f)
    print(f"D={D:.1f}, ζ={zeta:.2f} ({regime}): Δf_max={max_f:.4f} Hz")
```

### 5.5 SCPN Layer 6 Configuration

```python
# SCPN Layer 6: Planetary timescale (τ = 86400s = 24h)
N_layer6 = 4  # 4 oscillators in layer 6
omega_planet = np.ones(N_layer6) * 2 * np.pi / 86400  # 1 cycle per day
M_planet = np.ones(N_layer6) * 1.0  # normalised inertia
D_planet = np.ones(N_layer6) * 0.5  # moderate damping
P_planet = np.zeros(N_layer6)       # no external power

knm_planet = np.full((N_layer6, N_layer6), 0.5 / N_layer6)
np.fill_diagonal(knm_planet, 0.0)

engine_planet = InertialKuramotoEngine(N_layer6, dt=100.0)  # 100s timestep
th_p, od_p, _, _ = engine_planet.run(
    rng.uniform(0, 2*np.pi, N_layer6), np.zeros(N_layer6),
    P_planet, knm_planet, M_planet, D_planet, n_steps=864,
)
print(f"Planetary coherence: {engine_planet.coherence(th_p):.4f}")
```

### 5.6 Transient Stability Assessment

```python
# Run post-fault simulation and check if system re-synchronises
# Critical clearing time: how long a fault can persist before instability

fault_durations = [0.1, 0.2, 0.5, 1.0, 2.0]  # seconds
for t_fault in fault_durations:
    n_fault = int(t_fault / 0.01)
    # During fault: one generator decoupled (row/col of knm zeroed)
    knm_fault = knm.copy()
    knm_fault[0, :] = 0.0
    knm_fault[:, 0] = 0.0

    th, od = theta.copy(), np.zeros(N)
    # Fault period
    th, od, _, _ = engine.run(th, od, power, knm_fault, inertia, damping, n_fault)
    # Post-fault recovery
    th, od, _, od_post = engine.run(th, od, power, knm, inertia, damping, 5000)

    max_dev = np.max(np.abs(od_post[-100:])) / (2 * np.pi)
    stable = max_dev < 0.5
    print(f"Fault {t_fault:.1f}s: {'STABLE' if stable else 'UNSTABLE'} "
          f"(Δf_max={max_dev:.3f} Hz)")
```

### 5.7 Energy Monitoring

```python
# Track energy-like function along trajectory
def compute_energy(th, od, knm, inertia, power):
    kinetic = 0.5 * np.sum(inertia * od**2)
    potential = 0.0
    for i in range(len(th)):
        for j in range(i+1, len(th)):
            potential -= knm[i,j] * np.cos(th[j] - th[i])
    return kinetic + potential

th_f, od_f, th_traj, od_traj = engine.run(
    theta, np.zeros(N), power, knm, inertia, damping, n_steps=1000
)
energies = [compute_energy(th_traj[t], od_traj[t], knm, inertia, power)
            for t in range(0, 1000, 10)]
print(f"Energy: {energies[0]:.2f} → {energies[-1]:.2f} (should decrease)")
```

### 5.8 Frequency Nadir Detection

```python
# Find the worst frequency deviation after a disturbance (frequency nadir)
_, _, _, od_traj = engine.run(
    theta, np.zeros(N), power, knm, inertia, damping, n_steps=3000
)

freq_dev_hz = np.abs(od_traj) / (2 * np.pi)
nadir_time = np.argmax(np.max(freq_dev_hz, axis=1))
nadir_value = np.max(freq_dev_hz[nadir_time])
nadir_gen = np.argmax(freq_dev_hz[nadir_time])
print(f"Frequency nadir: {nadir_value:.4f} Hz at step {nadir_time}, "
      f"generator {nadir_gen}")
```

---

## 6. Technical Reference

### 6.1 Module API

::: scpn_phase_orchestrator.upde.inertial
    options:
        show_root_heading: true
        members_order: source

### 6.2 Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n` | `int` | — | Number of oscillators |
| `dt` | `float` | `0.01` | Integration timestep |

### 6.3 step() Parameters

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `theta` | `(N,)` | Rotor angles |
| `omega_dot` | `(N,)` | Angular velocity deviations |
| `power` | `(N,)` | Mechanical power $P_i$ |
| `knm` | `(N,N)` | Coupling matrix |
| `inertia` | `(N,)` | Inertia constants $M_i$ |
| `damping` | `(N,)` | Damping coefficients $D_i$ |

Returns: `(theta_new, omega_dot_new)` — both `(N,)` arrays.

### 6.4 run() Returns

`(theta_final, omega_final, theta_trajectory, omega_trajectory)`

- `theta_trajectory`: shape `(n_steps, N)`
- `omega_trajectory`: shape `(n_steps, N)`

### 6.5 Diagnostic Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `frequency_deviation(omega_dot)` | `float` | $\max_i |\dot{\omega}_i| / 2\pi$ (Hz) |
| `coherence(theta)` | `float` | $R = |⟨e^{i\theta}⟩|$ |

---

## 7. Performance Benchmarks

### 7.1 Rust Speedup

| N | Python (µs/step) | Rust (µs/step) | Speedup |
|---|-------------------|----------------|---------|
| 8 | 85 | 1.4 | 60x |
| 16 | 140 | 2.8 | 50x |
| 64 | 1400 | 35 | 40x |
| 256 | 22000 | 560 | 39x |

RK4 requires 4 coupling evaluations per step (each $O(N^2)$), making
it 4x more expensive than Euler per step but with $O(\Delta t^4)$ accuracy.

### 7.2 Memory

State: $2N$ (theta + omega_dot) + trajectory: $2 \times n_{\text{steps}} \times N$.

| N | n_steps | Trajectory memory |
|---|---------|-------------------|
| 16 | 1000 | 250 KB |
| 64 | 5000 | 5 MB |
| 256 | 10000 | 40 MB |

### 7.3 Complexity

| Operation | Time | Space |
|-----------|------|-------|
| `step()` | $4 \times O(N^2)$ | $O(N^2)$ scratch |
| `run(n)` | $4n \times O(N^2)$ | $O(n \cdot N)$ trajectory |

### 7.4 Timestep Selection

The swing equation is moderately stiff for large $M/D$ ratios. The CFL-like
stability condition for Euler would be $\Delta t < 2D / K$. RK4 is more
permissive — stable for $\Delta t \leq 0.01 / \sqrt{K/M}$ typically.

| M/D ratio | Recommended dt | Steps for 10s simulation |
|-----------|----------------|--------------------------|
| 0.1 (overdamped) | 0.05 | 200 |
| 1.0 (balanced) | 0.01 | 1,000 |
| 10 (underdamped) | 0.002 | 5,000 |
| 100 (very stiff) | 0.0005 | 20,000 |

For very stiff systems ($M/D > 100$), consider implicit methods (not yet
implemented in SPO).

### 7.5 Comparison with UPDEEngine

For the same N=64, K=3.0/N network:

| Engine | Method | µs/step (Rust) | State dim | Use case |
|--------|--------|----------------|-----------|----------|
| `UPDEEngine` | Euler | 57 | N | Fast, no inertia |
| `UPDEEngine` | RK4 | 257 | N | Accurate, no inertia |
| `InertialKuramotoEngine` | RK4 | 35 (!) | 2N | Inertia + damping |

Note: Inertial Rust is faster than UPDEEngine RK4 Python because Rust
eliminates intermediate array allocation.

---

## 8. Citations

1. **Filatrella G., Nielsen A.H., Pedersen N.F.** (2008). Analysis of a
   power grid using a Kuramoto-like model. *European Physical Journal B*
   **61**(4):485–491. doi:10.1140/epjb/e2008-00098-8

2. **Dörfler F., Bullo F.** (2012). Synchronization and transient stability
   in power networks and nonuniform Kuramoto oscillators. *SIAM Journal
   on Control and Optimization* **50**(3):1616–1642.
   doi:10.1137/110851584

3. **Rohden M., Sorge A., Timme M., Witthaut D.** (2012). Self-organized
   synchronization in decentralized power grids. *Physical Review Letters*
   **109**(6):064101. doi:10.1103/PhysRevLett.109.064101

4. **Dörfler F., Bullo F.** (2014). Synchronization in complex networks
   of phase oscillators: A survey. *Automatica* **50**(6):1539–1564.
   doi:10.1016/j.automatica.2014.04.012

5. **Park R.H.** (1929). Two-reaction theory of synchronous machines:
   Generalized method of analysis — Part I. *Transactions of the AIEE*
   **48**(3):716–727. doi:10.1109/T-AIEE.1929.5055275

6. **Bergen A.R., Hill D.J.** (1981). A structure preserving model for
   power system stability analysis. *IEEE Transactions on Power Apparatus
   and Systems* **PAS-100**(1):25–35. doi:10.1109/TPAS.1981.316883

---

## Test Coverage

- `tests/test_inertial.py` — 13 tests: step shape, run trajectory shapes,
  zero power convergence, frequency deviation bounds, coherence range,
  high damping overdamped behaviour, low damping oscillatory behaviour,
  heterogeneous inertia, Rust parity, energy-like conservation

Total: **13 tests**.

---

## Source

- Python: `src/scpn_phase_orchestrator/upde/inertial.py` (105 lines)
- Rust: `spo-kernel/crates/spo-engine/src/inertial.rs`
- FFI: `spo-kernel/crates/spo-ffi/src/lib.rs` (PyInertialStepper)
