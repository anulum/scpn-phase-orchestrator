# Neural Network Module (nn)

Differentiable Kuramoto dynamics for neural network integration via
JAX and equinox. Every function and layer is JIT-compilable, vmap-compatible,
and fully differentiable — enabling gradient-based coupling inference,
synchronisation optimisation, and physics-informed machine learning.

**Requires:** `pip install scpn-phase-orchestrator[nn]` (installs jax + equinox + optax)

## Architecture

```
                        ┌─────────────────────────┐
                        │   Functional API (JAX)   │
                        │  kuramoto_forward()      │
                        │  winfree_forward()       │
                        │  simplicial_forward()    │
                        │  stuart_landau_forward() │
                        │  order_parameter()       │
                        └───────────┬─────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ↓               ↓               ↓
            KuramotoLayer    SimplicialLayer   StuartLandauLayer
            (eqx.Module)    (eqx.Module)      (eqx.Module)
                    │               │               │
                    ↓               ↓               ↓
              training.py     UDEKuramotoLayer   BOLDGenerator
              (loss + optim)  (physics + MLP)    (hemodynamics)
                    │
          ┌─────────┼──────────┐
          ↓         ↓          ↓
     InverseKuramoto  Reservoir   OIM
     (coupling inference) (readout) (combinatorial)
```

---

## Functional API

Pure JAX functions — no state, no side effects. Each function is
decorated with `@jax.jit` internally or designed to be JIT'd by the caller.

### Kuramoto model

| Function | Signature |
|----------|-----------|
| `kuramoto_step` | `(phases, omegas, K, dt) → phases` |
| `kuramoto_rk4_step` | `(phases, omegas, K, dt) → phases` |
| `kuramoto_forward` | `(phases, omegas, K, dt, n_steps, method="rk4") → (final, traj)` |

Masked (sparse) variants append `_masked` and take an additional
`mask: jax.Array` parameter for selective coupling.

### Winfree model

| Function | Signature |
|----------|-----------|
| `winfree_step` | `(phases, omegas, K, dt) → phases` |
| `winfree_rk4_step` | `(phases, omegas, K, dt) → phases` |
| `winfree_forward` | `(phases, omegas, K, dt, n_steps, method="rk4") → (final, traj)` |

Winfree coupling: dθ_i/dt = ω_i + K · Q(θ_i) · Σ P(θ_j) where
Q is the sensitivity function and P is the pulse function.

### Simplicial (3-body) model

| Function | Signature |
|----------|-----------|
| `simplicial_step` | `(phases, omegas, K, dt, sigma2=0.0) → phases` |
| `simplicial_rk4_step` | `(phases, omegas, K, dt, sigma2=0.0) → phases` |
| `simplicial_forward` | `(phases, omegas, K, dt, n_steps, sigma2=0.0, method="rk4") → (final, traj)` |

The `sigma2` parameter controls the 3-body interaction strength.
When sigma2=0, reduces to standard Kuramoto.

### Stuart-Landau model

| Function | Signature |
|----------|-----------|
| `stuart_landau_step` | `(phases, amps, omegas, mu, K, K_r, dt, eps=1.0) → (phases, amps)` |
| `stuart_landau_rk4_step` | `(phases, amps, omegas, mu, K, K_r, dt, eps=1.0) → (phases, amps)` |
| `stuart_landau_forward` | `(phases, amps, omegas, mu, K, K_r, dt, n_steps, eps=1.0, method="rk4") → (phases, amps, phase_traj, amp_traj)` |

### Analysis functions

| Function | Returns | Description |
|----------|---------|-------------|
| `order_parameter(phases)` | scalar | Kuramoto R = \|Σ exp(iθ)\|/N |
| `plv(trajectory)` | (N,N) array | Pairwise phase-locking value |
| `coupling_laplacian(K)` | (N,N) array | Graph Laplacian L = D - K |
| `saf_order_parameter(K, omegas)` | scalar | Self-consistent analytical R |
| `saf_loss(K, omegas, budget)` | scalar | Differentiable SAF loss |

::: scpn_phase_orchestrator.nn.functional

---

## KuramotoLayer

Equinox module wrapping Kuramoto dynamics as a learnable layer.

```python
KuramotoLayer(
    n: int,              # number of oscillators
    n_steps: int = 50,   # integration steps per forward pass
    dt: float = 0.01,    # timestep
    K_scale: float = 0.1, # initialisation scale for K
    mask: jax.Array | None = None,  # sparse coupling mask
    key: jax.Array,      # PRNG key
)
```

**Learnable parameters:** `K` (coupling matrix), `omegas` (frequencies).

| Method | Signature | Description |
|--------|-----------|-------------|
| `__call__` | `(phases) → final_phases` | Forward pass |
| `forward_with_trajectory` | `(phases) → (final, trajectory)` | With full trajectory |
| `sync_score` | `(phases) → R` | Order parameter after forward pass |

::: scpn_phase_orchestrator.nn.kuramoto_layer

---

## SimplicialKuramotoLayer

Extends KuramotoLayer with learnable 3-body interaction strength σ₂.

```python
SimplicialKuramotoLayer(
    n: int,
    n_steps: int = 50,
    dt: float = 0.01,
    K_scale: float = 0.1,
    sigma2_init: float = 0.0,  # initial 3-body strength
    key: jax.Array,
)
```

**Learnable parameters:** `K`, `omegas`, `sigma2`.

When sigma2=0, output matches KuramotoLayer (verified in tests).

::: scpn_phase_orchestrator.nn.simplicial_layer

---

## StuartLandauLayer

Phase + amplitude dynamics with learnable bifurcation parameters.

```python
StuartLandauLayer(
    n: int,
    n_steps: int = 50,
    dt: float = 0.01,
    K_scale: float = 0.1,
    epsilon: float = 1.0,
    key: jax.Array,
)
```

**Learnable parameters:** `K`, `K_r` (amplitude coupling), `omegas`, `mu` (bifurcation).

| Method | Returns | Description |
|--------|---------|-------------|
| `__call__` | `(phases, amps)` | Forward phase + amplitude |
| `sync_score` | scalar | R from final phases |
| `mean_amplitude` | scalar | Mean amplitude after forward |

::: scpn_phase_orchestrator.nn.stuart_landau_layer

---

## BOLD Signal Generator

Balloon-Windkessel hemodynamic model converting oscillator amplitudes
to simulated fMRI BOLD signal. Differentiable for gradient-based
fMRI fitting.

| Function | Description |
|----------|-------------|
| `balloon_windkessel_step` | One Euler step of BW hemodynamics |
| `bold_signal` | V,Q → BOLD observation equation |
| `bold_from_neural` | Full neural → BOLD conversion |

State variables: signal (s), flow (f), volume (v), deoxyhemoglobin (q).

::: scpn_phase_orchestrator.nn.bold

---

## Reservoir Computing

Kuramoto-based echo state network with linear readout.

| Function | Description |
|----------|-------------|
| `reservoir_features(phases)` | cos/sin feature extraction |
| `reservoir_drive(phases, omegas, K, W_in, u, dt, n_steps)` | Driven reservoir dynamics |
| `ridge_readout(features, targets, alpha=1e-4)` | Ridge regression readout weights |
| `reservoir_predict(features, W_out)` | Prediction from trained readout |

Universal approximation near edge-of-bifurcation (arXiv:2407.16172).

::: scpn_phase_orchestrator.nn.reservoir

---

## UDE-Kuramoto (Universal Differential Equation)

Physics backbone (sin(Δθ) coupling) plus a learned neural residual.
The MLP handles model mismatch that the analytical Kuramoto model
cannot capture.

### CouplingResidual (eqx.Module)

```python
CouplingResidual(hidden: int = 16, key: jax.Array)
```

Small MLP: Linear(1, hidden) → tanh → Linear(hidden, 1).

### UDEKuramotoLayer (eqx.Module)

```python
UDEKuramotoLayer(n, n_steps=50, dt=0.01, K_scale=0.1, hidden=16, key=key)
```

**Learnable:** `K`, `omegas`, `residual` (CouplingResidual MLP).

::: scpn_phase_orchestrator.nn.ude

---

## Inverse Kuramoto

Gradient-based inference of K and ω from observed phase trajectories.

| Function | Description |
|----------|-------------|
| `infer_coupling(observed, dt, n_epochs, lr, ...)` | Full gradient descent inference |
| `analytical_inverse(observed, dt, alpha)` | Closed-form least-squares |
| `hybrid_inverse(observed, dt, ...)` | Analytical + gradient refinement |
| `inverse_loss(K, omegas, observed, dt, l1)` | Differentiable loss |
| `coupling_correlation(K_true, K_inferred)` | Pearson r for validation |

::: scpn_phase_orchestrator.nn.inverse

---

## Oscillator Ising Machine (OIM)

Combinatorial optimisation via phase clustering. Maps graph colouring,
max-cut, and QUBO to Kuramoto dynamics.

| Function | Description |
|----------|-------------|
| `oim_solve(adj, n_colors, key, ...)` | Full solver with annealing + restarts |
| `oim_forward(phases, adj, n_colors, dt, n_steps)` | Forward integration |
| `extract_coloring(phases, n_colors)` | Hard colour assignment |
| `coloring_violations(colors, adj)` | Count constraint violations |
| `coloring_energy(phases, adj, n_colors)` | Continuous energy |

First open-source OIM simulator.

::: scpn_phase_orchestrator.nn.oim

---

## Training Utilities

### Loss functions

| Function | Description |
|----------|-------------|
| `sync_loss(model, phases, target_R=1.0)` | (1 - R)² loss |
| `trajectory_loss(model, phases, observed)` | MSE on phase trajectory |
| `coupling_sparsity_loss(K, target_density)` | L1 sparsity penalty |

### Training loop

```python
from scpn_phase_orchestrator.nn.training import train

model, losses = train(
    model=layer,
    loss_fn=lambda m: sync_loss(m, phases),
    optimizer=optax.adam(1e-3),
    n_epochs=500,
    callback=lambda ep, m, l: print(f"Epoch {ep}: loss={float(l):.4f}"),
)
```

### Data generation

| Function | Returns |
|----------|---------|
| `generate_kuramoto_data(N, T, dt, K_scale, key)` | `(K, omegas, phases_init, trajectory)` |
| `generate_chimera_data(N, T, dt, coupling, range, key)` | `(K, omegas, trajectory)` |

::: scpn_phase_orchestrator.nn.training

---

## Physics validation

The nn module includes 13 physics validation test files
(`test_nn_physics_validation_p1` through `_p13`) verifying:

- Energy conservation under Hamiltonian coupling
- Gradient correctness via finite-difference comparison
- Order parameter convergence for strong coupling
- Stuart-Landau bifurcation (subcritical → supercritical)
- Simplicial explosive synchronisation
- BOLD hemodynamic response shape
- Reservoir echo state property
- UDE residual convergence
- OIM graph colouring correctness
- Inverse coupling recovery (r > 0.95)
