# Tutorial: Differentiable Kuramoto — From Zero to Gradient

This tutorial walks you through SPO's differentiable phase dynamics from
first principles. By the end you will understand:

1. What the Kuramoto model computes and why it matters
2. How to run differentiable Kuramoto dynamics in JAX
3. How to optimize coupling matrices via gradient descent
4. How to use the inverse pipeline to infer coupling from data

**Prerequisites:** Python 3.10+, `pip install scpn-phase-orchestrator[nn]`

---

## Step 1: The Kuramoto Model in 30 Seconds

N oscillators, each with a phase θ_i ∈ [0, 2π) and natural frequency ω_i.
They interact through a coupling matrix K:

```
dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j - θ_i)
```

When K is strong enough, oscillators synchronize (phases converge).
The order parameter R = |⟨exp(iθ)⟩| measures synchronization:
R = 1 means perfect sync, R ≈ 0 means incoherent.

## Step 2: Your First Differentiable Simulation

```python
import jax
import jax.numpy as jnp
from scpn_phase_orchestrator.nn import (
    kuramoto_forward, order_parameter
)

# Setup: 8 oscillators with random frequencies
key = jax.random.PRNGKey(42)
k1, k2, k3 = jax.random.split(key, 3)

N = 8
phases = jax.random.uniform(k1, (N,), maxval=2 * jnp.pi)
omegas = jax.random.normal(k2, (N,)) * 0.5
K = jnp.ones((N, N)) * 0.3  # uniform coupling
K = K.at[jnp.diag_indices(N)].set(0.0)  # no self-coupling

# Run 200 RK4 steps at dt=0.01 (2 seconds of dynamics)
final_phases, trajectory = kuramoto_forward(phases, omegas, K, dt=0.01, n_steps=200)

# Measure synchronization
R_initial = order_parameter(phases)
R_final = order_parameter(final_phases)
print(f"R: {float(R_initial):.3f} → {float(R_final):.3f}")
```

## Step 3: Differentiate Through the Dynamics

The key feature: you can compute gradients of R with respect to K.
This means you can OPTIMISE the coupling matrix to maximise synchronisation.

```python
# Define loss: minimise negative R (= maximise sync)
def sync_loss(K):
    final, _ = kuramoto_forward(phases, omegas, K, dt=0.01, n_steps=100)
    return -order_parameter(final)

# Compute gradient of sync w.r.t. coupling matrix
grad_K = jax.grad(sync_loss)(K)
print(f"Gradient shape: {grad_K.shape}")  # (8, 8)
print(f"Gradient norm: {float(jnp.linalg.norm(grad_K)):.4f}")
```

The gradient tells you: "to increase synchronization, strengthen these
couplings and weaken those ones."

## Step 4: Optimize Coupling via Gradient Descent

```python
K_opt = K.copy()
for step in range(50):
    g = jax.grad(sync_loss)(K_opt)
    K_opt = K_opt - 0.1 * g
    K_opt = K_opt.at[jnp.diag_indices(N)].set(0.0)  # maintain zero diagonal

final_opt, _ = kuramoto_forward(phases, omegas, K_opt, dt=0.01, n_steps=200)
R_optimized = order_parameter(final_opt)
print(f"R after optimization: {float(R_optimized):.3f}")
# Should be higher than R_final
```

## Step 5: Inverse Problem — Infer Coupling from Data

The real-world use case: you observe phase trajectories (from EEG, sensors,
market prices) and want to know the coupling structure.

```python
from scpn_phase_orchestrator.nn import infer_coupling, coupling_correlation

# Generate synthetic "observed" data with known coupling
K_true = jax.random.normal(k3, (N, N)) * 0.3
K_true = (K_true + K_true.T) / 2  # symmetric
K_true = K_true.at[jnp.diag_indices(N)].set(0.0)

_, observed_trajectory = kuramoto_forward(
    phases, omegas, K_true, dt=0.02, n_steps=100
)
# Prepend initial conditions
observed = jnp.concatenate([phases[jnp.newaxis, :], observed_trajectory])

# Infer coupling from the observed data
K_inferred, omegas_inferred, losses = infer_coupling(
    observed, dt=0.02, n_epochs=200, lr=0.01, l1_weight=0.001
)

# Evaluate
corr = coupling_correlation(K_true, K_inferred)
print(f"Coupling correlation: {float(corr):.3f}")
print(f"Loss: {losses[0]:.4f} → {losses[-1]:.4f}")
```

## What's Next

- [Guide: Differentiable Kuramoto](../guide/differentiable_kuramoto.md) — full API reference with Stuart-Landau, reservoir computing, spectral topology, UDE, and OIM
- [nn/ API Reference](../reference/api/nn.md) — auto-generated API docs for all nn/ modules
