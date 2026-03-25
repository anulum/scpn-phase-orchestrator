# Differentiable Kuramoto Layer

The `nn` module exposes Kuramoto phase dynamics as a differentiable building block
for JAX-based machine learning pipelines.

## Installation

```bash
pip install scpn-phase-orchestrator[nn]
```

This installs `jax>=0.4` and `equinox>=0.11`.

## Functional API

Use standalone functions when you need full control over parameters:

```python
import jax
import jax.numpy as jnp
from scpn_phase_orchestrator.nn import (
    kuramoto_step,
    kuramoto_rk4_step,
    kuramoto_forward,
    order_parameter,
    plv,
)

key = jax.random.PRNGKey(0)
N = 16
phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
omegas = jax.random.normal(key, (N,))
K = jnp.eye(N) * 0.5  # coupling matrix

# Single RK4 step
new_phases = kuramoto_rk4_step(phases, omegas, K, dt=0.01)

# Run 100 steps, get trajectory
final, trajectory = kuramoto_forward(phases, omegas, K, dt=0.01, n_steps=100)

# Measure synchronization
R = order_parameter(final)          # scalar in [0, 1]
P = plv(trajectory)                 # (N, N) phase-locking value matrix
```

All functions are `jax.jit`-compilable and `jax.vmap`-compatible:

```python
# Batch over initial conditions
batch_phases = jax.random.uniform(key, (32, N), maxval=2.0 * jnp.pi)
batch_step = jax.vmap(kuramoto_step, in_axes=(0, None, None, None))
batch_out = batch_step(batch_phases, omegas, K, 0.01)  # (32, N)
```

## Gradient-Based Optimization

Differentiate through the full Kuramoto forward pass:

```python
def sync_loss(K):
    final, _ = kuramoto_forward(phases, omegas, K, dt=0.01, n_steps=50)
    return -order_parameter(final)  # minimize to maximize sync

grad_K = jax.grad(sync_loss)(K)  # (N, N) gradient of loss w.r.t. coupling
```

## KuramotoLayer (equinox)

For integration into neural network architectures:

```python
import equinox as eqx
from scpn_phase_orchestrator.nn import KuramotoLayer

key = jax.random.PRNGKey(42)
layer = KuramotoLayer(n=16, n_steps=50, dt=0.01, key=key)

# Forward pass
output_phases = layer(phases)

# Differentiable sync score
R = layer.sync_score(phases)

# Gradient w.r.t. learnable parameters (K and omegas)
@eqx.filter_grad
def compute_grads(model):
    return model.sync_score(phases)

grads = compute_grads(layer)
# grads.K is (16, 16), grads.omegas is (16,)
```

## Simplicial (3-Body) Kuramoto

Higher-order interactions beyond pairwise coupling. The 3-body term produces
explosive (first-order) synchronization transitions (Gambuzza et al. 2023,
Nature Physics).

```python
from scpn_phase_orchestrator.nn import (
    simplicial_step,
    simplicial_forward,
    order_parameter,
)

# sigma2 controls 3-body coupling strength (0 = standard Kuramoto)
new_phases = simplicial_step(phases, omegas, K, dt=0.01, sigma2=0.5)

# Full trajectory with higher-order dynamics
final, trajectory = simplicial_forward(
    phases, omegas, K, dt=0.01, n_steps=100, sigma2=0.5
)

# Differentiate through 3-body dynamics
def loss(sigma2):
    final, _ = simplicial_forward(phases, omegas, K, 0.01, 50, sigma2)
    return order_parameter(final)

grad_sigma2 = jax.grad(loss)(0.5)  # gradient of sync w.r.t. 3-body strength
```

This is the first differentiable implementation of simplicial Kuramoto dynamics.
Existing libraries (XGI, HyperGraphX) can simulate but cannot differentiate.

## GPU Acceleration

JAX automatically uses GPU when available. On Linux (or WSL2):

```bash
pip install jax[cuda12]
```

No code changes needed — the same functions run on GPU transparently.
XLA compilation happens once per function signature, then runs at native speed.
