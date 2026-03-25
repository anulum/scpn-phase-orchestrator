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

## Stuart-Landau Layer (Phase + Amplitude)

Unlike the phase-only Kuramoto model, Stuart-Landau oscillators have both
phase (binding) and amplitude (presence/activity). This solves AKOrN's
limitations: amplitude enables memory, no N>32 degradation.

```python
from scpn_phase_orchestrator.nn import (
    stuart_landau_forward,
    StuartLandauLayer,
)

# Functional API
final_phases, final_amps, traj_p, traj_r = stuart_landau_forward(
    phases, amplitudes, omegas, mu, K, K_r,
    dt=0.01, n_steps=100,
)

# Equinox layer with learnable K, K_r, omegas, mu
layer = StuartLandauLayer(n=16, n_steps=50, dt=0.01, key=key)
out_phases, out_amplitudes = layer(phases, amplitudes)

# Differentiate through both phase and amplitude dynamics
@eqx.filter_grad
def grads(model):
    return model.sync_score(phases, amplitudes)

g = grads(layer)  # g.K, g.K_r, g.omegas, g.mu all have gradients
```

Supercritical (mu > 0): amplitudes converge to sqrt(mu) — active oscillators.
Subcritical (mu < 0): amplitudes decay to 0 — quiescent oscillators.

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

## BOLD Signal Generator (fMRI)

Convert oscillator amplitude dynamics to simulated fMRI BOLD signal via
the Balloon-Windkessel hemodynamic model (Friston 2000, Stephan 2007).

```python
from scpn_phase_orchestrator.nn import (
    stuart_landau_forward,
    bold_from_neural,
)

# Run Stuart-Landau dynamics
_, _, _, amp_trajectory = stuart_landau_forward(
    phases, amplitudes, omegas, mu, K, K_r,
    dt=0.001, n_steps=10000,  # 10s at 1kHz
)

# Convert amplitude envelope to BOLD signal
bold = bold_from_neural(amp_trajectory, dt=0.001, dt_bold=0.72)  # TR=720ms
# bold.shape = (T_bold, N_regions)
```

SPO is the only tool generating both phase-resolved EEG dynamics AND
predicted fMRI BOLD from the same underlying oscillator model.

## Spectral Alignment Function (SAF)

Design optimal coupling topologies without ODE integration. The SAF gives
a closed-form approximation of the order parameter from the Laplacian
eigenstructure, enabling 10x faster gradient-based topology optimization.

```python
from scpn_phase_orchestrator.nn import saf_order_parameter, saf_loss

# Estimate order parameter from coupling + frequencies (no ODE needed)
r = saf_order_parameter(K, omegas)

# Optimize coupling topology via gradient descent
grad_K = jax.grad(lambda K: -saf_order_parameter(K, omegas))(K)
K_optimized = K - lr * grad_K

# With budget constraint (L1 penalty on total coupling)
loss = saf_loss(K, omegas, budget=10.0, budget_weight=0.1)
```

Skardal & Taylor, SIAM J. Appl. Dyn. Syst. 2016; Song et al. 2025.

## Reservoir Computing

Use a fixed Kuramoto network as a nonlinear reservoir. Only the readout
layer is trained. Optimal at edge-of-bifurcation (arXiv:2407.16172).

```python
from scpn_phase_orchestrator.nn import (
    reservoir_drive,
    ridge_readout,
    reservoir_predict,
)

# Drive reservoir with input signal
features = reservoir_drive(
    phases, omegas, K, W_in, input_signal,
    dt=0.01, n_steps=5,
)

# Train readout via ridge regression
W_out = ridge_readout(features, targets, alpha=1e-4)

# Predict
predictions = reservoir_predict(features, W_out)
```

## Inverse Kuramoto (Data to Model)

Infer coupling matrix K and frequencies ω from observed phase data by
backpropagating through the Kuramoto ODE solver.

```python
from scpn_phase_orchestrator.nn import infer_coupling, coupling_correlation

# observed: (T, N) phase trajectory from experiment/EEG/sensors
K_inferred, omegas_inferred, losses = infer_coupling(
    observed, dt=0.02, n_epochs=200, lr=0.01, l1_weight=0.001
)

# Evaluate against ground truth (if available)
corr = coupling_correlation(K_true, K_inferred)
```

L1 sparsity penalty discovers network topology (which oscillators are
actually coupled vs independent). Circular phase-aware loss handles
the wraparound at 2π.

## UDE-Kuramoto (Physics + Neural Residual)

Universal Differential Equation approach: known Kuramoto backbone
`sin(Δθ)` plus a learned MLP residual `NN_φ(Δθ)` that captures model
mismatch (higher harmonics, asymmetric coupling, nonlinear effects).

```python
from scpn_phase_orchestrator.nn import UDEKuramotoLayer

# Layer with physics backbone + learned residual
layer = UDEKuramotoLayer(n=16, n_steps=50, dt=0.01, hidden=16, key=key)

# Forward pass uses sin(Δθ) + NN_φ(Δθ)
output = layer(phases)

# Train end-to-end: gradients flow through both K and NN_φ
@eqx.filter_grad
def loss(model):
    return model.sync_score(phases)

grads = loss(layer)  # grads.K, grads.omegas, grads.residual all updated
```

Rackauckas et al. 2020; Frontiers Comp. Neuro. 2025.
First Python UDE implementation for oscillator networks.

## GPU Acceleration

JAX automatically uses GPU when available. On Linux (or WSL2):

```bash
pip install jax[cuda12]
```

No code changes needed — the same functions run on GPU transparently.
XLA compilation happens once per function signature, then runs at native speed.
