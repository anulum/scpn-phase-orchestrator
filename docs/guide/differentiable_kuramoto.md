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

## Oscillator Ising Machine (Graph Coloring)

Solve NP-hard combinatorial problems via Kuramoto phase clustering.
Oscillators connected by graph edges repel from the same phase cluster.

```python
from scpn_phase_orchestrator.nn import (
    oim_forward, extract_coloring, coloring_violations, coloring_energy,
)

# Define graph adjacency matrix
A = jnp.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])  # K3

# Run OIM dynamics (3 colors for K3)
phases0 = jax.random.uniform(key, (3,), maxval=2*jnp.pi)
final, traj = oim_forward(phases0, A, n_colors=3, dt=0.1, n_steps=500)

# Extract integer coloring
colors = extract_coloring(final, n_colors=3)
violations = coloring_violations(colors, A)  # 0 = valid coloring

# Differentiable energy for gradient-based optimization
energy = coloring_energy(final, A, n_colors=3)
```

Nature Scientific Reports 2017; Böhm & Schumacher 2020.
First open-source oscillator Ising machine simulator.

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

## Winfree Model (Pulse-Coupled)

The Winfree model (1967) uses separate pulse and phase response functions
instead of sinusoidal coupling. Scalar coupling strength (all-to-all).

```python
from scpn_phase_orchestrator.nn import (
    winfree_step,
    winfree_forward,
)

# K is a scalar, not a matrix
final, trajectory = winfree_forward(
    phases, omegas, K=0.5, dt=0.01, n_steps=200
)
```

## Theta Neuron (Excitable Systems)

The canonical model for Type I neuronal excitability (Ermentrout & Kopell
1986). Unlike Kuramoto oscillators which always oscillate, theta neurons
can be **excitable** (eta < 0) — they fire only when driven by sufficient
synaptic input.

```python
from scpn_phase_orchestrator.nn import (
    theta_neuron_forward,
    ThetaNeuronLayer,
)

# Functional API
eta = jnp.full(N, -0.5)  # excitable regime
K = jnp.ones((N, N)) * 0.1
final, trajectory = theta_neuron_forward(phases, eta, K, dt=0.01, n_steps=500)

# Equinox layer with learnable K and eta
layer = ThetaNeuronLayer(n=16, n_steps=100, eta_mean=-0.5, key=key)
output = layer(phases)
```

## Chimera State Detection

Detect chimera states — coexistence of synchronised and incoherent domains
(Kuramoto & Battogtokh 2002). All functions are differentiable.

```python
from scpn_phase_orchestrator.nn import (
    local_order_parameter,
    chimera_index,
    detect_chimera,
    generate_chimera_data,
)

# Generate chimera-producing data on a ring
K, phases0, trajectory = generate_chimera_data(
    N=64, T=2000, coupling_strength=0.5, coupling_range=4, key=key
)

# Detect chimera at final timestep
R_local = local_order_parameter(trajectory[-1], K)   # (N,) per-oscillator R
chi = chimera_index(trajectory[-1], K)                # scalar: high = chimera
coherent, incoherent = detect_chimera(trajectory[-1], K)  # boolean masks

# Gradient-based search for chimera-producing coupling
grad_K = jax.grad(lambda K: chimera_index(trajectory[-1], K))(K)
```

## Spectral Analysis (Topology Metrics)

Differentiable spectral metrics for coupling matrix analysis. Gradient
flows through `jnp.linalg.eigh` for topology optimisation.

```python
from scpn_phase_orchestrator.nn import (
    laplacian_spectrum,
    algebraic_connectivity,
    eigenratio,
    sync_threshold,
)

eigs = laplacian_spectrum(K)              # (N,) sorted eigenvalues
lambda2 = algebraic_connectivity(K)       # Fiedler value (0 = disconnected)
ratio = eigenratio(K)                     # lambda_N / lambda_2 (lower = better)
Kc = sync_threshold(K, omegas)            # critical coupling estimate

# Optimise topology: minimise eigenratio (maximise synchronisability)
grad_K = jax.grad(eigenratio)(K)
```

## Analytical Inverse (Pikovsky 2008)

Faster and more accurate than gradient-based inverse. O(N^3) linear
regression on sin(Delta_theta) basis, no ODE backprop.

```python
from scpn_phase_orchestrator.nn import (
    analytical_inverse,
    hybrid_inverse,
    coupling_correlation,
)

# Analytical: exact for noiseless Kuramoto, seconds not minutes
K_est, omegas_est = analytical_inverse(observed, dt=0.01)
corr = coupling_correlation(K_true, K_est)  # typically > 0.95

# Hybrid: analytical init + gradient refinement for noisy data
K_est, omegas_est, losses = hybrid_inverse(
    observed, dt=0.01, n_refine=50, lr=0.005, window_size=10
)
```

## Training Loop

End-to-end training with optax optimisers:

```python
import optax
from scpn_phase_orchestrator.nn import (
    KuramotoLayer,
    sync_loss,
    trajectory_loss,
    train,
)

layer = KuramotoLayer(n=16, n_steps=50, dt=0.01, key=key)

# Train to maximise synchronisation
def loss_fn(model):
    return sync_loss(model, phases, target_R=1.0)

trained_layer, losses = train(
    layer,
    loss_fn,
    optax.adam(1e-3),
    n_epochs=200,
)

# Or fit to observed data
def data_loss(model):
    return trajectory_loss(model, phases, observed_trajectory)

trained_layer, losses = train(layer, data_loss, optax.adam(1e-3), 100)
```

## GPU Acceleration

JAX automatically uses GPU when available. On Linux:

```bash
pip install jax[cuda12]
```

No code changes needed — the same functions run on GPU transparently.
XLA compilation happens once per function signature, then runs at native speed.

### Local benchmark (GTX 1060 6GB, 2026-03-29)

| N | JAX GPU (ms) | NumPy CPU (ms) | Speedup |
|---|---|---|---|
| 512 | 517 | 460 | 0.9x |
| 1024 | 593 | 3,039 | **5.1x** |
| 2048 | 873 | 16,902 | **19.4x** |

Crossover at N ~ 512–1024. For N > 1024, always use JAX GPU.

For the complete API reference with mathematical formulations,
see [nn/ Module Reference](../reference/nn.md).
