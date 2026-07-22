# Neural Network Module (nn)

GPU-first differentiable Kuramoto dynamics for neural network integration via
JAX and equinox. Every function and layer is JIT-compilable, vmap-compatible,
and fully differentiable — enabling gradient-based coupling inference,
synchronisation optimisation, and physics-informed machine learning.

**Requires:** `pip install scpn-phase-orchestrator[nn]` (installs jax + equinox + optax)

## Runtime API

Production ML jobs should make the accelerator contract explicit at start-up:

```python
from scpn_phase_orchestrator.nn import (
    KuramotoLayer,
    jax_runtime_info,
    require_accelerator,
)

print(jax_runtime_info())
device = require_accelerator()
```

`require_accelerator()` raises when JAX is installed but only CPU devices are
visible. Use `require_accelerator(allow_cpu=True)` only for CI, notebooks, and
smoke tests that intentionally run without a GPU/TPU.

::: scpn_phase_orchestrator.nn.runtime

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
          ┌─────────┼──────────┬───────────────┐
          ↓         ↓          ↓               ↓
     InverseKuramoto  Reservoir   OIM   DifferentiableSupervisor
     (coupling inference) (readout) (combinatorial) (closed-loop policy)
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

Use Winfree dynamics when the observed coupling is pulse-driven rather than a
smooth sinusoidal pull. Typical cases include biological pacemakers, circadian
or neural populations with event-like signalling, flashing or firing oscillator
ensembles, endocrine or chemical pulse trains, and sensor networks where one
unit perturbs another through brief impulses. The model is the right API choice
when the phase response curve and pulse waveform are part of the hypothesis,
not just incidental numerical details.

### Simplicial (3-body) model

| Function | Signature |
|----------|-----------|
| `simplicial_step` | `(phases, omegas, K, dt, sigma2=0.0) → phases` |
| `simplicial_rk4_step` | `(phases, omegas, K, dt, sigma2=0.0) → phases` |
| `simplicial_forward` | `(phases, omegas, K, dt, n_steps, sigma2=0.0, method="rk4") → (final, traj)` |

The `sigma2` parameter controls the 3-body interaction strength.
When sigma2=0, reduces to standard Kuramoto.

Use simplicial dynamics when pairwise edges are not enough to represent the
interaction mechanism. The 3-body term models triadic or group constraints:
neural assemblies whose synchrony depends on co-active triplets, social or
multi-agent triads, reaction loops, power-network group modes, and topology
extracted from simplicial complexes or hypergraphs. This surface is intended
for regressions where cluster states, abrupt synchronization transitions, or
learned coupling cannot be explained by pairwise Kuramoto alone.

Winfree and simplicial models answer different modelling questions. Winfree
changes the timing law from smooth coupling to pulse-response coupling.
Simplicial Kuramoto changes the interaction topology from pairwise edges to
group terms. Combining them conceptually is useful for event-driven systems
with group structure, for example spiking neural assemblies, biological tissue
motifs, swarm coordination with triadic constraints, or industrial networks
whose failure modes depend on triangular dependencies rather than isolated
links.

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
| `saf_order_parameter(K, omegas, solver="auto")` | scalar | Self-consistent analytical R |
| `saf_loss(K, omegas, budget, solver="auto")` | scalar | Differentiable SAF loss |

### Spectral Alignment Function use cases

SAF estimates the synchrony of a Kuramoto network directly from the coupling
Laplacian and natural frequencies, without rolling out the ODE. Use it when the
question is "which topology should synchronise this frequency field?" rather
than "what is the phase trajectory at each time step?"

Concrete uses:

- Coupling-topology optimisation under a wiring or energy budget.
- Fast screening of candidate graphs before expensive time-domain simulation.
- Differentiable regularisation for learned `K` matrices in neural pipelines.
- Review of `auto_initial_k` matrices produced by auto-binding before runtime
  actuation.
- Sensitivity analysis for which frequency modes are poorly aligned with the
  graph Laplacian.

Solver modes:

- `solver="eigh"` computes the exact dense Laplacian eigendecomposition and is
  appropriate for small and medium dense systems where eigenvectors are needed
  for auditability.
- `solver="cg"` uses the equivalent Laplacian pseudoinverse formulation and
  conjugate-gradient matrix-vector products. This avoids full eigendecomposition
  and is the preferred GPU path for large dense systems.
- `solver="auto"` uses `eigh` up to `exact_size_limit` and switches to `cg`
  above that size.

Scaling limits: both paths still consume a dense `(N, N)` coupling matrix. The
CG path removes the cubic eigensolver bottleneck, but it does not make dense
memory disappear. For very sparse networks, keep a sparse or masked coupling
representation upstream and materialise dense `K` only when the SAF audit size
fits device memory.

Boundary contract: `K` must be a square real-valued JAX-compatible coupling
matrix and `omegas` must be a real-valued one-dimensional vector with matching
length. Boolean and complex payloads are rejected before Laplacian construction.
SAF solver controls are finite positive scalars or integers, while `saf_loss`
budget controls are finite non-negative scalars.

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

## Differentiable Chimera Metrics

JAX-native local order-parameter and chimera-index helpers for gradient-aware
topology searches.

::: scpn_phase_orchestrator.nn.chimera

## Differentiable Spectral Metrics

JAX-native graph Laplacian metrics used by topology and synchronisability
experiments.

::: scpn_phase_orchestrator.nn.spectral

## Theta Neuron Dynamics

Differentiable Ermentrout-Kopell theta-neuron dynamics for excitable systems.

::: scpn_phase_orchestrator.nn.theta_neuron

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

## Neural-ODE continuous adjoint (diffrax)

The explicit Euler map stores every step, so reverse-mode gradients cost
`O(n_steps)` memory. `solve_ude_adjoint` integrates the same UDE-Kuramoto
vector field with an adaptive solver (`diffrax.Tsit5` by default) under a
configurable adjoint — `RecursiveCheckpointAdjoint` (logarithmic checkpointing)
or `BacksolveAdjoint` (`O(1)` memory). Integration runs on the unwrapped phase
(the coupling is `2π`-periodic, so the field is wrap-invariant while an adaptive
solver must not see the `% 2π` discontinuities); wrapping is applied once, to the
returned states. The solver never mutates the global `jax_enable_x64` flag, so
the dtype of every intermediate follows the input arrays.

```python
solve_ude_adjoint(phases, omegas, K, residual_fn, *, t1, dt0=0.01,
                  solver=None, adjoint=None, rtol=1e-6, atol=1e-6,
                  max_steps=4096, saveat_ts=None, wrap=True)
```

Requires the `diffrax` dependency (the `nn`, `jax`, or `full` extra).

::: scpn_phase_orchestrator.nn.neural_ode

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

## Phase autoencoder

`nn.phase_autoencoder` learns the asymptotic phase, isochrons and
phase-sensitivity function of a limit-cycle oscillator from state time series
alone (Yawata, Fukami, Taira & Nakao 2024, *Chaos* 34, 063111). The encoder maps
the state to a three-component latent whose first two components lie on the unit
circle so that `θ = atan2(Y₂, Y₁)` is the asymptotic phase; the latent evolves by
an exactly-linear normal-form flow with learnable frequency `ω` and decay `λ`,
trained against a four-term reconstruction/phase/deviation/centring loss. The
trained weights are extracted to the pure-NumPy `oscillators.phase_reduction`
evaluator so the phase and the phase response curve are available on the control
path without JAX.

::: scpn_phase_orchestrator.nn.phase_autoencoder

---

## Differentiable Supervisor

`nn.supervisor` provides the differentiable neural policy surface for
closed-loop Kuramoto control. It is intentionally separate from
`supervisor.policy.SupervisorPolicy`: the neural policy remains a
JAX/equinox module trained over simulator or replay rollouts, while live
actuation still flows through `ControlAction`, mapper limits, and safety gates.

The built-in objective maximizes good-partition synchrony while penalizing
bad-partition synchrony, control energy, and abrupt action changes. The module
also includes a squashed-Gaussian action sampler and clipped PPO loss/train
step for on-policy RL experiments. This is a production-quality differentiable
training surface, not a claim that large-scale RL benchmarks or a preprint have
already been completed.

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from scpn_phase_orchestrator.nn import (
    DifferentiableSupervisorConfig,
    DifferentiableSupervisorPolicy,
    KuramotoSupervisorScenario,
    supervisor_train_step,
)

scenario = KuramotoSupervisorScenario(
    phases=jnp.array([0.0, 0.1, 2.7, 3.1]),
    omegas=jnp.array([0.04, 0.03, -0.03, -0.04]),
    base_K=jnp.full((4, 4), 0.03) - jnp.eye(4) * 0.03,
    good_mask=jnp.array([1.0, 1.0, 0.0, 0.0]),
    bad_mask=jnp.array([0.0, 0.0, 1.0, 1.0]),
    dt=0.02,
    inner_steps=4,
    horizon=3,
)
policy = DifferentiableSupervisorPolicy(
    DifferentiableSupervisorConfig(n_oscillators=4),
    key=jax.random.PRNGKey(0),
)
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(eqx.filter(policy, eqx.is_array))
policy, opt_state, loss = supervisor_train_step(
    policy,
    scenario,
    opt_state,
    optimizer,
)
```

::: scpn_phase_orchestrator.nn.supervisor

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
