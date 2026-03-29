# nn/ Module — Complete API Reference

The `nn/` module is SPO's differentiable JAX backend. Every function is
JIT-compilable, `vmap`-compatible, and fully differentiable via JAX autodiff.
It turns oscillator dynamics into gradient-trainable building blocks.

**Installation:** `pip install scpn-phase-orchestrator[nn]`
(installs `jax>=0.4`, `equinox>=0.11`, `optax>=0.2`)

---

## Architecture

```
nn/
├── functional.py          Pure JAX functions (no state)
├── kuramoto_layer.py      KuramotoLayer (equinox Module)
├── stuart_landau_layer.py StuartLandauLayer (equinox Module)
├── simplicial_layer.py    SimplicialKuramotoLayer (equinox Module)
├── theta_neuron.py        ThetaNeuronLayer + functional
├── ude.py                 UDEKuramotoLayer + CouplingResidual
├── inverse.py             Coupling matrix inference (3 methods)
├── oim.py                 Oscillator Ising Machine
├── bold.py                Balloon-Windkessel hemodynamic model
├── reservoir.py           Kuramoto reservoir computing
├── chimera.py             Chimera state detection
├── spectral.py            Laplacian spectral analysis
├── training.py            Loss functions + training loop
└── __init__.py            Lazy-loaded public API (90 symbols)
```

All imports are lazy: `import scpn_phase_orchestrator.nn` succeeds without
JAX installed. Symbols resolve on first attribute access.

---

## 1. Functional API (`functional.py`)

Stateless functions operating on JAX arrays. No side effects, no mutable
state. Every function listed here accepts and returns `jax.Array`.

### 1.1 Kuramoto Model

The standard Kuramoto model (Kuramoto 1975):

$$\frac{d\theta_i}{dt} = \omega_i + \sum_j K_{ij} \sin(\theta_j - \theta_i)$$

| Function | Integrator | Signature |
|---|---|---|
| `kuramoto_step` | Euler | `(phases, omegas, K, dt) → phases` |
| `kuramoto_rk4_step` | RK4 | `(phases, omegas, K, dt) → phases` |
| `kuramoto_forward` | scan(RK4\|Euler) | `(phases, omegas, K, dt, n_steps, method) → (final, trajectory)` |

**Parameters:**

- `phases`: `(N,)` oscillator phases in [0, 2π)
- `omegas`: `(N,)` natural frequencies
- `K`: `(N, N)` coupling matrix
- `dt`: `float` integration timestep
- `n_steps`: `int` number of integration steps
- `method`: `"rk4"` (default) or `"euler"`

**Returns:**

- `kuramoto_step`, `kuramoto_rk4_step`: `(N,)` updated phases, wrapped to [0, 2π)
- `kuramoto_forward`: tuple `(final, trajectory)` where `final` is `(N,)` and `trajectory` is `(n_steps, N)`

`kuramoto_forward` uses `jax.lax.scan` internally, making it efficient for
XLA compilation and enabling gradient flow through the full trajectory.

### 1.2 Masked (Sparse) Kuramoto

Identical to standard Kuramoto but with a binary mask for sparse coupling:

$$\frac{d\theta_i}{dt} = \omega_i + \sum_j K_{ij} \cdot M_{ij} \cdot \sin(\theta_j - \theta_i)$$

| Function | Integrator | Extra parameter |
|---|---|---|
| `kuramoto_step_masked` | Euler | `mask: (N, N)` binary |
| `kuramoto_rk4_step_masked` | RK4 | `mask: (N, N)` binary |
| `kuramoto_forward_masked` | scan | `mask: (N, N)` binary |

The mask is static (not learnable). Use this when network topology is known
but coupling weights are learnable.

### 1.3 Winfree Model

The Winfree model (Winfree 1967) — pulse-coupled oscillators with separate
sensitivity and influence functions:

$$\frac{d\theta_i}{dt} = \omega_i + \frac{K}{N} \cdot Q(\theta_i) \cdot \sum_j P(\theta_j)$$

where $P(\theta) = 1 + \cos(\theta)$ (pulse function) and $Q(\theta) = -\sin(\theta)$ (phase response curve).

| Function | Integrator |
|---|---|
| `winfree_step` | Euler |
| `winfree_rk4_step` | RK4 |
| `winfree_forward` | scan(RK4\|Euler) |

**Parameters:** `(phases, omegas, K, dt)` where `K` is a **scalar** coupling
strength (not a matrix). The all-to-all coupling is implicit.

### 1.4 Simplicial (3-Body) Kuramoto

Extends Kuramoto with higher-order 3-body interactions (Gambuzza et al.
2023, Nature Physics; Tang et al. 2025):

$$\frac{d\theta_i}{dt} = \omega_i + \sum_j K_{ij} \sin(\theta_j - \theta_i) + \frac{\sigma_2}{N^2} \sum_{j,k} \sin[(\theta_j - \theta_i) + (\theta_k - \theta_i)]$$

The 3-body term is computed efficiently as $\frac{2\sigma_2}{N^2} S_i C_i$
where $S_i = \sum_j \sin(\theta_j - \theta_i)$ and $C_i = \sum_j \cos(\theta_j - \theta_i)$.

| Function | Extra parameter |
|---|---|
| `simplicial_step` | `sigma2: float` (default 0.0) |
| `simplicial_rk4_step` | `sigma2: float` |
| `simplicial_forward` | `sigma2: float` |

When `sigma2=0`, these reduce to standard pairwise Kuramoto. Nonzero `sigma2`
produces explosive (first-order) synchronisation transitions — a qualitatively
different phenomenon from the continuous (second-order) transition of
standard Kuramoto.

### 1.5 Stuart-Landau Model

Coupled phase-amplitude oscillators. Unlike Kuramoto (phase only),
Stuart-Landau carries amplitude dynamics — enabling representation of
feature presence (amplitude > 0) alongside binding (phase).

Phase:
$$\frac{d\theta_i}{dt} = \omega_i + \sum_j K_{ij} \sin(\theta_j - \theta_i)$$

Amplitude:
$$\frac{dr_i}{dt} = (\mu_i - r_i^2) r_i + \epsilon \sum_j K^r_{ij} \cdot r_j \cdot \cos(\theta_j - \theta_i)$$

| Function | Returns |
|---|---|
| `stuart_landau_step` | `(new_phases, new_amplitudes)` |
| `stuart_landau_rk4_step` | `(new_phases, new_amplitudes)` |
| `stuart_landau_forward` | `(final_p, final_r, traj_p, traj_r)` |

**Additional parameters:**

- `amplitudes`: `(N,)` oscillator amplitudes (r >= 0)
- `mu`: `(N,)` bifurcation parameters. `mu > 0`: supercritical (oscillators converge to amplitude sqrt(mu)). `mu < 0`: subcritical (amplitudes decay to 0).
- `K_r`: `(N, N)` amplitude coupling matrix
- `epsilon`: `float` amplitude coupling strength (default 1.0)

Amplitudes are clamped to >= 0 after each step.

### 1.6 Order Parameters and Metrics

| Function | Formula | Returns |
|---|---|---|
| `order_parameter(phases)` | $R = \|\langle e^{i\theta} \rangle\|$ | Scalar R in [0, 1]. R=1: perfect sync. R~0: incoherent. |
| `plv(trajectory)` | $\text{PLV}_{ij} = \|\langle e^{i(\theta_i - \theta_j)} \rangle_t\|$ | `(N, N)` matrix in [0, 1]. |
| `coupling_laplacian(K)` | $L = D - K$ | `(N, N)` Laplacian. |
| `saf_order_parameter(K, omegas)` | $r \approx 1 - \frac{1}{2N}\sum_{j=2}^N \lambda_j^{-2} \langle v^j, \omega \rangle^2$ | Scalar estimated R. |
| `saf_loss(K, omegas, budget, budget_weight)` | $-r + w \cdot \max(\|K\|_1 - B, 0)$ | Scalar loss for topology optimisation. |

The SAF (Spectral Alignment Function) provides a closed-form estimate of the
order parameter from Laplacian eigenstructure (Skardal & Taylor 2016; Song et
al. 2025). It avoids ODE integration entirely, making it ~10x faster for
gradient-based topology search.

---

## 2. Equinox Layers

Stateful modules with learnable parameters. All inherit from `equinox.Module`.
Use with `optax` for training.

### 2.1 KuramotoLayer

```python
KuramotoLayer(n, n_steps=50, dt=0.01, K_scale=0.1, mask=None, *, key)
```

**Learnable parameters:**

| Parameter | Shape | Initialisation |
|---|---|---|
| `K` | `(n, n)` | Symmetric Gaussian, scale `K_scale` |
| `omegas` | `(n,)` | Gaussian |

**Static config:** `n_steps`, `dt`, `n`, `mask`

**Methods:**

| Method | Signature | Returns |
|---|---|---|
| `__call__` | `(phases) → phases` | Final phases after `n_steps` |
| `forward_with_trajectory` | `(phases) → (final, trajectory)` | Final + `(n_steps, n)` trajectory |
| `sync_score` | `(phases) → R` | Scalar order parameter |

If `mask` is provided (binary `(n, n)` array), uses `kuramoto_forward_masked`
internally — topology is fixed, only weights are learnable.

### 2.2 StuartLandauLayer

```python
StuartLandauLayer(n, n_steps=50, dt=0.01, K_scale=0.1, epsilon=1.0, *, key)
```

**Learnable parameters:**

| Parameter | Shape | Initialisation |
|---|---|---|
| `K` | `(n, n)` | Symmetric Gaussian |
| `K_r` | `(n, n)` | Symmetric Gaussian |
| `omegas` | `(n,)` | Gaussian |
| `mu` | `(n,)` | `0.5 + 0.1 * N(0,1)` (supercritical by default) |

**Methods:**

| Method | Signature | Returns |
|---|---|---|
| `__call__` | `(phases, amplitudes) → (phases, amplitudes)` | Final state |
| `forward_with_trajectory` | `(phases, amplitudes) → (fp, fr, traj_p, traj_r)` | Full trajectories |
| `sync_score` | `(phases, amplitudes) → R` | Phase order parameter |
| `mean_amplitude` | `(phases, amplitudes) → scalar` | Mean final amplitude |

### 2.3 SimplicialKuramotoLayer

```python
SimplicialKuramotoLayer(n, n_steps=50, dt=0.01, K_scale=0.1, sigma2_init=0.0, *, key)
```

**Learnable parameters:**

| Parameter | Shape | Description |
|---|---|---|
| `K` | `(n, n)` | Pairwise coupling |
| `omegas` | `(n,)` | Natural frequencies |
| `sigma2` | scalar | 3-body coupling strength |

When `sigma2=0`, identical to `KuramotoLayer`. Gradient flows through
`sigma2`, enabling learning whether higher-order interactions are needed.

### 2.4 ThetaNeuronLayer

```python
ThetaNeuronLayer(n, n_steps=50, dt=0.01, K_scale=0.1, eta_mean=-0.5, *, key)
```

The theta neuron (Ermentrout & Kopell 1986) — canonical model for Type I
neuronal excitability:

$$\frac{d\theta_i}{dt} = (1 - \cos\theta_i) + (1 + \cos\theta_i)(\eta_i + I_{\text{syn},i})$$

where $I_{\text{syn},i} = \sum_j K_{ij}(1 - \cos\theta_j)$.

Unlike Kuramoto oscillators which always oscillate, theta neurons can be
**excitable** ($\eta < 0$): they fire only when driven by sufficient synaptic
input. This makes them suitable for modelling spiking neural networks.

**Learnable parameters:**

| Parameter | Shape | Description |
|---|---|---|
| `K` | `(n, n)` | Synaptic coupling |
| `eta` | `(n,)` | Excitability. `eta > 0`: oscillatory. `eta < 0`: excitable. |

**Functional API:** `theta_neuron_step`, `theta_neuron_rk4_step`, `theta_neuron_forward`
with signature `(phases, eta, K, dt)`.

### 2.5 UDEKuramotoLayer

```python
UDEKuramotoLayer(n, n_steps=50, dt=0.01, K_scale=0.1, hidden=16, *, key)
```

Universal Differential Equation (Rackauckas et al. 2020): known physics
backbone + learned neural residual:

$$\frac{d\theta_i}{dt} = \omega_i + \sum_j K_{ij} \cdot [\sin(\theta_j - \theta_i) + \text{NN}_\phi(\theta_j - \theta_i)]$$

The `sin(Delta_theta)` term provides the mechanistic backbone. The
`CouplingResidual` MLP (3-layer, tanh activations) handles model mismatch:
higher harmonics, asymmetric coupling, amplitude-dependent effects.

**Learnable parameters:**

| Parameter | Shape | Description |
|---|---|---|
| `K` | `(n, n)` | Coupling matrix |
| `omegas` | `(n,)` | Natural frequencies |
| `residual` | `CouplingResidual` | 3-layer MLP (1 → hidden → hidden → 1) |

**CouplingResidual architecture:**

```
Δθ (scalar) → Linear(1, 16) → tanh → Linear(16, 16) → tanh → Linear(16, 1) → correction (scalar)
```

Applied per-pair via double `jax.vmap` over the `(N, N)` phase-difference
matrix — no explicit loops.

---

## 3. Inverse Problem (`inverse.py`)

Recover coupling matrix $K$ and natural frequencies $\omega$ from observed
phase trajectories.

### 3.1 analytical_inverse (Pikovsky 2008)

```python
analytical_inverse(observed, dt, alpha=0.0) → (K, omegas)
```

Exploits Kuramoto structure directly. Finite-difference approximation of
$d\theta/dt$, build $\sin(\Delta\theta)$ basis matrix, solve via least
squares per oscillator.

- **Complexity:** $O(N^3)$ per oscillator (lstsq)
- **Accuracy:** correlation > 0.95 for noiseless data
- **Speed:** seconds (no ODE backprop)
- **Phase wrapping:** uses `atan2(sin, cos)` for central differences, correctly handling 2π boundaries

**Parameters:**

- `observed`: `(T, N)` phase trajectory, $T \geq 3$
- `dt`: integration timestep
- `alpha`: Tikhonov (ridge) regularisation. 0 = no regularisation.

**Always try this first.** Only fall back to gradient methods if data is noisy
or the underlying model isn't pure Kuramoto.

### 3.2 hybrid_inverse

```python
hybrid_inverse(observed, dt, alpha=0.0, n_refine=50, lr=0.005, window_size=10)
    → (K, omegas, losses)
```

Analytical init + gradient refinement via multiple shooting. Handles model
mismatch (noise, higher harmonics) by starting from the analytical solution
and running Adam epochs with windowed loss.

**Multiple shooting:** the trajectory is split into windows of `window_size`
steps. Each window's loss is computed independently and averaged. This prevents
gradient vanishing through long ODE integration and is fully JIT-compatible
via `vmap` across windows.

### 3.3 infer_coupling (legacy)

```python
infer_coupling(observed, dt, n_epochs=200, lr=0.01, l1_weight=0.001,
               seed=0, window_size=0, grad_clip=1.0) → (K, omegas, losses)
```

Pure gradient descent through ODE solver with Adam optimiser. Slow (minutes)
and lower accuracy than analytical methods. Kept for backward compatibility
and for cases where the forward model is not standard Kuramoto.

**Loss function:** `inverse_loss` — runs forward model from `observed[0]`,
compares prediction against observed trajectory using circular distance
$\text{mean}(1 - \cos(\Delta\theta))$.

### 3.4 coupling_correlation

```python
coupling_correlation(K_true, K_inferred) → scalar
```

Pearson correlation on upper-triangle entries (excluding diagonal).
Use this to evaluate inference quality.

---

## 4. Oscillator Ising Machine (`oim.py`)

Maps NP-hard combinatorial problems (graph colouring, max-cut, QUBO) to
coupled oscillator dynamics. Connected oscillators repel from the same phase
cluster.

**Coupling function:**

$$\frac{d\theta_i}{dt} = \kappa \sum_j A_{ij} \sin(n_c \cdot (\theta_i - \theta_j))$$

where $n_c$ is the number of colours. This drives connected nodes to
phase separations of $2\pi / n_c$ — i.e., different colour assignments.

**Energy function:**

$$E = \sum_{(i,j) \in \mathcal{E}} \cos(n_c \cdot (\theta_i - \theta_j))$$

Minimised when connected nodes are maximally separated. Differentiable.

### Functions

| Function | Description |
|---|---|
| `oim_step(phases, adjacency, n_colors, dt, coupling_strength)` | Single Euler step |
| `oim_forward(phases, adjacency, n_colors, dt, n_steps, coupling_strength)` | Trajectory via scan |
| `oim_solve(adjacency, n_colors, *, key, ...)` | Full solver with annealing + multi-restart |
| `extract_coloring(phases, n_colors)` | Floor-bucket assignment |
| `extract_coloring_soft(phases, n_colors)` | Circular-distance assignment (more accurate) |
| `coloring_violations(colors, adjacency)` | Count same-colour edges |
| `coloring_energy(phases, adjacency, n_colors)` | Continuous energy (differentiable) |

### oim_solve

```python
oim_solve(adjacency, n_colors, *, key, dt=0.05, k_min=0.1, k_max=10.0,
          n_anneal=1000, n_refine=500, n_restarts=10)
    → (best_colors, best_phases, best_energy)
```

Fully vectorised solver. All restarts run in parallel via `vmap`. Annealing
and refinement use `jax.lax.scan` — no Python loops. For 2-colouring,
automatically switches to `sin(Delta_theta)` coupling (anti-phase at π).

---

## 5. BOLD Signal Generator (`bold.py`)

Balloon-Windkessel hemodynamic model (Friston et al. 2000, Stephan et al.
2007). Converts neural activity to simulated fMRI BOLD signal.

### State variables

| Variable | Symbol | Description | Resting value |
|---|---|---|---|
| `s` | vasodilatory signal | Neural-vascular coupling | 0 |
| `f` | blood inflow | Normalised cerebral blood flow | 1 |
| `v` | blood volume | Normalised venous volume | 1 |
| `q` | deoxyhemoglobin | Normalised dHb content | 1 |

### Equations

Oxygen extraction: $E(f) = 1 - (1 - E_0)^{1/f}$

State dynamics:
$$\frac{ds}{dt} = x - \kappa s - \gamma(f - 1)$$
$$\frac{df}{dt} = s$$
$$\frac{dv}{dt} = \frac{1}{\tau}(f - v^{1/\alpha})$$
$$\frac{dq}{dt} = \frac{1}{\tau}\left(\frac{f \cdot E(f)}{E_0} - \frac{v^{1/\alpha} \cdot q}{v}\right)$$

BOLD output: $y = V_0 \cdot [k_1(1-q) + k_2(1-q/v) + k_3(1-v)]$

### Functions

| Function | Description |
|---|---|
| `balloon_windkessel_step(s, f, v, q, x, dt, ...)` | Single Euler step |
| `bold_signal(v, q)` | Compute BOLD from volume and dHb |
| `bold_from_neural(neural, dt, dt_bold=0.5)` | Full pipeline: `(T, N)` neural → `(T_bold, N)` BOLD |

### Default parameters (Stephan et al. 2007)

| Parameter | Value | Description |
|---|---|---|
| `kappa` | 0.65 | Signal decay rate (1/s) |
| `gamma` | 0.41 | Flow-dependent elimination (1/s) |
| `tau` | 0.98 | Hemodynamic transit time (s) |
| `alpha` | 0.32 | Grubb's vessel stiffness exponent |
| `E0` | 0.4 | Resting oxygen extraction fraction |
| `V0` | 0.02 | Resting blood volume fraction |
| `k1` | 2.8 | BOLD coefficient (= 7·E0) |
| `k2` | 2.0 | BOLD coefficient |
| `k3` | 1.6 | BOLD coefficient (= 2·E0 - 0.2) |

---

## 6. Reservoir Computing (`reservoir.py`)

Kuramoto oscillator network as a nonlinear reservoir. Only the readout
layer is trained. Theory: universal approximation near edge-of-bifurcation
(arXiv:2407.16172, 2024).

### Pipeline

```
Input u(t) → W_in → modulate ω → Kuramoto dynamics → extract features → W_out → prediction
```

Input signal is injected into natural frequencies:
$\omega_i(t) = \omega_i + (W_{\text{in}} \cdot u(t))_i$

### Feature extraction

```python
reservoir_features(phases) → (2N+1,)
```

Features: `[cos(θ_1), sin(θ_1), ..., cos(θ_N), sin(θ_N), R]`

### Functions

| Function | Description |
|---|---|
| `reservoir_drive(phases, omegas, K, W_in, u, dt, n_steps)` | Drive reservoir, collect `(T, 2N+1)` features |
| `ridge_readout(features, targets, alpha=1e-4)` | Train linear readout: $W = (F^TF + \alpha I)^{-1}F^TY$ |
| `reservoir_predict(features, W_out)` | Apply readout: $\hat{Y} = F \cdot W$ |

---

## 7. Chimera Detection (`chimera.py`)

Chimera states: spatiotemporal patterns where synchronised and incoherent
domains coexist (Kuramoto & Battogtokh 2002). All functions are
differentiable, enabling gradient-based search for chimera-producing
coupling matrices.

### Local order parameter

$$R_i = \left|\frac{1}{|\mathcal{N}_i|}\sum_{j \in \mathcal{N}_i} e^{i(\theta_j - \theta_i)}\right|$$

Neighbours defined by nonzero entries in $K$.

### Functions

| Function | Returns |
|---|---|
| `local_order_parameter(phases, K)` | `(N,)` local R per oscillator |
| `chimera_index(phases, K)` | Scalar variance of local R. High = chimera. |
| `detect_chimera(phases, K, coherent_threshold=0.8, incoherent_threshold=0.3)` | `(coherent_mask, incoherent_mask)` boolean arrays |

---

## 8. Spectral Analysis (`spectral.py`)

Differentiable spectral metrics via `jnp.linalg.eigh`. Gradient flows
through eigendecomposition for topology optimisation.

### Functions

| Function | Formula | Description |
|---|---|---|
| `laplacian_spectrum(K)` | eigenvalues of $L = D - K$ | `(N,)` ascending |
| `algebraic_connectivity(K)` | $\lambda_2$ | Fiedler value. 0 iff disconnected. |
| `eigenratio(K)` | $\lambda_N / \lambda_2$ | MSF synchronisability (Barahona & Pecora 2002). Lower = more synchronisable. |
| `sync_threshold(K, omegas)` | $\max|\omega_i - \omega_j| / \lambda_2$ | Critical coupling estimate (Dorfler & Bullo 2014). |

---

## 9. Training Utilities (`training.py`)

End-to-end training loop for equinox layers with optax optimisers.

### Loss functions

| Function | Description |
|---|---|
| `sync_loss(model, phases, target_R=1.0)` | $(R - R_{\text{target}})^2$ |
| `trajectory_loss(model, phases, observed)` | Mean circular distance to observed data |
| `coupling_sparsity_loss(K, target_density=0.1)` | L1 penalty toward target sparsity |

### Training loop

```python
train_step(model, loss_fn, opt_state, optimizer) → (model, opt_state, loss)
```

Single step using `eqx.filter_value_and_grad` + optax update.

```python
train(model, loss_fn, optimizer, n_epochs, *, callback=None) → (model, losses)
```

Full loop. The inner step is `eqx.filter_jit`-compiled.

### Data generation

| Function | Returns |
|---|---|
| `generate_kuramoto_data(N, T, dt, K_scale, *, key)` | `(K_true, omegas_true, phases0, trajectory)` |
| `generate_chimera_data(N, T, dt, coupling_strength, coupling_range, *, key)` | `(K, phases0, trajectory)` on a 1D ring |

`generate_chimera_data` uses non-local ring coupling (Kuramoto & Battogtokh
2002) seeded with a partially coherent initial state.

---

## 10. GPU Benchmark Results

All benchmarks use the same `tools/gpu_benchmark.py` suite. 9 benchmark
suites, checkpoint/resume, saves after each benchmark.

### Hardware comparison (Kuramoto forward pass, 100 steps, mean µs/step)

| N | L40S (cloud) | GTX 1060 (local) | Ratio |
|---|---|---|---|
| 8 | 408 | 4,271 | 10.5x |
| 64 | 472 | 1,904 | 4.0x |
| 128 | 514 | 1,870 | 3.6x |
| 256 | 775 | 2,193 | 2.8x |
| 512 | 760 | 3,939 | 5.2x |

### JAX vs NumPy crossover (500 Kuramoto steps)

| N | GTX 1060 JAX (ms) | NumPy CPU (ms) | Speedup |
|---|---|---|---|
| 128 | 649 | 24 | 0.04x |
| 256 | 870 | 78 | 0.09x |
| 512 | 517 | 460 | 0.9x |
| **1024** | **593** | **3,039** | **5.1x** |
| **2048** | **873** | **16,902** | **19.4x** |

**Crossover at N ≈ 512–1024.** Below that, NumPy on CPU is faster due to
JAX kernel launch overhead. Above N=1024, GPU parallelism dominates.

### Batched Kuramoto (vmap, 64 oscillators, 200 steps)

| Batch size | Total (ms) | Per instance (µs) |
|---|---|---|
| 1 | 674 | 674,014 |
| 4 | 752 | 187,886 |
| 16 | 406 | 25,387 |
| 64 | 631 | 9,854 |
| 256 | 595 | 2,324 |

Batching amortises kernel launch overhead. At batch=256, per-instance
cost is 290x lower than batch=1.

### Inverse coupling accuracy (gradient method, 500 epochs)

| N | Correlation | RMSE | Final loss |
|---|---|---|---|
| 4 | 0.605 | 0.122 | 2e-06 |
| 8 | 0.588 | 0.161 | 2e-06 |
| 16 | 0.543 | 0.172 | 2e-05 |
| 32 | 0.327 | 0.190 | 1.3e-05 |

### Analytical vs gradient inverse (measured 2026-03-27, L40S)

| N | Analytical corr | Analytical time (s) | Gradient corr | Gradient time (s) | Speedup |
|---|---|---|---|---|---|
| 4 | 1.000 | 0.73 | 0.676 | 81.6 | 112x |
| 8 | 0.959 | 0.55 | 0.564 | 77.6 | 142x |
| 16 | 0.845 | 0.59 | 0.527 | 84.0 | 142x |
| 32 | 0.680 | 0.62 | 0.331 | 84.2 | 136x |

`analytical_inverse` is both faster and more accurate. Use gradient methods
only when the underlying model deviates from standard Kuramoto.

---

## 11. Design Decisions

**Why JAX, not PyTorch?** `jax.lax.scan` compiles the full ODE integration
loop into a single XLA kernel. PyTorch's eager mode would require one kernel
launch per step, 50–200x overhead for typical `n_steps`.

**Why equinox, not flax/haiku?** Equinox treats modules as pytrees. Layers
are plain dataclasses with `jax.Array` fields. No separate `state` vs `params`
dictionaries. `filter_jit` and `filter_grad` operate on the module directly.

**Why lazy imports?** `import scpn_phase_orchestrator.nn` must succeed
without JAX installed. The package is used in CI environments (linting,
typing, documentation) where GPU dependencies are not available.

**Why manual Adam in inverse.py?** The inverse functions pre-date the
`training.py` module and operate on raw arrays, not equinox modules. They
use hand-rolled Adam to avoid an optax dependency for users who only need
the functional API.

---

## 12. References

- Kuramoto, Y. (1975). Self-entrainment of a population of coupled non-linear oscillators.
- Winfree, A.T. (1967). Biological rhythms and the behavior of populations of coupled oscillators. *J. Theor. Biol.* 16(1):15–42.
- Ermentrout, G.B. & Kopell, N. (1986). Parabolic bursting in an excitable system coupled with a slow oscillation. *SIAM J. Appl. Math.* 46(2):233–253.
- Pikovsky, A. (2008). Reconstruction of a scalar potential from a time series of actions. *Phys. Rev. Lett.* 100:214101.
- Friston, K.J. et al. (2000). Nonlinear responses in fMRI: the Balloon model. *NeuroImage* 12(4):466–477.
- Stephan, K.E. et al. (2007). Comparing hemodynamic models with DCM. *NeuroImage* 38(3):387–401.
- Barahona, M. & Pecora, L.M. (2002). Synchronization in small-world systems. *Phys. Rev. Lett.* 89(5):054101.
- Skardal, P.S. & Taylor, D. (2016). Optimal synchronization of directed complex networks. *SIAM J. Appl. Dyn. Syst.* 15(1):458–489.
- Rackauckas, C. et al. (2020). Universal Differential Equations for Scientific Machine Learning. *arXiv:2001.04385*.
- Kuramoto, Y. & Battogtokh, D. (2002). Coexistence of coherence and incoherence in nonlocally coupled phase oscillators. *Nonlinear Phenom. Complex Syst.* 5(4):380–385.
- Gambuzza, L.V. et al. (2023). Stability of synchronization in simplicial complexes. *Nature Physics* 17(7):1093–1098.
- Dorfler, F. & Bullo, F. (2014). Synchronization in complex networks of phase oscillators: A survey. *Automatica* 50(6):1539–1564.
- Böhm, F. & Schumacher, J. (2020). Graph coloring with physics-inspired graph neural networks. *arXiv:2009.00490*.

