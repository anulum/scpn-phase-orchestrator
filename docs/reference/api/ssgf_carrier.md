# SSGF Geometry Carrier

## 1. Mathematical Formalism

### The Autopoietic Geometry Loop

The Self-Stabilizing Gauge Field (SSGF) framework treats the coupling
matrix $W_{ij}$ as a dynamic carrier field that co-evolves with the
phase dynamics. The `GeometryCarrier` implements the outer cycle of
this autopoietic loop:

$$z \xrightarrow{\text{decode}} W(z) \xrightarrow{\text{microcycles}} U_{\text{total}}(W, \theta) \xrightarrow{\nabla_z} z - \eta \nabla_z U$$

### Latent Space Parametrisation

Instead of optimising the $N^2$ entries of $W$ directly, the carrier
parametrises the geometry through a low-dimensional latent vector
$z \in \mathbb{R}^d$ (typically $d = 8$):

$$W = \text{decode}(z) = \text{softplus}(A \cdot z), \quad W_{ii} = 0$$

where $A \in \mathbb{R}^{N^2 \times d}$ is a fixed random projection
matrix initialised as $A_{ij} \sim \mathcal{N}(0, 1/\sqrt{d})$.

### Softplus Activation

The softplus function ensures non-negative coupling:

$$\text{softplus}(x) = \log(1 + e^x)$$

Properties:
- $\text{softplus}(x) \geq 0$ for all $x$
- $\text{softplus}(x) \to x$ as $x \to +\infty$ (linear regime)
- $\text{softplus}(x) \to 0$ as $x \to -\infty$ (suppressed coupling)
- $\text{softplus}(0) = \ln 2 \approx 0.693$
- Smooth everywhere (unlike ReLU), enabling gradient-based optimisation

### Numerical Stability

The Rust implementation uses a three-regime approximation:

$$\text{softplus}(x) = \begin{cases} 0 & x < -20 \\ \log(1 + e^x) & -20 \leq x \leq 20 \\ x & x > 20 \end{cases}$$

This avoids `exp` overflow for large positive $x$ and underflow for
large negative $x$.

### Gradient Computation

The SSGF outer cycle needs $\nabla_z U_{\text{total}}$. Since
$U_{\text{total}}$ is only available as a black-box function of $W$
(through the microcycle integration), the carrier uses **central
finite differences**:

$$\frac{\partial U}{\partial z_i} \approx \frac{U(\text{decode}(z + \varepsilon e_i)) - U(\text{decode}(z - \varepsilon e_i))}{2\varepsilon}$$

This requires $2d$ decode + cost evaluations per gradient step.
For $d = 8$, that is 16 evaluations — manageable when the microcycle
cost function is fast.

### Latent Space Dimensionality

The latent dimension $d$ controls the expressiveness of the geometry:

| $d$ | Degrees of freedom | Expressiveness |
|-----|-------------------|----------------|
| 1 | Scalar scaling | Uniform coupling only |
| 4 | Low-rank | Distance-decay-like patterns |
| 8 | Default | Rich but smooth topologies |
| $N$ | Medium | Arbitrary symmetric patterns |
| $N^2$ | Full rank | Any matrix (defeats the purpose) |

The default $d = 8$ balances expressiveness (can represent hierarchical,
small-world, modular topologies) with optimisation tractability (8D
gradient descent is well-conditioned).

---

## 2. Theoretical Context

### Geometry as a Dynamic Variable

In classical coupled oscillator theory, the coupling matrix $W$ is
a fixed parameter. The SSGF framework (inspired by gauge field theory
in physics) promotes $W$ to a dynamic variable that evolves on a
slow timescale:

- **Fast timescale:** Phases $\theta$ evolve under $W$ (Kuramoto)
- **Slow timescale:** $W$ (via $z$) evolves to minimise the free
  energy $U_{\text{total}}$

This two-timescale separation is the hallmark of SSGF and
distinguishes it from standard Kuramoto + Hebbian plasticity.

### Relation to Variational Autoencoders

The decode architecture $z \mapsto W$ is structurally similar to
the decoder in a variational autoencoder (VAE):
- $z$ is the latent code
- $A$ is the decoder weight matrix
- softplus is the output activation

Unlike a VAE, there is no encoder: the latent vector $z$ is
optimised directly by gradient descent on the physical cost
$U_{\text{total}}$, not reconstructed from data.

### Relation to Active Inference

In the Free Energy Principle (Friston, 2010), an agent minimises
its variational free energy by updating internal parameters. The
`GeometryCarrier` implements a form of active inference where:
- **Internal model:** $z$ parametrises the agent's belief about
  optimal connectivity
- **Free energy:** $U_{\text{total}}$ measures the mismatch between
  the current geometry and the dynamics' requirements
- **Gradient descent:** Minimises free energy by adapting $z$

### Historical Context

- **Friston, K. J.** (2010): "The free-energy principle: A unified
  brain theory?" Proposed that all biological self-organisation
  minimises variational free energy.
- **Breakspear, M.** (2017): "Dynamic models of large-scale brain
  activity." Reviewed two-timescale models for brain connectivity.
- **Deco, G. et al.** (2015): "Rethinking segregation and integration:
  Contributions of whole-brain modelling." Used dynamic coupling in
  whole-brain simulation.
- **Haken, H.** (1983): *Synergetics* — the slaving principle
  (fast variables enslaved to slow order parameters) underpins
  the SSGF timescale separation.
- **Kingma, D. P. & Welling, M.** (2014): "Auto-Encoding Variational
  Bayes." The decoder architecture in the carrier is analogous.

### Autopoiesis

The carrier implements **autopoiesis** (Maturana & Varela, 1980):
the system produces its own boundary conditions (coupling topology)
from its internal dynamics. The coupling matrix $W$ emerges from
the system's own phase dynamics, not from external specification.

---

## 3. Pipeline Position

```
 ┌── GeometryCarrier(n, z_dim=8, lr=0.01) ──────┐
 │                                                │
 │  z (latent, d-dim)                             │
 │  ↓ decode()                                    │
 │  W = softplus(A·z), diag=0, shape (n, n)      │
 │                                                │
 │  W → UPDEEngine.run(phases, ω, W, ...)        │
 │     → phases_new                                │
 │     → PGBO.observe(phases_new, W) → snapshot   │
 │     → SSGFCosts.evaluate(snapshot) → U_total   │
 │                                                │
 │  U_total → carrier.update(U_total, cost_fn)   │
 │     → ∇_z U via finite differences             │
 │     → z -= lr · ∇_z U                         │
 │     → new SSGFState(z, W, cost, grad_norm)     │
 │                                                │
 └────────────────────────────��───────────────────┘
```

### Input Contracts

**Constructor:**

| Parameter | Type | Default | Meaning |
|-----------|------|---------|---------|
| `n_oscillators` | `int` | — | Number of oscillators |
| `z_dim` | `int` | 8 | Latent space dimension |
| `lr` | `float` | 0.01 | Gradient descent learning rate |
| `seed` | `int \| None` | `None` | Random seed for initialisation |

**decode:**

| Parameter | Type | Shape | Meaning |
|-----------|------|-------|---------|
| `z` | `NDArray[float64] \| None` | `(d,)` | Latent vector (default: internal) |

**update:**

| Parameter | Type | Default | Meaning |
|-----------|------|---------|---------|
| `cost` | `float` | — | Current SSGF cost value |
| `cost_fn` | `Callable \| None` | `None` | Cost function $W \mapsto U$ |
| `epsilon` | `float` | $10^{-4}$ | Finite difference step |

### Output Contracts

| Method | Returns | Type |
|--------|---------|------|
| `decode(z)` | Coupling matrix | `NDArray[float64]` shape `(N, N)` |
| `update(cost, cost_fn)` | SSGF state snapshot | `SSGFState` |
| `z` (property) | Current latent vector | `NDArray[float64]` shape `(d,)` |

---

## 4. Features

- **Latent space parametrisation** — reduces $N^2$ coupling parameters
  to $d$-dimensional latent vector
- **Softplus activation** — ensures non-negative coupling weights
- **Random projection decoder** — $A \sim \mathcal{N}(0, 1/\sqrt{d})$
  enables rich topology generation
- **Central finite differences** — gradient computation for black-box
  cost functions
- **Autopoietic loop** — geometry produces dynamics, dynamics produce
  cost, cost drives geometry
- **Rust FFI for decode** — native softplus + matrix-vector product
  (note: slower than NumPy BLAS for large $N$ — see §7)
- **SSGFState dataclass** — captures snapshot of outer cycle state
- **Reset mechanism** — re-initialise $z$ with new seed
- **Configurable learning rate** — controls geometry adaptation speed
- **Zero diagonal** — no self-coupling, enforced in both backends

---

## 5. Usage Examples

### Basic: Decode and Inspect

```python
import numpy as np
from scpn_phase_orchestrator.ssgf.carrier import GeometryCarrier

gc = GeometryCarrier(n_oscillators=16, z_dim=8, seed=42)
W = gc.decode()
print(f"W shape: {W.shape}")          # (16, 16)
print(f"Non-negative: {(W >= 0).all()}")  # True
print(f"Diagonal zero: {(np.diag(W) == 0).all()}")  # True
print(f"Mean coupling: {W[W > 0].mean():.4f}")
```

### One SSGF Outer Step

```python
import numpy as np
from scpn_phase_orchestrator.ssgf.carrier import GeometryCarrier

gc = GeometryCarrier(n_oscillators=8, z_dim=8, lr=0.01, seed=42)

# Define a simple cost: sync deficit = 1 - R
def sync_cost(W):
    return float(W.sum())  # Placeholder cost

state = gc.update(cost=1.0, cost_fn=lambda W: sync_cost(W.reshape(8, 8)))
print(f"Step: {state.step}")
print(f"Cost: {state.cost:.4f}")
print(f"Grad norm: {state.grad_norm:.6f}")
print(f"z: {state.z[:3]}...")
```

### Full SSGF Loop

```python
import numpy as np
from scpn_phase_orchestrator.ssgf.carrier import GeometryCarrier
from scpn_phase_orchestrator.ssgf.costs import SSGFCosts
from scpn_phase_orchestrator.ssgf.pgbo import PGBO
from scpn_phase_orchestrator.upde.engine import UPDEEngine

N = 16
gc = GeometryCarrier(N, z_dim=8, lr=0.005, seed=42)
eng = UPDEEngine(N, dt=0.01)
pgbo = PGBO()
costs = SSGFCosts()
rng = np.random.default_rng(42)
phases = rng.uniform(0, 2 * np.pi, N)
omegas = np.ones(N)
alpha = np.zeros((N, N))

for outer in range(20):
    W = gc.decode()
    # Microcycle: integrate phases under current W
    for _ in range(100):
        phases = eng.step(phases, omegas, W, 0.0, 0.0, alpha)
    # Observe and compute cost
    snapshot = pgbo.observe(phases, W)
    U = costs.evaluate(snapshot)

    # Define cost function for gradient
    def cost_fn(W_flat):
        W_tmp = W_flat.reshape(N, N)
        for _ in range(50):
            phases_tmp = eng.step(phases.copy(), omegas, W_tmp, 0.0, 0.0, alpha)
        s = pgbo.observe(phases_tmp, W_tmp)
        return costs.evaluate(s)

    state = gc.update(U, cost_fn=cost_fn)
    print(f"Step {outer}: U={U:.4f}, grad={state.grad_norm:.6f}")
```

### Decode with Custom z

```python
import numpy as np
from scpn_phase_orchestrator.ssgf.carrier import GeometryCarrier

gc = GeometryCarrier(8, z_dim=8, seed=42)

# All zeros → softplus(A·0) = ln(2) uniformly
z_zero = np.zeros(8)
W0 = gc.decode(z_zero)
print(f"Uniform coupling at z=0: {W0[0, 1]:.4f}")  # ≈ 0.693

# Large positive → strong coupling
z_pos = np.ones(8) * 3.0
Wp = gc.decode(z_pos)
print(f"Strong coupling at z=3: {Wp[0, 1]:.4f}")

# Large negative → near-zero coupling
z_neg = -np.ones(8) * 5.0
Wn = gc.decode(z_neg)
print(f"Suppressed coupling at z=-5: {Wn[0, 1]:.6f}")
```

---

## 6. Technical Reference

### Class: GeometryCarrier

::: scpn_phase_orchestrator.ssgf.carrier

### Dataclass: SSGFState

```python
@dataclass
class SSGFState:
    z: NDArray       # Current latent vector, shape (d,)
    W: NDArray       # Decoded coupling matrix, shape (N, N)
    cost: float      # Current SSGF cost U_total
    grad_norm: float # ||∇_z U||₂
    step: int        # Outer step counter
```

### Rust Engine Functions

```rust
// Decode z → W via softplus(A·z), diagonal zeroed
pub fn decode(z: &[f64], a: &[f64], n: usize) -> Vec<f64>

// Numerically stable softplus
fn softplus(x: f64) -> f64

// Finite-difference gradient of cost w.r.t. z
pub fn finite_diff_gradient(
    z: &[f64], a: &[f64], n: usize,
    cost_fn: &dyn Fn(&[f64]) -> f64,
    epsilon: f64,
) -> Vec<f64>
```

### Auto-Select Logic

```python
try:
    from spo_kernel import carrier_decode_rust as _rust_decode
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False
```

Only `decode()` dispatches to Rust. The `update()` method and
gradient computation remain in Python because the cost function
callback cannot cross the FFI boundary.

### Internal State

| Field | Type | Shape | Initialisation |
|-------|------|-------|---------------|
| `_z` | `NDArray[float64]` | `(d,)` | $\mathcal{N}(0, 0.1)$ |
| `_A` | `NDArray[float64]` | `(N^2, d)` | $\mathcal{N}(0, 1/\sqrt{d})$ |
| `_step` | `int` | scalar | 0 |

---

## 7. Performance Benchmarks

Measured on Intel Core i5-11600K @ 3.90 GHz, 32 GB DDR4-2400.
z_dim = 8, median of 200 iterations.

### decode() Latency

| N | Python (µs) | Rust (µs) | Speedup |
|---|-------------|-----------|---------|
| 16 | 7.9 | 9.1 | **0.9x** |
| 32 | 13.0 | 33.8 | **0.4x** |
| 64 | 32.7 | 119.5 | **0.3x** |

### Why Is Rust Slower?

The Python path uses NumPy's `A @ z` (BLAS-accelerated matrix-vector
multiply) followed by vectorised `np.log1p(np.exp(...))`. NumPy
dispatches to optimised BLAS (OpenBLAS or MKL) for the matrix-vector
product, which uses SIMD instructions.

The Rust path uses a naive double loop for the dot product:
```rust
for i in 0..nn {
    for j in 0..z_dim {
        raw += a[i * z_dim + j] * z[j];
    }
}
```

This does not use BLAS or SIMD. For the decode function to benefit
from Rust, it would need:
- A BLAS binding (e.g., `ndarray-linalg` with LAPACK)
- Or manual SIMD vectorisation of the inner loop

**Current recommendation:** The Rust decode path exists for
correctness testing and for environments without NumPy. For
production use, the Python NumPy path is faster.

### update() Latency

The `update()` cost depends on the `cost_fn` callback. With $d = 8$
latent dimensions, it calls `decode()` $2d = 16$ times plus $16$
cost function evaluations. For a typical cost function (100 Kuramoto
steps on $N = 16$ oscillators):

- ~16 × 0.008 ms (decode) + 16 × 1 ms (microcycle) ≈ **16 ms**

The bottleneck is the microcycle, not the decode.

### Memory Usage

| Component | Size | Formula |
|-----------|------|---------|
| Latent $z$ | 64 B | $8d$ bytes |
| Projection $A$ | $8 N^2 d$ B | e.g., 16 KB for $N = 32$, $d = 8$ |
| Decoded $W$ | $8 N^2$ B | e.g., 8 KB for $N = 32$ |

### Test Coverage

- **Rust tests:** 8 (carrier module in spo-engine)
  - Decode zero z, diagonal zero, non-negative, softplus large positive,
    softplus large negative, softplus zero, gradient zero at minimum,
    gradient direction
- **Python tests:** 12 (`tests/test_carrier.py` and pipeline tests)
  - Constructor, decode shape, decode non-negative, decode diagonal,
    update step increment, update cost recorded, reset, z property,
    SSGFState fields, pipeline wiring, softplus values
- **Source lines:** 164 (Rust) + 130 (Python) = 294 total

---

## 8. Citations

1. **Friston, K. J.** (2010).
   "The free-energy principle: A unified brain theory?"
   *Nature Reviews Neuroscience* 11(2):127-138.
   DOI: [10.1038/nrn2787](https://doi.org/10.1038/nrn2787)

2. **Haken, H.** (1983).
   *Synergetics: Introduction and Advanced Topics.*
   Springer. ISBN: 978-3-540-12162-1.

3. **Maturana, H. R. & Varela, F. J.** (1980).
   *Autopoiesis and Cognition: The Realization of the Living.*
   D. Reidel. ISBN: 978-90-277-1016-1.

4. **Breakspear, M.** (2017).
   "Dynamic models of large-scale brain activity."
   *Nature Neuroscience* 20(3):340-352.
   DOI: [10.1038/nn.4497](https://doi.org/10.1038/nn.4497)

5. **Deco, G., Tononi, G., Boly, M., & Kringelbach, M. L.** (2015).
   "Rethinking segregation and integration: Contributions of
   whole-brain modelling."
   *Nature Reviews Neuroscience* 16(7):430-439.
   DOI: [10.1038/nrn3963](https://doi.org/10.1038/nrn3963)

6. **Kingma, D. P. & Welling, M.** (2014).
   "Auto-Encoding Variational Bayes."
   In *Proc. ICLR 2014*. arXiv:1312.6114.

7. **Glorot, X. & Bengio, Y.** (2010).
   "Understanding the difficulty of training deep feedforward neural
   networks."
   In *Proc. AISTATS 2010*. pp. 249-256.
   (Motivation for $A_{ij} \sim \mathcal{N}(0, 1/\sqrt{d})$ initialisation.)

8. **Kuramoto, Y.** (1984).
   *Chemical Oscillations, Waves, and Turbulence.*
   Springer. ISBN: 978-3-642-69691-6.

---

## Edge Cases and Limitations

### z_dim = 1

With a single latent dimension, the geometry is a scalar multiple
of a fixed random pattern. All topological structure is determined
by $A$; the only degree of freedom is the overall coupling scale.

### Large |z| Values

For $|z_i| > 5$, the softplus outputs become either very large
($\sim A \cdot z$) or near-zero ($\sim 0$). This can lead to
very strong or vanishing coupling. The `reset()` method can
re-initialise $z$ to small values.

### Non-Symmetric W

The decoded $W$ is **not** symmetric in general. The random
projection $A$ maps $z$ to all $N^2$ entries independently.
If symmetric coupling is required, symmetrise post-decode:
$W \leftarrow (W + W^T) / 2$.

### Cost Function Discontinuities

The finite-difference gradient assumes the cost function is smooth.
If the cost function has discontinuities (e.g., from regime
switches), the gradient estimate may be noisy or biased. In this
case, reduce $\varepsilon$ or use a smoothed cost function.

---

## Troubleshooting

### Issue: Gradient Norm is Zero Despite Non-Zero Cost

**Diagnosis:** If `cost_fn` is `None`, the `update()` method skips
gradient computation and returns `grad_norm = 0`. Always pass a
cost function for gradient-based optimisation.

### Issue: W Entries Grow Very Large

**Diagnosis:** The learning rate is too high, driving $z$ toward
large positive values. Softplus becomes linear for large inputs,
so coupling grows without bound.

**Solution:** Reduce `lr` or add L2 regularisation to $z$:
$U_{\text{reg}} = U_{\text{total}} + \lambda ||z||^2$.

### Issue: Decode is Slow for Large N

**Diagnosis:** For $N \geq 32$, the Rust decode path is slower
than Python NumPy due to lack of BLAS optimisation.

**Solution:** Ensure `_HAS_RUST = False` for this module, or
keep using the default auto-select (Python will be used when NumPy
is available). Future work: integrate BLAS bindings in Rust.

---

## Integration with Other SPO Modules

### With SSGFCosts

The `SSGFCosts` evaluator computes $U_{\text{total}}$ from a PGBO
snapshot. The carrier's `cost_fn` callback wraps the full pipeline:

```python
def cost_fn(W_flat):
    W = W_flat.reshape(N, N)
    # Run microcycles under W
    for _ in range(100):
        phases = eng.step(phases, omegas, W, 0.0, 0.0, alpha)
    snapshot = pgbo.observe(phases, W)
    return costs.evaluate(snapshot)
```

### With PGBO

The PGBO curvature metric $K_g$ provides a geometry-specific cost
signal. Low $K_g$ indicates the geometry is misaligned with the phase
dynamics — the carrier should increase its learning rate.

### With RegimeManager

The `SSGFState.grad_norm` serves as a convergence indicator:
- `grad_norm > 0.1` → geometry is still evolving rapidly
- `grad_norm < 0.01` → geometry has converged to a local minimum
- `grad_norm ≈ 0` → either converged or `cost_fn` is `None`

The `RegimeManager` can use this signal to decide whether to
continue geometry optimisation or freeze the topology.

### With OttAntonsenReduction

The OA reduction predicts the theoretical $R_{ss}$ for a given
effective coupling. The carrier can use this to set a target:

$$U_{\text{target}} = 1 - R_{ss}(K_{\text{eff}})$$

where $K_{\text{eff}} = \frac{1}{N} \sum_{ij} W_{ij}$.
