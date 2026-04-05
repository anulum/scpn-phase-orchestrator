# C15_sec Ethical Cost Term

## 1. Mathematical Formalism

### The Ethical Lagrangian

The SCPN framework augments the standard SSGF cost with an ethical
constraint term from Layer 15 (the ethical governance layer):

$$\mathcal{L}_{\text{ethical}} = U_{\text{total}} + w_{c15} \cdot C_{15,\text{sec}}$$

where $C_{15,\text{sec}}$ is the **Social Ethical Cost** — a scalar
measuring the ethical compliance of the synchronisation dynamics.

### C15_sec Decomposition

$$C_{15,\text{sec}} = (1 - J_{\text{sec}}) + \kappa \cdot \Phi_{\text{ethics}}$$

The cost has two terms:
1. **(1 - J_sec):** How far from ethically optimal the system is
2. **κ · Φ_ethics:** Penalty for violating hard safety constraints

### SEC Functional (Social Ethical Compliance)

$$J_{\text{sec}} = \alpha \cdot R + \beta \cdot K_{\text{norm}} + \gamma \cdot Q - \nu \cdot S_{\text{dev}}$$

where:

| Symbol | Definition | Meaning |
|--------|-----------|---------|
| $R$ | $\frac{1}{N}\|\sum_j e^{i\theta_j}\|$ | Kuramoto order parameter (coherence) |
| $K_{\text{norm}}$ | $\lambda_2(L) / N$ | Normalised algebraic connectivity |
| $Q$ | $\frac{\text{nnz}(W)}{N(N-1)}$ | Coupling density (quality) |
| $S_{\text{dev}}$ | $\text{std}(\theta) / \pi$ | Phase deviation from uniformity |
| $\alpha, \beta, \gamma, \nu$ | weights | Default: 0.4, 0.3, 0.2, 0.1 |

$J_{\text{sec}} \in [0, 1]$ approximately. Higher = more ethically
compliant. The four terms balance:
- **Coherence** ($\alpha R$): System should be synchronised (collective benefit)
- **Connectivity** ($\beta K_{\text{norm}}$): Network should be globally
  integrated (Wiener cybernetic ethics — no isolated subgroups)
- **Quality** ($\gamma Q$): Coupling should be dense enough to function
- **Equity** ($-\nu S_{\text{dev}}$): Phase deviation penalises systems
  where some oscillators are left behind

### CBF Constraint Penalties (Control Barrier Functions)

$$\Phi_{\text{ethics}} = \sum_{k=1}^{3} \max(0, g_k)^2$$

where the constraint functions $g_k$ encode hard safety boundaries:

| Constraint | $g_k$ | Meaning |
|-----------|-------|---------|
| Non-harm | $R_{\min} - R$ | Minimum coherence (system must function) |
| Connectivity | $\lambda_{2,\min} - \lambda_2$ | Network must not fragment |
| Coupling limit | $\max(K_{ij}) - K_{\max}$ | Coupling must not exceed safety bound |

The quadratic penalty $\max(0, g_k)^2$ is zero when the constraint
is satisfied and grows quadratically with the violation magnitude.
This is a relaxed CBF (Control Barrier Function) formulation.

### Fiedler Value (Algebraic Connectivity)

The algebraic connectivity $\lambda_2$ is the second-smallest
eigenvalue of the graph Laplacian:

$$L = D - |W|$$

where $D = \text{diag}(\sum_j |W_{ij}|)$ is the degree matrix.

$\lambda_2 > 0$ if and only if the graph is connected. A larger
$\lambda_2$ indicates stronger global integration — harder to
partition the network into disconnected components.

### Jacobi Eigenvalue Algorithm (Rust)

The Rust implementation computes eigenvalues of the symmetric
Laplacian using the classical Jacobi rotation method:

1. Find the largest off-diagonal element $|L_{pq}|$
2. Apply a Givens rotation to zero out $L_{pq}$
3. Repeat until all off-diagonal elements are below $\varepsilon = 10^{-12}$
4. Diagonal entries are the eigenvalues

The algorithm converges quadratically for well-separated eigenvalues
and has complexity $O(N^2)$ per rotation with $O(N^2)$ rotations
in the worst case, giving $O(N^4)$ total for dense matrices.

---

## 2. Theoretical Context

### Ethical Governance in Autonomous Systems

The C15_sec term operationalises ethical principles as mathematical
constraints within the synchronisation dynamics. This approach
draws from four traditions:

### Harsanyi Aggregation (1955)

The SEC functional $J_{\text{sec}}$ is a weighted sum of individual
metrics, analogous to Harsanyi's utilitarian aggregation theorem:
the social welfare function is a weighted sum of individual utilities
under certain rationality axioms. The weights $(\alpha, \beta, \gamma, \nu)$
encode the relative importance of different ethical principles.

### MacAskill's Effective Altruism (2022)

The **equity term** $-\nu S_{\text{dev}}$ reflects the principle
that a system which benefits some oscillators at the expense of
others is less ethically compliant. This operationalises the
concern for "leaving no one behind" in resource allocation.

### Lyapunov/CBF Safety (Ames et al. 2017)

Control Barrier Functions (CBFs) provide forward invariance
guarantees: if the system starts in a safe set, it remains there.
The constraint functions $g_k$ define the safe set boundary. The
quadratic penalty is a relaxation — it does not guarantee hard
invariance but provides a continuous optimisation signal.

### Wiener Cybernetic Ethics (1950)

Norbert Wiener's *The Human Use of Human Beings* argued that
communication systems should maintain connectivity and resist
fragmentation. The $\beta K_{\text{norm}}$ term directly implements
this: penalising network topologies that allow subgroups to become
disconnected.

### Historical Context

- **Harsanyi, J. C.** (1955): "Cardinal welfare, individualistic
  ethics, and interpersonal comparisons of utility." Weighted
  utilitarian aggregation theorem.
- **Wiener, N.** (1950): *The Human Use of Human Beings: Cybernetics
  and Society.* Ethical implications of communication and control.
- **Ames, A. D. et al.** (2017): "Control barrier function based
  quadratic programs for safety critical systems." CBF framework.
- **MacAskill, W.** (2022): *What We Owe the Future.* Long-termist
  ethical framework for AI systems.
- **Floridi, L.** (2013): *The Ethics of Information.* Information
  ethics as a foundation for AI governance.

### Layer 15 in the SCPN Stack

In the 15+1 layer SCPN architecture, Layer 15 is the **Ethical
Governance Layer**. It receives telemetry from all lower layers
and applies ethical constraints before the Layer 16 Director
makes high-level decisions. The C15_sec cost is the quantitative
output of Layer 15.

---

## 3. Pipeline Position

```
 UPDEEngine.step() ──→ phases
 CouplingBuilder/SSGF ──→ W (knm)
                    │        │
                    ↓        ↓
 ┌── compute_ethical_cost(phases, knm, ...) ──────┐
 │                                                 │
 │  Step 1: Compute SEC inputs                    │
 │    R = order parameter                          │
 │    λ₂ = fiedler_value(L(W))                    │
 │    Q = coupling density                         │
 │    S_dev = phase deviation                      │
 │                                                 │
 │  Step 2: Compute J_sec (weighted sum)          │
 │                                                 │
 │  Step 3: Compute CBF penalties (Φ_ethics)      │
 │    g₁: R_min - R                                │
 │    g₂: connectivity_min - λ₂                   │
 │    g₃: max(K) - max_coupling                    │
 │                                                 │
 │  Step 4: Assemble C15_sec = (1-J) + κΦ         │
 │                                                 │
 │  Output: EthicalCost(J_sec, Φ, C15, n_violated)│
 └─────────────────────────────────────────────────┘
                    │
                    ↓
         SSGFCosts adds w_c15 · C15_sec to U_total
                    │
                    ↓
         GeometryCarrier minimises L_ethical
```

### Input Contracts

| Parameter | Type | Default | Range | Meaning |
|-----------|------|---------|-------|---------|
| `phases` | `NDArray[float64]` | — | $[0, 2\pi)$ | Current phases |
| `knm` | `NDArray[float64]` | — | $\geq 0$ | Coupling matrix |
| `alpha_R` | `float` | 0.4 | $[0, 1]$ | Coherence weight |
| `beta_K` | `float` | 0.3 | $[0, 1]$ | Connectivity weight |
| `gamma_Q` | `float` | 0.2 | $[0, 1]$ | Quality weight |
| `nu_S` | `float` | 0.1 | $[0, 1]$ | Equity weight |
| `kappa` | `float` | 1.0 | $> 0$ | CBF penalty multiplier |
| `R_min` | `float` | 0.2 | $[0, 1]$ | Min. coherence (CBF) |
| `connectivity_min` | `float` | 0.1 | $\geq 0$ | Min. $\lambda_2$ (CBF) |
| `max_coupling` | `float` | 5.0 | $> 0$ | Max. coupling (CBF) |

### Output Contract

```python
@dataclass
class EthicalCost:
    J_sec: float             # SEC functional, ∈ [0, 1] approx.
    phi_ethics: float        # CBF penalty, ≥ 0
    c15_sec: float           # Total ethical cost
    constraints_violated: int # Number of violated constraints (0-3)
```

---

## 4. Features

- **Weighted SEC functional** — multi-objective ethical compliance
  (coherence, connectivity, quality, equity)
- **CBF constraint penalties** — hard safety boundaries with
  quadratic relaxation
- **Three safety constraints** — non-harm (minimum R), connectivity
  (minimum $\lambda_2$), coupling limit
- **Algebraic connectivity** — Fiedler value via Jacobi eigenvalues (Rust)
  or NumPy eigvalsh (Python)
- **Violation counting** — reports how many constraints are violated
- **Decomposable cost** — $C_{15}$ = deficit + penalties, separable
  for diagnosis
- **Rust FFI acceleration** — 5.7x faster for small N (N ≤ 8)
- **Configurable weights** — all 4 SEC weights and 3 CBF thresholds
  adjustable
- **Pipeline composable** — feeds into SSGFCosts as an additive term

---

## 5. Usage Examples

### Basic: Compute Ethical Cost

```python
import numpy as np
from scpn_phase_orchestrator.ssgf.ethical import compute_ethical_cost

N = 8
rng = np.random.default_rng(42)
phases = rng.uniform(0, 2 * np.pi, N)
knm = np.full((N, N), 0.5)
np.fill_diagonal(knm, 0.0)

result = compute_ethical_cost(phases, knm)
print(f"J_sec = {result.J_sec:.4f}")
print(f"Φ_ethics = {result.phi_ethics:.4f}")
print(f"C15_sec = {result.c15_sec:.4f}")
print(f"Violations: {result.constraints_violated}")
```

### Synchronised vs Desynchronised

```python
import numpy as np
from scpn_phase_orchestrator.ssgf.ethical import compute_ethical_cost

N = 8
knm = np.full((N, N), 0.5); np.fill_diagonal(knm, 0.0)

# Synchronised: all same phase
sync_phases = np.ones(N) * 1.0
sync_cost = compute_ethical_cost(sync_phases, knm)

# Desynchronised: uniform spread
desync_phases = np.linspace(0, 2 * np.pi, N, endpoint=False)
desync_cost = compute_ethical_cost(desync_phases, knm)

print(f"Sync: J={sync_cost.J_sec:.4f}, C15={sync_cost.c15_sec:.4f}")
print(f"Desync: J={desync_cost.J_sec:.4f}, C15={desync_cost.c15_sec:.4f}")
# Sync should have lower C15 (more ethical)
```

### Custom Safety Thresholds

```python
import numpy as np
from scpn_phase_orchestrator.ssgf.ethical import compute_ethical_cost

N = 8
phases = np.ones(N) * 0.5  # synchronised
knm = np.full((N, N), 0.3); np.fill_diagonal(knm, 0.0)

# Strict safety: high R_min, high connectivity_min
strict = compute_ethical_cost(
    phases, knm,
    R_min=0.8,
    connectivity_min=0.5,
    max_coupling=1.0,
)

# Relaxed safety: low thresholds
relaxed = compute_ethical_cost(
    phases, knm,
    R_min=0.1,
    connectivity_min=0.01,
    max_coupling=10.0,
)

print(f"Strict: C15={strict.c15_sec:.4f}, violations={strict.constraints_violated}")
print(f"Relaxed: C15={relaxed.c15_sec:.4f}, violations={relaxed.constraints_violated}")
```

### Integration with SSGF Loop

```python
import numpy as np
from scpn_phase_orchestrator.ssgf.carrier import GeometryCarrier
from scpn_phase_orchestrator.ssgf.ethical import compute_ethical_cost
from scpn_phase_orchestrator.upde.engine import UPDEEngine

N = 8
gc = GeometryCarrier(N, z_dim=8, lr=0.005, seed=42)
eng = UPDEEngine(N, dt=0.01)
rng = np.random.default_rng(42)
phases = rng.uniform(0, 2 * np.pi, N)
omegas = np.ones(N)
alpha = np.zeros((N, N))

for outer in range(10):
    W = gc.decode()
    for _ in range(100):
        phases = eng.step(phases, omegas, W, 0.0, 0.0, alpha)

    ethical = compute_ethical_cost(phases, W)

    def cost_fn(W_flat):
        W_tmp = W_flat.reshape(N, N)
        for _ in range(50):
            p = eng.step(phases.copy(), omegas, W_tmp, 0.0, 0.0, alpha)
        return compute_ethical_cost(p, W_tmp).c15_sec

    state = gc.update(ethical.c15_sec, cost_fn=cost_fn)
    print(f"Step {outer}: C15={ethical.c15_sec:.4f}, violations={ethical.constraints_violated}")
```

---

## 6. Technical Reference

### Function: compute_ethical_cost

::: scpn_phase_orchestrator.ssgf.ethical

### Dataclass: EthicalCost

```python
@dataclass
class EthicalCost:
    J_sec: float             # SEC functional value
    phi_ethics: float        # CBF penalty sum
    c15_sec: float           # Total: (1 - J_sec) + κ·Φ
    constraints_violated: int # Count of g_k > 0
```

### Rust Engine Functions

```rust
pub fn compute_ethical_cost(
    phases: &[f64], knm: &[f64], n: usize,
    alpha_r: f64, beta_k: f64, gamma_q: f64, nu_s: f64, kappa: f64,
    r_min: f64, connectivity_min: f64, max_coupling: f64,
) -> (f64, f64, f64, usize)  // (j_sec, phi_ethics, c15_sec, n_violated)
```

Internal helpers:
- `compute_sec_inputs` — R, λ₂, Q, S_dev
- `compute_cbf_penalties` — penalty sum and violation count
- `fiedler_value_inline` — Laplacian construction + Jacobi eigenvalues
- `jacobi_eigenvalues` — iterative eigenvalue solver
- `find_max_offdiag` — largest off-diagonal element
- `jacobi_rotate` — single Givens rotation

### Auto-Select Logic

```python
try:
    from spo_kernel import compute_ethical_cost_rust as _rust_ethical_cost
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False
```

### Python Eigenvalue Path

The Python path uses `fiedler_value()` from `coupling.spectral`,
which calls `numpy.linalg.eigvalsh` — a LAPACK wrapper using the
symmetric divide-and-conquer algorithm ($O(N^3)$ but with very
small constant factor due to BLAS optimisation).

---

## 7. Performance Benchmarks

Measured on Intel Core i5-11600K @ 3.90 GHz, 32 GB DDR4-2400.
Random phases and coupling, median of 50-100 iterations.

| N | Python (µs) | Rust (µs) | Speedup |
|---|-------------|-----------|---------|
| 8 | 82.6 | 14.5 | **5.7x** |
| 16 | 124.4 | 94.8 | **1.3x** |
| 32 | 735.2 | 1711.5 | **0.4x** |

### Why Does Rust Slow Down at Large N?

The bottleneck is the Fiedler value computation:
- **Python:** `numpy.linalg.eigvalsh` → LAPACK dsyevd, $O(N^3)$
  with highly optimised BLAS (SIMD, cache-aware)
- **Rust:** Jacobi rotation, $O(N^4)$ worst case with naive loops

For N=8, the Jacobi overhead is small and Rust wins via reduced
Python overhead. For N=32, the $O(N^4)$ Jacobi dominates.

**Recommendation:** Use the Rust path for $N \leq 16$ and the
Python path for $N > 16$. Future work: integrate LAPACK bindings
(ndarray-linalg) in Rust for $O(N^3)$ eigenvalue computation.

### Cost Breakdown (N=16)

| Component | Python (µs) | Rust (µs) |
|-----------|-------------|-----------|
| Order parameter R | ~5 | ~2 |
| Fiedler value λ₂ | ~100 | ~80 |
| Density Q | ~1 | ~0.5 |
| Phase deviation S_dev | ~2 | ~1 |
| CBF penalties | ~1 | ~0.5 |
| **Total** | **~124** | **~95** |

### Memory Usage

- Laplacian: $N^2$ floats (temporary, 8 KB for N=32)
- Working copy for Jacobi: $N^2$ floats
- Output: 4 scalars

### Test Coverage

- **Rust tests:** 7 (ethical module in spo-engine)
  - Empty phases, synchronised high coupling, no coupling violation,
    high coupling violation, C15 decomposition, Fiedler complete
    graph, Fiedler disconnected
- **Python tests:** 13 (`tests/test_closure_ethical.py`)
  - Shape/type, synchronised lower cost, constraints detected,
    decomposition identity, coupling limit, kappa scaling, weight
    sensitivity, pipeline wiring, edge cases
- **Source lines:** 277 (Rust) + 125 (Python) = 402 total

---

## 8. Citations

1. **Harsanyi, J. C.** (1955).
   "Cardinal welfare, individualistic ethics, and interpersonal
   comparisons of utility."
   *Journal of Political Economy* 63(4):309-321.
   DOI: [10.1086/257678](https://doi.org/10.1086/257678)

2. **Wiener, N.** (1950).
   *The Human Use of Human Beings: Cybernetics and Society.*
   Houghton Mifflin. ISBN: 978-0-306-80320-8.

3. **Ames, A. D., Xu, X., Grizzle, J. W., & Tabuada, P.** (2017).
   "Control barrier function based quadratic programs for safety
   critical systems."
   *IEEE Transactions on Automatic Control* 62(8):3861-3876.
   DOI: [10.1109/TAC.2016.2638961](https://doi.org/10.1109/TAC.2016.2638961)

4. **MacAskill, W.** (2022).
   *What We Owe the Future.*
   Basic Books. ISBN: 978-1-5416-1862-6.

5. **Floridi, L.** (2013).
   *The Ethics of Information.*
   Oxford University Press. ISBN: 978-0-19-964132-1.

6. **Fiedler, M.** (1973).
   "Algebraic connectivity of graphs."
   *Czechoslovak Mathematical Journal* 23(98):298-305.

7. **Jacobi, C. G. J.** (1846).
   "Über ein leichtes Verfahren die in der Theorie der
   Säcularstörungen vorkommenden Gleichungen numerisch aufzulösen."
   *Journal für die reine und angewandte Mathematik* 30:51-94.

8. **Russell, S.** (2019).
   *Human Compatible: Artificial Intelligence and the Problem of
   Control.*
   Viking. ISBN: 978-0-525-55861-3.

---

## Edge Cases and Limitations

### Empty Phases (N = 0)

Returns $J_{\text{sec}} = 0$, $\Phi = 0$, $C_{15} = 1.0$ (maximum
cost), 0 violations.

### Zero Coupling Matrix

When $W = 0$:
- $\lambda_2 = 0$ (disconnected graph)
- $Q = 0$ (no connections)
- Connectivity constraint is violated ($g_2 = \lambda_{2,\min} > 0$)
- $J_{\text{sec}}$ is low (only $\alpha R$ contributes)

### Perfect Synchronisation (R = 1)

With $R = 1$, all phases are identical:
- $S_{\text{dev}} = 0$ (no deviation)
- $J_{\text{sec}} = \alpha + \beta K_{\text{norm}} + \gamma Q$
  (maximum without equity penalty)
- Non-harm constraint satisfied (R ≥ R_min)

### All Weights Zero

If $\alpha = \beta = \gamma = \nu = 0$:
- $J_{\text{sec}} = 0$
- $C_{15} = 1 + \kappa \Phi$ (only CBF penalties remain)

---

## Troubleshooting

### Issue: C15_sec is Negative

**Diagnosis:** $J_{\text{sec}} > 1$ is possible when R, K_norm, and
Q are all high. Then $1 - J_{\text{sec}} < 0$ and $C_{15}$ can be
negative if $\Phi = 0$.

**Solution:** This indicates excellent ethical compliance — all metrics
are high and no constraints violated. Negative $C_{15}$ is valid.

### Issue: Constraints Always Violated

**Diagnosis:** The thresholds $R_{\min}$, $\lambda_{2,\min}$, or
$K_{\max}$ may be set too strictly for the current dynamical regime.

**Solution:** Reduce thresholds or increase coupling strength.

### Issue: Fiedler Value Very Slow for Large N

**Diagnosis:** The Rust Jacobi algorithm is $O(N^4)$.

**Solution:** For $N > 16$, ensure the Python path is used
(`_HAS_RUST = False` for this module). The Python path uses
LAPACK eigvalsh at $O(N^3)$.

---

## Integration with Other SPO Modules

### With SSGFCosts

The ethical cost integrates into the SSGF cost functional as an
additive term:

$$U_{\text{total}} = w_1 C_1 + w_2 C_2 + w_3 C_3 + w_4 C_4 + w_{c15} C_{15,\text{sec}}$$

The weight $w_{c15}$ controls the relative importance of ethical
compliance versus synchronisation efficiency.

### With GeometryCarrier

The geometry carrier can minimise the ethical cost directly:

```python
def ethical_cost_fn(W_flat):
    W = W_flat.reshape(N, N)
    for _ in range(100):
        phases = eng.step(phases, omegas, W, 0.0, 0.0, alpha)
    return compute_ethical_cost(phases, W).c15_sec

state = carrier.update(current_cost, cost_fn=ethical_cost_fn)
```

### With RegimeManager

The `constraints_violated` count provides a direct signal for
regime transitions:
- 0 violations: NOMINAL regime
- 1 violation: DEGRADED — supervisor should act
- 2-3 violations: CRITICAL — immediate intervention required

### With ActiveInferenceAgent

The ethical cost serves as the variational free energy for the
active inference agent's ethical beliefs. The agent's policy
selection minimises expected $C_{15,\text{sec}}$ over future states.
