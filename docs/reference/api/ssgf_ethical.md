# C15_sec diagnostic cost

`compute_ethical_cost` reports a weighted synchronization score, three squared
constraint residuals, their combined cost and a violation count. These numerical
quantities do not establish ethical compliance, safety, forward invariance or a
clinical decision. Callers choose their own weights and thresholds.

## Numerical contract

For a phase vector of length $N$ and a matching matrix $W$:

$$J_{\mathrm{sec}} = \alpha_R R + \beta_K \frac{\lambda_2}{N}
                    + \gamma_Q Q - \nu_S S_{\mathrm{dev}},$$

$$\texttt{phi\_ethics} = \kappa\sum_{k=1}^{3}\max(0,g_k)^2,
\qquad C_{15}=1-J_{\mathrm{sec}}+\texttt{phi\_ethics}.$$

The multiplier $\kappa$ is applied **once**. The returned `phi_ethics` already
includes it; callers must not multiply this field again.

| Quantity | Implemented definition |
|---|---|
| $R$ | $|\sum_j e^{i\theta_j}|/N$ |
| Adjacency | $A_{ij}=(|W_{ij}|+|W_{ji}|)/2$ for $i\ne j$, $A_{ii}=0$ |
| Laplacian | $L=\operatorname{diag}(A\mathbf{1})-A$ |
| $\lambda_2$ | Second-smallest Laplacian eigenvalue, zero for $N<2$ |
| $Q$ | Exactly nonzero **matrix entries**, divided by $N(N-1)$; zero for $N<2$ |
| $S_{\mathrm{dev}}$ | Population standard deviation of **raw** phases, divided by $\pi$ |
| $g_1$ | `R_min` minus $R$ |
| $g_2$ | `connectivity_min` minus $\lambda_2$ |
| $g_3$ | Largest matrix entry minus `max_coupling` if any entry is positive; otherwise zero |

Signed and asymmetric weights are accepted. Self-loops count in $Q$ and can
contribute to $g_3$, but do not enter the Laplacian. Arbitrarily small nonzero
entries count in density; there is no edge cutoff. Consequently $Q$ can exceed
one when diagonal entries are present. Normalized connectivity and the SEC score
are not constrained to $[0,1]$.

Phases need not lie in $[0,2\pi)$. The dispersion term is linear population
standard deviation, not circular variance: `[0, 2*pi]` has $S_{\mathrm{dev}}=1$
even though both phases represent the same direction.

The violation count is the number of **strictly positive** residuals. Equality
is satisfied, and a zero or negative multiplier does not change this count.
Signed finite weights and thresholds remain supported; negative `kappa` can make
`phi_ethics` negative, and a negative total is a numerical outcome rather than a
statement about compliance.

Empty `(0,)` phases and `(0,0)` coupling return `(0,0,1,0)` after validating all
scalar parameters. A single node has zero connectivity, density and dispersion
but still contributes its coherence score.

## Inputs and refusal

| Parameter | Default |
|---|---:|
| `alpha_R` | 0.4 |
| `beta_K` | 0.3 |
| `gamma_Q` | 0.2 |
| `nu_S` | 0.1 |
| `kappa` | 1.0 |
| `R_min` | 0.2 |
| `connectivity_min` | 0.1 |
| `max_coupling` | 5.0 |

The public function requires finite real phase and coupling measurements, a
one-dimensional phase vector, and a matching square matrix. Boolean, text,
complex and temporal aliases are refused before float conversion. Plain real
numeric object arrays are supported. Scalar parameters must be representable
finite non-boolean reals; even an empty input does not bypass their validation.

The function leaves input storage unchanged. Strided, read-only and unaligned
numeric views are adapted to aligned contiguous float64 buffers for FFI.
Finite inputs can still overflow graph, dispersion, residual-square or final
cost arithmetic; the operation then raises `ValueError` rather than returning
NaN or infinity. The Python adapter converts floating-point and eigensolver
failures to this same public refusal.

## Actual implementations

The default function uses `spo_kernel.compute_ethical_cost_rust` when its
original compiled callable is available. Otherwise it computes the score in
Python using the public order-parameter and spectral primitives. There is no
per-call ethical-cost backend selector. A base installation without the optional
kernel provides the genuine Python path; changing `_HAS_RUST` does not select a
backend and is not a supported runtime control.

The Rust core builds the reciprocal-magnitude Laplacian and uses largest-pivot
Jacobi rotations. Its working matrix is scaled before applying a relative
`tolerance = 8 * f64::EPSILON * N`; eigenvalues are restored to the original scale.
This preserves tiny graph connectivity instead of terminating at an absolute
`1e-12` edge magnitude. The dense search costs $O(N^2)$ per rotation.

```rust
pub fn compute_ethical_cost(
    phases: &[f64], knm: &[f64], n: usize,
    alpha_r: f64, beta_k: f64, gamma_q: f64, nu_s: f64,
    kappa: f64, r_min: f64, connectivity_min: f64, max_coupling: f64,
) -> Result<(f64, f64, f64, usize), String>
```

The Python FFI retains the same eleven arguments. Its arrays must be contiguous,
aligned one-dimensional float64 buffers. It rejects alias parameter types and
requires genuine nonnegative integer dimension metadata. Shape, finite-value
and arithmetic errors from the public Rust core become `ValueError`.

## Usage and real composition

```python
import numpy as np
from scpn_phase_orchestrator.ssgf.ethical import compute_ethical_cost

phases = np.zeros(2)
weights = np.array([[0.0, 0.5], [0.5, 0.0]])
result = compute_ethical_cost(phases, weights)
# R=1, lambda2=1, Q=1, dispersion=0:
assert np.isclose(result.J_sec, 0.75)
assert np.isclose(result.c15_sec, 0.25)
assert result.constraints_violated == 0
```

`compute_ssgf_costs` and `CyberneticClosure` use the existing four-term SSGF
objective. They do not automatically add a `w_c15` term. A caller can compose
that objective with this diagnostic explicitly:

```python
from scpn_phase_orchestrator.ssgf.costs import compute_ssgf_costs

base = compute_ssgf_costs(weights, phases).u_total
combined = base + 0.2 * compute_ethical_cost(phases, weights).c15_sec
```

A `GeometryCarrier` accepts a callback receiving a **matrix**, not a flattened
buffer. Freeze the observed phase state when evaluating its finite-difference
objective; do not advance a mutable engine inside repeated callback evaluations.

```python
from scpn_phase_orchestrator.ssgf.carrier import GeometryCarrier

carrier = GeometryCarrier(2, z_dim=3, lr=0.01, seed=42)
observed_phases = phases.copy()

def diagnostic_objective(candidate_weights):
    return compute_ethical_cost(observed_phases, candidate_weights).c15_sec

initial = diagnostic_objective(carrier.decode())
state = carrier.update(initial, cost_fn=diagnostic_objective)
```

The executed consumer chain is closure-produced coupling → `UPDEEngine.step`
→ observed phases → public cost. Regime transitions, active-inference policies
and actuator decisions are not automatically wired to this diagnostic.

## Current local measurements

The [recorded comparison](../data/ethical_cost_comparison.local.json) contains
source and native-binary SHA-256, actual numerical outputs, raw batch means,
separately observed individual-call latencies and execution metadata. Both
original installed owners match the independent scalar/LAPACK comparison oracle.
Interpreter startup is excluded; public validation and FFI are included.

The seed-42 fixture uses phases uniformly in `[0, 2*pi)`, coupling uniformly in
`[0, 0.5)` and a zero diagonal, with default parameters. Each owner and size has
20 batches of 100 calls plus 2,000 individually timed calls. Batch means and
call percentiles describe different sample populations.

| N | Python median batch mean (µs) | Rust median batch mean (µs) | Python / Rust |
|---:|---:|---:|---:|
| 8 | 384.75 | 86.72 | 4.44 |
| 16 | 403.75 | 168.17 | 2.40 |
| 32 | 546.75 | 1550.96 | 0.35 |

These are local observations from 2026-10-08, not universal speed guarantees or
an automatic runtime-selection policy. The previous September timing table used
different code and execution conditions and is superseded by this record.

```bash
python -m benchmarks.ethical_cost_benchmark --sizes 8 16 32 \
  --calls 100 --batches 20 \
  --python-profile /path/to/kernel-absent/bin/python \
  --rust-profile /path/to/native-installed/bin/python
```

The original native Criterion ring fixture was also rerun without changing its
sizes, weights or score parameters. It retained all 20 samples at each size.
Median per-iteration estimates were 86.58 µs at 16 nodes, 21.58 ms at 64 nodes,
and 6.29 s at 256 nodes. This different workload excludes Python/FFI overhead and
must not be compared directly with the dense public fixture. The large-node
cost is a concrete limit of the current Jacobi solver.

## API

::: scpn_phase_orchestrator.ssgf.ethical
