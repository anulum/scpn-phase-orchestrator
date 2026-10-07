# Attention-inspired phase coupling

`coupling.attention_residuals` implements SPO's state-dependent coupling law:
Fourier phase features, multi-head query/key/value attention over existing graph
neighbours, an output projection, and a regularised cosine readout. The public
`attnres_modulate` name remains for compatibility.

The [Attention Residuals paper](https://arxiv.org/pdf/2603.15031), section 3 and
figure 2, instead aggregates preceding network-layer outputs with a learned
pseudo-query and RMS-normalised keys. SPO's spatial oscillator attention is an
adaptation of the broader attention idea. It does not implement that depth
operator, a complete Transformer decoder, a training loop, or the paper's
Block AttnRes. Here `block_size` is an oscillator index-band radius.

## Coupling law

Input `K` is an undirected, finite real `(N, N)` coupling matrix with zero
diagonal; `theta` is a finite real phase vector in radians. Signed edges are
supported. The matrix uses target-row/source-column indexing. Coupling retains
its input units; attention scores and modulation strength are dimensionless.
Seeded feature projections depend on the absolute phase origin. Integral full
turns preserve periodicity; a common arbitrary phase shift need not preserve
the modulated graph.

For even feature width `D`, define

\[
 x_i=[\cos\theta_i,\sin\theta_i,\ldots,\cos(D\theta_i/2),\sin(D\theta_i/2)].
\]

Each of `H` heads has width `d_h = D/H`:

\[
 Q_h=XW_h^Q,\quad S_h=XW_h^K,\quad V_h=XW_h^V.
\]

Allowed neighbours exclude the diagonal, exactly zero input edges, and pairs
outside the optional index band. For an allowed pair, the head weight is

\[
 A_{h,ij}=\frac{\exp(\ell_{h,ij}-m_{h,i})}
 {\sum_{j'\in M_i}\exp(\ell_{h,ij'}-m_{h,i})},
 \quad\ell_{h,ij}=\frac{Q_{h,i}\cdot S_{h,j}}{\sqrt{d_h}\,\tau}.
\]

`m` is the largest allowed row logit. Masked entries and rows without neighbours
have zero weight. Concatenated head outputs are projected into `O`:

\[
 O=[A_1V_1\|\cdots\|A_HV_H]W^O,
 \quad a_{ij}=\tfrac12\left(1+
 \frac{O_i}{\|O_i\|+10^{-12}}\cdot
 \frac{O_j}{\|O_j\|+10^{-12}}\right)\mathbf1[(i,j)\in M].
\]

The symmetric result is

\[
 K'_{ij}=\tfrac12K_{ij}(1+\lambda a_{ij})+
          \tfrac12K_{ji}(1+\lambda a_{ji}),\qquad K'_{ii}=0.
\]

Non-negative `lambda_` boosts edge magnitudes while preserving signs and absent
edges. The computed cosine is clipped to `[-1,1]` to keep rounding at parallel
and antiparallel endpoints within the mathematical score range. This matters
when a large gain amplifies tiny rounding errors. It provides no down-weighting.
Out-of-band edges retain the input value
for exactly symmetric inputs. Floating-point topology comparisons use
`rtol=atol=1e-12` for symmetry and `atol=1e-12` for zero-diagonal/output zero-edge
validation. Inputs near those tolerances require particular care at large gains.
The edge mask uses exact input zeros. A weak nonzero edge such as `9e-13`
remains an edge and may grow beyond the output-zero tolerance under modulation.

`lambda_=0` returns an independent, bit-equal input copy after public input and
projection validation. The public empty-system result is an empty matrix after the same projection
validation; optional runtimes are unnecessary for these identity paths.

## Public API

```python
import numpy as np
from scpn_phase_orchestrator.coupling.attention_residuals import (
    attnres_modulate, default_projections,
)

K = np.array([[0.0, 0.3], [0.3, 0.0]])
phases = np.array([0.0, 0.7])
weights = default_projections(n_heads=2, d_model=4, seed=42)
K_mod = attnres_modulate(
    K, phases,
    w_q=weights[0], w_k=weights[1], w_v=weights[2], w_o=weights[3],
    n_heads=2, temperature=1.0, lambda_=0.1, backend="python",
)
```

Default width is 8 and default head count is 4. Other even widths are supported
when `H` divides `D`. Projections Q/K/V have shape `(H,D,D/H)`; the output matrix
has shape `(D,D)`. If any projection is omitted, its seeded default uses width 8.
Partially supplied projections must be compatible with those defaults.

All four default matrices retain the historical seeded Gaussian variance
`2/(D+D/H)`. This is Xavier scaling for Q/K/V, but differs from output-projection
Xavier variance `1/D`. The random parameters are reproducible, not learned.
An integer non-negative `projection_seed` controls missing projections.

`temperature` must be positive and finite; `lambda_` non-negative and finite;
`block_size` is `None` or a positive integer. Boolean, complex, numeric-string
and temporal aliases are rejected before numeric coercion. Python and direct
Go/Julia/Mojo adapters accept genuine real numeric object arrays.

Non-finite inputs, malformed shapes/topology and unrepresentable numerical
intermediates raise `ValueError`. A finite input does not guarantee that all
products, norms or the result are representable in float64. Invalid numerical
results are never published as coupling matrices.

::: scpn_phase_orchestrator.coupling.attention_residuals

## Runtime selection and compatibility

`backend=None` preserves the automatic loader chain: Rust, Mojo, Julia, Go,
then NumPy. This is a fixed selection order, not a measured performance ranking.
`ACTIVE_BACKEND` and `AVAILABLE_BACKENDS` report import-time loader discovery.
A loader probe does not establish numerical execution or a current benchmark.
Runtime/numerical failures during a selected computation propagate.

Use `backend="rust"`, `"mojo"`, `"julia"`, `"go"`, or `"python"` to require a
particular owner. An unavailable named runtime raises `ImportError`; it cannot
silently produce another backend's result. Invalid backend names raise
`ValueError`. Identity and empty public calls need no runtime even when a name
is supplied. Benchmarks use this explicit API without changing dispatch globals.

| Backend | Build or dependency | Native boundary |
|---|---|---|
| Rust | `maturin develop --release --manifest-path spo-kernel/crates/spo-ffi/Cargo.toml --locked` | PyO3 `attnres_modulate_rust`; one-dimensional float64 arrays and plain scalar controls |
| Go | `go build -buildmode=c-shared -o go/libattnres.so go/attnres.go` | Dimension-aware `AttnResModulateV2`; ctypes ingress validates buffers first |
| Julia | `juliacall` and `julia/attnres.jl` | Lazily loaded `AttnRes.attnres_modulate` over row-major buffers |
| Mojo | `mojo build mojo/attnres.mojo -o mojo/attnres_mojo -Xlinker -lm` | Text protocol V2 carries `D`; bridge negotiates `PROTOCOL` before admission |
| NumPy | Always present | Reference computation using the same coupling equations |

Go's original `AttnResModulate` symbol remains a **width-8-only** ABI. Both C
entry points require correctly sized valid pointers; V2 carries `D` explicitly,
refuses invalid dimensions before creating slices, and leaves output unchanged
on numerical error. Its projections contain `D*D` doubles. The Python bridge
requires V2 and refuses an old library rather than reading the wrong buffer size.

Mojo accepts both the original width-8 five-field request and
`V2 N H block_size temperature lambda D`, followed by `N*N` coupling values,
`N` phases and four `D*D` projection buffers. It checks exact cardinality before
allocation. `PROTOCOL` returns `2`. The bridge rejects obsolete executables.
The subprocess boundary adds serialization and process-start costs. Exactly
`N*N` finite stdout records are required before shared output validation.

All ports target numerical parity within stated float64 tolerances. Small
nonzero differences are not bit-exact results. Text round-tripping and differing
summation/transcendental implementations affect the observed error.

## Integration and scientific limits

```python
from scpn_phase_orchestrator.upde.engine import UPDEEngine

engine = UPDEEngine(n_oscillators=2, dt=0.01, method="euler")
omegas = np.array([0.8, 1.1])
alpha = np.zeros((2, 2))
for _ in range(10):
    K_mod = attnres_modulate(K, phases, lambda_=0.1, backend="python")
    phases = engine.step(phases, omegas, K_mod, 0.0, 0.0, alpha)
```

This feeds a time-varying matrix into the existing integrator; it does not alter
the integration method. Coupling changes can alter trajectories, calibration
coefficients and stability. No global five-percent order-parameter or anchor
bound, non-positive Lyapunov exponent, or performance budget follows from the
attention formula. For example, `lambda_=0.5` permits edge magnitudes to grow
by up to fifty percent, so it cannot guarantee a five-percent anchor tolerance.
A frozen-coupling Lyapunov calculation describes that snapshot, not the complete
state-dependent feedback Jacobian. Stability tests are bounded fixture evidence.

The [2026-10-07 diagnostics](../data/attention_residuals_real_runtime_benchmark_2026-10-07.json)
record all five original owners, source and binary hashes, independent coupled
trajectory checks, and 20 raw repetitions per case. Each sample starts from the
same phases and runs three Euler steps. The table shows median milliseconds per
step, including public validation, dispatch and subprocess costs:

| N | Baseline | NumPy | Rust | Go | Julia | Mojo |
|---|---|---|---|---|---|---|
| 4 | 0.126 | 1.491 | 1.963 | 2.406 | 1.886 | 197.680 |
| 16 | 0.147 | 2.360 | 1.657 | 2.655 | 2.822 | 217.762 |
| 64 | 0.202 | 10.743 | 8.861 | 19.537 | 21.073 | 239.570 |

These are shared-workstation diagnostics with the `powersave` governor and
recorded host load and affinity. They establish neither isolated speedups nor
production latency acceptance. The largest observed coupled-trajectory error
was `8.88e-16`; numerical agreement does not qualify a performance budget.
The [2026-09-26 measurements](../data/attention_residuals_measurement_types_benchmark_2026-09-26.json)
retain their historical source and host identities. Reproduce current results
with:

```bash
python benchmarks/attnres_modulation_benchmark.py --sizes 4 16 64 --steps 3 \
  --repeats 20 --backends python rust go julia mojo --output results.json
```

Dedicated public, bridge, measurement-ingress and stability tests exercise the
coupling law. Independent scalar and analytic oracles, signed/absent/banded
edges and non-default projection widths distinguish real execution from an
identity stand-in. Native tests additionally qualify Rust's exported numerical
boundary. Required-runtime profiles must prove that the named compiled owners
actually ran; optional-dependency refusal is a separate contract.
