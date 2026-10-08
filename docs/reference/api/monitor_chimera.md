# Instantaneous local coherence and chimera classification

`monitor.chimera` reports local phase coherence on directed positive non-self
adjacency. It classifies a snapshot; a persistent dynamical chimera needs
trajectory and frequency evidence beyond this statistic.

## Equation and populations

For $N(i)=\{j\ne i:K_{ij}>0\}$, the monitor computes

$$
R_i=\left|\frac{1}{|N(i)|}\sum_{j\in N(i)}e^{i(\theta_j-\theta_i)}\right|
   =\left|\frac{1}{|N(i)|}\sum_{j\in N(i)}e^{i\theta_j}\right|.
$$

An empty neighbourhood has $R_i=0$. Positive edge magnitudes count equally;
zero and negative edges do not count. The diagonal is excluded explicitly,
including small positive residues admitted by the zero-diagonal tolerance.
The factorization removes the centre's unit phasor before taking the magnitude,
avoiding overflowing subtraction between opposite finite extreme angles.
This is an algebraic identity, without approximate phase wrapping.

| Population | Condition |
|---|---|
| Coherent | $R_i>0.7$ |
| Incoherent | $R_i<0.3$ |
| Boundary | $0.3\le R_i\le0.7$ |

`chimera_index = boundary_count / N`; the empty population returns zero.
The 0.7/0.3 thresholds are implementation choices. A zero boundary fraction
can accompany coherent and incoherent populations together. Global phase shifts
preserve the exact statistic; floating point comparisons use tolerances.
A connected synchronized neighbourhood has unit magnitude. Uniformly spaced
phases with all-to-all non-self adjacency have the exact value `1/(N-1)`.

[Kuramoto and Battogtokh's primary paper](https://arxiv.org/pdf/cond-mat/0210694)
uses a spatially weighted nonlocal field and dynamical self-consistency.
This unweighted adjacency monitor does not reproduce that full model or derive
its thresholds from the paper. The spatial/temporal classification in
[Kemeth et al.](https://arxiv.org/pdf/1603.01110) also uses different observables.

## Public API and measurements

```python
from scpn_phase_orchestrator.monitor.chimera import (
    ACTIVE_BACKEND, AVAILABLE_BACKENDS, ChimeraState,
    local_order_parameter, detect_chimera,
)

local = local_order_parameter(phases, knm, backend="python")
state = detect_chimera(phases, knm, backend="python")
```

Both functions accept `backend: str | None` as a keyword-only argument.
Phases must retain shape `(N,)` and coupling `(N, N)`. Finite real measurements
and representable real numeric object arrays are accepted. Boolean, textual,
complex and temporal aliases, scalar-to-population promotion, ragged inputs,
nonfinite values, wrong dimensions and nonzero diagonals are refused before
execution. The zero-diagonal tolerance is `atol=1e-15, rtol=0`.
Backend results must be finite real `(N,)` vectors within `[0,1]`.

`ChimeraState` holds copied coherent/incoherent index lists and a finite index
in `[0,1]`. Construction checks nonnegative unique disjoint indices. Its frozen
fields prevent reassignment; its lists remain mutable. Direct construction does
not know the original population size.

## Actual owner selection

Automatic resolution caches the first admitted owner in the order
**Rust → Mojo → Julia → Go → Python**. Valid explicit names request exactly that
owner. An unknown name raises `ValueError`; an unavailable named owner raises
`ImportError`, even for empty measurements. Numerical execution and invalid
output errors propagate. Execution failure does not qualify another owner.

| Owner | Original runtime | Parity tolerance |
|---|---|---:|
| Rust | Installed `spo_kernel.detect_chimera_rust` extension | `1e-12` |
| Mojo | Pinned-toolchain `mojo/chimera_mojo` stdin executable | `1e-9` |
| Julia | `juliacall` and actual loaded `julia/chimera.jl` | `1e-12` |
| Go | `go/libchimera.so` with admitted V2 ABI | `1e-12` |
| Python | NumPy public implementation | `1e-12` against scalar oracle |

The Rust adapter extracts the local-order vector from its existing four-tuple;
Python applies the monitor's population thresholds. Direct Go/Julia/Mojo APIs
retain flat float64 inputs and a nonboolean nonnegative integer `n`, with exact
`n`/`n*n` extents. Their valid empty transport calls return empty vectors before
loading, and do not qualify native availability.

Go V2 checks dimensions, supplied extents and pointers before `unsafe.Slice`,
then numerical inputs before writing. Status 1 denotes invalid buffer/dimension
admission and status 2 invalid numerical domain. The original ABI remains
available for valid legacy consumers; Python requires V2. Julia validates before
`@inbounds`; Mojo validates its `CHI` header, counts, finite inputs and exact
scalar output rows before returning. Rust fallible engine functions report
invalid domains; legacy local-order returns an empty sentinel and legacy detect
returns a NaN-index sentinel on invalid input. The FFI uses the fallible path.

## Distinct JAX contract

[`nn.chimera`](../nn_chimera_contract.md) counts **all nonzero edges**, including
negative and self edges. Its index is **local-order variance**, and its masks
use inclusive default thresholds `>=0.8` and `<=0.3`. Phase gradients apply on
fixed topology away from zero phasors. Hard adjacency has zero coupling-amplitude
gradients on fixed support. These index/mask models are separate from the monitor.

## Current comparison

The 2026-10-08 collection measured 20 completed calls per owner at each of
N=16,64,256 after two untimed warm-ups. JAX JIT compilation is excluded and each
call synchronizes its result. All owners share the positive zero-diagonal graph,
the common local-order domain; their classification/index models are not mixed.
Mean milliseconds per completed public call:

| N | Rust | Mojo | Julia | Go | Python | JAX JIT |
|---|---:|---:|---:|---:|---:|---:|
| 16 | 0.121789 | 102.691809 | 0.339175 | 0.237051 | 0.645868 | 0.067708 |
| 64 | 0.151222 | 88.713500 | 0.440440 | 0.308219 | 2.617802 | 0.189669 |
| 256 | 1.469174 | 131.571343 | 4.385303 | 2.318871 | 7.245365 | 0.754322 |

[Raw timings, distributions, workload/source/native hashes and host metadata](../data/chimera_real_runtime_benchmark_2026-10-07.json)
identify the actual measurement timestamps. The filename follows this task's
2026-10-07 activation. The host was shared; these are diagnostic timings, without
controlled deployment deadlines, speedup claims or a fastest-owner ranking.

```bash
python -m benchmarks.chimera_benchmark --comparison --sizes 16 64 256 --calls 20 --seed 2026 --density 0.3 --output chimera_comparison.json
python -m benchmarks.chimera_benchmark --parity-gate --sizes 16 --calls 20 --require-backends rust
```

The original parity-gate consumer fields remain available. Native JSON output
restores blocking on Julia-initialized original stdout so pipe backpressure does
not truncate a comparison record. Failed parity acceptance exits nonzero. A provisioned lane
uses `--require-backends` to refuse missing owners rather than treating their
unavailable records as executed parity. Its timed operation bundles five
contracts and is separately labelled from the one-call comparison.

## Verification and cost

The original algorithm, backend, stability, measurement, dispatch and winding
modules remain exercised. `tests/test_chimera_real_runtime.py` covers actual
owners, admitted self residues, extreme phases and original import consumers.
The benchmark contracts exercise CLI routes, raw data and deliberate negative
output/custody controls. Native tests cover original boundaries and actual old
Go or malformed Julia artifacts in isolated installations.

```bash
make chimera-quality
python -m pytest tests/test_chimera_real_runtime.py -k python
cargo bench --locked -p spo-engine --bench parallel_bench -- chimera_local_order
```

Dense adjacency traversal costs `O(N^2)` time. The coupling input costs `O(N^2)`;
additional allocation depends on the implementation and transport. The
factorization computes only `N` trigonometric phasors. The Rust benchmark uses
20 Criterion samples and checks a two-neighbour interior row against `cos(0.17)`
before timing. Synthetic EEG examples exercise UPDE-to-monitor wiring without
claiming clinical interpretation.
