# Fractal dimension

`scpn_phase_orchestrator.monitor.dimension` computes a correlation integral,
an empirical correlation-dimension estimate, and the Kaplan–Yorke spectrum
formula. Each public function accepts keyword-only `backend="auto"`,
`"python"`, `"rust"`, `"mojo"`, `"julia"` or `"go"`.

## Mathematical contract

For samples \(x_t\in\mathbb R^d\), full-pair evaluation computes

\[
 C(\varepsilon)=\frac{1}{T(T-1)/2}
   \sum_{t<s}\mathbf 1\{\lVert x_t-x_s\rVert_2<\varepsilon\}.
\]

The comparison is **strict**: a distance equal to the radius is excluded.
Radii must be finite and nonnegative. Public calls sort radii and return
values in that sorted order, including duplicate radii. Direct native bridges
require already nondecreasing radii. `C` is finite, nondecreasing and in
`[0, 1]`; radius zero gives zero. With fewer than two samples, or with no
retained sampled pairs, every count is zero. Identical points with retained
pairs give one at every positive radius. Infinity is not an admitted radius.

Distances use iterative `hypot`, avoiding overflow/underflow from squaring
finite coordinates. If subtraction or the true distance exceeds float64
range, that pair is outside every finite radius. Rounding at a threshold can
still depend on the floating-point implementation; tolerance-based parity
is not a bit-exact guarantee.

For descending finite exponents \(\lambda_1\ge\dots\ge\lambda_N\), let
\(j\) be the largest nonnegative prefix index. The Kaplan–Yorke formula is

\[
 D_{KY}=j+\frac{\sum_{i=1}^j\lambda_i}{|\lambda_{j+1}|}.
\]

An empty spectrum or a negative largest exponent gives zero. If all prefixes
are nonnegative, the result is `N`, including an all-zero spectrum. All owners
sort a copy and use scaled compensated prefixes to avoid cumulative overflow.
The result lies in `[0, N]`. This evaluates a spectrum formula; identifying
it with an information dimension requires dynamical assumptions that this
API does not establish.

## Public API

```text
correlation_integral(trajectory, epsilons, max_pairs=50000, seed=42, *, backend="auto")
correlation_dimension(trajectory, n_epsilons=30, max_pairs=50000, seed=42, *, backend="auto")
kaplan_yorke_dimension(lyapunov_exponents, *, backend="auto")
```

The signatures above describe the API; `*` denotes keyword-only selection.
Trajectories have shape `(T, d)` with `d >= 1`; a one-dimensional signal means
`T` scalar samples. Spectra and radii are one-dimensional. Inputs must be
finite real numeric values. Boolean, complex and numeric-string aliases,
including aliases hidden in object arrays, are refused before coercion.
Real numeric object arrays are supported. Calls preserve caller arrays.

`max_pairs` is a positive integer fitting the native pointer-width unsigned
range, and `seed` is an unsigned 64-bit integer. Boolean controls are refused.
`n_epsilons >= 2` must fit native float64 array byte capacity, even for an
empty/constant trajectory. These are representability checks, not a guarantee
that a requested allocation fits available memory.

`CorrelationDimensionResult` contains `D2`, `epsilons`, `C_eps`, `slope` and
`scaling_range`. It validates finite nonnegative dimension, positive strictly
increasing radii, matching physical counts, aligned finite slopes, and ordered
scaling endpoints. Its arrays are copied from constructor inputs.

### Correlation-dimension estimator

The estimator uses radii from 1% of the estimated diameter to slightly below
the diameter. Diameter uses all points up to 200 samples; larger trajectories
use a fixed seeded subset. A nonzero diameter must generate finite positive,
strictly increasing radii; unrepresentable grids are refused. The local
`log(C)`/`log(epsilon)` slopes use positive counts only. The window of up to
five adjacent slopes with lowest variance supplies the reported mean, clamped
to zero. Leading zero-count intervals retain zero slopes in the aligned array.
Fewer than three positive counts give `D2=0`.

A zero-diameter trajectory returns the sentinel radius `[1.0]`, slope `[0.0]`,
range `(1.0, 1.0)` and `D2=0`; its count is still computed under the selected
owner and sampling contract. Empty/singleton/no-retained-pair cases give zero
counts; identical points with retained pairs give one.

This is a finite-data window estimator, not proof of an asymptotic dimension,
chaos or a scaling law. Sampling density, temporal correlation, embedding,
noise and window choice affect the estimate. No Theiler-window exclusion is
implemented.

### Pair sampling

When `T*(T-1)//2 <= max_pairs`, every unordered pair is evaluated. Otherwise
Python prepares independent seeded index arrays for Python/Go/Julia/Mojo,
drops diagonal draws without replacement draws, and normalises by the actual
retained count. Repeated off-diagonal pairs are allowed.

Before any kernel call, the public API returns zero if its prepared pair list
is empty. This includes an all-diagonal NumPy draw even when Rust is selected.
When reached, the Rust kernel retains its legacy thread-dependent LCG sampling law. It retries diagonal
draws until its requested valid-pair quota is filled and normalises by that
quota. Thread count affects its stream partitioning. Its seed does not select
the same sample as NumPy's generator. Subsampled Rust results receive physical
shape/range validation; they are not compared with a different generator's
sample as if it were an exact reference. Full-pair counts receive an exact
reference comparison within the owner's tolerance.

## Runtime selection and native boundaries

Automatic admission probes Rust, Mojo, Julia, Go, then Python. Runtime status
is exposed in `ACTIVE_BACKEND` and `AVAILABLE_BACKENDS`; pass `backend=` for
reproducible required ownership. Dispatch carries the actual callable's ABI
owner, including fallback after a failed admission. Explicit owners are
required even on degenerate inputs; an absent owner raises `RuntimeError`.

Automatic calls may use Python after an import/runtime/OS/registration failure.
Explicit-owner execution failures are refused. Invalid numerical output raises
`ValueError`; divergence is not silently accepted as successful fallback.
Successful non-Rust bridges validate their output against stable float64
references in addition to shape, finiteness and physical bounds. Public KY
validates every native result. Mojo uses absolute tolerance `1e-9`, other
native owners `1e-12`; the benchmark's Python oracle tolerance is zero.

| Owner | Admission and execution |
| --- | --- |
| Python | NumPy float64 geometry and scaled compensated spectrum prefixes |
| Rust | Installed `spo_kernel` with both callable CI/KY exports; PyO3 validates real finite arrays and plain integer metadata |
| Go | Loadable `libdimension.so` with both C exports; ctypes uses C-int counts and int64 pair indices |
| Julia | Actual `juliacall` runtime plus an included side-file exposing both functions |
| Mojo | Existing executable `dimension_mojo`; complete decimal/count protocol parsing and exact output cardinality |

Go refuses counts above signed C-int range, including virtual oversized spectra
before copying them. The C ABI validates metadata, null/count combinations,
finite values, radii and indices before computation. Actual caller buffer
capacity remains a C ABI precondition; arbitrary pointer validity cannot be
inferred from counts. Julia checks shapes and indices before indexed loops.
Mojo refuses malformed/trailing tokens, bad counts, nonfinite numbers and
nonzero process exits. Native kernels do not build themselves during dispatch.

The standalone Rust `correlation_dimension` helper is not exported through
PyO3. Its legacy strided diameter sample and radius/window policy differ from
the public Python estimator. `backend="rust"` on the public estimator selects
the integral runtime while retaining Python's estimator policy.

Build the desired source artifacts before use:

```bash
python tools/install_spo_kernel.py --release
(cd go && go build -buildmode=c-shared -o libdimension.so dimension.go)
mojo build mojo/dimension.mojo -o mojo/dimension_mojo -Xlinker -lm
julia julia/test_dimension.jl
```

Installed wheels need side-file/native assets in the location expected by their
bridges. Loader admission alone does not establish numerical fidelity; the
owning tests execute the actual public and raw native boundaries.

## Runnable example

```python
import numpy as np
from scpn_phase_orchestrator.monitor.embedding import delay_embed
from scpn_phase_orchestrator.monitor.dimension import (
    correlation_dimension, correlation_integral, kaplan_yorke_dimension,
)

signal = np.sin(np.linspace(0.0, 20.0, 200))
trajectory = delay_embed(signal, delay=3, dimension=2)
result = correlation_dimension(trajectory, backend="python")
assert result.D2 >= 0.0
assert correlation_integral([[0.0], [1.0]], [1.0, 2.0], backend="python").tolist() == [0.0, 1.0]
assert kaplan_yorke_dimension([1.0, -2.0], backend="python") == 1.5
```

## Comparison benchmark

```bash
python -m benchmarks.dimension_benchmark --T-list 50 150 400 --d 3 --n-k 20 \
  --calls 3 --warmups 1 --repeats 3 --require-backends rust mojo julia go python \
  --output dimension-comparison.json
```

The CLI requires all five owners by default. `--require-backends python julia`
requests a labelled subset; unavailable other owners remain visible and
`comparison_complete=0`. The portable reference-suite Python API requires
Python by default. Every record includes all five slots, required scope,
independent scalar `math.dist` full-pair counts, and exact rational KY prefix
references. Missing required owners or invalid output give exit 1. Invalid
controls give exit 2. Failed slots have no timings or success hashes; aggregate
failure has no benchmark success hash. JSON output is written on both
acceptance and parity refusal. Numerical hashes exclude timing and diagnostic
text and bind array shape and little-endian float64 bytes.

Timings measure **both public CI and KY**, including production validation and
independent oracle checks. Median repeated measurements exclude warmup. They
are not directly comparable with the former CI-only timing table.
[Measured results and source/runtime provenance](../data/dimension_real_runtime_2026-10-23.json)
were recorded on 23 October 2026.
They are non-isolated shared-host regression observations, using a debug Rust artifact with LLVM branch instrumentation, and do not establish production speedup or native-only cost.

## Verification surfaces

The owning tests cover independent geometry and high-precision prefixes,
public D2/result lifecycle, actual engine-to-embedding consumption, deterministic
full pairs and separately qualified sampling laws. Fresh subprocess profiles
exercise missing, unusable, nonexporting and valid Rust/Go/Julia assets and
revoked Mojo execution permission. Raw PyO3/Go ABI and Mojo protocol tests
exercise safe malformed requests. Failure-only contract injections are negative
refusal evidence, not successful native execution. CI prepares actual required
runtimes before collecting the corresponding tests.

References: Grassberger & Procaccia (1983), *Physical Review Letters* 50,
346–349; Kaplan & Yorke (1979), *Lecture Notes in Mathematics* 730, 228–237.
