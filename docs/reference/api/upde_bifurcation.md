# Coupling sweeps and threshold searches

The historical `bifurcation` API measures finite-horizon Kuramoto responses on
an independent coupling grid or searches for an `R = 0.1` classification
boundary. Its integrator, graph orientation and measurement window are the
[basin trial law](upde_basin_stability.md). Every coupling trial starts from the
same NumPy-seeded phase vector.

This computation does not implement pseudo-arclength continuation, track
solution branches, calculate eigenvalues, or certify a dynamical bifurcation.
Classical infinite-population critical-coupling formulas are not acceptance
oracles for a particular finite, directed, lagged network and finite window.

## Independent sweep

`trace_sync_transition` evaluates `n_points` uniformly spaced couplings in
`K_range`. It returns the sampled `K` and mean `R` values. When the samples have
an upcrossing `R[i] < 0.1 <= R[i+1]`, the first such interval produces

$$
K_{\mathrm{critical}}=K_i+
\frac{0.1-R_i}{R_{i+1}-R_i}(K_{i+1}-K_i).
$$

Without an upcrossing, `K_critical` is `None`. Starting above the threshold does
not itself count as an upcrossing. Reusing the initial condition improves
comparability across grid points; it does not perform a hysteresis sweep with
terminal-state carryover. `stable=True` is a historical compatibility marker,
not a measured or inferred stability certificate.

## Binary search

`find_critical_coupling` first measures the upper endpoint `K = 20`. A value
below `0.1` returns `NaN`. Otherwise it bisects `[0, 20]`, retaining the lower
endpoint after a subthreshold midpoint and the upper endpoint otherwise.
It stops when the interval width is less than `tol`, or after 30 iterations,
and returns the interval midpoint.

This search assumes a monotone response. It does not establish that assumption
or check that the lower endpoint starts below threshold. A singleton population,
for example, has `R = 1` throughout and produces a small positive search result
rather than a certified critical point at zero. `tol` bounds a search interval,
not physical, statistical or timestep error. An empty measurement window yields
zero `R`, no sweep upcrossing, and `NaN` from the search.

The lower-bracket convention also applies to finite networks with R already
above 0.1 at zero coupling. For a zero-coupling four-node template and seeds 42
or 1, every trial is constant and above threshold: the sweep has no upcrossing,
while search with `tol=0.05` returns `0.01953125`. This is the historical final
interval midpoint, not evidence of a transition. The Python, Rust and delegated
owners preserve it; qualify a lower/upper bracket separately before interpreting
a returned value as a classification boundary.

## Dispatch and validation

Both functions accept a keyword-only `backend`: `"rust"`, `"go"`, `"julia"`,
`"mojo"`, `"python"`, or `None`. Automatic selection uses the original batched
Rust composites when installed. Otherwise the Python composite delegates each
trial to the basin preference chain. Naming Rust uses its batched composite when both composite exports exist;
an older partial Rust installation can use its original per-trial kernel.
Naming another owner delegates all trials to that owner. Unavailable named
owners raise `ImportError`, without fallback. Computation faults propagate.

The public sweep requires a nonempty finite real frequency vector, a finite
nonnegative increasing `K_range`, and at least two grid points. The template
must have a zero diagonal within the existing `1e-12` admission tolerance.
An omitted template is all-to-all with off-diagonal entries `1/N`; custom
matrices receive no implicit normalization. Nonzero finite lag matrices are
supported by the sweep. The search uses zero lag. Step counts and seeds are
nonnegative integers; `dt` and `tol` are finite and positive. Boolean, complex
and numeric-string aliases are rejected.

Rust's direct native grid API separately retains its historical valid singleton,
equal-endpoint and negative-coupling grids. Its checked `try_*` functions report
invalid dimensions and numerical overflow; Python FFI converts those errors to
`ValueError`. This native compatibility domain does not broaden the Python
public sweep's graph or range contract.

Composite output arrays are checked before constructing public records: their
shapes must match the requested grid; `K` values must be finite, monotone and
inside the requested range and match the requested uniform grid; `R` values
must be finite and inside `[0, 1]`.
The optional critical coupling must be finite and nonnegative, or `NaN` for an
absent crossing, and agree with the first sampled upcrossing and interpolation. Injected invalid outputs are tested as negative controls.

## Public API

::: scpn_phase_orchestrator.upde.bifurcation
    options:
        show_root_heading: true
        members_order: source

```python
import numpy as np
from scpn_phase_orchestrator.upde.bifurcation import (
    trace_sync_transition, find_critical_coupling,
)

omega = np.array([-0.2, 0.1, 0.4])
diagram = trace_sync_transition(omega, K_range=(0.0, 3.0), n_points=5,
                               n_transient=2, n_measure=4, backend="python")
assert diagram.K_values.shape == diagram.R_values.shape == (5,)
if diagram.K_critical is not None:
    print(diagram.K_critical)
critical = find_critical_coupling(omega, n_transient=2, n_measure=4,
                                 tol=0.1, backend="python")
print(critical)  # May be NaN; no result is fabricated for an absent crossing.
```

## Evidence and numerical limits

`tests/test_basin_bifurcation_real_runtime.py` compares original public consumers
against a separate scalar Euler oracle, with all five named owners required in
the native qualification profile. `tests/test_bifurcation_dispatch.py` observes
original owner calls; injected malformed results are negative controls.
`native-tests/test_basin_bifurcation_profiles.py` exercises actual installed native
and genuinely absent environments, with source identity checks.

The grid has finite resolution, and every value depends on the timestep,
transient, window, frequency sample, graph and seed. Vary these inputs before
using a classification in a scientific conclusion. Previous unqualified timing,
precision, calibration and linear-speedup estimates are removed. Actual
comparison diagnostics must retain their source/binary provenance and host load.
