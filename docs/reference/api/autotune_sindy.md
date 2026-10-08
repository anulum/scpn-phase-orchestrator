<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Phase-SINDy public numerical contract -->

# Phase-SINDy symbolic discovery

`PhaseSINDy` estimates constant frequencies and directed sine couplings from
sampled phase trajectories. It fits a specified Kuramoto feature library;
interactions outside that library are not discovered by this estimator.
The fitted model is an offline inference result requiring operator review.

## Equations, sampling and units

For target node $i$, the assumed model is

$$
\dot\theta_i = \omega_i + \sum_{j\ne i} K_{ij}\sin(\theta_j-\theta_i).
$$

`phases` has shape `(T, N)` and contains real finite angles in radians. `dt` is
a positive finite sample period in seconds. The fit requires `N >= 1` and
`T - 1 >= N`; one oscillator with two samples is sufficient for admission.
Admission does not establish that the data identify a physical model.

Derivatives use adjacent unwrapped increments divided by `dt`. Whole-turn
aliases are reduced by arbitrary integer multiples of $2\pi$. Exact positive
and negative half turns retain the sign of the original increment. The Python
path uses [NumPy unwrapping](https://numpy.org/doc/stable/reference/generated/numpy.unwrap.html);
the native path implements the same principal-increment convention.
Floating-point operations need not be bit-identical across implementations.

Physical angular velocity is identifiable from these sampled increments only
when the sampling resolves the rotation between observations. Unwrapping
cannot recover missing complete turns. Finite differences also introduce
sampling error and amplify measurement noise.

Rows of $K$ are targets and columns are sources: $K_{ij}$ describes $j\to i$.
Both $\omega_i$ and $K_{ij}$ are reported in radians per second. This estimator
allows signed couplings and omits self-coupling terms.

## Sparse regression and numerical rank

For each target, the Python library consists of a constant column followed by
$\sin(\theta_j-\theta_i)$ for ascending source indices other than the target.
Its shape is `(T - 1, N)`.

Sequential thresholded least squares starts with a rectangular least-squares
fit, sets terms with magnitude strictly below `threshold` to zero, and refits
the retained features. It performs `max_iter` threshold/refit rounds. A term
exactly equal to the threshold is retained. If no features remain, the
coefficient vector is zero.

This follows the sparse-library approach of
[Brunton, Proctor and Kutz (2016)](https://doi.org/10.1073/pnas.1517384113).
The selected features and sampling assumptions constrain the equations that
can be inferred; sparsity is not a guarantee of global optimisation or truth.

The Python implementation uses
[SciPy least squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html)
with `cond` set to double-precision machine epsilon times the larger matrix
dimension, matching [NumPy's rank cutoff](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html).
Rust uses the existing
`nalgebra` rectangular SVD primitive with the corresponding relative cutoff,
without forming normal equations. The native solve scales the target to avoid
projection overflow and limits SVD convergence iterations. Invalid derived
arithmetic or a failed solve is refused.

Both owners order the regression's constant column first. Rust remaps the fitted
coefficients back to its public diagonal-frequency layout after solving.

Dependent feature columns yield a minimum-norm least-squares solution. For
example, two trajectories with a constant phase offset have constant sine
features, which cannot independently distinguish a frequency from its coupling.
Even a small residual does not make those separate coefficients identifiable.
Full-rank, sufficiently excited data and a suitable model library are needed
for unique coefficient recovery. Near-dependent features require particular
care when interpreting a fit.

## Public coefficient layout and state

`fit` returns `N` double-precision vectors, each of shape `(N,)`. For three
nodes, the layout is:

| Target | Returned vector |
|---|---|
| 0 | `[omega_0, K_01, K_02]` |
| 1 | `[omega_1, K_10, K_12]` |
| 2 | `[omega_2, K_20, K_21]` |

The native engine returns a row-major `(N, N)` matrix with frequencies on the
diagonal and couplings off the diagonal. The Python public wrapper remaps it
into the layout above. `feature_names` matches every returned vector.

`get_equations()` returns one equation per target, formatted to four decimal
places. Terms of magnitude at most `1e-6` are omitted from text; the numerical
coefficients remain available independently. An equation with no displayed
terms is `d(theta_i)/dt = 0`. Calling this method before a successful fit raises
`RuntimeError`.

Boolean aliases, complex phases, invalid controls, non-finite values,
inconsistent dimensions and invalid backend output are rejected. A trajectory
with too few samples or zero nodes clears the previous fit. Other failed fits
preserve previous coefficients and equations. The Python path publishes its
new state only after every target has fitted successfully.

## Example

This self-contained example generates independently specified Euler samples of
a two-node network, fits them and prints the inferred equations. It uses
synthetic data; successful recovery does not validate a measured physical system.

```python
import numpy as np
from scpn_phase_orchestrator.autotune.sindy import PhaseSINDy

omega = np.array([1.0, 1.7])
k = np.array([[0.0, 0.2], [-0.1, 0.0]])
dt = 0.02
phases = np.empty((400, 2), dtype=np.float64)
phases[0] = [0.0, 0.8]
for sample in range(1, len(phases)):
    previous = phases[sample - 1]
    derivative = omega + np.array([
        k[0, 1] * np.sin(previous[1] - previous[0]),
        k[1, 0] * np.sin(previous[0] - previous[1]),
    ])
    phases[sample] = (previous + dt * derivative) % (2 * np.pi)

model = PhaseSINDy(threshold=0.01)
coefficients = model.fit(phases, dt)
np.testing.assert_allclose(coefficients, [[1.0, 0.2], [1.7, -0.1]], atol=1e-9)
print(*model.get_equations(), sep="\n")
```

For CSV onboarding, phase-like columns feed the same estimator through
[auto-binding](../../guide/auto_binding.md). The discovery record preserves
signed source/target edges, equations, derivative sample counts and residual
quality. The proposal remains subject to binding validation and operator
review; a self-fit does not receive external-validation status.

## Real runtime diagnostics

Run the same scoped source, test and diagnostic quality checks locally and in CI:

```bash
make phase-sindy-quality PYTHON=.venv/bin/python
```

The maintained diagnostic exercises directed recovery, arbitrary whole-turn
aliases, analytical dependent features and empty support through `PhaseSINDy`.
It observes the installed native callable or SciPy call without replacing it,
checks independent references, and retains repeated timings with source,
binary and environment identity.

```bash
.venv/bin/python benchmarks/phase_sindy_benchmark.py --expect-backend native --repeats 20
```

Use `--expect-backend python` in a genuinely separate installation without
`spo-kernel`. A required backend mismatch or failed numerical reference exits
non-zero. The optional estimator uses Python when the native entry point is
unavailable; requiring native execution is a separate explicit runtime contract.
The kernel runs in process during a fit and needs no permanently running service.

Shared-workstation timings are local diagnostics. They do not establish an
isolated speedup, real-data discovery quality or production latency guarantees.

### Source-bound coverage admission

`tools/phase_sindy_coverage_policy.json` binds the owning source files and locked
native dependency to the reviewed measurement. Changes invalidate that binding
until new real measurements and review replace it. The admission command checks
complete Python statement and branch membership and rejects additional gaps:

```bash
python tools/phase_sindy_coverage.py --python-report python.json --native-report native.json
```

The eight statements in the original-owner profile observers execute while
Python tracing is suspended. They remained unrecorded with coverage.py 7.16.1's
monitoring core on Python 3.12.3 in both actual runtime profiles. Their exact
lines and branch outcomes are measurement exclusions, not numerical coverage.
If a report records them, no exclusion is credited.

Native SVD nonconvergence remains an uncovered defensive path, including its
error propagation during refitting. The 100000-iteration bound and refusal are
retained. Native statement coverage is below 100%; the admission receipt states
this explicitly. The separate solve-error closure is source-bound to nalgebra
0.35.0's retained U/V and non-negative cutoff invariant. Neither condition
changes the repository's global or per-module coverage thresholds.

The small recorded exports under `tests/fixtures/phase_sindy_coverage/` exercise
the admission protocol and source binding in CI. They contain real branch and
region vectors, not substituted successful solver results. They do not replace
fresh runtime tests or establish current hosted coverage by themselves.
The command validates report structure and source scope; it does not authenticate
the measuring process. Updating source hashes alone cannot qualify a changed
implementation. Source changes, including changes to the owning tests and
observers, require new raw measurements, source receipts and review. The repository
does not automate the complete native/Python report join.

## API reference

::: scpn_phase_orchestrator.autotune.sindy
    options:
      docstring_style: numpy

The native Rust entry point is:

```rust
pub fn sindy_fit(
    phases: &[f64],
    n_osc: usize,
    n_time: usize,
    dt: f64,
    threshold: f64,
    max_iter: usize,
) -> Result<Vec<f64>, String>
```

The PyO3 binding exposes `spo_kernel.sindy_fit_rust`; rejected native inputs or
numerical errors become Python `ValueError`. Public input/output shape, units
and source/target orientation remain the same for both supported implementations.
