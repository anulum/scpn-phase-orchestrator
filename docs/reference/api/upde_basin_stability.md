# Basin stability — finite-horizon threshold classification

`basin_stability` reports the fraction of seeded Kuramoto trials whose measured
order parameter meets a chosen threshold. Each trial uses explicit Euler over a
finite transient and measurement window. This statistic does not certify
convergence to an attractor, its asymptotic basin volume, or linear stability.

[Menck et al. (2013)](https://www.nature.com/articles/nphys2516) introduced basin
stability as a complement to local stability analysis. The distinction between
that asymptotic concept and a numerical finite-window classification matters
here; see also [Schultz et al., *Potentials and limits to basin stability
estimation*](https://arxiv.org/abs/1603.01844). This implementation does not
reproduce the inertial network model in
[Ji et al. (2014)](https://www.nature.com/articles/srep04783).

## Trial law and measurement

The graph uses target rows and source columns. Phases and lags are in radians,
frequencies in radians per second, and coupling entries in inverse seconds:

$$
\theta_i^{s+1}=\theta_i^s+\Delta t\left(\omega_i+
\kappa\sum_j K_{ij}\sin(\theta_j^s-\theta_i^s-\alpha_{ij})\right).
$$

Every derivative uses the complete old phase snapshot. There is no hidden
$1/N$ normalization. Signed, directed and self-couplings are admitted by this
trial API. Only exactly zero scaled edges are skipped. Phases remain unwrapped;
the trigonometric readout is periodic.

After discarding `n_transient` steps, each measurement follows another Euler
step. For `n_measure = m > 0` the result is

$$
\bar R=\frac{1}{m}\sum_{s=n_t+1}^{n_t+m}
\left|\frac{1}{N}\sum_j e^{i\theta_j^s}\right|.
$$

An empty measurement window returns `0.0` after input and named-owner validation,
without executing an unused transient. For `n_samples = M`, the estimator
returns `count(R_final >= R_threshold) / M`; its empty-sample convention is zero.
`R_final` contains window averages, despite its historical name.

## Backend selection and reproducibility

All three public functions accept the keyword-only `backend` argument:
`"python"`, `"rust"`, `"go"`, `"julia"`, or `"mojo"`. `None` uses the declared
Rust/Mojo/Julia/Go/Python preference. That order is not a measured speed ranking.
An unavailable named owner raises `ImportError`; arithmetic and output-contract
errors propagate. A named request never returns another owner's computation.

The Python Monte Carlo loop draws initial phases with
`np.random.default_rng(seed).uniform(0, 2*np.pi, N)`. All selected owners receive
the same trial vectors, including the multi-threshold API. Cross-language
comparison uses floating-point tolerances. Bit-exact equality is not promised.
The direct legacy `spo_kernel.basin_stability_rust` instead uses an LCG and is a
separate API, with a different seed-to-phase sequence.

## Validation and native boundaries

Public vectors must be nonempty and finite; graph and lag matrices must match
`(N, N)`. Boolean and complex aliases are rejected;
numeric-string aliases are rejected before float coercion. `dt` must be
finite and positive, `k_scale` finite, counts nonnegative integers, and thresholds
in `[0, 1]`. Nonfinite scaled couplings, active phase differences, velocities,
Euler iterates and results are errors. A disconnected graph does not evaluate
unused overflowing pair differences.

Native Rust provides fallible `try_*` APIs. Python FFI converts their errors to
`ValueError`, before indexed computation. Legacy Rust scalar/sweep signatures
remain available; their documented failure sentinels are not Python results.

The Go Python bridge requires `SteadyStateRV2`, with four explicit buffer-length
fields and signed 64-bit dimension/count metadata. The library checks that
metadata before constructing slices. Actual readable pointer storage remains a
C caller obligation. `SteadyStateR` retains its original ABI for valid callers,
which must provide `N`, `N`, `N*N`, and `N*N` readable doubles. Rebuild older Go
libraries before using the current checked bridge.

Julia checks native dimensions, finite values and step counts before indexing.
Mojo checks the complete `STEADY` request before allocating buffers and requires
exactly one finite scalar stdout record. Native zero-window calls validate their
inputs before returning zero.

## Public API

::: scpn_phase_orchestrator.upde.basin_stability
    options:
        show_root_heading: true
        members_order: source

`multi_basin_stability` keeps labels such as `"R>=0.30"`. Distinct thresholds must have distinct two-decimal labels; ambiguous labels
raise an error instead of silently dropping a result. Repeating the identical
threshold retains its original single-label behaviour. Result records enforce
count/fraction consistency with their thresholded trial array; supplied float32
fraction rounding is accepted within relative tolerance 1e-7 and canonicalized
to the exact count/sample ratio. `n_converged` is the historical field name for the count of
trials satisfying the finite-window threshold.

```python
import numpy as np
from scpn_phase_orchestrator.upde.basin_stability import (
    basin_stability, multi_basin_stability, steady_state_r,
)

phases = np.array([0.0, np.pi / 2])
omegas = np.zeros(2)
coupling = np.array([[0.0, 0.5], [0.5, 0.0]])
trial = steady_state_r(phases, omegas, coupling, dt=0.04,
                       n_transient=2, n_measure=3, backend="python")
result = basin_stability(omegas, coupling, n_samples=8, seed=29,
                         n_transient=2, n_measure=3, backend="python")
thresholds = multi_basin_stability(omegas, coupling, n_samples=8, seed=29,
                                  R_thresholds=(0.3, 0.6, 0.8),
                                  n_transient=2, n_measure=3, backend="python")
assert 0 <= trial <= 1
assert result.S_B == result.n_converged / result.n_samples
```

## Evidence and limits

Independent scalar Euler and analytic two-oscillator contracts are in
`tests/test_basin_bifurcation_real_runtime.py`. Genuine installed native and
kernel-absent profiles are in `native-tests/test_basin_bifurcation_profiles.py`.
Native contracts are exercised in Rust, Go, Julia and the Mojo executable.
Injected malformed outputs are negative boundary controls, not parity evidence.

Threshold sensitivity, sample variation, timestep error and finite-window bias
must be assessed for the chosen model. The API supplies no validated operating
threshold, confidence interval, attraction certificate or production safety
classification. Previously unqualified speed estimates are removed; comparison
runs belong to the actual source-qualified benchmark record.


## Source-qualified comparison diagnostics

The [2026-10-07 raw record](../data/basin_bifurcation_real_runtime_benchmark_2026-10-07.json)
contains twenty repetitions of the trial, two-sample Monte Carlo, three-point
sweep and threshold search for N=4/16/64. Every original named owner is observed
and numerically checked against the independent scalar Euler oracle. Maximum
observed difference is 5.56e-17. The record includes all 1,440 raw durations
(including the scalar baseline), fourteen source hashes, three native binary
hashes, affinity, governor and before/after load.

Single-trial medians, in milliseconds (two transient and three measurement steps):

| N | Python | Rust | Go | Julia | Mojo |
|---|---:|---:|---:|---:|---:|
| 4 | 0.426 | 0.229 | 0.450 | 0.347 | 101.660 |
| 16 | 1.310 | 1.130 | 1.756 | 1.833 | 61.247 |
| 64 | 20.404 | 17.880 | 25.930 | 22.102 | 102.084 |

These measurements ran on a shared workstation with a powersave governor.
They include public input validation and boundary overhead, including a subprocess
per Mojo trial. The Rust Python binding uses the recorded development build.
They establish current diagnostic comparisons, not an isolated speed ranking or
production performance acceptance.
