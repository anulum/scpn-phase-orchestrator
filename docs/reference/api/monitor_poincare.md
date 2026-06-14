# Poincare section monitor API reference

`scpn_phase_orchestrator.monitor.poincare` extracts Poincare-section crossings
from continuous trajectories and phase histories.
For oscillator systems this turns a high-dimensional flow into a sequence of
section crossings, making periodic locking, quasi-periodicity, return-time
spread, and chaotic recurrence structure measurable without assuming a single
phase model.

The monitor provides:

- a generic hyperplane-crossing function,
- a phase-oscillator crossing function,
- a return-time convenience function,
- a validated result dataclass,
- a deterministic Python implementation,
- optional Rust, Mojo, Julia, and Go direct backend dispatch.

The API is observational.
It does not mutate solver state.
It does not tune couplings.
It does not actuate a controller.
It reads validated trajectory samples and returns validated recurrence evidence.

## Primary use cases

- Detect stable limit-cycle entrainment from convergent return times.
- Compare oscillator regimes before and after coupling changes.
- Compare open-loop and supervised runs without depending on a single order
  parameter.
- Build low-dimensional recurrence maps for nonlinear trajectory diagnostics.
- Gate simulator regressions when crossing counts or interpolation times drift.
- Measure return-time spread for periodic, quasi-periodic, or chaotic regimes.
- Validate optional accelerator backends against the Python crossing contract.
- Feed monitor evidence into reports, notebooks, and review-only dashboards.

## Exported symbols

| Symbol | Kind | Purpose |
| --- | --- | --- |
| `PoincareResult` | dataclass | Validated crossings, crossing times, return times, and summary statistics. |
| `poincare_section` | function | Generic hyperplane Poincare section for 1-D or 2-D trajectories. |
| `phase_poincare` | function | Phase-oscillator section for crossings of one oscillator modulo `2*pi`. |
| `return_times` | function | Convenience wrapper returning only generic-section return times. |
| `ACTIVE_BACKEND` | string | First available backend in the fallback chain. |
| `AVAILABLE_BACKENDS` | list | Available backend names, always including `python`. |

## Backend chain

The monitor resolves backends in this order:

1. Rust,
2. Mojo,
3. Julia,
4. Go,
5. Python.

`python` is always available.
Optional direct backends are used only when their runtime and callable surface
load successfully.
If a selected direct backend raises or returns malformed output, public section
functions fall back to the Python implementation for the same validated input.

The backend chain preserves the public contract:
invalid input fails before optional runtime loading, and invalid backend output
is rejected before assembly into a `PoincareResult`.

## Mathematical model

A Poincare section records where a trajectory crosses a codimension-one surface.
For the generic function the surface is a hyperplane:

```text
normal . state - offset = 0
```

The crossing direction determines which sign changes count:

- `positive`: signed distance moves from negative to non-negative,
- `negative`: signed distance moves from positive to non-positive,
- `both`: signed distance changes sign in either direction.

For phase oscillators the section is defined by one oscillator crossing a target
phase modulo `2*pi`.
The phase monitor unwraps the target oscillator internally for crossing
detection while returning full phase vectors at crossing times.

## Interpolation model

Both generic and phase monitors linearly interpolate between adjacent samples.
When a crossing lies between sample `i` and `i + 1`, the monitor computes a
fractional crossing time `i + alpha` and an interpolated state vector.

For the generic hyperplane section the crossing fraction is
`alpha = -d0 / (d1 - d0)`, where `d0`/`d1` are the signed distances at samples
`i`/`i + 1`.

For the phase section the wrapped phase advances from `shifted[i]` up through
the `2*pi`-equivalent-`0` section boundary into `shifted[i + 1]`, so the
fraction at which it reaches the boundary is

```text
alpha = (2*pi - shifted[i]) / ((2*pi - shifted[i]) + shifted[i + 1])
```

i.e. the remaining distance to the boundary over the full wrapped step. This
makes the interpolated section-oscillator value land exactly on `section_phase`
modulo `2*pi`.

This means crossing times are sample-index times, not physical seconds, unless
the caller's sample spacing is one second.
To convert to physical time, multiply return times by the sampling interval used
for the trajectory.

## Public boundary contract

`poincare_section(trajectory, normal, offset=0.0, direction="positive")` accepts
finite real 1-D or 2-D trajectories and a finite real section normal.

The public boundary rejects:

- boolean aliases in trajectory data,
- complex trajectory values,
- non-numeric trajectory values,
- non-finite trajectory values,
- trajectories with rank other than one or two,
- boolean aliases in the normal vector,
- complex normal values,
- non-numeric normal values,
- non-finite normal values,
- normal vectors whose length does not match the state dimension,
- non-finite or non-real `offset`,
- boolean alias `offset`,
- unknown `direction` values.

A one-dimensional trajectory is treated as a one-dimensional state history with
shape `(T, 1)` after validation.
A two-dimensional trajectory is interpreted as `(timesteps, dimension)`.

## Phase boundary contract

`phase_poincare(phases, oscillator_idx=0, section_phase=0.0)` accepts finite
real phase histories with shape `(T, N)`.

The public boundary rejects:

- boolean aliases in phase data,
- complex phase values,
- non-numeric phase values,
- non-finite phase values,
- arrays with rank other than one or two,
- non-integral oscillator indexes,
- boolean alias oscillator indexes,
- negative oscillator indexes,
- oscillator indexes outside `[0, N)`,
- non-finite section phases,
- non-real section phases,
- boolean alias section phases.

A one-dimensional phase history is treated as a single-oscillator history with
shape `(T, 1)` after validation.

## PoincareResult

`PoincareResult` is the validated public result record.

Fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `crossings` | `FloatArray` | Crossing states with shape `(n_crossings, dimension)`. |
| `crossing_times` | `FloatArray` | Strictly increasing fractional sample-index crossing times. |
| `return_times` | `FloatArray` | Differences between consecutive crossing times. |
| `mean_return_time` | `float` | Mean of `return_times`, or `0.0` when no returns exist. |
| `std_return_time` | `float` | Standard deviation of `return_times`, or `0.0` when no returns exist. |

The dataclass validates and normalizes its fields in `__post_init__`.
It rejects malformed records even if a caller constructs the dataclass directly.

## PoincareResult invariants

A valid result satisfies:

- `crossings` is a finite two-dimensional `float64` array,
- `crossing_times` is a finite one-dimensional `float64` array,
- crossing-time count equals crossing row count,
- crossing times are strictly increasing when more than one crossing exists,
- `return_times` is a finite one-dimensional `float64` array,
- return-time length equals `max(0, n_crossings - 1)`,
- every return time is non-negative within numerical tolerance,
- return times match `np.diff(crossing_times)`,
- `mean_return_time` is finite and non-negative,
- `std_return_time` is finite and non-negative,
- mean and standard deviation match the supplied return-time array.

These invariants protect downstream reports from accepting hand-assembled or
backend-produced recurrence records with inconsistent statistics.

## poincare_section

`poincare_section` computes generic hyperplane crossings.

Parameters:

| Parameter | Meaning |
| --- | --- |
| `trajectory` | State history with shape `(timesteps,)` or `(timesteps, dimension)`. |
| `normal` | Hyperplane normal with length equal to `dimension`. |
| `offset` | Hyperplane offset in `normal . state - offset = 0`. |
| `direction` | One of `positive`, `negative`, or `both`. |

Return:

- a `PoincareResult` containing interpolated crossing points and fractional
  crossing times.

If the normal vector has zero magnitude, the function returns an empty valid
result.
This avoids division by zero and makes the degenerate section explicit.

## Generic section example

```python
import numpy as np

from scpn_phase_orchestrator.monitor.poincare import poincare_section

t = np.linspace(0.0, 6.0 * np.pi, 3000)
trajectory = np.column_stack([np.sin(t), np.cos(t)])
result = poincare_section(
    trajectory,
    normal=np.array([1.0, 0.0]),
    offset=0.0,
    direction="positive",
)

assert result.crossings.shape[1] == 2
assert result.mean_return_time >= 0.0
```

## return_times

`return_times(trajectory, normal, offset=0.0)` is a convenience wrapper around
`poincare_section(..., direction="positive")`.
It returns only the return-time array from the resulting `PoincareResult`.

Use it when a caller needs recurrence spacing but not crossing states.
Use the full `poincare_section` result when crossing coordinates or crossing
times are required for diagnostics.

## phase_poincare

`phase_poincare` computes section crossings for phase-oscillator histories.
It detects when `phases[:, oscillator_idx]` crosses `section_phase` modulo
`2*pi` and returns the full phase vector at each crossing.

Parameters:

| Parameter | Meaning |
| --- | --- |
| `phases` | Phase history with shape `(timesteps,)` or `(timesteps, oscillators)`. |
| `oscillator_idx` | Oscillator used to define the section. |
| `section_phase` | Target phase, in radians. |

Return:

- a `PoincareResult` whose crossing dimension equals the oscillator count.

## Phase section example

```python
import numpy as np

from scpn_phase_orchestrator.monitor.poincare import phase_poincare

steps = 500
phases = np.zeros((steps, 2), dtype=np.float64)
omegas = np.array([0.1, 0.2], dtype=np.float64)
for idx in range(1, steps):
    phases[idx] = phases[idx - 1] + omegas

result = phase_poincare(phases, oscillator_idx=0, section_phase=0.0)

assert result.crossings.shape[1] == 2
```

## Engine pipeline wiring

The monitor consumes solver output directly.
The dedicated pipeline test advances `UPDEEngine`, stores the phase trajectory,
and sends that trajectory to `phase_poincare`.

The pipeline shape is:

1. `UPDEEngine.step` advances phases,
2. phase vectors are copied into a trajectory array,
3. `phase_poincare` validates the trajectory,
4. crossings and return times are computed,
5. downstream reports inspect `PoincareResult`.

This is a real monitor pipeline boundary, not an import-only smoke test.

## Direct accelerator boundary contract

The direct Go, Julia, and Mojo wrappers validate before loading optional
runtimes.
The validation layer is shared through
`experimental.accelerators.monitor._poincare_validation`.

Section backend inputs must satisfy:

- flattened trajectory buffer is a real finite `float64` vector,
- no boolean aliases are present in the trajectory buffer,
- no complex aliases are present in the trajectory buffer,
- `t` is a non-boolean positive integer,
- `d` is a non-boolean positive integer,
- flattened trajectory length exactly equals `t*d`,
- normal vector is real finite `float64`,
- normal vector length equals `d`,
- `offset` is a finite real scalar,
- `direction_id` is one of `0`, `1`, or `2`.

Phase backend inputs must satisfy:

- flattened phase buffer is a real finite `float64` vector,
- no boolean aliases are present in the phase buffer,
- no complex aliases are present in the phase buffer,
- `t` is a non-boolean positive integer,
- `n` is a non-boolean positive integer,
- flattened phase length exactly equals `t*n`,
- `oscillator_idx` is in `[0, n)`,
- `section_phase` is a finite real scalar.

These checks keep Python, Go, Julia, and Mojo surfaces aligned.
Invalid physics states fail in Python before optional runtime loading or FFI
marshalling.

## Backend output validation

Direct backend return payloads are validated before they are returned to the
public assembly boundary.

Output contracts:

- crossing buffers are finite real `float64` vectors,
- crossing buffer length is exactly `t*dim`,
- time buffers are finite real `float64` vectors,
- time buffer length is exactly `t`,
- crossing count is a non-boolean integer,
- crossing count is non-negative,
- crossing count does not exceed the `t - 1` sampled intervals,
- active crossing times stay inside sampled interval bounds,
- active crossing times are strictly increasing.

Malformed backend payloads fail closed.
The public section functions catch backend failure and fall back to Python for
validated inputs.

## Mojo text parsing contract

The Mojo direct backend communicates through subprocess text output.
The parser preserves raw stdout line cardinality, including blank records.
It requires an explicit crossing-count header.
It rejects crossing counts that exceed the `t - 1` sampled intervals.
Malformed Mojo text output is converted into deterministic `ValueError`
failures instead of leaking index or conversion errors from parser internals.

## Backend dispatch contract

`ACTIVE_BACKEND` is the first resolved backend in `AVAILABLE_BACKENDS`.
`AVAILABLE_BACKENDS` always contains `python`.
Backend loaders are cached after successful loading.
If a loader fails, dispatch continues to the next available backend.
If no optional backend is usable, dispatch returns `None` and the Python path is
used.

The dispatch tests cover:

- Python fallback when a loader fails,
- backend loader caching,
- active backend consistency,
- non-empty available backend list.

## Direction identifiers

Direct backends receive integer direction IDs:

| Public direction | Backend ID |
| --- | --- |
| `positive` | `0` |
| `negative` | `1` |
| `both` | `2` |

The public function validates direction strings before any backend call.
The direct backend validation rejects IDs outside `0`, `1`, and `2`.

## Numerical interpretation

Return times are expressed in sample-index units.
If samples are separated by timestep `dt`, physical return times are
`result.return_times * dt`.

For a constant-frequency phase oscillator with angular velocity `omega` and
sample spacing `dt`, the expected return time in samples is approximately:

```text
2*pi / (omega * dt)
```

The algorithmic tests check this relation with tolerance for discrete sampling
and endpoint effects.

## Empty and degenerate cases

No crossings is a valid result.
The returned arrays are empty, and both summary statistics are `0.0`.

A zero normal vector for `poincare_section` is treated as a degenerate section
and returns no crossings.
A short phase trajectory may also return no crossings.
These cases are not failures; they are valid recurrence evidence showing that
the selected section was not crossed in the sampled interval.

## Error semantics

The public monitor raises `ValueError` for invalid inputs and invalid result
records.
Optional direct backends also raise `ValueError` for validation failures before
runtime loading.

Backend runtime failures are not exposed to callers when the Python fallback can
produce a valid result from already validated input.
This preserves the monitor result contract while allowing optional acceleration
to be deployed opportunistically.

## Polyglot and benchmark notes

The phase-section interpolation fraction is the wrapped-step boundary fraction
`(2*pi - shifted[i]) / ((2*pi - shifted[i]) + shifted[i + 1])` across all five
backends (Python, Rust, Mojo, Julia, Go). An earlier
`shifted[i] / (shifted[i] - shifted[i + 1] + 2*pi)` form drifted the crossing
fraction toward the step midpoint, leaving the interpolated crossing roughly
`0.09 rad` off the section; the corrected fraction recovers `section_phase`
exactly.

The polyglot parity gate `benchmark_poincare_polyglot_parity_gate`
(`benchmarks/poincare_benchmark.py`, wired into
`benchmarks/reference_suite.py` as `poincare_polyglot`) times every available
backend, verifies bit-close parity of the generic and phase crossing points,
times, and counts against the Python reference, and enforces the geometric
contracts: section crossings lie on the hyperplane, phase crossings recover the
section phase, both crossing-time sequences are strictly increasing, both
produce at least one crossing, and every backend reports the same crossing
count. Run it with
`PYTHONPATH=.:src python benchmarks/poincare_benchmark.py --parity-gate`.

Future backend changes must preserve:

- pre-runtime input validation,
- output payload validation,
- crossing-count bounds,
- strictly increasing crossing times,
- direction ID semantics,
- phase modulo semantics,
- Python fallback behaviour,
- `PoincareResult` invariants.

## Behavioural tests guarding this reference

The dedicated monitor test surfaces are:

- `tests/test_poincare.py`, covering public input validation, result-record
  validation, backend dispatch, fallback, and UPDE pipeline wiring,
- `tests/test_poincare_algorithm.py`, covering algorithmic crossing properties,
  direction filtering, interpolation accuracy, phase wraparound, constant
  frequency return times, empty trajectory safety, and invalid direction
  rejection,
- `tests/test_poincare_backends.py`, covering direct Go, Julia, and Mojo input
  validation, output validation, type-hint parity, Mojo text parsing, and
  backend parity expectations,
- `tests/test_poincare_stability.py`, covering stability-oriented recurrence
  behaviour,
- `tests/test_prop_embedding_poincare.py`, covering property-style embedding
  and Poincare monitor invariants,
- `tests/test_poincare_benchmark.py`, covering the polyglot parity-gate
  contracts, control validation, and per-backend wall-clock timing.

The reference depth guard is `tests/test_reference_api_monitor_poincare.py`.
It prevents this page from regressing to a shallow overview and checks that the
public monitor, backend, validation, and recurrence contracts remain documented.

## Operational checklist

- Confirm trajectory samples are finite before section extraction.
- Confirm trajectory samples do not contain boolean aliases.
- Confirm trajectory samples do not contain complex aliases.
- Confirm trajectory rank is one or two.
- Confirm normal length matches trajectory dimension.
- Confirm normal values are finite real numbers.
- Confirm offset is a finite real scalar.
- Confirm direction is positive, negative, or both.
- Confirm phase histories are finite before phase section extraction.
- Confirm phase histories do not contain boolean aliases.
- Confirm phase histories do not contain complex aliases.
- Confirm oscillator_idx is an in-range integer.
- Confirm section_phase is a finite real scalar.
- Use poincare_section for generic hyperplane recurrence.
- Use phase_poincare for oscillator phase wraparound recurrence.
- Use return_times only when crossing coordinates are not needed.
- Interpret return times as sample-index intervals.
- Multiply return times by dt when physical time is required.
- Treat no crossings as valid evidence, not an exception.
- Treat zero normal vectors as degenerate empty sections.
- Review direction filtering before comparing crossing counts.
- Review endpoint effects before asserting exact crossing counts.
- Review interpolation tolerance for sampled sinusoidal trajectories.
- Review phase wraparound near 2*pi boundaries.
- Review constant-frequency return times against expected periods.
- Review quasi-periodic regimes with return-time spread.
- Review chaotic regimes with crossing-map dispersion.
- Keep PoincareResult crossing times strictly increasing.
- Keep PoincareResult return times matched to crossing-time differences.
- Keep PoincareResult summary statistics derived from return_times.
- Reject hand-assembled result records with inconsistent statistics.
- Validate direct backend inputs before runtime loading.
- Validate direct backend outputs before public assembly.
- Keep direct backend crossing count bounded by sampled intervals.
- Keep direct backend crossing times inside sampled intervals.
- Keep direct backend crossing times strictly increasing.
- Preserve Mojo stdout cardinality during parser validation.
- Reject missing Mojo crossing-count headers.
- Reject malformed Mojo crossing-count headers.
- Reject Mojo crossing counts exceeding sampled intervals.
- Keep ACTIVE_BACKEND aligned with AVAILABLE_BACKENDS.
- Keep python in AVAILABLE_BACKENDS.
- Fall back to Python when optional backend loading fails.
- Fall back to Python when optional backend execution fails.
- Do not fall back for invalid public input.
- Do not load optional runtimes before validation failures are raised.
- Do not expose backend parser internals as public errors.
- Do not mutate solver trajectories during monitoring.
- Do not tune coupling matrices from this monitor.
- Do not actuate from this monitor.
- Store recurrence evidence with the replay or report when needed.
- Use UPDE pipeline tests to guard monitor integration.
- Use algorithmic tests to guard interpolation semantics.
- Use backend tests to guard direct wrapper parity.
- Use property tests to guard shape and invariant stability.
- Avoid broad mixed-module bucket tests for monitor coverage.
- Avoid import-only tests for Poincare monitor evidence.
- Avoid mock-only tests that bypass public monitor behaviour.
- Document any future backend change before exposing it.
- Document any future result-field change before exposing it.
- Document any future direction convention change before exposing it.
- Document any future time-unit change before exposing it.
- Document benchmark impact only after measurement.
- Document polyglot compatibility only after parity evidence.
- Keep tests/test_poincare.py aligned with public errors.
- Keep tests/test_poincare_algorithm.py aligned with mathematical semantics.
- Keep tests/test_poincare_backends.py aligned with direct wrappers.
- Keep tests/test_poincare_stability.py aligned with recurrence stability.
- Keep tests/test_prop_embedding_poincare.py aligned with property invariants.
- Keep API docs aligned with monitor facade exports.
- Keep changelog entries factual when this page changes.
- Keep roadmap entries tied to documented evidence.
- Keep closure evidence module-specific.
- Review whether optional Rust acceleration changes return shapes.
- Review whether optional Go acceleration changes return shapes.
- Review whether optional Julia acceleration changes return shapes.
- Review whether optional Mojo acceleration changes return shapes.
- Review whether backend output validation catches malformed crossing buffers.
- Review whether backend output validation catches malformed time buffers.
- Review whether backend output validation catches boolean crossing counts.
- Review whether backend output validation catches negative crossing counts.
- Review whether backend output validation catches too many crossings.
- Review whether backend output validation catches non-monotone times.
- Review whether public validation catches object-complex arrays.
- Review whether public validation catches object-boolean arrays.
- Review whether public validation catches non-numeric nested lists.
- Review whether public validation catches non-finite offsets.
- Review whether public validation catches non-finite section phases.
- Review whether public validation catches invalid oscillator indexes.
- Review whether return-time reports state sample units clearly.
- Review whether reports convert sample units when dt is known.
- Review whether recurrence maps include crossing coordinates when needed.
- Review whether recurrence maps include crossing times when needed.
- Review whether periodic examples use deterministic trajectories.
- Review whether chaotic examples use documented seeds.
- Review whether monitor examples avoid private datasets.
- Review whether notebooks import public monitor symbols.
- Review whether dashboards label return-time units.
- Review whether Studio panels avoid treating monitor output as actuation.
- Review whether future docs keep observational boundaries visible.
- Review whether future docs keep accelerator boundaries visible.
- Review whether future docs keep mathematical limits visible.
- Review whether future docs keep fallback semantics visible.
- Review whether future docs keep validation semantics visible.
- Review whether future docs keep result invariants visible.
- Review whether future docs keep backend parity evidence visible.
- Review whether future docs keep benchmark limitations visible.
- Review whether future docs keep polyglot notes visible.
- Review whether future docs keep examples runnable.
- Review whether future docs keep examples deterministic.
- Review whether future docs guard against shallow overview regression.
- Review whether future docs require new reference guard phrases.
- Review whether future docs require changelog updates.
- Review whether future docs require roadmap updates.
- Review whether future docs require public roadmap updates.
- Review whether future monitor changes require CI matrix updates.
- Review whether future monitor changes require optional dependency docs.
- Review whether future monitor changes require direct backend package notes.
- Review whether future monitor changes require Rust parity notes.
- Review whether future monitor changes require Go parity notes.
- Review whether future monitor changes require Julia parity notes.
- Review whether future monitor changes require Mojo parity notes.
- Review whether future monitor changes preserve Python fallback.
- Review whether future monitor changes preserve fail-closed validation.
- Review whether future monitor changes preserve recurrence invariants.
- Review whether future monitor changes preserve engine pipeline wiring.
- Review whether future monitor changes preserve public result compatibility.
- Review whether future monitor changes preserve public import compatibility.
- Review whether future monitor changes preserve type-hint compatibility.
- Review whether future monitor changes preserve domain-report compatibility.
- Review whether future monitor changes preserve replay-report compatibility.
- Review whether future monitor changes preserve documentation completeness.

## Release-note summary

The Poincare monitor API provides validated recurrence evidence for generic
state trajectories and phase-oscillator histories. It documents the public
Python monitor contract, `PoincareResult` invariants, sample-index return-time
semantics, optional Rust/Mojo/Julia/Go backend validation, backend-output
fail-closed behaviour, and Python fallback semantics. The API remains
observational and does not mutate, tune, or actuate solver state.

::: scpn_phase_orchestrator.monitor.poincare
