<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SCPN Phase Orchestrator — causal coupling inference API
-->

# Coupling — Causal Inference

`scpn_phase_orchestrator.coupling.infer` packages transfer-entropy-based
coupling discovery as `auto-coupling-estimation`.

The estimator accepts oscillator phase time series with shape
`(oscillators, timesteps)` and returns a directed source-to-target matrix:
`knm[i, j]` is the inferred influence from oscillator `i` to oscillator `j`.
Standard UPDE matrices use target-by-source orientation, so
`CouplingInferenceResult.to_upde_knm()` returns the transposed matrix for direct
engine input.

The current production backend is transfer entropy over binned phase states.
`granger` and `notears` are reserved method names and raise
`NotImplementedError` until those estimators have benchmarked implementations.
The boundary rejects boolean aliases, complex values, non-finite samples, and
non-2-D phase series before inference. Backend transfer-entropy matrices must be
finite, non-negative, correctly shaped, and zero on the diagonal; invalid
backend scores fail closed instead of being silently normalised.

## Primary use cases

- Review-only oscillator discovery from phase traces produced by a domainpack,
  replay, sensor extractor, or notebook experiment.
- Data-driven proposal of an initial directed coupling graph before a human
  operator accepts or edits a binding specification.
- Directional dependency evidence for phase models where oscillator `i` may
  drive oscillator `j` with a lagged information flow.
- Reproducible audit records for auto-binding and coupling-review pipelines.
- CLI-driven inspection of CSV or `.npy` phase matrices before a full UPDE run.
- Regression tests for transfer-entropy orientation, thresholding, and diagonal
  safety semantics.

## Safety boundary

The module is an inference and review surface.
It does not actuate hardware.
It does not mutate a domainpack.
It does not write a binding file.
It does not claim that a discovered edge is physically causal by itself.

The caller must still decide:

- whether the source trace is representative,
- whether the sample rate is meaningful for the domain,
- whether a proposed edge should become an operational coupling,
- whether a threshold is conservative enough for the deployment,
- whether the resulting UPDE matrix needs additional physics constraints,
- whether domain-specific validation accepts the proposal.

The module guarantees that malformed input and malformed backend output fail
closed before a `CouplingInferenceResult` is returned.

## Exported symbols

| Symbol | Kind | Purpose |
| --- | --- | --- |
| `CouplingInferenceConfig` | dataclass | Inference method, binning, threshold, normalisation, and timestep limits. |
| `CouplingInferenceResult` | dataclass | Directed coupling estimate, raw scores, support mask, and audit diagnostics. |
| `auto_coupling_estimation` | function | Packaged review profile using transfer entropy and default thresholds. |
| `infer_coupling_from_timeseries` | function | Full inference entry point for explicit configuration. |
| `InferenceMethod` | type alias | Literal method names: `transfer_entropy`, `granger`, `notears`. |
| `NormalisationMode` | type alias | Literal normalisation names: `max`, `none`. |
| `FloatArray` | type alias | `numpy.float64` array type for real-valued matrices. |
| `BoolArray` | type alias | boolean array type for directed support masks. |

## Data orientation

Input phases use oscillator-by-time orientation.
The expected shape is `(oscillators, timesteps)`.
Rows are oscillator trajectories.
Columns are observation steps.

For CLI input, `--orientation time-by-oscillator` transposes a CSV matrix whose
rows are time samples and columns are oscillator channels.
Use `--orientation oscillator-by-time` when the source already follows the API
shape.

The result orientation is source-to-target.
`result.knm[i, j]` means source oscillator `i` has inferred directional
influence on target oscillator `j`.

UPDE engine matrices commonly use target-by-source orientation.
`result.to_upde_knm()` returns `result.knm.T` as a contiguous `float64` matrix.
That conversion is explicit so review tools cannot confuse inference direction
with runtime coupling direction.

## CouplingInferenceConfig

`CouplingInferenceConfig` is immutable and uses dataclass slots.
It collects the inference contract without starting inference at construction
time.
Validation occurs when the config is passed into
`infer_coupling_from_timeseries`.

Fields:

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `method` | `InferenceMethod` | `transfer_entropy` | Inference backend selector. |
| `n_bins` | `int` | `8` | Number of phase-state bins for transfer entropy. |
| `threshold_quantile` | `float | None` | `0.75` | Quantile over positive off-diagonal scores. |
| `threshold_absolute` | `float | None` | `None` | Absolute support threshold; takes precedence when present. |
| `normalisation` | `NormalisationMode` | `max` | Coupling weight scaling mode. |
| `min_timesteps` | `int` | `4` | Minimum accepted number of observations. |

### Method validation

Accepted method names are:

- `transfer_entropy`, implemented today,
- `granger`, reserved and fail-closed,
- `notears`, reserved and fail-closed.

Reserved method names are intentionally accepted by the config validator so
callers can request them and receive a clear `NotImplementedError` rather than
falling back to a weaker estimator.
Unknown method names raise `ValueError`.

### Bin validation

`n_bins` must be an integer greater than or equal to `2`.
Boolean aliases are rejected even though `bool` is a subclass of `int` in
Python.
A bin count below `2` is invalid because transfer entropy over one state cannot
represent directional state transitions.

### Threshold validation

`threshold_quantile` may be `None` or a finite real value in `[0, 1]`.
`threshold_absolute` may be `None` or a finite non-negative real value.
Boolean aliases are rejected for both threshold fields.
`threshold_absolute takes precedence` when provided.

When neither threshold is available, the support threshold is `0.0` and support
is defined by strictly positive off-diagonal scores.

### Normalisation validation

Accepted normalisation modes are:

- `max`, divide supported scores by the largest supported score,
- `none`, preserve supported transfer-entropy scores as raw weights.

Unknown normalisation names raise `ValueError`.
If `max` is selected and no supported score is positive, the resulting `knm` is
a zero matrix.

### Timestep validation

`min_timesteps` must be an integer at least `4`.
Boolean aliases are rejected.
The default is conservative for unit tests and small examples; real domain
experiments normally require substantially longer traces before a coupling
proposal should be trusted.

## Phase-series validation

`infer_coupling_from_timeseries` validates `phase_series` before backend
execution.

The boundary rejects:

- boolean aliases anywhere in the input,
- complex values,
- non-numeric values,
- non-finite values,
- arrays that are not two-dimensional,
- fewer than two oscillators,
- fewer than `min_timesteps` observations.

The accepted phase series is copied into a contiguous `float64` array.
This prevents later mutations of the caller's input from changing the inference
record after validation.

## Transfer-entropy backend contract

The implemented backend delegates to
`scpn_phase_orchestrator.monitor.transfer_entropy.transfer_entropy_matrix`.
The input to that backend is the validated oscillator-by-time `float64` phase
series.
The backend receives the configured `n_bins`.

The returned score matrix must satisfy all of these invariants:

- shape is `(n_oscillators, n_oscillators)`,
- every value is finite,
- every value is non-negative within numerical tolerance,
- every diagonal value is zero within numerical tolerance.

A malformed backend matrix raises `RuntimeError`.
The inference function does not repair invalid backend structure.
It only clips tiny negative numerical noise after the explicit negative-score
boundary has passed.

## Thresholding model

Thresholding is applied to positive off-diagonal scores.
The diagonal is excluded from threshold estimation.
A score must be strictly positive and greater than or equal to the selected
threshold to become support.
The diagonal is always forced to `False` in `support_mask`.

Threshold selection order:

1. use `threshold_absolute` when it is not `None`,
2. otherwise use `threshold_quantile` over positive off-diagonal scores,
3. otherwise use `0.0`.

This makes the support rule deterministic for the same score matrix and config.

## Normalisation model

The support mask is applied before normalisation.
Unsupported edges are set to `0.0`.
The diagonal is set to `0.0`.

With `normalisation="max"`, supported scores are divided by the largest
supported score.
The resulting non-zero weights lie in `(0, 1]`.
With `normalisation="none"`, supported scores remain in transfer-entropy score
units.

The normalisation step does not create new support edges.
It only scales already-supported scores.

## CouplingInferenceResult

`CouplingInferenceResult` is the immutable output record.

Fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `knm` | `FloatArray` | Source-to-target directed coupling weights. |
| `score_matrix` | `FloatArray` | Raw validated transfer-entropy scores. |
| `support_mask` | `BoolArray` | Directed off-diagonal support decisions. |
| `method` | `str` | Resolved inference method. |
| `score_kind` | `str` | Score semantics, currently `transfer_entropy`. |
| `n_bins` | `int` | Transfer-entropy bin count used for the run. |
| `threshold` | `float` | Effective support threshold. |
| `normalisation` | `str` | Effective normalisation mode. |
| `shape` | `tuple[int, int]` | Validated `(oscillators, timesteps)` input shape. |
| `package` | `str` | Stable package name, `auto-coupling-estimation`. |
| `orientation` | `str` | Stable orientation label, `source_to_target`. |
| `active_backend` | `str` | Active transfer-entropy backend label. |
| `available_backends` | `tuple[str, ...]` | Transfer-entropy backend labels visible at runtime. |

The result object is intended for review, audit, and explicit handoff.
It does not imply that a runtime binding has been accepted.

## edge_count property

`edge_count` returns the number of non-zero directed off-diagonal support edges.
It is computed from `support_mask`.
It does not count diagonal entries.
It does not count unsupported non-zero raw scores.

Use this value to check whether a proposal is empty, sparse, or unexpectedly
full before presenting it to an operator.

## density property

`density` reports directed graph density over possible off-diagonal edges.
For `n` oscillators, the possible directed off-diagonal count is `n * (n - 1)`.
A one-oscillator result cannot be produced by the public validator, but the
property still returns `0.0` if the possible edge count is zero.

Density is a review diagnostic, not a physics proof.
Domain-specific validation should still inspect which edges were proposed and
whether they are plausible for the system.

## to_upde_knm

`to_upde_knm()` returns a contiguous `float64` matrix in UPDE
,target-by-source orientation.
It is the explicit bridge from inference convention to engine convention.

Example:

```python
upde_k = result.to_upde_knm()
assert upde_k.shape == result.knm.shape
```

If `result.knm[0, 1]` is non-zero, then `upde_k[1, 0]` is non-zero.
That means source oscillator `0` pulls target oscillator `1` in UPDE notation.

## to_audit_record

`to_audit_record()` returns a JSON-safe dictionary.
It includes:

- package name,
- method,
- score kind,
- orientation,
- input shape,
- bin count,
- threshold,
- normalisation,
- active backend,
- available backends,
- `knm` as nested lists,
- `score_matrix` as nested lists,
- `support_mask` as nested lists,
- diagnostics containing `edge_count`, `density`, `score_min`, and `score_max`.

The audit record can be serialized with `json.dumps` without custom encoders.
It is suitable for replay metadata, CLI JSON output, and review reports.

## auto_coupling_estimation

`auto_coupling_estimation(phase_series, *, n_bins=8, threshold_quantile=0.75,
threshold_absolute=None, normalisation="max")` is the packaged convenience
profile.

It always uses:

- `method="transfer_entropy"`,
- the supplied bin count,
- the supplied threshold parameters,
- the supplied normalisation mode,
- the default minimum timestep contract from `CouplingInferenceConfig`.

Use this function when a caller wants the standard review profile and does not
need to construct a config object explicitly.

## infer_coupling_from_timeseries

`infer_coupling_from_timeseries(phase_series, *, config=None)` is the explicit
entry point.
When `config` is omitted, it uses `CouplingInferenceConfig()`.
When `config` is provided, every config field is validated before phase data is
accepted.

The function returns a `CouplingInferenceResult` only after all input,
backend-output, thresholding, and normalisation boundaries have passed.

## Python example

```python
import numpy as np

from scpn_phase_orchestrator.coupling import (
    CouplingInferenceConfig,
    infer_coupling_from_timeseries,
)

time = np.arange(240, dtype=np.float64)
driver = 0.13 * time
follower = 0.72 * np.roll(driver, 1) + 0.03
independent = 0.09 * time + 1.7
phases = np.vstack([driver, follower, independent]) % (2.0 * np.pi)

result = infer_coupling_from_timeseries(
    phases,
    config=CouplingInferenceConfig(
        method="transfer_entropy",
        n_bins=8,
        threshold_quantile=0.65,
        normalisation="max",
    ),
)

assert result.orientation == "source_to_target"
assert result.knm.shape == (3, 3)
assert result.to_upde_knm().shape == (3, 3)
```

## CLI

```bash
spo auto-coupling-estimation phases.csv \
  --orientation time-by-oscillator \
  --n-bins 8 \
  --threshold-quantile 0.75 \
  --json-out
```

CSV and `.npy` sources are supported.
Use `--orientation oscillator-by-time` when rows are oscillators and columns are
timesteps.
Use `--orientation time-by-oscillator` when rows are observations and columns are
oscillators.

The CLI emits either a compact text summary or the JSON-safe audit record.
The JSON output uses the same `to_audit_record()` contract as the Python API.

## CLI review workflow

A conservative review workflow is:

1. export phase traces from the replay or extractor pipeline,
2. run `spo auto-coupling-estimation` with explicit orientation,
3. inspect `edge_count`, `density`, and the support graph,
4. compare proposed directionality with domain expectations,
5. convert to UPDE orientation only after review,
6. feed the candidate matrix into a review-only binding proposal,
7. validate the binding with domain-specific thresholds,
8. keep the audit record with the replay artefacts.

This preserves a human review boundary between inferred statistical support and
operational coupling.

## Failure boundaries

The public API raises:

- `TypeError` for invalid integer-like config types such as boolean aliases,
- `ValueError` for malformed config values and malformed phase series,
- `NotImplementedError` for reserved but unavailable methods,
- `RuntimeError` for malformed backend score matrices.

Failure messages identify the violated boundary.
They are not intended to expose source paths or internal stack details.

## Physics and mathematics notes

Transfer entropy estimates directional information flow from lagged state
transitions.
In this module it is used as a coupling proposal signal, not as a complete proof
of mechanistic causality.

Important interpretation limits:

- common drivers can create apparent directed support,
- undersampled traces can hide or invert directionality,
- phase wrapping can change bin occupancy,
- bin count controls resolution and estimator variance,
- thresholds change support sparsity,
- normalisation changes coupling weight scale but not support membership,
- domain constraints may require symmetry, boundedness, or topology masks after
  inference.

For scientific use, treat the result as an auditable candidate that must be
validated against known physics, controlled interventions, replay error, or
held-out trajectories.

## Backend parity notes

The transfer-entropy implementation is the only production backend for this
module today.
`granger` and `notears` are reserved names because they are likely future
extensions, but they must not silently alias to transfer entropy.
A caller requesting either reserved method receives a fail-closed
`NotImplementedError`.

There is no Go, Julia, Mojo, or Rust counterpart for this inference facade in
this repository at the time of this reference update.
The runtime evidence surface is the Python API plus CLI wrapper.
Future counterparts must preserve the same input validation, orientation,
thresholding, audit-record, and fail-closed reserved-method semantics before
being treated as compatible.

## Benchmark and evidence notes

This documentation update does not change production inference behaviour or
benchmark results.
The existing benchmark-relevant claim is limited to deterministic behavioural
coverage through the dedicated module tests.
If a new backend is implemented later, it must include measured quality and
runtime evidence before being advertised as a production estimator.

## Behavioural tests guarding this reference

The dedicated module test surface is `tests/test_coupling_infer.py`.
It checks:

- directed support recovery from a deterministic driver/follower trace,
- JSON-safe audit records,
- invalid phase-series rejection,
- boolean alias rejection,
- complex alias rejection,
- short-trace rejection,
- reserved `notears` fail-closed behaviour,
- CLI JSON output,
- backend negative-score rejection,
- backend self-score rejection,
- invalid threshold config rejection,
- invalid timestep config rejection.

The reference depth guard is `tests/test_reference_api_coupling_infer.py`.
It prevents this page from regressing to a shallow index and checks that the
public contract terms remain documented.

## Operational checklist

- Confirm the phase matrix orientation before inference.
- Use oscillator-by-time orientation for direct Python calls.
- Use time-by-oscillator only when the source table stores time on rows.
- Reject traces with fewer than two oscillators.
- Reject traces shorter than the configured minimum timestep count.
- Reject boolean phase aliases before float coercion.
- Reject complex phase aliases before float coercion.
- Reject non-finite phase samples before backend execution.
- Keep n_bins at least two.
- Avoid treating one-bin transfer entropy as meaningful.
- Review bin count against sample count.
- Review threshold_quantile against expected graph sparsity.
- Use threshold_absolute when a domain threshold has been calibrated.
- Do not combine threshold_absolute with hidden post-filtering.
- Record the effective threshold in audit evidence.
- Record the normalisation mode in audit evidence.
- Record active and available backend labels in audit evidence.
- Treat source-to-target orientation as the review convention.
- Use to_upde_knm for target-by-source engine input.
- Do not pass result.knm directly to a UPDE engine without checking orientation.
- Do not treat edge_count as proof of physical causality.
- Do not treat density as proof of model adequacy.
- Inspect support_mask before using normalised weights.
- Inspect score_matrix before accepting sparse support.
- Keep the diagonal zero after every transformation.
- Reject backend matrices with non-zero self scores.
- Reject backend matrices with negative scores.
- Reject backend matrices with non-finite scores.
- Reject backend matrices with the wrong shape.
- Keep malformed backend output as RuntimeError rather than silent repair.
- Use JSON audit output for reproducible CLI reviews.
- Store the audit record with replay artefacts when coupling proposals matter.
- Keep reserved granger and notears methods fail-closed.
- Do not silently downgrade reserved methods to transfer entropy.
- Add benchmark evidence before implementing a new estimator backend.
- Add module-specific tests before implementing a new estimator backend.
- Add orientation tests before implementing a polyglot counterpart.
- Add audit-record compatibility tests before implementing a polyglot counterpart.
- Add malformed-input tests before implementing a polyglot counterpart.
- Do not use mock-only tests as estimator evidence.
- Prefer deterministic synthetic traces for regression tests.
- Prefer held-out domain traces for validation studies.
- Keep auto-binding use review-only until a binding validator accepts the proposal.
- Keep actuation disabled during inference review.
- Document source data provenance in downstream review records.
- Document sample-rate assumptions in downstream review records.
- Document phase extractor settings in downstream review records.
- Document any topology mask applied after inference.
- Document any symmetry constraint applied after inference.
- Document any boundedness constraint applied after inference.
- Document any manual edge deletion after inference.
- Document any manual edge addition after inference.
- Compare proposed driver/follower direction with domain knowledge.
- Compare support against replay residuals before runtime use.
- Compare support against known interventions when available.
- Use domain-specific validation thresholds before accepting auto_initial_k.
- Avoid claiming causality from transfer entropy alone.
- Avoid claiming production readiness for reserved estimators.
- Avoid claiming hardware impact from this review-only module.
- Avoid writing binding files from this module directly.
- Keep CLI input loading separate from inference validation.
- Keep CLI JSON output aligned with to_audit_record.
- Keep text summaries compact and non-authoritative.
- Keep documentation examples explicit about orientation.
- Keep examples small enough to run locally.
- Keep examples free of private datasets.
- Keep examples deterministic.
- Keep future docs aligned with exported symbol names.
- Keep future docs aligned with tests/test_coupling_infer.py.
- Keep future docs aligned with CLI option names.
- Keep future changelog entries factual.
- Keep future roadmap entries tied to evidence.
- Keep future TODO closure evidence module-specific.
- Review score_min and score_max in the audit diagnostics.
- Review whether edge_count changes when n_bins changes.
- Review whether density changes when threshold_quantile changes.
- Review whether normalisation none is needed for calibrated downstream weights.
- Review whether max normalisation is appropriate for exploratory proposals.
- Review whether the trace has enough timesteps for the chosen bin count.
- Review whether phase wrapping is handled before inference.
- Review whether preprocessing introduced boolean masks into phase arrays.
- Review whether preprocessing introduced complex analytic signals instead of phases.
- Review whether preprocessing introduced NaN gaps.
- Review whether interpolation changed lag structure.
- Review whether the same trace segment is used for all oscillators.
- Review whether channels are aligned in time.
- Review whether channel labels are preserved outside this numeric API.
- Review whether downstream binding maps matrix indices to oscillator names.
- Review whether result.shape matches the reviewed source file.
- Review whether active_backend is expected in the runtime environment.
- Review whether available_backends changed after dependency updates.
- Review whether support edges are stable under small threshold changes.
- Review whether support edges are stable under small bin-count changes.
- Review whether support edges are stable across replay windows.
- Review whether support edges are stable across held-out traces.
- Review whether domainpack validators accept the proposed coupling.
- Review whether supervisor safety constraints allow the proposed coupling.
- Review whether audit records are retained for external review.
- Review whether CLI failures are surfaced clearly to operators.
- Review whether future backend changes require docs updates.
- Review whether future backend changes require benchmarks.
- Review whether future backend changes require compatibility notes.
- Review whether future backend changes require migration notes.
- Review whether future backend changes require operator guidance.
- Review whether future CLI changes require reference updates.
- Review whether future auto-binding changes require reference links.
- Review whether future UPDE orientation changes require this page to change.
- Review whether future result fields require audit-record tests.
- Review whether future result fields remain JSON-safe.
- Review whether future config fields reject boolean aliases.
- Review whether future config fields reject non-finite values.
- Review whether future config fields have documented defaults.
- Review whether future methods preserve fail-closed semantics.
- Review whether future methods expose score_kind honestly.
- Review whether future methods document estimator limits.
- Review whether future methods document data requirements.
- Review whether future methods document benchmark evidence.
- Review whether future methods document validation evidence.
- Review whether future methods document reserved-name migration.
- Review whether future examples avoid shallow shape-only claims.
- Review whether future tests would catch plausible regressions.
- Review whether future tests remain module-specific.
- Review whether future tests avoid broad bucket naming.
- Review whether future tests avoid mock-only estimator evidence.
- Review whether future tests validate orientation and diagonal invariants.
- Review whether future tests validate JSON audit stability.
- Review whether future tests validate CLI orientation handling.
- Review whether future tests validate backend failure paths.
- Review whether future tests validate reserved method errors.
- Review whether future tests validate threshold precedence.
- Review whether future tests validate max normalisation.
- Review whether future tests validate raw-score normalisation.
- Review whether future tests validate support_mask diagonal exclusion.
- Review whether future tests validate edge_count and density.
- Review whether future tests validate to_upde_knm transposition.
- Review whether future tests validate audit diagnostics.
- Review whether future docs keep this module review-only.
- Review whether future docs avoid unsupported production claims.
- Review whether future docs keep physics interpretation limits visible.
- Review whether future docs keep mathematics threshold rules visible.
- Review whether future docs keep safety boundaries visible.
- Review whether future docs keep benchmark limitations visible.
- Review whether future docs keep polyglot limitations visible.
- Review whether future docs keep CLI examples aligned with real options.
- Review whether future docs keep Python examples aligned with public imports.
- Review whether future docs keep audit examples aligned with result fields.
- Review whether future docs keep operational checklist actionable.
- Review whether future docs keep closure evidence in the internal TODO.
- Review whether future docs keep public roadmap claims evidence-backed.

## Release-note summary

The coupling inference API converts validated oscillator phase traces into a
reviewable directed source-to-target coupling proposal using transfer entropy.
It preserves explicit orientation conversion for UPDE input, rejects malformed
input and backend scores before returning a result, emits JSON-safe audit
records, and keeps unimplemented estimator names fail-closed until measured
production implementations exist.

::: scpn_phase_orchestrator.coupling.infer
