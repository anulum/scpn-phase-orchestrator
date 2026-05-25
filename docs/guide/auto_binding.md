<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Auto-binding guide -->

# Auto-Binding Proposals

The auto-binding prototype converts raw source families into reviewable
`binding_spec.yaml` proposals. It never overwrites a domainpack and never marks
the proposal as trusted without validation.

Supported inputs:

- Time-series CSV with a header, one optional time column, and one or more
  numeric signal columns. If `--sample-rate-hz` is not supplied, a strictly
  increasing regular `time`, `timestamp`, or `t` column is used to infer the
  sampling rate.
- Event-log JSON arrays containing event records.
- Graph JSON containing `nodes` and optional `edges`.

Each proposal returns a `StudioProjectState` with:

- deterministic source hash and counts,
- proposed binding YAML,
- inferred channels,
- review-only extractor-parameter proposals,
- review-only initial `K` template binding for time-series sources,
- confidence factors,
- provenance,
- deterministic discovery evidence for time-series sources,
- binding-validator diagnostics.

Example:

```python
from scpn_phase_orchestrator.autotune.binding_proposal import (
    propose_binding_from_time_series_csv,
)

state = propose_binding_from_time_series_csv(
    "t,a,b\n0,0.1,0.4\n1,0.2,0.5\n",
    sample_rate_hz=None,
    project_name="sensor_review",
)

print(state.binding.yaml_text)
print(state.binding.validation_errors)
print(state.binding.provenance["discovery_evidence"])
```

For time-series CSV imports, provenance includes derivative sparse-regression
evidence, phase-aware Kuramoto SINDy evidence when columns are phase-like, a
residual-scored SINDy library selection record, a correlation graph,
lagged directed graph inference record, connected-component clusters, and the
sampling-rate inference path. Non-phase tables carry an explicit phase-SINDy
skipped status instead of a fitted phase model. These records are audit evidence
for operator review; they do not enable automatic actuation.

Time-series proposals also bind each inferred oscillator family to concrete
extractor parameters in the generated YAML. The family `config` records the
source column, column index, sampling rate, sample period, finite sample count,
basic distribution statistics, and the review-only status. The same records are
available under `binding.provenance["extractor_parameter_proposals"]` for JSON
audit export.

Generated binding YAML is emitted through the structured YAML serialiser rather
than string interpolation. Raw source column names are preserved as scalar
values and safely quoted when they contain `:`, `{`, `[`, `#`, newlines, or
other YAML-significant characters.

The generated binding includes a coupling template named `auto_initial_k`.
Its matrix is oriented as `target_by_source`, has a zero diagonal, is scaled
from the combined lagged-graph, phase-SINDy, and correlation evidence, and is
mirrored into validator-accepted `cross_channel_couplings`. This is an initial
operator proposal only: downstream code must still validate the domainpack and
explicitly accept the proposed matrix before using it for runtime actuation.

For oscillator families with phase-like channels, the generated
`auto_initial_k` matrix can be reviewed with the JAX Spectral Alignment
Function before simulation or actuation:

```python
from scpn_phase_orchestrator.nn import saf_loss, saf_order_parameter

r_est = saf_order_parameter(auto_initial_k, omegas, solver="auto")
audit_loss = saf_loss(auto_initial_k, omegas, budget=10.0, solver="cg")
```

Use this as a review signal, not as automatic approval. SAF answers whether the
proposed topology is spectrally aligned with the frequency field; it does not
prove that the raw source data are valid, that the inferred channels are
causal, or that a domainpack is safe for runtime actuation.

The reference benchmark suite includes a deterministic
`auto_binding_synthetic_quality` fixture set. It measures extractor coverage,
binding-validator acceptance, expected initial-K support recall, generated edge
count, wall-clock time, and fixtures per second. Treat that benchmark as a
reproducible synthetic regression surface, not as a live-dataset acceptance
claim.

The output is suitable for human review in SPO Studio or for tests that need a
deterministic proposal package.

Symbolic binding proposals carry the same review-only boundary. Free-text
intent is sanitised before compilation, generated output is checked against the
expected binding schema, and the result includes review metadata. Treat the
proposal as an auditable draft until an operator approves and validates the
domainpack through the normal binding loader.
