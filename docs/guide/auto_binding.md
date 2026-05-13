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
connected-component clusters, and the sampling-rate inference path. Non-phase
tables carry an explicit phase-SINDy skipped status instead of a fitted phase
model. These records are audit evidence for operator review; they do not enable
automatic actuation.

The output is suitable for human review in SPO Studio or for tests that need a
deterministic proposal package.
