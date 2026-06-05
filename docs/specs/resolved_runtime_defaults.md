<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Resolved runtime defaults -->

# Resolved Runtime Defaults

## Why resolved defaults are documented

Operators often debug failures in the gap between declared bindings and inferred
execution state. This page closes that gap by listing exact inferences made from
input declarations before runtime.

That makes execution behavior reviewable before a long simulation starts.

This page makes YAML-default behavior explicit by documenting runtime choices
that are inferred from `binding_spec.yaml` and shown via:

- `spo validate <binding_spec>`
- `spo inspect <binding_spec>`
- `spo run <binding_spec>`
- audit header metadata (`binding_summary` and backward-compatible `binding_config`)

## Inferred values

| Runtime field | Inference rule | Source |
| --- | --- | --- |
| `name`/`version`/`safety_tier` | Loaded from binding metadata and echoed unchanged | `name`, `version`, `safety_tier` |
| `engine_mode` | `stuart_landau` when `amplitude:` is present; otherwise `kuramoto` | `binding_spec.amplitude` |
| `sample_period_s`/`control_period_s` | Loaded directly from YAML | `sample_period_s`, `control_period_s` |
| `control_interval_steps` | `max(1, round(control_period_s / sample_period_s))` | `control_period_s`, `sample_period_s` |
| `layer_count` | Number of `layers` entries | `layers` |
| `oscillator_count` | Sum of `oscillator_ids` in all layers | `layers[].oscillator_ids` |
| `unassigned_layer_count` | Number of layers where `family` is unset | `layers[].family` |
| `features.amplitude` | Enabled when amplitude config is declared | `amplitude:` block |
| `features.geometry_prior` | Enabled when geometry prior config is declared | `geometry_prior:` block |
| `features.imprint_model` | Enabled when imprint config is declared | `imprint_model:` block |
| `features.protocol_net` | Enabled when protocol net config is declared | `protocol_net:` block |
| `resolved_extractor_type` | Canonical extractor class resolved from family `extractor_type` alias | `oscillator_families.*.extractor_type` |
| `channels` | Inferred from layers + families and sorted for deterministic output. Includes role/replay/supervisor metadata when declared | `oscillator_families`, `channels` |
| `families` | Derived family summary including extractor alias and declared config keys | `oscillator_families` |
| `layers` | Inferred layer summaries with resolved channel and oscillator span | `layers`, `oscillator_families` |
| `channel role/replay/supervisor metadata` | Uses declared channel metadata; omitted when channel is undeclared | `channels:` block |
| `channel_groups` | Copied through and normalised by name | `channel_groups` |
| `cross_channel_couplings` | Copied through from explicit declarations | `cross_channel_couplings` |
| `coupling` | Copies base coupling structure fields | `coupling` |
| `objectives` | Copies objective weights and layer partitioning | `objectives` |
| `boundaries` | Boundaries with names and severity only | `boundaries` |
| `actuators` | Actuator names, knobs, scopes, and limits | `actuators` |

## Visibility contract

This contract keeps run intent readable in two places: CLI output and audit
records. The same summary is visible in both review surfaces so one can validate
an outcome without recomputing all defaults from scratch.

1. CLI output prints the resolved summary before execution.
2. Audit header stores the same `binding_summary` payload for replay/report.
3. Driver values are not copied into summary metadata; only structural keys are
   retained to avoid leaking deployment-local endpoint details.

## Operator workflow

1. Run `spo validate` and confirm resolved summary.
2. Run `spo inspect` when reviewing channel algebra and cross-channel coupling.
3. Run `spo run --audit ...` and verify `binding_summary` is present in the
   generated audit header before promoting the run artifact.

## Operational consequence

Resolved defaults are the boundary between intent and execution:

- they make the runtime effective configuration explicit before simulation begins,
- they preserve operator trust by making inferred execution defaults reproducible,
- and they reduce ambiguity when comparing failures across environments.

When an operator reports unexpected behavior, this page is the first artifact to
compare: if the runtime summary differs from the declared spec intent, the issue
is usually in binding declaration or adapter pre-processing, not in core dynamics.
