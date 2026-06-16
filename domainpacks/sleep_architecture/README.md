<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Sleep Architecture Domainpack -->

# Sleep Architecture Domainpack

Maps sleep-band EEG dynamics to coupled oscillators. Delta, theta,
alpha, and gamma layers support sleep-stage phase analysis and boundary
checks for arousal and band-power conditions.

## Layers

| Layer | Oscillators | Channel | Purpose |
|-------|-------------|---------|---------|
| delta | 2 | P | Slow-wave sleep band |
| theta | 2 | P | NREM/REM transition band |
| alpha | 2 | P | Wake and arousal band |
| gamma | 2 | P | REM and binding-band activity |

## Boundaries

- `delta_power_ceiling`: soft delta-power ratio ceiling.
- `arousal_index`: hard arousals-per-hour ceiling.

## Value-Alignment Guard

The binding spec includes a `value_alignment` template for review-time
sleep-architecture control checks. It bounds global coupling, circadian-drive,
and phase-advance proposals, then falls back to a zero-drive safe hold when a
candidate action exceeds those priors.

This template is for simulation, replay, and policy review. It is not a live
sleep intervention protocol or medical-device control policy.

## Run

```bash
spo validate domainpacks/sleep_architecture/binding_spec.yaml
spo run domainpacks/sleep_architecture/binding_spec.yaml --steps 100
python domainpacks/sleep_architecture/run.py
```

## Operator Runbook

Use this pack as a research-tier sleep-architecture simulation, replay, and
review artefact. It is not a sleep-staging medical device, closed-loop sleep
intervention, alarm system, or treatment protocol.

### 1. Validate the Binding

```bash
spo validate domainpacks/sleep_architecture/binding_spec.yaml
```

Expected result: the binding is valid and reports the `research` safety tier.

### 2. Run the Pack-Owned Scenario

```bash
PYTHONPATH=src python domainpacks/sleep_architecture/run.py
```

Expected result: the deterministic sleep-band scenario executes end to end and
prints a compact JSON summary for review.

### 3. Produce and Verify an Audit Log

```bash
mkdir -p /tmp/spo_sleep_architecture_runbook
spo run domainpacks/sleep_architecture/binding_spec.yaml \
  --steps 40 \
  --audit /tmp/spo_sleep_architecture_runbook/audit.json

spo replay /tmp/spo_sleep_architecture_runbook/audit.json --verify
```

Expected result: the research-tier run admits an audit log, and replay
verification passes deterministically.

### 4. Generate Review Artefacts

```bash
spo report /tmp/spo_sleep_architecture_runbook/audit.json \
  > /tmp/spo_sleep_architecture_runbook/report.txt

spo explain /tmp/spo_sleep_architecture_runbook/audit.json \
  --markdown-out /tmp/spo_sleep_architecture_runbook/explain.md
```

Expected result: the report text and explanation are written under
`/tmp/spo_sleep_architecture_runbook/`.

Policy export is currently blocked because the referenced policy file contains
no rules:

```bash
spo formal-export domainpacks/sleep_architecture/binding_spec.yaml \
  --export policy \
  --output /tmp/spo_sleep_architecture_runbook/policy.prism
```

Expected result today: the command fails closed with a policy-no-rules error.
Full formal packages and STL monitor exports also remain blocked until this
binding adds `protocol_net` and `stl_monitors` evidence.

### 5. Live-Use Readiness Boundary

Do not connect this pack to PSG/EEG hardware, stimulation devices, bedside
alarms, wearables, or clinical sleep-lab workflows from the local runbook. A
live research or clinical deployment needs, at minimum:

- subject/data-governance approval and consent path for the intended dataset;
- hardware adapter provenance, isolation, and latency evidence;
- admitted audit-log environment with replay/report/explain retention;
- reviewed policy rules and monitor surfaces for the intended protocol;
- operator approval, version pinning, and rollback procedure;
- explicit statement that SPO does not diagnose, stage, or treat sleep
  disorders.

## Read Next

- [Sleep Staging API](../../docs/reference/api/monitor_sleep_staging.md)
- [Analysis Toolkit](../../docs/guide/analysis_toolkit.md)
- [Notebook 16](../../notebooks/16_sleep_staging.ipynb)
