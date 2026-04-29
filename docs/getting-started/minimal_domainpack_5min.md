<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Minimal viable domainpack guide -->

# Minimal Viable Domainpack in 5 Minutes

This path starts with three tiny source streams and ends with a validated
domainpack, a run, an audit log, and a replay check. It is intentionally
small: one physical signal, one event stream, and one state-machine trace.

## 1. Create a Workspace

```bash
spo scaffold pump_lab
mkdir -p domainpacks/pump_lab/data
```

The scaffold gives you the domainpack directory and starter YAML files. The
`data/` directory records the raw source streams that the binding will name.

## 2. Add Raw Source Data

Physical channel `P`: a sampled pressure wave.

```csv title="domainpacks/pump_lab/data/pressure.csv"
t_s,pressure_bar
0.00,2.00
0.01,2.19
0.02,2.36
0.03,2.47
0.04,2.50
0.05,2.44
0.06,2.29
0.07,2.10
0.08,1.90
0.09,1.71
```

Informational channel `I`: event timestamps for retry or alarm pulses.

```csv title="domainpacks/pump_lab/data/events.csv"
t_s,event
0.000,start
0.120,pulse
0.235,pulse
0.355,pulse
0.490,pulse
```

Symbolic channel `S`: state-machine trace.

```csv title="domainpacks/pump_lab/data/states.csv"
t_s,state
0.00,idle
0.10,prime
0.20,run
0.40,recover
0.55,run
```

At this point you have enough data to define three oscillator families:
continuous waveform, event cadence, and state-cycle phase.

## 3. Bind the Data to Oscillators

Replace the scaffolded `binding_spec.yaml` with a minimal binding like this:

```yaml title="domainpacks/pump_lab/binding_spec.yaml"
name: pump_lab
version: "0.1.0"
safety_tier: research
sample_period_s: 0.01
control_period_s: 0.1

layers:
  - name: sensor
    index: 0
    oscillator_ids: [pressure_wave]
  - name: control
    index: 1
    oscillator_ids: [event_cadence, state_cycle]

oscillator_families:
  pressure_wave:
    channel: P
    extractor_type: physical
    config:
      source: data/pressure.csv
      time_column: t_s
      value_column: pressure_bar
  event_cadence:
    channel: I
    extractor_type: event
    config:
      source: data/events.csv
      time_column: t_s
  state_cycle:
    channel: S
    extractor_type: ring
    config:
      source: data/states.csv
      time_column: t_s
      state_column: state
      states: [idle, prime, run, recover]

coupling:
  base_strength: 0.35
  decay_alpha: 0.2
  templates: {}

drivers:
  physical:
    zeta: 0.0
    psi: 0.0
  informational:
    zeta: 0.02
  symbolic:
    zeta: 0.02

objectives:
  good_layers: [0, 1]
  bad_layers: []
  good_weight: 1.0
  bad_weight: 1.0

boundaries:
  - name: coherence_floor
    variable: R
    lower: 0.2
    upper: null
    severity: soft

actuators:
  - name: coupling_global
    knob: K
    scope: global
    limits: [0.0, 3.0]

policy: policy.yaml
```

The binding answers four questions:

| Question | Field |
| --- | --- |
| What cycles exist? | `oscillator_families` |
| Where do they live in the hierarchy? | `layers` |
| Which coherence should increase or decrease? | `objectives` |
| What can the supervisor change? | `actuators` |

## 4. Add a Policy

Use one rule that raises global coupling when the good-layer coherence drops.

```yaml title="domainpacks/pump_lab/policy.yaml"
rules:
  - name: restore_coherence
    regime: [DEGRADED, RECOVERY, CRITICAL]
    condition:
      metric: R_good
      op: "<"
      threshold: 0.5
    action:
      knob: K
      scope: global
      value: 0.15
      ttl_s: 5.0
```

This is deliberately boring. The first domainpack should prove the path before
you add compound triggers, cooldowns, amplitude mode, Petri-net protocols, or
custom extractors.

## 5. Validate and Run

```bash
spo validate domainpacks/pump_lab/binding_spec.yaml
spo run domainpacks/pump_lab/binding_spec.yaml --steps 300 --audit pump_lab_audit.jsonl --seed 42
spo replay pump_lab_audit.jsonl --verify
spo report pump_lab_audit.jsonl
```

The built-in run path validates the binding, then uses the declared oscillator
families to initialise a deterministic seeded phase probe. The `source` entries
above are the intake contract for custom extractors and adapter tooling; keep
them beside the binding so the handoff from real measurements to oscillator
families is explicit.

The run prints final coherence and regime state. The audit log records the
phase vector, layer order parameters, actions, and chained hashes. Replay
verifies the audit chain before the report turns the run into a readable
summary.

## 6. Know What Each Knob Did

| Knob | First-use meaning |
| --- | --- |
| `K` | Coupling strength. Higher values pull connected oscillators towards shared phase. |
| `alpha` | Phase lag. Use when coupling has a known delay or offset. |
| `zeta` | External drive strength. Use sparingly for temporary forcing. |
| `Psi` | Target phase for the external drive. |

For a first domainpack, only expose `K`. Add `zeta`, `alpha`, and `Psi` after
you can explain which real actuator changes them.

## 7. Promotion Checklist

Before treating the domainpack as more than a tutorial:

- Replace toy CSV rows with representative source files.
- Add boundaries for measured safety variables, not only `R`.
- Add a domain README with sensor names, units, and ownership.
- Run `spo replay --verify` for every audit you keep.
- Add regression tests for the binding spec and any custom extractor.

Next: use the [New Domain Checklist](../tutorials/01_new_domain_checklist.md)
when the minimal path works end to end.
