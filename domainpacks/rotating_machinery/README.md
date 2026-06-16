# Rotating Machinery Domainpack

Maps rotor dynamics and vibration monitoring to SPO's Kuramoto/UPDE framework.

## Why Kuramoto Fits This Domain

Rotating machinery vibration is fundamentally a coupled oscillator problem.
Shaft harmonics (1X, 2X, 3X), bearing defect frequencies (BPFI, BPFO),
blade-pass frequency, and structural modes all interact through mechanical
coupling (shared shaft, bearing stiffness, casing).  Phase relationships
between these harmonics diagnose fault type: 1X+2X in-phase = misalignment,
specific bearing frequencies = inner/outer race defect.

## Layers

| Layer | Oscillators | Channel | Freq (×RPM) | Purpose |
|-------|------------|---------|-------------|---------|
| shaft_rotation | 3 | P (hilbert) | 1X/2X/3X | Running speed harmonics |
| blade_dynamics | 2 | P (hilbert) | 5X/0.3X | Blade pass + subsync flutter |
| bearing_condition | 3 | I (event) | 1.68/1.2/0.07 | BPFI/BPFO/FTF (SKF 6205) |
| structural_resonance | 2 | I (event) | 3.6X/7.2X | FEA-derived mode shapes |

## Boundaries

- **vibration_velocity**: < 7.1 mm/s (hard) — ISO 10816-3 zone C/D
- **shaft_displacement**: < 100 um (soft) — API 670
- **bearing_temperature**: < 105°C (hard)

## Actuators

| Actuator | Knob | Physical Meaning |
|----------|------|-----------------|
| speed_setpoint | zeta | VFD speed reference |
| bearing_stiffness | K | Support stiffness (pedestal adjustment) |
| damper_viscosity | alpha | Squeeze-film damper viscosity |

The binding spec includes a `value_alignment` template for review-time speed
setpoint, blade-damper, and bearing-stiffness actuation guards.

## Imprint

Bearing wear: ISO 15243 spalling progression accumulates and modulates
mechanical coupling, representing gradual bearing degradation.

## Scenario

200 steps: cold start ramp -> 1st critical speed crossing -> nominal speed ->
bearing degradation onset -> blade flutter event -> policy response.

## Operator Runbook

Use this sequence for local review evidence. It validates the binding, runs the
pack-owned non-actuating scenario, records the expected local-runtime refusal
for the consumer-tier binding, and exports a policy model for formal review.

```bash
mkdir -p /tmp/spo-rotating-machinery
.venv/bin/spo validate domainpacks/rotating_machinery/binding_spec.yaml
.venv/bin/python domainpacks/rotating_machinery/run.py \
  > /tmp/spo-rotating-machinery/scenario.txt
set +e
.venv/bin/spo run domainpacks/rotating_machinery/binding_spec.yaml \
  --steps 500 \
  --seed 42 \
  --audit /tmp/spo-rotating-machinery/run.jsonl \
  > /tmp/spo-rotating-machinery/blocked-run.txt 2>&1
test $? -ne 0
set -e
.venv/bin/spo formal-export domainpacks/rotating_machinery/binding_spec.yaml \
  --export policy \
  --output /tmp/spo-rotating-machinery/policy.prism
```

Expected artefacts:

- `/tmp/spo-rotating-machinery/scenario.txt`: deterministic local scenario
  output from this pack's `run.py`.
- `/tmp/spo-rotating-machinery/blocked-run.txt`: expected refusal showing that
  the local runtime will not execute a consumer-tier binding through
  `spo run --audit`.
- `/tmp/spo-rotating-machinery/policy.prism`: policy export for formal review.

`spo replay`, `spo report`, and `spo explain` require an audit log from an
admitted execution path. They are intentionally not part of the current local
consumer-tier command chain. Full formal-package and STL exports also remain
blocked until this binding defines a `protocol_net` and `stl_monitors`.

Do not use this runbook as a live machinery controller. The actuators above map
to VFD speed reference, support stiffness, and damper-viscosity concepts, but
the current domainpack only emits review evidence. Any plant connection remains
blocked until a senior operator provides all of the following evidence:

- calibrated vibration, temperature, and tachometer source mapping for every
  oscillator in `binding_spec.yaml`;
- read-only historian or condition-monitoring ingestion before any write-capable
  connector is enabled;
- independent overspeed, vibration-trip, and bearing-temperature protection
  outside SPO;
- operator-approved target hashes for the reviewed binding, connector plan, and
  audit-storage location;
- site-specific acceptance thresholds for ISO 10816-3/API 670 limits and a
  rollback plan for any VFD or damper advisory.
