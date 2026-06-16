# Manufacturing SPC Domainpack

Statistical process control for discrete manufacturing lines, mapped to
SPO's Kuramoto/UPDE framework.

## Why Kuramoto Fits This Domain

Sensor signals (vibration, temperature, pressure) oscillate around
setpoints.  Tool wear causes systematic drift that correlates sensor
phases -- exactly the kind of synchronisation Kuramoto detects.
When sensors synchronise (high R on bad layer), it signals correlated
drift from a common root cause.

## Layers

| Layer | Oscillators | Channel | Source |
|-------|------------|---------|--------|
| sensor | 4 | P (physical) | Accelerometers, thermocouples, transducers |
| machine | 3 | I (informational) | PLC/SCADA: OEE, cycle time, throughput |
| line | 2 | S (symbolic) | Quality inspection: yield rate, defect class |

## Boundaries

- **temp_high**: temperature < 85°C (hard) -- OEM limit
- **pressure_low**: pressure > 2.0 bar (hard) -- minimum process pressure
- **vibration_warning**: vibration < 4.5 mm/s RMS (soft)

## Actuators

| Actuator | Knob | Physical Meaning |
|----------|------|-----------------|
| coupling_global | K | Inter-station signal coupling |
| lag_sensor | alpha | Sensor-to-controller delay |
| damping_global | zeta | PID damping injection |

## Value-Alignment Guard

The binding spec includes a `value_alignment` template for review-time
process-control checks. It bounds inter-station coupling, sensor lag, and
damping-drive steps, then falls back to a zero-damping safe hold when a
candidate action exceeds those priors.

This template is for simulation, replay, and policy review. It is not a live
machine-safety controller.

## Imprint

Tool wear history: progressive degradation accumulates and modulates
coupling, representing gradual loss of machining precision.

## Scenario

200 steps: normal production -> systematic tool wear (sensor drift) ->
random machine failure -> golden sample reference injection -> recovery.

## Operator Runbook

Use this sequence for local review evidence. It validates the binding, runs the
pack-owned non-actuating scenario, records the expected local-runtime refusal
for the consumer-tier binding, and exports a policy model for formal review.

```bash
mkdir -p /tmp/spo-manufacturing-spc
.venv/bin/spo validate domainpacks/manufacturing_spc/binding_spec.yaml
.venv/bin/python domainpacks/manufacturing_spc/run.py \
  > /tmp/spo-manufacturing-spc/scenario.txt
set +e
.venv/bin/spo run domainpacks/manufacturing_spc/binding_spec.yaml \
  --steps 500 \
  --seed 42 \
  --audit /tmp/spo-manufacturing-spc/run.jsonl \
  > /tmp/spo-manufacturing-spc/blocked-run.txt 2>&1
test $? -ne 0
set -e
.venv/bin/spo formal-export domainpacks/manufacturing_spc/binding_spec.yaml \
  --export policy \
  --output /tmp/spo-manufacturing-spc/policy.prism
```

Expected artefacts:

- `/tmp/spo-manufacturing-spc/scenario.txt`: deterministic local scenario output
  from this pack's `run.py`.
- `/tmp/spo-manufacturing-spc/blocked-run.txt`: expected refusal showing that
  the local runtime will not execute a consumer-tier binding through
  `spo run --audit`.
- `/tmp/spo-manufacturing-spc/policy.prism`: policy export for formal review.

`spo replay`, `spo report`, and `spo explain` require an audit log from an
admitted execution path. They are intentionally not part of the current local
consumer-tier command chain. Full formal-package and STL exports also remain
blocked until this binding defines a `protocol_net` and `stl_monitors`.

Do not use this runbook as a live PLC, SCADA, or machine-safety controller. The
actuators above map to coupling, sensor lag, and damping concepts for review
only. Any production-line connection remains blocked until a senior operator
provides all of the following evidence:

- read-only historian, MES, or SCADA tag mapping for every oscillator and
  boundary in `binding_spec.yaml`;
- proof that the first connector path cannot write PLC tags, setpoints, drive
  parameters, or safety-interlock state;
- independent OEM, PLC, and safety-relay limits outside SPO for temperature,
  pressure, vibration, and emergency stop conditions;
- operator-approved target hashes for the reviewed binding, connector plan, and
  audit-storage location;
- site-specific acceptance thresholds for scrap-rate, cycle-time, and tool-wear
  review plus a rollback plan for any advisory workflow.
