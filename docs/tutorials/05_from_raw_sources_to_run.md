# 05 — End-to-End From Raw Sources

Build a production-like path from raw source files to a full `spo run`, then
finish by inspecting supervisor decisions and actuation output.

This tutorial uses local synthetic files so you can reproduce it in minutes.

## 1. Create a New Domainpack

```bash
spo scaffold valve_tune
mkdir -p domainpacks/valve_tune/data
```

Keep the scaffold files and add three raw inputs that represent the three
P/I/S channels.

## 2. Add Raw Source Files

### 2.1 CSV sensor stream (P channel)

```csv title="domainpacks/valve_tune/data/pressure_sensor.csv"
t_s,pressure_bar
0.00,2.00
0.02,2.08
0.04,2.15
0.06,2.28
0.08,2.41
0.10,2.36
0.12,2.18
0.14,2.04
0.16,1.90
0.18,1.86
0.20,1.93
```

### 2.2 Event log (I channel)

```csv title="domainpacks/valve_tune/data/event_log.csv"
t_s,event
0.000,start
0.060,pulse
0.120,pulse
0.190,pulse
0.260,pulse
0.320,alarm
0.450,pulse
```

### 2.3 State-machine trace (S channel)

```csv title="domainpacks/valve_tune/data/state_trace.csv"
t_s,state
0.00,idle
0.05,fill
0.15,prime
0.22,steady
0.38,drift
0.55,recover
0.78,steady
```

## 3. Wiring the Binding Spec

Replace `domainpacks/valve_tune/binding_spec.yaml` with a three-channel spec:

```yaml title="domainpacks/valve_tune/binding_spec.yaml"
name: valve_tune
version: "0.1.0"
safety_tier: research
sample_period_s: 0.01
control_period_s: 0.1

layers:
  - name: sensor_layer
    index: 0
    oscillator_ids: [pressure_osc]
  - name: rhythm_layer
    index: 1
    oscillator_ids: [event_osc, mode_osc]

oscillator_families:
  pressure_osc:
    channel: P
    extractor_type: physical
    config:
      source: data/pressure_sensor.csv
      time_column: t_s
      value_column: pressure_bar
  event_osc:
    channel: I
    extractor_type: event
    config:
      source: data/event_log.csv
      time_column: t_s
  mode_osc:
    channel: S
    extractor_type: ring
    config:
      source: data/state_trace.csv
      time_column: t_s
      state_column: state
      states: [idle, fill, prime, steady, drift, recover]

coupling:
  base_strength: 0.38
  decay_alpha: 0.25
  templates: {}

drivers:
  physical:
    zeta: 0.0
    psi: 0.0
  informational:
    zeta: 0.03
  symbolic:
    zeta: 0.03

objectives:
  good_layers: [0, 1]
  bad_layers: []
  good_weight: 1.0
  bad_weight: 1.0

boundaries:
  - name: coherence_floor
    variable: R
    lower: 0.25
    upper: null
    severity: soft

actuators:
  - name: coupling_global
    knob: K
    scope: global
    limits: [0.0, 3.5]

policy: policy.yaml
```

## 4. Policy for Supervisor Actions

Create `domainpacks/valve_tune/policy.yaml`:

```yaml
rules:
  - name: recover_coherence
    regime: [DEGRADED, RECOVERY]
    condition:
      metric: R_good
      layer: 0
      op: "<"
      threshold: 0.45
    action:
      knob: K
      scope: global
      value: 0.12
      ttl_s: 6.0
```

The supervisor has one job: raise coupling when the physical layer drops.

## 5. Validate the Contract

```bash
spo validate domainpacks/valve_tune/binding_spec.yaml
```

Expected output includes a resolved structure summary and `channels=I, P, S`.

## 6. Run and Capture Audit

```bash
spo run domainpacks/valve_tune/binding_spec.yaml --steps 180 --seed 7 --audit valve_tune_audit.jsonl
```

The CLI prints the final `R_good`, `R_bad`, and final regime.

## 7. Inspect Supervisor Decisions and Actuation

Show action rows and boundary events from the audit log:

```bash
python - <<'PY'
import json
from pathlib import Path

for line in Path("valve_tune_audit.jsonl").read_text().splitlines():
    row = json.loads(line)
    if row.get("actions"):
        print(f"step={row['step']} actions={row['actions']}")
    if row.get("event_type") == "actuation":
        print(f"actuation={row}")
PY
```

You now have a complete raw-source -> binding -> run -> decision chain.

## 8. Quick Visual Check

```python
from scpn_phase_orchestrator.audit.replay import ReplayEngine
from scpn_phase_orchestrator.reporting.plots import CoherencePlot

entries = ReplayEngine("valve_tune_audit.jsonl").load()

plotter = CoherencePlot(entries)
plotter.plot_r_timeline("valve_tune_r.png")
plotter.plot_regime_timeline("valve_tune_regime.png")
plotter.plot_action_audit("valve_tune_actions.png")
```

## 9. Finish with Replay

```bash
spo replay valve_tune_audit.jsonl --verify --output valve_tune_replay.json
spo report valve_tune_audit.jsonl --json-out > valve_tune_report.json
```

`--verify` confirms the same initial conditions reproduce the same step-by-step
trajectory and detects any divergence.
