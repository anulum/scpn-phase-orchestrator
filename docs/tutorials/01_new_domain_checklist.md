# New Domain Checklist

Steps to bind a new domain to the phase orchestrator.

## 1. Identify Oscillators

For each cyclic process in the domain, determine:

- **Channel:** P (continuous waveform), I (event timestamps), or S (discrete state sequence).
- **Frequency range:** Expected oscillation frequency in Hz.
- **Observability:** Can you measure it directly, or must it be inferred?

Use the [Oscillator Hunt Sheet](02_oscillator_hunt_sheet.md) to catalogue candidates.

## 2. Define Hierarchy Layers

Group oscillators into layers by timescale or scope:

- **Micro:** fastest cycles (e.g., individual queue depths)
- **Meso:** intermediate aggregates (e.g., service latency)
- **Macro:** slowest / system-wide (e.g., error rate, throughput)

Each layer gets a name, index, and list of oscillator_ids.

## 3. Write binding_spec.yaml

Use `spo scaffold <domain_name>` to generate a template. Fill in:

- `name`, `version`, `safety_tier`
- `sample_period_s` (data sampling interval)
- `control_period_s` (supervisor decision interval, >= sample_period_s)
- `layers` list
- `oscillator_families` dict
- `coupling` section (base_strength, decay_alpha)
- `objectives` (good_layers, bad_layers)
- `boundaries` (soft/hard limits)
- `actuators` (available knobs and their limits)

Validate: `spo validate domainpacks/<domain>/binding_spec.yaml`

## 4. Implement Custom Extractors (If Needed)

Built-in extractors cover most cases:

- `physical` -- Hilbert transform for continuous signals
- `informational` -- inter-event interval for timestamps
- `symbolic` -- ring-phase or graph-walk for state sequences

If your signal needs special processing, subclass `PhaseExtractor`. See [Plugin API](../specs/plugin_api.md).

## 5. Build Knm Templates

Start with the default exponential decay. Refine from domain knowledge:

- Which oscillators interact directly?
- Are there known coupling strengths from data or physics?
- Should coupling change under fault conditions?

See [Build Knm Templates](03_build_knm_templates.md).

## 6. Define Objectives

Partition layers into good (maximise R) and bad (suppress R):

```yaml
objectives:
  good_layers: [2]   # coordination layer
  bad_layers: [0]     # retry storm layer
```

## 7. Set Boundaries

Define soft/hard limits on observable variables:

```yaml
boundaries:
  - name: queue_overflow
    variable: queue_depth
    upper: 10000
    severity: hard
```

## 8. Map Actuators

Declare what knobs are available and their limits:

```yaml
actuators:
  - name: coupling_control
    knob: K
    scope: global
    limits: [0.0, 3.0]
```

## 9. Write policy.yaml

Declare supervisor rules that fire when regime/metric conditions match:

```yaml
rules:
  - name: suppress_fault
    regime: [DEGRADED, CRITICAL]
    condition:
      metric: R_bad
      op: ">"
      threshold: 0.5
    action:
      knob: zeta
      scope: global
      value: 0.2
      ttl_s: 5.0

  - name: restore_target
    regime: [DEGRADED]
    condition:
      metric: R_good
      op: "<"
      threshold: 0.4
    action:
      knob: K
      scope: global
      value: 0.3
      ttl_s: 5.0
```

Reference `policy.yaml` in binding_spec.yaml: `policy: policy.yaml`.

## 10. Add Imprint (If Applicable)

If the domain has slow accumulation effects (wear, aging, drift), add
an `imprint_model` section to binding_spec.yaml:

```yaml
imprint_model:
  decay_rate: 0.002
  saturation: 4.0
  modulates: ["K"]
```

In run.py, integrate the ImprintModel:

```python
from scpn_phase_orchestrator.imprint.state import ImprintModel, ImprintState

imprint_model = ImprintModel(
    decay_rate=spec["imprint_model"]["decay_rate"],
    saturation=spec["imprint_model"]["saturation"],
)
imprint_state = ImprintState(n=n_osc)

# Before engine.step:
coupling = imprint_model.modulate_coupling(coupling, imprint_state)
# After actions:
exposure = np.abs(phases - np.mean(phases))
imprint_state = imprint_model.update(imprint_state, exposure, dt)
```

## 11. Write README.md

Every domainpack README must include these sections:

1. Title and one-line summary
2. **Why Kuramoto Fits This Domain** — cite at least one paper
3. **Layers** — table with Layer, Oscillators, Channel, Purpose
4. **Boundaries** — list with standard/source
5. **Actuators** — table with Actuator, Knob, Physical Meaning
6. **Imprint** — physical basis or "None" with justification
7. **Scenario** — step count and narrative arc

See `domainpacks/cardiac_rhythm/README.md` as the gold standard.

## 12. Validate and Run

```bash
spo validate domainpacks/<domain>/binding_spec.yaml
spo run domainpacks/<domain>/binding_spec.yaml --steps 200
```

Check R_good and R_bad in the output. Iterate on coupling strength and objectives.
