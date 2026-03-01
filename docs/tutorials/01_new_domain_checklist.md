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

## 9. Validate and Run

```bash
spo validate domainpacks/<domain>/binding_spec.yaml
spo run domainpacks/<domain>/binding_spec.yaml --steps 200
```

Check R_good and R_bad in the output. Iterate on coupling strength and objectives.
