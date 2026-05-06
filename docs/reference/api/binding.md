# Binding System

The binding system connects domain-specific signals to SPO's universal
oscillator framework. A *binding specification* (YAML file) declares
the complete interface between a domain and the phase dynamics engine.

## Pipeline position

The binding system is the **configuration layer** of the SPO pipeline.
It is loaded once at startup and configures all downstream subsystems:

```
binding_spec.yaml
       │
       ↓
  load_binding_spec()
       │
       ↓
  validate_binding_spec()
       │
       ↓
  BindingSpec
  ├── layers[] ──→ Oscillator Extractors (P/I/S)
  ├── coupling  ──→ CouplingBuilder.build()
  ├── policy    ──→ PolicyEngine rules
  └── actuators ──→ ActuationMapper mappings
```

Without a valid binding spec, SPO cannot start. The spec declares
*what* to observe, *how* to couple, *when* to intervene, and *where*
to actuate.

## Role in the Architecture

The binding system is the first stage of the SPO pipeline:

```
Domain Data ─► binding_spec.yaml ─► Loader ─► Validator ─► BindingSpec
                                                              │
                                                    Oscillator Extractors
                                                    Coupling Templates
                                                    Policy Rules
                                                    Actuator Mappings
```

Every domainpack ships a binding specification. When SPO starts,
the loader reads the YAML, the validator checks it against the schema,
and the resulting `BindingSpec` configures all downstream subsystems.

## Specification Structure

A binding spec declares:

```yaml
name: power_grid
version: "1.0"
layers:
  - name: generator_phase
    channel: P
    extractor: hilbert
    frequency_range: [49.5, 50.5]
  - name: load_demand
    channel: I
    extractor: event_rate
coupling:
  template: distance_decay
  K_base: 0.47
  decay_alpha: 0.25
policy:
  rules:
    - condition: R < 0.6
      action: boost_K(0.1)
actuators:
  - name: governor
    knob: K
    scope: layer_0
    limits: [0.0, 2.0]
```

The schema is defined in `docs/specs/binding_spec.schema.json` and
enforced by the validator at load time.

## Resolved Runtime Summary

`validate`/`inspect`/`run` commands rely on a resolved summary that is
produced from the YAML and includes inferred defaults (for example
`control_interval_steps` and `engine_mode`). The full contract is documented in
`Resolved Runtime Defaults` and exposed as a CLI summary plus audit metadata.
The summary now embeds `channel_algebra`, so audit consumers can read required
channels, optional channels, derived channels, group membership, coupling
participants, and missing required channel evidence from the same resolved
configuration record.

::: scpn_phase_orchestrator.binding.resolved

## N-Channel Algebra Summary

`build_channel_algebra_report()` produces a deterministic, JSON-safe view of
declared channels, required/optional status, derived channels, group
membership, supervisor visibility, coupling participation, and cross-channel
edges. It is intended for audit, replay, and reporting surfaces that need a
channel-count-agnostic view without re-parsing YAML.

The same report classifies delayed and uncertain channels from existing
`role`, `metric_semantics`, and `replay_semantics` metadata. This lets audit and
reporting surfaces expose delayed/uncertain policy evidence without changing
the binding schema.

The report also emits runtime policy records for every declared channel.
Delayed channels use `hold_last_runtime_evidence`, uncertain channels use
`confidence_weight_runtime_contribution`, missing required channels use
`block_required_channel`, and missing optional channels use
`drop_optional_channel`. This gives supervisor/runtime callers deterministic
handling semantics without adding new binding-schema fields.

```python
from scpn_phase_orchestrator.binding import (
    build_channel_algebra_report,
    load_binding_spec,
)

spec = load_binding_spec("domainpacks/power_safety_nchannel/binding_spec.yaml")
report = build_channel_algebra_report(spec)
audit_record = report.to_audit_record()
```

This report is read-only. It complements `validate_binding_spec()` rather than
replacing validation gates.

::: scpn_phase_orchestrator.binding.channel_algebra

## Types

Core type definitions shared across the binding subsystem.

### BindingSpec (dataclass)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | `str` | yes | Domainpack name |
| `version` | `str` | yes | Spec version |
| `safety_tier` | `str` | yes | Safety classification |
| `sample_period_s` | `float` | yes | Input sampling interval |
| `control_period_s` | `float` | yes | Control loop interval |
| `layers` | `list[HierarchyLayer]` | yes | Oscillator layers |
| `oscillator_families` | `dict[str, OscillatorFamily]` | yes | P/I/S families |
| `coupling` | `CouplingSpec` | yes | K_nm parameters |
| `drivers` | `DriverSpec` | yes | External drive config |
| `objectives` | `ObjectivePartition` | yes | Optimisation targets |
| `boundaries` | `list[BoundaryDef]` | yes | Safety boundaries |
| `actuators` | `list[ActuatorMapping]` | yes | Output actuators |
| `imprint_model` | `ImprintSpec \| None` | no | Memory dynamics |
| `geometry_prior` | `GeometrySpec \| None` | no | Spatial constraints |
| `protocol_net` | `ProtocolNetSpec \| None` | no | Petri net FSM |
| `amplitude` | `AmplitudeSpec \| None` | no | Stuart-Landau params |

### Other types

- `ActuatorMapping` — maps a control knob to a named actuator
  with scope and limits
- `HierarchyLayer` — declares a layer with channel, extractor, and
  frequency range
- `VALID_KNOBS` — recognised control knobs: `K`, `alpha`, `zeta`, `Psi`

::: scpn_phase_orchestrator.binding.types

## Loader

Loads binding specifications from YAML files. Supports:

- Single-file specs (most domainpacks)
- Multi-file specs with `$ref` template references
- Environment variable interpolation for credentials and endpoints
- Default value injection for optional fields

The loader does *not* validate — that is the validator's job. This
separation allows testing with deliberately invalid specs.

::: scpn_phase_orchestrator.binding.loader

## Validator

Schema validation for binding specifications. Checks:

- Field types and required fields (against JSON schema)
- Cross-references: actuator knobs must match declared layers
- Frequency ranges: `f_min < f_max`, both positive
- Channel constraints: at least one layer, no duplicate names
- Template resolution: referenced templates must exist

Validation errors are collected (not raised on first failure) so that
users see all problems at once.

::: scpn_phase_orchestrator.binding.validator

## Security

The binding loader enforces security constraints:

1. **Path traversal rejection** — `../` sequences in file paths are
   rejected to prevent reading outside the domainpack directory
2. **Schema validation** — all fields are checked against the JSON
   schema before any data is used
3. **Environment variable interpolation** — only whitelisted env vars
   are substituted; arbitrary code execution is not possible
4. **Size limits** — binding specs exceeding 1 MB are rejected

These protections are tested in `tests/test_binding_loader_security.py`
with adversarial inputs including malicious YAML, oversized files,
and path traversal attempts.

## Domainpacks

A **domainpack** is a directory containing a binding spec plus optional
data files (coupling templates, calibration data, policy rules). SPO
ships with built-in domainpacks for common domains:

| Domainpack | Layers | Channels | Description |
|------------|--------|----------|-------------|
| `power_grid` | generators, loads | P, I | AC power system sync |
| `neural_eeg` | cortical regions | P | EEG phase dynamics |
| `microservices` | API endpoints | I | IT infrastructure sync |
| `tokamak` | plasma + magnetics | P | Fusion plasma control |
| `smart_factory` | machines, queues | P, I, S | Manufacturing sync |

Each domainpack is validated at load time against the schema. Invalid
domainpacks produce detailed error messages listing all violations.

**Performance:** `load_binding_spec()` < 10 ms.

## Symbolic Binding Compiler

The `SemanticDomainCompiler` is the first review-gated symbolic-to-binding
path. It translates a domain intent string into a `BindingSpec` and can also
emit a complete artefact bundle:

- `binding_spec.yaml` for the domain interface
- `policy.yaml` with a conservative low-coherence recovery rule
- `audit.json` with confidence, matched keywords, validation status, dry-run
  coherence, and Petri-net review reachability metadata
- `README.md` for the generated domainpack directory

The compiler remains deterministic and local. It extracts layer counts,
domain-family keywords, oscillator counts, channel declarations, safe default
actuator mappings, and a review transition in `protocol_net`. The generated
binding is passed through `validate_binding_spec()` and a short
`UPDEEngine` dry run before artefacts are returned.

CLI usage:

```bash
spo generate "A 3-layer cardiac rhythm suppression system" \
  --name cardiac_review \
  --output-dir domainpacks/cardiac_review
spo validate domainpacks/cardiac_review/binding_spec.yaml
```

::: scpn_phase_orchestrator.binding.semantic
