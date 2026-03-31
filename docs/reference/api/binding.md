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

## Types

Core type definitions shared across the binding subsystem.

`ActuatorMapping` — maps a control knob to a named actuator with scope and limits.
`ChannelSpec` — declares a P/I/S channel with extractor type and parameters.
`VALID_KNOBS` — the set of control knobs recognised by the actuation layer (`K`, `alpha`, `zeta`, `Psi`).

::: scpn_phase_orchestrator.binding.types

## Loader

Loads binding specifications from YAML files. Supports:

- Single-file specs (most domainpacks)
- Multi-file specs with `$ref` template references
- Environment variable interpolation for secrets (API keys, endpoints)
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
