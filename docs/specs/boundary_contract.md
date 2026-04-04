# Boundary Contract

The Boundary Contract defines how measured system variables are checked
against operational limits. Boundaries are the safety layer between the
UPDE integration engine and the physical or informational systems it
controls. When a measured value falls outside its defined range, the
boundary observer raises a violation that the supervisor uses to trigger
regime transitions.

## Boundary Types

| Severity | Meaning | Consequence |
|----------|---------|-------------|
| `soft` | Warning. System operating outside preferred range. | Logged. May contribute to DEGRADED regime. |
| `hard` | Violation. System in unsafe state. | Forces CRITICAL regime immediately. |

### Severity Design Rationale

The two-tier system matches industrial control practice:

- **Soft boundaries** define the nominal operating envelope. Crossing
  them indicates drift but not danger. Examples: coherence R dropping
  below 0.5, latency exceeding a SLO threshold, EEG amplitude rising
  above typical range. Soft violations are logged in the audit trace
  and may accumulate to trigger a DEGRADED regime assessment, but no
  single soft violation forces a regime change.

- **Hard boundaries** define absolute safety limits. Crossing them
  indicates the system is in a state that could cause harm or data
  corruption. Examples: queue depth exceeding memory capacity, coupling
  strength diverging to infinity, phase quality collapsing across all
  channels. A single hard violation immediately forces CRITICAL regime,
  which reduces coupling and activates recovery protocols.

Unknown severity values are treated as `"hard"` with a warning log,
following the fail-safe principle.

## BoundaryDef

Boundary definitions are declared in the binding spec YAML under the
`boundaries` list. Each definition specifies what variable to monitor,
its acceptable range, and the violation severity.

### YAML Schema

```yaml
boundaries:
  - name: queue_depth_limit
    variable: queue_depth
    lower: null
    upper: 10000
    severity: hard
  - name: latency_warning
    variable: p99_latency_ms
    lower: null
    upper: 500
    severity: soft
  - name: coherence_floor
    variable: R
    lower: 0.1
    upper: null
    severity: hard
  - name: coupling_ceiling
    variable: max_K
    lower: null
    upper: 50.0
    severity: hard
  - name: temperature_range
    variable: core_temp_C
    lower: 10.0
    upper: 85.0
    severity: soft
```

### Field Reference

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Human-readable identifier. Must be unique within the binding spec. |
| `variable` | `str` | Key in the values dict passed to `observe()`. |
| `lower` | `float` or `null` | Lower bound (inclusive). `null` = no lower check. |
| `upper` | `float` or `null` | Upper bound (inclusive). `null` = no upper check. |
| `severity` | `str` | `"soft"` or `"hard"`. |

### Range Semantics

- `lower: 10, upper: 90` — value must be in `[10, 90]`. Both bounds
  checked.
- `lower: null, upper: 100` — value must be `<= 100`. No lower bound.
- `lower: 0, upper: null` — value must be `>= 0`. No upper bound.
- `lower: null, upper: null` — no check performed (boundary is inert).

Violation occurs when `value < lower` or `value > upper`. Equality
with the bound is NOT a violation.

## BoundaryObserver

The `BoundaryObserver` is the runtime component that evaluates measured
values against the boundary definitions. It is stateless except for an
optional step counter and event bus attachment.

### Construction

```python
from scpn_phase_orchestrator.monitor import BoundaryObserver

observer = BoundaryObserver(boundary_defs=spec.boundaries)
```

The `boundary_defs` argument is a list of `BoundaryDef` dataclasses,
typically loaded from the binding spec via `load_binding_spec()`.

### Observation

```python
state = observer.observe({
    "queue_depth": 12000,
    "p99_latency_ms": 300,
    "R": 0.75,
    "max_K": 12.5,
    "core_temp_C": 42.0,
})
```

The `observe()` method:

1. Iterates over all boundary definitions.
2. Looks up each `variable` key in the provided values dict.
3. Skips boundaries whose variable is absent from the dict (missing
   values are not violations — they indicate the sensor is offline).
4. Checks `value < lower` and `value > upper`.
5. For each violation, formats a message string and appends it to the
   appropriate list in `BoundaryState`.
6. If any violations occurred and an `EventBus` is attached, posts a
   `RegimeEvent` with `kind="boundary_breach"`.

### Step Tracking

```python
state = observer.observe(values, step=42)
```

The optional `step` parameter records the integration step at which the
observation was made. This is included in the `RegimeEvent` posted to
the event bus, enabling temporal correlation in the audit log.

## BoundaryState

The return value of `observe()` is a `BoundaryState` dataclass:

| Field | Type | Content |
|-------|------|---------|
| `violations` | `list[str]` | All violated boundaries (soft + hard). |
| `soft_violations` | `list[str]` | Soft-severity violations only. |
| `hard_violations` | `list[str]` | Hard-severity violations only. |

### Message Format

Each violation message follows the pattern:

```
{name}: {variable}={value:.4g} outside [{lower}, {upper}]
```

Example: `"queue_depth_limit: queue_depth=1.2e+04 outside [None, 10000]"`

The `.4g` format ensures readable numbers for both small and large
values. `None` appears for unset bounds.

## Event Bus Integration

The observer can be connected to the supervisor's event bus:

```python
from scpn_phase_orchestrator.supervisor import EventBus

bus = EventBus()
observer.set_event_bus(bus)
```

When attached, every call to `observe()` that finds at least one
violation posts a `RegimeEvent`:

```python
RegimeEvent(
    kind="boundary_breach",
    step=current_step,
    detail="queue_depth_limit: queue_depth=1.2e+04 outside [None, 10000]"
)
```

The `RegimeManager` subscribes to these events. Its `evaluate()` method
checks `boundary_state.hard_violations` first — any hard violation
forces CRITICAL regardless of R values or other metrics.

## Integration with RegimeManager

The regime escalation path:

1. **No violations** — regime determined by R thresholds alone.
2. **Soft violations only** — logged in audit trace. If R is also low,
   the combination may push the system from NOMINAL to DEGRADED.
3. **Hard violations** — CRITICAL forced immediately. The supervisor
   reduces coupling (`zeta` is lowered, coupling matrix scaled down)
   and waits for the violation to clear before allowing recovery.

The exact R thresholds and regime transition rules are defined in
[regime_manager.md](regime_manager.md).

## Rust FFI Implementation

The `BoundaryObserver` has a Rust counterpart in `spo-supervisor`
(`boundaries.rs`). The Rust implementation mirrors the Python API:

```rust
let observer = BoundaryObserver::new(defs);
let state = observer.observe(&values);
assert!(state.hard_violations.is_empty());
```

The FFI wrapper (`PyBoundaryObserver` in `spo-ffi`) accepts Python
dicts and converts them to Rust `HashMap<String, f64>`. The Rust path
is faster for systems with many boundaries (>100) due to avoiding
Python dict lookup overhead.

## Domain-Specific Boundary Examples

### Neuroscience (EEG/BCI)

```yaml
boundaries:
  - name: eeg_amplitude_ceiling
    variable: eeg_max_uV
    lower: null
    upper: 200.0
    severity: hard   # > 200 uV indicates artifact
  - name: alpha_power_floor
    variable: alpha_band_power
    lower: 0.5
    upper: null
    severity: soft   # low alpha may indicate drowsiness
```

### Industrial Control (Plasma Tokamak)

```yaml
boundaries:
  - name: plasma_current_range
    variable: Ip_MA
    lower: 0.5
    upper: 2.5
    severity: hard
  - name: disruption_proximity
    variable: disruption_score
    lower: null
    upper: 0.8
    severity: hard
```

### Financial Markets

```yaml
boundaries:
  - name: correlation_breakdown
    variable: min_sector_corr
    lower: -0.3
    upper: null
    severity: soft
  - name: vol_spike
    variable: vix
    lower: null
    upper: 40.0
    severity: hard
```

## Testing the Boundary Contract

The test suite `tests/test_boundary_observer.py` verifies:

1. Soft violations are correctly partitioned from hard violations.
2. Missing variables in the values dict do not produce violations.
3. Boundary equality (value == bound) is not a violation.
4. Unknown severity is treated as hard with a warning log.
5. Event bus receives `RegimeEvent` when violations occur.
6. Empty boundary list produces no violations for any input.
7. Both bounds can be set simultaneously (range check).
8. Message format matches the documented pattern.

The Rust-Python parity test (`tests/test_ffi_parity.py`) additionally
verifies that the Rust and Python implementations produce identical
`BoundaryState` for a battery of test inputs.

## Design Decisions

**Why two severities, not three?** Early prototypes had a `warning`
level between soft and hard. In practice, operators struggled to
distinguish warning from soft — both required human review but no
automated action. Collapsing to two levels simplified the supervisor
logic and eliminated ambiguous escalation paths.

**Why skip missing variables?** A boundary definition for a variable
that is not present in the values dict is silently skipped rather than
raising an error. This allows a single boundary spec to cover multiple
deployment configurations where some sensors may be absent. The
alternative (fail-loud on missing) was rejected because it would force
per-deployment boundary specs.

**Why format violations as strings?** Structured violation objects
were considered but add serialisation overhead for audit logging. The
string format is grep-friendly and sufficient for both human review
and automated parsing (the pattern is fixed and machine-readable).

## References

- Regime escalation logic: [regime_manager.md](regime_manager.md).
- Boundary severity configuration: binding spec schema.
- Audit trace format: [audit_trace.md](audit_trace.md).
- Event bus API: `scpn_phase_orchestrator.supervisor.events`.
