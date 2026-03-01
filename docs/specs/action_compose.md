# Action Compose

## ControlAction

```python
@dataclass
class ControlAction:
    knob: str       # "K", "alpha", "zeta", or "Psi"
    scope: str      # "global" or "layer_{n}"
    value: float    # target value or delta
    ttl_s: float    # time-to-live in seconds
    justification: str
```

Valid knobs: `{"K", "alpha", "zeta", "Psi"}`.

## ActuationMapper

Maps `ControlAction` to domain-specific actuator commands.

Initialised with `list[ActuatorMapping]` from the binding spec:

```yaml
actuators:
  - name: coupling_knob
    knob: K
    scope: global
    limits: [0.0, 5.0]
  - name: damping_knob
    knob: zeta
    scope: global
    limits: [0.0, 2.0]
```

`map_actions(actions)` returns a list of command dicts:

```python
[
    {
        "actuator": "coupling_knob",
        "knob": "K",
        "scope": "global",
        "value": 0.5,   # clipped to limits
        "ttl_s": 10.0,
    }
]
```

Values are clamped to `[limits[0], limits[1]]`. Scope matching: a global action maps to all actuators for that knob. A layer-scoped action maps only to matching-scope actuators.

`validate_action(action)` returns True if the action's knob is valid and value is within limits for at least one matching actuator.

## ActionProjector

Rate-limits and clips control actions relative to the previous value:

```python
projector = ActionProjector(
    rate_limits={"K": 0.1, "alpha": 0.05, "zeta": 0.2},
    value_bounds={"K": (0.0, 5.0), "alpha": (-3.14, 3.14)},
)
projected = projector.project(action, previous_value=0.45)
```

1. Clamp value to `[lo, hi]`.
2. If `|value - previous| > rate_limit`, cap the delta at `rate_limit`.
3. Re-clamp after rate limiting.

Returns a new `ControlAction` with the projected value.

## Composition Order

```
SupervisorPolicy.decide()
    → list[ControlAction]
    → ActionProjector.project() (per action)
    → ActuationMapper.map_actions()
    → list[dict] (actuator commands)
    → Domain-specific execution
```

## References

Rate limits and value bounds are empirical — see [ASSUMPTIONS.md](../ASSUMPTIONS.md) § Rate Limits. Knob definitions follow the Sakaguchi–Kuramoto model documented in [knobs_K_alpha_zeta_Psi.md](../concepts/knobs_K_alpha_zeta_Psi.md).
