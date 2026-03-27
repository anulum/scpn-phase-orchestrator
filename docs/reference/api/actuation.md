# Actuation

The actuation subsystem converts high-level supervisor decisions into
bounded, rate-limited control commands. It sits between the supervisor
(which decides *what* to change) and the physical/virtual actuators
(which execute the change). This separation of concerns is critical
for safety: the supervisor can propose aggressive actions, but the
actuation layer enforces physical constraints.

## Architecture

```
Supervisor ─► ControlAction ─► ActuationMapper ─► ActionProjector ─► Actuator
                                    │                    │
                              knob routing         value clamping
                              limit enforcement    rate limiting
```

## Control Actions

A `ControlAction` is the universal message format between supervisor and actuators:

| Field | Type | Description |
|-------|------|-------------|
| `knob` | `str` | Parameter to change: `K`, `alpha`, `zeta`, or `Psi` |
| `scope` | `str` | Target: `global` (all layers) or `layer_{n}` (single layer) |
| `value` | `float` | Target parameter value |
| `ttl_s` | `float` | Time-to-live in seconds (action expires after this) |
| `justification` | `str` | Human-readable reason (logged to audit trail) |

## Actuation Mapper

Maps control actions to actuator-specific command dictionaries.
Each `ActuatorMapping` declares:

- **name** — actuator identifier (e.g. `"coupling_amplifier"`)
- **knob** — which control knob it responds to
- **scope** — which layer(s) it affects
- **limits** — `(lo, hi)` value bounds for this actuator

The mapper routes actions by knob and scope, and clamps values to
actuator limits. A global-scope action is routed to *all* actuators
that handle the specified knob.

### Usage

```python
from scpn_phase_orchestrator.actuation.mapper import ActuationMapper, ControlAction
from scpn_phase_orchestrator.binding.types import ActuatorMapping

mappings = [
    ActuatorMapping(name="K_amp", knob="K", scope="global", limits=(0.0, 5.0)),
    ActuatorMapping(name="zeta_drive", knob="zeta", scope="global", limits=(0.0, 1.0)),
]
mapper = ActuationMapper(mappings)

actions = [ControlAction(knob="K", scope="global", value=3.0, ttl_s=5.0,
                         justification="MPC pre-emptive boost")]
commands = mapper.map_actions(actions)
# [{"actuator": "K_amp", "knob": "K", "scope": "global", "value": 3.0, "ttl_s": 5.0}]
```

::: scpn_phase_orchestrator.actuation.mapper

## Action Projector

Safety layer that enforces value bounds and rate limits on control
actions before they reach actuators. Two constraints are applied:

1. **Value bounds** — absolute limits per knob (e.g. $K \in [0, 10]$)
2. **Rate limits** — maximum change per step per knob (e.g. $|\Delta K| \leq 0.1$)

Rate limits prevent discontinuous jumps that destabilise the phase
dynamics. A coupling strength that jumps from 0.1 to 5.0 in one
step can cause the Euler integrator to diverge. The projector
ensures smooth transitions.

### Rate Limit Semantics

Given previous value $v_{\text{prev}}$ and proposed value $v_{\text{new}}$:

$$v_{\text{projected}} = \text{clamp}\left(v_{\text{prev}} + \text{clamp}(\Delta, -r, r),\; v_{\min},\; v_{\max}\right)$$

where $\Delta = v_{\text{new}} - v_{\text{prev}}$ and $r$ is the rate limit.

Bounds and rate limits are empirical; see `docs/ASSUMPTIONS.md § Rate Limits`
for the derivation from CFL stability conditions.

::: scpn_phase_orchestrator.actuation.constraints
