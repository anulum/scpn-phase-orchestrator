# Actuation

The actuation subsystem converts high-level supervisor decisions into
bounded, rate-limited control commands. It sits between the supervisor
(which decides *what* to change) and the physical/virtual actuators
(which execute the change). This separation of concerns is critical
for safety: the supervisor can propose aggressive actions, but the
actuation layer enforces physical constraints.

## Pipeline position

```
SupervisorPolicy.decide()
         │
         ↓
  list[ControlAction]
         │
         ↓
  ActuationMapper.map_actions()     ← routing by knob + scope
         │
         ↓
  ActionProjector.project()         ← rate limit + value bounds
         │
         ↓
  Actuator commands (dict)          → Modbus/gRPC/HTTP output
         │
         ↓
  UPDEEngine.step(knm + ΔK, zeta + Δζ, ...)  ← next cycle
```

The actuation subsystem is the **output adapter** of the SPO pipeline.
Without it, supervisor decisions would be unbounded floating-point
values that could crash the integrator.

---

## Control Actions

A `ControlAction` is the universal message format between supervisor
and actuators:

### ControlAction (dataclass)

| Field | Type | Description |
|-------|------|-------------|
| `knob` | `str` | Parameter to change: `K`, `zeta`, `psi`, `alpha` |
| `scope` | `str` | Target: `global` or `layer_{n}` |
| `value` | `float` | Proposed parameter value |
| `ttl_s` | `float` | Time-to-live in seconds |
| `justification` | `str` | Human-readable reason (audit trail) |

### Knob semantics

| Knob | Engine parameter | Effect |
|------|-----------------|--------|
| `K` | Coupling strength K_ij | Increases/decreases synchronisation pull |
| `zeta` | External drive amplitude ζ | Damping or excitation |
| `psi` | External drive phase Ψ | Phase of external reference |
| `alpha` | Phase lag α_ij | Shifts preferred phase relationships |

---

## Actuation Mapper

Maps control actions to actuator-specific command dictionaries.

### ActuatorMapping (dataclass)

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Actuator identifier |
| `knob` | `str` | Which control knob it responds to |
| `scope` | `str` | Which layer(s) it affects |
| `limits` | `tuple[float, float]` | (lo, hi) value bounds |

### ActuationMapper

```python
ActuationMapper(mappings: list[ActuatorMapping])
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `map_actions` | `(actions: list[ControlAction]) → list[dict]` | Route actions to actuators |
| `validate_action` | `(action: ControlAction) → bool` | Check if any actuator handles this knob+scope |

### Routing rules

1. Action's `knob` must match an actuator's `knob`
2. Action's `scope` must match an actuator's `scope` (or `"global"` matches all)
3. Action's `value` is clamped to actuator's `limits`
4. Unroutable actions are silently dropped (no matching actuator)

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
```

### Edge cases

| Input | Behaviour |
|-------|-----------|
| Empty mappings | `map_actions()` returns [] for any input |
| Empty actions | Returns [] |
| No matching actuator | Action silently dropped |
| `validate_action()` on unroutable | Returns False |

**Performance:** `map_actions()` < 10 μs.

::: scpn_phase_orchestrator.actuation.mapper

---

## Action Projector

Safety layer that enforces value bounds and rate limits on control
actions before they reach actuators.

### Safety requirements

| ID | Requirement | Enforcement |
|----|-------------|-------------|
| SR-1 | Output value within [lo, hi] | Value bounds clamp |
| SR-2 | Maximum change ≤ rate_limit per step | Rate limit clamp |

### Constructor

```python
ActionProjector(
    rate_limits: dict[str, float],    # {"K": 0.1, "zeta": 0.05}
    value_bounds: dict[str, tuple[float, float]],  # {"K": (0.0, 1.0)}
)
```

### project()

```python
def project(action: ControlAction, previous_value: float) -> ControlAction
```

Given previous value v_prev and proposed value v_new:

```
Δ = v_new - v_prev
Δ_clamped = clamp(Δ, -rate_limit, +rate_limit)
v_projected = clamp(v_prev + Δ_clamped, v_min, v_max)
```

The returned ControlAction has the same knob, scope, ttl_s, and
justification — only `value` is modified.

### Rate limit motivation

Rate limits prevent discontinuous jumps that destabilise the phase
dynamics. A coupling strength that jumps from 0.1 to 5.0 in one
step can cause the Euler integrator to diverge (CFL violation).
The projector ensures smooth transitions.

### Consecutive-step guarantee

Over N consecutive steps, the maximum total change is bounded by
N × rate_limit. This provides a formal guarantee on the maximum
slew rate of any actuated parameter.

### Unbounded knobs

Knobs not in `rate_limits` or `value_bounds` pass through unmodified.
This allows domain-specific knobs to bypass the projector when safety
constraints are not applicable.

**Performance:** `project()` < 10 μs.

::: scpn_phase_orchestrator.actuation.constraints

---

## Closed-loop feedback example

```python
from scpn_phase_orchestrator.actuation.constraints import ActionProjector
from scpn_phase_orchestrator.actuation.mapper import ActuationMapper, ControlAction
from scpn_phase_orchestrator.supervisor.policy import SupervisorPolicy
from scpn_phase_orchestrator.supervisor.regimes import RegimeManager
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

# Setup
eng = UPDEEngine(n=8, dt=0.01)
pol = SupervisorPolicy(RegimeManager())
proj = ActionProjector(
    rate_limits={"K": 0.1, "zeta": 0.05},
    value_bounds={"K": (0.0, 5.0), "zeta": (0.0, 1.0)},
)

# Feedback loop
k_current = 0.5
zeta_current = 0.0
for _ in range(1000):
    phases = eng.step(phases, omegas, knm, zeta_current, 0.0, alpha)
    r, psi = compute_order_parameter(phases)
    state = build_upde_state(r, psi)
    actions = pol.decide(state, boundary)
    for a in actions:
        if a.knob == "K":
            safe = proj.project(a, previous_value=k_current)
            k_current = safe.value
        elif a.knob == "zeta":
            safe = proj.project(a, previous_value=zeta_current)
            zeta_current = safe.value
```

---

## Performance summary

| Operation | Budget | Notes |
|-----------|--------|-------|
| `ActuationMapper.map_actions()` | < 10 μs | Dict construction |
| `ActionProjector.project()` | < 10 μs | Two clamp operations |
| Full closed-loop overhead | < 70 μs | decide + project + map |
