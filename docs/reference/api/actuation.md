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

## Output protocols

The actuation subsystem can drive multiple output protocols:

| Protocol | Adapter | Use case |
|----------|---------|----------|
| Modbus/TLS | `modbus_tls` | Industrial controllers (PLC, DCS) |
| gRPC | `grpc_service` | Distributed SPO nodes |
| HTTP/REST | `server` | Web dashboard, external APIs |
| Redis | `redis_store` | State persistence, pub/sub |
| Direct | In-process | Same-process engine feedback |

For in-process use (most common), the actuation output feeds
directly back into the next `UPDEEngine.step()` call without
any serialisation overhead.

## TTL (time-to-live) semantics

Each `ControlAction` carries a `ttl_s` field. The actuation layer
tracks active actions and expires them after TTL elapses. This
prevents stale control commands from persisting indefinitely if
the supervisor stops producing updates.

| TTL | Meaning |
|-----|---------|
| 1.0 s | Short-lived corrective action |
| 5.0 s | Standard policy action |
| 30.0 s | Sustained regime response |
| ∞ | Permanent override (not recommended) |

## Safety invariants

The actuation subsystem guarantees:

1. **Bounded output:** every actuated value is within [lo, hi]
2. **Bounded rate:** |Δv| ≤ rate_limit per step
3. **Monotonic convergence:** consecutive project() calls converge
   toward the proposed value at the rate limit
4. **No side effects:** project() is pure — same inputs → same output
5. **Metadata preservation:** project() only modifies `value`,
   all other ControlAction fields are immutable

---

## Performance summary

| Operation | Budget | Notes |
|-----------|--------|-------|
| `ActuationMapper.map_actions()` | < 10 μs | Dict construction |
| `ActionProjector.project()` | < 10 μs | Two clamp operations |
| Full closed-loop overhead | < 70 μs | decide + project + map |

## CFL stability interaction

The ActionProjector's rate limits interact with the CFL stability
condition `dt × (max_ω + max_K) < π`:

- If `rate_limit_K` is too large, a single step could violate CFL
- The recommended rate limit is `rate_limit_K ≤ π/(dt × N) - max_ω/N`
- The numerics module's `check_stability()` should be called after
  applying actuation to verify the new K_nm is stable

This is **not** enforced automatically — the rate limits in the binding
spec must be chosen to be CFL-compatible. The `docs/ASSUMPTIONS.md`
file documents the derivation.

## Relationship to other subsystems

| Subsystem | Interaction |
|-----------|-------------|
| Supervisor | Produces ControlAction list |
| Binding | Declares ActuatorMapping list |
| Numerics | CFL check after actuation |
| Audit | Logs every actuation command |
| Engine | Consumes modified K_nm, ζ, Ψ |

## HDL Synthesis Compiler

The `KuramotoVerilogCompiler` provides a path from high-level topological
learning to **hard real-time hardware execution**. It compiles a
stabilized Kuramoto network ($K_{nm}$, $\omega$) directly into structural
Verilog code.

### Use Case: Nanosecond-Scale Control

In applications like **Nuclear Fusion Plasma Control**, the latency
requirements for suppressing tearing modes are in the microsecond or
nanosecond range. Even the high-performance Rust kernel running on a
standard CPU may introduce jitter due to OS context switching.

By compiling the synchronization manifold into an FPGA bitstream, the
control logic is executed in parallel hardware, achieving:
- **Zero Jitter:** Deterministic execution timing.
- **Nanosecond Latency:** Direct mapping of the UPDE integration loop to gates.
- **Massive Parallelism:** All $O(N^2)$ interactions computed concurrently.

### Implementation Details

The compiler generates a structural Verilog module that implements:
1. **State Registers:** Fixed-point or floating-point registers for each $\theta_i$.
2. **Interaction Matrix:** Parallel instantiation of sine-calculators (CORDIC or LUT).
3. **Euler Integration:** Single-clock cycle updates for the entire manifold.

::: scpn_phase_orchestrator.actuation.hdl_compiler
