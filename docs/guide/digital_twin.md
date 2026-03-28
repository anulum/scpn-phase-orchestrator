# Digital Twin Pattern

SPO fits naturally into digital twin architectures as the **coherence
control layer** — the component that monitors synchronisation health
and intervenes when the physical system drifts.

## Architecture

```
Physical System ──► Sensors ──► Phase Extraction (P/I/S)
                                      │
                                      ▼
                              ┌── SPO Engine ──┐
                              │  UPDE dynamics  │
                              │  15+ monitors   │
                              │  Regime FSM     │
                              │  Supervisor     │
                              └───────┬─────────┘
                                      │
                              ┌───────┴─────────┐
                              │   Actuation      │
                              │   (K, α, ζ, Ψ)  │
                              └───────┬─────────┘
                                      │
                                      ▼
Physical System ◄── Actuators ◄── Control Commands
```

The digital twin runs SPO in parallel with the physical system.
Sensor data feeds the oscillator extractors; control commands flow
back through the actuation mapper.

## Implementation

```python
from scpn_phase_orchestrator.binding.loader import load_binding_spec
from scpn_phase_orchestrator.server import SimulationState

# 1. Load domain binding
spec = load_binding_spec("domainpacks/power_grid/binding_spec.yaml")
twin = SimulationState(spec)

# 2. Feed sensor data each tick
while True:
    sensor_data = read_sensors()       # your data source
    state = twin.step()                # advance twin
    R = state["R_global"]
    regime = state["regime"]

    # 3. Act on supervisor decisions
    if regime == "critical":
        increase_coupling()            # your actuator
    elif regime == "degraded":
        log_warning(R)

    # 4. Audit trail for compliance
    audit_logger.log_step(state)
```

## Deployment Options

| Mode | Latency | Use case |
|------|---------|----------|
| **Embedded** (Python library) | <1ms/step | Edge devices, PLCs |
| **REST API** (FastAPI server) | ~10ms | Cloud dashboards |
| **gRPC streaming** | ~5ms | High-frequency telemetry |
| **WASM** (browser) | ~1ms | Monitoring dashboards |
| **Docker** (full stack) | N/A | Production deployment |

## Real-World Examples

- **Tokamak plasma**: MHD mode coupling → SPO detects mode locking precursors
- **Power grid**: generator swing equations → SPO predicts cascade failures
- **Manufacturing**: SPC quality oscillations → SPO tunes process parameters
- **Traffic**: signal timing drift → SPO maintains green wave coherence

Each maps onto the same pattern: extract phases → run engine → monitor → act.
