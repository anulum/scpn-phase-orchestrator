# Hardware & Deployment

SPO supports multiple deployment targets from browser to FPGA.

## Rust FFI Ecosystem

12 PyO3 bindings providing native-speed implementations of core modules:

| Binding | Purpose |
|---------|---------|
| PyUPDEStepper | ODE integration (euler, rk4, rk45) |
| PyCouplingBuilder | Matrix construction + projection |
| PyRegimeManager | FSM regime transitions with hysteresis |
| PyCoherenceMonitor | R_good, R_bad, phase-lock detection |
| PyBoundaryObserver | Boundary violation detection |
| PyImprintModel | History-dependent modulation |
| PyActionProjector | Rate-limiting and value bounding |
| PyLagModel | Phase lag matrix management |
| PySupervisorPolicy | Rule engine execution |
| PyStuartLandauStepper | Amplitude dynamics |
| PyPetriNet | Formal state machine |

Install: `pip install spo-kernel` or build from `spo-kernel/` with maturin.

## FPGA Kernel (Sub-15μs Real-Time)

Hardware-accelerated Kuramoto solver for Xilinx Zynq-7020:

- 16-oscillator system
- Fixed-point Q16.16 arithmetic
- 4-stage pipelined CORDIC for sin/cos
- 240 evaluations/step at 100 MHz ≈ 15 μs/step
- AXI-Lite runtime reconfiguration
- Resource: ~8K LUTs, ~4K FFs, 1 BRAM, 2 DSP48 (45% utilization)

Deterministic latency for closed-loop control (BCI, DBS, power grid).

Source: `spo-kernel/crates/spo-fpga/`

## WebAssembly (Browser-Based)

Self-contained Kuramoto in any browser or edge runtime via WASM:

```javascript
const spo = await init();
spo.init(16);  // 16 oscillators
const R = spo.step(omegas_json, coupling, dt);
const phases = spo.get_phases();  // JSON array
```

No server needed. Enables real-time visualization, educational demos,
and distributed computing on edge devices.

Source: `spo-kernel/crates/spo-wasm/`

## JAX GPU Acceleration

The `nn/` module runs on GPU transparently via JAX. On Linux (or WSL2):

```bash
pip install scpn-phase-orchestrator[nn]
pip install jax[cuda12]  # Linux only — Windows needs WSL2
```

XLA compilation happens once per function signature, then runs at
native GPU speed. All `nn/` functions are JIT-compiled.

## Docker Deployment

```dockerfile
FROM python:3.12-slim
RUN pip install scpn-phase-orchestrator[full]
# Or with Rust acceleration:
RUN pip install scpn-phase-orchestrator[full] spo-kernel
```

## Production Monitoring

The audit trail + supervisor + regime manager provide production-grade
observability:

- SHA256-chained audit log (deterministic replay)
- Regime FSM with hysteresis (no state oscillation)
- Boundary observer with configurable thresholds
- Prometheus-compatible metrics export

See: [Control Systems Guide](control_systems.md)
