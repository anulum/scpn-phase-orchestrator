# Hardware & Deployment

SPO supports multiple deployment targets from browser to FPGA.

## Deployment decision model

Use this page as a lane-selection map, not a ranking list. All lanes share a common
control model and same binding/spec contract; they differ in latency envelope,
determinism profile, and operational dependency footprint.

Choose a lane by answering:

1. What is the maximum acceptable control loop latency?
2. Is hardware determinism required for every actuation cycle?
3. Is local/offline edge execution a hard constraint?
4. Are audit and replay requirements satisfied in the target path?

If a lane cannot satisfy one requirement directly, use a fallback lane with
explicitly documented performance trade-offs rather than changing core control logic.

## Why this matters for safety review

The same supervisory policy can run in all lanes, but only some lanes meet strict
timing and audit constraints for regulated deployment:

- **Rust/FFI and FPGA**: low-latency deterministic behavior.
- **JAX**: higher throughput when differentiable training or sensitivity analysis is a priority.
- **WASM and browser**: low-friction validation, training, and visualization use cases.
- **Docker + managed hosting**: strong packaging and CI reproducibility for team rollout.

Document the chosen lane in domainpack onboarding so operators can trace why an
implementation path was selected.

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

## Deployment lane guidance

This page spans multiple environments that differ in deployment constraints:

- **Edge / browser:** Wasm path for deterministic local execution and lightweight
  visualization.
- **Performance-sensitive services:** Rust/FFI path for lower-latency control loops.
- **GPU-rich workloads:** JAX path where batch throughput and differentiable backends
  are prioritized.
- **Deterministic industrial I/O:** FPGA path where fixed-latency closed-loop behaviour
  is required.

Use these lanes as a routing choice, not a replacement order. The same binding and
supervisor contracts are the baseline regardless of backend shape.

## Why fixed-latency paths matter

For control workloads that need bounded actuation windows, fixed-latency execution
paths reduce uncertainty in supervisory timing. The design intent is to preserve
correctness under latency pressure:

- deterministic control cadence,
- auditable fallback behavior when optional paths are missing,
- and a shared audit chain across lanes.

## Deployment readiness check

Before a production promotion, map each required surface to an explicit lane and
confirm the corresponding prerequisites:

1. install profile success,
2. optional dependency availability,
3. backend health checks,
4. deterministic audit export in a dry-run path.

This keeps deployment sign-off tied to evidence and avoids accidental rollout where
critical path assumptions are absent.

See: [Control Systems Guide](control_systems.md)

## Deployment evidence checks

Select a lane only after matching deployment assumptions to explicit checks:

- hardware determinism required vs. acceptable jitter,
- audit export availability for the active lane,
- and fallback behavior when optional dependencies are missing.

When a lane is promoted to service, keep the corresponding lane choice and
health checks in the release package. That avoids silent drift when same binding
specs are run on different infrastructure later.
