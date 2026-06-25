# Subsystem: `adapters` / `apps` / `grpc_gen` — ecosystem bridges & applications

Connects SPO to sibling repositories, hardware, and external protocols.
`adapters` 23 files (~7.9k LOC), `apps` 8, `grpc_gen` 5.

## Ecosystem bridges

Sibling bridges are **not hard dependencies**. Most are pure data-shape
validators or HTTP clients; only `QuantumControlBridge` performs a real (lazy)
import of a sibling, and only when a quantum method is called.

| Bridge | Target | Coupling |
|--------|--------|----------|
| `SCPNControlBridge` | `scpn-control` | wire format only (no import) |
| `PlasmaControlBridge` | plasma physics | wire format only; local physics thresholds |
| `QuantumControlBridge` | `scpn-quantum-control` | lazy import on `solve_q_upde` / `build_openqasm_manifest` |
| `FusionCoreBridge` | `scpn-fusion-core` | wire format only |
| `RemanentiaBridge` | `remanentia` | HTTP client (localhost:8001) |
| `Synapse*Bridge`, `NeurocoreBridge`, `SNNControllerBridge` | `sc-neurocore` | validator / HTTP |
| `GaianMeshNode` | mesh gossip | HTTP |

## Hardware & protocols

Lazy-imported, environment-gated, read-only where applicable: BrainFlow (EEG),
Modbus / Modbus-TLS (IEC 62443), OPC-UA, MQTT, Lab Streaming Layer, Redis,
Prometheus (urllib), OpenTelemetry (delegates to `runtime.observability`).

## Published cross-repo wire formats

The interlock surface siblings and STUDIO consume:

- **K_nm handshake** — `(N, N)` array, zero diagonal, finite.
- **Phase-gossip JSON** — `{regime, stability, layers:[{R, psi, locks}], cross_alignment}`.
- **`quantum_compiler_manifest` v1** — frequency/coupling terms, OpenQASM,
  `manifest_sha256`, `qpu_execution_permitted=False`.
- **`scpn_quantum_target_readiness_v1`** — blocked reasons, readiness level.
- **Plasma physics-invariant violations** — `{variable, value, threshold, severity}`.
- **Coherence-memory snapshot** — for Remanentia consolidation.
- **`PhaseState`** — the canonical phase record.

Contracts are unidirectional wire formats (manifest publication, physics
verdicts, HTTP gossip), not synchronous RPC.

## Apps

`apps/queuewaves/` — a cascade-failure detector for distributed systems: a
Prometheus collector → phase pipeline → anomaly detector (retry-storm / cascade /
chronic) → webhook alerter, served by FastAPI + WebSocket. Standalone and opt-in.

## Wiring & scope boundaries

Adapters are **not wired into the core simulation loop or server by default**:
callers instantiate them explicitly, and production deployments must keep their
endpoint, mode, and version in audit metadata. `spo doctor` now reports the two
package-local review/export adapter surfaces that should always ship with the
package:

- `fmi-cosimulation` — `CoSimulationSlave`, `generate_model_description`,
  `write_fmu`, and `cosimulate` for FMI 3.0 co-simulation export/review.
- `hybrid-cocompiler` — manifest, readiness, and operator handoff builders for
  non-executing neuromorphic/quantum co-compilation review.

There is no live `FMICosimulationBridge` or `HybridCocompilerBridge` class. The
FMI surface is a co-simulation slave/export package, and the hybrid surface is a
deterministic manifest builder with execution permissions forced false.
`grpc_gen` and the OpenTelemetry/metrics modules are forwarding aliases to the
`runtime` layer.
