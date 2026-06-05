# Adapters

Bridges between SPO and external systems in the SCPN ecosystem,
observability platforms, and hardware controllers. Each adapter
translates SPO's internal representations (phases, coupling matrices,
regime states) into the wire format expected by the target system.

## Pipeline position

```
UPDEEngine ──→ UPDEState ──→ Adapters (output)
                                │
                  ┌─────────────┼──────────────────┐
                  ↓             ↓                  ↓
          SCPN Ecosystem   Observability      Hardware
          │                │                  │
          ├─ scpn_control  ├─ OpenTelemetry   ├─ Modbus/TLS
          ├─ fusion_core   ├─ Prometheus      └─ gRPC
          ├─ neurocore     └─ Grafana
          ├─ plasma_control
          ├─ quantum_control
          └─ snn_bridge

External Systems ──→ Adapters (input) ──→ Oscillator Extractors
```

Adapters are bidirectional: **output** adapters export SPO state to
external systems; **input** adapters import external signals for
phase extraction.

## SCPN Ecosystem Bridges

These adapters connect SPO to sibling packages in the SCPN ecosystem.
They share the Kuramoto/UPDE phase representation but differ in scope:

| Adapter | Target | Data flow |
|---------|--------|-----------|
| `scpn_control_bridge` | scpn-control (v0.18.0) | Bidirectional: phases, coupling, regime |
| `fusion_core_bridge` | SCPN-Fusion-Core (v3.9.3) | Export: sync metrics for fusion analysis |
| `neurocore_bridge` | sc-neurocore (v3.13.3) | Export: phase states for SNN processing |
| `plasma_control_bridge` | Plasma control systems | Import: magnetic diagnostics as P-channel |
| `quantum_control_bridge` | scpn-quantum-control (v0.9.1) | Export: coherence metrics for QPU scheduling |
| `hybrid_cocompiler` | Quantum + neuromorphic review package | Export: shared audit envelope for simulator handoff |
| `snn_bridge` | local SNN daemon | Export: local phase-dynamics signal packets |

### scpn-control Bridge

`SCPNControlBridge(scpn_config: dict)` — bidirectional adapter.

| Method | Signature | Description |
|--------|-----------|-------------|
| `import_knm` | `(scpn_knm: NDArray) → CouplingState` | Wrap external K_nm |
| `import_omega` | `(scpn_omega: NDArray) → NDArray` | Validate frequencies |
| `export_state` | `(upde_state: UPDEState) → dict` | Telemetry export |

`import_knm` validates non-empty finite real-valued square matrices with a zero
self-coupling diagonal. `import_omega` validates non-empty finite real-valued
1-D vectors with strictly positive natural frequencies. `export_state` produces
a dict with `regime`, `stability`, `layers` (each with R, ψ, lock signatures).

::: scpn_phase_orchestrator.adapters.scpn_control_bridge

### Fusion Core Bridge

`FusionCoreBridge` is a non-executing review bridge for
`scpn-fusion-core` equilibrium summaries. It maps positive q-profile
bounds, non-negative normalised beta, confinement time, sawtooth/ELM event
counts, and non-negative MHD amplitude into bounded phase channels. The
feedback path rejects empty phase vectors before computing the complex order
parameter, so exported `R_global`, mean phase, and mean frequency records stay
finite. Stability checks also reject negative beta and confinement-ratio
payloads instead of silently converting them into ordinary soft violations.

::: scpn_phase_orchestrator.adapters.fusion_core_bridge

### Plasma Control Bridge

`PlasmaControlBridge` is the non-actuating plasma telemetry boundary. It
accepts finite layer-coupling matrices, phase snapshots, Lyapunov review
scores, and invariant payloads only after rejecting boolean numeric aliases,
non-zero layer self-coupling, empty phase snapshots, negative beta and
Greenwald ratios, and non-positive safety-factor minima. This keeps the
Kronecker-expanded `K_nm` graph off-diagonal and prevents placeholder
phase-state exports from entering downstream review paths.

::: scpn_phase_orchestrator.adapters.plasma_control_bridge

### Quantum Control Bridge

::: scpn_phase_orchestrator.adapters.quantum_control_bridge

### Hybrid Co-Compiler

::: scpn_phase_orchestrator.adapters.hybrid_cocompiler

### SNN Bridge

`SNNControllerBridge` maps finite UPDE layer order-parameter magnitudes in
`[0, 1]` to LIF input currents, validates non-negative real-valued spike rates
before action projection, and rejects boolean or complex array aliases before
schedule-manifest generation.

::: scpn_phase_orchestrator.adapters.snn_bridge

### Neurocore Bridge

`NeurocoreBridge` maps bounded UPDE layer coherence to stochastic LIF input
currents, accepts only non-negative deterministic seeds, and validates
real-valued non-negative rate vectors from action inputs or Rust backend output
before producing coupling actions.

::: scpn_phase_orchestrator.adapters.neurocore_bridge

## Observability

Adapters for production monitoring and tracing.

### OpenTelemetry

Exports SPO metrics and traces to any OTLP-compatible backend
(Jaeger, Zipkin, Grafana Tempo). Requires `opentelemetry-api`.

**OTelExporter API:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `record_step` | `(upde_state, step_idx)` | Record metrics for one engine step |
| `record_regime_change` | `(old, new)` | Record regime transition event |

**Metrics exported:**

| Metric | Type | Description |
|--------|------|-------------|
| `spo.order_parameter` | Gauge | Current R value |
| `spo.regime` | Gauge | Current regime (0-3) |
| `spo.step_latency_ms` | Histogram | Engine step duration |
| `spo.coupling_mean` | Gauge | Mean K_nm value |

::: scpn_phase_orchestrator.adapters.opentelemetry

### Prometheus

Fetches Prometheus instant and range metrics as a validated telemetry input
boundary. The adapter rejects malformed decoded JSON, malformed result/sample
structures, non-finite JSON constants, boolean/negative/non-real sample
timestamps, and non-finite sample values before returning arrays or scalars.

**PrometheusAdapter API:**

```python
PrometheusAdapter(endpoint: str, timeout: float = 5.0)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `fetch_metric` | `(query, start, end, step) -> NDArray[np.float64]` | Fetch a range-vector metric as finite values |
| `fetch_instant` | `(query) -> float` | Fetch one instant-vector scalar |

::: scpn_phase_orchestrator.adapters.prometheus

### Metrics Exporter

Lightweight metrics export helpers used by services that do not need the full
OpenTelemetry adapter.

::: scpn_phase_orchestrator.adapters.metrics_exporter

### Redis Store

Optional Redis-backed state exchange for deployments that need shared runtime
state outside the local process.

::: scpn_phase_orchestrator.adapters.redis_store

## Hardware Adapters

### Modbus/TLS

Industrial control interface for power grids, HVAC, and manufacturing.
Translates ControlAction to Modbus register writes over TLS.

::: scpn_phase_orchestrator.adapters.modbus_tls

### Hardware I/O

Generic hardware I/O abstraction for digital/analogue outputs.

Hardware I/O sample buffers and simulated-board frequency configuration
accept only finite real sensor amplitudes and finite positive real frequencies;
boolean and complex aliases are rejected before buffering or synthetic EEG
generation so flags and phasors cannot enter real sensor channels.

::: scpn_phase_orchestrator.adapters.hardware_io

## Gaian Mesh Bridge

The  implements **Layer 12: Distributed Mesh** of the SCPN architecture. It provides decentralized inter-node synchronization via stateless UDP heartbeats.

Multiple independent instances of SPO running across different machines can "couple" together. Instead of exchanging raw (N)$ phases, nodes exchange their macroscopic **Order Parameters** ({global}, \Psi_{global}$). The bridge integrates the peer fields and translates them into external forcing parameters ($\zeta$, $\Psi$) for the local .

### Features
- **Stateless UDP Broadcasting:** Designed for high-frequency, loss-tolerant mesh topologies.
- **Topological Consensus:** Enables thousands of independent agents (drones, servers) to synchronize without a central command node.
- **Timeout-Aware:** Automatically drops stale peers from the mean-field calculation to prevent phantom drag.

Peer and local `psi` values are finite real phases on the circle; negative
finite phases are canonicalised modulo `2*pi` before mesh-drive computation.

::: scpn_phase_orchestrator.adapters.gaian_mesh_bridge

## LSL BCI Entrainment Bridge

The `LSLBCIBridge` implements **Phase 9: Biological Integration** of the SCPN
augmentation roadmap. It establishes a real-time feedback loop between human
neural oscillations and the phase orchestrator.

By utilizing the **Lab Streaming Layer (LSL)** protocol, the bridge can ingest
live EEG data from a wide range of hardware (OpenBCI, Muse, Neuralink, etc.).
It extracts the instantaneous phase of target brainwaves (e.g., Alpha or Gamma
rhythms) and provides them as input to the `ActiveInferenceAgent` for
predictive entrainment.

Captured samples must be finite real EEG amplitudes, not boolean aliases, and
LSL timestamps must be finite non-negative values before samples enter the
Hilbert phase buffer.

### Features
- **Real-Time Phase Extraction:** Uses Hilbert transforms on sliding windows
  to track neural phase state.
- **Hardware Agnostic:** Supports any EEG device with an LSL outlet.
- **Closed-Loop Feedback:** Enables the orchestrator to steer human brainwave
  coherence via adaptive auditory or visual stimulation targets.

::: scpn_phase_orchestrator.adapters.lsl_bci_bridge

## Remanentia Bridge

::: scpn_phase_orchestrator.adapters.remanentia_bridge

## Synapse Bridges

The synapse bridges translate phase-channel and coupling data into sibling
service contracts while keeping the normal audit path unchanged.
The channel bridge treats hub WebSocket frames as untrusted JSON: decoded
messages must use finite JSON values and unique object keys before sender,
type, or payload fields can affect phase-channel state.

::: scpn_phase_orchestrator.adapters.synapse_channel_bridge

::: scpn_phase_orchestrator.adapters.synapse_coupling_bridge
