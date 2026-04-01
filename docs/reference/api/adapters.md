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
| `snn_bridge` | SNN daemon (04_ARCANE_SAPIENCE) | Export: reasoning stimuli from phase dynamics |

### scpn-control Bridge

`SCPNControlBridge(scpn_config: dict)` — bidirectional adapter.

| Method | Signature | Description |
|--------|-----------|-------------|
| `import_knm` | `(scpn_knm: NDArray) → CouplingState` | Wrap external K_nm |
| `import_omega` | `(scpn_omega: NDArray) → NDArray` | Validate frequencies |
| `export_state` | `(upde_state: UPDEState) → dict` | Telemetry export |

`import_knm` validates square matrix shape. `import_omega` validates
1-D and all-positive. `export_state` produces a dict with `regime`,
`stability`, `layers` (each with R, ψ, lock signatures).

::: scpn_phase_orchestrator.adapters.scpn_control_bridge

### Fusion Core Bridge

::: scpn_phase_orchestrator.adapters.fusion_core_bridge

### Plasma Control Bridge

::: scpn_phase_orchestrator.adapters.plasma_control_bridge

### Quantum Control Bridge

::: scpn_phase_orchestrator.adapters.quantum_control_bridge

### SNN Bridge

::: scpn_phase_orchestrator.adapters.snn_bridge

### Neurocore Bridge

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

Exposes SPO metrics as a Prometheus scrape target.

**PrometheusAdapter API:**

```python
PrometheusAdapter(endpoint: str, timeout: float = 5.0)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `update` | `(upde_state)` | Push current metrics |

::: scpn_phase_orchestrator.adapters.prometheus

## Hardware Adapters

### Modbus/TLS

Industrial control interface for power grids, HVAC, and manufacturing.
Translates ControlAction to Modbus register writes over TLS.

::: scpn_phase_orchestrator.adapters.modbus_tls

### Hardware I/O

Generic hardware I/O abstraction for digital/analogue outputs.

::: scpn_phase_orchestrator.adapters.hardware_io

## Gaian Mesh Bridge

The  implements **Layer 12: Distributed Mesh** of the SCPN architecture. It provides decentralized inter-node synchronization via stateless UDP heartbeats.

Multiple independent instances of SPO running across different machines can "couple" together. Instead of exchanging raw (N)$ phases, nodes exchange their macroscopic **Order Parameters** ({global}, \Psi_{global}$). The bridge integrates the peer fields and translates them into external forcing parameters ($\zeta$, $\Psi$) for the local .

### Features
- **Stateless UDP Broadcasting:** Designed for high-frequency, loss-tolerant mesh topologies.
- **Topological Consensus:** Enables thousands of independent agents (drones, servers) to synchronize without a central command node.
- **Timeout-Aware:** Automatically drops stale peers from the mean-field calculation to prevent phantom drag.

::: scpn_phase_orchestrator.adapters.gaian_mesh_bridge
