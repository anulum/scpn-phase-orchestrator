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
