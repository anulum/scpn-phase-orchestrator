# Adapters

Bridges between SPO and external systems in the SCPN ecosystem,
observability platforms, and hardware controllers. Each adapter
translates SPO's internal representations (phases, coupling matrices,
regime states) into the wire format expected by the target system.

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

Exports SPO metrics (R, regime, step latency) and traces (per-step
spans with phase snapshot attributes) to any OTLP-compatible backend
(Jaeger, Zipkin, Grafana Tempo). Requires `opentelemetry-api`.

::: scpn_phase_orchestrator.adapters.opentelemetry

### Prometheus

Exposes SPO metrics as a Prometheus scrape target. Gauges for R,
regime, and coupling strength; histograms for step latency.

::: scpn_phase_orchestrator.adapters.prometheus
