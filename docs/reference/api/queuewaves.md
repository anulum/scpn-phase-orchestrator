# QueueWaves

Real-time cascade failure detector for microservice architectures.
QueueWaves models each service as a phase oscillator driven by its
request queue depth. When services desynchronise (queue depths diverge),
a cascade failure is developing. SPO detects this via R(t) drop and
alerts before the cascade reaches user-facing services.

## Pipeline position

QueueWaves is a **complete SPO application** — it instantiates the
full pipeline from data collection to alerting:

```
Prometheus/StatsD (queue metrics)
         │
         ↓
   Collector.poll()
         │
         ↓
   InformationalExtractor → PhaseState[]
         │
         ↓
   CouplingBuilder.build() → K_nm
         │
         ↓
   UPDEEngine.step() → phases
         │
         ↓
   compute_order_parameter() → R
         │
         ↓
   RegimeManager.evaluate() → Regime
         │
         ↓
   detect_chimera() → chimera_index
         │
         ↓
   Alerter → Slack / PagerDuty / webhook
```

## Architecture

```
Collector → Pipeline → Detector → Alerter
   │            │          │          │
 Prometheus   phase     chimera    Slack/
 /StatsD    extraction  + regime   PagerDuty
            + UPDE      analysis   webhook
```

The collector polls queue metrics, the pipeline extracts phases and
runs UPDE integration, the detector evaluates synchronisation health,
and the alerter fires notifications when thresholds are crossed.

## Theory

Microservice queue depths oscillate around steady state. In normal
operation, these oscillations are loosely synchronised (R ≈ 0.4-0.7).
When a cascade develops:

1. **Upstream services** accumulate requests (queue depth rises)
2. **Downstream services** starve (queue depth falls)
3. R drops sharply as phase coherence breaks

QueueWaves detects the R drop 10-30 seconds before the cascade
reaches user-facing latency thresholds, enabling pre-emptive
circuit breaking or load shedding.

## Configuration

::: scpn_phase_orchestrator.apps.queuewaves.config

## Pipeline

::: scpn_phase_orchestrator.apps.queuewaves.pipeline

## Detector

::: scpn_phase_orchestrator.apps.queuewaves.detector

## Alerter

::: scpn_phase_orchestrator.apps.queuewaves.alerter

## Collector

::: scpn_phase_orchestrator.apps.queuewaves.collector

## Server

::: scpn_phase_orchestrator.apps.queuewaves.server
