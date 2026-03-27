# QueueWaves

Real-time cascade failure detector for microservice architectures.
QueueWaves models each service as a phase oscillator driven by its
request queue depth. When services desynchronise (queue depths diverge),
a cascade failure is developing. SPO detects this via R(t) drop and
alerts before the cascade reaches user-facing services.

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
