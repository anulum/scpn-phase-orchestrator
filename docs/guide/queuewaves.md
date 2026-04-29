# QueueWaves Cascade Failure Detector

QueueWaves maps microservice health metrics onto Kuramoto oscillators and
detects cascade failures by monitoring phase synchronisation. When error-rate
oscillators lock in phase (a "retry storm"), QueueWaves fires alerts before
the cascade reaches user-facing endpoints.

## Architecture

```
PrometheusCollector ─┐
  (scrape PromQL)    │
                     ├──► PhaseComputePipeline ──► AnomalyDetector ──► WebhookAlerter
                     │       (UPDE step,             (3 anomaly         (Slack / generic
                     │        order params,            types)             webhook POST)
                     │        PLV matrix)
                     └──► REST API + WebSocket ──► Dashboard (HTML)
```

Each service maps to one oscillator. Services are grouped into hierarchy layers
(micro, meso, macro). `ConfigCompiler` converts `QueueWavesConfig` into a
standard SPO `BindingSpec`, so the full UPDE pipeline -- coupling, imprint,
policy, boundaries -- runs underneath.

## Configuration

`QueueWavesConfig` is loaded from YAML:

```yaml
prometheus_url: http://prometheus:9090
scrape_interval_s: 15.0
buffer_length: 64

services:
  - name: order-api
    promql: "rate(http_requests_total{service='order-api',code=~'5..'}[1m])"
    layer: micro
    channel: P
  - name: payment-svc
    promql: "rate(http_requests_total{service='payment-svc',code=~'5..'}[1m])"
    layer: micro
    channel: P
  - name: gateway-p99
    promql: "histogram_quantile(0.99, rate(http_request_duration_seconds_bucket{service='gateway'}[5m]))"
    layer: meso
    channel: P
  - name: total-throughput
    promql: "sum(rate(http_requests_total[5m]))"
    layer: macro
    channel: P
  - name: retry-budget
    promql: "retry_budget_remaining"
    layer: macro
    channel: RetryBudget
    extractor_type: event

thresholds:
  r_bad_warn: 0.50
  r_bad_critical: 0.70
  plv_cascade: 0.85
  imprint_chronic: 1.5
  cooldown_seconds: 300.0

coupling:
  strength: 0.50
  decay: 0.25

alert_sinks:
  - url: https://hooks.slack.com/services/T.../B.../xxx
    format: slack
  - url: https://ops.example.com/api/alert
    format: generic

server:
  host: 0.0.0.0
  port: 8080

security:
  mode: production
  api_key_env: QUEUEWAVES_API_KEY
  rate_limit_per_minute: 120
```

The production-ready template lives at
`domainpacks/queuewaves/queuewaves.production.yaml`. It sets
`security.mode: production`, `api_key_env: QUEUEWAVES_API_KEY`, and a positive
request rate limit, so deployments do not need to remember those fields from
scratch.

### Key Fields

| Field | Default | Meaning |
|-------|---------|---------|
| `prometheus_url` | required | Prometheus base URL |
| `scrape_interval_s` | 15.0 | Seconds between scrape cycles |
| `buffer_length` | 64 | Rolling buffer length per service |
| `services[].channel` | P | Binding channel ID. `P`, `I`, and `S` have default extractors; named channels are allowed. |
| `services[].extractor_type` | channel default | Required for named channels; accepts the normal binding extractor names and aliases. |
| `thresholds.r_bad_warn` | 0.50 | R_bad warning threshold |
| `thresholds.r_bad_critical` | 0.70 | R_bad critical threshold |
| `thresholds.plv_cascade` | 0.85 | PLV cascade propagation threshold |
| `thresholds.imprint_chronic` | 1.5 | Imprint chronic degradation threshold |
| `thresholds.cooldown_seconds` | 300.0 | Alert deduplication cooldown |
| `security.mode` | development | `production` requires API key auth and rate limiting |
| `security.api_key_env` | QUEUEWAVES_API_KEY | Environment variable containing the API key |
| `security.rate_limit_per_minute` | 120 | Per-key request limit in production mode |

### ConfigCompiler

`ConfigCompiler.compile(cfg)` converts `QueueWavesConfig` into a `BindingSpec`:

- Each layer group becomes a `HierarchyLayer`
- `micro` layer oscillators go to `bad_layers` (retry storms synchronise here)
- `meso`/`macro` go to `good_layers`
- Coupling matrix built with `base_strength` and `decay`
- Boundaries set from `r_bad_warn` (soft) and `r_bad_critical` (hard)

## Running

### Continuous server

```bash
export QUEUEWAVES_API_KEY="$(openssl rand -hex 32)"
spo queuewaves serve --config domainpacks/queuewaves/queuewaves.production.yaml
```

Starts a FastAPI application. Scrapes Prometheus on each interval, runs the
UPDE pipeline, detects anomalies, fires alerts, and pushes state over WebSocket.
When `security.mode: production` is set, REST and WebSocket clients must send
`X-API-Key: <value of QUEUEWAVES_API_KEY>`.

### One-shot check

```bash
spo queuewaves check --config queuewaves.yaml
```

Scrapes once, runs the pipeline, prints `R_good`/`R_bad`/regime. Exits 0 if
no anomalies, 1 if anomalies detected. Use in CI or cron for periodic health
checks.

## REST API

All endpoints are prefixed with `/api/v1/`.

| Method | Path | Returns |
|--------|------|---------|
| GET | `/api/v1/health` | `{"status": "ok", "tick": N}` |
| GET | `/api/v1/state` | Latest `PipelineSnapshot` dict |
| GET | `/api/v1/state/history?n=100` | Last N snapshots |
| GET | `/api/v1/anomalies` | Active anomaly list |
| GET | `/api/v1/services` | Per-service phase, omega, imprint |
| GET | `/api/v1/plv` | Cross-layer PLV matrix |
| GET | `/api/v1/metrics/prometheus` | Prometheus text exposition format |
| POST | `/api/v1/check` | One-shot scrape-and-analyze |

## WebSocket

Connect to `/ws/stream`. Messages are JSON:

```json
{"type": "tick", "data": {"r_good": 0.12, "r_bad": 0.55, "regime": "degraded", ...}}
{"type": "anomaly", "data": {"type": "retry_storm_forming", "severity": "warning", ...}}
```

The server pushes a `tick` message after each scrape cycle and an `anomaly`
message for each detected anomaly.

## Anomaly Types

| Type | Trigger | Severity |
|------|---------|----------|
| `retry_storm_forming` | `R_bad > r_bad_warn` | warning |
| `retry_storm_forming` | `R_bad > r_bad_critical` | critical |
| `cascade_propagation` | Any PLV pair `> plv_cascade` | warning |
| `chronic_degradation` | Service imprint `> imprint_chronic` | warning |

**retry_storm_forming**: micro-layer oscillators (error rates, retry counters)
are synchronising. High R_bad means services are failing in lockstep.

**cascade_propagation**: high phase-locking value between two layers means
failures are propagating across the hierarchy.

**chronic_degradation**: a service's imprint score has accumulated past
threshold, indicating sustained poor health (memory leaks, connection pool
exhaustion).

## Alerting

### Slack format

```json
{
  "attachments": [{
    "color": "#FF0000",
    "blocks": [{
      "type": "section",
      "text": {"type": "mrkdwn", "text": ":rotating_light: *retry_storm_forming* [critical]\nR_bad=0.75 > 0.70 -- retry storm imminent"}
    }]
  }]
}
```

### Generic webhook format

```json
{
  "type": "retry_storm_forming",
  "severity": "critical",
  "service": "cluster",
  "value": 0.75,
  "threshold": 0.70,
  "tick": 42,
  "message": "R_bad=0.750 > 0.7 -- retry storm imminent",
  "suppressed_count": 3
}
```

Cooldown deduplication prevents alert floods. The same anomaly type + service
combination is suppressed for `cooldown_seconds` (default 300s). The
`suppressed_count` field reports how many events were dropped since the last
firing.

## Dashboard

A single-file HTML dashboard is served at `/` (or `/static/dashboard.html`).
It connects to `/ws/stream` for live updates and renders:

- Phase wheel per oscillator
- R_good / R_bad gauges
- Anomaly log
- Service table with imprint history

## Production Deployment

```bash
export QUEUEWAVES_API_KEY="$(openssl rand -hex 32)"
spo queuewaves serve --config domainpacks/queuewaves/queuewaves.production.yaml
```

QueueWaves is single-process (shared pipeline state). Run behind nginx or
Traefik for TLS termination. The production template already sets
`security.mode: production`; the server refuses to start unless
`QUEUEWAVES_API_KEY` is present. For multiple clusters, run one QueueWaves
instance per Prometheus source.
