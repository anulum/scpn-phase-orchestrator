# Production Deployment

## Docker

Build a multi-stage image:

```dockerfile
FROM python:3.12-slim AS base
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir .

FROM base AS runtime
COPY src/ src/
COPY domainpacks/ domainpacks/
ENTRYPOINT ["spo"]
```

```bash
docker build -t spo .
docker run --rm spo run domainpacks/queuewaves/binding_spec.yaml --steps 1000
```

For Rust acceleration, add a Rust build stage:

```dockerfile
FROM rust:1.75-slim AS rust-build
WORKDIR /build
COPY spo-kernel/ spo-kernel/
RUN cd spo-kernel && cargo build --release -p spo-ffi

FROM python:3.12-slim AS runtime
COPY --from=rust-build /build/spo-kernel/target/release/libspo_ffi.so /usr/local/lib/
# ... install Python package ...
```

## QueueWaves Server

Run behind uvicorn with a reverse proxy:

```bash
spo queuewaves serve --config /etc/queuewaves/config.yaml --host 127.0.0.1 --port 8080
```

Or directly with uvicorn for more control:

```bash
uvicorn scpn_phase_orchestrator.apps.queuewaves.server:create_app \
    --factory --host 127.0.0.1 --port 8080 --workers 1 --log-level info
```

QueueWaves is single-process (shared pipeline state). Do not use multiple
workers. Place behind nginx or Traefik for TLS, rate limiting, and static file
serving.

Example nginx config:

```nginx
upstream queuewaves {
    server 127.0.0.1:8080;
}

server {
    listen 443 ssl;
    server_name qw.example.com;

    location / {
        proxy_pass http://queuewaves;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

The `Upgrade`/`Connection` headers are required for WebSocket passthrough.

## OpenTelemetry

Configure the `OTelExporter` to send traces and metrics to a collector:

```python
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry import trace, metrics

trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(
    BatchSpanExporter(OTLPSpanExporter(endpoint="http://jaeger:4317"))
)
metrics.set_meter_provider(MeterProvider())

from scpn_phase_orchestrator.adapters import OTelExporter
exporter = OTelExporter(service_name="spo-production")
```

Exported metrics:

| Metric | Type | Description |
|--------|------|-------------|
| `spo.r_global` | gauge | Global Kuramoto order parameter R |
| `spo.stability_proxy` | gauge | Mean R across layers |
| `spo.steps_total` | counter | Total UPDE integration steps |

Spans: `spo.regime_change` with attributes `spo.regime.old`, `spo.regime.new`.

Without the `otel` extra installed, `OTelExporter` silently discards all
spans and metrics (no-op fallback).

## Prometheus

QueueWaves exposes a `/api/v1/metrics/prometheus` endpoint in text exposition
format:

```
queuewaves_r_good 0.123456
queuewaves_r_bad 0.654321
queuewaves_regime{name="degraded"} 1
queuewaves_tick 42
queuewaves_phase{service="order-api"} 3.141592
queuewaves_imprint{service="order-api"} 0.523000
```

Add this endpoint as a Prometheus scrape target. Build Grafana dashboards from
`queuewaves_r_good`, `queuewaves_r_bad`, and per-service `queuewaves_phase`
/ `queuewaves_imprint` series.

## Audit Logging

Enable audit logging for deterministic replay and compliance:

```bash
spo run binding_spec.yaml --steps 10000 --audit /var/log/spo/run.jsonl
```

The audit log is a JSONL file containing:

- Header record (oscillator count, dt, seed, amplitude mode)
- Per-step records (phases, omegas, knm, alpha, zeta, psi, state, actions)
- Event records (regime transitions, boundary violations)
- SHA-256 hash chain for tamper detection

Verify integrity and determinism:

```bash
spo replay /var/log/spo/run.jsonl --verify
```

This re-executes the simulation from the header parameters and checks that
every transition matches the logged state. Hash chain integrity is verified
independently.

Write audit logs to a persistent volume. Do not write to ephemeral container
storage.

## Health Checks

QueueWaves `/api/v1/health` returns 200 with `{"status": "ok", "tick": N}`.
Use as a Kubernetes liveness probe:

```yaml
livenessProbe:
  httpGet:
    path: /api/v1/health
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 30
```

For batch `spo run`, health is indicated by the exit code (0 = success).

## Security

- **AGPL compliance**: all deployments must make source available per AGPL v3
  terms. Commercial licensing available for proprietary use.
- **No secrets in binding specs**: binding YAML files are checked into
  version control. Prometheus URLs with auth tokens should use environment
  variable substitution or a secrets manager.
- **Audit log integrity**: the SHA-256 hash chain detects tampering. Store
  logs on append-only storage for compliance workloads.
- **WebSocket**: QueueWaves WebSocket has no built-in authentication. Add
  auth at the reverse proxy layer.

## Scaling

One SPO instance per domain. The UPDE engine is stateful (phases, coupling,
imprint) and not designed for horizontal scaling within a single domain.

For QueueWaves monitoring multiple clusters, run one instance per Prometheus
source. Each instance maintains its own phase state and alert cooldown timers.

## Monitoring Checklist

Track these signals via OTel gauges or Prometheus metrics:

| Signal | Source | Alert condition |
|--------|--------|-----------------|
| `R_global` | order parameter | Sustained drop below 0.3 |
| Regime | `RegimeManager.current` | Transition to `critical` |
| Boundary violations | `BoundaryObserver` | Any `hard` violation |
| Policy actions | `SupervisorPolicy.decide()` | Action rate spike |
| Step latency | wall clock per step | >20% above baseline |
| Audit log size | file system | Disk usage threshold |
