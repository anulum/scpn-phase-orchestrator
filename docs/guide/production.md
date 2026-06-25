# Production Deployment

## Production Purpose and Risk Scope

## Reader intent for this guide

This guide is for teams that have already completed a replay-validated control
loop and now need to move under operational governance. It is not a first-step
tuning tutorial; it is the transition contract from validated simulation to service
runtime.

Treat every subsection as a governance checkpoint, and keep the audit/replay
surface active until the review policy explicitly allows external action.

This guide is for teams promoting SPO from a controlled simulation path to a
reviewable service path. The key control objective is not speed first, but
failure predictability and controlled blast radius.

SPO is suitable for production only when each layer in the path has clear
evidence and boundaries:

- container and dependency immutability for repeatable startup,
- explicit health checks for simulator and regime state,
- bounded actuator interfaces and explicit rate limits,
- durable audit logging plus replay checks,
- clear separation between simulation outputs and any external write action.

That structure is what makes the deployment claim auditable instead of merely
operational.

## Docker

The repository includes a production-ready `Dockerfile` with three stages:

1. **Rust builder** (`rust:1.83-slim`) — builds spo-kernel via maturin
2. **Python builder** (`python:3.12-slim`) — installs SPO + Rust wheel
3. **Production** (`python:3.12-slim`) — minimal image, non-root user

```bash
docker build -t spo .
docker run --rm spo run domainpacks/queuewaves/binding_spec.yaml --steps 1000
```

## Deployment readiness checkpoints

The production deployment path is treated as an evidence gate, not a one-step
promotion. Before external traffic reaches this path, operators should record:

- backend selection evidence (Python fallback or `spo-kernel`),
- reproducible image metadata (`--sha`-pinned base images and lockfiles),
- health endpoint behaviour under restart, degraded, and normal modes,
- deterministic audit replay command and retention policy,
- explicit actuator approval policy (policy DSL + projection + execution mapper).

These checkpoints are meant to separate simulation-only validation from any
runtime write path. The same checklist is reused in incident response:
if any checkpoint fails to produce a fresh artifact, suspend deployment and
continue in replay-only mode.

The image runs as non-root user `spo` (UID 1000) and includes a deep
HEALTHCHECK against `/api/health`. Base images are pinned by SHA digest
for reproducible builds.

### Production defaults

The container defaults are intentionally locked down for production:

- non-root runtime user (`spo`, UID 1000),
- pinned base-image digests for reproducibility,
- hash-locked dependency installation,
- healthcheck wired to `/api/health`,
- entrypoint constrained to SPO CLI execution.

For hardened deployments, keep these defaults and override only runtime
configuration (bind mounts, environment, network policy).

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

from scpn_phase_orchestrator.runtime.observability import OTelExporter
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

For a parallel protobuf event stream, add `--audit-stream`. The shared
simulation core flushes and verifies that event stream once at run end and
surfaces the result on the returned `SimulationResult`; it does not run an
O(n) stream scan before every control proposal.

```bash
spo run binding_spec.yaml \
  --steps 10000 \
  --audit /var/log/spo/run.jsonl \
  --audit-stream /var/log/spo/run.spoa
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

### Core API (`/api/health`)

The core REST API exposes a deep health endpoint that checks engine state,
R finiteness, and regime subsystem:

```json
GET /api/health → 200
{
  "status": "healthy",
  "checks": {
    "engine": "ok",
    "R_finite": "ok",
    "regime": "ok"
  }
}
```

If any subsystem degrades, `status` changes to `"degraded"` with the
failing check identified. Use as a Kubernetes readiness probe:

```yaml
readinessProbe:
  httpGet:
    path: /api/health
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 15
```

The Dockerfile HEALTHCHECK uses this endpoint to verify the server
is genuinely functional, not just importable.

### QueueWaves (`/api/v1/health`)

QueueWaves has its own health endpoint returning `{"status": "ok", "tick": N}`.
Use as a liveness probe:

```yaml
livenessProbe:
  httpGet:
    path: /api/v1/health
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 30
```

For batch `spo run`, health is indicated by the exit code (0 = success).

## Container Registry (GHCR)

Tagged releases are built, scanned, and pushed to GitHub Container Registry:

```bash
# Pull the latest release
docker pull ghcr.io/anulum/scpn-phase-orchestrator:latest

# Pull a specific version
docker pull ghcr.io/anulum/scpn-phase-orchestrator:0.5.0
```

The publish pipeline (`.github/workflows/publish.yml`) performs:

1. **Build** — multi-stage Dockerfile with Rust FFI + Python
2. **Scan** — Trivy and Grype checks for CRITICAL/HIGH CVEs (either gate blocks publish on failure)
3. **Push** — tagged version + `latest` to `ghcr.io/anulum/`

Container images include the Rust kernel for maximum performance.
The pure-Python fallback wheel is published separately to PyPI.

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

For hardening adapter traffic, deploy sane environment defaults:

```bash
export SPO_ADAPTER_RATE_LIMIT_PER_MINUTE=120
export QUEUEWAVES_API_KEY="$(openssl rand -hex 32)"
export PROMETHEUS_TOKEN_PATH=/run/secrets/prometheus/token

# TLS client certs for SCADA bindings come from mounted secret volume.
export SPO_MODBUS_TLS_CERT_PATH=/run/secrets/scada/client.pem
export SPO_MODBUS_TLS_KEY_PATH=/run/secrets/scada/client.key
export SPO_MODBUS_CA_CERT_PATH=/run/secrets/scada/ca.pem
```

Use host/port validation and rate-limit checks before each write path so malformed
production inputs fail fast with generic errors.

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
