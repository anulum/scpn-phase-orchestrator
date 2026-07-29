# From Notebook to Production

SPO supports the full lifecycle from exploratory analysis to
production deployment. This guide traces the path.

## Transition Principle

Moving from notebook analysis to production is not a command swap; it is a
control-boundary upgrade.

The notebook phase optimises understanding and hypotheses. The production phase
adds three non-negotiable properties:

- **Reproducibility**: every result can be re-run from deterministic inputs,
- **Reviewability**: every intervention path has explicit policy and rate controls,
- **Auditability**: every run can be replayed and traced to decision events.

Treat the transition itself as risk control, not convenience plumbing.

## Stage 1: Explore (Notebook)

Start with a Jupyter notebook. Load data, extract phases, run the engine.

```python
# notebooks/explore_my_domain.ipynb
from scpn_phase_orchestrator.binding.loader import load_binding_spec
from scpn_phase_orchestrator.runtime.server import SimulationState

spec = load_binding_spec("domainpacks/my_domain/binding_spec.yaml")
sim = SimulationState(spec)

R_history = []
for _ in range(1000):
    state = sim.step()
    R_history.append(state["R_global"])

# Plot, analyse, iterate on binding spec
import matplotlib.pyplot as plt
plt.plot(R_history)
plt.ylabel("R"); plt.xlabel("Step")
plt.show()
```

21 example notebooks ship in `notebooks/`. Start from the closest
domain and modify.

## Stage 2: Validate (CLI)

Once the binding spec is tuned, validate and run from the CLI.

```bash
# Validate binding spec
spo validate domainpacks/my_domain/binding_spec.yaml

# Run batch simulation
spo run domainpacks/my_domain/binding_spec.yaml \
    --steps 10000 --seed 42 --audit audit.jsonl

# Replay for determinism check
spo replay audit.jsonl --verify
```

## Stage 3: Integrate a service surface

The generic REST/WebSocket surface is a library factory rather than a `spo`
subcommand. Create an ASGI module explicitly so the binding path is reviewable:

```python title="app.py"
from scpn_phase_orchestrator.runtime.server import create_app

app = create_app("domainpacks/my_domain/binding_spec.yaml")
```

```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --workers 1

# Or with full stack (Redis + Prometheus + Grafana)
cd deploy && docker compose up
```

The gRPC package exposes `PhaseStreamServicer` for an integration-owned
`grpc.Server`; SPO does not currently ship a top-level gRPC launcher command.

Endpoints:
- `GET /api/state` — current R, regime, phases
- `GET /api/studio-feed` — live `studio.control-feed.v1` envelope for STUDIO ingestion
- `GET /api/health` — deep health check
- `GET /api/metrics` — Prometheus exposition
- `WS /ws/stream` — real-time WebSocket observer

## Stage 4: Monitor (Observability)

Connect to your existing monitoring stack.

```python
# OpenTelemetry integration
from scpn_phase_orchestrator.runtime.observability import OTelExporter
exporter = OTelExporter(service_name="spo-production")
```

Prometheus scrapes `/api/metrics`. Grafana dashboards visualise
R(t), regime transitions, and per-layer coherence.

## Stage 5: Harden (Production)

- **Audit logging**: SHA-256 chained JSONL for regulatory compliance
- **Deterministic replay**: reproduce any incident from the audit trail
- **Container scanning**: Trivy reports configured vulnerability severities in CI
- **Health checks**: `/api/health` verifies engine + R + regime subsystems
- **Rate limiting**: actuation projector prevents discontinuous jumps

### Stage 6: Evidence Gate for Live Use

Before routing a run to any external control surface, keep the following checks
as hard requirements:

- fixed seed and deterministic inputs recorded in metadata,
- full CLI `spo validate` success on the deployment binding,
- replay verification of at least one pre-prod run,
- policy and safety boundaries confirmed for the target regime profile,
- health and metrics continuity for the intended traffic pattern.

If any one of these checks is missing, keep that surface in review-only mode.

This gate is the operational difference between a technical demo and an
auditable deployment path.

## Stage 7: Scale

- **Single instance:** benchmark the exact engine, monitors, audit mode, and
  host at the intended oscillator count.
- **Rust kernel:** compare the selected native hot paths with the Python
  reference on the deployment workload; local speedups are not portable.
- **JAX:** benchmark differentiable batches separately from the default runtime
  loop and record device, precision, compilation, and transfer costs.
- **Docker/Helm:** validate capacity, liveness/readiness behaviour, rollback,
  and resource limits under representative load.

## Checklist

- [ ] Binding spec validated (`spo validate`)
- [ ] Notebook analysis confirms R dynamics match domain
- [ ] Policy rules cover regime transitions
- [ ] Audit logging enabled for production runs
- [ ] Prometheus + Grafana dashboards configured
- [ ] Container image scanned (Trivy)
- [ ] Health check endpoint verified
- [ ] Deterministic replay tested
