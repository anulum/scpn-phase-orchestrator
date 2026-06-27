# Interfaces

The public surfaces through which SPO is driven. All are thin wrappers over the
same core pipeline (`binding → coupling → upde → monitor → supervisor →
actuation → audit`).

## 1. Python library API

The stable import surface (`api.py`, re-exported from the package root):

| Symbol | Purpose |
|--------|---------|
| `Orchestrator` | High-level Kuramoto runner; `Orchestrator.from_yaml(path)`, `.run(steps, seed) → OrchestratorState`. |
| `OrchestratorState` | Immutable final-state record (phases, omegas, K_nm, alpha, order parameter, mean phase, sample period). |
| `evaluate_binding_spec(spec, …)` | Full-fidelity evaluation of any spec → `SimulationResult`. |

The package root additionally re-exports the building blocks: `BindingSpec`,
`CouplingBuilder`, `UPDEEngine` / `StuartLandauEngine` / `SparseUPDEEngine` /
`SheafUPDEEngine`, `BoundaryObserver`, `RegimeManager`, `SupervisorPolicy`,
`AuditLogger`, `ControlAction`, `PhaseExtractor`, `PhaseState`,
`BifurcationDiagram`, `lyapunov_spectrum`, and related helpers. This is the
import contract for sibling repositories.

## 2. CLI (`spo`)

The CLI is a package (`runtime/cli/`, ~15 modules + a plugins group), not a
single file. Primary verbs (each registered in its module):

`run`, `validate`, `inspect`, `auto-bind`, `auto-coupling-estimation`,
`replay`, `watch`, `scaffold`, `generate`, `formal-export`, `policy-dry-run`,
`quickstart`, `koopman-mpc`, `assurance-case`, `twin-confidence`, `doctor`,
`meta-transfer-manifest`, plus `digital-twin-*` bundle/dashboard/playbook
commands and a `queuewaves serve|check` subgroup.

A `plugins` command group exposes lifecycle / storage / scheduler / execution /
revocation / supervisor subcommands.

I/O: a domainpack YAML + flags in; a SHA-256-chained JSONL audit log and a console
summary out.

## 3. REST (FastAPI)

`runtime/server.py`. Nine routes; dataclass models (not Pydantic); no OpenAPI
schema is generated. Optional `X-API-Key` auth (when `SPO_API_KEY` is set) and
optional per-minute rate limiting.

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | HTML dashboard |
| GET | `/api/state` | Current simulation snapshot |
| GET | `/api/studio-feed` | Read-only `studio.control-feed.v1` live feed |
| GET | `/api/config` | Domain / oscillator configuration |
| GET | `/api/metrics` | Order parameter and regime metrics |
| GET | `/api/health` | Health probe |
| POST | `/api/step` | Advance the simulation (auth) |
| POST | `/api/reset` | Reset state (auth) |
| WS | `/ws/stream` | Streaming phase telemetry |

## 4. gRPC

`runtime/server_grpc.py` + `grpc_gen/` stubs. Falls back to a hand-written
dataclass message layer when `protobuf` is absent.

| RPC | Type | Request → Response |
|-----|------|--------------------|
| `GetState` | unary | `StateRequest → StateResponse` |
| `Step` | unary | `StepRequest(n_steps) → StateResponse` |
| `Reset` | unary | `ResetRequest → StateResponse` |
| `GetConfig` | unary | `ConfigRequest → ConfigResponse` |
| `StreamPhases` | server-stream | `StreamRequest(max_steps, interval_s) → stream StateResponse` |

`StateResponse`: `step`, `R_global`, `regime`, repeated `LayerState`,
`amplitude_mode`, `mean_amplitude`. Optional API-key + rate limiting via
`SPO_GRPC_*` environment variables.

## 5. STUDIO surface

`studio/` exposes builder functions (`build_canvas_graph`,
`build_runtime_snapshot`, `build_deployment_readiness`, `run_binding_spec_replay`,
`build_studio_control_feed`, …) and a registry of 13 review panels. All panels
are `execution_disabled=True` and `operator_review_required=True`. The surface
emits Python-dataclass `ExportManifest` records and a read-only
`studio.control-feed.v1` envelope for live STUDIO ingestion. The schema-A
federation manifest is local-first, reports measured and curated evidence, and
ships no UI module while external fleet acceptance remains pending. See
[subsystems/studio-reporting.md](subsystems/studio-reporting.md).

## 6. Reporting and visualisation

- `reporting/` — `CoherencePlot` (Matplotlib PNG/SVG), narrative explainability,
  tabular summaries, and an operator-copilot advisor. Real, functional.
- `visualization/` — JSON serialisers for a Three.js torus (`torus_points_json`,
  `phase_wheel_json`), a D3 network graph (`network_graph_json`,
  `coupling_heatmap_json`), and a WebSocket frame streamer. The client-side
  JavaScript consuming these payloads lives outside this repository.

## 7. Deployment targets

CLI / Python library / FastAPI + WebSocket (QueueWaves app) / Docker + Helm /
JAX GPU (`[nn]`) / gRPC streaming / WASM (browser) / FPGA Verilog (Zynq-7020,
unsynthesised). Availability and maturity vary — see `backends.md`.
