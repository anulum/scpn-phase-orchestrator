# API Reference

Auto-generated from source docstrings via
[mkdocstrings](https://mkdocstrings.github.io/).
Python classes auto-delegate to Rust when the `spo_kernel` extension is
available; the public interface is identical in both backends.

If you are still deciding what SPO is for, start with the
[Use Cases and Value Map](../../getting-started/use_cases.md). This API index
is organised for implementation work after the domain, data source, and safety
boundary are known.

## API by Job

| Job | Primary API | Supporting docs |
|-----|-------------|-----------------|
| Run a reviewed binding from Python | [Python Facade](api.md) | [Quickstart](../../getting-started/quickstart.md) |
| Validate and resolve domain assumptions | [Binding](binding.md) | [New Domain Checklist](../../tutorials/01_new_domain_checklist.md) |
| Extract phase from data | [Oscillators](oscillators.md) | [From Raw Sources to Run](../../tutorials/05_from_raw_sources_to_run.md) |
| Simulate coupled dynamics | [UPDE](upde.md) | [UPDE Numerics](../../specs/upde_numerics.md) |
| Infer or build coupling | [Coupling](coupling.md) | [Build K_nm Templates](../../tutorials/03_build_knm_templates.md) |
| Detect coherence and instability | [Monitor](monitor.md) | [Analysis Toolkit](../../guide/analysis_toolkit.md) |
| Propose bounded control | [Supervisor](supervisor.md) and [Actuation](actuation.md) | [Production Guide](../../guide/production.md) |
| Replay and audit decisions | [Audit](audit.md) | [Deterministic Replay](../../tutorials/06_deterministic_replay_for_debugging.md) |
| Optimise differentiable oscillator models | [nn API](nn.md) | [Differentiable Kuramoto](../../tutorials/04_differentiable_kuramoto.md) |

## Public API entry point

```python
from scpn_phase_orchestrator import (
    AuditLogger,
    BifurcationDiagram,
    BindingSpec,
    BoundaryObserver,
    ControlAction,
    CouplingBuilder,
    Orchestrator,
    OrchestratorState,
    PhaseExtractor,
    PhaseState,
    QPUDataArtifact,
    RegimeManager,
    SPOError,
    SparseUPDEEngine,
    SheafUPDEEngine,
    StuartLandauEngine,
    SupervisorPolicy,
    UPDEEngine,
    compile_domain_to_qpu_artifact,
    emit_qpu_data_artifact,
    find_critical_coupling,
    lyapunov_spectrum,
    trace_sync_transition,
    validate_qpu_data_artifact,
)
```

The frozen top-level manifest is tracked in
[`public_api_manifest.txt`](../../specs/public_api_manifest.txt). Any change to
`scpn_phase_orchestrator.__all__` must update that manifest, this import block,
and release notes in the same change so semantic-versioning review can classify
the compatibility impact.

## Module index

| Module | Description |
|--------|-------------|
| [Python Facade](api.md) | High-level local simulation API for notebooks and applications |
| [Core & Exceptions](core.md) | Exception hierarchy, Rust/Python compat constants |
| [Binding](binding.md) | Configuration loading and validation for binding specs |
| [UPDE Engine](upde.md) | Kuramoto ODE, Stuart-Landau amplitude ODE, metrics, PAC |
| [Oscillators](oscillators.md) | Phase extraction: Physical, Informational, Symbolic channels |
| [Coupling](coupling.md) | K_nm matrix construction, geometry constraints, lag estimation |
| [Supervisor](supervisor.md) | Regime management, policy engine, Petri net FSM, event bus |
| [Monitor](monitor.md) | Boundary violation detection, coherence monitoring |
| [Actuation](actuation.md) | Control output mapping, action projection |
| [Imprint](imprint.md) | History-dependent coupling modulation |
| [Drivers](drivers.md) | External forcing functions (Physical, Informational, Symbolic) |
| [Audit](audit.md) | SHA256-chained audit logging and deterministic replay |
| [Reporting](reporting.md) | Matplotlib coherence visualizations |
| [Artifacts](artifacts.md) | Portable run artifacts and QPU data packaging/validation |
| [Distributed Sync](distributed.md) | Transport-neutral phase-vector gossip protocol for UPDE nodes |
| [Scaffold](scaffold.md) | Domainpack scaffolding, including fail-closed LLM-guided proposals |
| [Visualization](visualization.md) | Real-time WebXR/network/torus visualization surfaces |
| [Adapters](adapters.md) | Bridges to SCPN ecosystem, observability, SNN controllers |
| [Plugins](plugins.md) | Extension manifests and compatibility checks |
| [QueueWaves](queuewaves.md) | Real-time cascade failure detector for microservices |
| [SSGF](ssgf.md) | Self-Stabilizing Gauge Field: geometry carrier, ethical cost, PGBO, TCBO |
| [Autotune](autotune_sindy.md) | Auto-calibration: SINDy, coupling estimation, phase extraction, frequency ID, reward evaluation, replay policy search |
| [Meta-Transfer](meta.md) | Replay-backed cross-domain initial policy proposals |
| [CLI](../cli.md) | `spo` command-line interface |
