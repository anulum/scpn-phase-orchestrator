# API Reference

Auto-generated from source docstrings via
[mkdocstrings](https://mkdocstrings.github.io/).
Python classes auto-delegate to Rust when the `spo_kernel` extension is
available; the public interface is identical in both backends.

## Public API entry point

```python
from scpn_phase_orchestrator import (
    AuditLogger,
    BifurcationDiagram,
    BindingSpec,
    BoundaryObserver,
    ControlAction,
    CouplingBuilder,
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
| [Scaffold](scaffold.md) | Domainpack scaffolding, including fail-closed LLM-guided proposals |
| [Visualization](visualization.md) | Real-time WebXR/network/torus visualization surfaces |
| [Adapters](adapters.md) | Bridges to SCPN ecosystem, observability, SNN controllers |
| [Plugins](plugins.md) | Extension manifests and compatibility checks |
| [QueueWaves](queuewaves.md) | Real-time cascade failure detector for microservices |
| [SSGF](ssgf.md) | Self-Stabilizing Gauge Field: geometry carrier, ethical cost, PGBO, TCBO |
| [Autotune](autotune_sindy.md) | Auto-calibration: SINDy, coupling estimation, phase extraction, frequency ID, reward evaluation, replay policy search |
| [Meta-Transfer](meta.md) | Replay-backed cross-domain initial policy proposals |
| [CLI](../cli.md) | `spo` command-line interface |
