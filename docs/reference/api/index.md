# API Reference

Auto-generated from source docstrings via
[mkdocstrings](https://mkdocstrings.github.io/).
Python classes auto-delegate to Rust when the `spo_kernel` extension is
available; the public interface is identical in both backends.

## Public API entry point

```python
from scpn_phase_orchestrator import (
    UPDEEngine, StuartLandauEngine, CouplingBuilder, BindingSpec,
    PhaseExtractor, PhaseState, RegimeManager, SupervisorPolicy,
    BoundaryObserver, AuditLogger, SPOError,
)
```

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
| [Adapters](adapters.md) | Bridges to SCPN ecosystem, observability, SNN controllers |
| [QueueWaves](queuewaves.md) | Real-time cascade failure detector for microservices |
| [SSGF](ssgf.md) | Self-Stabilizing Gauge Field: geometry carrier, ethical cost, PGBO, TCBO |
| [Autotune](autotune_sindy.md) | Auto-calibration: SINDy, coupling estimation, phase extraction, frequency ID |
| [CLI](../cli.md) | `spo` command-line interface |
