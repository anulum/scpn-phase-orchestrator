# System Overview

![Synchronization Manifold](../assets/synchronization_manifold.png)

## Core Thesis

Any system with coupled cycles maps onto Kuramoto phase dynamics. The orchestrator treats synchrony as a universal state-space: extract phases, integrate coupling, measure coherence, act on knobs.

## Pipeline

```
Domain Signals
    |
    v
Domain Binder -----> BindingSpec (YAML)
    |                   declares oscillators, layers, coupling, objectives
    v
Oscillator Extractors
    |  P: Hilbert phase from continuous waveform
    |  I: inter-event frequency from timestamps
    |  S: ring-phase from discrete state sequence
    v
UPDEEngine
    |  dtheta_i/dt = omega_i
    |              + sum_j K_ij sin(theta_j - theta_i - alpha_ij)
    |              + zeta sin(Psi - theta_i)
    |
    |  Methods: Euler (default), RK4
    |  Output: phases, R per layer, cross-layer alignment
    v
Supervisor (RegimeManager + SupervisorPolicy)
    |  Reads: R_good, R_bad, boundary state
    |  Decides: ControlActions on {K, alpha, zeta, Psi}
    |  Regime: NOMINAL / DEGRADED / CRITICAL / RECOVERY
    v
ActuationMapper + ActionProjector
    |  Maps ControlActions to domain-specific actuator commands
    |  Clips values, enforces rate limits
    v
Domain Actuators (external)
```

## Dual Objective: R_good / R_bad

The `ObjectivePartition` divides layers into two groups:

- **R_good** (good_layers): coherence to maximise. High R_good = healthy synchronisation.
- **R_bad** (bad_layers): coherence to suppress. High R_bad = pathological lock-in.

The supervisor seeks to raise R_good while lowering R_bad. This captures systems where some synchrony is desirable (coordinated service calls) and some is harmful (retry storms, seizure-like cascades).

## Domain-Agnostic Approach

The engine has no domain knowledge. All domain semantics live in the `BindingSpec`:

- Which signals are oscillators (P/I/S channel)
- How oscillators group into hierarchy layers
- What coupling template to use
- Which boundaries constitute violations
- What actuators exist

A new domain requires writing a binding spec and (optionally) custom extractors. No engine code changes.

## Key Data Structures

| Structure | Module | Purpose |
|-----------|--------|---------|
| `BindingSpec` | `binding.types` | Domain declaration |
| `PhaseState` | `oscillators.base` | Extracted phase per oscillator |
| `CouplingState` | `coupling.knm` | Knm + alpha matrices + active template |
| `UPDEState` | `upde.metrics` | R per layer, cross-layer alignment, regime |
| `BoundaryState` | `monitor.boundaries` | Violations (soft/hard) |
| `ControlAction` | `actuation.mapper` | Knob adjustment command |
| `ImprintState` | `imprint.state` | Memory imprint vector |

## Audit and Replay

Every step writes a JSONL record (timestamp, regime, layer states, actions). Deterministic replay from audit logs verifies reproducibility.

## References

- **[kuramoto1975]** Y. Kuramoto (1975). Self-entrainment of a population of coupled non-linear oscillators. *Lecture Notes in Physics* 39, 420–422.
- **[acebron2005]** J. A. Acebrón et al. (2005). The Kuramoto model: a simple paradigm for synchronization phenomena. *Rev. Mod. Phys.* 77, 137–185.
- **[sakaguchi1986]** H. Sakaguchi & Y. Kuramoto (1986). A soluble active rotater model showing phase transitions via mutual entertainment. *Prog. Theor. Phys.* 76, 576–581.
