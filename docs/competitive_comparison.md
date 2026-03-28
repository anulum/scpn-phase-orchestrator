# What SPO Does That No One Else Can

This page documents specific capabilities that distinguish SPO from
every other oscillator / synchronisation library. Claims are grounded
in the actual codebase, not aspirations.

## Closed-Loop Supervisory Control

**SPO has it. No one else does.**

| Library | Simulation | Monitoring | Control loop | Supervisor |
|---------|:----------:|:----------:|:------------:|:----------:|
| **SPO** | 9 engines | 15+ monitors | Yes | Petri FSM + MPC |
| Brian2 | SNN sim | Spike monitors | No | No |
| TVB (The Virtual Brain) | Neural mass | Basic metrics | No | No |
| neurolib | Neural mass | R(t) | No | No |
| kuramoto (fabridamicelli) | Euler sim | None | No | No |
| jaxkuramoto | JAX sim | None | No | No |
| pyDSTool | Bifurcation | None | No | No |

Every other library stops at simulation. SPO closes the loop:
detect degradation → decide action → execute → verify.

## Domain Agnosticism

**SPO compiles any domain. Others are domain-locked.**

SPO treats the domain as a YAML binding spec. The engine, monitors,
supervisor, and actuation layer are universal. 32 domainpacks ship
out of the box.

No other library has this abstraction. Brian2 is for neurons. TVB
is for brains. pyDSTool is for ODEs. SPO is for *any coupled-cycle
system*.

## What Specifically Cannot Be Done Elsewhere

| Capability | SPO module | Closest alternative | Gap |
|-----------|-----------|-------------------|-----|
| Regime FSM with hysteresis | `supervisor/regimes.py` | Manual if-else | Formal state machine with cooldown |
| Petri net formal verification | `supervisor/petri_net.py` | None | Deadlock/liveness checkable |
| MPC via Ott-Antonsen | `supervisor/predictive.py` | None | 10-step predictive control |
| Three-factor Hebbian plasticity | `coupling/plasticity.py` | None | Coupling that learns |
| TE-adaptive coupling | `coupling/te_adaptive.py` | None | Causal-directed adaptation |
| Hodge decomposition | `coupling/hodge.py` | None | Gradient/curl/harmonic flow |
| SSGF cybernetic closure | `ssgf/closure.py` | None | Self-organising geometry |
| Imprint memory | `imprint/update.py` | None | History-dependent coupling |
| 3-channel P/I/S extraction | `oscillators/*.py` | None | Physical + Informational + Symbolic |
| SHA-256 audit replay | `audit/logger.py` | None | Deterministic, tamper-evident |
| Chimera detection | `monitor/chimera.py` | Manual | Automatic coherent/incoherent |
| 9 ODE engines | `upde/*.py` | 1-2 per library | Standard + SL + inertial + market + swarmalator + ... |
| Rust FFI + WASM + FPGA | `spo-kernel/` | None | Sub-μs performance path |

## Measured Advantages

From `examples/supervisor_advantage.py`:

- Supervised loop achieves higher R than open-loop at same coupling
- Recovery from fault injection: supervisor detects and compensates

From `test_physics_benchmarks.py`:

- Analytical predictions (K_c, √μ, OA) verified against simulation
- Cross-engine parity: 5 engine pairs agree on common scenarios

## When NOT to Use SPO

- **Pure SNN simulation**: use Brian2 or NEST (spike-level, not phase)
- **Connectome-only analysis**: use TVB (no control needed)
- **Simple ODE exploration**: use pyDSTool (lighter weight)
- **Production ML pipelines**: use standard MLOps (SPO is for physics)

SPO is for systems where you need to *control* synchronisation,
not just *observe* it.
