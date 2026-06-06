# What SPO Does That No One Else Can

This page documents specific capabilities that distinguish SPO from
every other oscillator / synchronisation library. Claims are grounded
in the actual codebase, not aspirations.

## Evidence-first usage notes

The table and list are best-read in this order:

1. identify required control behavior (loop, actuation limits, safety boundaries),
2. check whether the related module path is present and documented, and
3. validate with the linked experiment or test command before procurement.

The goal is to avoid assuming equivalence from terminology alone. A module path with
replay evidence and validation gates is treated as higher operational confidence than a
standalone benchmark row.

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
supervisor, and actuation layer are universal. 36 domainpacks ship
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

## Market and operator interpretation

This comparison page is written as a positioning document, not a benchmark table.
Its intent is to reduce technical ambiguity for teams deciding whether SPO matches
their workflow.

In operational terms, SPO is strongest when the target problem is:
- persistent phase coupling control,
- auditable safety gates,
- and reproducible intervention evidence.

The table therefore maps directly to deployment posture: if the requirement is
closed-loop control with traceable action outcomes, SPO is the library that
already assumes that workflow end-to-end.

## Evidence and validation boundary

Claims above are grounded in repo-visible modules, tests, and benchmark suites.
For each differentiated capability, the documentation should be cross-checked
against:
- implementation presence in linked modules,
- validation tests in the corresponding suite,
- and audit records when policy actions are involved.

## How to interpret this page for adoption decisions

This document is a technical fit check, not a general leaderboard. Use it to reduce
integration risk with a simple sequence:

1. Confirm your problem needs supervised synchronisation control (not pure simulation).
2. Verify whether your required safety envelope is present (bounded actions, replay, regime checks).
3. Confirm benchmark relevance to your workload scale and hardware constraints.
4. Compare SPO's differentiable path only when learning/gradient tooling is required.
5. Read linked validation pages before committing to benchmark claims.

If step 2 fails, SPO should not be selected as the primary engine despite SPO
having partial feature overlap.

## Practical adoption sequence

Use this page only as the first selection filter. After deciding SPO is in scope:

- map your requirement to the closest operational control profile (supervision,
  safety enforcement, replay requirements),
- match required monitors and regulators in the linked concept and guide pages,
- then switch to implementation-level validation before any production trial.

The final decision should include:

- a working command path from this architecture page into your target guide page,
- explicit evidence that safety gates are enforced on the same branch you plan to
  promote,
- and a release note or roadmap checkpoint that acknowledges the governance boundary.
