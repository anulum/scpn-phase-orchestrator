<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Executive Overview -->

# Executive Overview

SCPN Phase Orchestrator is a coherence-control compiler for systems with
repeating behaviour. It turns waves, events, and discrete states into phase
variables; evaluates synchrony, coupling, lag, causality, and regime risk; and
produces reviewable control proposals instead of opaque automation.

The short operational question is:

> Which parts of this system are locking together, should that lock exist, and
> what bounded change can be reviewed before anything reaches production?

## Why This Exists

Modern organisations already collect cyclic telemetry, but it is usually split
between dashboards, incident logs, notebooks, controller code, and domain tools.
SPO provides a shared phase contract so different teams can discuss the same
phenomenon with the same variables: `theta`, `omega`, `K`, `alpha`, `zeta`,
`Psi`, `R_good`, and `R_bad`.

That matters when the wrong intervention is expensive. In grids, plasma,
industrial operations, robotics, clinical research, and distributed systems,
control proposals must be bounded, replayable, and rejectable. SPO is designed
around that evidence boundary.

## What It Is

| Layer | Description | User-facing value |
|-------|-------------|-------------------|
| Domainpack compiler | YAML binding specs describe sources, oscillator families, channels, coupling, objectives, and boundaries | domain assumptions are explicit and versioned |
| Oscillator extraction | physical, informational, and symbolic signals become phase channels | heterogeneous telemetry becomes comparable |
| Dynamics engine | Kuramoto, UPDE, Stuart-Landau, delay, stochastic, simplicial, inertial, and related models evolve the phase state | synchrony hypotheses can be tested before deployment |
| Monitors | order parameter, PLV, PAC, Lyapunov, entropy, transfer entropy, Hodge, chimera, STL, and related metrics | operators see more than a single scalar dashboard |
| Supervisor | regime FSMs, Petri nets, policies, value guards, and projectors constrain proposed changes | actuation stays bounded and reviewable |
| Evidence layer | audit logs, replay, benchmark snapshots, Studio panels, and generated docs | decisions can be reproduced, explained, or rejected |

## What It Is For

SPO is most useful where a system has repeated behaviour and synchrony affects
risk, performance, or diagnosis.

| Domain | Typical use case | SPO contribution |
|--------|------------------|------------------|
| Power and energy | generator/inverter oscillations, weak damping, cascading risk | inertial Kuramoto modelling and bounded stability proposals |
| Fusion and plasma | MHD modes, transport oscillations, actuator timing | multi-rate phase binding and review-only stabilisation evidence |
| Cloud operations | retry storms, queue cascades, heartbeat lock-in | harmful synchrony detection and desynchronisation proposals |
| Manufacturing | vibration, tool wear, cyclic defects, process drift | root-cause evidence across coupled process loops |
| Neuroscience/cardiology | band coupling, rhythm coherence, seizure or arrhythmia research | reproducible phase metrics without autonomous treatment claims |
| Robotics/swarms | gait, formation, leader-follower coherence | simulation-first phase-policy development |
| Traffic/logistics | signal coordination, platoon waves, queue waves | congestion-wave evidence before live signal changes |
| Digital twins | plant/twin drift and residual coherence | replayable mismatch and maintenance hypotheses |
| ML research | differentiable oscillator layers and inverse coupling | gradient-based topology, coupling, and policy search |

## What Makes It Different

SPO is not only an oscillator simulator. The differentiator is the chain from
source binding to bounded operational evidence:

1. Bind the domain assumptions.
2. Extract phase consistently across physical, informational, and symbolic data.
3. Simulate or infer coupled dynamics with explicit numerical contracts.
4. Measure synchrony, causality, and safety monitors.
5. Propose only bounded, rate-limited, review-gated actions.
6. Preserve replay and benchmark evidence for promotion decisions.

This is why the repository includes CLI, Python API, JAX layers, Rust kernels,
notebooks, domainpacks, Studio panels, formal proof hooks, and benchmark gates.
They are separate surfaces, but they serve the same review-first control path.

## Adoption Routes

| If you are | Start with | Then move to |
|------------|------------|--------------|
| Evaluating value | [Use Cases and Value Map](use_cases.md) | [Quickstart](quickstart.md) |
| Building a domain | [New Domain Checklist](../tutorials/01_new_domain_checklist.md) | [Raw Sources to Run](../tutorials/05_from_raw_sources_to_run.md) |
| Embedding in Python | [Python Facade API](../reference/api/api.md) | [Production Deployment](../guide/production.md) |
| Training differentiable models | [Differentiable Kuramoto](../guide/differentiable_kuramoto.md) | [nn API](../reference/api/nn.md) |
| Reviewing operations | [Notebook to Production](../guide/notebook_to_production.md) | [Release Hygiene](../RELEASE_HYGIENE.md) |
| Checking roadmap maturity | [Public Roadmap](../roadmap.md) | benchmark and CI artefacts |

## Evidence and Safety Boundaries

- A domainpack is executable evidence only after its assumptions and boundaries
  are reviewed for the domain.
- Benchmark numbers are dated snapshots and must be reproduced before being
  used as current performance claims.
- Hardware adapters are opt-in and adapter-scoped; review-only quantum and
  neuromorphic compiler targets do not execute on hardware by default.
- The Python facade is for deterministic local simulation, not live actuation.
- Studio panels are operator review surfaces, not proof that a domain is
  calibrated.
- Frontier tracks such as safe RL, causal topology mutation, formal export, and
  hardware-native compilation remain evidence-bound until the relevant artefacts
  are published.

## Market Value Summary

The market value is a unified language for synchrony risk. Instead of separate
teams arguing through dashboards, notebooks, and controller logs, SPO gives them
a common artefact chain: binding spec, phase state, coupling matrix, regime,
proposal, replay log, and benchmark context.

That is most valuable in domains where drift, lock-in, cascade, or unsafe
control can become expensive: energy, fusion, industrial operations, clinical
research, robotics, aerospace, cloud platforms, and high-value digital twins.
