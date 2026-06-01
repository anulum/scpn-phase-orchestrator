<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Use Cases and Value Map -->

# Use Cases and Value Map

SCPN Phase Orchestrator is a coherence-control compiler. It turns cyclic,
event-driven, and state-machine behaviour into phase variables, runs those
variables through coupled-oscillator dynamics, and produces reviewable control
signals for systems where synchrony is useful, harmful, or diagnostic.

The practical question it answers is:

> Which parts of this system are locking together, which parts should not be
> locking together, and which safe control knob can move the system back toward
> the intended regime?

## The Short Explanation

Most complex systems contain repeating processes: heartbeats, queues, retry
storms, grid frequency, machine vibration, market cycles, sleep stages, traffic
lights, robotic gaits, plasma modes, or software-service heartbeats. SPO maps
those processes into three oscillator channels:

| Channel | What it captures | Example inputs |
|---------|------------------|----------------|
| Physical | Continuous measured waves | voltage, pressure, vibration, EEG, PPG, temperature |
| Informational | Timed events and decisions | requests, retries, trades, alarms, messages |
| Symbolic | Discrete states and modes | sleep stage, machine state, traffic phase, workflow state |

Once every relevant signal has a phase, SPO can compute order parameters,
phase-locking metrics, causal coupling evidence, regime transitions, and
bounded control proposals with deterministic replay evidence.

## Where It Creates Value

| Market or domain | Operational problem | SPO value path |
|------------------|---------------------|----------------|
| Power and energy | Grid oscillations, weak damping, inverter coordination | detect harmful coherence, test coupling policies, produce audit-backed control proposals |
| Fusion and plasma | MHD mode coupling, transport oscillations, control timing | map multi-rate observables into coupled phase channels and review stabilising proposals |
| Cloud and platform operations | Retry storms, queue cascades, service-heartbeat lock-in | detect synchronised failure modes and desynchronise overloaded layers |
| Manufacturing and machinery | Vibration, process drift, cyclic defects, SPC patterns | expose phase coupling between process variables and control loops |
| Cardiology and neuroscience | rhythm coherence, seizure precursors, band coupling | analyse phase locking, PAC/PLV/ITPC, and multi-scale synchrony without hidden actuation |
| Robotics and swarms | gait coordination, leader-follower locking, swarm consensus | prototype phase policies before physical deployment |
| Traffic and logistics | signal timing, platoon waves, queue waves | simulate coordination policies and detect unstable synchronisation patterns |
| Finance and markets | regime coupling, sector synchrony, crash precursors | review cyclic coherence and coupling shifts as risk evidence, not trading advice |
| Digital twins | plant/twin drift and residual coherence | bind real telemetry to phase-state twins with replayable audit records |
| Research platforms | differentiable oscillator models and inverse coupling | optimise coupling matrices and infer topology from trajectory data |

## Core User Journeys

| User | First useful path | Production path |
|------|-------------------|-----------------|
| Domain expert | identify oscillators and boundaries | maintain a domainpack with cited assumptions and validation evidence |
| ML researcher | use differentiable Kuramoto layers and SAF losses | train coupling/topology proposals with accelerator checks and reproducible seeds |
| Platform engineer | run QueueWaves or a minimal domainpack | wire metrics to bounded policies, audit replay, and dashboard review |
| Control engineer | inspect K, alpha, zeta, and Psi knobs | keep actuation behind rate limits, safety tiers, and human review gates |
| Documentation reader | start with this page and the quickstart | follow the tutorial sequence and API reference for the selected surface |
| Investor or product reviewer | inspect use cases, evidence boundaries, and roadmap | review validated capabilities, commercial licensing path, and open frontier items |

## What SPO Is Not

SPO is not a black-box automation system. Hardware writes are adapter-scoped,
review-gated, and disabled for quantum or neuromorphic compiler targets until a
verified external execution pipeline exists.

SPO is not a generic time-series dashboard. It specifically models phase,
coherence, coupling, synchronisation, desynchronisation, and safe control
proposal boundaries.

SPO is not a claim that every cyclic system is identical. Each domainpack must
state its measurement source, phase extractor, coupling assumptions, safety
boundaries, and validation evidence.

## Capability Map

| Capability | What to read | Why it matters |
|------------|--------------|----------------|
| Binding specs | [Binding API](../reference/api/binding.md) | makes domain assumptions explicit and reviewable |
| Oscillator extraction | [Oscillators API](../reference/api/oscillators.md) | maps physical, informational, and symbolic data to phase |
| UPDE engines | [UPDE API](../reference/api/upde.md) | runs Kuramoto, Stuart-Landau, delay, stochastic, and higher-order dynamics |
| Supervisor | [Supervisor API](../reference/api/supervisor.md) | classifies regimes and proposes bounded actions |
| Audit replay | [Audit API](../reference/api/audit.md) | makes runs reproducible and tamper-evident |
| Differentiable dynamics | [Differentiable Kuramoto](../guide/differentiable_kuramoto.md) | supports gradient-based topology and coupling optimisation |
| Autotune | [Autotune API](../reference/api/autotune.md) | extracts phases, identifies frequencies, infers coupling, and proposes replay-only policies |
| Hardware adapters | [Adapter Bridges](../guide/adapters.md) | keeps optional integrations explicit and safety-bounded |
| Notebooks | [Notebooks and Demos](../galleries/notebooks_and_demos.md) | gives runnable learning routes for each major capability |

## Learning Routes

| Goal | Route |
|------|-------|
| Understand the product | This page -> [System Overview](../concepts/system_overview.md) -> [Pipeline Execution](../concepts/pipeline_execution.md) |
| Run the first local simulation | [Installation](installation.md) -> [Quickstart](quickstart.md) -> [Hello World](hello_world.md) |
| Build a domainpack | [New Domain Checklist](../tutorials/01_new_domain_checklist.md) -> [Oscillator Hunt Sheet](../tutorials/02_oscillator_hunt_sheet.md) -> [Build K_nm Templates](../tutorials/03_build_knm_templates.md) |
| Move from data to a run | [From Raw Sources to Run](../tutorials/05_from_raw_sources_to_run.md) -> [Deterministic Replay](../tutorials/06_deterministic_replay_for_debugging.md) |
| Work with ML surfaces | [Differentiable Kuramoto Tutorial](../tutorials/04_differentiable_kuramoto.md) -> [nn API](../reference/api/nn.md) -> [SAF guide](../guide/differentiable_kuramoto.md) |
| Review production readiness | [Production Guide](../guide/production.md) -> [Release Hygiene](../RELEASE_HYGIENE.md) -> [Roadmap](../roadmap.md) |

## Evidence Boundary

The repository contains a broad capability surface, but each claim must be read
with its evidence boundary:

- committed tests demonstrate behavioural contracts, not universal domain truth;
- benchmark pages report dated local or CI snapshots, not guaranteed throughput
  on every machine;
- hardware adapters document safety boundaries and optional dependencies;
- roadmap items distinguish implemented surfaces from research or deferred
  tracks;
- domainpacks are examples until calibrated against real domain data and reviewed
  by the responsible domain expert.

That boundary is intentional. SPO is designed for systems where incorrect
control claims are expensive, so documentation must separate executable
capability, review-only evidence, and future research scope.
