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

| Market or domain | Operational problem | SPO value path | Buyer-facing value |
|------------------|---------------------|----------------|--------------------|
| Power and energy | Grid oscillations, weak damping, inverter coordination | detect harmful coherence, test coupling policies, produce audit-backed control proposals | fewer unreviewed tuning changes and clearer stability evidence for grid operators |
| Fusion and plasma | MHD mode coupling, transport oscillations, control timing | map multi-rate observables into coupled phase channels and review stabilising proposals | one evidence layer between plasma diagnostics, control timing, and safety review |
| Cloud and platform operations | Retry storms, queue cascades, service-heartbeat lock-in | detect synchronised failure modes and desynchronise overloaded layers | lower incident blast radius through earlier cascade evidence and replayable remediation proposals |
| Manufacturing and machinery | Vibration, process drift, cyclic defects, SPC patterns | expose phase coupling between process variables and control loops | faster root-cause triage for cyclic defects and drift before scrap or downtime grows |
| Cardiology and neuroscience | rhythm coherence, seizure precursors, band coupling | analyse phase locking, PAC/PLV/ITPC, and multi-scale synchrony without hidden actuation | reproducible research and clinical-review evidence without claiming autonomous treatment |
| Robotics and swarms | gait coordination, leader-follower locking, swarm consensus | prototype phase policies before physical deployment | safer progression from simulation to robot trials with bounded control envelopes |
| Traffic and logistics | signal timing, platoon waves, queue waves | simulate coordination policies and detect unstable synchronisation patterns | evidence for congestion-wave mitigation before changing live signal plans |
| Finance and markets | regime coupling, sector synchrony, crash precursors | review cyclic coherence and coupling shifts as risk evidence, not trading advice | transparent market-regime diagnostics for analysts, not opaque prediction claims |
| Digital twins | plant/twin drift and residual coherence | bind real telemetry to phase-state twins with replayable audit records | measurable plant/twin mismatch and reviewable predictive-maintenance hypotheses |
| Research platforms | differentiable oscillator models and inverse coupling | optimise coupling matrices and infer topology from trajectory data | one package for theory, reproducible experiments, and accelerator-backed topology search |

## How to Recognise a Good SPO Use Case

A good candidate has all of these properties:

| Question | Good sign | Poor fit |
|----------|-----------|----------|
| Is there repeated behaviour? | waves, cycles, retries, rotations, stages, heartbeats, events, or state loops | one-off static records with no temporal phase meaning |
| Can each source be timestamped or ordered? | samples, events, and states can be aligned to a common clock or sequence | measurements have no reliable ordering or cadence |
| Does synchrony matter? | some coherence is desirable, harmful, diagnostic, or causal | the domain only needs scalar threshold alerting |
| Can assumptions be written down? | coupling, lag, forcing, and safety boundaries are reviewable | the domain requires undocumented black-box decisions |
| Is actuation sensitive? | proposals need replay, audit, rate limits, and human review | a simple uncontrolled script is acceptable |

## Why the Market Angle Matters

Many organisations already collect cyclic telemetry but keep it split across
signal dashboards, incident logs, notebooks, and controller-specific tools. SPO
turns those fragments into a common phase contract. That matters because the
same questions reappear across sectors: which components are phase-locked, what
coupling path caused it, whether the lock is good or bad, and which bounded
knob could move the system without violating review policy.

The commercial value is highest where wrong control is expensive: energy,
industrial operations, clinical research, cyber-physical infrastructure,
robotics, aerospace, and high-value simulation. In those settings the audit
surface is as important as the mathematics. A proposal that cannot be replayed,
rejected, or traced should not reach production.

## Core User Journeys

| User | First useful path | Production path |
|------|-------------------|-----------------|
| Domain expert | identify oscillators and boundaries | maintain a domainpack with cited assumptions and validation evidence |
| ML researcher | use differentiable Kuramoto layers and SAF losses | train coupling/topology proposals with accelerator checks and reproducible seeds |
| Platform engineer | run QueueWaves or a minimal domainpack | wire metrics to bounded policies, audit replay, and dashboard review |
| Control engineer | inspect K, alpha, zeta, and Psi knobs | keep actuation behind rate limits, safety tiers, and human review gates |
| Documentation reader | start with this page and the quickstart | follow the tutorial sequence and API reference for the selected surface |
| Investor or product reviewer | inspect use cases, evidence boundaries, and roadmap | review validated capabilities, commercial licensing path, and open frontier items |

## Expected value pattern

Across domains, value is most predictable when teams keep the same sequence:

1. define what should synchronize and what should remain decoupled,
2. represent inputs consistently across physical, informational, and symbolic channels,
3. test bounded control candidates in replay-first mode,
4. keep only candidates with explicit regime, metric, and evidence gates.

This produces a consistent operating language for operators, platform owners, and
product reviewers because "good coherence", "harmful coherence", and "bounded
response" are handled in the same contracts for different sectors.

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

## Adoption sequence by role

For each use case, apply this sequence before live recommendations:

1. define the target coherence outcome (desirable lock, avoidable lock, or
   neutral synchrony),
2. map signals into physical/informational/symbolic channels,
3. run at least one audit-backed deterministic run,
4. validate policy and monitor settings on replay,
5. only then promote to review cadence with bounded actuation.

This keeps the evidence boundary explicit and prevents overreach from
unvalidated inference.

## Use this page to select a first domain

Before opening implementation guides, define:

- target signal class (continuous, event, state),
- expected control direction (synchronise, desynchronise, classify),
- and the first evidence package (minimal domainpack + replay file).

If those three items are not agreed, the remaining process should stay in
evaluation mode and avoid actuator exports.

Use this page as the final business-value filter before any advanced engine
choice is made.

For the three focus tracks — industrial predictive maintenance, critical
infrastructure, and biosignal/clinical — with their lead domainpacks, runnable
commands, validation status, and safety-tier posture, see
[Beachhead Verticals](../galleries/beachhead_verticals.md).
