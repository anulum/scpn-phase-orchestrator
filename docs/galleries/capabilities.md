<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Capabilities and application boundaries -->

# Capabilities & Applications

This page answers three separate questions:

1. What can the software execute now?
2. What evidence supports those capabilities?
3. Where could the platform create operational or commercial value after
   domain validation?

Keeping those questions separate prevents a working API, a simulation
scaffold, or a benchmark from being mistaken for a field-proven product.

## Capability maturity at a glance

| Maturity | Current SPO surface | What the label permits |
|---|---|---|
| Stable software contract | 24 top-level Python exports, CLI, binding validation, simulation, audit/replay | downstream integration under semantic versioning |
| Externally checked scientific niche | grid modal-damping estimation against ANDES small-signal eigenvalues | a bounded grid-modal claim under the documented study protocol |
| Reproducible evaluation method | matched-false-alarm detector audit, permutation significance, sealed record | honest comparison of a detector with event/null data |
| Tested engineering capability | phase extraction, coupled dynamics, monitors, policy projection, deterministic replay, optional Rust parity | local R&D and pre-deployment evaluation |
| Domain scaffold | 36 binding specs spanning engineering, physical, biological, and digital systems | a starting model whose assumptions still require domain evidence |
| Review-only integration | PLC, PMU, quantum, neuromorphic, clinical, fusion, and other hardware-adjacent bridges | artefact generation or bounded dry runs, not autonomous actuation |
| Experimental research | broad transfer, metaphysical/identity models, frontier formal and accelerator tracks | hypothesis generation only |

The [fact-based overview](../FACT_BASED_OVERVIEW.md) and
[validation report](../VALIDATION_REPORT.md) provide the evidence history. The
[public roadmap](../roadmap.md) separates stable, active, deferred, and research
work.

## What SPO can do now

### Turn timing data into a common model

SPO maps continuous waves, event timing, and discrete state transitions into
physical, informational, and symbolic phase channels. A versioned
`binding_spec.yaml` records what each oscillator represents, how oscillators
couple, which synchrony is useful or harmful, and where safety boundaries sit.

This makes assumptions reviewable before they become controller logic.

### Execute coupled-dynamics hypotheses

The Python engine implements Kuramoto/UPDE, Stuart-Landau, inertial, delay,
stochastic, simplicial, hypergraph, geometric, swarmalator, and related model
families. A researcher can compare coupling, phase-lag, forcing, topology, and
integration choices on deterministic seeded runs.

These are executable models, not proof that a selected model represents a
particular plant, patient, market, or network.

### Measure more than one coherence number

The monitor surface includes order parameters, PLV, PAC, chimera indices,
Lyapunov and recurrence measures, transfer entropy, Hodge decomposition,
spectral metrics, winding, STL traces, and domain-specific observers. This
supports competing explanations instead of hiding a decision behind one score.

### Produce reviewable proposals and evidence

The supervisor can classify regimes and produce rate-limited, projected
control candidates. Audit records are hash-linked and can be replayed
deterministically. Hardware and external-service writes remain behind separate
adapter, policy, credential, and operator gates.

### Audit early-warning claims honestly

`spo audit-detector` and `scpn_phase_orchestrator.evaluation` compare
event and null scores at a matched false-alarm rate, compute a permutation
p-value, and seal the result. The method can evaluate an SPO detector, a
classical baseline, or a black-box model on the same footing.

Negative results remain results: the current real-data studies do not support a
general tipping-point prediction claim.

### Accelerate selected local workloads

The optional `spo-kernel` Rust workspace accelerates selected numerical,
monitor, supervisor, and extraction paths through PyO3. Backend choice,
fallback, validation, and parity are explicit. Timing tables in the
[performance guide](../guide/performance.md) are dated local regression
snapshots, not portable real-time guarantees.

## Application and value map

| Area | Candidate problem | SPO contribution | Evidence still required |
|---|---|---|---|
| Power systems | oscillatory-mode tracking and damping review | inertial dynamics, PMU ingestion, modal evidence, replay | utility data, operating-envelope validation, operator acceptance |
| Cloud platforms | retry or queue synchronisation that amplifies incidents | event-phase mapping, harmful-coherence metrics, QueueWaves | service-specific baselines, false-alarm study, production load tests |
| Industrial systems | interacting machine, process, or controller cycles | binding contract, delay/coupling hypotheses, bounded proposals | plant model, hazard analysis, deterministic timing evidence |
| Robotics and swarms | alignment, dephasing, hand-off, or collective motion | swarmalator/coupling simulation and regime traces | hardware-in-loop tests and fleet safety case |
| Neuroscience and physiology research | reproducible phase relationships across channels | extraction, phase metrics, honest detector evaluation | cohort protocol, clinical statistics, ethics and regulatory review |
| Fusion and plasma research | cross-scale mode-locking hypotheses | multi-channel bindings, phase/coupling analysis, review artefacts | device data, physics validation, control-room approval |
| ML and inverse modelling | learn coupling or topology through dynamics | differentiable JAX layers, inverse Kuramoto, SAF loss | held-out benchmarks and task-specific generalisation |
| Assurance-heavy R&D | reconstruct how a model produced a proposal | sealed audit, replay, explicit claim and adapter boundaries | organisation-specific governance and deployment controls |

## Where the market value can come from

SPO's defensible value proposition is **integration and evidence discipline**,
not universal prediction.

- **Shorter model-integration cycles:** phase-bearing signals, coupling
  assumptions, solver settings, and review boundaries use one contract.
- **Lower model-risk ambiguity:** implemented, benchmarked, externally checked,
  scaffold, and research surfaces are labelled separately.
- **Reproducible technical review:** a buyer, regulator, scientist, or operator
  can inspect the binding, replay the run, and reject an unsupported proposal.
- **Cross-domain reuse:** the engine and assurance workflow can be reused while
  each domain keeps its own validation burden.
- **Deployment optionality:** pure Python supports evaluation; optional Rust,
  service, telemetry, and hardware adapters can be promoted only when their
  evidence exists.

Potential commercial forms include an engineering library, an evaluation and
assurance toolkit, domain-specific integration work, controlled operator
services, and commercial licensing. No revenue, market-size, adoption, or
return-on-investment figure is claimed without external evidence.

## What SPO does not establish

SPO does not, by itself:

- predict plasma disruptions, seizures, market crashes, or infrastructure
  failures;
- provide clinical diagnosis or treatment;
- certify functional safety, cybersecurity, or regulatory compliance;
- guarantee hard real-time deadlines on Python, Rust, FPGA, or network paths;
- validate all 36 domainpacks as deployable detectors or controllers;
- show that a local benchmark transfers to another host or workload;
- authorise autonomous writes to PLCs, medical devices, robots, grids, fusion
  equipment, quantum systems, or neuromorphic hardware.

Those outcomes need exact domain data, preregistered success criteria,
independent review, deployment evidence, and operator authority.

## Choose the next evidence path

| Goal | Next page |
|---|---|
| decide whether the problem has real phase structure | [Use Cases and Value Map](../getting-started/use_cases.md) |
| run a sealed result | [Quickstart](../getting-started/quickstart.md) |
| understand scientific claim boundaries | [Fact-Based Overview](../FACT_BASED_OVERVIEW.md) |
| select an engine | [Choosing an Engine](../guide/choosing_an_engine.md) |
| inspect the API | [API Reference](../reference/api/index.md) |
| move a notebook toward production | [Notebook to Production](../guide/notebook_to_production.md) |
| review benchmarks and release evidence | [Release Hygiene](../RELEASE_HYGIENE.md) |
| inspect what remains open | [Public Roadmap](../roadmap.md) |
