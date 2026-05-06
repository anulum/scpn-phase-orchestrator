<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Public Roadmap -->

# Public Roadmap

This roadmap is a public planning view. It avoids dates unless a release is
already published, and it separates stable surfaces from research or deferred
maintenance tracks.

## Current Stable Surface

| Surface | Status |
|---------|--------|
| UPDE/Kuramoto engines | Python engine family with optional Rust acceleration |
| Stuart-Landau amplitude dynamics | documented guide, API reference, examples, and tests |
| Supervisor and policy DSL | regime management, policy rules, Petri net sequencing, audit trace |
| Domainpacks | 36 bundled domainpacks with README coverage and gallery inventory |
| Notebooks and demos | 19 notebooks, 27 terminal examples, Streamlit tools, CLI demo, WASM demo |
| API documentation | package-level and detailed MkDocs API pages with public-module autodoc coverage guarded by tests |
| Deployment docs | production, backend fallback, dependency locks, interactive tools |
| Audit/replay | deterministic audit trace and replay documentation |

## Active Documentation Track

| Track | Current state | Next useful additions |
|-------|---------------|-----------------------|
| Onboarding | role-based first-hour handbook exists | keep failure paths and contributor paths linked from every first-run page |
| Troubleshooting | install, notebooks, docs, FFI, validation, audit replay, and demos covered | add issue templates if repeated failure classes emerge |
| Notebook operations | execution matrix documents extras and CI expectation | keep matrix updated when notebooks are added or made local-only |
| API examples | public modules have mkdocstrings coverage; package-level API pages exist | add compact copy/paste examples to subsystem pages that lack them |
| Domainpack docs | 36 README files exist | keep examples and gallery aligned with every new domainpack |

## Usability Moat Track

This track is the main post-docs priority for lowering the learning curve for
control engineers who do not already live in phase dynamics.

| Item | Target outcome |
|------|----------------|
| One-click SPO Studio web UI | drag/drop oscillators, live `R`/`Psi`/`K` visualisation, real-time knob tuning, and export/deploy paths for Docker, WASM, and FPGA |
| Auto-binding prototype | infer a proposed `binding_spec.yaml` from raw time-series, event logs, or graph signals using SINDy-style discovery or graph-learning methods |
| Guided deployment path | take a validated Studio project to Docker, browser WASM, or hardware-targeted output without requiring manual wiring |
| Beginner control-engineer mode | explain phase, coupling, objectives, and supervisor decisions in terms of the user's domain signals rather than only Kuramoto terminology |

## v1.0 Focus

| Focus | Acceptance shape |
|-------|------------------|
| API freeze discipline | top-level public manifest and drift test are in place; future manifest changes must carry compatibility notes |
| N-channel rollout | P/I/S remains default while additional typed channels are documented and replayable |
| Benchmark visibility | benchmark pages label commands, environment, and historical snapshot dates |
| Production hardening | deployment defaults, security scans, dependency locks, and fallback paths remain documented |
| Docs as entry point | users can move from install to run, validate, replay, and deploy without reading source first |

## v1.x Differentiator Track

| Item | Roadmap stance |
|------|----------------|
| Dynamic higher-order topology adaptation | foundation, plasma-control demo, traffic-flow transfer-entropy demo, and pairwise-support policy hardening are implemented; broader Lyapunov policy validation remains open |
| Causal intervention engine | counterfactual rollout foundation and cardiac attribution demo are implemented; deeper causal-model learning and broader attribution demos remain open |
| RL/autotune layer on JAX `nn` backend | reward evaluation, replay ranking, offline candidate generation, proposal records, replay-only policy search, and adaptive replay refinement are implemented; next step is optional PPO/SAC or hybrid physics learners behind the same non-actuating gates |
| FEP / predictive-coding supervisor backend | predictive supervisor foundation and power-grid hierarchy proof are implemented; broader hierarchy/domainpack proof work remains open |
| Full N-channel algebra | channel algebra summary, resolved-config integration, audit-header coverage, report JSON/text exposure, reusable report summary payloads, delayed/uncertain classification, and runtime policy records are implemented; deeper supervisor execution remains open |
| Hierarchical orchestration | nested supervisors plus edge/cloud synchronisation protocol for distributed coherence control |
| Formal supervisor verification | export Petri-net and policy surfaces to PRISM, TLA+, or equivalent model-checking workflows for safety properties |
| STL runtime verification | robustness monitor foundation, policy YAML `stl_monitors` integration, and PRISM export linkage are implemented; full monitoring automata synthesis remains open |
| Symbolic-to-binding compiler | compiler foundation plus local retrieval evidence, confidence factors, and generated review notebook are implemented; richer long-form retrieval and notebook execution evidence remain open |
| Cross-domain meta-transfer | replay-backed proposals, multi-audit fitting, training summaries, and deterministic JSON package export are implemented; larger real audit-history corpora and optional `scpn-meta` packaging remain open |
| Plugin ecosystem | manifest registry foundation is implemented for entry-point discovery, capability declarations, compatibility checks, audit records, marketplace catalogue packaging, a runnable catalogue example, and CLI catalogue export; Rust-side integration remains open |

## Deferred Maintenance Track

These items are acknowledged but not the current documentation slice:

| Item | Status |
|------|--------|
| Typed NumPy signature sweep | tracked as maintenance; STL fallback robustness and strange-loop supervisor helper arrays are parameterised; continue scoped module-by-module sweeps |
| Visual and batch-heavy features | deferred unless a user-facing workflow needs them immediately |
| Broad benchmark-file expansion | useful, but benchmark numbers must be measured and reproducible before publication |
| Thin-test strengthening | quality backlog item; prefer focused property or parity tests tied to changed code |
| Auxiliary backend review | keep non-primary backends experimental unless measured value justifies maintenance cost |

## Research And Experimental Track

These areas may evolve behind explicit experimental flags or separate guides:

| Area | Roadmap stance |
|------|----------------|
| ML-assisted binding proposals | promising, but outputs must remain reviewable binding specs |
| Learned supervisor policies | research path only until auditability and safety constraints are clear |
| Distributed edge orchestration | useful for multi-node deployments; keep audit and replay semantics central |
| Formal verification exports | desirable for safety-critical policies, but separate from normal user onboarding |
| Neuromorphic and quantum-native backends | bridge work remains experimental unless a maintained workload depends on it |

## Long-Horizon Research Backlog

These are candidate research tracks. Each needs a proof-of-concept, safety
boundary, audit format, and reproducible domainpack demo before it can move
into the active roadmap.

| Track | First acceptance gate |
|-------|-----------------------|
| Self-modelling embodied digital twin | `self_model_error` monitor plus one hardware-in-the-loop or replay-backed reconfiguration demo |
| Evolutionary supervisor policy search | offline evolution over `audit.jsonl` histories with STL/counterfactual safety filter before any live hot-patch path |
| Information-geometry control layer | JAX-native Fisher-Rao or Wasserstein control primitive with audit-visible curvature or geodesic metrics |
| Sheaf-cohomology control | supervisor foundation is implemented; heterogeneous-domain demos and obstruction hardening remain open |
| Federated meta-orchestrator | privacy-preserving policy-gradient aggregation across nodes without raw time-series exchange |
| Byzantine-fault-tolerant meta-orchestrator | three-node consensus demo over signed policy/topology proposals and hash-linked audit evidence |
| Quantum-native compiler target | Qiskit/PennyLane or OpenPulse/QASM output backend for `quantum_simulation` with co-simulation validation |
| Neuromorphic compiler target | Lava/PyNN or HDL output from a binding spec plus supervisor policy with simulator parity evidence |
| Hybrid neuromorphic-quantum co-compiler | co-simulation hook that keeps quantum and spiking targets under the same N-channel audit semantics |
| Value-alignment supervisor guard | guard foundation is implemented; binding-spec templates and counterfactual violation reporting remain open |
| Autopoietic lineage sandbox | resource-bounded offline child-policy lineage over audit replays, with merge only through reviewable policy diffs |
| Temporal-causal hypergraph experiments | explicitly experimental time-symmetric rollout research; no production claim without a conventional causal baseline |
| Intergenerational policy inheritance | signed lineage metadata for child orchestrators, inherited policy genomes, multi-objective replay fitness, and merge-only reviewed hot patches |
| Sheaf-theoretic coherence manifold | obstruction-aware control primitive over a sheaf Laplacian with audit-visible cohomology dimensions |
| Constitutional value-alignment guard | Pareto objective constraints in binding specs, counterfactual violation logs, and forced safe fallback path |
| Strange-loop meta-orchestrator | monitor foundation is implemented; long-run drift scenarios and studio surfacing remain open |
| Morphogenetic field topology | field foundation is implemented; field snapshot visualisation and domainpack demos remain open |
| Integrated-information monitor | monitor foundation and audit-report summary integration are implemented; benchmarked approximations remain open |
| Topos-theoretic semantic binding | categorical validation prototype for binding and policy composition with explicit proof obligations |
| Multiverse counterfactual simulator | vectorised JAX branch rollouts over knob/topology ensembles before committing high-risk actuation |
| Entanglement-aware hybrid order parameters | quantum co-simulation monitor that reports entanglement entropy alongside classical `R` and `Psi` |

Priority order for the first implementation tranche:

1. ~~Causal intervention engine foundation.~~
2. ~~STL runtime verification foundation.~~
3. ~~Dynamic higher-order topology adaptation foundation.~~
4. ~~FEP / predictive-coding supervisor backend foundation.~~
5. ~~Symbolic-to-binding compiler foundation.~~

Speculative priority order after the first tranche:

1. ~~Strange-loop meta-orchestrator foundation.~~
2. ~~Morphogenetic field topology foundation.~~
3. ~~Integrated-information monitor foundation.~~
4. ~~Sheaf-theoretic coherence manifold foundation.~~
5. ~~Constitutional value-alignment guard foundation.~~

## Minor Polish Before v1.0

| Item | Required before claiming done |
|------|-------------------------------|
| Public benchmark suite | reference-suite metadata, dated JSON, and public snapshot page are in place; broader cross-host/backend comparison coverage remains open |
| Windows Rust FFI stability | remove the experimental warning only after CI, installation, and parity evidence support it |
| End-to-end hardware example | at least one real FPGA or neuromorphic output path with documented command, artefact, and verification result |

## Non-Goals For The Current Docs Track

- Do not present historical benchmark snapshots as current measurements.
- Do not promise release dates for research features.
- Do not hide optional dependency fallback paths.
- Do not add public APIs without corresponding docs and examples.
- Do not make users infer commands from source code when a guide can state them.

## Source Of Truth

Use this page for public orientation. Use `ROADMAP.md` in the repository root
for the detailed historical release plan and internal planning notes.
