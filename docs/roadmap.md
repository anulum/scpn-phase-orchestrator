<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Public Roadmap -->

# Public Roadmap

This roadmap is a public planning view. It avoids dates unless a release is
already shipped, and it separates stable surfaces from research or deferred
maintenance tracks.

## Current Stable Surface

| Surface | Status |
|---------|--------|
| UPDE/Kuramoto engines | Python engine family with optional Rust acceleration |
| Stuart-Landau amplitude dynamics | documented guide, API reference, examples, and tests |
| Supervisor and policy DSL | regime management, policy rules, Petri net sequencing, audit trace |
| Domainpacks | 36 bundled domainpacks with README coverage and gallery inventory |
| Notebooks and demos | 19 notebooks, 27 terminal examples, Streamlit tools, CLI demo, WASM demo |
| API documentation | package-level and detailed MkDocs API pages |
| Deployment docs | production, backend fallback, dependency locks, interactive tools |
| Audit/replay | deterministic audit trace and replay documentation |

## Active Documentation Track

| Track | Current state | Next useful additions |
|-------|---------------|-----------------------|
| Onboarding | role-based first-hour handbook exists | keep failure paths and contributor paths linked from every first-run page |
| Troubleshooting | install, notebooks, docs, FFI, validation, audit replay, and demos covered | add issue templates if repeated failure classes emerge |
| Notebook operations | execution matrix documents extras and CI expectation | keep matrix updated when notebooks are added or made local-only |
| API examples | package-level API pages exist | add compact copy/paste examples to subsystem pages that lack them |
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
| API freeze discipline | public entry points documented and compatibility boundaries stated |
| N-channel rollout | P/I/S remains default while additional typed channels are documented and replayable |
| Benchmark visibility | benchmark pages label commands, environment, and historical snapshot dates |
| Production hardening | deployment defaults, security scans, dependency locks, and fallback paths remain documented |
| Docs as entry point | users can move from install to run, validate, replay, and deploy without reading source first |

## v1.x Differentiator Track

| Item | Roadmap stance |
|------|----------------|
| RL/autotune layer on JAX `nn` backend | learn `K`, `alpha`, `zeta`, and `Psi` policies from rewards such as coherence minus `R_bad` and unsafe-actuation penalties |
| Full N-channel algebra | formalise channel groups, required/optional channels, derived channels, cross-channel coupling, replay, and reporting |
| Hierarchical orchestration | nested supervisors plus edge/cloud synchronisation protocol for distributed coherence control |
| Formal supervisor verification | export Petri-net and policy surfaces to PRISM, TLA+, or equivalent model-checking workflows for safety properties |
| Plugin ecosystem | stable interfaces for domainpacks, extractors, actuators, bridges, and compatibility tests so domain experts can publish extensions without forking |

## Deferred Maintenance Track

These items are acknowledged but not the current documentation slice:

| Item | Status |
|------|--------|
| Typed NumPy signature sweep | tracked as maintenance; current public docs should not claim runtime validation beyond verified checks |
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

## Minor Polish Before v1.0

| Item | Required before claiming done |
|------|-------------------------------|
| Public benchmark suite | reproducible numbers against reference Strogatz/Pikovsky-style implementations, with commands and environment labels |
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
