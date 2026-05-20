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
| One-click SPO Studio web UI | Streamlit operator prototype now covers domainpack load, raw-source import, binding review, beginner-mode guidance with ordered walkthrough, oscillator edit review artefacts, layer/channel canvas review artefacts, canvas layout manifests, canvas topology patch artefacts, validated binding rewrite candidates, signed hash-checked binding apply with backups, deterministic Canvas interaction state for browser controls, live `R`/`Psi`/`K` metrics, replay tuning, hierarchy monitor, connector ownership plans, dry-run records, owned no-I/O runtime boundary validation for REST/gRPC/Kafka/hardware, and localhost service-process manifests with health checks, deployment readiness/package/materialisation manifests, hardware target and verified-evidence package manifests, recovery reports, and Docker/WASM/project export manifests; it is not yet a good standalone product, because real FPGA/neuromorphic evidence remains open |
| Auto-binding prototype | deterministic proposal builders now infer reviewable `binding_spec.yaml` records from time-series CSV, event-log JSON, and graph JSON with confidence factors, validation diagnostics, time-column sample-rate inference, sparse-derivative/correlation-graph/clustering evidence, lagged directed graph inference, transfer-entropy coupling audit export, extractor-parameter proposals, validator-accepted `auto_initial_k`, and benchmark-gated domain thresholds; `spo auto-bind` exports review-only binding YAML or audit JSON from local raw sources without writing files or enabling actuation; private or partner live-dataset gates remain future work |
| Guided deployment path | deployment readiness, deployment package, package materialisation, hardware target, and verified hardware evidence manifests now carry target status, required artefacts, export hashes, operator-invoked commands, blocked reasons, safety gates, hardware evidence requirements, SHA-256 checks, parity status, and connector handoff review; real target evidence remains open |
| Beginner control-engineer mode | beginner-mode guidance now explains reviewed signals, coupling knobs, objectives, supervisor regime, validation state, ordered walkthrough status, and next actions from the current replay |

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
| Dynamic higher-order topology adaptation | foundation, plasma-control demo, traffic-flow/network-security transfer-entropy demos, Lyapunov mutation validation, and pairwise-support policy hardening are implemented; broader multi-domain Lyapunov policy validation remains open |
| Causal intervention engine | counterfactual rollout foundation, live causal-graph learning, and cardiac/power-grid/traffic-flow/network-security attribution demos are implemented; larger causal-model learners and additional attribution demos remain open |
| RL/autotune layer on JAX `nn` backend | reward evaluation, replay ranking, offline candidate generation, proposal records, replay-only policy search, adaptive replay refinement, PPO-like/SAC-like/hybrid-physics proposal generators, a differentiable Equinox supervisor objective, and deterministic multi-scenario replay learner benchmark gates are implemented behind non-actuating replay gates; real trained PPO/SAC experiments, baselines, and publication artefacts remain open |
| FEP / predictive-coding supervisor backend | predictive supervisor foundation, reusable hierarchy assessment, and power-grid/cardiac hierarchy proofs are implemented; deeper predictive-coding world-model integration remains open |
| Full N-channel algebra | channel algebra summary, resolved-config integration, audit-header coverage, report JSON/text exposure, reusable report summary payloads, delayed/uncertain classification, runtime policy records, delayed/uncertain supervisor execution with audit evidence, and replay-only channel-weight/cross-coupling optimisation surfaces are implemented; deeper learner-backed optimisation remains open |
| Hierarchical orchestration | reduced-summary parent orchestration, bounded escalation audit records, deterministic edge/cloud sync envelopes, strict transport runtime validation, decoded JSONL/REST/frame adapter boundaries, power-grid/cardiac/edge-consensus transport demos, offline gossip consensus replay, and transport-neutral phase-vector gossip for UPDE nodes are implemented; live owned transports and broader deployment remain open |
| Digital-twin binding standard | deterministic binding contract export, transport-neutral payload validation, JSONL file replay, in-memory reference adapter, adapter manifest compatibility checks, dependency-free REST/gRPC/Kafka boundary adapters, a no-I/O hardware frame boundary, and transport-neutral operator evidence are implemented for timing, layers, actuators, N-channel algebra, sync capabilities, contract hashes, directions, auth posture, topics, registered devices, safety interlocks, sequence checks, residual extrema, mismatch reasons, adapter health, and replay/live parity; real hardware evidence remains tracked separately |
| Formal supervisor verification | Petri-net and policy surfaces export to PRISM and TLA+; PRISM STL label export exists; benchmark-gated deterministic formal verification packages now bind exported artefact hashes, named safety properties, disabled checker execution, external PRISM/TLC checker command manifests, and checker-readiness records without invoking tools locally; `spo formal-export --export package --include-checker-readiness` exposes the same no-execution package manifest plus optional PRISM/TLC readiness evidence; non-executing external-checker readiness audits now report executable availability, resolved paths, ready/missing status, and disabled execution; deeper external model-checker CI execution remains open |
| STL runtime verification | robustness monitor foundation, policy YAML `stl_monitors` integration, PRISM export linkage, builtin monitoring automata synthesis, non-actuating controller-candidate synthesis, policy-gated candidate projection, offline closed-loop synthesis plans, and reference-suite gates for deterministic non-actuating plan quality are implemented; runtime integration through the full safety/actuation stack remains open |
| Symbolic-to-binding compiler | compiler foundation, domainpack/docs retrieval evidence, confidence factors, generated review notebook, notebook preflight execution evidence, deterministic retrieval ranking diagnostics, and reference-suite retrieval-ranking gates are implemented; optional corpus expansion with larger live-doc/domainpack corpora remains open |
| LLM-guided domainpack scaffolding | `spo scaffold --llm` now turns natural-language intent into a strict JSON proposal, deterministic binding YAML, validator-approved domainpack files, and an audit record; live providers must be explicitly configured or an offline proposal file must be supplied |
| Cross-domain meta-transfer | replay-backed proposals, multi-audit fitting, nested audit-directory corpus loading, benchmark-gated nested corpus proposal quality, training summaries, deterministic JSON package export, `scpn-meta` packaging-readiness manifests, CLI manifest export, package entry-point metadata, package entry-point release evidence, direct console-entry invocation evidence, non-publishing distribution evidence, pre-publish enforcement, and reference-suite manifest-readiness gates are implemented; larger private or partner real audit-history corpora and actual optional package publication remain open |
| Plugin ecosystem | manifest registry foundation is implemented for entry-point discovery, capability declarations, compatibility checks, audit records, marketplace catalogue packaging, a runnable catalogue example, CLI catalogue export, Rust-facing flattened registry export, and benchmark-gated guarded metadata-only Rust runtime handoff manifests/CLI export with grouped dispatch records, target hashes, blocked incompatible entries, mutual-exclusive Rust output modes, deterministic hashes, and loading disabled by default; deeper owned runtime loading remains open |

## Next High-Impact Differentiated Moves

These items are the next product/research moves that would most clearly
separate SPO from ordinary simulation or dashboard tooling. They are TODO
items until they have implementation evidence, tests, and reproducible docs.

| Move | Acceptance shape |
|------|------------------|
| Auto-discovery / data-driven binding | raw CSV, sensor logs, and event streams produce reviewable oscillator, channel, initial `K`, and binding proposals via SINDy, coupling estimation, clustering, and validation diagnostics; zero-config CSV sample-rate inference, sparse-derivative evidence, phase-aware Kuramoto SINDy evidence, residual-scored SINDy library selection, correlation graph evidence, lagged directed graph inference, transfer-entropy coupling audit export, extractor-parameter proposals, validator-accepted `auto_initial_k` binding, and reference-suite proposal-quality benchmarks are in place; deterministic domain-like fixtures now enforce domain-specific gates for extractor coverage, expected-edge recall, validation errors, sample count, and proposed-edge multiplier before any automatic runtime use; private or partner live datasets remain future work under the same review-only threshold contract |
| Bayesian UPDE uncertainty | sampled `omega` and `K_nm` distributions propagate through the existing UPDE kernel and report posterior-predictive `R ± sigma`, credible intervals, sampled final phases, and JSON-safe audit diagnostics; Gaussian posterior fitting from observed Kuramoto trajectories is benchmark-gated for residual quality, parameter recovery, uncertainty width, zero-diagonal non-negative coupling, and posterior rollout sample count; reserved NumPyro/BlackJAX sampler names are benchmarked as fail-closed until real validated sampler implementations exist |
| RL / optimisation on knobs | JAX `nn/` and `autotune` train replay-only or simulator-backed policies for `K`, `alpha`, `zeta`, `Psi`, channel weights, and cross-channel gains using `R_good - penalty(R_bad, safety, regime churn)` objectives; `nn.supervisor` now provides a differentiable closed-loop policy/loss surface, and the reference suite now gates PPO-like, SAC-like, and hybrid-physics replay proposals across deterministic multi-scenario fixtures while full PPO/SAC experiments, baselines, and publication artefacts remain open |
| Hierarchical and distributed orchestration | nested edge supervisors maintain local coherence, exchange bounded aggregate evidence upward, and synchronise UPDE phase vectors through digest-checked, sequence-checked, bounded-correction gossip; live owned transports and transport smoke gates now sit on replay and adapter semantics proof for production ownership |
| Formal verification export | Petri nets, policy DSL, STL monitors, and supervisor transitions export to PRISM, SPIN, TLA+, SMT, or equivalent proof workflows with reproducible safety artefacts for plasma, power, and medical-style domainpacks |
| Neuromorphic and quantum tighter integration | validated phase-control plans emit Lava/BrainScaleS-style schedules and QPU control schedules through partner bridges, with execution disabled until real target evidence, hashes, parity, and operator approval exist |
| Plugin ecosystem | standard Python and Rust extension interfaces let domain experts add extractors, monitors, actuators, and bridges with schema validation, capability metadata, audit records, compatibility tests, and no core fork |
| Observability and digital-twin polish | Prometheus/Grafana live-run telemetry, digital-twin residual channels, adapter health, replay linkage, and twin mismatch evidence become first-class operator surfaces; digital-twin sync validations now reduce to a shared live/replay operator evidence record, and runtime Prometheus text exports accepted/rejected sync counts, adapter health, latest sequence, residual extrema, status, capabilities, directions, and mismatch reasons from the same record |

## Deferred Maintenance Track

These items are acknowledged but not the current documentation slice:

| Item | Status |
|------|--------|
| Typed NumPy signature sweep | tracked as maintenance; STL fallback robustness and strange-loop supervisor helper arrays are parameterised; continue scoped module-by-module sweeps |
| Visual and batch-heavy features | deferred unless a user-facing workflow needs them immediately |
| Broad benchmark-file expansion | useful, but benchmark numbers must be measured and reproducible before publication |
| Thin-test strengthening | long-running maintenance backlog; do opportunistically when touching a module, but do not let broad test-hardening displace active feature/module roadmap work |
| Superficial assertion audit | initial audit is recorded in the local internal ledger; remediate flagged candidates incrementally as modules are changed by replacing shape/type/existence-only assertions with behavioural, invariant, error-semantic, parity, or pipeline-effect checks |
| Auxiliary backend review | keep non-primary backends experimental unless measured value justifies maintenance cost |

## Research And Experimental Track

These areas may evolve behind explicit experimental flags or separate guides:

| Area | Roadmap stance |
|------|----------------|
| ML-assisted binding proposals | promising, but outputs must remain reviewable binding specs |
| Learned supervisor policies | differentiable `nn.supervisor` policy surface exists for replay/simulator training; live use remains research-only until auditability, safety constraints, and benchmark evidence are complete |
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
| Sheaf-cohomology control | supervisor foundation, edge-consensus, power-grid, and network-security heterogeneous-domain demos, and obstruction-summary triage are implemented; additional heterogeneous demos and deeper obstruction hardening remain open |
| Federated meta-orchestrator | privacy-preserving policy-gradient aggregation across nodes without raw time-series exchange |
| Byzantine-fault-tolerant meta-orchestrator | offline three-node BFT consensus manifests now verify signed policy proposals, hash-linked audit parents, quorum winners, rejected nodes, and non-actuating review gates; live distributed transport remains open |
| Quantum-native compiler target | deterministic OpenQASM 3 compiler manifests now cover Qiskit/PennyLane handoff with Z-frequency terms, symmetrised XY coupling terms, co-simulation parity evidence, SHA-256 hashes, QPU execution disabled, non-executing target-readiness audits for backend, credential, operator-approval, and stable-hash evidence, and reference-suite gates for blocked/ready target-readiness quality; real QPU target execution remains open |
| Neuromorphic compiler target | deterministic Lava/PyNN schedule manifests from `UPDEState` now include population records, projections, control-action review records, simulator parity evidence, SHA-256 coverage, disabled hardware writes, non-executing hardware target-readiness audits, and reference-suite gates for blocked/ready target-readiness quality; real neuromorphic target execution remains open |
| Hybrid neuromorphic-quantum co-compiler | deterministic hybrid manifests now combine quantum compiler and neuromorphic schedule artefacts under shared N-channel audit semantics, component hashes, parity status, disabled execution gates, non-executing hybrid target-readiness audits that link quantum and neuromorphic readiness hashes before operator handoff, deterministic non-executing operator handoff packages, and reference-suite gates for hash-linked ready/blocked readiness and package evidence; real hybrid target execution remains open |
| Value-alignment supervisor guard | guard foundation, binding-spec templates, counterfactual violation/score reporting, all current domainpack templates, and deterministic replay-calibration evidence are implemented, including cardiac rhythm, power grid, autonomous vehicle, satellite, power-safety N-channel, network-security, financial-markets, chemical-reactor, manufacturing SPC, robotic CPG, swarm-robotics, traffic-flow, plasma-control, fusion-equilibrium, neuroscience EEG, brain-connectome, sleep-architecture, circadian-biology, epidemic SIR, agent-coordination, quantum-simulation, identity-coherence, PLL-clock, digital-twin N-channel, edge-consensus N-channel, firefly-swarm, QueueWaves, rotating-machinery, laser-array, bio-stub, vortex-shedding, musical-acoustics, gene-oscillator, geometry-walk, metaphysics-demo, and minimal-domain; real deployment calibration remains open |
| Autopoietic lineage sandbox | deterministic resource-bounded offline child-policy lineage manifests over audit replay summaries are implemented with review-only policy diffs, accepted/rejected child evidence, disabled live merge/hot patch/actuation flags, and reference-suite gates; richer replay corpora and operator UI surfacing remain open |
| Temporal-causal hypergraph experiments | explicitly experimental time-symmetric rollout research; no production claim without a conventional causal baseline |
| Intergenerational policy inheritance | signed lineage metadata for child orchestrators, inherited policy genomes, multi-objective replay fitness, and merge-only reviewed hot patches |
| Sheaf-theoretic coherence manifold | obstruction-aware control primitive over a sheaf Laplacian with audit-visible cohomology dimensions |
| Constitutional value-alignment guard | Pareto objective constraints in binding specs, counterfactual violation logs, and forced safe fallback path |
| Strange-loop meta-orchestrator | monitor foundation is implemented; long-run drift scenarios and studio surfacing remain open |
| Morphogenetic field topology | field foundation, dependency-free snapshot visualisation, passive SVG rendering, plus swarm-robotics split-flock, power-grid stressed-replay, traffic-flow corridor-spillback, plasma-control edge-localised, and network-security lateral-movement demos are implemented; additional domainpack demos and richer Studio UI rendering remain open |
| Integrated-information monitor | monitor foundation, audit-report summary integration, and expanded deterministic approximation benchmark cases are implemented; broader empirical replay benchmark corpus remains open |
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
