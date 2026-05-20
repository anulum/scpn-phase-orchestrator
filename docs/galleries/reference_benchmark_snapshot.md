<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Reference Benchmark Snapshot -->

# Reference Benchmark Snapshot

This page publishes a dated benchmark snapshot from the reference suite. Treat
these numbers as historical measurements for the listed environment, not as
fresh validation unless the command is rerun and the JSON artefact is updated.

## Reproduction Metadata

| Field | Value |
|-------|-------|
| Snapshot date | `2026-05-20` |
| Suite version | `reference_suite_v1` |
| Command | `PYTHONPATH=src python benchmarks/reference_suite.py` |
| Backend | `python_numpy` |
| Python | `CPython 3.12.3` |
| NumPy | `2.2.6` |
| Platform | `Linux-6.17.0-23-generic-x86_64-with-glibc2.39` |
| Executable | `/media/anulum/724AA8E84AA8AA75/aaa_God_of_the_Math_Collection/03_CODE/SCPN-PHASE-ORCHESTRATOR/.venv/bin/python` |
| JSON artefact | `benchmarks/results/reference_suite.json` |

## Historical Results

| Suite ID | Reference surface | Size | Steps | Wall time (s) | Steps/s | Summary value |
|----------|-------------------|------|-------|---------------|---------|---------------|
| `auto_binding_synthetic_quality` | Synthetic auto-binding extractor/K proposal quality | 4 fixtures | 4 domain gates | 0.04307294404134154 | 92.86572090732382 | validation errors = 0; extractor coverage = 1.0; expected edge recall = 1.0; proposed edges = 33; accepted domains = 4/4 |
| `semantic_retrieval_ranking_quality` | Symbolic compiler retrieval ranking quality | 3 evidence records | 3 ranked records | 0.015583062020596117 | 192.51672078535677 | top source = domainpack; top domainpack = power_grid; feature-complete = 3; retrieval score = 1.0; SHA-256 = 88f658e0c7222d27a3e1125be74fda54ff07f272ac1deb90f545393df8a55b2d |
| `replay_policy_candidate_quality` | Replay-only PPO/SAC/hybrid policy candidate quality | 3 scenarios | 9 learner gates | 0.01423552498454228 | 632.221151645106 | accepted scenarios = 3/3; accepted learners = 9/9; min coherence improvement = 0.035689587760827646; unsafe acceptances = 0; non-actuating = yes |
| `bayesian_posterior_fit_quality` | Bayesian posterior fit from observed Kuramoto phases | 96 samples | 128 posterior rollouts | 2.1922598049859516 | 43.790430213455096 | residual RMSE = 3.904347277377099e-07; omega max error = 0.007744271156763904; K max error = 0.029439030191471344; interval width = 0.002121338455159605; accepted = yes |
| `bayesian_backend_fail_closed` | Bayesian backend availability and fail-closed gate | 3 backends | 3 backend probes | 0.2778991900268011 | 10.79528155411563 | available backends = 1; fail-closed backends = 2; unexpected reserved successes = 0; accepted = yes |
| `formal_export_artifact_quality` | PRISM/TLA/STL formal-export artefact quality | 5 artefacts | 3 package properties | 0.0005473069613799453 | 9135.641153537157 | identifier maps = 22; fail-closed = 5; checker commands = 3; checker readiness = 2 ready/1 missing; checker execution disabled = yes; package SHA-256 = b1d5207b71b84ecc674b0d203206371f0861bd8cc03667592dfa060bac171a92 |
| `stl_closed_loop_plan_quality` | Offline STL closed-loop synthesis plan quality | 3 plans | 1 projected action | 0.0003578549949452281 | 8383.283850653443 | projected actions = 1; rejected candidates = 1; blocked reasons = 3; non-actuating = yes; SHA-256 = c5e8bc18e6edef4cd0913b4d3fef1acf7f27865ffc64ef2903e15457d64a4c28 |
| `domain_formal_safety_exports` | Plasma, power-grid, and medical-style formal safety artefacts | 3 domains | 9 artefacts | 0.0006097939913161099 | 14759.0827856067 | accepted domains = 3/3; SHA-256 = ca29f17d051e8206fcd9b7a56063a79dd6e6d16746b7ce800482e4e7297c504b |
| `hybrid_cocompiler_review_gate` | Hybrid quantum/neuromorphic review manifest gate | 1 manifest | 2 blocked probes | 0.00011891900794580579 | 8409.084613754294 | target backends = 4; component hashes = 3; non-actuating = yes; SHA-256 = e5510f11f3339e62ad54b723a53e737835b5c5c4d2a0274f3539533099073fa7 |
| `quantum_target_readiness_gate` | Non-executing QPU target-readiness audit gate | 2 readiness records | 1 blocked / 1 ready | 0.0001809889799915254 | 11050.39654952278 | ready = 1; blocked = 1; blocked reasons = 2; non-executing = yes; readiness SHA-256 = aa0f85ce5bbfd35acf04d96e29d3bb64edf7ce5b091193263b13712d98f6134c |
| `neuromorphic_target_readiness_gate` | Non-executing neuromorphic hardware target-readiness audit gate | 2 readiness records | 1 blocked / 1 ready | 0.00024182797642424703 | 8270.341709725644 | ready = 1; blocked = 1; blocked reasons = 3; non-executing = yes; readiness SHA-256 = fbff4ea82152b5fb51733f179661b8ec117b1afc076c80972e610af7717368d0 |
| `hybrid_target_readiness_gate` | Non-executing hybrid target-readiness audit gate | 2 readiness records | 1 blocked / 1 ready | 0.00011475704377517104 | 17428.124097709828 | ready = 1; blocked = 1; blocked reasons = 1; component hash linked = 1; non-executing = yes; readiness SHA-256 = 5dbf280c524594e46047c0fb342383df767713c6ebdd71d43eb5fdc5a0b5cc64 |
| `hybrid_operator_handoff_package_gate` | Non-executing hybrid operator handoff package gate | 2 packages | 1 blocked / 1 ready | 0.0001218619872815907 | 16412.008737216234 | ready packages = 1; blocked packages = 1; blocked reasons = 1; hash chain linked = 1; non-executing = yes; package SHA-256 = c742f1c3a2ba7bfad9e1266f743c43120df8cf5c9e12da865dbcf0436b879eb5 |
| `meta_transfer_audit_corpus_quality` | Cross-domain meta-transfer audit-corpus proposal gate | 6 records | 3 neighbours | 0.0032925489940680563 | 1822.296345721737 | domains = 4; top neighbour = power_grid; confidence = 0.9977061617091371; deterministic hash = yes; proposal SHA-256 = bfef70f740fbdedc765080f9c9bb0156ec046fd0351dab4f4117c7c259a781fb |
| `meta_transfer_package_manifest_quality` | Cross-domain meta-transfer package manifest gate | 4 records | 4 domain gates | 0.00042936898535117507 | 9315.99658212028 | feature keys = 5; knobs = 4; digest linked = yes; execution disabled = yes; manifest SHA-256 = bf551f9836e581eba6469309784c56e3b3fd5cbfee23e55cee6018f0163df6af |
| `plugin_ecosystem_catalog_quality` | Python/Rust plugin marketplace capability gate | 3 manifests | 6 handoff target hashes | 0.0002764470409601927 | 10851.988104412334 | compatible plugins = 2/3; incompatible monitor rejection = 1; required kinds = 4/4; loading disabled = yes; handoff SHA-256 = db0fd80e5a3d3468412f0314558b017f9a2f4473d5d7a9ab768e40d86eaf3f77 |
| `kuramoto_reference_strogatz_2000` | Strogatz-style all-to-all Kuramoto reference | 64 oscillators | 1000 | 0.13140711496816948 | 7609.938017756712 | final `R` = 1.0 |
| `stuart_landau_reference_pikovsky_2001` | Pikovsky-style coupled amplitude/phase reference | 64 oscillators | 1000 | 0.2567110530217178 | 3895.4302443510283 | final mean amplitude = 3.6193922141707704 |
| `petri_net_reachability` | Supervisor reachability traversal | 4 places | 5000 | 0.018718710052780807 | 267112.42312646494 | reachable markings = 4 |

## Auto-Binding Acceptance Gates

The auto-binding benchmark now evaluates larger deterministic domain-like
datasets instead of only toy smoke fixtures. Each fixture has explicit
domain-specific thresholds for minimum extractor coverage, expected-edge recall,
maximum validation errors, minimum sample count, and maximum proposed-edge
multiplier. The 2026-05-20 snapshot passed all four gates:

| Domain fixture | Samples | Expected-edge recall | Extractor coverage | Proposed-edge multiplier | Accepted |
|----------------|--------:|---------------------:|-------------------:|-------------------------:|----------|
| `phase_chain` | 128 | 1.0 | 1.0 | 6.0 | yes |
| `industrial_sensor_chain` | 128 | 1.0 | 1.0 | 6.0 | yes |
| `cardiac_rhythm_surrogate` | 160 | 1.0 | 1.0 | 4.5 | yes |
| `power_grid_surrogate` | 192 | 1.0 | 1.0 | 6.0 | yes |

## Semantic Retrieval Ranking Acceptance Gates

The semantic retrieval benchmark uses a deterministic synthetic domainpack/docs
corpus to exercise symbolic-to-binding retrieval ranking. The gate requires
ranked evidence records, complete ranking features, domainpack precedence for
the best matching power-grid corpus item, positive retrieval score, and a stable
ranking hash.

| Metric | Snapshot value | Gate |
|--------|---------------:|------|
| Evidence records | 3 | >= 3 |
| Ranked records | 3 | >= 3 |
| Feature-complete records | 3 | >= 3 |
| Domainpack top-rank flag | 1 | required |
| Top source | `domainpack` | domainpack |
| Top domainpack | `power_grid` | power_grid |
| Retrieval score | 1.0 | > 0 |
| Deterministic hash | 1 | required |
| Ranking SHA-256 | `88f658e0c7222d27a3e1125be74fda54ff07f272ac1deb90f545393df8a55b2d` | stable |

## Replay-Policy Acceptance Gates

The replay-policy benchmark evaluates deterministic PPO-like, SAC-like, and
hybrid-physics learner proposals across deterministic replay scenarios against
a simulator-backed coherence surface. All proposals remain review-only
(`actuation_permitted = false`) and must pass minimum acceptance rate, minimum
coherence improvement, zero unsafe accepted candidates, and non-actuating output
gates.

| Replay scenario | Learners | Baseline coherence | Minimum coherence improvement | Unsafe acceptances | Accepted |
|-----------------|---------:|-------------------:|------------------------------:|-------------------:|----------|
| `two_channel_low_coupling` | 3 | 0.793 | 0.05827974999403174 | 0 | yes |
| `three_channel_cross_gain` | 3 | 0.7758666666666668 | 0.05663105080295605 | 0 | yes |
| `stability_recovery` | 3 | 0.8022666666666668 | 0.035689587760827646 | 0 | yes |

## Bayesian Posterior-Fit Acceptance Gates

The Bayesian posterior benchmark fits Gaussian `omega` and `K_nm`
distributions from a deterministic Kuramoto trajectory, then runs the fitted
posterior through the existing Bayesian UPDE rollout path. The gate requires
finite audit diagnostics, non-negative zero-diagonal coupling, bounded residual
error, bounded parameter-recovery error, bounded credible interval width, and
at least 96 posterior rollout samples.

| Metric | Snapshot value | Gate |
|--------|---------------:|------|
| Residual RMSE | 3.904347277377099e-07 | <= 0.0025 |
| Max omega mean absolute error | 0.007744271156763904 | <= 0.03 |
| Max `K_nm` mean absolute error | 0.029439030191471344 | <= 0.06 |
| Credible interval width | 0.002121338455159605 | <= 0.01 |
| Posterior rollout samples | 128 | >= 96 |
| Finite audit record | 1 | required |
| Zero-diagonal coupling | 1 | required |
| Non-negative coupling | 1 | required |

## Bayesian Backend Fail-Closed Gates

The backend benchmark probes the shipped NumPy implementation and the reserved
sampler names through the same `bayesian_upde_run()` execution path. NumPy must
complete with samples; reserved names must fail closed until they have real,
validated sampler implementations.

| Backend | Available | Fail closed | Sample count |
|---------|-----------|-------------|-------------:|
| `numpy` | yes | no | 16 |
| `numpyro` | no | yes | 0 |
| `blackjax` | no | yes | 0 |

## Formal-Export Acceptance Gates

The formal-export benchmark emits deterministic PRISM/TLA/STL artefacts for a
bounded Petri protocol, policy-rule set, and STL monitor set. It also probes
malformed nets, rules, and STL predicates to ensure unsupported shapes fail
closed before text generation. External checker readiness is recorded as a
non-executing audit: commands are generated, executable availability is
reported, and missing tools remain fail-closed instead of being invoked.

| Metric | Snapshot value | Gate |
|--------|---------------:|------|
| Artefact count | 5 | >= 5 |
| Identifier-map entries | 22 | >= 12 |
| Fail-closed malformed probes | 5 | >= 4 |
| Formal package properties | 3 | >= 3 |
| Formal checker commands | 3 | >= 3 |
| Checker readiness records | 3 | >= 3 |
| Ready checker records | 2 | evidence |
| Missing checker records | 1 | >= 1 fail-closed |
| Checker execution disabled | 1 | required |
| Checker-readiness execution disabled | 1 | required |
| Deterministic hash | 1 | required |
| Package SHA-256 | `b1d5207b71b84ecc674b0d203206371f0861bd8cc03667592dfa060bac171a92` | stable |
| Petri PRISM bytes | 1012 | > 0 |
| Petri TLA bytes | 1281 | > 0 |
| Policy PRISM bytes | 1116 | > 0 |
| Policy TLA bytes | 1370 | > 0 |
| STL PRISM bytes | 808 | > 0 |

## STL Closed-Loop Plan Acceptance Gates

The STL closed-loop benchmark evaluates offline synthesis plans for builtin STL
monitor automata. The gate requires deterministic non-actuating plans, at least
one policy-projected bounded action, fail-closed blockers for unmapped
candidates, no-action behaviour for already satisfied monitors, and a stable
plan hash. These records are review artefacts; they do not mutate runtime state
or bypass the safety/actuation stack.

| Metric | Snapshot value | Gate |
|--------|---------------:|------|
| Plan count | 3 | >= 3 |
| Projected actions | 1 | >= 1 |
| Rejected candidates | 1 | evidence |
| Blocked reasons | 3 | >= 3 |
| Non-actuating flag | 1 | required |
| Deterministic hash | 1 | required |
| Plan SHA-256 | `c5e8bc18e6edef4cd0913b4d3fef1acf7f27865ffc64ef2903e15457d64a4c28` | stable |

## Domain Formal-Safety Gates

The domain formal-safety benchmark emits policy PRISM, policy TLA, and STL
PRISM artefacts for plasma-control, power-grid, and medical/cardiac-style
profiles. Each domain must provide at least two policy rules, at least two STL
specifications, three deterministic artefacts, and required reachability/STL
labels.

| Domain profile | Artefacts | Rules | STL specs | Identifier maps | Accepted |
|----------------|----------:|------:|----------:|----------------:|----------|
| `plasma_control` | 3 | 2 | 2 | 12 | yes |
| `power_grid` | 3 | 2 | 2 | 12 | yes |
| `medical_cardiac` | 3 | 2 | 2 | 12 | yes |

## Hybrid Co-Compiler Review Gates

The hybrid co-compiler benchmark combines a deterministic quantum compiler
manifest and a deterministic neuromorphic schedule manifest under one shared
audit envelope. It requires simulator parity, component hashes, linked target
backends, and forced non-actuation; permission leaks and parity failures must
produce blocked review manifests.

| Metric | Snapshot value | Gate |
|--------|---------------:|------|
| Target backends | 4 | >= 4 |
| Component hashes | 3 | = 3 |
| Quantum term count | 3 | >= 3 |
| Neuromorphic sample count | 2 | >= 2 |
| Blocked probes | 2 | >= 2 |
| Non-actuating flags | 1 | required |
| Deterministic hash | 1 | required |

## Quantum Target-Readiness Gates

The quantum target-readiness benchmark validates the non-executing handoff audit
for declared QPU backends. The gate requires one blocked record, one
ready-not-executed record, credential and operator-approval blockers, stable
readiness hashes, operator commands, and forced disabled QPU execution and
actuation. These records are target-readiness evidence only; they do not submit
jobs, run simulators, or permit external actuation.

| Metric | Snapshot value | Gate |
|--------|---------------:|------|
| Readiness records | 2 | >= 2 |
| Ready-not-executed records | 1 | >= 1 |
| Blocked records | 1 | >= 1 |
| Blocked reasons | 2 | >= 2 |
| Operator commands | 6 | >= 6 |
| Non-executing flag | 1 | required |
| Deterministic hash | 1 | required |
| Manifest SHA-256 | `e323283dbcdc138915a6d2a9728fdcce9dfa9600245428298d60c21b3a5ac30d` | stable |
| Ready readiness SHA-256 | `aa0f85ce5bbfd35acf04d96e29d3bb64edf7ce5b091193263b13712d98f6134c` | stable |

## Neuromorphic Target-Readiness Gates

The neuromorphic target-readiness benchmark validates the non-executing hardware
handoff audit for declared Lava/PyNN-style targets. The gate requires one
blocked record, one ready-not-executed record, credential, operator-approval,
and external simulator-parity blockers, stable readiness hashes, operator
commands, and forced disabled hardware writes and actuation. These records are
target-readiness evidence only; they do not submit jobs, run external
simulators, or permit hardware actuation.

| Metric | Snapshot value | Gate |
|--------|---------------:|------|
| Readiness records | 2 | >= 2 |
| Ready-not-executed records | 1 | >= 1 |
| Blocked records | 1 | >= 1 |
| Blocked reasons | 3 | >= 3 |
| Operator commands | 6 | >= 6 |
| Non-executing flag | 1 | required |
| Deterministic hash | 1 | required |
| Manifest SHA-256 | `b6d66744f1488a0711c5b40a7fef273c3ab81e8a8eb030ab873e4e75f831600a` | stable |
| Ready readiness SHA-256 | `fbff4ea82152b5fb51733f179661b8ec117b1afc076c80972e610af7717368d0` | stable |

## Hybrid Target-Readiness Gates

The hybrid target-readiness benchmark validates the non-executing handoff audit
that links the quantum and neuromorphic readiness records to the combined
co-compiler manifest. The gate requires one blocked record, one
ready-not-executed record, explicit hybrid operator approval for readiness,
component-hash linkage, stable readiness hashes, operator commands, and forced
disabled QPU execution, hardware writes, and actuation. These records are
readiness evidence only; they do not submit jobs, run external simulators, or
permit hybrid actuation.

| Metric | Snapshot value | Gate |
|--------|---------------:|------|
| Readiness records | 2 | >= 2 |
| Ready-not-executed records | 1 | >= 1 |
| Blocked records | 1 | >= 1 |
| Blocked reasons | 1 | >= 1 |
| Operator commands | 6 | >= 6 |
| Component hash linked | 1 | required |
| Non-executing flag | 1 | required |
| Deterministic hash | 1 | required |
| Hybrid manifest SHA-256 | `e5510f11f3339e62ad54b723a53e737835b5c5c4d2a0274f3539533099073fa7` | stable |
| Ready readiness SHA-256 | `5dbf280c524594e46047c0fb342383df767713c6ebdd71d43eb5fdc5a0b5cc64` | stable |

## Hybrid Operator Handoff Package Gates

The hybrid operator handoff benchmark validates deterministic, non-executing
packages prepared for external approved operator workflows. The gate requires
one blocked package, one ready-not-executed package, blocked-reason preservation,
operator commands, package-hash determinism, manifest/readiness hash-chain
linkage, and forced disabled execution, QPU execution, hardware writes, and
actuation. These packages are review artefacts only; they do not submit jobs,
run external simulators, or permit hybrid actuation.

| Metric | Snapshot value | Gate |
|--------|---------------:|------|
| Packages | 2 | >= 2 |
| Ready-not-executed packages | 1 | >= 1 |
| Blocked packages | 1 | >= 1 |
| Blocked reasons | 1 | >= 1 |
| Operator commands | 8 | >= 8 |
| Hash chain linked | 1 | required |
| Non-executing flag | 1 | required |
| Deterministic hash | 1 | required |
| Ready package SHA-256 | `c742f1c3a2ba7bfad9e1266f743c43120df8cf5c9e12da865dbcf0436b879eb5` | stable |

## Meta-Transfer Audit-Corpus Gates

The meta-transfer audit-corpus benchmark loads a deterministic nested JSONL
corpus with mixed SPO audit shapes, fits a review-only cross-domain transfer
model, and verifies proposal quality against a held-out power-grid-like query.
The gate requires recursive corpus discovery, multi-domain coverage,
feature/knob coverage, enough nearest-neighbour evidence, high confidence, a
required top-domain match, and a stable proposal hash. It does not actuate,
train online, load plugins, or execute a packaged model.

| Metric | Snapshot value | Gate |
|--------|---------------:|------|
| Training records | 6 | >= 6 |
| Domains | 4 | >= 4 |
| Feature keys | 5 | >= 5 |
| Knobs | 4 | >= 4 |
| Proposal knobs | 4 | >= 4 |
| Neighbours | 3 | >= 3 |
| Top neighbour domain | `power_grid` | `power_grid` |
| Confidence | 0.9977061617091371 | >= 0.97 |
| Deterministic hash | 1 | required |
| Proposal SHA-256 | `bfef70f740fbdedc765080f9c9bb0156ec046fd0351dab4f4117c7c259a781fb` | stable |

## Meta-Transfer Package Manifest Gates

The meta-transfer manifest benchmark packages a deterministic synthetic
multi-domain replay corpus into the review-only `scpn-meta` manifest surface.
The gate requires training-summary coverage, digest linkage between the JSON
package and manifest, public import/console metadata, disabled execution, and a
stable manifest hash. It does not build, install, upload, or execute an optional
package.

| Metric | Snapshot value | Gate |
|--------|---------------:|------|
| Training records | 4 | >= 4 |
| Domains | 4 | >= 4 |
| Feature keys | 5 | >= 5 |
| Knobs | 4 | >= 4 |
| Package bytes | 1950 | > 0 |
| Package digest matches manifest | 1 | required |
| Execution disabled | 1 | required |
| Deterministic hash | 1 | required |
| Package name | `scpn-meta` | `scpn-meta` |
| Import target | `scpn_phase_orchestrator.meta` | public meta facade |
| Console script | `scpn-meta` | proposed optional entry point |
| Package SHA-256 | `533acf3b37aa233b7a53da1903c99865a7e34055d3d5bcacef3501c3b9fd273f` | stable |
| Manifest SHA-256 | `bf551f9836e581eba6469309784c56e3b3fd5cbfee23e55cee6018f0163df6af` | stable |

## Plugin Ecosystem Catalog Gates

The plugin ecosystem benchmark packages validated Python plugin manifests into
the marketplace catalogue and Rust-facing registry. It requires extractor,
monitor, actuator, and bridge capabilities to appear in compatible metadata,
requires incompatible manifests to remain visible to review jobs, and requires a
deterministic registry hash for release evidence.

| Metric | Snapshot value | Gate |
|--------|---------------:|------|
| Compatible plugin count | 2 | >= 2 |
| Full plugin count | 3 | >= 3 |
| Incompatible plugin count | 1 | >= 1 |
| Compatible capability count | 5 | >= 5 |
| Handoff target hashes | 6 | >= 5 |
| Blocked handoff capabilities | 1 | >= 1 |
| Handoff loading disabled | 1 | required |
| Required capability kinds observed | 4 | = 4 |
| Extractor capabilities | 1 | >= 1 |
| Monitor capabilities | 2 | >= 1 |
| Actuator capabilities | 1 | >= 1 |
| Bridge capabilities | 1 | >= 1 |
| Deterministic hash | 1 | required |
| Registry SHA-256 | `4dc86c339a42dba16bfe99c79fd6197051c87c97c7fbbf2a93dd86c1585ff25b` | stable |
| Handoff SHA-256 | `db0fd80e5a3d3468412f0314558b017f9a2f4473d5d7a9ab768e40d86eaf3f77` | stable |

## Use Policy

- Re-run `PYTHONPATH=src python benchmarks/reference_suite.py` before using the
  values for a release note, paper table, or performance claim.
- Keep the command, backend, versions, platform, and snapshot date next to any
  copied result.
- Do not compare this page against a different host or backend without adding a
  separate dated artefact.
