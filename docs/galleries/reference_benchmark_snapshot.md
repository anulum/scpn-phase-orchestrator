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
| NumPy | `2.4.4` |
| Platform | `Linux-6.17.0-23-generic-x86_64-with-glibc2.39` |
| Executable | `/media/anulum/724AA8E84AA8AA75/aaa_God_of_the_Math_Collection/03_CODE/SCPN-PHASE-ORCHESTRATOR/.venv/bin/python` |
| JSON artefact | `benchmarks/results/reference_suite.json` |

## Historical Results

| Suite ID | Reference surface | Size | Steps | Wall time (s) | Steps/s | Summary value |
|----------|-------------------|------|-------|---------------|---------|---------------|
| `auto_binding_synthetic_quality` | Synthetic auto-binding extractor/K proposal quality | 4 fixtures | 4 domain gates | 0.05236929497914389 | 76.38063490434621 | validation errors = 0; extractor coverage = 1.0; expected edge recall = 1.0; proposed edges = 33; accepted domains = 4/4 |
| `replay_policy_candidate_quality` | Replay-only PPO/SAC/hybrid policy candidate quality | 3 scenarios | 9 learner gates | 0.013055607967544347 | 689.358934671874 | accepted scenarios = 3/3; accepted learners = 9/9; min coherence improvement = 0.035689587760827646; unsafe acceptances = 0; non-actuating = yes |
| `bayesian_posterior_fit_quality` | Bayesian posterior fit from observed Kuramoto phases | 96 samples | 128 posterior rollouts | 2.3823742069653235 | 40.295936599433354 | residual RMSE = 3.904347277377099e-07; omega max error = 0.007744271156763904; K max error = 0.029439030191471344; interval width = 0.002121338455159605; accepted = yes |
| `bayesian_backend_fail_closed` | Bayesian backend availability and fail-closed gate | 3 backends | 3 backend probes | 0.2867484980379231 | 10.46212977758385 | available backends = 1; fail-closed backends = 2; unexpected reserved successes = 0; accepted = yes |
| `formal_export_artifact_quality` | PRISM/TLA/STL formal-export artefact quality | 5 artefacts | 3 package properties | 0.0009058300056494772 | 5519.799486455537 | identifier maps = 22; fail-closed = 5; checker commands = 3; checker readiness = 2 ready/1 missing; checker execution disabled = yes; package SHA-256 = b1d5207b71b84ecc674b0d203206371f0861bd8cc03667592dfa060bac171a92 |
| `stl_closed_loop_plan_quality` | Offline STL closed-loop synthesis plan quality | 3 plans | 1 projected action | 0.0005155940307304263 | 5818.531288560482 | projected actions = 1; rejected candidates = 1; blocked reasons = 3; non-actuating = yes; SHA-256 = c5e8bc18e6edef4cd0913b4d3fef1acf7f27865ffc64ef2903e15457d64a4c28 |
| `domain_formal_safety_exports` | Plasma, power-grid, and medical-style formal safety artefacts | 3 domains | 9 artefacts | 0.0006900079897604883 | 13043.327227448526 | accepted domains = 3/3; SHA-256 = ca29f17d051e8206fcd9b7a56063a79dd6e6d16746b7ce800482e4e7297c504b |
| `hybrid_cocompiler_review_gate` | Hybrid quantum/neuromorphic review manifest gate | 1 manifest | 2 blocked probes | 0.0002019969979301095 | 4950.568623529731 | target backends = 4; component hashes = 3; non-actuating = yes; SHA-256 = e5510f11f3339e62ad54b723a53e737835b5c5c4d2a0274f3539533099073fa7 |
| `plugin_ecosystem_catalog_quality` | Python/Rust plugin marketplace capability gate | 3 manifests | 6 handoff target hashes | 0.0004642080166377127 | 6462.619973108575 | compatible plugins = 2/3; incompatible monitor rejection = 1; required kinds = 4/4; loading disabled = yes; handoff SHA-256 = db0fd80e5a3d3468412f0314558b017f9a2f4473d5d7a9ab768e40d86eaf3f77 |
| `kuramoto_reference_strogatz_2000` | Strogatz-style all-to-all Kuramoto reference | 64 oscillators | 1000 | 0.15490243799285963 | 6455.676314443133 | final `R` = 1.0 |
| `stuart_landau_reference_pikovsky_2001` | Pikovsky-style coupled amplitude/phase reference | 64 oscillators | 1000 | 0.28508444398175925 | 3507.7326073392614 | final mean amplitude = 3.6193922141707704 |
| `petri_net_reachability` | Supervisor reachability traversal | 4 places | 5000 | 0.021801681024953723 | 229340.11346543004 | reachable markings = 4 |

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
