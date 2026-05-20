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
| `auto_binding_synthetic_quality` | Synthetic auto-binding extractor/K proposal quality | 4 fixtures | 4 domain gates | 0.05281981098232791 | 75.72916157042464 | validation errors = 0; extractor coverage = 1.0; expected edge recall = 1.0; proposed edges = 33; accepted domains = 4/4 |
| `replay_policy_candidate_quality` | Replay-only PPO/SAC/hybrid policy candidate quality | 3 learners | 3 acceptance gates | 0.003457240993157029 | 867.7439628703775 | accepted learners = 3/3; min coherence improvement = 0.05827974999403174; unsafe acceptances = 0; non-actuating = yes |
| `bayesian_posterior_fit_quality` | Bayesian posterior fit from observed Kuramoto phases | 96 samples | 128 posterior rollouts | 2.4910545359598473 | 38.537895744225246 | residual RMSE = 3.904347277377099e-07; omega max error = 0.007744271156763904; K max error = 0.029439030191471344; interval width = 0.002121338455159605; accepted = yes |
| `bayesian_backend_fail_closed` | Bayesian backend availability and fail-closed gate | 3 backends | 3 backend probes | 0.29438505397411063 | 10.19073475198857 | available backends = 1; fail-closed backends = 2; unexpected reserved successes = 0; accepted = yes |
| `formal_export_artifact_quality` | PRISM/TLA/STL formal-export artefact quality | 5 artefacts | 5 fail-closed probes | 0.0003819430130533874 | 13090.958151134206 | identifier maps = 22; fail-closed = 5; deterministic hash = yes; SHA-256 = 74217fecfc92b3cf0d3d87f7b58c4278d4c758a1309f11c6d99bac429a57e378 |
| `domain_formal_safety_exports` | Plasma, power-grid, and medical-style formal safety artefacts | 3 domains | 9 artefacts | 0.00038152397610247135 | 23589.605277081566 | accepted domains = 3/3; SHA-256 = ca29f17d051e8206fcd9b7a56063a79dd6e6d16746b7ce800482e4e7297c504b |
| `hybrid_cocompiler_review_gate` | Hybrid quantum/neuromorphic review manifest gate | 1 manifest | 2 blocked probes | 0.00010968500282615423 | 9117.016677156445 | target backends = 4; component hashes = 3; non-actuating = yes; SHA-256 = e5510f11f3339e62ad54b723a53e737835b5c5c4d2a0274f3539533099073fa7 |
| `kuramoto_reference_strogatz_2000` | Strogatz-style all-to-all Kuramoto reference | 64 oscillators | 1000 | 0.13413915602723137 | 7454.944772404798 | final `R` = 1.0 |
| `stuart_landau_reference_pikovsky_2001` | Pikovsky-style coupled amplitude/phase reference | 64 oscillators | 1000 | 0.26052704197354615 | 3838.3731394053893 | final mean amplitude = 3.6193922141707704 |
| `petri_net_reachability` | Supervisor reachability traversal | 4 places | 5000 | 0.023899629013612866 | 209208.26834391762 | reachable markings = 4 |

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
hybrid-physics learner proposals against a simulator-backed coherence surface.
All proposals remain review-only (`actuation_permitted = false`) and must pass
minimum acceptance rate, minimum coherence improvement, zero unsafe accepted
candidates, and non-actuating output gates.

| Learner proposal | Candidate count | Baseline coherence | Selected coherence | Coherence improvement | Accepted |
|------------------|----------------:|-------------------:|-------------------:|----------------------:|----------|
| `ppo_like_replay` | 15 | 0.793 | 0.8653832908831077 | 0.07238329088310769 | yes |
| `sac_like_replay` | 15 | 0.793 | 0.8512797499940318 | 0.05827974999403174 | yes |
| `hybrid_physics_replay` | 15 | 0.793 | 0.8581479119360509 | 0.06514791193605085 | yes |

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
closed before text generation.

| Metric | Snapshot value | Gate |
|--------|---------------:|------|
| Artefact count | 5 | >= 5 |
| Identifier-map entries | 22 | >= 12 |
| Fail-closed malformed probes | 5 | >= 4 |
| Deterministic hash | 1 | required |
| Petri PRISM bytes | 1012 | > 0 |
| Petri TLA bytes | 1281 | > 0 |
| Policy PRISM bytes | 1116 | > 0 |
| Policy TLA bytes | 1370 | > 0 |
| STL PRISM bytes | 808 | > 0 |

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

## Use Policy

- Re-run `PYTHONPATH=src python benchmarks/reference_suite.py` before using the
  values for a release note, paper table, or performance claim.
- Keep the command, backend, versions, platform, and snapshot date next to any
  copied result.
- Do not compare this page against a different host or backend without adding a
  separate dated artefact.
