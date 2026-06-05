# Evaluation Protocol

## Purpose and scope

This protocol defines the baseline proof surface for a SPO run before it is treated as a release-grade control artifact. It separates three goals:

- **Performance**: whether the configured intervention improves synchronisation and suppresses unhealthy layers;
- **Safety**: whether boundary rules are respected under the configured policy;
- **Evidence quality**: whether the same outcome can be replayed from immutable audit records.

The protocol is intentionally narrow and deterministic so a reviewer can reproduce outcomes in a fixed, finite window before launching a broader production experiment.

## Boundaries of this protocol

Use this protocol as the minimum quality surface for default releases and CI-backed comparisons. It is not the only valid benchmark style, and it is not a substitute for domain-specific acceptance tests in regulated environments.

When a domain has additional medical, grid, or transport constraints, extend this protocol by adding a domain annex that records those explicit thresholds while preserving the same fixed-seed and replay contract.

## Metrics

| Metric | Computation | Target |
|--------|------------|--------|
| R_good convergence | mean R over good_layers at final step | > 0.8 |
| R_bad suppression | mean R over bad_layers at final step | < 0.3 |
| Boundary compliance | fraction of steps with zero hard violations | 1.0 |
| Action count | total ControlActions issued during run | minimise |
| Convergence speed | steps until R_good first exceeds 0.7 | minimise |

## Procedure

1. Load binding spec.
2. Initialise phases from a fixed seed (`rng(42)`).
3. Run N steps (default 100).
4. Record R_good, R_bad, boundary state, and actions at every step.
5. Compute final metrics from the audit log.

## Deterministic Replay

Replay from an audit log must reproduce the same R_good / R_bad trajectory. Requirements:

- Fixed seed for initial phases.
- Audit log records all ControlActions with exact values.
- No stochastic components in the default supervisor policy.
- `ReplayEngine` loads JSONL and reconstructs step-by-step state.

## Ablation Protocol

To isolate the contribution of each subsystem, run with components disabled:

| Ablation | Modification |
|----------|--------------|
| No coupling | Set `K = 0` (zero matrix). Oscillators free-run. |
| No driver | Set `zeta = 0`. No external entrainment. |
| No supervisor | Disable `SupervisorPolicy`. No ControlActions. |
| No imprint | Disable `ImprintModel`. Static coupling. |

Compare R_good convergence across ablations. The full system should converge faster and higher than any ablation.

## Benchmark Domains

Run eval on all domainpacks in the `domainpacks/` directory:

- `minimal_domain` -- baseline sanity check
- `queuewaves` -- R_bad suppression scenario
- `geometry_walk` -- symbolic channel test
- `bio_stub` -- multi-channel, multi-layer stress test

## Provenance of Evaluation Thresholds

All metric targets in this protocol (R_good > 0.8, R_bad < 0.3, convergence
at 0.7, 100-step default) are empirical engineering judgements calibrated on
the bundled domainpacks. They are not derived from analytical results.
See [ASSUMPTIONS.md](../ASSUMPTIONS.md) § Evaluation Protocol for the full
constant registry.

## Why this protocol is structured this way

The protocol is arranged to separate measurement, control action, and evidence
generation into deterministic steps. That structure allows reproducible comparisons
between baseline runs and control-enhanced runs.

The default 100-step window is an engineering benchmark, not a theorem: it is
chosen to expose divergence or convergence tendencies quickly while keeping replay
time practical in CI and integration checks.

The replay rule makes this protocol production-grade because it requires the same
audit trail to regenerate the exact trajectory, rather than only reproducing a final
summary score.

## Operational usage

- Use `eval_protocol` for release gating before promoting a configuration to shared
  runtime.
- Run ablations to confirm each control subsystem contributes to the target metrics
  instead of masking regressions in another module.
