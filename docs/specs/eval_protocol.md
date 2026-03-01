# Evaluation Protocol

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
