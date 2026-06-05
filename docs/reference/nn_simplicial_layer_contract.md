# `scpn_phase_orchestrator.nn.simplicial_layer` contract

Purpose:
- Provide a differentiable simplicial Kuramoto layer with learnable 3-body
  coupling (`sigma2`) on top of pairwise coupling.

Public contract:
1. `SimplicialKuramotoLayer.__call__(phases)` preserves oscillator dimension and
   returns finite phase values in `[0, 2π)`.
2. `forward_with_trajectory(...)` returns deterministic `(final, trajectory)`
   with trajectory shape `(n_steps, n)`.
3. Non-zero `sigma2` changes dynamics relative to `sigma2=0` for non-degenerate
   inputs, confirming active 3-body path wiring.

Verification:
- `tests/test_simplicial_layer.py::TestForward::test_output_shape`
- `tests/test_simplicial_layer.py::TestTrajectory::test_trajectory_shape`
- `tests/test_simplicial_layer.py::TestSigma2Activation::test_sigma2_changes_dynamics`

## Operational meaning

This contract covers higher-order coupling support. In production terms, the
3-body term is the explicit mechanism for representing non-pairwise interactions
without reengineering the broader orchestration stack.

The contract distinguishes when the higher-order path is active (`sigma2 != 0`) so
teams can A/B compare pairwise and simplicial dynamics with measurable output
differences.

## When to engage simplicial coupling

- Use simplicial coupling when pairwise interaction models under-explain coherence
  transitions in empirical datasets.
- The active-path requirement (`sigma2 != 0`) is an explicit switch that lets teams
  separate model-complexity experiments from base-pairwise baseline experiments.
- Shape invariants and deterministic trajectory output keep higher-order terms
  auditable against baseline pairwise runs.

## Practical deployment notes

The simplicial contract is primarily a complexity dial: teams do not switch on
`sigma2` by default. They activate it when baseline pairwise dynamics fail to match
cross-time coherence structure in the target domain.

From an operations perspective, the strict difference contract between
`sigma2=0` and `sigma2!=0` avoids the most common source of ambiguity:
whether observed differences are from model structure or from random run variance.

When this contract is respected, higher-order coupling becomes a controlled
experiment variable rather than a hidden change in numerical behavior.
