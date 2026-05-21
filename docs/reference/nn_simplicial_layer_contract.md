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
