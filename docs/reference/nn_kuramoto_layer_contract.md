# `scpn_phase_orchestrator.nn.kuramoto_layer` contract

Purpose:
- Provide a differentiable phase-only oscillator layer with optional sparse
  coupling mask semantics.

Public contract:
1. `KuramotoLayer.__call__(phases)` preserves oscillator dimension.
2. `forward_with_trajectory(...)` returns `(final, trajectory)` with trajectory
   shape `(n_steps, n)`.
3. When `mask` is configured, masked integration path remains shape-stable and
   finite.

Verification:
- `tests/test_kuramoto_layer.py::TestKuramotoLayer::test_forward_shape`
- `tests/test_kuramoto_layer.py::TestKuramotoLayer::test_trajectory`
- `tests/test_kuramoto_layer.py::TestKuramotoLayer::test_masked_layer_path`
