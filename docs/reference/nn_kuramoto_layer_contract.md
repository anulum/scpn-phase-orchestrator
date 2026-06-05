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

## Why this is a core model contract

The differentiable layer contract defines the minimal guarantees needed for policy
optimization, control loops, and replacement-by-backend strategies. Stable shape
semantics are mandatory because they determine whether gradient-based methods can
be composed across model and supervisor layers.

Trajectory output is included because production workflows frequently depend on
time-series inspection, not just end state values.

## Enterprise usage notes

- This contract is the standard backbone for differentiable policy and inverse
  coupling surfaces because it guarantees both single-shot and trajectory outputs.
- Keeping output shapes stable allows shared controller code to run across Python,
  Rust, and experimental language adapters without per-call branching.
- Mask-aware finite behaviour is the guard rail for sparse-graph studies where
  topology is inferred automatically and may contain structured exclusions.
