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

## Operational overview

In practice, teams adopt this contract when they need a single canonical layer
that can be inserted into policy, optimization, or replay tooling without
rebuilding control glue.

The trajectory mode is not an optional analytics feature in production; it is the
artifact most operations teams use to prove that a control adjustment produced the
claimed state path over time. This is why the contract is explicit about output
dimensions and mask-path finiteness.

When this contract is upheld, the orchestrator can route Kuramoto experiments
through either research-oriented training runs or strict deployment runs with the
same shape expectations.

## Deployment boundary

The layer sits at the boundary between model discovery and policy control in both
phase-only and mixed topologies. Every caller assumes this contract when moving from
parameter search to replay and then to governance review.

Recommended usage:

- Keep trajectory mode enabled when evidence replay is required for operator approval.
- Use masked layers in sparse topology studies only after verifying the same shape contract under both
  dense and masked paths.
- If a downstream supervisor reads phase outputs from this layer, do not alter output semantics at the caller.
