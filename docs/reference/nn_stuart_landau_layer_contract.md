# `scpn_phase_orchestrator.nn.stuart_landau_layer` contract

Purpose:
- Provide differentiable phase-plus-amplitude oscillator evolution for the
  runtime neural layer surface.

Public contract:
1. `StuartLandauLayer.__call__(phases, amplitudes)` returns
   `(final_phases, final_amplitudes)` with unchanged oscillator dimension.
2. `forward_with_trajectory(...)` returns
   `(final_phases, final_amplitudes, phase_trajectory, amplitude_trajectory)`
   where trajectories have leading dimension `n_steps`.
3. `sync_score(...)` and `mean_amplitude(...)` return finite scalar summaries.

Verification:
- `tests/test_stuart_landau_nn.py::TestStuartLandauLayer::test_forward_shapes`
- `tests/test_stuart_landau_nn.py::TestStuartLandauLayer::test_trajectory`
- `tests/test_stuart_landau_nn.py::TestStuartLandauLayer::test_mean_amplitude_is_finite_scalar`

## Why this matters for deployment

The Stuart-Landau contract supports workflows where phase-only models are not
sufficient and amplitude dynamics are required for meaningful control signals.

The shape and finiteness guarantees are production-critical because controllers
and monitors downstream often assume synchronized tuple dimensions for trajectory
alignment.

## Deployment context

- Amplitude-aware models are required when phase-only assumptions fail (for
  example, when transient growth/decay is operationally meaningful).
- The dual output contract (`phases`, `amplitudes`) is what allows a single policy
  graph to monitor convergence in both state dimensions.
- Trajectory retention is essential for evidence replay in release audits because it
  preserves temporal shape alignment from discovery through replay.

## Deployment interpretation

Amplitude-sensitive control paths are typically introduced only after phase-only
experiments establish a baseline. This contract protects that migration by defining
exactly which dimensional guarantees must hold when the extra state enters the stack.

The contract is operationally useful because it makes phase and amplitude states
first-class peers; policy graphs can reason about divergence, damping, and lock-in
without introducing a parallel interface for amplitude fields.

The trajectory contract also prevents silent replay mismatches: every training,
validation, or audit run can be aligned on both phase and amplitude timelines.

## Practical deployment implication
- This contract is the first gate for adding amplitude-aware control into an established phase-only pipeline.
- It is intended to keep training, replay, and policy optimization paths shape-compatible.
- Use this interface when controllers need explicit damping and growth signals in addition to phase state.

## Engineering handoff note

Amplitude-aware control is typically introduced as a second phase after baseline
phase-only validation. Keep this contract fixed while you compare three runs:

1. phase-only baseline,
2. phase-only + controller change,
3. phase+amplitude control candidates.

This three-run structure helps isolate whether observed gains come from control law
changes or from amplitude channels entering the policy path. The contract is the
common shape anchor across all three.
