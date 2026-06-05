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
