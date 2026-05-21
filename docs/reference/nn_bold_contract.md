# `scpn_phase_orchestrator.nn.bold` contract

Purpose:
- Provide differentiable Balloon-Windkessel hemodynamic conversion from neural
  activity tensors to BOLD-like signals.

Public contract:
1. `balloon_windkessel_step` preserves vector shapes and enforces positive flow
   and volume state floors.
2. `bold_signal` returns tensor outputs with the same leading shape as `(v, q)`.
3. `bold_from_neural` returns a finite `(T_bold, N)` tensor and clamps
   subsampling with `max(1, int(dt_bold / dt))` so `dt_bold < dt` does not
   produce an invalid stride.

Verification:
- `tests/test_bold.py::TestBalloonWindkesselStep::test_output_shapes`
- `tests/test_bold.py::TestBOLDFromNeural::test_output_shape`
- `tests/test_bold.py::TestBOLDFromNeural::test_dt_bold_smaller_than_dt_uses_no_downsample`
