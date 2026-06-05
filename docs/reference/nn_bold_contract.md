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

## Operational context

The BOLD contract is the bridge between phase dynamics and hemodynamic-style
measurement abstractions. It is used when users need differentiable signals that
are compatible with neuroimaging-style pipelines without adding a separate
preprocessing layer.

The finite-state and stride clamps are explicitly part of the contract because they
prevent unstable numerical states from silently contaminating gradient paths.

## Practical positioning

- Use this layer when SPO workflows need model-compatible BOLD-like trajectories
  for neuroimaging-style downstream pipelines.
- The stride clamp protects against silent no-op downsampling in mixed `dt` regimes,
  which preserves continuity between physics and hemodynamic abstractions.
- Maintaining shape and state positivity is especially important when chained into
  ML baselines that assume finite non-negative input features.
