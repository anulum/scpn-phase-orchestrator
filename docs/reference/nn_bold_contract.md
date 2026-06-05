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

## Practical value

This contract enables a narrow but critical bridge between dynamic phase models and
measurement-like outputs. In multi-team settings, one of the common integration gaps
is not the algorithm choice itself, but whether the signal contract is stable enough
for another pipeline to consume it.

The contract here is deliberately conservative: it prioritizes state validity and shape
contract stability over advanced parameterization. That is aligned with production
expectations where downstream teams need predictable interfaces for longitudinal
analysis, model calibration, and compliance reporting.

## Operational overview

In teams that combine scientific simulation and operations reporting, the BOLD
layer is often the first place where phase dynamics are converted into
consumable monitoring features. The practical value is not only predictive power
but confidence that downstream dashboards, regressors, and replay systems can
reuse outputs with the same assumptions every run.

This contract makes the conversion behavior explicit: finite state floors and
shape constraints are preserved across execution paths, which keeps downstream
analysis deterministic even when experimental scheduling changes `dt` and
`dt_bold`.

### For deployment planning

- Use BOLD outputs as the measurable interface between oscillatory physics and
  downstream tooling that expects continuous observables.
- Keep stride clamp behaviour documented with your benchmark notes so teams can
  explain any downsample-driven latency changes.
- Treat BOLD contract checks as part of release smoke checks when a new
  preprocessing or export path is introduced.
