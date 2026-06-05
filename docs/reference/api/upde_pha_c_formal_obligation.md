# UPDE — PHA-C Lean Kinematic Proof Obligation

`PHACKinematicProofObligation` is the review-only bridge between a verified
runtime `PHACAcceptanceRecord` and the Lean module `SPOFormal.Kinematic`. It
converts the accepted PHA-C trajectory envelope into fixed-point natural-number
fields that match `KinematicBounds`, then signs the manifest with a
deterministic SHA-256 hash.

The obligation does not execute Lean, write to hardware, mutate a supervisor,
or change a coupling policy. It records the exact theorem and Boolean
certificate predicate that a reviewer or CI proof gate must use:

- Lean module: `SPOFormal.Kinematic`
- Predicate: `KinematicBounds.budgetCertificate`
- Theorem: `budget_certificate_discharges_budget`
- Continuous-envelope module: `SPOFormal.Continuous`
- Continuous predicate: `ContinuousEnvelopeBounds.budgetCertificate`
- Continuous theorem: `continuous_envelope_certificate_discharges_horizon`
- Phase-budget module: `SPOFormal.Kinematic`
- Phase predicate: `PhaseBudgetBounds.budgetCertificate`
- Phase theorem: `phase_budget_certificate_discharges_phase_lock`

## Use cases

Use this manifest when the PHA-C acceptance chain needs a formal review bridge:

- MIF/FRC handoff packages that need fixed-point assumptions to specialise
  into downstream constants;
- release evidence that must prove runtime PHA-C acceptance has a named Lean
  theorem target, not only empirical trajectory hashes;
- benchmark gates that must fail when the accepted trajectory no longer fits
  the finite-horizon Gronwall merge-window certificate;
- Studio or audit panels that need a compact proof-obligation hash without
  exposing raw trajectory arrays.

## Fixed-point mapping

The builder first verifies the source `PHACAcceptanceRecord`, then projects the
runtime envelope into integer units:

| Manifest field | Runtime source | Lean role |
|----------------|----------------|-----------|
| `time_step_s` | accepted integration `dt` | sampled-rate time step |
| `fixed_point_time_scale_s` | manifest time scale, default `1e-6` | sampled-rate fixed-point clock |
| `time_scale_units_per_second` | `ceil(1 / fixed_point_time_scale_s)` | sampled-rate denominator |
| `time_step_units` | `ceil(time_step_s / fixed_point_time_scale_s)` | sampled-rate numerator |
| `horizon_time_units` | `horizon_steps * time_step_units` | reviewed horizon duration |
| `initial_tolerance_units` | max observed spatial dispersion | `KinematicBounds.initialTolerance` |
| `lipschitz_step_gain_units` | explicit control, default `0` | `KinematicBounds.lipschitzStepGain` |
| `relative_velocity_rate_bound_units_per_second` | predictive slack divided by `dt` | `SampledRateKinematicBounds.relativeVelocityRateBound` |
| `relative_velocity_step_bound_units` | explicit predictive slack, default `0` | `KinematicBounds.relativeVelocityStepBound` |
| `configured_coupling_residual_step_bound_units` | explicit predictive residual slack, default `0` | residual-bound provenance |
| `coupling_residual_rate_bound_units_per_second` | max of configured residual slack and observed moving-frame residual, divided by `dt` | `SampledRateKinematicBounds.couplingResidualRateBound` |
| `coupling_residual_step_bound_units` | max of configured residual slack and observed moving-frame residual | `KinematicBounds.couplingResidualStepBound` |
| `continuous_drive_rate_bound_units_per_second` | velocity-rate plus residual-rate bound | `ContinuousEnvelopeBounds.driveRateBound` |
| `continuous_horizon_drive_bound_units` | sampled continuous drive over `horizon_time_units` | `ContinuousEnvelopeBounds.sampledDriveBoundAt` |
| `continuous_linear_budget_units` | initial dispersion plus sampled horizon drive | `ContinuousEnvelopeBounds.budgetAt` |
| `continuous_margin_units` | merge tolerance minus continuous budget | `ContinuousEnvelopeBounds.budgetCertificate` |
| `merge_window_tolerance_units` | spatial merge tolerance | `KinematicBounds.mergeWindowTolerance` |
| `horizon_steps` | accepted PHA-C step count | `KinematicBounds.horizonSteps` |
| `phase_tolerance_units` | accepted phase tolerance | phase-lock certificate input |
| `max_phase_dispersion_units` | max observed phase dispersion | replayed phase evidence |
| `configured_phase_drift_bound_units` | explicit predictive phase-drift slack, default `0` | phase-bound provenance |
| `phase_budget_units` | observed dispersion plus configured phase drift | phase-lock budget |
| `phase_margin_units` | phase tolerance minus phase budget | phase-lock certificate margin |

The default is a replay certificate. The observed spatial dispersion already
includes the accepted moving-frame trajectory, while the residual term proves
the ballistic coordinate update was mechanically valid. Downstream predictive
lanes can provide non-zero `relative_velocity_step_bound_m` and non-zero
`coupling_residual_step_bound_m` values when they need a reviewed residual
envelope beyond the observed replay residual. The verifier requires the
configured residual units to fit inside the sampled residual bound before the
same Lean theorem can certify a future horizon rather than the replay envelope.
Non-zero `lipschitz_step_gain_units` can then be added for finite-horizon
growth.

The phase side is also explicit. Downstream lanes can provide
`phase_drift_bound_rad` when the reviewed handoff must budget future phase
drift in addition to the accepted replay dispersion. The manifest records that
slack as `configured_phase_drift_bound_units`, records
`phase_budget_units = max_phase_dispersion_units +
configured_phase_drift_bound_units`, and derives `phase_margin_units` from the
budget rather than from replay dispersion alone. The manifest names the Lean
`PhaseBudgetBounds.budgetCertificate` predicate and
`phase_budget_certificate_discharges_phase_lock` theorem so the phase budget is
reviewed by a formal fixed-point mirror instead of only by Python arithmetic.

For non-zero gain, the runtime manifest replays the Lean recurrence
`previous + gain * previous + drive`, records the terminal
`gronwall_budget_units`, records the signed
`gronwall_budget_margin_units`, and hashes the full budget trace as
`gronwall_budget_trace_sha256`. The legacy `linear_budget_units` field remains
as the zero-gain reference budget; the merge-window margin is now derived from
the Gronwall terminal budget.

For continuous-rate handoffs, the manifest also records a sampled-rate mirror:
per-second relative-velocity and residual bounds are sampled through
`time_step_units / time_scale_units_per_second` before they enter the discrete
Lean budget. The Lean side names this bridge `SampledRateKinematicBounds` and
proves that a sampled-rate certificate discharges the same finite-horizon
merge-window budget after conversion to `KinematicBounds`.

The manifest also records the continuous-envelope theorem target. That layer
samples the same per-second rates over `horizon_time_units`, records the
continuous horizon drive, and requires the signed continuous margin to be
non-negative before `proof_obligations_discharged` can be true. The Lean side
names this boundary `ContinuousEnvelopeBounds`; it is a dependency-light
fixed-point continuous envelope, not a Mathlib real-analysis proof.

## Minimal example

```python
from scpn_phase_orchestrator.upde.pha_c_acceptance import (
    build_pha_c_acceptance_record,
)
from scpn_phase_orchestrator.upde.pha_c_formal_obligation import (
    build_pha_c_kinematic_proof_obligation,
    verify_pha_c_kinematic_proof_obligation,
)

record = build_pha_c_acceptance_record(...)
obligation = build_pha_c_kinematic_proof_obligation(record)

assert obligation.lean_theorem == "budget_certificate_discharges_budget"
assert obligation.proof_obligations_discharged
verify_pha_c_kinematic_proof_obligation(obligation)
```

## Verification boundary

`verify_pha_c_kinematic_proof_obligation(...)` checks:

- exact schema, evidence kind, claim boundary, Lean kinematic, continuous, and
  phase-budget module, predicate, and theorem names;
- review-only flags: `execution_disabled=True` and `actuating=False`;
- finite positive metric and phase fixed-point scales;
- finite positive time scale and accepted time-step sampling fields;
- natural-number fields and the Lean equations for drive, linear zero-gain
  reference budget, Gronwall budget trace, terminal budget, and merge-window
  margin;
- continuous-envelope theorem metadata, drive-rate sum, horizon-drive replay,
  continuous budget, and continuous margin;
- configured residual-bound provenance and its fit inside the sampled residual
  drive bound;
- configured phase-drift provenance, phase-budget replay, and phase tolerance
  margin consistency;
- lower-case SHA-256 fields; and
- canonical manifest hash replay.

If `proof_obligations_discharged` disagrees with the fixed-point certificate
math, verification fails closed.

::: scpn_phase_orchestrator.upde.pha_c_formal_obligation.PHACKinematicProofObligation

::: scpn_phase_orchestrator.upde.pha_c_formal_obligation.build_pha_c_kinematic_proof_obligation

::: scpn_phase_orchestrator.upde.pha_c_formal_obligation.pha_c_kinematic_proof_obligation_to_dict

::: scpn_phase_orchestrator.upde.pha_c_formal_obligation.verify_pha_c_kinematic_proof_obligation
