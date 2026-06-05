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
| `initial_tolerance_units` | max observed spatial dispersion | `KinematicBounds.initialTolerance` |
| `lipschitz_step_gain_units` | explicit control, default `0` | `KinematicBounds.lipschitzStepGain` |
| `relative_velocity_step_bound_units` | explicit predictive slack, default `0` | `KinematicBounds.relativeVelocityStepBound` |
| `coupling_residual_step_bound_units` | moving-frame kinematic residual | `KinematicBounds.couplingResidualStepBound` |
| `merge_window_tolerance_units` | spatial merge tolerance | `KinematicBounds.mergeWindowTolerance` |
| `horizon_steps` | accepted PHA-C step count | `KinematicBounds.horizonSteps` |

The default is a replay certificate. The observed spatial dispersion already
includes the accepted moving-frame trajectory, while the residual term proves
the ballistic coordinate update was mechanically valid. Downstream predictive
lanes can provide non-zero `relative_velocity_step_bound_m` and non-zero
`lipschitz_step_gain_units` when they want the same Lean theorem to certify a
future horizon rather than the replay envelope.

For non-zero gain, the runtime manifest replays the Lean recurrence
`previous + gain * previous + drive`, records the terminal
`gronwall_budget_units`, records the signed
`gronwall_budget_margin_units`, and hashes the full budget trace as
`gronwall_budget_trace_sha256`. The legacy `linear_budget_units` field remains
as the zero-gain reference budget; the merge-window margin is now derived from
the Gronwall terminal budget.

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

- exact schema, evidence kind, claim boundary, Lean module, predicate, and
  theorem names;
- review-only flags: `execution_disabled=True` and `actuating=False`;
- finite positive metric and phase fixed-point scales;
- natural-number fields and the Lean equations for drive, linear zero-gain
  reference budget, Gronwall budget trace, terminal budget, and merge-window
  margin;
- phase tolerance margin consistency;
- lower-case SHA-256 fields; and
- canonical manifest hash replay.

If `proof_obligations_discharged` disagrees with the fixed-point certificate
math, verification fails closed.

::: scpn_phase_orchestrator.upde.pha_c_formal_obligation.PHACKinematicProofObligation

::: scpn_phase_orchestrator.upde.pha_c_formal_obligation.build_pha_c_kinematic_proof_obligation

::: scpn_phase_orchestrator.upde.pha_c_formal_obligation.pha_c_kinematic_proof_obligation_to_dict

::: scpn_phase_orchestrator.upde.pha_c_formal_obligation.verify_pha_c_kinematic_proof_obligation
