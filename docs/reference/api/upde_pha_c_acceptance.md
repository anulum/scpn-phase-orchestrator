# UPDE — PHA-C Acceptance Chain

`PHACAcceptanceRecord` is the end-to-end PHA-C review gate. It advances a
moving-frame trajectory through the same physics surfaces used by the PHA-C
lane, then converts the result into timeline evidence and a deterministic
acceptance hash.

The acceptance record is review-only. It does not write to actuators, change
coupling policy, schedule hardware, or mutate supervisor state.

## Use cases

Use the PHA-C acceptance chain when reviewers need evidence that the whole lane
was exercised rather than a single isolated module:

- MIF/FRC readiness checks where spatial coupling, Doppler correction,
  moving-frame propagation, merge-window locking, handoff hashing, and timeline
  hashing must be reviewed together;
- release gates that need one compact record proving the PHA-C chain can run
  from initial state to final lock evidence;
- benchmark snapshots that aggregate the per-module Rust, Go, Julia, Mojo, and
  Python parity gates while labelling local timings as non-isolated regression
  evidence;
- Studio or replay panels that need an acceptance hash without exposing raw
  trajectory arrays.

## Contract

Inputs are the initial state plus schedules:

```text
phases_t0[i]              = initial oscillator phase
positions_t0[i]           = initial axial position
omega_schedule[t, i]      = natural frequency at schedule row t
velocity_schedule[t, i]   = axial velocity at schedule row t
knm[i, j]                 = zero-diagonal base coupling matrix
```

For each schedule row, the builder:

1. resolves distance-modulated coupling through `SpatialCouplingModulator`;
2. computes the graph-weighted Doppler correction;
3. advances one moving-frame UPDE step;
4. records phase and position trajectory rows;
5. computes the signed moving-frame kinematic residual
   `max |z[t+1] - (z[t] + v[t] * dt)|`;
6. builds a `PHACTimelineRecord` over the trajectory;
7. can project the verified acceptance payload into a Lean
   `PHACKinematicProofObligation` for `SPOFormal.Kinematic`; and
8. hashes schedules, trajectories, spatial couplings, Doppler trace, timeline,
   proof-obligation references, and final acceptance payload.

The record carries:

- first-lock index/time and final-lock state;
- lock sample, lock-loss, reset, and maximum consecutive-lock counts;
- maximum absolute Doppler correction and spatial coupling;
- maximum moving-frame kinematic residual, maximum absolute velocity, and
  maximum per-oscillator axial path length;
- final-position, maximum-velocity, and path-length equation replay under
  `PHA_C_ACCEPTANCE_KINEMATIC_SUMMARY_REPLAY_TOLERANCE`;
- fixed-point Lean proof-obligation compatibility through
  `build_pha_c_kinematic_proof_obligation(...)`;
- maximum phase/spatial dispersion, minimum signed phase/spatial margins,
  minimum Kuramoto order parameter, and maximum distance to reference;
- replay validation that each signed margin equals its tolerance minus the
  corresponding maximum dispersion under
  `PHA_C_ACCEPTANCE_MARGIN_REPLAY_TOLERANCE`;
- resolved tolerance profile provenance;
- `execution_disabled=True`, `actuating=False`, and the fixed claim boundary
  `pha_c_end_to_end_acceptance_review_only`.

## Minimal example

```python
import numpy as np
from scpn_phase_orchestrator.upde.pha_c_acceptance import (
    build_pha_c_acceptance_record,
    verify_pha_c_acceptance_record,
)
from scpn_phase_orchestrator.upde.pha_c_formal_obligation import (
    build_pha_c_kinematic_proof_obligation,
    verify_pha_c_kinematic_proof_obligation,
)

n = 5
phases = np.linspace(-0.002, 0.002, n)
positions = np.linspace(-0.0006, 0.0006, n)
omega = np.zeros((4, n))
knm = np.full((n, n), 0.04)
np.fill_diagonal(knm, 0.0)
velocities = np.vstack([np.linspace(0.10, 0.12, n) for _ in range(4)])

record = build_pha_c_acceptance_record(
    phases,
    positions,
    omega,
    knm,
    velocities,
    dt=1.0e-3,
    required_consecutive_samples=3,
    tolerance_profile="baseline_1x",
    backend="python",
)

assert record.first_lock_index == 2
assert record.final_lock_achieved
assert record.kinematic_residual_max_m <= 1.0e-12
assert record.execution_disabled
assert not record.actuating
acceptance_payload = record.to_dict()
verify_pha_c_acceptance_record(record)
obligation = build_pha_c_kinematic_proof_obligation(record)
assert obligation.proof_obligations_discharged
verify_pha_c_kinematic_proof_obligation(obligation)
```

Use `verify_pha_c_acceptance_record(...)` when replaying a stored acceptance
record. It rechecks sample/step consistency, first-lock semantics, review-only
flags, signed margin equations, the moving-frame kinematic residual bound,
kinematic equation replay flags and tolerance, SHA-256 fields, the timeline
digest reference, and the canonical acceptance hash without requiring the
original schedules or trajectories. The signed margin
replay rejects records whose positive-looking phase or spatial margin no longer
matches `tolerance - maximum_dispersion`.
Use `verify_pha_c_kinematic_proof_obligation(...)` when release review also
needs the accepted runtime envelope bound to the Lean
`budget_certificate_discharges_budget` theorem.

## Relationship to other PHA-C records

| Surface | Scope | Use when |
|---------|-------|----------|
| `PHACHandoffRecord` | one phase/position sample | a downstream lane needs one reviewed event-state atom |
| `PHACTimelineRecord` | complete phase/position trajectory | a downstream lane needs lock/loss/reset history |
| `PHACAcceptanceRecord` | full PHA-C physics chain plus timeline | a release, MIF/FRC, or benchmark gate needs end-to-end evidence |
| `PHACKinematicProofObligation` | fixed-point Lean assumptions derived from verified acceptance | a formal gate or downstream lane needs a signed theorem-specific obligation |

## Polyglot parity and benchmark snapshot

The benchmark gate records Rust, Mojo, Julia, Go, and Python source-contract
slots for the acceptance builder, then aggregates the existing PHA-C subgates:
spatial modulation, time-varying omega, Doppler, moving-frame, merge window,
handoff, and timeline. Acceptance rows also publish the minimum signed margins
copied from the timeline plus the moving-frame kinematic residual, so release
evidence exposes both the distance to the reviewed merge envelope and the
mechanical validity of the axial schedule. The benchmark row now also records
`phase_margin_equation_validated`, `spatial_margin_equation_validated`,
`signed_margin_equations_validated`, and `margin_replay_tolerance` for every
backend slot. It also records `final_position_equation_validated`,
`max_abs_velocity_equation_validated`, `path_length_equation_validated`,
`kinematic_equations_validated`, and `kinematic_summary_replay_tolerance` for
every backend slot so the aggregate acceptance record preserves the
moving-frame summary equations rather than only scalar residuals. The same
benchmark now builds and verifies a Lean kinematic
proof-obligation manifest for every backend row. The
acceptance gate fails unless the manifest names `SPOFormal.Kinematic`, targets
`budget_certificate_discharges_budget`, replays its canonical hash, replays the
finite-horizon Gronwall budget trace, and discharges the fixed-point
merge-window and phase-margin checks. The same manifest publishes sampled
continuous-rate fields for `dt`, horizon time, velocity-rate bounds, and
residual-rate bounds so Rust, Mojo, Julia, Go, and Python source-contract rows
carry the same time-normalised formal assumptions. The rows now also publish
the `SPOFormal.Continuous` horizon theorem, continuous drive-rate sum,
continuous horizon-drive replay, continuous budget, continuous margin, and a
Boolean continuous-envelope discharge flag.

```bash
uv run python benchmarks/pha_c_acceptance_benchmark.py \
  --parity-gate \
  --calls 1 \
  --output benchmarks/results/pha_c_acceptance.json
```

Committed benchmark JSON is local regression evidence only. It is not a
production timing claim unless rerun under the benchmark-isolation protocol.

## Failure boundaries

The acceptance builder fails closed on:

- empty, non-finite, complex, object-dtype, or boolean state vectors;
- mismatched phase and position shapes;
- malformed omega or velocity schedules;
- non-square, non-finite, or non-zero-diagonal coupling matrices;
- invalid moving-frame backend names;
- non-positive `dt`, tolerances, or integration controls;
- unknown tolerance profile names.

::: scpn_phase_orchestrator.upde.pha_c_acceptance.PHACAcceptanceRecord

::: scpn_phase_orchestrator.upde.pha_c_acceptance.build_pha_c_acceptance_record

::: scpn_phase_orchestrator.upde.pha_c_acceptance.pha_c_acceptance_record_to_dict

::: scpn_phase_orchestrator.upde.pha_c_acceptance.verify_pha_c_acceptance_record
