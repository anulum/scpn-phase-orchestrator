# Reactor Semantics

The U0 reactor-semantic API gives fusion, magneto-inertial-fusion, and control
systems one explicit vocabulary for saying what a signal means. It is a
description and review surface. It cannot admit a control action or actuate a
plant.

## Ownership boundary

| Concern | Owner | U0 rule |
|---------|-------|---------|
| Plant and solver truth | `SCPN-FUSION-CORE`, `SCPN-MIF-CORE` | SPO consumes declared values, units, frames, clocks, calibration, validity, and provenance; it does not invent them. |
| Semantic interpretation | `SCPN-PHASE-ORCHESTRATOR` | SPO assigns typed carrier meaning, checks comparability, and emits review evidence. |
| Admission and action | `SCPN-CONTROL` | CONTROL alone decides whether evidence is admissible for an action. U0 records remain `review_only`. |

`ReactorContext` is faceted instead of tokamak-shaped. Configuration/topology,
confinement, drivers, cadence, reaction/fuel, conversion boundary, facility,
coordinate frame, evidence class, event identity, and operating point are
independent fields. Pulsed, repetitive-target, and single-experiment contexts
require an opaque plant- or integrator-supplied `event_id`; U0 never invents a
`shot_id`.
The built-in registry includes 32 concrete configurations and an immutable,
namespaced extension path.

The deterministic reference portfolio exercises nine non-exclusive design
slices:

- A1 axisymmetric tokamak,
- N1 stellarator,
- C1 compact toroid/FRC,
- O1 open-field mirror/cusp,
- P1 Z-pinch/dense-focus,
- I1 inertial fusion/IFE,
- H1 MagLIF/MIF,
- E1 IEC/beam-target, and
- X1 fusion-fission/direct-conversion contexts.

These are scaffold fixtures, not experimental validation or reactor-readiness
claims.

## Five public contracts

- `ReactorContext` identifies the physical and facility context without a
  privileged reactor family.
- `ObservableDescriptor` binds a value to units, frame, spatial support,
  diagnostic/channel, clock, calibration, uncertainty, quality, validity, and
  provenance.
- `PhaseSemanticRecord` states the mathematical carrier and the declared
  origin, orientation, wrap convention, reference signal, extractor,
  observation operator, confidence, and observability threshold.
- `PhaseRelation` exists only after frame, clock, harmonic, context, and
  validity compatibility checks pass or explicit transforms are supplied.
- `RegimeEstimate` preserves independent regime axes, hysteresis, dwell time,
  threshold provenance, transition reason, and fail-closed validity while
  fixing its authority to review only.

## Eight carriers are not interchangeable

U0 preserves `cyclic_phase`, `complex_mode`, `field_phase`, `event_cycle`,
`bounded_feature`, `categorical_state`, `protocol_phase`, and
`numerical_phase`. A bounded feature, categorical state, protocol stage, or
raw event count cannot acquire an angle merely by normalization. Event-cycle
phase requires a declared reference signal. Numerical phase cannot claim
observed or experimental evidence.

Legacy `FusionCoreBridge.observables_to_phases()` normalizes scalar features
and event-count parity into oscillator coordinates. That method predates U0
and is not a valid U0 semantic producer. A consumer must cross the U0 contract
surface and satisfy the carrier-specific checks before comparing or exporting
reactor phase meaning.

## Fail-closed behavior

Every phase-bearing record declares its own observability threshold; U0 does
not impose a universal physics threshold. Below that threshold, validity must
be `unobservable` and `phase_rad` must be absent. `unknown`, `stale`,
`out_of_distribution`, `unobservable`, and `invalid` records cannot be compared
as usable phases. Non-usable regime evidence can only publish the top-level
state `unknown`.

Clock records distinguish plant-monotonic, simulation-monotonic,
shot-relative, facility-synchronised, wall-clock, model-tick, and unknown time
bases. They carry an explicit epoch and preserve a 0–999 picosecond offset in
addition to the integer nanosecond timestamp. Mixed time bases or epochs need
an explicit clock transform. `validate_observable_sequence` additionally
rejects mixed streams and non-monotonic or unusable samples.

Serialization uses a strict typed envelope and canonical sorted JSON. Unknown
fields, missing fields, duplicate JSON keys, unknown contract kinds, registry
drift, unsupported schema versions, and envelope/payload version disagreement
are refused. U0 currently accepts exactly schema `1.0.0`; compatibility is not
guessed.

The portable exchange shape is published as
[`reactor_semantics_u0.schema.json`](../../specs/reactor_semantics_u0.schema.json).
JSON Schema validates transport structure; the Python runtime additionally
enforces registry identity and the cross-field physics/epistemic invariants.

```python
from scpn_phase_orchestrator import (
    ObservableDescriptor,
    PhaseRelation,
    PhaseSemanticRecord,
    ReactorContext,
    RegimeEstimate,
)
from scpn_phase_orchestrator.reactor_semantics import (
    build_reactor_reference_portfolio,
    canonical_json,
)

reference = build_reactor_reference_portfolio()
tokamak_context: ReactorContext = reference[0].context
payload = canonical_json(tokamak_context)
```

The root package exports only the five stable contracts. Registry, evidence,
serialization, relation-building, enums, and portfolio helpers live under
`scpn_phase_orchestrator.reactor_semantics` so their use remains explicit.

::: scpn_phase_orchestrator.reactor_semantics
    options:
      show_root_heading: true
      show_source: false
      members_order: source
