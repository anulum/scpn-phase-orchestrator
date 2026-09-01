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

The separate `DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY` binds every one of
those 32 configurations to its device-family project without claiming that a
project assignment is already a semantic producer. Its versioned canonical
record is sealed to the reactor-registry digest and the cross-project
assignment-map SHA-256. Only `conventional_tokamak` through the exercised
FUSION/TORAX adapter and `frc_compression_mif` through the exercised MIF
adapter currently have `verified_review_adapter` ingress. The other thirty
records are explicitly `not_declared`: they advertise no producer, source
schema, adapter API, handoff schema, or semantic profile.

The profile registry is a descriptive SPO binding for producer and consumer
contract discovery. It does not replace device-owned `reactor-domain.json`,
declare capability evidence maturity, or determine Studio federation state.
It never supplies an implicit generic adapter. CONTROL adapter contracts remain
separately versioned fields and are currently absent. The public
`ReactorResearchControlIntent` research-hypothesis schema exists, but no profile
advertises it because no producer-to-CONTROL path has exercised it. All
profiles are `review_only`, `actionable=false`, and preserve independent
machine protection as the final veto.

`DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY` adds the machine-readable
candidate layer. It covers every built-in configuration with explicit
`direct_cyclic`, `derived_cyclic`, `event_relative`, `noncyclic_feature`, and
`numerical_only` routes plus a fail-closed `unobservable` result. Each
candidate fixes its admissible carrier set, reference/repetition/operator
requirements, minimum evidence fields, and the disposition when evidence is
missing. These records always carry `evidence_claimed=false`: applicability is
a research and schema requirement, never proof that the diagnostic or signal
exists.

The catalogue keeps equilibrium profiles, trajectories, yields, and blanket
response noncyclic; requires validated modal operators for MHD, pinch,
implosion, liner, and electrostatic oscillation phase; requires facility or
event references for drive and shot timing; and confines synthetic oscillator
coordinates to `numerical_phase`. It therefore provides a deterministic gap
map without allowing architecture-only projects to advertise observations.

## Device diagnostic-plan design review

`device_diagnostic_plan_review_from_producer_bytes()` is the reactor-family-
neutral intake for a device owner's portable diagnostic and clock plan. It
accepts exact manifest, envelope, and plan bytes plus two identities that the
current producer envelope cannot supply uniquely: a 40-character source Git
revision and the SHA-256 of the exact installed wheel artefact. The producer's
package revision remains a separate field. SPO never imports or executes the
device package while interpreting these bytes.

The v1 intake accepts only canonical `scpn.reactor-domain.v1` manifest bytes
and canonical `scpn.reactor-diagnostic-plan-envelope.v1` version `1.1.0`
bytes. It recomputes the raw manifest and plan digests, rejects recursive key
drift, and resolves project ownership, configurations, candidate IDs,
observability classes, carriers, evidence slots, and registry pins against
SPO's installed frozen registries. Planned and explicitly deferred candidates
must cover the applicable catalogue exactly.

Each `DeviceDiagnosticSignalReview` preserves the candidate, its
`derived_cyclic`, `event_relative`, `noncyclic_feature`, or `numerical_only`
class, the admissible carrier, clock identity, and exact required evidence-slot
names. It does not claim that the slots contain observations. A bounded feature
therefore remains bounded, an event cycle remains event-relative, and a model
oscillator remains numerical phase.

Producer clock vocabulary is not coerced into SPO vocabulary.
`simulation` is only a `simulation_monotonic` candidate,
`shot_event_epoch` is only shot-relative compatible, and
`facility_monotonic` remains explicitly unmapped. A declared offset bound is
not facility synchronisation, wall time, or correlation evidence.

`DeviceDiagnosticPlanReview` embeds the exact three source documents and seals
their digests, source commit, artefact digest, registry identities, typed signal
rows, clock rows, candidate coverage, and reference frames. Its fixed verdict
is design-declaration-only: no measurement, physical observation, facility
binding, classification, semantic ingress, control intent, action, or actuation
is created. The portable envelope is defined by
[`device_diagnostic_plan_review.schema.json`](../../specs/device_diagnostic_plan_review.schema.json).

Exact producer fixtures currently exercise this intake for tokamak, dense-
plasma-focus, MagLIF, and mechanical-or-liquid-liner MIF device plans. The
dense-plasma-focus review keeps a shot-relative discharge-current `event_cycle`,
a facility-clocked neck-mode `complex_mode`, and a simulation-clocked
`numerical_phase` distinct. Both liner-MIF reviews keep compression-trajectory
`bounded_feature`, liner-arrival `event_cycle`, resolved-asymmetry
`complex_mode`, and model `numerical_phase` distinct without equating their
device identities or timescales. Facility clocks remain unmapped, event timing
uncertainties remain explicit, and all three device-plan-only semantic-ingress
profiles remain `not_declared`. Passing a design review does not add source
evidence to the reactor configuration evidence matrix.

## FAIR-MAST magnetic physical-source review

`mast_magnetic_source_review_from_producer_bytes()` accepts the exact canonical
complete-magnetic archive envelope and diagnostic-qualification bytes emitted
by SCPN-FUSION-CORE. The intake imports no sibling package. Its caller must pin
the full producer Git revision and the SHA-256 of the exact producer wheel;
the review separately preserves both source documents, their outer and payload
digests, the FAIR-MAST ingestion revision, and the source tree state.

The v1 contract is deliberately specific to the complete shot-27707 FAIR-MAST
inventory: 72 arrays, 11 measurement families, 132 qualified channel records,
and four reproduced archive grids. Every array, clock, measurement, empirical-
quality row, channel-quality row, identifier-only geometry row, completeness
claim, event limitation, and archive-to-qualification binding is replayed from
the embedded bytes. SCPN-FUSION-CORE is the physical-evidence producer;
SCPN-TOKAMAK-CORE remains the spherical-tokamak device owner. This review does
not alter the semantic-profile registry's `not_declared` producer state.

The producer qualification records applied transforms, but it supplies no
calibration lineage or transfer functions. It also supplies no physical
geometry join, provider quality flags, uncertainty, instrument-clock relation,
or resolved facility event identity. The four clock rows are therefore only
shot-relative candidates for derived archive grids. They are not instrument
clocks and are not mapped to facility, plant-monotonic, simulation-monotonic,
or wall time.

`MastMagneticSourceReview` records authentic physical-source custody while
fixing `observation_admitted=false`, `qualified_phase_evidence=false`,
`phase_inference_performed=false`, `semantic_ingress_declared=false`, and
`classification_performed=false`. It cannot create CONTROL intent, execution
permission, direct actuation, or actionability; independent machine protection
retains the final veto. The canonical review envelope is defined by
[`mast_magnetic_source_review.schema.json`](../../specs/mast_magnetic_source_review.schema.json).

`DEFAULT_REACTOR_REGIME_MODE_ONTOLOGY` closes the remaining open-text meaning
above U0 without changing the U0 `1.0.0` wire format. It defines eight
independent axes: plant readiness, diagnostic observability, confinement or
assembly, stability or symmetry, driver synchronization, power or burn,
exhaust or boundary, and evidence maturity. Every axis has a closed label
vocabulary and explicit universal or context-dependent applicability.

`ReactorRegimeAxisAssignment` keeps three cases distinct. An `applicable` axis
requires a defined non-unknown label and classification evidence;
`not_applicable` requires a context basis and forbids a physics label; and
`unknown` reports that applicability or classification is unresolved. The
last two project into U0 as literal `not_applicable` and `unknown` labels with
zero confidence. Neither can silently become `nominal`.

The same ontology names six physical mode families for closed-field MHD,
open-field interchange, self-magnetic pinch instability, inertial asymmetry,
magneto-inertial asymmetry, and IEC bunching. A physical
`ReactorModeBinding` is valid only with an exact reactor configuration,
compatible observability candidate and carrier, physical harmonic basis,
operator, frame, reference, orientation, origin, wrap convention,
observability threshold, validity, quality, provenance, and admissible
evidence class. The separate all-configuration synthetic-oscillator definition
admits only `numerical_phase` with simulation evidence and no physical
harmonic. It is an explicit fallback, not a claim of equivalence to a plasma
mode.

The ontology record is sealed to the exact reactor and observability registry
versions and SHA-256 digests. It remains `review_only` and
`actionable=false`; it validates identity and refusal rules but performs no
classification, extraction, CONTROL admission, machine-protection decision,
or actuation.

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

## Coupled-transport review handoff

`coupled_transport_handoff_from_fusion_bytes()` is the public producer adapter.
It accepts only canonical
`scpn-fusion-core.torax-runtime-review-envelope.v1` bytes and independently
checks the byte digest, nested payload digest, exact source and
model-intersection schemas, Git revision, event identity, provenance digests,
simulation clock, completion, solver numerics, U0 registry identity, and the
declared non-empirical calibration. It has no runtime dependency on FUSION,
TORAX, NumPy, Julia, or an accelerator.

The accepted evidence set is closed: four radial profiles (electron density,
electron temperature, ion temperature, and poloidal flux), five global source
totals (driven current, electron heat, ion-electron exchange, ion heat, and
particles), and three state budgets (particle inventory, poloidal-flux L2, and
thermal energy). The normalised `rho` axis is spatial support, not a thirteenth
observable. Profile rows and scalar series must match every clock sample before
the adapter selects the final sample. Each quantity has one exact unit and a
category-matched `absolute_rms_difference` plus `relative_l2`; the absolute RMS
is carried in U0's standard-deviation field with confidence fixed to zero and
provenance stating that it is numerical, not statistical or empirical.

The frozen TORAX deck contains deuterium main ions and neon impurity but models
no fusion reaction, burn, or fusion power. `deuterium_deuterium` therefore
identifies only the evidence-supported fuel class. The producer must retain the
explicit `deuterium_only_input_no_fusion_power_or_burn_model` qualifier; SPO
refuses an unqualified reaction label.

`ReactorSemanticHandoff` carries one exact canonical FUSION producer envelope
alongside its U0 interpretation. The source bytes and the complete handoff
payload have independent SHA-256 seals, so a downstream reviewer can verify the
producer-to-semantics chain without trusting a live Python object graph.

The coupled-transport profile is deliberately narrow. Every observable retains
FUSION provenance and shares one simulation-monotonic clock domain and epoch.
SPO emits exactly one `bounded_feature` record per observable, with zero phase
confidence and observability, `unobservable` phase validity, and explicit
unknown phase quality. This expected phase-extraction state does not invalidate
an otherwise usable nonphase observable or prevent review admission. Amplitude,
angle, frequency, bandwidth, mode,
harmonic, origin, orientation, wrap, reference signal, observation operator,
and phase relations are forbidden. The regime remains `unknown` with zero
confidence. The handoff fixes `authority="review_only"` and
`actionable=false`; it has no path to `ControlAction`.

Cross-project consumers pass the exact serialized bytes to
`handoff_from_bytes()`. It admits only the unique canonical UTF-8 encoding;
`handoff_from_json()` remains the Python string surface for local composition.
Importing `scpn_phase_orchestrator.reactor_semantics` does not initialise UPDE,
supervisor, native accelerator, or Julia runtimes. The decoder refuses
duplicate keys, unknown or extra fields, digest drift, registry or U0 version
drift, mixed clocks, missing event identity, FUSION ownership loss, phase
relabeling, relations, regime inference, and authority escalation. The portable
shape is published as
[`reactor_semantic_handoff.schema.json`](../../specs/reactor_semantic_handoff.schema.json).

Freshness is a consumer-owned admission policy. A consumer compares sample and
calibration timestamps only against an explicitly supplied reference clock in
the same domain and epoch; it must not compare simulation-monotonic evidence to
wall time implicitly.

## MIF merge-compression review handoff

`mif_merge_compression_handoff_from_mif_bytes()` is the dedicated adapter for
`scpn-mif-core.merge-compression-observation.v1`. It accepts exact canonical
SCPN-MIF-CORE bytes without importing or executing that sibling package. The
adapter independently verifies source and payload digests, source revision,
event identity, review-only authority, reactor vocabulary, complete simulation
clock, model-evidence validity, kinematic vector shape, derived geometry,
merge predicates, and trigger prerequisites.

The configuration is the registry's `frc_compression_mif`, whose family is
`magneto_inertial` and whose topology is a compressed field-reversed
configuration. The shorter `frc` alias is intentionally rejected because it
resolves to an uncompressed magnetic-closed FRC.

Each serialized oscillator angle becomes one `numerical_phase` record. Its
meaning is limited to the MIF model state: `[0,2pi)` wrap, positive model
evolution, event-start origin and reference, identity model-state observation
operator, simulation-monotonic clock, and simulation evidence. Position,
velocity, phase-lock error, Kuramoto order parameter, tolerances, streaks,
safety margin, and integrator error remain `bounded_feature`. Lock, trigger,
and gate values remain `categorical_state`. Their phase observability is zero;
none acquires an angle. V1 emits no phase relation and leaves the reactor regime
`unknown` because MIF supplies no versioned regime classifier.

`MIFMergeCompressionHandoff` embeds the exact source JSON and its SHA-256 beside
the complete U0 graph. `mif_merge_compression_handoff_to_bytes()` seals that
graph in a second canonical envelope;
`mif_merge_compression_handoff_from_bytes()` verifies the full chain. Both
source and handoff are fixed to `review_only` and `actionable=false`, and the
module has no control-action dependency. The portable shape is published as
[`mif_merge_compression_handoff.schema.json`](../../specs/mif_merge_compression_handoff.schema.json).

## Portable reactor regime assessment

`ReactorRegimeAssessment` is the digest-sealed portable identity for one
complete eight-axis regime vector. It binds the exact source handoff, event,
reactor context, producer and source revisions, clock and validity window, and
the exact reactor, semantic-profile, observability, and regime-ontology
registries. Axis rows are always serialized in lexicographic `axis_id` order.

Each `ReactorRegimeAxisAssessment` separates ontology-derived static
applicability from the result disposition: `classified`, `unknown`, or
`not_applicable`. A classified row requires a closed ontology label, positive
confidence and observability, probability uncertainty with a named basis,
usable validity and quality, provenance, evidence IDs, and a typed binding for
every evidence role required by that axis definition. Classifier identity is
therefore required for confinement/assembly, while owner declaration,
diagnostic inventory, clock/reference, reaction-model, boundary, or maturity
roles remain distinct rather than being mislabeled as classifiers.

`unknown` is the correct result when a statically applicable axis lacks enough
qualified evidence. It has no physics label, zero confidence, unit uncertainty
probability, and an explicit reason; its evidence list may be empty.
`not_applicable` is accepted only when the pinned ontology computes that result
for the exact configuration. It cannot be used to hide missing evidence.

`classification_performed=false` states that this codec validates supplied
results but never runs a classifier. The complete envelope and every axis are
permanently `review_only` and non-actionable. No collapsed nominal/critical
verdict, control target, actuator, CODAC, or interlock field exists. Consumers
must independently retrieve the referenced artifacts; a syntactically valid
digest is not proof of their contents. The current MIF handoff still supports
only an all-unknown applicable vector plus ontology-derived non-applicability;
the assessment contract does not upgrade numerical model state into measured
physical evidence.

The transport shape is published as
[`reactor_regime_assessment.schema.json`](../../specs/reactor_regime_assessment.schema.json).

## Reactor research ControlIntent hypothesis

`ReactorResearchControlIntent` is a digest-sealed SPO research hypothesis for
possible later CONTROL review. Its name does not confer CONTROL authority. The
contract permanently fixes `authority="review_only"`, `actionable=false`, and
`execution_permitted=false`; it imports no supervisor, actuation, device
adapter, or SCPN-CONTROL runtime.

Each hypothesis binds the exact SPO producer revision and artifact, source
semantic handoff and producer revision, a prior CONTROL review-admission
decision, a classified regime-axis assignment, semantic and evidence sets,
registry/observability/ontology identities, and a device-owner control-contract
schema and digest. Its candidate variable includes units, device bounds,
evidence-bound baseline, proposed value and delta, direction, maximum delta and
rate, proposed rate, and rate horizon. Complete clock identity, validity,
evidence class, quality, confidence subject, observability, and unit-aware
uncertainty remain explicit.

The objective vocabulary is restricted to five physical axes. The target
vocabulary is narrower still: it refuses harmful and ambiguous targets and
permits only established confinement/assembly, symmetric or quiescent
stability, synchronized drivers, rising or sustained power/burn, and
conditioned or regulated boundary/exhaust hypotheses. CONTROL must still apply
its own objective/target/variable/direction allowlists.

The JSON Schema validates transport shape; the Python decoder additionally
enforces sorted identifiers, ontology applicability, source and target labels,
registry digests, baseline/delta/value equality, delta/rate/horizon equality,
clock ordering, evidence quality, and non-actuation invariants. Neither proves
an external digest merely by receiving it. A future consumer must independently
decode or retrieve the exact source handoff, prior CONTROL decision, regime
assignment, and device contract.

No current semantic profile declares this surface. In particular, the current
MIF handoff has an unknown regime, while this contract requires a classified,
valid, quality-qualified source assignment. CONTROL has no ControlIntent
consumer or action edge. The portable shape is published as
[`reactor_control_intent.schema.json`](../../specs/reactor_control_intent.schema.json).

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
    coupled_transport_handoff_from_fusion_bytes,
    handoff_from_bytes,
    handoff_to_bytes,
)

reference = build_reactor_reference_portfolio()
tokamak_context: ReactorContext = reference[0].context
payload = canonical_json(tokamak_context)

# Cross-project path: exact FUSION bytes -> SPO semantic bytes.
fusion_bytes = open("torax_runtime_review_envelope_v1.json", "rb").read()
semantic_bytes = handoff_to_bytes(
    coupled_transport_handoff_from_fusion_bytes(fusion_bytes)
)
```

The root package exports only the five stable contracts. Registry, evidence,
serialization, relation-building, enums, and portfolio helpers live under
`scpn_phase_orchestrator.reactor_semantics` so their use remains explicit.

::: scpn_phase_orchestrator.reactor_semantics
    options:
      show_root_heading: true
      show_source: false
      members_order: source

## Module reference

The package facade above is the supported discovery surface. The references
below document the modules that define and validate each part of the wire
contract, including the producer-specific adapters and the review-only
assessment and research-intent envelopes.

::: scpn_phase_orchestrator.reactor_semantics.abstaining_assessment
    options:
      show_root_heading: true
      show_source: false

::: scpn_phase_orchestrator.reactor_semantics.contracts
    options:
      show_root_heading: true
      show_source: false

::: scpn_phase_orchestrator.reactor_semantics.control_intent
    options:
      show_root_heading: true
      show_source: false

::: scpn_phase_orchestrator.reactor_semantics.coupled_transport
    options:
      show_root_heading: true
      show_source: false

::: scpn_phase_orchestrator.reactor_semantics.diagnostic_plan_review
    options:
      show_root_heading: true
      show_source: false

::: scpn_phase_orchestrator.reactor_semantics.evidence
    options:
      show_root_heading: true
      show_source: false

::: scpn_phase_orchestrator.reactor_semantics.handoff
    options:
      show_root_heading: true
      show_source: false

::: scpn_phase_orchestrator.reactor_semantics.mast_magnetic_review
    options:
      show_root_heading: true
      show_source: false

::: scpn_phase_orchestrator.reactor_semantics.mif_merge_compression
    options:
      show_root_heading: true
      show_source: false

::: scpn_phase_orchestrator.reactor_semantics.observability_profiles
    options:
      show_root_heading: true
      show_source: false

::: scpn_phase_orchestrator.reactor_semantics.reference_portfolio
    options:
      show_root_heading: true
      show_source: false

::: scpn_phase_orchestrator.reactor_semantics.regime_assessment
    options:
      show_root_heading: true
      show_source: false

::: scpn_phase_orchestrator.reactor_semantics.regime_ontology
    options:
      show_root_heading: true
      show_source: false

::: scpn_phase_orchestrator.reactor_semantics.registry
    options:
      show_root_heading: true
      show_source: false

::: scpn_phase_orchestrator.reactor_semantics.semantic_profiles
    options:
      show_root_heading: true
      show_source: false

::: scpn_phase_orchestrator.reactor_semantics.serialization
    options:
      show_root_heading: true
      show_source: false

::: scpn_phase_orchestrator.reactor_semantics.vocabulary
    options:
      show_root_heading: true
      show_source: false
