# Device physical-evidence request

`device_physical_evidence_request_from_plan_review()` turns one accepted,
byte-canonical device diagnostic-plan review into a configuration-specific
request for the producer's next physical-evidence object. It is the common L3
boundary for reactor families whose design plans have passed structural review
but contain no physical samples.

The request embeds the exact source review, pins its source commit and package
digest, and replays the current reactor, observability, and semantic-profile
registry bindings. It is not a generic evidence adapter and does not execute or
import the producer package.

## Direct-drive laser-ICF instance

The first materialised request is scoped only to
`laser_icf_direct_drive` and its owner `SCPN-ICF-LASER-CORE`. It does not grant
evidence to indirect-drive or fast/shock-ignition ICF, even though those
configurations share the accepted producer plan.

The request preserves five candidate meanings:

- beam timing is an event-relative `event_cycle` and needs a physical event
  reference plus repeated-cycle evidence;
- resolved implosion asymmetry is `derived_cyclic` through a validated
  `complex_mode` observation operator;
- implosion trajectory and shot outcome remain separate
  `noncyclic_feature` quantities;
- the synthetic oscillator coordinate remains `numerical_only` and is never
  eligible as a physical selection target.

The plan's facility and shot clocks still need physical correlation. Its
simulation clock remains synthetic and is ineligible as a physical reference.
A declared clock topology or timing uncertainty is not clock-correlation
evidence.

## Ion-beam ICF instance

The second materialised request uses exact `SCPN-ICF-BEAM-CORE` fixture bytes,
source revision, and reproducible wheel digest, but selects only
`ion_beam_icf`. Its bunch timing remains an event-relative `event_cycle`, its
resolved asymmetry remains `derived_cyclic`, its trajectory and shot outcome
remain noncyclic, and its model oscillator remains numerical-only.

The accepted producer review also names `pulsed_electron_beam_icf`. That shared
review identity does not merge the configurations: constructing each request
produces a distinct request ID, and only the ion-beam request is materialised
here. Neither request inherits a physical sample, phase, validity, or CONTROL
admission from the other.

The host-independent
`tools/materialize_device_physical_evidence_request.py` reconstructs the review
from exact local manifest and plan-envelope fixture bytes. It accepts the
immutable source revision and package-artifact digest explicitly, imports no
producer module, writes canonical bytes atomically, and has a fail-closed
`--check` mode. Absolute input/output paths make the command independent of the
caller's working directory.

## Projectile/impact ICF instance

The third materialised request selects only `projectile_or_impact_icf` from
exact `SCPN-ICF-IMPACT-CORE` fixture bytes, source revision, and reproducible
wheel digest. Impact timing remains an event-relative `event_cycle`; resolved
asymmetry remains `derived_cyclic`; trajectory and shot outcome remain
noncyclic bounded features; and the model oscillator remains numerical-only.

The producer review owns exactly this combined configuration. It cannot be
used to construct requests for laser, ion-beam, pulsed-electron-beam, or
generic beam-target configurations: the public factory refuses each as a
configuration mismatch. No topology resemblance transfers a physical sample,
phase, validity, CONTROL admission, or authority.

## Producer obligations

All thirteen prerequisites are explicitly missing: physical sample identity,
configuration-specific diagnostic identity, phenomenon identity, physical
reference, clock and epoch correlation, observation operator or calibration,
uncertainty, validity, producer evidence-state semantics, quality, provenance
and reproducibility, a predeclared observability gate, and independent
validation.

The producer evidence-state contract must distinguish `unknown`,
`out_of_distribution`, `low_observability`, and `stale` about current plant
truth. These map to validity and force physical-regime abstention. Quality is
orthogonal and cannot substitute for the cause.

New peer discoveries enter the atlas as explicit gaps or provisional,
source-bound candidates. SPO adds them to a released registry only after
configuration ownership, physical meaning, carrier, units, frame, clock,
evidence requirements, non-applicability, provenance, and validation are
reviewed. A peer report alone is not physical evidence and cannot silently
change an existing meaning.

## Fail-closed authority

The request fixes `selected_candidate_id=null`,
`physical_source_present=false`, `observation_admitted=false`,
`phase_inference_eligible=false`, `semantic_ingress_declared=false`,
`control_admission_requested=false`, `actionable=false`,
`execution_permitted=false`, and `direct_actuation=false`. Authority remains
`review_only`, and `machine_protection_final_veto=true`.

The canonical direct-drive request is
[`laser_icf_direct_drive_physical_evidence_request.v1.json`](data/laser_icf_direct_drive_physical_evidence_request.v1.json),
validated by
[`device_physical_evidence_request.schema.json`](../specs/device_physical_evidence_request.schema.json).

- Request ID: `3f273e5ef1fb68e7a928913a7f7a8c9b5e6055a7649c722598911fa39458111a`
- Canonical envelope SHA-256: `f42a9817dcef628caefab5ba5681853327bae9b21ba72459eb9588e14c2ed6a9`
- Embedded review ID: `0dac2e7bf5043eab60f5979b1fbf73a5331928816b2a7152c6ad41b27151d083`
- Embedded review SHA-256: `5cb5824bd6058a148d8ab71ead7a0d35939a30b8ddd8d40c1f68cad3caaf0467`

The canonical ion-beam request is
[`ion_beam_icf_physical_evidence_request.v1.json`](data/ion_beam_icf_physical_evidence_request.v1.json).

- Request ID: `b381e5d5dc8aaff311da8f7d0453ed458f154f3930dd1a3297df07d366d93854`
- Canonical envelope SHA-256: `c36256af2280a5caf786953c0c1e293b552f128acb02704123ea8073c5153b9b`
- Embedded review ID: `5da4be074476c8b3bd4a16c199d5f9f359e11f4e1fa36554765a1c880bf41719`
- Embedded review SHA-256: `6200379b8ec7284f05c2f271a0a3fda72c1e0efe3fbfaa97aef49a01a7700b3d`

The canonical projectile/impact request is
[`projectile_or_impact_icf_physical_evidence_request.v1.json`](data/projectile_or_impact_icf_physical_evidence_request.v1.json).

- Request ID: `27a576dd67b149069bd4eefa1ef343c570a0084688acd3370721e6a34023ac62`
- Canonical envelope SHA-256: `ccdda701953cdec025d3b7f63f026bbaf92efed54ecebddfbb510eb83eab64e1`
- Embedded review ID: `eeefac32254f871dc94ce655353b60327f2aa1e7dde566bd92c89c86cb8eaa84`
- Embedded review SHA-256: `5035b44a327b916f662125cc452777fb30bc43c6ee37642d354d2c46c2ff60e3`

Any source byte, registry binding, candidate meaning, clock boundary,
obligation, or authority change alters the seal and requires deliberate
review.
