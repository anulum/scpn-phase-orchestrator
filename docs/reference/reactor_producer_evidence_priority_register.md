# Reactor producer-evidence priority register

This register turns the evidence atlas and exact SCPN custody map into a
conservative sequence of producer-to-SPO intake lanes. It covers all **34
registered configurations** across **9 confinement families**.

The project counts are intentionally distinct:

- **22 Reactor Systems device repositories** are in the diagnostic-plan
  portfolio;
- the registry has **23 distinct `device_project` owners** after adding
  SCPN-MIF-CORE;
- adding SCPN-FUSION-CORE yields **24 upstream reactor projects**; and
- SCPN-CONTROL is the separate 25th project boundary.

The register asks which exact evidence boundary closes next. It does not rank
reactor technologies scientifically, economically, strategically, or
commercially.

## Method and precedence

Every row is an exact join of the sealed configuration-coverage,
diagnostic-plan, occurrence-ledger, and technology-atlas artifacts. Intake
precedence is deterministic:

1. qualify reviewed physical-source custody;
2. extend an exercised byte-canonical review adapter;
3. create a missing versioned diagnostic plan;
4. build physical evidence from an accepted plan; or
5. repair a refused plan before physical intake.

No opaque or additive priority score is emitted, and rows within one lane are
deliberately unordered. External `E5` through `E0` evidence ranks remain
context only and never alter the lane or grant authority.

## Lane result

| Lane | Exact next boundary | Configurations |
|---|---|---:|
| `L0_qualify_existing_physical_source` | Complete qualification of reviewed physical-source custody | 1 |
| `L1_extend_exercised_review_adapter` | Supply a physical producer payload through an existing review boundary | 2 |
| `L2_build_missing_diagnostic_plan` | Publish a versioned, configuration-specific diagnostic plan | 2 |
| `L3_build_from_accepted_plan` | Supply a configuration-specific physical sample envelope | 13 |
| `L4_repair_refused_plan_before_intake` | Repair the exact plan contract, then supply physical evidence | 16 |

### L0 — physical-source qualification

`spherical_tokamak` is the only L0 row. SCPN-FUSION-CORE supplies reviewed
FAIR-MAST physical-source bytes, while SCPN-TOKAMAK-CORE remains the device
owner. The source has no declared portable semantic-ingress profile. The
materialized `mast_phase_qualification_request_from_source_review()` request
therefore remains blocked on controlled phenomenon identity, reproducible
source-ingestion state, calibration, geometry/frame and modal observation operators,
clock correlation, uncertainty, validity, producer-owned plant-truth-state
semantics, observability, and independent evidence.

Its request ID is
`3aae1686abf2b3854d2136079118c7f68e6c75e769906d4c45f0f4adda7bc722` and
canonical envelope SHA-256 is
`33156dd1759a4c1e3209f3cf166ef69270bb82150ccadb71bbf9bdfb962bf006`.

### L1 — exercised review adapters

`conventional_tokamak` routes to SCPN-FUSION-CORE through
`conventional_tokamak_physical_payload_request()`. `frc_compression_mif`
routes to SCPN-MIF-CORE through
`frc_compression_mif_physical_payload_request()`. Both current adapters are
simulation-only, forbidding reuse as physical evidence.

The requests require immutable source/package identity, canonical bytes,
independent validation, and distinct `unknown`, `out_of_distribution`,
`low_observability`, and `stale` producer dispositions about current plant
truth. Those are validity causes, not physical reactor regimes; quality labels
cannot substitute for them.

### L2 — missing diagnostic plans

The two namespaced extensions
`scpn.reactor_systems:lattice_confinement_fusion` and
`scpn.reactor_systems:muon_catalysed_fusion` are architecture-only projects.
Their next boundary is a producer-owned, versioned diagnostic plan. Literature
evidence, registry identity, or green software workflows do not create such a
plan.

### L3 — accepted plans

Thirteen configurations map to the seven accepted producer objects:
SCPN-ICF-BEAM-CORE, SCPN-ICF-IMPACT-CORE, SCPN-ICF-LASER-CORE, SCPN-IEC-CORE,
SCPN-LEVITATED-DIPOLE-CORE, SCPN-MAGNETIC-CUSP-CORE, and
SCPN-STELLARATOR-CORE. Their plans declare intended channels, carriers, frames,
clocks, and evidence slots; they do not contain physical samples.

The first materialised L3 boundary is the direct-drive laser-ICF
`device_physical_evidence_request_from_plan_review()` request. It embeds the
accepted SCPN-ICF-LASER-CORE review while remaining specific to
`laser_icf_direct_drive`; the other two laser-ICF configurations inherit no
evidence. Its request ID is
`3f273e5ef1fb68e7a928913a7f7a8c9b5e6055a7649c722598911fa39458111a`
and canonical envelope SHA-256 is
`f42a9817dcef628caefab5ba5681853327bae9b21ba72459eb9588e14c2ed6a9`.

### L4 — refused plans

Sixteen configurations map to thirteen refused producer objects. Seven
projects lack the canonical shared-kernel owner exclusion. Six have plan
envelopes whose source digests do not match the current manifest and plan
bytes; their exact-head `CI` runs also fail. The owner must publish corrected
canonical objects before SPO can review physical evidence. Refusal is not
silently bypassed by a related topology or a previously accepted version.

## Required physical evidence

Every configuration-specific evidence request requires at least:

1. a physical sample and phenomenon identity;
2. reference and clock epoch;
3. observation operator or calibration;
4. uncertainty and validity;
5. quality and provenance; and
6. an evaluated observability gate.

The artifact must bind the exact source revision, reproducible package
identity, and canonical producer bytes. Independent validation is mandatory.
A diagnostic name, facility page, abstract, design plan, model output, or
topology resemblance is not a substitute.

## CONTROL and machine-safety boundary

All 34 rows remain `review_only`, `actionable=false`,
`direct_actuation_authorized=false`, and
`machine_protection_final_veto=true`. The register reports zero complete
physical evidence chains, zero qualified physical observations, zero qualified
physical phases, and zero CONTROL admissions.

CONTROL must not consume a lane, external evidence rank, plan status,
occurrence ID, candidate ID, or producer request as signal, regime, intent,
execution, or actuation evidence.

## Machine-readable custody

The canonical
[`reactor_producer_evidence_priority_register.v1.json`](data/reactor_producer_evidence_priority_register.v1.json)
is validated by
[`reactor_producer_evidence_priority_register.schema.json`](../specs/reactor_producer_evidence_priority_register.schema.json).

- Schema version: `1.2.0`
- Payload SHA-256: `ba09efdf2bfc9b36b92aeabb2c0b1f306d10a735d5ed989204dea6ddaa38929b`
- Configuration evidence payload: `4ee797e9bd03d646d538b30ecbf468a90e70be4bdada922b09eb84e483f3b730`
- Diagnostic-plan portfolio payload: `13bdcfd794cab002903d4861a378056536e0fcb98beca64863a9b36cc71558a5`
- Signal occurrence payload: `8210dc2310a7031ccad1a1675677e3e92007a2dd82e696c39d25202d2f9f022f`
- Technology atlas payload: `21dcfa1b4c54e09e6b860101bed5df927655d887e974460a105f5e97cb4138ed`

Any input artifact, custody state, plan result, lane, blocker, producer route,
readiness axis, or authority change alters the seal and requires deliberate
review.
