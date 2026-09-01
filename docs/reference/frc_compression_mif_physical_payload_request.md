# FRC-compression MIF physical-payload request

SPO materialises the `L1_extend_exercised_review_adapter` next boundary as the
digest-sealed schema
`scpn-phase-orchestrator.frc-compression-mif-physical-payload-request.v1`
version `1.1.0`. The public builder is
`frc_compression_mif_physical_payload_request()`.

The request binds the existing SCPN-MIF-CORE merge-compression adapter, source
schema, handoff schema, semantic profile, semantic-profile registry, and
observability registry. This binding proves that a byte-canonical
producer-to-SPO review path exists. It does **not** convert numerical
merge-compression output into a physical sample:
`source_kind=simulation`, `physical_source_present=false`, and
`reusable_as_physical_evidence=false` are fixed contract members.

## Producer obligations

SCPN-MIF-CORE must allocate a configuration-specific canonical physical
payload and satisfy all thirteen prerequisites:

1. immutable physical sample identity and sampled values;
2. FRC-compression-MIF-specific diagnostic, channel, geometry, and frame
   identity;
3. controlled phenomenon identity and semantic carrier;
4. physical reference identity;
5. diagnostic-to-facility and event clock correlation;
6. a validated observation operator or calibration lineage;
7. uncertainty;
8. validity;
9. a producer-owned plant-truth state vocabulary that distinctly represents
   `unknown`, `out_of_distribution`, `low_observability`, and `stale`;
10. provider and derived quality semantics, orthogonal to plant-truth cause;
11. immutable provenance and reproducibility, including source and package
    identity;
12. a predeclared and evaluated observability gate; and
13. independent validation without same-shot circularity.

The plant-truth state contract must define classification criteria,
precedence, transitions, and interval semantics and bind each state to the
physical sample, correlated clock, validity, calibration or observation
operator, and observability-gate result. Generic `accepted`, `degraded`, or
`rejected` quality labels cannot replace or erase the physical cause. This
makes the `STATE-01` producer obligation explicit; it does not close the gap.
`plant_truth_state_contract_present=false` remains fixed until an immutable
SCPN-MIF-CORE physical payload supplies and validates that evidence.

The four applicable catalogue entries are carried as unselected candidate
requirements: driver arrival, a resolved asymmetry mode, translation and
compression trajectory, and the simulation-only synthetic oscillator
coordinate. Applicability is not evidence. New peer discoveries enter the
atlas as gaps or provisional candidates and require a versioned registry
change before this request can reference them canonically.

## Authority boundary

The request fixes `selected_candidate_id=null`,
`physical_payload_schema_allocated=false`, `physical_source_present=false`,
`plant_truth_state_contract_present=false`,
`observation_admitted=false`, `phase_inference_eligible=false`,
`semantic_ingress_extended=false`, `control_admission_requested=false`,
`actionable=false`, `execution_permitted=false`, and
`direct_actuation=false`. It is review-only and machine protection retains the
final veto.

Neither the external technology rank, the existing SCPN-MIF-CORE simulation
adapter, its numerical trigger decision, nor this request may be consumed as a
physical observation, physical phase, CONTROL intent, action, execution, or
actuation authority.

The normative JSON Schema is
[`frc_compression_mif_physical_payload_request.schema.json`](../specs/frc_compression_mif_physical_payload_request.schema.json).
The canonical producer-priority register pins the current request identifier
and complete envelope SHA-256.
