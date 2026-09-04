# Conventional-tokamak physical-payload request

SPO materialises the `L1_extend_exercised_review_adapter` next boundary as the
digest-sealed schema
`scpn-phase-orchestrator.conventional-tokamak-physical-payload-request.v1`
version `1.1.0`. The public builder is
`conventional_tokamak_physical_payload_request()`.

The request binds the existing SCPN-FUSION-CORE TORAX adapter, source schema,
handoff schema, semantic profile, semantic-profile registry, and observability
registry. This binding proves that a byte-canonical producer-to-SPO review path
exists. It does **not** convert TORAX simulation output into a physical sample:
`source_kind=simulation`, `physical_source_present=false`, and
`reusable_as_physical_evidence=false` are fixed contract members.

## Producer obligations

SCPN-FUSION-CORE must allocate a configuration-specific canonical physical
payload and satisfy all thirteen prerequisites:

1. immutable physical sample identity and sampled values;
2. conventional-tokamak-specific diagnostic, channel, geometry, and frame
   identity;
3. controlled phenomenon identity and semantic carrier;
4. physical reference identity;
5. diagnostic-to-facility and event clock correlation;
6. a validated observation operator or calibration lineage;
7. uncertainty;
8. validity;
9. distinct producer evidence-state semantics for `unknown`,
   `out_of_distribution`, `low_observability`, and `stale`;
10. provider and derived quality semantics;
11. immutable provenance and reproducibility, including source and package
    identity;
12. a predeclared and evaluated observability gate; and
13. independent validation without same-shot circularity.

The four evidence dispositions describe why current plant truth cannot support
a physical classification. They map to U0 validity `unknown`,
`out_of_distribution`, `unobservable`, and `stale`, respectively, and every
mapping forces an unclassified `RegimeState.UNKNOWN` result. They are not
plasma modes or operating regimes. Provider quality is orthogonal and cannot
replace the disposition; a merely small score above a predeclared observability
gate is not `low_observability`.

The four applicable catalogue entries are carried as unselected candidate
requirements: equilibrium profiles, a recurrent transient, a resolved MHD
mode, and the simulation-only synthetic oscillator coordinate. Applicability
is not evidence. New peer discoveries enter the atlas as gaps or provisional
candidates and require a versioned registry change before this request can
reference them canonically.

## Authority boundary

The request fixes `selected_candidate_id=null`,
`physical_payload_schema_allocated=false`, `physical_source_present=false`,
`observation_admitted=false`, `phase_inference_eligible=false`,
`producer_evidence_state_contract_present=false`,
`quality_state_may_substitute_for_evidence_state=false`,
`semantic_ingress_extended=false`, `control_admission_requested=false`,
`actionable=false`, `execution_permitted=false`, and
`direct_actuation=false`. It is review-only and machine protection retains the
final veto.

Neither the external technology rank, the accepted SCPN-TOKAMAK-CORE design
plan, the existing simulation adapter, nor this request may be consumed as a
physical observation, physical phase, CONTROL intent, action, execution, or
actuation authority.

The normative JSON Schema is
[`conventional_tokamak_physical_payload_request.schema.json`](../specs/conventional_tokamak_physical_payload_request.schema.json).
The canonical producer-priority register pins request ID
`3f01f59e422421bdb98bfa51aff7f3f5378d96e3b5c8ffbdc80ae4899f027aba`
and complete envelope SHA-256
`ed3515b4c41ba911ba6172d4a1d22b76f1c9e9aa5bb6627e02d6975bcd65945f`.
