# Reactor producer-evidence priority register

This register turns the reactor technology atlas and current SCPN evidence map
into a conservative sequence of producer-to-SPO intake lanes. It covers all
**32 built-in configurations** across **8 confinement families** and the full
system scope of **22 upstream reactor projects plus SCPN-CONTROL**. The exact
identity model keeps three related counts distinct: 20 Reactor Systems projects
are present in the diagnostic-plan portfolio; the registry has 21 distinct
`device_project` owners because it additionally assigns `frc_compression_mif`
to SCPN-MIF-CORE; and the 22-project upstream set additionally includes
SCPN-FUSION-CORE as a physical/simulation evidence producer for configurations
whose device owner is another project.

The register answers one bounded question: what exact evidence boundary must
be closed next before a configuration can move toward a qualified physical
observation? It does not decide which reactor technology is scientifically,
economically, strategically, or commercially preferable.

## Methodology

Every row is an exact join of four sealed SPO artifacts:

- the configuration evidence coverage matrix;
- the diagnostic-plan portfolio status;
- the signal occurrence ledger; and
- the external technology and diagnostic atlas.

The register applies deterministic custody precedence:

1. reviewed physical-source custody already exists;
2. an exercised byte-canonical review adapter already exists;
3. an exact-custody diagnostic-plan fixture is structurally accepted; or
4. the diagnostic plan is structurally refused and must be repaired first.

This order describes **integration readiness**, not research importance. No
opaque or additive priority score is emitted, and rows within one lane are
deliberately unordered. Selecting one row within a lane requires an explicit
review of safety value, calibrated machine-readable data availability,
clock/reference/operator completeness, uncertainty, validity, quality,
provenance, observability gates, independent validation, and producer capacity.

External `E5` through `E0` evidence ranks remain context only. They never alter
the intake lane. For example, the `beam_target` row carries `E5` external
context but stays in L3 because its current diagnostic plan is structurally
refused. The `colliding_beam` row carries `E1` and stays in the same L3 lane for
the same exact producer-custody reason.

## Lane result

| Lane | Exact next boundary | Configurations |
|---|---|---:|
| `L0_qualify_existing_physical_source` | Complete qualification of already-reviewed physical-source custody | 1 |
| `L1_extend_exercised_review_adapter` | Supply a physical producer payload through an existing byte-canonical adapter boundary | 2 |
| `L2_build_from_accepted_plan` | Convert an accepted design declaration into a configuration-specific physical sample envelope | 13 |
| `L3_repair_refused_plan_before_intake` | Repair and regenerate canonical plan bytes, then supply a physical sample envelope | 16 |

### L0 — qualify existing physical source

`spherical_tokamak` is the only L0 row. SCPN-FUSION-CORE already supplies the
reviewed FAIR-MAST physical source, while SCPN-TOKAMAK-CORE remains the device
owner. The next gate is not another literature review or generic tokamak plan.
It is completion of calibration lineage, physical geometry/frame binding,
modal observation operator and harmonic basis, provider quality, uncertainty,
validity, instrument-to-facility clock correlation, resolved event identity,
observability threshold, and independent multi-shot or classifier evidence.

L0 still has no qualified observation, physical phase, CONTROL admission, or
actuation authority.

### L1 — extend exercised review adapters

- `conventional_tokamak` routes its next exact producer request to
  SCPN-FUSION-CORE.
- `frc_compression_mif` routes its next exact producer request to
  SCPN-MIF-CORE.

Both configurations have exercised byte-canonical review adapters, but those
adapters carry simulation evidence only. The next input must be a physical
sample envelope with configuration-specific diagnostic identity, clock and
reference binding, physical observation operator or calibration, uncertainty,
validity, quality, provenance, and an evaluated observability gate.

`SCPN-MIF-CORE` is the registry's 21st distinct `device_project` owner and is
not one of the 20 projects in the diagnostic-plan portfolio. Together with
SCPN-FUSION-CORE, it completes the two upstream physics cores above those 20
device-plan producers. Its L1 classification comes from its existing adapter,
not from an absent plan row.

### L2 — build from accepted plans

The 13 L2 configurations are:

- `dense_plasma_focus`;
- `ion_beam_icf` and `pulsed_electron_beam_icf`;
- `laser_icf_direct_drive`, `laser_icf_indirect_drive`, and
  `laser_icf_fast_or_shock_ignition`;
- `projectile_or_impact_icf`;
- `maglif`, `mechanical_or_liquid_liner_mif`, and `plasma_jet_mif`;
- `sheared_flow_z_pinch`, `theta_pinch`, and `z_pinch`.

Their exact-custody plans define intended channels, clocks, carriers, and
evidence slots. They contain no sampled physical values. The next gate is a
configuration-specific canonical evidence envelope owned by the named device
project, with immutable source revision and package identity. Shared plans do
not equate configurations: the three laser-ICF rows, two beam-ICF rows, two
Z-pinch rows, and three separate MIF device projects remain independent.

### L3 — repair refused plans before intake

The 16 L3 configurations are:

- `beam_target` and `colliding_beam`;
- `cusp`;
- `field_reversed_configuration`;
- `fusion_fission_hybrid`;
- `gas_dynamic_mirror`, `simple_magnetic_mirror`, and `tandem_mirror`;
- `gridded_iec` and `polywell`;
- `heliotron`, `stellarator`, and `torsatron`;
- `levitated_dipole`;
- `reversed_field_pinch`; and
- `spheromak`.

Each current producer fixture omits the mandatory
`timing_uncertainty_s` member. The producer must emit the member explicitly,
regenerate the canonical plan and envelope, refresh its fixture digest and
package identity, and let SPO replay the new bytes. Only then can a separate
physical sample envelope be designed. SPO must not infer JSON `null`, relax the
schema, or use external technology evidence to bypass this gate.

## Required physical evidence

Every configuration-specific producer request carries the same minimum field
vector:

1. physical sample;
2. phenomenon identity;
3. reference;
4. clock epoch;
5. observation operator or calibration;
6. uncertainty;
7. validity;
8. quality;
9. provenance; and
10. observability gate.

The artifact must also bind the exact source revision, reproducible package
identity, and canonical producer bytes. Independent validation is mandatory.
A diagnostic name, facility page, source abstract, design plan, model output,
or topology resemblance is not a substitute.

## CONTROL and machine-safety boundary

All 32 rows remain `review_only`, `actionable=false`,
`direct_actuation_authorized=false`, and
`machine_protection_final_veto=true`. The register reports zero complete
physical evidence chains, zero qualified physical observations, zero qualified
physical phases, and zero CONTROL admissions.

The lane name is not a command or an admission decision. CONTROL must not
consume a lane, external evidence rank, plan status, occurrence ID, candidate
ID, or producer request as signal, regime, intent, execution, or actuation
evidence. A future byte-canonical SPO review remains a separate fail-closed
boundary.

## Machine-readable custody

The canonical artifact is
[`reactor_producer_evidence_priority_register.v1.json`](data/reactor_producer_evidence_priority_register.v1.json).
It is validated by
[`reactor_producer_evidence_priority_register.schema.json`](../specs/reactor_producer_evidence_priority_register.schema.json)
and sealed over canonical JSON payload bytes.

- Schema: `scpn-phase-orchestrator.reactor-producer-evidence-priority-register.v1`
- Schema version: `1.0.0`
- Payload SHA-256: `e287c4160d7c070b0b22aca3cd0d53c54946421357438cc99de0f60549c3966c`
- Configuration evidence payload:
  `a41d53e9c0dce5482a131ecaf996442f56c02e8b1f0737067c2d3cccb677f7d8`
- Diagnostic-plan portfolio payload:
  `5b02e7fb2302c2e66bf0fc4a25dae82de673b00c5921e14bb0c8a73a8ecaa1dd`
- Signal occurrence payload:
  `b7bffc61956dc32ee5ee1c1c9d399ee3546af9e374cd52d18f7c85602ba32c22`
- Technology atlas payload:
  `eb8e2ffbbc98241ac2458044455bcec425860f33ab3ba9d5ea4fa3b86870d3d3`

Any input artifact, custody state, plan result, lane, blocker, producer route,
readiness axis, or authority change alters the payload seal and requires
deliberate review.
