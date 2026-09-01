# Reactor configuration evidence coverage

This matrix answers a deliberately narrow question: for each built-in reactor
configuration known to SPO, what evidence source and portable semantic ingress
exist in the exact SCPN-PHASE-ORCHESTRATOR, SCPN-FUSION-CORE, SCPN-MIF-CORE,
and SCPN-CONTROL snapshot represented by the occurrence ledger?

Configuration identity, a device-project assignment, and candidate observability
requirements are design declarations. They do not prove that a producer exists,
that a signal was observed in a reactor, or that a phase meaning is qualified.
No row inherits evidence from another configuration merely because both share a
confinement family, device class, or similar topology.

## Coverage result

- All **32 built-in configurations** across **8 confinement families** have an
  explicit row.
- **2 configurations** have an exercised, byte-canonical, review-only producer
  adapter: `conventional_tokamak` and `frc_compression_mif`.
- **1 configuration** has a reviewed physical archive source but no qualified
  observation or physical phase: `spherical_tokamak`.
- **2 configurations** have only local-model or synthetic-replay source evidence:
  `field_reversed_configuration` and `stellarator`.
- **27 configurations** have no configuration-specific source occurrence in the
  exact four-project snapshot.
- **30 configurations** have no portable producer-to-SPO semantic ingress profile.
- **0 configurations** have a qualified physical observation, qualified physical
  phase, direct actuation authority, or permission to bypass machine protection.

“No source evidence” is an explicit producerless result for this captured scope;
it is not a claim that no implementation or experiment exists elsewhere. Likewise,
“physical source, unqualified” records custody of physical-source material without
promoting derived time grids, channel arrays, or source labels into observations.

## Design declarations are not evidence ingress

Separate byte-canonical diagnostic-plan reviews exercise exact tokamak, dense-
plasma-focus, MagLIF, mechanical-or-liquid-liner MIF, plasma-jet MIF,
laser-ICF, and ICF-beam producer fixtures. These are design declarations about
intended channels, clocks, carriers, and evidence slots; they contain no sampled
signal values. Consequently, dense plasma focus, the three MIF configurations,
and the three laser-ICF plus two ICF-beam configurations do not change their nine
matrix rows: each still has no configuration-specific source evidence, no
semantic producer, and `semantic_ingress_state=not_declared`. Neither MagLIF,
mechanical-or-liquid-liner MIF, nor plasma-jet MIF inherits the verified
`frc_compression_mif` adapter merely because they are magneto-inertial
configurations; the three MIF design reviews also do not provide evidence for
one another. Direct-drive, indirect-drive, and fast/shock-ignition laser ICF do
not provide evidence for one another or inherit ion/electron-beam, projectile,
or impact ICF evidence. Ion-beam and pulsed-electron-beam ICF likewise do not
provide evidence for one another or inherit evidence from laser, projectile,
impact, or generic beam-target configurations.

## Exact matrix

Occurrence IDs resolve into the [reactor signal occurrence ledger](reactor_signal_occurrence_ledger.md).
Candidate IDs resolve through SPO’s observability registry. `not_declared` means
no complete, exercised producer-to-SPO adapter is advertised for that configuration.

| Family | Configuration | Candidate IDs | Evidence state | Occurrences | Semantic ingress |
|---|---|---|---|---|---|
| `magnetic_closed` | `conventional_tokamak` | `closed.equilibrium_profiles`<br>`closed.recurrent_transient`<br>`closed.resolved_mhd_mode`<br>`model.synthetic_oscillator_coordinate` | verified simulation review adapter | `FUS-008`, `SPO-006`, `CTRL-001` | `verified_review_adapter` |
| `magnetic_closed` | `field_reversed_configuration` | `closed.equilibrium_profiles`<br>`closed.resolved_mhd_mode`<br>`model.synthetic_oscillator_coordinate` | local model only | `FUS-004`, `FUS-005` | `not_declared` |
| `magnetic_closed` | `heliotron` | `closed.equilibrium_profiles`<br>`closed.resolved_mhd_mode`<br>`model.synthetic_oscillator_coordinate` | no source evidence | — | `not_declared` |
| `magnetic_closed` | `reversed_field_pinch` | `closed.equilibrium_profiles`<br>`closed.resolved_mhd_mode`<br>`model.synthetic_oscillator_coordinate` | no source evidence | — | `not_declared` |
| `magnetic_closed` | `spherical_tokamak` | `closed.equilibrium_profiles`<br>`closed.recurrent_transient`<br>`closed.resolved_mhd_mode`<br>`model.synthetic_oscillator_coordinate` | physical source, unqualified | `FUS-009`, `FUS-010`, `SPO-009` | `not_declared` |
| `magnetic_closed` | `spheromak` | `closed.equilibrium_profiles`<br>`closed.resolved_mhd_mode`<br>`model.synthetic_oscillator_coordinate` | no source evidence | — | `not_declared` |
| `magnetic_closed` | `stellarator` | `closed.equilibrium_profiles`<br>`closed.resolved_mhd_mode`<br>`model.synthetic_oscillator_coordinate` | synthetic replay only | `FUS-011` | `not_declared` |
| `magnetic_closed` | `torsatron` | `closed.equilibrium_profiles`<br>`closed.resolved_mhd_mode`<br>`model.synthetic_oscillator_coordinate` | no source evidence | — | `not_declared` |
| `magnetic_open` | `cusp` | `model.synthetic_oscillator_coordinate`<br>`open.drive_reference`<br>`open.equilibrium_and_loss`<br>`open.resolved_interchange_mode` | no source evidence | — | `not_declared` |
| `magnetic_open` | `gas_dynamic_mirror` | `model.synthetic_oscillator_coordinate`<br>`open.drive_reference`<br>`open.equilibrium_and_loss`<br>`open.resolved_interchange_mode` | no source evidence | — | `not_declared` |
| `magnetic_open` | `levitated_dipole` | `model.synthetic_oscillator_coordinate`<br>`open.drive_reference`<br>`open.equilibrium_and_loss`<br>`open.resolved_interchange_mode` | no source evidence | — | `not_declared` |
| `magnetic_open` | `polywell` | `iec.resolved_bunching`<br>`iec.steady_state`<br>`model.synthetic_oscillator_coordinate`<br>`open.drive_reference`<br>`open.equilibrium_and_loss`<br>`open.resolved_interchange_mode` | no source evidence | — | `not_declared` |
| `magnetic_open` | `simple_magnetic_mirror` | `model.synthetic_oscillator_coordinate`<br>`open.drive_reference`<br>`open.equilibrium_and_loss`<br>`open.resolved_interchange_mode` | no source evidence | — | `not_declared` |
| `magnetic_open` | `tandem_mirror` | `model.synthetic_oscillator_coordinate`<br>`open.drive_reference`<br>`open.equilibrium_and_loss`<br>`open.resolved_interchange_mode` | no source evidence | — | `not_declared` |
| `self_magnetic` | `dense_plasma_focus` | `model.synthetic_oscillator_coordinate`<br>`self_magnetic.drive_waveform`<br>`self_magnetic.resolved_instability_mode` | no source evidence | — | `not_declared` |
| `self_magnetic` | `sheared_flow_z_pinch` | `model.synthetic_oscillator_coordinate`<br>`self_magnetic.drive_waveform`<br>`self_magnetic.resolved_instability_mode` | no source evidence | — | `not_declared` |
| `self_magnetic` | `theta_pinch` | `model.synthetic_oscillator_coordinate`<br>`self_magnetic.drive_waveform`<br>`self_magnetic.resolved_instability_mode` | no source evidence | — | `not_declared` |
| `self_magnetic` | `z_pinch` | `model.synthetic_oscillator_coordinate`<br>`self_magnetic.drive_waveform`<br>`self_magnetic.resolved_instability_mode` | no source evidence | — | `not_declared` |
| `magneto_inertial` | `frc_compression_mif` | `magneto_inertial.driver_arrival`<br>`magneto_inertial.resolved_asymmetry_mode`<br>`magneto_inertial.translation_and_compression`<br>`model.synthetic_oscillator_coordinate` | verified simulation review adapter | `MIF-001`, `SPO-007`, `CTRL-002` | `verified_review_adapter` |
| `magneto_inertial` | `maglif` | `magneto_inertial.driver_arrival`<br>`magneto_inertial.resolved_asymmetry_mode`<br>`magneto_inertial.translation_and_compression`<br>`model.synthetic_oscillator_coordinate` | no source evidence | — | `not_declared` |
| `magneto_inertial` | `mechanical_or_liquid_liner_mif` | `magneto_inertial.driver_arrival`<br>`magneto_inertial.resolved_asymmetry_mode`<br>`magneto_inertial.translation_and_compression`<br>`model.synthetic_oscillator_coordinate` | no source evidence | — | `not_declared` |
| `magneto_inertial` | `plasma_jet_mif` | `magneto_inertial.driver_arrival`<br>`magneto_inertial.resolved_asymmetry_mode`<br>`magneto_inertial.translation_and_compression`<br>`model.synthetic_oscillator_coordinate` | no source evidence | — | `not_declared` |
| `inertial` | `ion_beam_icf` | `inertial.driver_timing`<br>`inertial.implosion_trajectory`<br>`inertial.resolved_asymmetry_mode`<br>`inertial.shot_outcome`<br>`model.synthetic_oscillator_coordinate` | no source evidence | — | `not_declared` |
| `inertial` | `laser_icf_direct_drive` | `inertial.driver_timing`<br>`inertial.implosion_trajectory`<br>`inertial.resolved_asymmetry_mode`<br>`inertial.shot_outcome`<br>`model.synthetic_oscillator_coordinate` | no source evidence | — | `not_declared` |
| `inertial` | `laser_icf_fast_or_shock_ignition` | `inertial.driver_timing`<br>`inertial.implosion_trajectory`<br>`inertial.resolved_asymmetry_mode`<br>`inertial.shot_outcome`<br>`model.synthetic_oscillator_coordinate` | no source evidence | — | `not_declared` |
| `inertial` | `laser_icf_indirect_drive` | `inertial.driver_timing`<br>`inertial.implosion_trajectory`<br>`inertial.resolved_asymmetry_mode`<br>`inertial.shot_outcome`<br>`model.synthetic_oscillator_coordinate` | no source evidence | — | `not_declared` |
| `inertial` | `projectile_or_impact_icf` | `inertial.driver_timing`<br>`inertial.implosion_trajectory`<br>`inertial.resolved_asymmetry_mode`<br>`inertial.shot_outcome`<br>`model.synthetic_oscillator_coordinate` | no source evidence | — | `not_declared` |
| `inertial` | `pulsed_electron_beam_icf` | `inertial.driver_timing`<br>`inertial.implosion_trajectory`<br>`inertial.resolved_asymmetry_mode`<br>`inertial.shot_outcome`<br>`model.synthetic_oscillator_coordinate` | no source evidence | — | `not_declared` |
| `beam_target` | `beam_target` | `beam.rf_bunch_phase`<br>`beam.target_outcome`<br>`model.synthetic_oscillator_coordinate` | no source evidence | — | `not_declared` |
| `beam_target` | `colliding_beam` | `beam.rf_bunch_phase`<br>`beam.target_outcome`<br>`model.synthetic_oscillator_coordinate` | no source evidence | — | `not_declared` |
| `electrostatic` | `gridded_iec` | `iec.resolved_bunching`<br>`iec.steady_state`<br>`model.synthetic_oscillator_coordinate` | no source evidence | — | `not_declared` |
| `hybrid` | `fusion_fission_hybrid` | `hybrid.source_blanket_response`<br>`model.synthetic_oscillator_coordinate` | no source evidence | — | `not_declared` |

## Evidence and authority boundaries

The evidence axis and the semantic-ingress axis are independent. For example,
`spherical_tokamak` has an exact physical-source review but still has
`semantic_ingress_state=not_declared`; its MAST arrays are not qualified
observations and do not establish a physical phase. Conversely, the two verified
adapters carry simulation evidence into deterministic review records, not physical
observations or controller commands.

Every row is `authority=review_only`, `actionable=false`,
`direct_actuation_authorized=false`, and
`machine_protection_final_veto=true`. CONTROL or another consumer must not infer
actuation permission from registry membership, evidence presence, regime labels,
or review admission.

## Machine-readable custody

The canonical artifact is
[`reactor_configuration_evidence_coverage.v1.json`](data/reactor_configuration_evidence_coverage.v1.json).
It is validated by
[`reactor_configuration_evidence_coverage.schema.json`](../specs/reactor_configuration_evidence_coverage.schema.json)
and sealed over canonical JSON payload bytes.

- Schema: `scpn-phase-orchestrator.reactor-configuration-evidence-coverage.v1`
- Schema version: `1.0.0`
- Payload SHA-256: `a41d53e9c0dce5482a131ecaf996442f56c02e8b1f0737067c2d3cccb677f7d8`
- Configuration registry: `1.0.0` / `786d9542ce76c56dd7748fa948b17efed6c073525e527ce90e6d5e29a2d00090`
- Observability registry: `1.0.0` / `d70c0de696534e5a77066ef8420cf7ca17bc4d7321984b0ac83523dbc1dce609`
- Semantic-profile registry: `1.0.0` / `6ac7f3863e1a5f50af297c572ec0b80b60820a23de1a769fda6bb0a831243ec3`
- Occurrence-ledger payload: `b7bffc61956dc32ee5ee1c1c9d399ee3546af9e374cd52d18f7c85602ba32c22`

Any change to a registry, source digest, occurrence binding, evidence state, or
authority field changes the payload seal and requires deliberate review.
