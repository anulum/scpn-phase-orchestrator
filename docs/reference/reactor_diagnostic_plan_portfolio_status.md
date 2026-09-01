# Reactor diagnostic-plan portfolio status

This snapshot answers one narrow interoperability question: which exact
diagnostic-plan fixtures from the 20 Reactor Systems device repositories can
SPO parse at its public, review-only boundary, and which fixtures are refused
before any semantic interpretation?

The observation was made at `2026-09-01T20:08:41+02:00` against each producer's
committed `reactor-domain.json` and
`tests/data/plan_envelope_fixture.json`. Each pair was passed to
`device_diagnostic_plan_review_from_producer_bytes()` without importing or
executing producer source.

## Result

- **20 producers** were examined.
- **10 fixtures** are structurally accepted and have byte-identical fixture
  custody in SPO.
- **10 fixtures** fail closed with `plan_structure_mismatch` because every
  channel omits the required `timing_uncertainty_s` member.
- **0 fixtures** constitute a qualified physical observation or physical phase.
- **0 fixtures** create CONTROL intent, action, execution, actuation, or
  machine-protection authority.

Structural acceptance means only that the synthetic design declaration obeys
the exact exchange contract. It does not validate hardware, calibration,
geometry, clocks, observability, measurements, regimes, or operation. Exact
custody means SPO holds identical fixture bytes; it does not make a historical
producer commit equivalent to the newer observed commit when unrelated files
have changed.

## Accepted and held byte-identically

| Producer | Configurations | Plan | Observed revision |
|---|---|---|---|
| `SCPN-DENSE-PLASMA-FOCUS-CORE` | `dense_plasma_focus` | `dpf_reference_plan` | `32c843c85e59` |
| `SCPN-ICF-BEAM-CORE` | `ion_beam_icf`, `pulsed_electron_beam_icf` | `icf_beam_reference_plan` | `3ee15a5bf56b` |
| `SCPN-ICF-IMPACT-CORE` | `projectile_or_impact_icf` | `icf_impact_reference_plan` | `397f1f2a5fb2` |
| `SCPN-ICF-LASER-CORE` | `laser_icf_direct_drive`, `laser_icf_fast_or_shock_ignition`, `laser_icf_indirect_drive` | `icf_laser_reference_plan` | `bc041638eefc` |
| `SCPN-MIF-LINER-CORE` | `mechanical_or_liquid_liner_mif` | `mif_liner_reference_plan` | `8b1ee018a8c8` |
| `SCPN-MIF-MAGLIF-CORE` | `maglif` | `maglif_reference_plan` | `d3b6230e9b77` |
| `SCPN-MIF-PLASMA-JET-CORE` | `plasma_jet_mif` | `plasma_jet_reference_plan` | `418b5cce6fb8` |
| `SCPN-THETA-PINCH-CORE` | `theta_pinch` | `theta_pinch_reference_plan` | `df38049bf4e6` |
| `SCPN-TOKAMAK-CORE` | `conventional_tokamak`, `spherical_tokamak` | `tokamak_reference_plan` | `9eef91eb9086` |
| `SCPN-Z-PINCH-CORE` | `sheared_flow_z_pinch`, `z_pinch` | `z_pinch_reference_plan` | `acf5dfe49de1` |

These shared plans do not equate their listed configurations and do not
transfer evidence within or between reactor families.

## Refused producer fixtures

Every listed channel omits the member; none contains an explicit JSON `null`.
All affected channels are currently non-event carriers, so producer-side
regeneration is expected to emit explicit `null`. SPO does not write that value
for a producer, because omission and a declared non-applicable timing bound are
different source claims.

| Producer | Configurations | Affected channel: carrier@clock |
|---|---|---|
| `SCPN-BEAM-TARGET-CORE` | `beam_target`, `colliding_beam` | `ch_rf_bunch_phase: cyclic_phase@clk_facility`<br>`ch_synthetic_oscillator: numerical_phase@clk_sim`<br>`ch_target_outcome_set: bounded_feature@clk_shot` |
| `SCPN-FRC-CORE` | `field_reversed_configuration` | `ch_excluded_flux_set: bounded_feature@clk_shot`<br>`ch_interferometer: bounded_feature@clk_shot`<br>`ch_mirnov_array: complex_mode@clk_facility`<br>`ch_synthetic_oscillator: numerical_phase@clk_sim` |
| `SCPN-FUSION-FISSION-HYBRID-CORE` | `fusion_fission_hybrid` | `ch_blanket_thermal_set: bounded_feature@clk_shot`<br>`ch_neutron_flux_set: bounded_feature@clk_shot`<br>`ch_synthetic_oscillator: numerical_phase@clk_sim` |
| `SCPN-IEC-CORE` | `gridded_iec`, `polywell` | `ch_bunching_probe: complex_mode@clk_facility`<br>`ch_drive_reference: cyclic_phase@clk_facility`<br>`ch_interchange_probe_array: complex_mode@clk_facility`<br>`ch_loss_profile_set: bounded_feature@clk_shot`<br>`ch_steady_state_set: bounded_feature@clk_shot`<br>`ch_synthetic_oscillator: numerical_phase@clk_sim` |
| `SCPN-LEVITATED-DIPOLE-CORE` | `levitated_dipole` | `ch_drive_reference: cyclic_phase@clk_facility`<br>`ch_flux_loop_set: bounded_feature@clk_shot`<br>`ch_interchange_probe_array: complex_mode@clk_facility`<br>`ch_interferometer: bounded_feature@clk_shot`<br>`ch_synthetic_oscillator: numerical_phase@clk_sim` |
| `SCPN-MAGNETIC-CUSP-CORE` | `cusp` | `ch_cusp_loss_probes: bounded_feature@clk_shot`<br>`ch_diamagnetic_loop: bounded_feature@clk_shot`<br>`ch_drive_reference: cyclic_phase@clk_facility`<br>`ch_flute_probe_array: complex_mode@clk_facility`<br>`ch_synthetic_oscillator: numerical_phase@clk_sim` |
| `SCPN-MIRROR-CORE` | `gas_dynamic_mirror`, `simple_magnetic_mirror`, `tandem_mirror` | `ch_diamagnetic_loop: bounded_feature@clk_shot`<br>`ch_drive_reference: cyclic_phase@clk_facility`<br>`ch_end_loss_array: bounded_feature@clk_shot`<br>`ch_flute_probe_array: complex_mode@clk_facility`<br>`ch_synthetic_oscillator: numerical_phase@clk_sim` |
| `SCPN-RFP-CORE` | `reversed_field_pinch` | `ch_flux_loop_set: bounded_feature@clk_shot`<br>`ch_rogowski_coil: bounded_feature@clk_shot`<br>`ch_synthetic_oscillator: numerical_phase@clk_sim`<br>`ch_toroidal_probe_array: complex_mode@clk_facility` |
| `SCPN-SPHEROMAK-CORE` | `spheromak` | `ch_flux_loop_set: bounded_feature@clk_shot`<br>`ch_surface_probe_array: complex_mode@clk_facility`<br>`ch_synthetic_oscillator: numerical_phase@clk_sim`<br>`ch_thomson_profiles: bounded_feature@clk_shot` |
| `SCPN-STELLARATOR-CORE` | `heliotron`, `stellarator`, `torsatron` | `ch_flux_loop_set: bounded_feature@clk_shot`<br>`ch_interferometer: bounded_feature@clk_shot`<br>`ch_mirnov_array: complex_mode@clk_facility`<br>`ch_synthetic_oscillator: numerical_phase@clk_sim`<br>`ch_thomson_profiles: bounded_feature@clk_shot` |

The exact refusal detail is:

```text
channels[] key mismatch: missing=['timing_uncertainty_s'], unknown=[]
```

Producer fix-forward must add the member to every channel, regenerate the
canonical plan and envelope digests, update the combined fixture and pinned
tests, and provide a new exact revision and reproducible package identity. SPO
must then replay the new bytes; it must not relax the schema or infer defaults.

## Machine-readable custody

The [digest-sealed status data](data/reactor_diagnostic_plan_portfolio_status.v1.json)
records all 20 full revisions, manifest and fixture SHA-256 values,
configurations, plans, custody paths, refusal codes, and affected channel IDs.
Its [Draft 2020-12 schema](../specs/reactor_diagnostic_plan_portfolio_status.schema.json)
enforces the 10/10 split and the review-only, non-actionable boundary.

This structural register complements the
[reactor configuration evidence coverage](reactor_configuration_evidence_coverage.md):
the latter tracks actual evidence and semantic ingress. A row can therefore
have an accepted diagnostic design plan while correctly remaining
`not_declared` and producerless for physical evidence.
