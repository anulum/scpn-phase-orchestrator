# Reactor technology and diagnostic atlas

This atlas maps the strongest cited external technology and diagnostic evidence
for every reactor configuration in SPO's current registry. It covers all **34
registered configurations** across **9 confinement families**; it is deliberately broader
than tokamaks and includes closed- and open-field magnetic confinement,
self-magnetic pinches, inertial fusion, magneto-inertial fusion,
electrostatic systems, beam-target systems, fusion-fission hybrids,
lattice-confinement experiments, and muon-catalysed fusion.

The atlas answers two questions without conflating them:

1. What has an external facility, government programme, or peer-reviewed source
   reported for this exact reactor configuration?
2. What evidence would still be required before SPO could admit a physical
   observation, give a phase-bearing signal physical meaning, or expose a
   review result to CONTROL?

The research cutoff is **2026-09-04**. The machine-readable artifact contains
**37 primary sources** and exact source-to-configuration bindings. It is a
representative evidence map for the registered configurations, not an exhaustive
directory of every facility, diagnostic, experiment, company, or publication.

## Methodology

Each configuration was joined by exact identity to
`DEFAULT_REACTOR_REGISTRY` and the corresponding observability candidates. A
source qualifies only for the configuration it names or directly studies;
related topologies never inherit evidence. Facility-owner and government
programme material remains labelled as such, separately from peer-reviewed
literature. Lifecycle wording describes the cited source date, not a claim of
current operation beyond that source.

The external evidence rank records the strongest cited result:

| Rank | Meaning |
|---|---|
| `E5_integrated_fusion_observation` | An integrated device reports a fusion-product observation. This is not a power-plant or net-energy claim. |
| `E4_integrated_plasma_experiment` | An integrated plasma experiment and relevant diagnostics are documented, without an atlas-qualified integrated fusion observation. |
| `E3_component_or_driver_experiment` | A driver, target, liner, compression, or other essential subsystem experiment is documented. |
| `E2_engineering_or_facility_development` | Engineering or facility development is documented without a stronger qualifying experiment. |
| `E1_concept_or_simulation` | The cited evidence is a concept, design study, or simulation. |
| `E0_no_qualifying_source` | No qualifying source was captured by this version of the atlas. |

These ranks are **not** technology-readiness levels, investment advice,
economic rankings, reactor-readiness claims, SCPN implementation maturity,
signal admissibility, physical-phase qualification, CONTROL admission, or
actuation authority. The absence of a stronger rank means only that this atlas
does not cite stronger exact-configuration evidence.

## Coverage result

| Strongest external evidence | Configurations |
|---|---:|
| Integrated fusion observation (`E5`) | 9 |
| Integrated plasma experiment (`E4`) | 18 |
| Component or driver experiment (`E3`) | 5 |
| Concept or simulation (`E1`) | 2 |
| Engineering-only (`E2`) or no qualifying source (`E0`) | 0 |

The distribution shows why a single label such as "fusion reactor" is too
coarse for phase orchestration. The 34 configurations have different drivers,
clocks, observables, physical carriers, diagnostic operators, and reference
events. Even two configurations at the same external evidence rank can require
incompatible phase semantics.

## Configuration map

Source identifiers resolve in the [primary-source register](#primary-source-register).
The rank is external evidence only; every row still has
`admission_state=refused_no_producer_evidence` at SPO's physical-observation
boundary.

| Family | Configuration | Rank | Representative system or programme | Sources |
|---|---|---|---|---|
| `magnetic_closed` | `conventional_tokamak` | E5 | JET; ITER diagnostic programme | SRC-001, SRC-002 |
| `magnetic_closed` | `spherical_tokamak` | E4 | MAST Upgrade | SRC-003 |
| `magnetic_closed` | `stellarator` | E4 | Wendelstein 7-X | SRC-004 |
| `magnetic_closed` | `heliotron` | E4 | Large Helical Device | SRC-005 |
| `magnetic_closed` | `torsatron` | E4 | Advanced Toroidal Facility | SRC-006 |
| `magnetic_closed` | `reversed_field_pinch` | E4 | RFX-mod2 | SRC-007 |
| `magnetic_closed` | `spheromak` | E4 | Sustained Spheromak Physics Experiment | SRC-008 |
| `magnetic_closed` | `field_reversed_configuration` | E4 | C-2W Norman | SRC-009 |
| `magnetic_open` | `simple_magnetic_mirror` | E4 | Wisconsin HTS Axisymmetric Mirror | SRC-010 |
| `magnetic_open` | `gas_dynamic_mirror` | E4 | Gas Dynamic Trap | SRC-011 |
| `magnetic_open` | `tandem_mirror` | E4 | GAMMA 10 | SRC-012 |
| `magnetic_open` | `cusp` | E4 | WB-8 high-beta cusp experiment | SRC-013 |
| `magnetic_open` | `polywell` | E4 | WB-8 Polywell experiment | SRC-013 |
| `magnetic_open` | `levitated_dipole` | E4 | Levitated Dipole Experiment | SRC-014 |
| `self_magnetic` | `sheared_flow_z_pinch` | E5 | FuZE and FuZE-Q | SRC-015, SRC-016 |
| `self_magnetic` | `theta_pinch` | E4 | Scylla theta-pinch programme | SRC-017 |
| `self_magnetic` | `dense_plasma_focus` | E5 | Dense-plasma-focus neutron sources | SRC-018 |
| `self_magnetic` | `z_pinch` | E4 | Sandia Z machine | SRC-034 |
| `inertial` | `laser_icf_indirect_drive` | E5 | National Ignition Facility | SRC-019 |
| `inertial` | `laser_icf_direct_drive` | E4 | OMEGA Laser Facility | SRC-020 |
| `inertial` | `laser_icf_fast_or_shock_ignition` | E4 | OMEGA and fast-ignition campaigns | SRC-020, SRC-021 |
| `inertial` | `ion_beam_icf` | E3 | NDCX ion-beam target platform | SRC-022 |
| `inertial` | `pulsed_electron_beam_icf` | E3 | Electron Beam Fusion Accelerator programme | SRC-023 |
| `inertial` | `projectile_or_impact_icf` | E4 | Hypervelocity and gas-gun target platforms | SRC-024, SRC-025 |
| `magneto_inertial` | `maglif` | E5 | MagLIF on the Z facility | SRC-026 |
| `magneto_inertial` | `plasma_jet_mif` | E3 | Plasma Liner Experiment | SRC-027 |
| `magneto_inertial` | `mechanical_or_liquid_liner_mif` | E3 | Liquid-liner test beds | SRC-028 |
| `magneto_inertial` | `frc_compression_mif` | E3 | Trenta FRC compression prototype | SRC-029 |
| `electrostatic` | `gridded_iec` | E5 | Cylindrical IEC neutron source | SRC-030 |
| `beam_target` | `beam_target` | E5 | Accelerator-based 14 MeV neutron generator | SRC-031 |
| `beam_target` | `colliding_beam` | E1 | Colliding Beam Fusion Reactor proposal | SRC-032 |
| `hybrid` | `fusion_fission_hybrid` | E1 | FDS-EM conceptual design | SRC-033 |
| `extension` | `scpn.reactor_systems:lattice_confinement_fusion` | E5 | Bremsstrahlung-irradiated deuterated-metal experiment | SRC-035 |
| `extension` | `scpn.reactor_systems:muon_catalysed_fusion` | E5 | Muon-catalysed D-T experiments | SRC-036, SRC-037 |

The two extension rows report measured fusion-product observations only. The
lattice source does not establish a self-sustaining lattice reactor, and the
muon sources do not establish net energy gain. Both rows explicitly retain
`current_integrated_device` and `power_conversion_demonstration` as gaps.

## Diagnostic capability and gaps

The cited record includes configuration-specific capability claims such as
magnetic, density, temperature, spectroscopic, imaging, charged-particle,
electrical-waveform, shock-timing, compression-trajectory, beam, neutron, and
fusion-product measurements. A capability is marked `observed_integrated` only
when the cited source documents it on the representative integrated device.
`observed_component` remains a driver or subsystem result; `planned` and
`not_demonstrated_in_cited_source` do not become observations.

This diagnostic inventory is intentionally not exhaustive. More importantly,
a paper or facility page is not a producer payload. For every configuration,
SPO still lacks the complete byte-canonical evidence chain required at its
public boundary:

- a physical sample tied to the named phenomenon;
- an explicit frame, reference, origin, orientation, and clock epoch;
- the observation operator or calibration lineage;
- uncertainty, validity interval, quality, and provenance;
- the applicable observability gate and its measured result; and
- an exact producer-to-SPO adapter with immutable source and artifact identity.

The atlas separately records broader development gaps where applicable,
including a current integrated device, integrated fusion output, public
machine-readable calibrated data, reactor-environment diagnostic
qualification, power-conversion demonstration, and CONTROL admission. A gap is
not silently filled by a neighbouring configuration, a source abstract, or an
SPO design declaration.

## Relation to SCPN projects

The registry assigns the 34 configurations to 23 device-owner projects. The
diagnostic-plan portfolio covers 22 Reactor Systems device repositories; the
additional registry owner is SCPN-MIF-CORE. Adding SCPN-FUSION-CORE yields 24
upstream reactor projects, and SCPN-CONTROL is the separate 25th system
boundary. External sources establish context for those identities; they do not
change the exact-project [occurrence ledger](reactor_signal_occurrence_ledger.md),
the [configuration evidence coverage](reactor_configuration_evidence_coverage.md),
or the [diagnostic-plan portfolio status](reactor_diagnostic_plan_portfolio_status.md).

SCPN-FUSION-CORE and SCPN-MIF-CORE remain owners of their exact producer facts.
SPO owns the semantic qualification and refusal rules. SCPN-CONTROL may consume
only a separately qualified, byte-canonical SPO review contract; it must not
convert this literature atlas into a measurement, regime decision, or command.

## Safety and authority boundary

All 34 rows are `review_only`, `actionable=false`,
`direct_actuation_authorized=false`, and
`machine_protection_final_veto=true`. All 34 have zero admitted physical
observations, zero qualified physical phases, and zero CONTROL admissions.
External technology evidence therefore cannot authorize an action, bypass an
independent protection system, or weaken a device-owner interlock.

## Machine-readable custody

The canonical artifact is
[`reactor_technology_diagnostic_atlas.v1.json`](data/reactor_technology_diagnostic_atlas.v1.json).
It is validated by
[`reactor_technology_diagnostic_atlas.schema.json`](../specs/reactor_technology_diagnostic_atlas.schema.json)
and sealed over canonical JSON payload bytes.

- Schema: `scpn-phase-orchestrator.reactor-technology-diagnostic-atlas.v1`
- Schema version: `1.1.0`
- Payload SHA-256: `2b239e1a3a81fde2091886d68e84894d46458c1a7be8155ce796a05479572da9`
- Configuration registry: `1.1.0` / `6741f25892d81b24aa621ee4f56b5e785e8323eca6ccf9d9009ce2c8e53f4912`
- Observability registry: `1.1.0` / `0aaf9bc7234113bedb98de51f2acd124a21da579e4d1ab1234e5b30ebc7880e0`

Any source, rank, configuration binding, capability status, missing-evidence
field, or authority change alters the payload seal and requires deliberate
review.

## Primary-source register

1. **SRC-001:** UKAEA, [JET final DT high-fusion-power scenario](https://scientific-publications.ukaea.uk/papers/insights-of-the-jet-high-fusion-power-scenario-in-the-final-dt-campaign/) (2025).
2. **SRC-002:** ITER Organization, [ITER diagnostics](https://www.iter.org/machine/supporting-systems/diagnostics) (2023).
3. **SRC-003:** UKAEA, [MAST Upgrade Research Plan](https://ccfe.ukaea.uk/wp-content/uploads/2019/12/MAST-U_RP_2019_v1.pdf) (2019).
4. **SRC-004:** Max Planck Institute for Plasma Physics, [Wendelstein 7-X diagnostics](https://www.ipp.mpg.de/3812950/diagnostik) (2024).
5. **SRC-005:** [LHD diagnostics and data acquisition](https://doi.org/10.1016/S0920-3796(00)00121-6) (2000).
6. **SRC-006:** Oak Ridge National Laboratory, [ORNL fusion history](https://www.ornl.gov/igniting-innovation-ornl-fusion-history) (2025).
7. **SRC-007:** [RFX-mod2 diagnostic capability enhancements](https://doi.org/10.1088/1741-4326/ad490a) (2024).
8. **SRC-008:** [Neutral-particle analysis on SSPX](https://doi.org/10.1063/1.2737756) (2007).
9. **SRC-009:** [C-2W integrated diagnostic suite](https://doi.org/10.1063/5.0043807) (2021).
10. **SRC-010:** University of Wisconsin-Madison, [Wisconsin HTS Axisymmetric Mirror](https://wham.physics.wisc.edu/) (2024).
11. **SRC-011:** [Gas Dynamic Trap confinement and stability](https://doi.org/10.1585/pfr.14.2402030) (2019).
12. **SRC-012:** [GAMMA 10 VUV and soft-X-ray diagnostics](https://doi.org/10.1016/0368-2048(96)02974-X) (1996).
13. **SRC-013:** [High-energy electron confinement in a magnetic cusp](https://doi.org/10.1103/PhysRevX.5.021024) (2015).
14. **SRC-014:** [Levitated Dipole Experiment microwave interferometer](https://doi.org/10.1063/1.3095684) (2009).
15. **SRC-015:** [Sustained neutron production from a sheared-flow Z pinch](https://doi.org/10.1103/PhysRevLett.122.135001) (2019).
16. **SRC-016:** [Extreme-ultraviolet spectroscopy on a sheared-flow Z pinch](https://pubmed.ncbi.nlm.nih.gov/38065162/) (2023).
17. **SRC-017:** [X-ray crystal spectroscopy of a theta-pinch plasma](https://doi.org/10.1103/PhysRev.131.1891) (1963).
18. **SRC-018:** [Dense plasma focus as a neutron source](https://doi.org/10.1016/0029-554X(77)90569-9) (1977).
19. **SRC-019:** [Diagnosing inertial-confinement-fusion ignition](https://doi.org/10.1088/1741-4326/ad703b) (2024).
20. **SRC-020:** Laboratory for Laser Energetics, [OMEGA diagnostics](https://www.lle.rochester.edu/diagnostics/) (2026).
21. **SRC-021:** [Diagnostics for fast-ignition science](https://doi.org/10.1063/1.2978199) (2008).
22. **SRC-022:** [Ion-beam high-energy-density diagnostics](https://doi.org/10.1063/1.3479112) (2010).
23. **SRC-023:** Sandia National Laboratories, [Particle-beam fusion research](https://www.sandia.gov/research/publications/details/particle-beam-fusion-research-at-sandia-national-laboratories-1978-12-31/) (1978).
24. **SRC-024:** [Hypervelocity impact facility for shock compression](https://doi.org/10.1016/j.proeng.2017.09.756) (2017).
25. **SRC-025:** UKAEA, [review of gas-gun-driven target experiments](https://firstlightfusion.com/science-hub/review-of-first-light-fusion-ltds-experimental-report-validate-production-of-neutrons-from-gas-gun-driven-targets/) (2022).
26. **SRC-026:** [MagLIF performance scaling](https://doi.org/10.1103/PhysRevLett.125.155002) (2020).
27. **SRC-027:** [Formation of a plasma liner for plasma-jet MIF](https://doi.org/10.1063/5.0204213) (2024).
28. **SRC-028:** [Rotating liquid-liner shape manipulation](https://doi.org/10.1016/j.fusengdes.2023.114087) (2024).
29. **SRC-029:** Helion Energy, [Trenta FRC prototype report](https://www.helionenergy.com/wordpress/uploads/2021/06/fusion-scientific-breakthroughts-helion-62221-converted.pdf) (2021).
30. **SRC-030:** [Neutron imaging with an IEC fusion source](https://doi.org/10.1364/AO.447180) (2022).
31. **SRC-031:** [Commissioning an accelerator-based 14 MeV neutron generator](https://doi.org/10.1016/j.fusengdes.2025.115158) (2025).
32. **SRC-032:** [Colliding Beam Fusion Reactor](https://doi.org/10.1126/science.278.5342.1419) (1997).
33. **SRC-033:** IAEA, [fusion-fission hybrid reactor design study](https://www-pub.iaea.org/MTCD/Meetings/FEC2008/ft_p3-21.pdf) (2008).
34. **SRC-034:** Sandia National Laboratories, [Z machine history](https://www.sandia.gov/labnews/2021/09/24/look-whos-turning-25/) (2021).
35. **SRC-035:** [Novel nuclear reactions observed in bremsstrahlung-irradiated deuterated metals](https://doi.org/10.1103/PhysRevC.101.044610) (2020).
36. **SRC-036:** [Experimental investigation of muon-catalyzed fusion](https://doi.org/10.1103/PhysRevLett.51.1757) (1983).
37. **SRC-037:** [Temperature-dependent muon-catalyzed fusion in solid deuterium-tritium mixtures](https://doi.org/10.1103/PhysRevLett.90.043401) (2003).
