<!-- SPDX-FileCopyrightText: © Concepts 1996–2026 Miroslav Šotek. -->
<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->

# Beachhead Verticals

A beachhead is a small set of domains where the same core capability — detecting
and steering **phase (de)synchronisation** in a network of coupled oscillators —
maps directly onto an expensive, well-understood operational failure mode. The
three verticals below are the focus tracks: each ships lead domainpacks whose
binding specs validate, whose `run.py` scenarios execute end to end, and which
are covered by dedicated tests.

Every pack is a worked example, not a benchmarked product claim. Where a
quantified baseline-vs-orchestrated comparison exists, it lives on the linked
benchmark page; the commands here reproduce the scenario locally.

Run any pack two ways:

```bash
# Self-contained narrated scenario
python domainpacks/<pack>/run.py

# Through the CLI on the binding spec
spo validate domainpacks/<pack>/binding_spec.yaml
spo run domainpacks/<pack>/binding_spec.yaml --steps 500 --audit run.jsonl
```

---

## Industrial predictive maintenance

**Pain:** unplanned downtime and catastrophic failure in rotating equipment,
batch reactors, and production lines. The diagnostic signal is the *phase
relationship* between harmonics, stages, or stations — misalignment, bearing
defects, and process drift each have a phase signature before they breach an
amplitude limit.

| Pack | Layers | Oscillators | Channels | What it maps |
|------|-------:|------------:|----------|--------------|
| `rotating_machinery` | 4 | 10 | P/I | Shaft harmonics, blade dynamics, bearing defect frequencies, structural modes (ISO 10816-3 / API 670 boundaries) |
| `manufacturing_spc` | 3 | 9 | P/I/S | Sensor → machine → line coupling, bad-layer suppression under policy rules |
| `chemical_reactor` | 4 | 10 | P/I | Reaction kinetics, heat transfer, pressure vessel, feed flow with hard temperature/pressure limits |

**Validation:** all three validate, run end to end, and carry dedicated
domainpack tests (spec structure, layer map, full `main()` pipeline).

---

## Critical infrastructure

**Pain:** cascading instability in power grids and request-driven service
meshes. Generator swing-equation coherence and queue retry-storms are both
synchronisation phenomena — the control objective is to hold or break phase lock
before a cascade.

| Pack | Layers | Oscillators | Channels | What it maps |
|------|-------:|------------:|----------|--------------|
| `power_grid` | 5 | 12 | P/I | Inertial Kuramoto swing dynamics, generator-trip transients ([benchmark](power_grid_benchmark.md)) |
| `queuewaves` | 3 | 6 | P/I | Upstream/downstream queue phase-lock, retry-storm desync; ships a production server template |

**Validation:** the most mature vertical — `power_grid` carries five demo tests
and a benchmark page; `queuewaves` is a productised app with a server, pipeline,
and a full async/e2e test suite, plus a `queuewaves.production.yaml` deployment
template.

---

## Biosignal / clinical

**Pain:** rhythm and network-state disorders where coherence *is* the clinical
variable — arrhythmia, seizure dynamics, and sleep-stage architecture are all
read out from oscillatory phase coherence.

| Pack | Layers | Oscillators | Channels | What it maps |
|------|-------:|------------:|----------|--------------|
| `cardiac_rhythm` | 4 | 10 | P/I | SA node → conduction → depolarisation → repolarisation; sinus → PVC → VT → drug → pacing ([benchmark](cardiac_rhythm_benchmark.md)) |
| `neuroscience_eeg` | 6 | 14 | P/I | Delta/theta/alpha/beta/gamma bands plus a network layer; seizure and arousal boundaries |
| `sleep_architecture` | 4 | 8 | P/I/S | AASM sleep-stage architecture inferred from band order parameters |

**Validation:** this vertical previously had no dedicated tests; all three now
carry end-to-end domainpack suites (spec validation, layer structure, objective
sanity, boundaries/actuators, and full `main()` execution).

> Clinical use is gated on regulatory validation (CE/MDR, clinical studies).
> These packs are research and engineering references, not medical devices.

---

## Why these three

The same engine and binding-spec workflow covers all three, so a vertical slice
in one hardens the shared core for the others. They differ in go-to-market
profile: industrial has the lightest regulatory drag and the clearest ROI,
infrastructure is the most product-ready, and biosignal has the highest
differentiation but the heaviest validation burden. See the
[Domainpack Gallery](domainpack_gallery.md) for the full catalogue.
