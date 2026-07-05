# PSML grid — real power-grid early-warning evidence

This directory holds the **third domain** the SCPN Phase Orchestrator's
early-warning design was proven on: the *same* three-member detector suite
(critical slowing down, rising synchronisation, ordinal-transition entropy) and
the *same* matched-false-alarm harness that screened scalp-EEG seizures and
cardiac AF onsets here screen a **growing power-grid oscillation** — a
generator-trip-triggered inter-area mode whose amplitude climbs toward a
protective endpoint — through nothing but a grid adapter. Every detector's alarm
or silence is sealed into a hash-addressed, claim-bounded `EarlyWarningEvidence`
record.

Across three independent physical domains — brain, heart, grid — it is the same
suite and the same harness behind one per-domain adapter each, which is the point:
the design is genuinely **domain-adaptable**.

## The result, stated honestly

At a matched false-alarm rate (≤ 10 % over 24 damped-disturbance null trials),
**detection is sparse and no detector shows a robust early-warning advantage**:

| Detector | Instabilities led | Median lead |
|----------|------------------:|------------:|
| critical_slowing_down | 3 / 12 | 1 s |
| synchronisation | 0 / 12 | — |
| transition_entropy | 2 / 12 | 1 s |
| ensemble_weighted (fusion) | 2 / 12 | 1 s |

Critical slowing down — the classical variance / autocorrelation rise — leads the
most growing oscillations, exactly as its theory predicts of a declining-damping
instability; rising synchronisation leads none (the buses are electrically coupled
and already coherent, so their coherence *level* barely rises); and the fusion
leads **no more** instabilities than critical slowing down alone, so there is no
robust fusion advantage. The leads are short (≈ 1 s), on a sub-second analysis
grid.

The false-alarm threshold is set continuously from the null (the quantile of the
null alarm scores) rather than searched on a fixed grid, so a detector is matched
to the target exactly and is never silently clipped to a grid maximum. This
matters here: critical slowing down's matched threshold on the electrically-coupled
grid is very high (its damped-disturbance nulls are variance-heavy), so a grid that
stopped at a modest maximum would have *over-counted* its detections — the honest,
uncapped calibration is what yields the 3 / 12 above.

This mirrors the scalp-EEG and cardiac findings a third time: on real data, at an
honest operating point, the detection is a **commodity**. The deliverable is the
**auditable, reproducible, claim-bounded sealed evidence** — not the lead, and not
a claim that the suite predicts grid instability.

## The source data (not included here)

The raw co-simulation data is **citation-only** and is **not redistributed** here.
Obtain it directly from Zenodo:

- Dataset: *PSML: A Multi-scale Time-series Dataset for Machine Learning in
  Decarbonized Energy Grids* (<https://zenodo.org/record/5130612>,
  `PSML.zip`, 5.2 GB), published under **CC BY 4.0**. The millisecond-level
  transmission scenarios are under
  `PSML/Millisecond-level PMU Measurements/Natural Oscillation/row_*`.
- Cite as requested on the dataset page:
  - X. Zheng, N. Xu, L. Trinh, D. Wu, T. Huang, S. Sivaranjani, Y. Liu, L. Xie,
    *PSML: A Multi-scale Time-series Dataset for Machine Learning in Decarbonized
    Energy Grids*, NeurIPS Datasets and Benchmarks, 2021.

Each scenario's `info.csv` names the disturbance `type`, `start`, and `end`; each
`trans.csv` carries the 23 transmission-bus voltage magnitudes at ~238 Hz.

## Methodology

- **Transition and null.** A **transition** is a `gen_trip` scenario whose
  oscillation *grows* between the trip (`start`) and the annotated `end` (late-third
  cross-bus deviation ≥ 1.3× the early third); its **onset** is `end`, the
  instability endpoint the growing oscillation leads to. The **false-alarm null** is
  the *damped* `bus_fault` / `branch_trip` scenarios whose response does not grow.
  So the calibration asks the fair question — does the suite lead a *growing*
  instability more often than it false-alarms on a *stable* disturbance? 27 growing
  transitions and 187 damped nulls were found; the first 12 transitions were
  evaluated on 24 null trials.
- **Observable pipeline.** Each recording is band-passed 0.2–5 Hz (the
  electromechanical mode band, which also removes the DC operating point), turned
  into a per-bus Hilbert analytic phase, and read as the 23-bus order parameter.
  Window 0.5 s, step 0.1 s.
- **Fixed pre-onset segment.** Each transition is scored on the 2 s segment ending
  at `end`: a leading baseline (its first third) then a detection horizon, so the
  growing oscillation is the pre-onset precursor and any alarm is a genuine lead.
- **Matched false alarm.** Each detector's threshold is set continuously to the
  tightest value holding the trial false-alarm rate at or below 10 % — the quantile
  of the null alarm scores, with no grid ceiling. The calibrated robust-z
  thresholds were ≈ critical slowing down 480, rising synchronisation 3.3,
  transition entropy 3.7, fusion 157 (the two very high thresholds reflect
  variance-heavy damped-disturbance nulls, which a bounded grid would have clipped).
  Each detector's achieved false-alarm rate is recorded in the aggregate.
- **Sealing.** Each evaluated transition yields four `EarlyWarningEvidence` records
  (one per detector), including a sealed silence where a detector did not fire. Each
  record's `content_hash` is a canonical-JSON SHA-256, the same seal the
  assurance-case bundle and the NERC PRC evidence use.

## Reproducing the sealed evidence

With the PSML millisecond scenarios in a directory `DATA/`, the shipped module
regenerates the committed artefact (the pipeline and detectors are deterministic —
no randomness — so a fresh run reproduces the sealed records byte for byte):

```bash
python bench/early_warning_leadtime_grid.py DATA OUT
```

`OUT/` then contains the twelve `row_<n>_early_warning_evidence.json` records and
`early_warning_leadtime_grid_results.json`, identical to the files committed here.
The raw PSML data is read but never copied; only the derived sealed records are.

## Scope and limits (what this is not)

- **Review-only, offline.** The `EarlyWarningEvidence` disclaimer applies: this is a
  technical evidence-mapping artefact, not an operational, safety, or dispatch
  decision, nor a certification. It never actuates.
- **Simulated, one system.** PSML is a transmission-and-distribution co-simulation
  of one 23-bus system, not field PMU data; this shows the shipped suite on its
  growing-oscillation scenarios, not a fleet-wide instability benchmark.
- **Sparse detection.** The classical critical-slowing-down member leads the most
  instabilities, the fusion no more; the leads are short. The honest reading is that
  detection is a commodity, which is exactly why the auditable sealed evidence — not
  the lead — is the deliverable, and why the same conclusion holds across three
  independent physical domains.
