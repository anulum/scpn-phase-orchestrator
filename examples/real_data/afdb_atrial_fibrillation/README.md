# MIT-BIH AFDB — real cardiac-ECG early-warning evidence

This directory holds the **second domain** the SCPN Phase Orchestrator's
early-warning design was proven on: the *same* three-member detector suite
(critical slowing down, rising synchronisation, ordinal-transition entropy) and
the *same* matched-false-alarm harness that screened scalp-EEG seizures here
screen the onset of **atrial fibrillation (AF)** in the surface ECG, through
nothing but a different adapter. Every detector's alarm — or silence — is sealed
into a hash-addressed, claim-bounded `EarlyWarningEvidence` record.

It exists to show the design is genuinely **domain-adaptable** (one suite, one
harness, a per-domain adapter), and to record plainly what the commodity detectors
do and do **not** show on real AF onsets.

## The result, stated honestly

At a matched false-alarm rate (≤ 10 % over 20 sinus-rhythm null trials),
**detection is sparse and no detector shows a robust early-warning advantage**:

| Detector | AF onsets led | Median lead |
|----------|--------------:|------------:|
| critical_slowing_down | 1 / 6 | 529 s |
| synchronisation | 2 / 6 | 134 s |
| transition_entropy | 0 / 6 | — |
| ensemble_weighted (fusion) | 2 / 6 | 377 s |

Two records are led — `04043` (rising synchronisation + fusion) and `04908`
(critical slowing down + rising synchronisation + fusion); the other four are
sealed silences. The fusion leads **no more** onsets than its best single member
(synchronisation, 2 / 6), so there is no robust fusion advantage, and a lead on
two onsets is not evidence of a reliable precursor.

This mirrors the scalp-EEG capstone's finding: on real data, at an honest
operating point, the detection is a **commodity**. The deliverable is the
**auditable, reproducible, claim-bounded sealed evidence** — including the sealed
silences — not the lead, and not a claim that the suite predicts AF.

A label-permutation significance test (10 000 relabellings, seed 0, recorded in the
aggregate as `permutation_significance`) confirms this: the best members —
synchronisation and the fusion — each lead 2 / 6 against 0.9 expected by chance at the
matched false-alarm rate, **p ≈ 0.22**, not significant. No detector beats chance.

### Two honest caveats specific to this domain

- **Thin oscillator population.** AFDB carries only two ECG leads, so the
  cross-`node` Kuramoto order parameter is a two-lead inter-lead phase coherence —
  a genuine but thin population next to the 23-channel scalp-EEG field. The
  synchrony detector's power is correspondingly limited here.
- **Opposite transition direction.** A seizure onset is a synchronisation *rise*;
  an AF onset is a *loss* of organised atrial activity — a desynchronisation. The
  suite's rising-synchronisation member is therefore not the natural fit for AF
  (critical slowing down, a direction-agnostic variance rise, is), and the result
  is reported as-is rather than tuned to flatter the suite.

## The source data (not included here)

The raw two-lead ECG recordings are **citation-only** and are **not
redistributed** in this repository. Obtain them directly from PhysioNet:

- Database: *MIT-BIH Atrial Fibrillation Database*
  (<https://physionet.org/content/afdb/1.0.0/>). Download the record files
  (`<record>.hea` / `.dat` / `.atr`).
- Cite as requested on the dataset page:
  - G. B. Moody and R. G. Mark, *A new method for detecting atrial fibrillation
    using R-R intervals*, Computers in Cardiology 10:227–230, 1983.
  - A. L. Goldberger, L. A. N. Amaral, L. Glass, J. M. Hausdorff, P. C. Ivanov,
    R. G. Mark, J. E. Mietus, G. B. Moody, C.-K. Peng, H. E. Stanley,
    *PhysioBank, PhysioToolkit, and PhysioNet*, Circulation 101(23):e215–e220,
    2000.

The AF onsets are the `(AFIB` rhythm-change annotations in each record's `atr`
stream. The evaluated AF records are `04043`, `04048`, `04746`, `04908`, `05091`,
`07879` (each scored on its first onset with a full clean pre-onset sinus
segment); the false-alarm null is the longest sinus stretch of `04015` and
`04126`.

## Methodology

- **Observable pipeline.** Each recording is band-passed 5–20 Hz (zero-phase
  Butterworth), turned into a per-lead Hilbert analytic phase, and decimated
  250 → 50 Hz by decimating the continuous `sin φ` / `cos φ` and reconstructing the
  phase with `atan2` (a wrapped phase is never low-pass filtered). All three
  members read that one decimated field. Window 200, step 25 (4 s / 0.5 s).
- **Fixed pre-onset segment.** Each onset is scored on the 900 s segment ending at
  onset: a 300 s leading sinus baseline (guaranteed AF-free) then a 600 s detection
  horizon. The onset sits at the segment end, so every window is pre-onset and any
  alarm is a genuine lead. An onset without a full clean pre-onset segment is
  **excluded**, not counted as a silent null.
- **Matched false alarm.** The longest sinus stretch of each null record is cut
  into non-overlapping 900 s trials (20 trials total); every detector's threshold
  is set continuously to the tightest value holding the trial false-alarm rate at
  or below 10 % (the quantile of the null alarm scores, with no grid ceiling). The
  calibrated robust-z thresholds were ≈ critical slowing down 5.7, rising
  synchronisation 4.2, transition entropy 2.7, fusion 2.3, and each detector's
  achieved false-alarm rate is recorded in the aggregate.
- **Sealing.** Each evaluated onset yields four `EarlyWarningEvidence` records (one
  per detector), including a sealed silence where a detector did not fire. Each
  record's `content_hash` is a canonical-JSON SHA-256, the same seal the
  assurance-case bundle and the NERC PRC evidence use.

## Reproducing the sealed evidence

With the raw AFDB records in a directory `DATA/` and the optional `cardiac` extra
installed (`pip install -e .[cardiac]`, `wfdb`), the shipped module regenerates
the committed artefact (the pipeline and detectors are deterministic — no
randomness — so a fresh run reproduces the sealed records byte for byte):

```bash
python bench/early_warning_leadtime_cardiac.py DATA OUT
```

`OUT/` then contains the six `<record>_early_warning_evidence.json` records and
`early_warning_leadtime_cardiac_results.json`, identical to the files committed
here. The raw ECG is read but never copied; only the derived sealed records are.

## Scope and limits (what this is not)

- **Review-only, offline.** The `EarlyWarningEvidence` disclaimer applies: this is
  a technical evidence-mapping artefact, not a clinical, operational, or safety
  decision, nor a certification. It never actuates.
- **Six onsets, two leads.** It shows the shipped suite on six AFDB onsets through
  a two-lead adapter; it is not a population study or an AF-prediction benchmark.
- **Sparse detection.** Two of six evaluated onsets are led at matched false alarm,
  with no robust fusion advantage. The honest reading is that detection is a
  commodity, which is exactly why the auditable sealed evidence — not the lead — is
  the deliverable, and why the same conclusion holds across two independent
  physiological domains.
```
