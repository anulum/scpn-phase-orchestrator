# CHB-MIT chb01 — real scalp-EEG early-warning evidence

This directory holds the second **non-synthetic** sealed artefact the SCPN Phase
Orchestrator produced, and the empirical capstone of the early-warning detector
suite: the three-member suite (critical slowing down, rising synchronisation,
ordinal-transition entropy) and its weighted fusion run on real annotated seizures
from the CHB-MIT Scalp EEG Database, calibrated to a matched false-alarm rate, with
every detector's alarm — or silence — sealed into a hash-addressed, claim-bounded
`EarlyWarningEvidence` record.

It exists so an outsider can inspect a real, honest result without taking our word
for it, and to record plainly what the commodity detectors do and do **not** show
on real seizures.

## The result, stated honestly

At a matched false-alarm rate (≤ 10 % over 20 interictal null trials), **detection
is sparse and no detector shows a robust early-warning advantage**:

| Detector | Seizures led | Median lead |
|----------|-------------:|------------:|
| critical_slowing_down | 2 / 6 | 409 s |
| synchronisation | 1 / 6 | 442 s |
| transition_entropy | 0 / 6 | — |
| ensemble_weighted (fusion) | 1 / 6 | 452 s |

Two seizures are led: `chb01_04` (critical slowing down, rising synchronisation,
and the fusion, ≈ 7 min before onset) and `chb01_26` (critical slowing down only).
Transition entropy leads no seizure; **the fusion detects no more seizures than
critical slowing down alone** (its best single member), so there is no robust
fusion advantage. Leads on two of six seizures are not evidence of a reliable
precursor. The false-alarm threshold is set continuously to the exact 10 % target
(the quantile of the null alarm scores), not searched on a fixed grid, so the
matched operating point is exact and no detector is clipped.

This is the point of the programme, made concrete: on real data, at an honest
operating point, the detection is a **commodity** — the SCPN value is not "a better
detector" but the **auditable, reproducible, claim-bounded sealed evidence** here,
which records the sparse-detection result faithfully, including the sealed
*silences*.

A label-permutation significance test (10 000 relabellings, seed 0, recorded in the
aggregate as `permutation_significance`) confirms this: even critical slowing down,
the best member, leads 2 / 6 against 0.9 expected by chance at the matched false-alarm
rate — **p ≈ 0.22**, not significant. No detector beats chance, so the sparse detection
is exactly what the matched false-alarm rate produces.

## The source data (not included here)

The raw scalp-EEG recordings are **citation-only** and are **not redistributed**
in this repository. Obtain them directly from PhysioNet:

- Database: *CHB-MIT Scalp EEG Database*, subject `chb01`
  (<https://physionet.org/content/chbmit/1.0.0/>). Download the `chb01/` records
  and `chb01-summary.txt` (the annotated seizure onset times).
- Cite as requested on the dataset page:
  - A. H. Shoeb, *Application of Machine Learning to Epileptic Seizure Onset
    Detection and Treatment*, PhD thesis, Massachusetts Institute of Technology,
    2009.
  - A. L. Goldberger, L. A. N. Amaral, L. Glass, J. M. Hausdorff, P. C. Ivanov,
    R. G. Mark, J. E. Mietus, G. B. Moody, C.-K. Peng, H. E. Stanley,
    *PhysioBank, PhysioToolkit, and PhysioNet*, Circulation 101(23):e215–e220,
    2000.

The seizure onset times used here are the clinician annotations from
`chb01-summary.txt` (verified against that file): `chb01_03` 2996 s, `chb01_04`
1467 s, `chb01_15` 1732 s, `chb01_16` 1015 s, `chb01_18` 1720 s, `chb01_21`
327 s, `chb01_26` 1862 s. The interictal (seizure-free) null uses `chb01_01`,
`chb01_02`, `chb01_05`, `chb01_06`, `chb01_07`.

## Methodology

- **Observable pipeline.** Each recording is band-passed 4–30 Hz (zero-phase
  Butterworth), turned into a per-channel Hilbert analytic phase, and decimated
  256 → 32 Hz by decimating the continuous `sin φ` / `cos φ` and reconstructing
  the phase with `atan2` (a wrapped phase is never low-pass filtered). All three
  members read that one decimated field. Window 128, step 16 (4 s / 0.5 s).
- **Fixed pre-onset segment.** Each seizure is scored on the 900 s segment ending
  at onset: a 300 s leading baseline (guaranteed pre-ictal, so it cannot be
  contaminated by ictal samples) then a 600 s detection horizon. The onset sits at
  the segment end, so every window is pre-ictal and any alarm is a genuine lead.
  `chb01_21` (onset 327 s) is **excluded**, not counted as a silent null, because
  it is too early for a clean baseline.
- **Matched false alarm.** Each interictal recording is cut into non-overlapping
  900 s null trials of the same structure (20 trials total); every detector's
  threshold is set continuously to the tightest value holding the trial false-alarm
  rate at or below 10 % (the quantile of the null alarm scores, with no grid
  ceiling). The calibrated robust-z thresholds were ≈ critical slowing down 4.3,
  rising synchronisation 3.7, transition entropy 0.0, fusion 2.9, and each
  detector's achieved false-alarm rate is recorded in the aggregate.
- **Sealing.** Each evaluated seizure yields four `EarlyWarningEvidence` records
  (one per detector), including a sealed silence where a detector did not fire.
  Each record's `content_hash` is a canonical-JSON SHA-256, the same seal the
  assurance-case bundle and the NERC PRC evidence use.

## Reproducing the sealed evidence

With the raw `chb01_*.edf` records and `chb01-summary.txt` in a directory
`DATA/`, the shipped module regenerates the committed artefact (the pipeline and
detectors are deterministic — no randomness — so a fresh run reproduces the
sealed records byte for byte):

```bash
python bench/early_warning_leadtime_eeg.py DATA OUT
```

`OUT/` then contains the six `chb01_<id>_early_warning_evidence.json` records and
`early_warning_leadtime_eeg_results.json`, identical to the files committed here.
The raw EDF is read but never copied; only the derived sealed records are.

## Scope and limits (what this is not)

- **Review-only, offline.** The `EarlyWarningEvidence` disclaimer applies: this is
  a technical evidence-mapping artefact, not a clinical, operational, or safety
  decision, nor a certification. It never actuates.
- **One subject, one montage.** It shows the shipped suite on `chb01`'s seizures;
  it is not a population study or a seizure-prediction benchmark.
- **Sparse detection.** Two of six evaluated seizures are led at matched false
  alarm, with no robust fusion advantage; the leads are not evidence of a reliable
  precursor. The honest reading is that detection is a commodity, which is exactly
  why the auditable sealed evidence — not the lead — is the deliverable.
