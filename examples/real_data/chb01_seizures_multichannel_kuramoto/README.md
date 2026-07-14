<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# CHB-MIT chb01 — multi-channel Kuramoto pre-ictal audit

This directory holds an honest detector audit that asks: on real scalp EEG from
subject `chb01` of the CHB-MIT database, does a multi-channel Kuramoto order
parameter separate the 5-minute pre-ictal window from seizure-free interictal
epochs? And can an adaptive or semi-adaptive channel-selection variant improve
over the textbook mean-R Kuramoto?

## The result, stated honestly

In the **4–30 Hz seizure dynamics band**, a new **global top-k PLV** detector is
the strongest performer. It selects the 15 channels with the highest mean
phase-locking value to the mean field and computes the unweighted mean-R over
that subset:

| Detector | Mean detection rate @ 10 % FA | Mean AUC | Geo-mean p-value | Fraction beating chance |
|----------|------------------------------:|---------:|-----------------:|------------------------:|
| `mean_kuramoto_delta` (0.5–4 Hz) | 0.016 | 0.379 | 0.9314 | 0.00 |
| `adaptive_kuramoto_delta` (0.5–4 Hz) | 0.063 | 0.402 | 0.7052 | 0.00 |
| `mean_kuramoto_seizure` (4–30 Hz) | 0.698 | 0.908 | 0.0008 | 0.86 |
| `adaptive_kuramoto_seizure` (4–30 Hz) | 0.429 | 0.797 | 0.0118 | 0.57 |
| `plv_kuramoto_seizure` (4–30 Hz) | 0.603 | 0.840 | 0.0035 | 0.57 |
| `topk15_plv_kuramoto_seizure` (4–30 Hz) | **0.810** | **0.940** | **0.0003** | **1.00** |

The top-k PLV detector alarms on **7 / 7** seizures at the matched false-alarm
rate (calibrated to exactly 10 % on 600 interictal epochs), improving both mean
detection rate and AUC over the simple mean-R detector. The original adaptive
quality-weighted variant (SNR + kurtosis) and the soft PLV-to-mean-field
weighting are inferior on this corpus; the gain comes from *hard* global
selection of the most phase-coherent channels.

The delta-band variants remain uninformative for pre-ictal detection here.

## What this means for the codebase

The adaptive Kuramoto module now has a variant that beats the simple mean-R
detector on this corpus:

- Prefer `topk15_plv_kuramoto_seizure` for seizure-EEG early-warning tasks in
  the 4–30 Hz band on `chb01`.
- The SNR/kurtosis quality weight and soft PLV weighting are not suitable here;
  the improvement comes from top-k global channel selection driven by mean PLV.
- Treat all weighting strategies as domain-specific hyperparameters, not
  default upgrades.

## Methodology

- **Corpus.** PhysioNet CHB-MIT Scalp EEG Database, subject `chb01`.
- **Event class.** For each annotated seizure, the 30-second epochs that fully
  lie inside the 5-minute window immediately before the clinician-marked onset.
  This yields 9 pre-ictal epochs per seizure.
- **Null class.** All 30-second epochs from the seizure-free recordings
  `chb01_01`, `chb01_02`, `chb01_05`, `chb01_06`, `chb01_07` (600 epochs total).
- **Channels.** All 23 bipolar EEG derivations are used; no channel selection.
- **Preprocessing.** Raw recordings are anti-alias resampled from 256 Hz to
  64 Hz, then band-pass filtered (Butterworth, zero-phase) to the target band.
  Hilbert phases are extracted and Kuramoto order parameters are computed per
  epoch.
- **Detectors.**
  - `mean_kuramoto`: unweighted mean of `exp(i φ_c)` across channels, epoch
    score = mean of `R(t)`.
  - `adaptive_kuramoto`: per-channel SNR weights penalised by excess kurtosis,
    weighted Kuramoto order parameter, epoch score = median of `R(t)`.
  - `plv_kuramoto`: per-channel phase-locking value to the instantaneous mean
    field used as weights, weighted Kuramoto order parameter, epoch score =
    median of `R(t)`.
  - `topk15_plv_kuramoto`: global selection of the 15 channels with the highest
    mean PLV to the mean field, then unweighted mean-R over that subset, epoch
    score = mean of `R(t)`.
- **Audit protocol.** Matched false-alarm rate = 10 %, calibrated on the pooled
  null epochs. Significance tested with 10 000 label permutations (seed 42).
  Every detector's scores, thresholds, p-values, and AUCs are sealed into
  content-addressed JSON records.

## Reproducing the audit

1. Download the raw `chb01_*.edf` files and `chb01-summary.txt` from PhysioNet
   into `DATA/`.
2. Run:

   ```bash
   python bench/chbmit_multichannel_kuramoto.py DATA examples/real_data/chb01_seizures_multichannel_kuramoto
   ```

The script fetches the seizure annotations from PhysioNet, reads the local EDFs,
and regenerates the sealed records and aggregate JSON committed here.

## Source data citation

- A. H. Shoeb, *Application of Machine Learning to Epileptic Seizure Onset
  Detection and Treatment*, PhD thesis, Massachusetts Institute of Technology,
  2009.
- A. L. Goldberger et al., PhysioBank, PhysioToolkit, and PhysioNet,
  *Circulation* 101(23):e215–e220, 2000.

Raw EDF files are **citation-only and are not redistributed** in this
repository.

## Scope and limits

- Offline, retrospective, single-subject audit — not a clinical or operational
  system.
- The null class is drawn from separate interictal recordings, not from the
  same files, so the operating point is honest but does not test within-file
  false-alarm control.
- The 64 Hz analysis rate preserves both bands; the resampler's anti-alias
  filter removes energy above 32 Hz.
