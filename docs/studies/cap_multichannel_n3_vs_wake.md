<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# Real-Data Case Study: CAP Multi-Channel N3 vs Wake

## Abstract

We extend the honest sleep-staging audit to a true multi-channel EEG corpus, the
PhysioNet CAP Sleep Database. On a four-recording panel — two controls plus
bruxism and narcolepsy — we compare three N3 slow-wave detectors at the same
matched false-alarm operating point: a normalized delta-band Hilbert envelope
averaged across channels, a multi-channel delta-phase Kuramoto order parameter,
and an SNR-weighted variant of the Kuramoto detector. All three detectors are
audited with label-permutation significance tests and sealed into
content-addressed records. The comparison is guarded by an integrity test that
needs only the committed artefacts, not the raw recordings.

## The question

Sleep-EDF provides only two EEG channels, which limits the spatial phase
coherence questions that SPO's Kuramoto machinery is designed for. The CAP
Sleep Database provides multiple EEG derivations, so we can ask: when more
channels are available, does a spatial delta-phase coherence detector separate
N3 from Wake better than, worse than, or comparably to a simple delta-band
amplitude envelope? Because the simple mean-R Kuramoto detector collapses on
all recordings except `n2`, we also test a diagnostic recommendation: an
SNR-weighted Kuramoto variant that weights each channel by the square root of
its local delta-band SNR. We answer these questions honestly at a fixed
false-alarm budget calibrated on Wake epochs.

## Data

Four recordings from the PhysioNet CAP Sleep Database:

| Recording | Condition | Wake epochs | N3 epochs |
|-----------|-----------|------------:|----------:|
| `n1`      | Control   | 39          | 321       |
| `n2`      | Control   | 142         | 197       |
| `brux2`   | Bruxism   | 127         | 289       |
| `narco2`  | Narcolepsy| 180         | 188       |

Raw files are **citation-only and are not redistributed**. Cite:

- Terrillon D, Mietus JE, Goldberger AL, Yuhas A, Cottrell GW. The CAP Sleep
  Database. *PhysioNet*, 2009.
- Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG,
  Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank, PhysioToolkit, and
  PhysioNet: components of a new research resource for complex physiologic
  signals. *Circulation* 101(23):e215–e220, 2000.

## Methods

### Channel selection

The script selects EEG channels by substring match, preferring monopolar
F3/F4/C3/C4/O1/O2 labels and falling back to bipolar 10–20 derivations. At
least three channels are required. All selected channels are resampled to a
common 100 Hz analysis rate before feature extraction.

### Detector 1 — normalized delta envelope

For each channel the score is the mean delta-band Hilbert envelope divided by
the mean broadband Hilbert envelope. The epoch score is the arithmetic mean of
this ratio across channels.

### Detector 2 — multi-channel delta-phase Kuramoto

For each channel the signal is band-pass filtered to 0.5–4 Hz and its Hilbert
phase is extracted. At every time sample the Kuramoto order parameter across
channels is

```
R(t) = | (1/C) sum_c exp(i phi_c(t)) |
```

and the epoch score is the mean of `R(t)` over the epoch. This measures the
spatial coherence of delta-band phase across the scalp.

### Detector 3 — SNR-weighted multi-channel delta-phase Kuramoto

For each channel the delta-band Hilbert phase is extracted as in detector 2.
Within each epoch the channel's weight is the square root of its local
delta-band SNR (delta power / total power). The weighted Kuramoto order
parameter is

```
R(t) = | sum_c w_c(t) exp(i phi_c(t)) | / sum_c w_c(t)
```

where `w_c(t)` is the per-channel, per-epoch SNR weight. The epoch score is the
mean of `R(t)` over the epoch. The sqrt softens the weighting so that a single
high-SNR channel cannot completely dominate the average.

### Audit protocol

- **Events:** expert-scored N3 epochs (`SLEEP-S3` and `SLEEP-S4`).
- **Nulls:** expert-scored Wake epochs (`SLEEP-S0`) from the same recording.
- **Calibration:** choose the smallest score threshold that holds the Wake
  false-alarm rate at or below 10 %.
- **Significance test:** 10 000 label permutations (fixed seed 42) of the pooled
  N3 + Wake alarm outcomes; one-sided p-value for the observed N3 alarm count.

The entire audit is performed by
`scpn_phase_orchestrator.evaluation.audit_detector` and sealed with
`seal_detector_audit`, producing a SHA-256 content hash over the corpus
provenance and verdict. The orchestration is built on the reusable
`bench/honest_dataset_audit.py` harness so the same protocol can be applied to
other datasets without rewriting the audit boilerplate.

## Results

See `examples/real_data/cap_multichannel_staging/cap_multichannel_aggregate.json`
for the full cross-subject numerical record. Per-recording sealed audits and
summaries live in the `n1/`, `n2/`, `brux2/`, and `narco2/` subdirectories.

### Cross-subject summary

| Quantity | Delta envelope | Multi-channel Kuramoto | SNR-weighted Kuramoto |
|----------|---------------:|-----------------------:|----------------------:|
| Mean N3 detection rate | 0.525 | 0.184 | 0.175 |
| Std. N3 detection rate | 0.360 | 0.070 | 0.075 |
| Mean achieved false alarm | 0.092 | 0.092 | 0.092 |
| Fraction beating chance | 3/4 (0.75) | 3/4 (0.75) | 3/4 (0.75) |
| Geometric-mean p-value | 0.001 | 0.015 | 0.016 |
| Recommendation | — | Do not refine | Do not refine |

### `n1` results (control)

| Quantity | Delta envelope | Multi-channel Kuramoto | SNR-weighted Kuramoto |
|----------|---------------:|-----------------------:|----------------------:|
| N3 epochs | 321 | 321 | 321 |
| Wake epochs | 39 | 39 | 39 |
| Mean N3 score | 0.762 | 0.629 | 0.629 |
| Mean Wake score | 0.702 | 0.604 | 0.605 |
| Matched threshold | 0.779 | 0.661 | 0.664 |
| Target false alarm | 0.100 | 0.100 | 0.100 |
| Achieved false alarm | 0.077 | 0.077 | 0.077 |
| N3 detection rate | 0.380 | 0.231 | 0.218 |
| Permutation p-value | < 0.001 | 0.017 | 0.025 |
| Beats chance (α = 0.05) | **yes** | **yes** | **yes** |

### `n2` results (control)

| Quantity | Delta envelope | Multi-channel Kuramoto | SNR-weighted Kuramoto |
|----------|---------------:|-----------------------:|----------------------:|
| N3 epochs | 197 | 197 | 197 |
| Wake epochs | 143 | 143 | 143 |
| Mean N3 score | 0.646 | 0.728 | 0.733 |
| Mean Wake score | 0.615 | 0.723 | 0.725 |
| Matched threshold | 0.815 | 0.791 | 0.792 |
| Target false alarm | 0.100 | 0.100 | 0.100 |
| Achieved false alarm | 0.098 | 0.098 | 0.098 |
| N3 detection rate | 0.005 | 0.223 | 0.218 |
| Permutation p-value | 1.000 | 0.002 | 0.002 |
| Beats chance (α = 0.05) | no | **yes** | **yes** |

### `brux2` results (bruxism)

| Quantity | Delta envelope | Multi-channel Kuramoto | SNR-weighted Kuramoto |
|----------|---------------:|-----------------------:|----------------------:|
| N3 epochs | 289 | 289 | 289 |
| Wake epochs | 127 | 127 | 127 |
| Mean N3 score | 0.757 | 0.687 | 0.687 |
| Mean Wake score | 0.465 | 0.683 | 0.686 |
| Matched threshold | 0.694 | 0.732 | 0.740 |
| Target false alarm | 0.100 | 0.100 | 0.100 |
| Achieved false alarm | 0.094 | 0.094 | 0.094 |
| N3 detection rate | 0.913 | 0.062 | 0.045 |
| Permutation p-value | < 0.001 | 0.913 | 0.984 |
| Beats chance (α = 0.05) | **yes** | no | no |

### `narco2` results (narcolepsy)

| Quantity | Delta envelope | Multi-channel Kuramoto | SNR-weighted Kuramoto |
|----------|---------------:|-----------------------:|----------------------:|
| N3 epochs | 188 | 188 | 188 |
| Wake epochs | 180 | 180 | 180 |
| Mean N3 score | 0.733 | 0.603 | 0.581 |
| Mean Wake score | 0.549 | 0.553 | 0.549 |
| Matched threshold | 0.658 | 0.760 | 0.759 |
| Target false alarm | 0.100 | 0.100 | 0.100 |
| Achieved false alarm | 0.100 | 0.100 | 0.100 |
| N3 detection rate | 0.803 | 0.218 | 0.218 |
| Permutation p-value | < 0.001 | 0.001 | 0.001 |
| Beats chance (α = 0.05) | **yes** | **yes** | **yes** |

## Recommendation

The SNR-weighted Kuramoto detector does not improve over the simple mean-R
Kuramoto detector on this panel. Its mean N3 detection rate is 0.175 versus
0.184 for the mean-R variant, and it is strictly worse on `n1` and `brux2`.
Consequently, further investment in this exact spatial-R feature is not
supported by the data. The normalized delta envelope remains the strongest
simple detector (mean detection rate 0.525), while the Kuramoto family requires
a different refinement direction — for example adaptive channel selection or a
temporal-stability criterion — if it is ever to catch the envelope.

## Reproduction

Install SPO with the EDF ingestion extra:

```bash
pip install "scpn-phase-orchestrator[eeg]"
```

Obtain the four `.edf`/`.txt` pairs from PhysioNet and run the batch script
with the committed manifest:

```bash
python bench/cap_multichannel_n3_vs_wake.py \
  --manifest examples/real_data/cap_multichannel_staging/cap_multichannel_manifest.csv \
  examples/real_data/cap_multichannel_staging
```

The script is deterministic, so regenerating the output reproduces the
committed sealed audit records and aggregate comparison JSON.

## Provenance and integrity

The sealed audit records live in
`examples/real_data/cap_multichannel_staging/`. Their content hashes are
recorded in the per-recording comparison JSON files.

`tests/test_cap_multichannel_staging_evidence.py` recomputes the content seals
from the recorded fields and pins the SHA-256 digests of the citation-only
source files, so the result can be guarded without redistributing the raw data.

## Scope and limitations

- **Review-only, offline.** This audit measures detector skill on four public
  recordings; it is not a clinical sleep-staging product.
- **Small panel.** The operating point is chosen on each recording's Wake null
  and validated by permutation; generalisation to other subjects, age groups, or
  sleep disorders is not claimed.
- **Channel selection is data-driven.** The exact labels used per recording are
  recorded in the comparison JSON.
- **Honest comparison.** All three detectors share the same matched-false-alarm
  protocol and no post-hoc tuning is used to improve any detector.

## Related work

- The Sleep-EDF case study (`sleep_staging_sleepedf.md`) introduced the honest
  N3-vs-Wake audit on a two-channel corpus.
- The early-warning matched-false-alarm study
  (`early_warning_matched_false_alarm.md`) introduced the sealed-evidence
  protocol used here.
