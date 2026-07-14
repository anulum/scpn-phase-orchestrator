<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# Real-Data Case Study: CHB-MIT Multi-Channel Kuramoto Pre-Ictal Audit

## Abstract

We compare the textbook multi-channel Kuramoto order parameter with three
adaptive or semi-adaptive variants on real annotated seizures from the CHB-MIT
Scalp EEG Database (subject `chb01`). The task is to separate the 5-minute
pre-ictal window from seizure-free interictal epochs at a matched 10 % false
alarm rate. A new **global top-k PLV** detector (mean-R over the 15 channels
with the highest mean phase-locking value to the mean field) is now the
strongest performer (mean detection rate 0.81, AUC 0.94), beating both the
simple mean-R baseline (mean detection rate 0.70, AUC 0.91) and the earlier
adaptive weightings. The original SNR+kurtosis adaptive variant lags
(mean detection rate 0.43, AUC 0.80), and the PLV-to-mean-field soft weighting
is a partial improvement (mean detection rate 0.60, AUC 0.84). The delta-band
variants remain uninformative. The result shows that the right form of adaptive
channel selection *can* improve on the textbook mean-R detector for this corpus,
but only when the selection is driven by cross-channel phase coherence and
restricted to a top-k subset rather than soft-weighting all channels.

**This within-subject advantage does not generalise.** A leave-one-subject-out
audit across `chb01`–`chb05` (see *Cross-subject generalisation* below), which
calibrates `k` only on the training subjects, finds top-k PLV beats mean-R on
just **1 / 5** held-out subjects, with both detectors averaging ≈ 0.50 AUC
(chance). The pre-ictal coherence signal is present only on `chb01` and, weakly,
`chb05`. The `chb01` headline is therefore **subject-specific, not a general
seizure detector**; the fixed-`k` advantage was an artefact of tuning `k` on the
evaluation subject.

## Artefacts

All sealed records, per-seizure summaries, and the aggregate comparison JSON are
committed under:

```text
examples/real_data/chb01_seizures_multichannel_kuramoto/
```

Run the audit yourself:

```bash
python bench/chbmit_multichannel_kuramoto.py DATA \
    examples/real_data/chb01_seizures_multichannel_kuramoto
```

where `DATA/` contains the raw `chb01_*.edf` files from PhysioNet.

## Headline results

| Detector | Band | Mean DR @ 10 % FA | Mean AUC | Geo-mean p | Beats chance |
|----------|------|------------------:|---------:|-----------:|-------------:|
| mean_kuramoto | 0.5–4 Hz | 0.016 | 0.379 | 0.9314 | 0 / 7 |
| adaptive_kuramoto | 0.5–4 Hz | 0.063 | 0.402 | 0.7052 | 0 / 7 |
| mean_kuramoto | 4–30 Hz | 0.698 | 0.908 | 0.0008 | 6 / 7 |
| adaptive_kuramoto | 4–30 Hz | 0.429 | 0.797 | 0.0118 | 4 / 7 |
| plv_kuramoto | 4–30 Hz | 0.603 | 0.840 | 0.0035 | 4 / 7 |
| **topk15_plv_kuramoto** | **4–30 Hz** | **0.810** | **0.940** | **0.0003** | **7 / 7** |

## Take-away

For seizure-EEG early-warning on `chb01`, the **top-k PLV global selection**
detector is now the preferred multi-channel Kuramoto variant. Selecting the 15
channels with the highest mean PLV to the mean field and computing the
unweighted mean-R over that subset improves both detection rate and AUC over
the textbook mean-R detector. The SNR+kurtosis and soft PLV weightings are
inferior on this corpus; the gain comes from *hard* channel selection driven by
cross-channel phase coherence, not from continuous quality weighting. **This
advantage is specific to `chb01` and does not transfer — see the cross-subject
audit below.**

## Cross-subject generalisation (leave-one-subject-out)

The `chb01` headline tunes `k` on the same subject it scores. To test whether
top-k PLV is a real detector rather than a per-subject artefact, we ran a
leave-one-subject-out audit on five subjects. For each held-out subject, `k` is
calibrated **only on the other four subjects** (the `k` maximising their mean
top-k PLV AUC), then top-k PLV and the mean-R baseline are scored on the
held-out subject's pre-ictal windows versus its own interictal nulls. No subject
influences its own `k`, so each figure is a genuine out-of-sample estimate.

| Held-out | Pre-ictal epochs | Calibrated `k` | top-k PLV AUC | mean-R AUC |
|----------|-----------------:|---------------:|--------------:|-----------:|
| chb01 | 63 | 23 | 0.908 | 0.908 |
| chb02 | 18 | 15 | 0.114 | 0.275 |
| chb03 | 63 | 23 | 0.309 | 0.309 |
| chb04 | 27 | 23 | 0.460 | 0.348 |
| chb05 | 45 | 23 | 0.707 | 0.707 |
| **mean** | | | **0.499** | **0.509** |

Out-of-sample, top-k PLV beats mean-R on **1 / 5** subjects — `chb04`, where both
detectors are below chance, so that "win" is between two failing detectors. For
four of five subjects the calibrated `k` is 23, which is all channels, i.e.
exactly the mean-R baseline: the training subjects do not support a beneficial
`k < 23`. The one time it does pick `k = 15` (`chb02`), it makes the held-out
result **worse** (0.11 vs 0.28). Mean out-of-sample AUC is ≈ 0.50 for both
detectors — indistinguishable from chance.

The deeper finding is about the signal, not the detector: pre-ictal Kuramoto
coherence rises only on `chb01` (AUC 0.91) and, weakly, `chb05` (0.71); on
`chb02`/`chb03`/`chb04` it is at or below chance. **Neither mean-R nor top-k PLV
is a viable subject-independent seizure predictor on this corpus.** The `chb01`
top-k advantage was `k`-tuning on the evaluation subject, not generalisation.

Reproduce:

```bash
python bench/chbmit_crosssubject_validation.py DATA \
    examples/real_data/chbmit_crosssubject_kuramoto \
    chb01 chb02 chb03 chb04 chb05
```

where `DATA/` holds each subject's `chbNN-summary.txt` and `chbNN_*.edf`.
Sealed under `examples/real_data/chbmit_crosssubject_kuramoto/`.

## Limitations

The single-subject `chb01` headline is real **within** `chb01` but is **not a
general seizure detector**, as the cross-subject audit above establishes:

- **Does not generalise across subjects.** Out-of-sample, top-k PLV matches or
  loses to mean-R on 4 / 5 subjects and both average ≈ 0.50 AUC (chance).
- **The fixed `k` advantage was tuning.** The winning `k = 15` was selected on
  the same `chb01` seizures it scored; under honest per-subject `k` calibration
  the advantage disappears (calibrated `k` collapses to 23 = mean-R).
- **The pre-ictal coherence signal itself is subject-specific.** It is strong on
  `chb01`, moderate on `chb05`, and absent (≤ chance) on `chb02`/`chb03`/`chb04`.

Any use of this detector must be per-patient calibrated, not deployed as a
subject-independent model. Cross-corpus validation (e.g. the CAP sleep database)
would further test whether even per-patient calibration transfers across domains.
