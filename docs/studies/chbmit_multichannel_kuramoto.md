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
cross-channel phase coherence, not from continuous quality weighting.
