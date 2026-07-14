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

We compare the textbook multi-channel Kuramoto order parameter with two
adaptive quality-weighted variants on real annotated seizures from the CHB-MIT
Scalp EEG Database (subject `chb01`). The task is to separate the 5-minute
pre-ictal window from seizure-free interictal epochs at a matched 10 % false
alarm rate. The simple mean-R detector in the 4–30 Hz band is the clear winner
(mean detection rate 0.70, AUC 0.91). The original SNR+kurtosis adaptive
variant lags (mean detection rate 0.43, AUC 0.80), and a new
PLV-to-mean-field weighting improves it (mean detection rate 0.60, AUC 0.84)
but still does not beat the unweighted mean-R. The delta-band variants are
uninformative. This is a concrete negative result for the adaptive refinement
on this corpus: neither SNR/kurtosis nor PLV-to-mean-field weighting is the
right default upgrade here.

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
| mean_kuramoto | 4–30 Hz | **0.698** | **0.908** | **0.0008** | **6 / 7** |
| adaptive_kuramoto | 4–30 Hz | 0.429 | 0.797 | 0.0118 | 4 / 7 |
| plv_kuramoto | 4–30 Hz | 0.603 | 0.840 | 0.0035 | 4 / 7 |

## Take-away

For seizure-EEG early-warning on `chb01`, prefer the simple mean-R Kuramoto in
the 4–30 Hz band. The SNR+kurtosis adaptive weights are not suitable here; the
PLV-to-mean-field weighting is a partial improvement but still leaves a gap to
the unweighted mean. Treat all weighting strategies as domain-specific
hyperparameters, not default upgrades.
