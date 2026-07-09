<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# CAP Sleep Database — Multi-Channel N3 vs Wake Detector Comparison

This directory holds sealed audit records for an honest comparison of three
slow-wave detectors on a small cross-subject panel from the PhysioNet CAP Sleep
Database.

A follow-up diagnostic study explains why the multi-channel delta-phase Kuramoto
detector wins on `n2` but collapses on the other recordings; see
`docs/studies/cap_kuramoto_diagnostic.md` and the committed
`cap_kuramoto_diagnostic.json`.

## Data

The panel contains four recordings:

| Recording | Condition | Wake epochs | N3 epochs |
|-----------|-----------|------------:|----------:|
| `n1`      | Control   | 39          | 321       |
| `n2`      | Control   | 142         | 197       |
| `brux2`   | Bruxism   | 127         | 289       |
| `narco2`  | Narcolepsy| 180         | 188       |

Raw `.edf` and `.txt` files are **citation-only and are not redistributed**.
Download them from:

```bash
for rec in n1 n2 brux2 narco2; do
  curl -L -o ${rec}.edf https://physionet.org/files/capslpdb/1.0.0/${rec}.edf
  curl -L -o ${rec}.txt https://physionet.org/files/capslpdb/1.0.0/${rec}.txt
done
```

Cite:

- Terrillon D, Mietus JE, Goldberger AL, Yuhas A, Cottrell GW. The CAP Sleep
  Database. *PhysioNet*, 2009.
- Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG,
  Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank, PhysioToolkit, and
  PhysioNet: components of a new research resource for complex physiologic
  signals. *Circulation* 101(23):e215–e220, 2000.

## Reproduction

Install SPO with the EDF ingestion extra:

```bash
pip install "scpn-phase-orchestrator[eeg]"
```

Run the batch script with the committed manifest:

```bash
python bench/cap_multichannel_n3_vs_wake.py \
  --manifest examples/real_data/cap_multichannel_staging/cap_multichannel_manifest.csv \
  examples/real_data/cap_multichannel_staging
```

The script is deterministic and reproduces the committed JSON artefacts in the
per-recording subdirectories and the aggregate comparison file.

## Detectors

| Detector | Description |
|----------|-------------|
| `normalized_delta_envelope` | Per-channel delta-band Hilbert envelope divided by broadband Hilbert envelope; epoch score is the mean across channels. |
| `multi_channel_delta_kuramoto` | Per-channel delta-band Hilbert phase; Kuramoto order parameter `R(t)` across channels at each sample; epoch score is the mean of `R(t)` over the epoch. |
| `snr_weighted_delta_kuramoto` | Per-channel delta-band Hilbert phase; weighted Kuramoto order parameter where channel weights are sqrt(delta-band SNR); epoch score is the mean of `R(t)` over the epoch. |

All three detectors are audited at the same matched false-alarm operating point
(`target_false_alarm = 0.10`) using 10 000 label permutations (seed 42).

## Results

See `cap_multichannel_aggregate.json` for the full cross-subject numerical
record. Per-recording sealed audits and summaries live in the `n1/`, `n2/`,
`brux2/`, and `narco2/` subdirectories.

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

## Scope and limitations

- **Small panel, review-only, offline.** This audit measures detector skill on
  four public recordings; it is not a clinical sleep-staging product.
- **Same-recording calibration.** The operating point is chosen on each
  recording's Wake epochs and validated by permutation; generalisation to other
  subjects, age groups, or sleep disorders is not claimed.
- **Channel selection is data-driven.** The script selects monopolar
  F3/F4/C3/C4/O1/O2 channels when present and falls back to bipolar derivations;
  the exact labels used per recording are recorded in the comparison JSON.
- **Honest comparison.** All three detectors share the same matched-false-alarm
  protocol and no post-hoc tuning is used to improve any detector.
