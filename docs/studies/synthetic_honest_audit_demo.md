<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# Synthetic Honest-Audit Harness Demo

## Abstract

We demonstrate the reusable honest-audit harness (`bench/honest_dataset_audit.py`)
on a synthetic, fully reproducible corpus. Events are AR(1) windows with rising
autocorrelation; nulls are white-noise windows matched in mean and variance. Two
detectors — lag-1 autocorrelation and window mean — are audited at a matched
10 % false-alarm rate. The autocorrelation detector beats chance on every
recording, while the window-mean control does not, confirming that the harness
produces a fair, domain-agnostic comparison.

## The question

The CAP and Sleep-EDF studies show the harness on sleep EEG. Does the same
protocol work on a non-EEG, non-clinical domain with a clear event/null split?
This demo answers yes with a controlled corpus where ground truth and detector
skill are known by construction.

## Data

Three synthetic recordings generated deterministically from seeds 0, 1, and 2.
Each recording contains 40 event windows and 200 null windows.

| Recording | Seed | Event windows | Null windows |
|-----------|-----:|--------------:|-------------:|
| `run_a`   | 0    | 40            | 200          |
| `run_b`   | 1    | 40            | 200          |
| `run_c`   | 2    | 40            | 200          |

Events are AR(1) with `φ = 0.8`; nulls are AR(1) with `φ = 0.0`. Both are
centred to zero mean and scaled to unit marginal variance, so only the
autocorrelation structure differs.

## Methods

### Detectors

- `lag1_autocorrelation`: sample lag-1 autocorrelation of each window.
- `window_mean_control`: arithmetic mean of each window (no-skill control).

### Audit protocol

- **Events:** AR(1) windows (`φ = 0.8`).
- **Nulls:** white-noise windows (`φ = 0.0`).
- **Calibration:** smallest threshold holding the null false-alarm rate at or
  below 10 %.
- **Significance test:** 10 000 label permutations (fixed seed 42) of the pooled
  event + null alarm outcomes; one-sided p-value for the observed event alarm
  count.

## Results

See `examples/real_data/synthetic_honest_audit_demo/synthetic_honest_audit_demo.json`
for the full aggregate record.

### Cross-recording summary

| Quantity | Autocorrelation | Window mean |
|----------|----------------:|------------:|
| Mean detection rate | 1.000 | 0.000 |
| Std. detection rate | 0.000 | 0.000 |
| Mean achieved false alarm | 0.100 | 0.000 |
| Fraction beating chance | 3/3 (1.0) | 0/3 (0.0) |
| Geometric-mean p-value | < 0.001 | 1.000 |
| Recommendation | **Use / refine** | Do not refine |

## Reproduction

```bash
python bench/synthetic_honest_audit_demo.py \
  examples/real_data/synthetic_honest_audit_demo
```

## Scope and limitations

- **Synthetic only.** The corpus is generated for demonstration; no real-world
  inference is claimed.
- **Known skill.** The autocorrelation detector is expected to win because the
  event/null construction makes it win — the point is to verify the harness
  reports this honestly.
- **Template.** The loader / detector / label-extractor pattern can be reused
  for any domain that provides per-recording event/null labels and scores.
