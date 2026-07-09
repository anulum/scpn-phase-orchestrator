<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# Synthetic Honest-Audit Harness Demo

This directory is a minimal cross-domain proof of the reusable honest-audit
harness (`bench/honest_dataset_audit.py`). It contains no citation-only raw data;
everything is generated deterministically from a fixed seed, so the committed
artefacts can be regenerated exactly.

## Corpus

Three synthetic recordings (`run_a`, `run_b`, `run_c`). Each recording contains:

- **Events:** 40 AR(1) windows with coefficient `φ = 0.8` (critical slowing down).
- **Nulls:** 200 white-noise windows (`φ = 0.0`).

Both arms are forced to zero mean and unit marginal variance, so the window mean
carries no discriminative signal.

## Detectors

| Detector | Description |
|----------|-------------|
| `lag1_autocorrelation` | Lag-1 autocorrelation of each window — skilful on rising-autocorrelation events. |
| `window_mean_control` | Arithmetic mean of each window — no-skill control. |

## Results

| Quantity | Autocorrelation | Window mean |
|----------|----------------:|------------:|
| Mean detection rate | 1.000 | 0.000 |
| Mean achieved false alarm | 0.100 | 0.000 |
| Fraction beating chance | 3/3 (1.0) | 0/3 (0.0) |
| Geometric-mean p-value | < 0.001 | 1.000 |
| Recommendation | **Use / refine** | Do not refine |

The autocorrelation detector separates events from nulls at the matched 10 %
false-alarm rate on every recording; the window-mean control does not, exactly
as the construction demands.

## Reproduction

```bash
python bench/synthetic_honest_audit_demo.py \
  examples/real_data/synthetic_honest_audit_demo
```

The script regenerates the manifests, sealed audit records, summaries, and the
aggregate comparison JSON.

## Scope

This is a pedagogical / regression artefact, not a scientific claim. It shows
that the harness generalises to a non-EEG domain once a loader, detector
registry, and label extractor are provided.
