<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# CAP Delta-Phase Kuramoto Variants: An Honest Detector Panel

## Abstract

The [Kuramoto diagnostic](cap_kuramoto_diagnostic.md) showed that the simple
mean-R delta-phase Kuramoto detector fails on CAP recordings such as `brux2`
because both N3 and Wake epochs carry a high *mean* order parameter — the mean
discards the N3-vs-Wake separation of `R(t)`, and a bruxism recording's Wake
epochs produce artefact-driven coherence with little genuine slow-wave power.
This study audits four new variants that each target that failure from a
different angle, alongside the three established detectors, on the same
four-recording CAP panel and at the same matched false-alarm operating point
(`target_false_alarm = 0.10`, 10 000-permutation significance test, seed 42).

The headline result: `coherent_sustained_kuramoto` is the **first
Kuramoto-family detector to beat the normalized delta envelope on this panel**
(mean detection rate 0.552 vs 0.525), and the decisive lever is *time-resolved
amplitude gating*, which raises the mean-R detector from 0.184 to 0.457 and
repairs the `brux2` failure (0.062 → 0.837).

## Variants

| Detector | Definition |
|----------|------------|
| `normalized_delta_envelope` | Delta-band Hilbert envelope / broadband envelope, averaged across channels (established baseline). |
| `multi_channel_delta_kuramoto` | Mean of `R(t)`, the across-channel delta-phase order parameter (established). |
| `snr_weighted_delta_kuramoto` | Per-channel, per-epoch SNR-weighted `R(t)` (established; shown not to improve on mean-R). |
| `amplitude_gated_delta_kuramoto` | Mean of `R(t)·A(t)`, where `A(t)` is the instantaneous mean delta amplitude — coherence contributes only where genuine slow-wave power exists. |
| `sustained_delta_kuramoto` | Lower quartile of `R(t)` within the epoch — the sustained coherence floor. |
| `adaptive_channel_kuramoto` | Mean `R(t)` over channels whose whole-recording delta SNR is at or above the median. |
| `coherent_sustained_kuramoto` | Lower quartile of `R(t)·A(t)` — sustained coherent slow-wave power (amplitude gate + sustained floor). |

## Results

### Cross-subject mean detection rate (matched FA 0.10)

| Detector | mean DR | std DR | geo-mean p | beats chance |
|----------|--------:|-------:|-----------:|-------------:|
| `coherent_sustained_kuramoto` | **0.552** | 0.339 | 0.0011 | 0.75 |
| `normalized_delta_envelope` | 0.525 | 0.360 | 0.0010 | 0.75 |
| `amplitude_gated_delta_kuramoto` | 0.457 | 0.287 | 0.0013 | 0.75 |
| `adaptive_channel_kuramoto` | 0.204 | — | — | 0.75 |
| `multi_channel_delta_kuramoto` | 0.184 | 0.070 | 0.0149 | 0.75 |
| `snr_weighted_delta_kuramoto` | 0.175 | — | — | 0.75 |
| `sustained_delta_kuramoto` | 0.173 | — | — | 0.75 |

### Per-recording detection rate

| Recording | Condition | envelope | mean-R | amplitude-gated | coherent-sustained |
|-----------|-----------|---------:|-------:|----------------:|-------------------:|
| `n1` | control | 0.380 | 0.231 | 0.386 | 0.377 |
| `n2` | control | 0.005 | **0.223** | 0.046 | 0.081 |
| `brux2` | bruxism | 0.913 | 0.062 | 0.837 | **0.893** |
| `narco2` | narcolepsy | 0.803 | 0.218 | 0.559 | **0.856** |

## Interpretation

- **Amplitude gating is the decisive lever.** Weighting `R(t)` by the
  instantaneous delta amplitude lifts the mean-R detector from 0.184 to 0.457.
  On `brux2` — where mean-R catastrophically fails (0.062) because bruxism Wake
  epochs are spuriously coherent — the gate recovers 0.837, close to the
  envelope's 0.913. This confirms the diagnostic hypothesis: mean-R counts
  coherence that carries no genuine slow-wave power.
- **The sustained floor helps only on top of the gate.** `sustained_delta_kuramoto`
  alone (0.173) is no better than mean-R (0.184); but the lower quartile of the
  *gated* series (`coherent_sustained`, 0.552) beats the amplitude gate alone
  (0.457) and the envelope (0.525). Requiring sustained coherent power, not a
  transient burst, is what pushes the Kuramoto family past the envelope.
- **`coherent_sustained_kuramoto` also beats the envelope on `narco2`** (0.856
  vs 0.803), not only on average.
- **Honest trade-off on `n2`.** `n2` is the one recording where pure phase
  coherence uniquely wins: mean-R scores 0.223 while the envelope collapses to
  0.005 and the gated variants score 0.046–0.081. `n2` has low delta SNR but
  high phase coherence, so amplitude gating suppresses exactly the signal that
  discriminates there. No single detector dominates every regime — this points
  to a regime-adaptive ensemble (pure phase coherence for low-SNR/high-coherence
  recordings, amplitude-gated sustained coherence otherwise) as the next step.
- **Channel selection and per-epoch SNR weighting are not the levers.**
  `adaptive_channel_kuramoto` (0.204) barely improves on mean-R, and
  `snr_weighted_delta_kuramoto` (0.175) does not improve at all. The
  discriminating structure is temporal (instantaneous amplitude, sustained
  floor), not which channels are used.

## Reproduction

```bash
PYTHONPATH=.:src python bench/cap_kuramoto_variants.py \
  examples/real_data/cap_kuramoto_variants
```

The script reads the four CAP recordings from `scratchpad/cap_data/`, audits all
seven detectors at matched FA 0.10 with a 10 000-permutation test, and writes
sealed audit records plus the aggregate comparison JSON. The committed evidence
is guarded by `tests/test_cap_kuramoto_variants_evidence.py`.

## Scope and limitations

- **Four recordings.** Cross-subject statistics on four recordings are
  descriptive, not powered; the per-recording verdicts (each a 10 000-permutation
  test on hundreds of epochs) are the primary evidence.
- **Raw EDF files are citation-only.** Only derived sealed records and the
  aggregate JSON are committed; the raw PhysioNet CAP recordings are not
  redistributed.
- **N3-vs-Wake only.** The audit separates deep sleep from wakefulness; it does
  not attempt full multi-stage classification.

## Related work

- [`cap_multichannel_n3_vs_wake.md`](cap_multichannel_n3_vs_wake.md) — the
  original three-detector honest audit.
- [`cap_kuramoto_diagnostic.md`](cap_kuramoto_diagnostic.md) — the diagnostic
  that motivated the amplitude-gated and sustained variants.
