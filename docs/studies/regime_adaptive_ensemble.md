<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# Regime-Adaptive Ensemble: An Honest Cross-Corpus Test

## Abstract

The [CAP variant panel](cap_kuramoto_variants.md) suggested that amplitude-gated
sustained coherence beats the delta envelope on a rich montage, and the
[Sleep-EDF transfer test](sleepedf_kuramoto_variants.md) showed the coherence
approach collapses on a sparse two-channel montage. Those two results motivated a
label-free **regime router**: choose the envelope on sparse montages and
`coherent_sustained_kuramoto` on rich ones, using only the observable channel
count. This study audits two such routers alongside every component detector on a
combined five-recording cross-corpus manifest (four CAP + one Sleep-EDF), at
matched false-alarm 0.10 with a 10 000-permutation test.

**The result is a negative one, reported as found: the channel-count router does
not beat the plain delta envelope.** The envelope is the most robust single
detector across the five recordings, and the coherence advantage seen on CAP
turns out to be recording-specific rather than a clean montage-size effect.

## Results

Per-recording detection rate (matched FA 0.10), with the montage router's route
in bold:

| Recording | channels | envelope | mean-R | coherent-sustained | montage router |
|-----------|---------:|---------:|-------:|-------------------:|---------------:|
| `n1` | 8 | 0.380 | 0.231 | 0.377 | **0.377** (coherent) |
| `n2` | 3 | 0.005 | 0.223 | 0.081 | **0.005** (envelope) |
| `brux2` | 6 | 0.913 | 0.062 | 0.893 | **0.893** (coherent) |
| `narco2` | 3 | 0.803 | 0.218 | 0.856 | **0.803** (envelope) |
| `SC4001E0` | 2 | 0.995 | 0.000 | 0.268 | **0.995** (envelope) |

Cross-corpus mean detection rate:

| Detector | mean DR |
|----------|--------:|
| `normalized_delta_envelope` | **0.619** |
| `regime_adaptive_montage` | 0.615 |
| `regime_adaptive_full` | 0.615 |
| `coherent_sustained_kuramoto` | 0.495 |
| `amplitude_gated_delta_kuramoto` | 0.421 |
| `adaptive_channel_kuramoto` | 0.163 |
| `multi_channel_delta_kuramoto` | 0.147 |
| `snr_weighted_delta_kuramoto` | 0.140 |
| `sustained_delta_kuramoto` | 0.139 |

## Interpretation

- **The plain delta envelope is the most robust single detector** (mean DR
  0.619). No detector and neither router exceeds it on the five-recording panel.
- **The channel-count router does not beat the envelope** (0.615 vs 0.619). It is
  bounded below the envelope and above `coherent_sustained` because it is a
  weighted mix of their per-recording scores. The router is a strict function of
  the envelope and coherent-sustained columns above.
- **The routing threshold is too crude, and for a revealing reason.** The router
  treats `n2` and `narco2` (three bipolar derivations) as sparse and sends them
  to the envelope. But on `narco2` the coherence detector is actually *better*
  (0.856 vs 0.803) — so the router leaves 0.053 of detection rate on the table
  there. Channel count is not a sufficient statistic for the regime.
- **The "montage richness" story from the CAP panel was too simple.**
  `coherent_sustained` does not uniformly win on rich montages: it *loses* on
  `brux2` (0.893 vs 0.913) and ties on `n1` (0.377 vs 0.380). Its CAP-panel
  advantage (0.552 vs 0.525) is driven almost entirely by `narco2` and by the
  near-zero-rate `n2`, not by montage size. The genuine, transferable finding is
  narrower than first framed: amplitude gating keeps the Kuramoto family alive
  where plain coherence dies (confirmed on both corpora), but it does not give a
  robust, montage-indexable advantage over the envelope.
- **The `n2` refinement is inert here.** `regime_adaptive_full` is identical to
  `regime_adaptive_montage` on this panel: `n2` is a three-channel recording, so
  the sparse-montage rule routes it to the envelope before the low-SNR /
  high-coherence check can fire. The in-sample `n2` axis therefore has no effect
  and remains unvalidated.

## What would be needed to beat the envelope

The evidence points to a regime signal finer than channel count — the recording
where coherence helps (`narco2`) and the recordings where it does not (`brux2`,
`n2`, Sleep-EDF) are not separated by montage size. A useful router would need a
label-free feature that predicts *when* spatial slow-wave coherence adds
information over amplitude alone, validated out-of-sample on more recordings than
this five-recording panel provides. Until then the honest recommendation is the
plain delta envelope, with amplitude-gated coherence reserved for montages and
recordings where it has been shown to help.

## Reproduction

```bash
PYTHONPATH=.:src python bench/regime_adaptive_ensemble.py \
  examples/real_data/regime_adaptive_ensemble
```

The script loads the four CAP recordings and the Sleep-EDF recording, audits all
seven component detectors plus the two routers at matched FA 0.10 with a
10 000-permutation test, and writes sealed audit records plus the aggregate
comparison JSON. The committed evidence is guarded by
`tests/test_regime_adaptive_ensemble_evidence.py`.

## Scope and limitations

- **Five recordings.** Four CAP + one Sleep-EDF is a small, heterogeneous panel;
  the cross-corpus means are descriptive, and the negative result is a finding
  about *this* panel and *this* router, not a proof that no router can help.
- **In-sample thresholds.** The router thresholds are read from the CAP
  diagnostic; the `n2` axis in particular is unvalidated and, as shown, inert
  here.
- **Raw EDF files are citation-only.** Only derived sealed records and the
  aggregate JSON are committed.

## Related work

- [`cap_kuramoto_variants.md`](cap_kuramoto_variants.md) — the rich-montage panel.
- [`sleepedf_kuramoto_variants.md`](sleepedf_kuramoto_variants.md) — the sparse-
  montage transfer test.
