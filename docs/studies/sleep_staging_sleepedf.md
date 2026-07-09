<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# Real-Data Case Study: Sleep-EDF N3 vs Wake Honest Audit

## Abstract

We apply SPO's detector-agnostic auditor to a simple slow-wave detector on a
public polysomnogram from PhysioNet Sleep-EDF Expanded. The detector scores each
30-second epoch by the normalized delta-band Hilbert envelope and is asked to
separate expert-scored N3 (slow-wave sleep) epochs from Wake epochs at a matched
false-alarm rate. On subject SC4001, first night, the detector fires on 95.9 %
of N3 epochs while holding the Wake false-alarm rate at 10.0 %, with a
permutation p-value < 0.001. The result is sealed into a content-addressed audit
record and guarded by an integrity test that needs only the committed artefacts,
not the raw recording.

## The question

Sleep staging is a classic cyclic-state classification problem: the EEG
oscillator field moves through Wake → N1 → N2 → N3 → REM and back. A common
heuristic in SPO's exploratory sleep-staging script is that cross-band
Kuramoto-R phase coherence can distinguish stages. We ask a stricter,
operational question: at a fixed false-alarm budget calibrated on Wake epochs,
does a detector score N3 epochs higher than the matched false-alarm rate
explains? This is the same honest-evaluation moat used in the early-warning
study, applied to a sleep domain where the "transition" is the onset of
slow-wave sleep.

## Data

The recording is subject SC4001, first night, from the PhysioNet Sleep-EDF
Expanded corpus:

- `SC4001E0-PSG.edf` — polysomnogram at 100 Hz, including EEG Fpz-Cz.
- `SC4001EC-Hypnogram.edf` — expert sleep-stage annotations at 30-second
  resolution.

Raw files are **citation-only and are not redistributed**. Cite:

- Kemp B, Zwinderman AH, Tuk B, Kamphuisen HAC, Oberye JJL. Analysis of a
  sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the
  EEG. *IEEE Transactions on Biomedical Engineering* 47(9):1185–1194, 2000.
- Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG,
  Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank, PhysioToolkit, and
  PhysioNet: components of a new research resource for complex physiologic
  signals. *Circulation* 101(23):e215–e220, 2000.

## Methods

### Detector

The detector extracts the Fpz-Cz channel, applies a 0.5–4 Hz Butterworth
band-pass, computes the Hilbert envelope of both the delta-filtered signal and
the raw broadband signal, and returns, for each 30-second epoch:

```
score = mean(|H(delta_filter(eeg))|) / mean(|H(eeg)|)
```

This is the fraction of the EEG's instantaneous broadband power concentrated in
the delta band. N3 is defined clinically by high-amplitude delta slow waves, so
this is an amplitude-envelope detector rather than a phase-coherence detector.

### Why not Kuramoto R?

During development we evaluated the cross-band Kuramoto-R score used by the
exploratory `experiments/sleep_staging_eeg.py` script. On this recording it did
not separate N3 from Wake at the matched-false-alarm bar (p ≈ 0.7). The honest
result therefore uses the oscillator observable that actually carries the
signal: the slow-wave amplitude envelope.

### Audit protocol

- **Events:** 220 expert-scored N3 epochs.
- **Nulls:** 1997 expert-scored Wake epochs from the same recording.
- **Calibration:** choose the smallest score threshold that holds the Wake
  false-alarm rate at or below 10 %.
- **Significance test:** 10 000 label permutations (fixed seed 42) of the pooled
  N3 + Wake alarm outcomes; one-sided p-value for the observed N3 alarm count.

The entire audit is performed by `scpn_phase_orchestrator.evaluation.audit_detector`
and sealed with `seal_detector_audit`, producing a SHA-256 content hash over the
corpus provenance and verdict.

## Results

| Quantity | Value |
|----------|------:|
| Epochs analysed | 2650 |
| N3 epochs (events) | 220 |
| Wake epochs (null) | 1997 |
| Mean N3 score | 0.837 |
| Mean Wake score | 0.654 |
| Matched threshold | 0.753415 |
| Target false alarm | 0.100 |
| Achieved false alarm | 0.100 |
| N3 detection rate | 0.959 |
| Permutation p-value | < 0.001 |
| Beats chance (α = 0.05) | **yes** |

The normalized delta envelope cleanly separates the two classes at the
matched-false-alarm operating point.

## Reproduction

Install SPO with the EDF ingestion extra:

```bash
pip install "scpn-phase-orchestrator[eeg]"
```

Obtain the two EDF files from PhysioNet and run:

```bash
python bench/sleep_staging_sleepedf.py \
  SC4001E0-PSG.edf \
  SC4001EC-Hypnogram.edf \
  examples/real_data/sleepedf_staging
```

The script is deterministic, so regenerating the output reproduces the
committed `sleepedf_n3_vs_wake_audit.json` and
`sleepedf_n3_vs_wake_summary.json`.

## Provenance and integrity

The sealed audit record lives at
`examples/real_data/sleepedf_staging/sleepedf_n3_vs_wake_audit.json`. Its
`content_hash` is:

```
836b9deda96455b31734c319cb3f30e87fb1ec005fe4a397a555619b57e690d0
```

`tests/test_sleepedf_staging_evidence.py` recomputes the content seal from the
recorded fields and pins the SHA-256 digests of the citation-only source EDFs,
so the result can be guarded without redistributing the raw data.

## Scope and limits

- **Review-only, offline.** This audit measures detector skill on one public
  recording; it is not a clinical sleep-staging product.
- **One subject, one night.** The operating point is chosen on the same
  recording's Wake null and validated by permutation; generalisation to other
  subjects, age groups, or sleep disorders requires a separate audit.
- **Not a universal N3 detector.** The honest result is specific to this Fpz-Cz
  channel, this scoring function, and this recording.
- **Amplitude, not phase coherence.** The detector uses the delta-band Hilbert
  envelope because a cross-band phase-coherence score did not clear the bar on
  this corpus. The finding is that slow-wave sleep is amplitude-defined here,
  not that SPO's phase-coherence machinery is generally unsuited to sleep.

## Related work

- The early-warning matched-false-alarm study (`early_warning_matched_false_alarm.md`)
  introduced the honest-evaluation protocol used here.
- The ISO-NE forced-oscillation case study (`../validation/iso_ne_case1_forced_oscillation.md`)
  shows the same sealed-evidence pattern on a power-system disturbance.
- `experiments/sleep_staging_eeg.py` contains the exploratory cross-band-R
  heuristic that motivated this audit.
