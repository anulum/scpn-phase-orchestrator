<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# Sleep-EDF Transfer Test: Do the CAP Kuramoto Variants Generalise?

## Abstract

The [CAP variant panel](cap_kuramoto_variants.md) showed that time-resolved
amplitude gating rescues the delta-phase Kuramoto detector on a rich (6–8
channel) montage, and that `coherent_sustained_kuramoto` beats the delta
envelope there. This study tests whether that mechanism **transfers** to a
different corpus and a much sparser montage: PhysioNet Sleep-EDF Expanded, using
its two EEG derivations (Fpz-Cz and Pz-Oz) as a two-oscillator montage. The same
seven detectors and the same matched false-alarm protocol
(`target_false_alarm = 0.10`, 10 000-permutation test, seed 42) are applied to
recording `SC4001E0` (220 N3 epochs, 1997 Wake epochs).

The finding is clean and bounded: **amplitude gating is the only Kuramoto-family
mechanism that survives the transfer** — it beats chance where every
pure-coherence variant scores exactly zero — but on a two-channel montage its
detection rate stays far below the pure delta envelope. The coherence factor
adds information only when the montage is rich enough to carry spatial structure.

## Results

Recording `SC4001E0`, N3 vs Wake, matched FA 0.10:

| Detector | detection rate | achieved FA | p-value | beats chance |
|----------|---------------:|------------:|--------:|-------------:|
| `normalized_delta_envelope` | **0.995** | 0.100 | 0.0001 | yes |
| `amplitude_gated_delta_kuramoto` | 0.277 | 0.100 | 0.0001 | yes |
| `coherent_sustained_kuramoto` | 0.268 | 0.100 | 0.0001 | yes |
| `multi_channel_delta_kuramoto` | 0.000 | 0.100 | 1.0000 | no |
| `snr_weighted_delta_kuramoto` | 0.000 | 0.100 | 1.0000 | no |
| `sustained_delta_kuramoto` | 0.000 | 0.100 | 1.0000 | no |
| `adaptive_channel_kuramoto` | 0.000 | 0.100 | 1.0000 | no |

## Interpretation

- **Pure phase coherence carries no N3-vs-Wake signal on two channels.** The
  mean-R detector and every variant built on the unweighted order parameter
  (`snr_weighted`, `sustained`, `adaptive_channel`) score exactly 0.000 at
  matched FA (p = 1.0). With only two derivations, the delta-phase order
  parameter `R(t)` is dominated by shared reference and volume conduction and is
  roughly constant across sleep stages, so it cannot separate N3 from Wake.
  On two channels `adaptive_channel_kuramoto` is identical to mean-R — there is
  no channel subset to select.
- **Amplitude gating transfers as the surviving lever.**
  `amplitude_gated_delta_kuramoto` (0.277) and `coherent_sustained_kuramoto`
  (0.268) are the only Kuramoto-family detectors that beat chance
  (p = 0.0001). The amplitude term `A(t)` — the instantaneous delta envelope —
  is the discriminative signal, and gating preserves it even where the coherence
  factor is uninformative. This confirms that amplitude gating is a real,
  corpus-transferable mechanism rather than a CAP-specific artefact.
- **But its advantage is montage-dependent.** On CAP's rich montage the
  coherence factor varies with genuine spatial slow-wave structure, so
  `R(t)·A(t)` beats the envelope (0.552 vs 0.525). On Sleep-EDF's two channels
  the coherence factor is near-constant, so multiplying the clean amplitude
  signal by it only dilutes it — the gated variants (0.27) fall far below the
  pure envelope (0.995). The value of the coherence approach scales with the
  number of channels.

## Design rule

The two corpora together yield a concrete, evidence-grounded rule for a
regime-adaptive detector:

- **Sparse montage (≤ 2–3 channels):** use the delta envelope; the coherence
  factor cannot help and dilutes the amplitude signal.
- **Rich montage (≥ 6 channels):** use `coherent_sustained_kuramoto`; the
  amplitude-gated sustained order parameter exploits real spatial structure and
  beats the envelope.

The `n2` trade-off seen on CAP (pure phase coherence uniquely winning on a
low-SNR/high-coherence recording) is a second axis of the same regime split.

## Reproduction

```bash
PYTHONPATH=.:src python bench/sleepedf_kuramoto_variants.py \
  examples/real_data/sleepedf_kuramoto_variants
```

The script reads `SC4001E0-PSG.edf` and `SC4001EC-Hypnogram.edf` from
`scratchpad/sleepedf_data/`, loads both EEG derivations, audits all seven
detectors at matched FA 0.10 with a 10 000-permutation test, and writes sealed
audit records plus the aggregate comparison JSON. The committed evidence is
guarded by `tests/test_sleepedf_kuramoto_variants_evidence.py`.

## Scope and limitations

- **One recording.** Sleep-EDF `SC4001E0` is a single recording; the per-detector
  verdicts (a 10 000-permutation test on 220 N3 and 1997 Wake epochs) are the
  primary evidence, not a cross-subject average.
- **Two EEG channels only.** Sleep-EDF cassette recordings provide Fpz-Cz and
  Pz-Oz; this is the sparse end of the montage-richness axis by construction.
- **Raw EDF files are citation-only.** Only derived sealed records and the
  aggregate JSON are committed.

## Related work

- [`cap_kuramoto_variants.md`](cap_kuramoto_variants.md) — the rich-montage panel
  where amplitude-gated sustained coherence beats the envelope.
- [`sleep_staging_sleepedf.md`](sleep_staging_sleepedf.md) — the original
  single-channel Sleep-EDF envelope audit.
