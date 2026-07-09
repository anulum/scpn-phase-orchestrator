<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Sleep-EDF N3 vs Wake real-data evidence -->

# Sleep-EDF N3 vs Wake Honest Audit

This directory contains the sealed derived evidence for a real-data case study
that audits a simple slow-wave detector on PhysioNet Sleep-EDF Expanded.

## Data source

- **Corpus:** PhysioNet Sleep-EDF Expanded, subject SC4001, first night.
- **Files used:**
  - `SC4001E0-PSG.edf` — polysomnographic recording (EEG Fpz-Cz at 100 Hz).
  - `SC4001EC-Hypnogram.edf` — expert hypnogram annotations.
- **Citation:**
  - Kemp B, Zwinderman AH, Tuk B, Kamphuisen HAC, Oberye JJL. Analysis of a
    sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the
    EEG. *IEEE-BME* 47(9):1185–1194, 2000.
  - Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG,
    Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank, PhysioToolkit, and
    PhysioNet: components of a new research resource for complex physiologic
    signals. *Circulation* 101(23):e215–e220, 2000.

The raw EDF files are **citation-only and are not redistributed here**. Obtain
them from the PhysioNet Sleep-EDF Expanded page and verify the SHA-256 digests
below if you need to reproduce the record byte-for-byte.

## Reproduction

Install SPO with the EDF ingestion extra:

```bash
pip install "scpn-phase-orchestrator[eeg]"
```

Download the two EDF files (e.g. into `DATA/`) and run:

```bash
python bench/sleep_staging_sleepedf.py \
  DATA/SC4001E0-PSG.edf \
  DATA/SC4001EC-Hypnogram.edf \
  examples/real_data/sleepedf_staging
```

The script is deterministic, so a fresh run reproduces the committed
`sleepedf_n3_vs_wake_audit.json` and `sleepedf_n3_vs_wake_summary.json`.

## Result

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
| Detection rate | 0.959 |
| Permutation p-value | < 0.001 |
| Beats chance (α = 0.05) | **yes** |

The detector is the normalized delta-band Hilbert envelope: the fraction of the
EEG's instantaneous broadband power that lives in the 0.5–4 Hz band. N3
(slow-wave sleep) is an amplitude-defined oscillatory phenomenon, and this score
clears the matched-false-alarm bar on the held-out Wake null epochs.

## Provenance

- Audit content hash: `836b9deda96455b31734c319cb3f30e87fb1ec005fe4a397a555619b57e690d0`
- Source PSG SHA-256: `2b40a18adf76af69a42d6db1f30f31d26b369f6d27ca0050ef30147ef892b131`
- Source hypnogram SHA-256: `a4cf67694ade1b52a0ddd06d5817fd45d2d3e8bac5302f640f3e9cfbbf12a996`

`tests/test_sleepedf_staging_evidence.py` recomputes the content seal and pins
the source digests without requiring the raw EDF files to be present.

## Scope and limits

- **Review-only, offline.** This is a detector-skill audit on one public
  recording, not a clinical sleep-staging system.
- **One subject, one night.** The result does not transfer to other recordings,
  age groups, or pathologies without a separate audit.
- **Not a phase-coherence claim.** A cross-band Kuramoto-R phase-coherence score
  was evaluated and did not separate N3 from Wake at the matched-false-alarm bar;
  the honest result comes from the delta-band amplitude envelope.
