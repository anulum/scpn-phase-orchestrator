<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->

# arXiv submission package — the matched-false-alarm study

`01_early_warning_matched_false_alarm.tex` is the arXiv manuscript of
`docs/studies/early_warning_matched_false_alarm.md` (the canonical text;
the two must stay in sync — regenerate this manuscript when the study
changes). Every number binds to the hash-sealed artefacts under
`examples/real_data/`; the manuscript adds no claim the sealed records do
not carry.

Build: `pdflatex 01_early_warning_matched_false_alarm.tex` twice (plain TeX Live, no external packages
beyond the standard set; 17 pages letter).

## Proposed metadata (OWNER approves before submission)

- **Title:** Generic early-warning detection is at chance across five
  modalities; a domain-specific detector clears the bar where the
  signature is deterministic
- **Authors:** Miroslav Šotek (ANULUM / Fortis Studio)
- **Primary category:** `physics.data-an` (Data Analysis, Statistics and
  Probability)
- **Cross-lists (proposal):** `eess.SY` (grid detector + deployment
  operating point), `nlin.AO` (Kuramoto collective-coordinate results),
  `q-bio.QM` (DNB modality)
- **License (recommendation):** arXiv non-exclusive distribution license
  (default; retains all rights). CC licenses grant more than needed —
  owner decision.
- **Comments field:** "17 pages. All evidence records are hash-sealed
  and byte-reproducible; source and sealed artefacts at
  https://github.com/anulum/scpn-phase-orchestrator"

### Metadata abstract (trimmed to arXiv's field limit; PDF carries the full one)

> Generic early-warning signals (EWS) — rising variance and lag-one
> autocorrelation of critical slowing down, and related synchronisation
> and ordinal-entropy indicators — are widely reported to precede abrupt
> transitions in the brain, the heart, power systems and the climate,
> but most of that evidence rests on a retrospective per-record trend
> test. We ask the operational question instead: at a fixed false-alarm
> budget, does a detector fire on transitions more often than on
> no-transition controls? One domain-adaptable suite and one
> matched-false-alarm harness are applied unchanged to four labelled
> corpora (scalp-EEG seizures, cardiac AF onsets, power-grid growing
> oscillations, palaeoclimate transitions), with label-permutation
> significance. No generic detector reaches significance in any domain,
> and the canonical Dakos AR(1)-Kendall-tau detector fares no better on
> the identical segments — including 0 of 6 on its own palaeoclimate
> records. A dynamical-network-biomarker extension to single-cell and
> bulk transcriptomics reaches the same conclusion once the null is
> granted the analysis's own selection freedom. Where the domain carries
> a deterministic signature, a domain-specific detector clears the same
> bar decisively: a grid modal envelope-growth detector leads 36 of 90
> transitions (p = 0.0001; held-out 24 of 45, p = 0.0002) where every
> generic member is at chance. Pre-registered cross-dataset legs (real
> ISO-NE captures; the WECC 240-bus OSL corpus) show the certified
> numeric operating point does not port in either direction: the
> deployable unit is the frozen detector shape plus per-system
> matched-false-alarm calibration, sealed in hash-addressed,
> byte-reproducible evidence records, including every silence.

## Submission checklist (submission itself is OWNER-GATED)

1. Owner reviews the built PDF (build locally; the PDF is not committed).
2. Owner approves metadata above (title / categories / license /
   abstract / comments).
3. Upload `01_early_warning_matched_false_alarm.tex` as the source (arXiv compiles it; no figures, no
   .bbl needed — the bibliography is inline).
4. First-time `physics.data-an` submissions may require endorsement —
   arXiv states it during submission if so.
5. After the arXiv identifier exists: add it to `CITATION.cff`,
   `.zenodo.json`, the study page, and the README citation block; that
   update is a normal follow-up commit.

## Claim discipline

No superlatives; every quantitative statement in the manuscript is
carried by a sealed artefact committed in this repository. The negative
results (four-domain null, DNB selection artefact, EEG/DNB transfer
boundaries, cross-dataset non-portability, streaming recall) are stated
with the same weight as the positive grid result — they are the point.
