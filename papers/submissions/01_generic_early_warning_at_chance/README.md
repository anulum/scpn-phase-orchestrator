<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->

# Paper 1 — the operational-protocol null result

`01_generic_early_warning_at_chance.tex` is the methodological /
negative-result half of the split (external review + triage:
`docs/internal/reviews/grok_2026-08-26_matched_fa_paper*.md`). Canonical
numbers: `docs/studies/early_warning_matched_false_alarm.md` and the
sealed artefacts under `examples/real_data/`. 8 pages on plain TeX Live;
build with `pdflatex` twice (PDF not committed).

## Proposed metadata (OWNER approves before submission)

- **Title:** Generic early-warning signals are at chance under a matched
  false-alarm protocol across brain, heart, grid, climate and molecular
  data
- **Authors:** Miroslav Šotek (Anulum Institute)
- **Primary category:** `physics.data-an`
- **Cross-lists (proposal):** `stat.ME` (the evaluation protocol),
  `q-bio.QM` (the DNB modality)
- **License (recommendation):** arXiv non-exclusive distribution license
- **Comments field:** "8 pages. Companion manuscript (grid positive
  result and eigenvalue regime map) in preparation from the same
  repository. All evidence records are hash-sealed and
  byte-reproducible; source and sealed artefacts at
  https://github.com/anulum/scpn-phase-orchestrator"

### Metadata abstract (within the arXiv field limit)

> Generic early-warning signals (rising variance and lag-one
> autocorrelation of critical slowing down, synchronisation and
> ordinal-entropy indicators) are widely reported to precede abrupt
> transitions, but most evidence rests on a retrospective per-record
> trend test. We evaluate the operational question instead: at a fixed
> false-alarm budget, does a detector fire on transitions more often
> than on no-transition controls? One detector suite and one
> matched-false-alarm harness are applied unchanged to four labelled
> corpora (scalp-EEG seizures, cardiac AF onsets, power-grid growing
> oscillations, palaeoclimate transitions), scored by label-permutation
> significance. No detector reaches significance in any domain, and the
> canonical Dakos AR(1)-Kendall-tau detector fares no better on the
> identical segments, including 0 of 6 on its own palaeoclimate
> records; its strongest showing anywhere is 3 of 6 on scalp EEG
> (p = 0.067). Extending the protocol to dynamical-network-biomarker
> indices reaches the same conclusion under modality-appropriate nulls,
> including a surrogate null granted the analysis's own
> module-selection freedom. The corpora are small, so these nulls bound
> demonstrated skill rather than proving early warning impossible; what
> they establish is that the operational bar is materially stricter
> than the retrospective test on which most published evidence rests.
> Every alarm and non-detection is sealed in content-addressed,
> byte-reproducible evidence records.

## Submission checklist (OWNER-GATED)

1. Owner reviews the locally built PDF and the metadata above.
2. Upload the single `.tex` as source (bibliography is inline).
3. Submit 01 BEFORE 02; after the arXiv id exists, update 02's
   companion reference, CITATION.cff, .zenodo.json, and the study page.
