<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->

# Paper 2 — the grid positive result and the eigenvalue regime map

`02_grid_modal_growth_regime_map.tex` is the positive / physical half of
the split (external review + triage: `docs/internal/reviews/`).
Canonical numbers: the study page and the sealed artefacts under
`examples/real_data/`. 8 pages on plain TeX Live; build with `pdflatex`
twice (PDF not committed).

## Proposed metadata (OWNER approves before submission)

- **Title:** A domain-specific modal-growth detector clears a matched
  false-alarm bar on power-grid instability, and an eigenvalue regime
  map shows when its form transfers
- **Authors:** Miroslav Šotek (ANULUM / Fortis Studio)
- **Primary category:** `eess.SY`
- **Cross-lists (proposal):** `physics.data-an`, `nlin.AO` (Kuramoto
  collective-coordinate results)
- **License (recommendation):** arXiv non-exclusive distribution license
- **Comments field:** "8 pages. Companion to arXiv:XXXX.XXXXX (the
  operational-protocol null result). All evidence records are
  hash-sealed and byte-reproducible; source and sealed artefacts at
  https://github.com/anulum/scpn-phase-orchestrator"
- **BEFORE SUBMITTING:** replace the companion placeholder above with
  Paper 1's real arXiv id (submit 01 first).

### Metadata abstract (within the arXiv field limit)

> Under a matched-false-alarm, permutation-tested protocol, generic
> early-warning detectors are statistically at chance on power-grid
> growing oscillations (companion manuscript). We show the
> complementary positive result and bound it. A detector reading the
> canonical wide-area-monitoring instability quantity --- the
> exponential growth rate of the most unstable bus's voltage-deviation
> envelope --- leads 36 of 90 generator-trip transitions of the PSML
> 23-bus corpus at a matched 10% false alarm (permutation p = 0.0001;
> held-out 24 of 45, p = 0.0002), on a non-circular disturbance-type
> split where the best generic detector is at chance. Stress tests
> bound the claim: run as a causal stream the certified per-window
> operating point is unusable and an honest recalibration with an
> exponential-fit-quality gate recovers 11 of 45 held-out transitions
> at a matched 10% stream false alarm; on two cross-dataset corpora
> (three real ISO-NE captures; 13 synthetic WECC 240-bus contest cases)
> the certified numeric threshold does not transfer in either
> direction, while the frozen shape with per-system calibration remains
> usable; and the form does not transfer to scalp EEG or few-timepoint
> molecular indices. Against ANDES small-signal eigenvalues, three
> analytic normal forms, and unimodal/bimodal mean-field Kuramoto
> models, the estimated quantity is confirmed to be the eigenvalue's
> real part, with a regime map: envelope growth sizes oscillatory
> modes, autocorrelation non-oscillatory ones. All evidence records are
> content-addressed and byte-reproducible.

## Submission checklist (OWNER-GATED)

1. Submit AFTER Paper 1; insert its arXiv id into the abstract/comments
   companion reference.
2. Owner reviews the locally built PDF and the metadata above.
3. Upload the single `.tex` as source (bibliography is inline).
