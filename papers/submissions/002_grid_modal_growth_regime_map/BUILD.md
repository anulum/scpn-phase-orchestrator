<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->

# Build

Plain TeX Live, no packages beyond the standard set, bibliography
inline (no BibTeX pass needed):

```bash
pdflatex -jobname=manuscript manuscript.tex
pdflatex -jobname=manuscript manuscript.tex
```

Two passes resolve cross-references. Output: `manuscript.pdf`
(8 pages, letter). The tracked `manuscript.pdf` is the inspected
review artefact rebuilt from this exact source; the Zenodo record
carries the published PDF of the same source (md5
`6027788713f79ee72b8079d8a4190605`).
