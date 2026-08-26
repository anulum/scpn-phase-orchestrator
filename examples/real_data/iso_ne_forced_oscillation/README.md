<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->

# ISO-NE cross-dataset evaluation — the operating-point portability test

This directory holds the sealed record of the External Validation Program's
**cross-dataset generalisation leg**: the PSML-certified grid modal-growth detector
(`../psml_modal_growth/`), with its shape **frozen**, evaluated on real ISO-NE
PMU captures of documented sustained oscillations it never saw. The question is
not "does the detector win here" but "what survives a dataset change" — and the
answer is sealed either way.

## Honest headline

* **The PSML operating point does not port.** Run verbatim (frozen-transfer branch:
  threshold `1.3203 σ/s`, two-second windows), ambient ISO-NE frequency noise
  alone crosses the frozen threshold in 71 of 219 case-2 pre-onset windows
  (~32 % per-window false alarm against the certified 9.09 %), and transition
  windows cross at the same rate. An operating point certified on 238 Hz
  fault-transient voltage envelopes is **not a portable constant** — which is
  why the product's deployment step calibrates per system.
* **The locally calibrated frozen shape detects 1 of 3 events** (local-calibration branch:
  window = five cycles of the documented mode, threshold matched to a 10 %
  false alarm on pooled pre-onset ambient windows only): the 1.13 Hz regional
  mode of case 3 is caught **57.1 s before the estimated onset**; cases 1-2
  stay silent. Permutation p = 0.332 at n = 3 — **no significance claim**; this
  leg is a pre-registered case study, and the statistical leg is WECC 240-bus.
* **The null base is thin and disclosed.** The pre-registered 60 s transition
  region consumes the whole pre-onset span of the early-onset cases, leaving
  six pooled null windows (case 2 only). No parameter was revised after seeing
  data; a revision for future legs is a disclosed protocol change decided
  before running.

## The source data (not included here)

The raw captures are **citation-only** and **not redistributed**. Obtain them
from the dataset authors' library:

- Page: <https://web.eecs.utk.edu/~kaisun/Oscillation/actualcases.html>
  (`ISO-NE-case1.zip` … `ISO-NE-case3.zip`; each contains `ISO-NE_caseN.csv`).
- Cite as requested there: S. Maslennikov, B. Wang, Q. Zhang, F. Ma, X. Luo,
  K. Sun, E. Litvinov, *A test cases library for methods locating the sources
  of sustained oscillations*, IEEE PES General Meeting, 2016.

Verifying these SHA-256 digests confirms bit-identical sources (case 1 equals
the digest sealed in `../iso_ne_case1/` since July 2026):

| Case | `ISO-NE_caseN.csv` SHA-256 |
|------|----------------------------|
| 1 | `ca5001bb64cfecced20ea71a6a007a5db8ad96acdcfa13cb021358f0f2575de0` |
| 2 | `e503003f2df7e02262a9412700ed3be6833bb9f95e120ceea21596b4a6f53f1f` |
| 3 | `f30d399b017aeee04987dadbf9cec0117444b4372456c93aa367e86ccba3772f` |

Cases 4-6 of the library are excluded from the corpus for reasons fixed before
any detector run and sealed in the payload (`corpus.excluded`): no separable
in-band onset (4, 5) and a single-substation channel degeneracy (6).

## Reproducing the sealed record

With the extracted cases under a directory laid out as
`case1/ISO-NE_case1.csv`, `case2/ISO-NE_case2.csv`, `case3/ISO-NE_case3.csv`:

```bash
python -m bench.early_warning_leadtime_isone \
  --data-dir <that-directory> \
  --output iso_ne_modal_growth_cross_dataset.json
```

The run is deterministic (fixed permutation seed 0): regenerating reproduces
`iso_ne_modal_growth_cross_dataset.json` byte for byte, and the committed
`content_hash`
`fd6285fc341a5884aab0d806e54a912c90eec1024e95ee269f71b0db4bd461b7` recomputes
from the payload alone. `tests/test_iso_ne_cross_dataset_evidence.py` guards both without
the raw data.

## What this record does and does not claim

It claims exactly: the frozen operating point fails to transfer (measured), the
frozen shape with per-system ambient calibration catches one of three real
events with a positive lead at a held false alarm, and the corpus is too small
for significance. It does **not** claim cross-dataset generalisation is
established, and it does not revise the detector — any variant search on this
data would disqualify the leg as a frozen-shape cross-dataset evaluation and restart it as a disclosed
maximisation round.
