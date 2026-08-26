<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->

# WECC 240-bus cross-dataset evaluation — the E2.G statistical leg

This directory holds the sealed record of the External Validation Program's
**E2.G statistical leg**: the PSML-certified grid modal-growth detector
(`../psml_modal_growth/`), with its shape **frozen**, evaluated on all 13
forced-oscillation cases of the 2021 IEEE-NASPI Oscillation Source Location
Contest — synthetic PMU exports of the reduced WECC 240-bus system it never
saw. The protocol was pre-registered before the first detector run and every
amendment is disclosed in the payload; the outcome is sealed either way.

## Honest headline

* **The PSML operating point does not port — now measured in BOTH
  directions.** Run verbatim (G-a branch: threshold `1.3203 σ/s`, two-second
  windows), the frozen threshold crosses **nothing** on WECC: 0 of 611 pooled
  ambient windows (0 % false alarm against the certified 9.09 % on PSML) and
  0 transition windows in any case — the operating point is far too
  conservative here, just as it was ~3.5× too hot on real ISO-NE frequency
  channels. A certified numeric threshold is **not a portable constant**;
  deployment calibrates per system on its own ambient data.
* **The early-warning window is structurally absent in most cases.** With the
  onset search pinned at the exact forcing start (t = 30 s, known by
  simulation design), 9 of 13 cases are already above three times the ambient
  in-band envelope at the forcing start — an effectively instantaneous onset
  with an empty early-warning region. Of the 4 cases with a resolvable
  region, the locally calibrated frozen shape (G-b: window = five cycles of
  the documented forcing fundamental, threshold matched to a 10 % false alarm
  on 130 pooled pre-disturbance ambient windows) **leads 2** (case 1:
  +5.4 s; case 4: +4.1 s). Permutation p = 0.41 — **no significance claim**
  for early-warning lead on this corpus.
* **Detection after onset is descriptive, and fast.** At the same matched
  operating point the detector alarms in 12 of 13 cases, 8 of them within the
  first two scoring windows after the forcing start (median latency ~1 s;
  the one miss is the HVDC-source case 13). The detection COUNT carries **no
  significance claim**: at a 10 % per-window false alarm, a 60 s region
  alarms by chance with probability up to `1-(0.9)^n ≈ 0.85-1.0` per case
  (the payload seals the per-case bound and the caveat); the informative
  quantity is the latency concentration.

## The source data (not included here)

The raw exports are **citation-only** and **not redistributed**. Obtain them
from the dataset authors' library:

- Page: <https://web.eecs.utk.edu/~kaisun/Oscillation/contestcases.html>
  (`All_cases.zip`, 41.3 MB, cases `Case1` … `Case13`, each with
  `CaseN_PMU/BusVolMag.txt` among the four PMU exports; the solution key and
  TSAT file description PDFs are linked on the same page).
- The scenario set was designed by the IEEE-NASPI OSL Committee on the
  reduced WECC 240-bus model developed by NREL (H. Yuan, R. S. Biswas,
  J. Tan, Y. Zhang, *Developing a Reduced 240-Bus WECC Dynamic Model for
  Frequency Response Study of High Renewable Integration*, 2020 IEEE/PES
  T&D); credit DOE/NREL/Alliance per the NREL disclaimer shipped in the
  archive, and cite the test-cases library: S. Maslennikov, B. Wang et al.,
  *A test cases library for methods locating the sources of sustained
  oscillations*, IEEE PES General Meeting, 2016.

Verifying these SHA-256 digests confirms bit-identical sources:

| Archive | SHA-256 |
|---------|---------|
| `All_cases.zip` | `ac02aef5c6259cf2184e8f690f9b6ade24ec104e1b467c8b082694925b2a9b93` |

| Case | `CaseN_PMU/BusVolMag.txt` SHA-256 |
|------|-----------------------------------|
| 1 | `94a5f89ef6d051c81b313b758b8736dc87f7086cfa101b58b9e6fa519de0b7b8` |
| 2 | `e3b4e1e6918a2e4ac92d48cf86df927256ed0be080c634dfa61ffea4db6a398c` |
| 3 | `d9a9988de751cec90c98dffa72baeab21961da59ec37ab41409889f41d614fdf` |
| 4 | `b2d455485f920bfd589691447f9affe879922130b734c2700b7f301c3ceeebf5` |
| 5 | `2f30dc65dd9c94bf602897e07edf4c933a70880d27d7ae865127bd37fe927712` |
| 6 | `8239bcf4e85e3e3594250406ab5622b09fe254316a2b264607eac721315fd1a1` |
| 7 | `fc4ec6c05b6bb416372288c4df7def5bfaeaa685783c20005a577568a58936d6` |
| 8 | `c735189b15db55fedc45eaf660971da7a08a7ea74e46919a0d9f61fc94dff192` |
| 9 | `76b048ed88c27a2c29ce3b8473d68f10a8168f663d285792d13973e68df0c3c8` |
| 10 | `32d321534729b90917753a67bae54e25485bcd98bca61bb531eb7187aa2b8d8a` |
| 11 | `00478d9adc42bb5e2b4415e60c091b88038b1c250d4c433ccda9201fe8163e0f` |
| 12 | `25396a74f7bd9fb0c77327709e4964b7628f2a4cfd03b710ec44d2b64ddecb40` |
| 13 | `b5fde906e1742246aec530819a9fdeae8c1e5d4c4d767d42ea690942365e8d07` |

## Protocol provenance

Fixed a priori (internal plan appendix A.16) before the first detector run:
observable = per-bus voltage magnitude (the certified observable family);
ground truth from the contest solution key (forced oscillation at t = 30 s in
every case, all masking disturbances at t ≥ 26 s); null windows end at or
before t = 25 s; the early-warning region is `(30 s, onset_est]`. Disclosed
amendments, both fixed before the first real-corpus run: A.16.1 pins the
onset search at the forcing start (acausal smoothing smear) and adds the
secondary detection branch; A.16.2 lets the parser drop TSAT's duplicated
discontinuity rows (the duplicate timestamps sit exactly at the documented
fault times). A.16.3, after the run, marks the detection-branch permutation
p-value as non-interpretable (region/window unit mismatch) — a disclosure,
not a protocol change; no scores were altered.

## Reproducing the sealed record

With `All_cases.zip` extracted so cases sit at
`All_cases/CaseN/CaseN_PMU/BusVolMag.txt`:

```bash
python -m bench.early_warning_leadtime_wecc \
  --data-dir <path-to>/All_cases \
  --output wecc_240_osl_modal_growth_cross_dataset.json
```

The run is deterministic (fixed permutation seed 0): regenerating reproduces
`wecc_240_osl_modal_growth_cross_dataset.json` byte for byte, and the
committed `content_hash`
`f77be580e06b532e2471b452ac72856f87a5df0444c2d38b782b4be306982b66` recomputes
from the payload alone. `tests/test_wecc_e2g_evidence.py` guards both without
the raw data.

## What this record does and does not claim

It claims exactly: the frozen operating point fails to transfer in the
conservative direction (measured 0 % false alarm and 0 % detection), the
early-warning window is structurally absent for instant-onset forced
oscillations at these record lengths (led 2 of 4 resolvable cases, p = 0.41,
not significant), and post-onset detection at a locally matched false alarm
is fast (median ~1 s) but its count is chance-compatible and claimed only
descriptively. It does **not** claim cross-dataset early-warning
generalisation, and it does not revise the detector — any variant search on
this data would disqualify the leg as E2.G and restart it as a disclosed
maximisation round.
