<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->

# Live demo — The Honest Test

**[Open the interactive demo](../demo/honest_test.html)** — the four-domain
falsification of generic early warning, charted from the committed record.

## What the demo shows

The best SCPN suite member and the classical Dakos et al. 2008
AR(1)-Kendall-tau detector were evaluated on real transitions in four domains
(EEG seizures, cardiac AF onsets, grid instabilities, palaeoclimate tippings)
at a matched 10% false alarm with label-permutation significance. No detector
reached significance in any domain; the strongest signal anywhere is the
classical detector on EEG at p = 0.067. The chart shows each detector's led
count per domain against its expected-by-chance mark.

Publishing a negative result about our own detectors is deliberate: the same
harness that demoted the generic suite is the one that certified the modal
detector which does clear the bar (see the
[Modal Head-to-Head demo](grid_modal_head_to_head_demo.md)), and it is the
harness offered for certifying any detector on any data. A certification you
can trust is one that is allowed to say no.

## Claim boundary

Corpora are small (6-12 transitions per domain) — the honest statement is "no
detector demonstrated significance here", not "early warning is impossible".
One classical competitor was implemented; the EEG corpus is single-subject.
The committed record is `examples/real_data/head_to_head_ar1_kendall/`; the
publication-form study is
[Early-Warning: Honest False-Alarm Evaluation](../studies/early_warning_matched_false_alarm.md).
