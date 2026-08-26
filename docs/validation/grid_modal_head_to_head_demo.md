<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->

# Live demo — Modal Head-to-Head

**[Open the interactive demo](../demo/grid_modal_head_to_head.html)** — the
certified comparison behind the grid modal-growth detector, charted from the
sealed record.

## What the demo shows

On 90 real generator-trip instabilities of the PSML 23-bus corpus (Zheng et
al. 2021), the domain-specific modal envelope-growth detector leads 36
transitions at a matched 10% false alarm (label-permutation p = 0.0001, and
24/45 on the held-out half, p = 0.0002) — while every generic early-warning
detector, including the classical critical-slowing-down baseline, stays
statistically at chance on the identical split. The chart shows each
detector's led count against its own expected-by-chance mark, with the
p-values sealed alongside.

The comparison is fair by construction: identical segments and calibration,
disturbance-type labels independent of the scored statistic, disclosed
data-quality drops, and a pre-registered operating point validated on a
held-out half. The whole record is one content-addressed artefact
(`examples/real_data/psml_modal_growth/`) guarded by an integrity test.

## Claim boundary

The certificate is corpus-specific: the numeric operating point does not port
across datasets (measured on ISO-NE captures — see the
[Grid Sentinel Replay demo](grid_sentinel_replay_demo.md)), so deployment
calibrates per system on its own ambient data. "Generic detectors at chance"
is a sealed statement about this task and operating point, not a blanket
dismissal of those methods elsewhere.

Related: [Early-Warning: Honest False-Alarm Evaluation](../studies/early_warning_matched_false_alarm.md).
