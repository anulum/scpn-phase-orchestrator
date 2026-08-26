<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->

# Live demo — Grid Sentinel Replay

**[Open the interactive demo](../demo/grid_sentinel_replay.html)** — a real
recorded grid oscillation replayed through the certified modal-growth stream
monitor, with every claim hash-sealed.

## What the demo shows

ISO-NE case 3 of the UTK oscillation test-case library (Maslennikov et al.
2016) — a documented 1.13 Hz regional oscillation from July 2017, real PMU
data — is replayed sample by sample through
`GridModalStreamMonitor`. The monitor's operating point is read only from the
sealed cross-dataset evaluation artefact
(`examples/real_data/iso_ne_forced_oscillation/`), whose content hash is
verified before any value is trusted. The first live alarm fires **57.1 s
before the estimated oscillation onset**, reproducing the sealed offline
decision within stream-step quantisation, and the replay itself seals a
record of its own — pacing the replay at wall-clock speed produces a
byte-identical seal, because the decisions are deterministic.

The page charts the live growth-rate score, the certified threshold, the
estimated onset, and the full alarm log, and lists the three
content-addressed hashes (raw capture, sealed evidence, sealed replay
record) so anyone with the cited public capture can recompute the claim.

## Claim boundary

The demo demonstrates the sealed live pipeline — it is **not** a statistical
generalisation claim. The threshold was calibrated on this system's own
pre-onset ambient windows (six windows, three events, permutation p = 0.33);
the powered cross-dataset leg (WECC 240-bus) is separate and in progress.
The sentinel is review-only: it observes and seals, and never actuates.

Related: the honest false-alarm methodology in
[Early-Warning: Honest False-Alarm Evaluation](../studies/early_warning_matched_false_alarm.md)
and the first real-data case study
[ISO-NE Forced Oscillation](iso_ne_case1_forced_oscillation.md).
