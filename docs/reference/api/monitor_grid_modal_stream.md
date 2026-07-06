<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 | Contact: www.anulum.li | protoscience@anulum.li -->

# Monitor — Grid Modal Stream

A causal, real-time streaming monitor that carries the certified grid modal-growth detector
online — the step from an offline benchmark to an operational early warning.

## What it is for

The offline head-to-head certifies the detector on fixed pre-onset segments. This monitor
runs the *same* detector **causally on a live stream**: it keeps a sliding window of the most
recent per-bus voltage samples and, every step, re-scores that window with the identical
primitives the detector uses, so the streaming score on a window is bit-for-bit the offline
`modal_growth_score` on that same window. That identity is what makes the offline-calibrated
threshold valid online — the monitor never recalibrates.

## Contract

- `GridModalStreamMonitor` — pushes one per-bus sample at a time (`update`) and raises a
  `StreamAlarm` when the live growth rate crosses the certified threshold, after an optional
  persistence debounce, latching until the growth falls back below so each instability
  episode raises one lead event.
- `StreamAlarm` — a lead event: the sample index and time it fired, the growth rate `σ`, the
  threshold crossed, and the most unstable bus.
- `GridModalStreamMonitor.from_evidence` — builds the monitor straight from a sealed
  head-to-head artefact (aggregation, recency weighting, and matched-false-alarm threshold
  from the certification), so the certified detector becomes the live monitor with no
  hand-set constants.

The honest live-deployment operating point — a stream is stricter than the pre-onset window,
because a damped fault has a transient growth window the continuous monitor also alarms on —
is measured and sealed in `examples/real_data/psml_modal_growth/`.

::: scpn_phase_orchestrator.monitor.grid_modal_stream
