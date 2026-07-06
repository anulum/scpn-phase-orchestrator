<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 | Contact: www.anulum.li | protoscience@anulum.li -->

# Monitor — Grid Modal Growth

The power-grid modal-growth detector — the first domain-specific early-warning detector in
the programme to beat the generic suite on a real corpus, and the operational core that a
live monitor scores on each window.

## What it is for

When a disturbance leaves an electromechanical mode under-damped, the amplitude of the
cross-bus voltage deviation grows *exponentially*: the real part `σ` of the dominant mode's
eigenvalue turns positive (negative damping), the canonical wide-area-monitoring
early-warning quantity. This module estimates that growth rate directly and passively — it
reads bus voltages and returns a growth rate; it never actuates and never recalibrates, so
its operating point is fixed by the offline certification (`bench.grid_modal_head_to_head`).

## Contract

- `cross_bus_deviation` — the per-sample mean absolute deviation of the bus voltages from
  their cross-bus mean, an amplitude envelope of the collective mode.
- `per_bus_deviation` — the same deviation kept *per bus*, so a growth rate can be measured
  on each bus and the most unstable one taken.
- `envelope_growth_rate` — the exponential growth rate `σ` of a deviation envelope (the
  slope of its log against time); `σ > 0` is a growing, unstable mode, `σ < 0` a damped one.
  A `recency_top` weighting lets later samples count for more, because a real instability
  accelerates toward onset.
- `modal_growth_score` — one segment's `σ` under the `"focal"` (most unstable bus) or
  `"mean"` (whole network) aggregation; the certified default is `"focal"` with a recency
  weighting.

On the PSML 23-bus corpus, labelling transitions by disturbance type (a label independent of
the growth statistic, so the comparison is not circular), the detector leads 36 of 90
growing-instability transitions at a matched ten-percent false alarm where every generic
member is at chance.

## References

- Zheng et al. 2021 — the PSML power-system dataset (23-bus millisecond-level PMU
  measurements) with disturbance-type annotations.
- Kundur 1994, *Power System Stability and Control* — small-signal (modal) stability: a
  mode's eigenvalue real part is its growth rate, the sign of instability.

::: scpn_phase_orchestrator.monitor.grid_modal_growth
