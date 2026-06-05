# Monitor — Merge Window

`MergeWindowMonitor` is the PHA-C.4 runtime gate for deciding when a
moving-frame phase cluster has actually merged. It combines two predicates:
wrapped phase dispersion around a reference phase and axial spatial dispersion
around a reference point. The monitor emits `lock_achieved=True` only after both
predicates remain inside tolerance for the configured number of consecutive
samples.

## What it is for

The monitor is intended for PHA-C pipelines where phase dynamics and physical or
abstract position are both meaningful. Examples include moving-frame UPDE runs,
coalescence studies, chamber or conveyor synchronization, plasma or beam
alignment experiments, robotic swarms, and digital-twin cells where a phase lock
alone is not enough evidence that the population has spatially merged.

## Contract

- Phase dispersion: `max_i |wrap(theta_i - theta_ref)| <= phase_tol_rad`.
- Spatial dispersion: `max_i |z_i - z_ref| <= spatial_tol_m`.
- Signed margins: `phase_margin_rad = phase_tol_rad - phase_dispersion_rad`
  and `spatial_margin_m = spatial_tol_m - spatial_dispersion_m`.
- Signed-margin replay tolerance: `MERGE_WINDOW_MARGIN_REPLAY_TOLERANCE`.
- Consecutive gate: both predicates must pass for `required_consecutive_samples`.
- Default tolerances: `phase_tol_rad=0.01`, `spatial_tol_m=0.002`,
  `required_consecutive_samples=3`.
- Named profiles multiply the reviewed baseline: `baseline_1x`, `buffer_3x`,
  and `review_5x`.
- Evidence boundary: benchmark timings are local regression evidence unless run
  under the documented isolated-core benchmark protocol.

```python
import numpy as np
from scpn_phase_orchestrator.monitor.merge_window import MergeWindowMonitor

monitor = MergeWindowMonitor(
    phase_tol_rad=0.01,
    spatial_tol_m=0.002,
    required_consecutive_samples=3,
)

for t in range(3):
    report = monitor.evaluate(
        np.array([0.0, 0.004, -0.005]),
        np.array([0.0, 0.001, -0.0015]),
        t=float(t),
    )

assert report.lock_achieved
```

## Tolerance profiles

PHA-C and MIF/FRC review lanes often need to separate the reviewed baseline
window from wider diagnostic buffers. `resolve_merge_window_tolerance_profile`
keeps that boundary explicit:

```python
from scpn_phase_orchestrator.monitor.merge_window import (
    MergeWindowMonitor,
    resolve_merge_window_tolerance_profile,
)

profile = resolve_merge_window_tolerance_profile("buffer_3x")
assert profile.spatial_tol_m == 0.006

monitor = MergeWindowMonitor(tolerance_profile="buffer_3x")
```

The default baseline is `0.01` rad and `0.002` m. Passing explicit
`phase_tol_rad` or `spatial_tol_m` with a profile treats those values as the
baseline before applying the multiplier.

## Polyglot surfaces

The Python monitor is the public runtime reference. Rust, Go, Julia, and Mojo
source-contract adapter modules are present for parity gates and downstream
accelerator wiring. The benchmark gate records all declared backend slots and
labels the local workstation timing data as non-isolated evidence. Adapter
parity includes the signed margin fields, so a backend cannot pass with only
boolean lock evidence. The benchmark payload also publishes
`phase_margin_equation_validated`, `spatial_margin_equation_validated`,
`signed_margin_equations_validated`, and `margin_replay_tolerance`; the gate
fails unless every declared backend row proves both signed-margin equations.

```bash
uv run python benchmarks/merge_window_benchmark.py --parity-gate
```

## Event/state handoff

When the merge-window report must cross into MIF, Studio, audit replay, or
another downstream PHA-C lane, use
`build_pha_c_handoff_record(...)`. It binds the merge report to the source
phase and position digests, adds signed margin and order-parameter evidence,
and fixes the
non-actuating claim boundary for later review.

```python
from scpn_phase_orchestrator.upde.pha_c_handoff import (
    build_pha_c_handoff_record,
)

handoff = build_pha_c_handoff_record(
    phases,
    positions,
    phase_tol_rad=0.01,
    spatial_tol_m=0.002,
    required_consecutive_samples=3,
)
```

## Operational role

- This monitor is the gate where “mostly synchronized” becomes “merged” in a
  replayable way.
- Signed-margin outputs are what operators use to tune recovery aggressiveness
  without guessing how close the signal was to the boundary.
- The handoff record keeps review lanes deterministic: MIF, Studio, and audit
  replay all receive the same lock/evidence contract.

## What this means for operations

Merge-window gates are most useful when teams need a binary operational decision
(`lock_achieved`) plus a continuous confidence context (`margins`).

The signed-margin design makes recovery tuning possible without guesswork:

- `phase_margin_rad` and `spatial_margin_m` quantify how far the current state is
  from the lock boundary.
- `required_consecutive_samples` filters transients and reduces false
  merge-on-spike behaviour.

In production review lanes, that combination is what allows teams to avoid both
premature lock declarations and excessive delay in returning to active control.

::: scpn_phase_orchestrator.monitor.merge_window
