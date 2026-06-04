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
- Consecutive gate: both predicates must pass for `required_consecutive_samples`.
- Default tolerances: `phase_tol_rad=0.01`, `spatial_tol_m=0.002`,
  `required_consecutive_samples=3`.
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

## Polyglot surfaces

The Python monitor is the public runtime reference. Rust, Go, Julia, and Mojo
source-contract adapter modules are present for parity gates and downstream
accelerator wiring. The benchmark gate records all declared backend slots and
labels the local workstation timing data as non-isolated evidence.

```bash
uv run python benchmarks/merge_window_benchmark.py --parity-gate
```

## Event/state handoff

When the merge-window report must cross into MIF, Studio, audit replay, or
another downstream PHA-C lane, use
`build_pha_c_handoff_record(...)`. It binds the merge report to the source
phase and position digests, adds the order parameter, and fixes the
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

::: scpn_phase_orchestrator.monitor.merge_window
