# UPDE — PHA-C Event Timeline

`PHACTimelineRecord` is the trajectory-level PHA-C evidence surface. It consumes
phase and axial-position matrices from a moving-frame run, threads the
single-sample `PHACHandoffRecord` contract through time, and emits a compact
lock/loss/reset timeline for downstream review lanes.

The timeline is intentionally review-only. It never writes to actuators, changes
coupling, schedules hardware, or mutates supervisor state.

For the full PHA-C chain from spatial modulation through moving-frame dynamics
and timeline hashing, use
[`PHACAcceptanceRecord`](upde_pha_c_acceptance.md).

## Use cases

Use the PHA-C event timeline when a downstream lane needs event evidence across
an entire trajectory instead of a single sample:

- MIF/FRC review where the first lock time, lock losses, and reset count must be
  carried into chamber or field-reconstruction analysis;
- Studio panels that need a stable summary of when a moving-frame run entered
  or left a reviewed merge window;
- replay ledgers where every sample handoff hash must roll into one canonical
  trajectory hash;
- safety reviews that need to separate observed lock evidence from any later
  policy or operator action;
- polyglot parity gates that compare Python, Rust, Go, Julia, and Mojo
  source-contract behavior for the same trajectory.

## Contract

Inputs are two real finite matrices with shape `(time, oscillator)`:

```text
phases_by_step[t, i]    = wrapped or unwrapped phase for oscillator i at sample t
positions_by_step[t, i] = axial position for oscillator i at sample t
times[t]                = strictly increasing sample time
```

For each sample, the builder calls `build_pha_c_handoff_record(...)` and carries
`consecutive_lock_samples` forward. The final timeline records:

- `first_lock_index` and `first_lock_time` for the first achieved consecutive
  joint lock;
- `final_lock_achieved` for the last sample;
- phase-lock, spatial-lock, full-lock, lock-loss, and reset counts;
- maximum phase dispersion, maximum spatial dispersion, minimum Kuramoto order
  parameter, maximum distance to the reference position, and minimum signed
  phase/spatial margins over the trajectory;
- replay validation that each minimum signed margin equals the resolved
  tolerance minus the corresponding maximum dispersion under
  `PHA_C_TIMELINE_MARGIN_REPLAY_TOLERANCE`;
- tolerance profile name, multiplier, resolved phase tolerance, and resolved
  spatial tolerance;
- SHA-256 digests for the time vector, all sample records, transition table,
  and final timeline;
- `execution_disabled=True`, `actuating=False`, and the fixed claim boundary
  `pha_c_event_timeline_review_only`.

## Minimal example

```python
import numpy as np
from scpn_phase_orchestrator.upde.pha_c_timeline import (
    build_pha_c_event_timeline,
    verify_pha_c_event_timeline,
)

phases = np.array(
    [
        [-0.02, 0.0, 0.02],
        [-0.002, 0.0, 0.002],
        [-0.0015, 0.0, 0.0015],
        [-0.001, 0.0, 0.001],
        [-0.02, 0.0, 0.02],
    ]
)
positions = np.array(
    [
        [-0.003, 0.0, 0.003],
        [-0.0005, 0.0, 0.0005],
        [-0.0004, 0.0, 0.0004],
        [-0.0003, 0.0, 0.0003],
        [-0.003, 0.0, 0.003],
    ]
)
times = np.arange(phases.shape[0]) * 0.5

timeline = build_pha_c_event_timeline(
    phases,
    positions,
    times=times,
    phase_tol_rad=0.01,
    spatial_tol_m=0.002,
    required_consecutive_samples=3,
    tolerance_profile="baseline_1x",
)

assert timeline.first_lock_index == 3
assert timeline.lock_loss_count == 1
assert timeline.execution_disabled
assert not timeline.actuating
evidence_payload = timeline.to_dict()
verify_pha_c_event_timeline(timeline)
```

Use `verify_pha_c_event_timeline(...)` when replaying a stored trajectory
record. It rechecks timeline counts, first-lock semantics, transition-count
bounds, review-only flags, signed margin equations, SHA-256 fields, and the
canonical timeline hash without requiring raw trajectory matrices. The signed
margin replay rejects records whose positive-looking phase or spatial margin no
longer matches `tolerance - maximum_dispersion`.

## Handoff versus timeline

| Surface | Scope | Main output | Use when |
|---------|-------|-------------|----------|
| `PHACHandoffRecord` | one phase/position sample | scalar lock evidence plus sample hashes | a downstream lane needs one reviewed event-state atom |
| `PHACTimelineRecord` | complete trajectory | lock acquisition, loss, reset, profile, and trajectory hashes | a downstream lane needs replayable event history |

The timeline is built from handoff records. If native accelerator kernels are
added later, they must preserve both the per-sample handoff hashes and the final
timeline hash, including the minimum signed margins.

## Polyglot parity

The benchmark gate records Rust, Mojo, Julia, Go, and Python source-contract
slots. The current timeline path is evidence construction, not a numerical hot
loop, so the non-Python slots validate parity against the Python reference
contract. If native kernels are later added, they must preserve the same hashes
signed margins, signed-margin equations, and fail-closed input boundaries. The
benchmark payload publishes `phase_margin_equation_validated`,
`spatial_margin_equation_validated`, `signed_margin_equations_validated`, and
`margin_replay_tolerance` for every backend row.

```bash
uv run python benchmarks/pha_c_timeline_benchmark.py \
  --parity-gate \
  --calls 1 \
  --output benchmarks/results/pha_c_timeline.json
```

Committed benchmark JSON is local regression evidence only. It is not a
production timing claim unless rerun under the benchmark-isolation protocol.

## Failure boundaries

The timeline fails closed on:

- empty, non-finite, complex, object-dtype, or boolean trajectory matrices;
- non-matrix phase or position inputs;
- mismatched phase and position shapes;
- missing, non-finite, non-vector, wrong-length, or non-increasing time vectors;
- negative tolerances;
- invalid consecutive-sample controls;
- unknown tolerance profile names.

::: scpn_phase_orchestrator.upde.pha_c_timeline.PHACTimelineRecord

::: scpn_phase_orchestrator.upde.pha_c_timeline.build_pha_c_event_timeline

::: scpn_phase_orchestrator.upde.pha_c_timeline.pha_c_event_timeline_to_dict

::: scpn_phase_orchestrator.upde.pha_c_timeline.verify_pha_c_event_timeline
