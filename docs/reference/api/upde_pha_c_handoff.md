# UPDE — PHA-C Handoff

`PHACHandoffRecord` is the PHA-C event/state bridge between moving-frame
simulation, merge-window monitoring, replay, MIF import, and Studio review.
It converts one phase-plus-position sample into deterministic scalar evidence:
phase dispersion, spatial dispersion, order parameter, lock status, source
digests, and a canonical record hash.

The handoff is intentionally review-only. It does not write to actuators,
modify coupling, schedule hardware, or mutate supervisor state.

## Use cases

Use the PHA-C handoff when a downstream lane needs a compact, verifiable event
record instead of raw phase and position arrays:

- MIF/FRC merger review where lock evidence must be carried into a separate
  field-reconstruction or chamber-analysis lane;
- Studio dashboards that show whether a moving-frame run reached a reviewed
  phase-space merge window without exposing full state vectors;
- deterministic replay where the same phase/position sample must regenerate the
  same event hash;
- operator evidence streams where the event can be logged before a human or
  policy layer decides whether any later action is allowed;
- polyglot benchmark gates that compare Python, Rust, Go, Julia, and Mojo
  source-contract behavior without making throughput claims.

## Contract

The handoff first evaluates `MergeWindowMonitor` semantics:

```text
phase_locked   = max_i |wrap(theta_i - theta_ref)| <= phase_tol_rad
spatial_locked = max_i |z_i - z_ref| <= spatial_tol_m
lock_achieved  = consecutive joint locks >= required_consecutive_samples
```

It then adds:

- `phase_order_parameter = |mean(exp(i theta_i))|`;
- `distance_to_reference_max_m = max_i |z_i - z_ref|`;
- the tolerance profile name and multiplier when `baseline_1x`, `buffer_3x`,
  or `review_5x` is used;
- SHA-256 digests for the phase vector, position vector, merge report,
  source-chain, and final record;
- `execution_disabled=True` and `actuating=False`;
- the fixed claim boundary `pha_c_event_state_handoff_review_only`.

The record contains scalar evidence and hashes only. It avoids serialising full
phase or position vectors into public evidence records.

## Minimal example

```python
import numpy as np
from scpn_phase_orchestrator.upde.pha_c_handoff import (
    build_pha_c_handoff_record,
)

record = build_pha_c_handoff_record(
    np.array([0.0, 0.003, -0.004]),
    np.array([0.0, 0.0005, -0.0008]),
    t=4.0,
    phase_tol_rad=0.01,
    spatial_tol_m=0.002,
    required_consecutive_samples=3,
    prior_consecutive_lock_samples=2,
    tolerance_profile="baseline_1x",
)

assert record.lock_achieved
assert record.execution_disabled
assert not record.actuating
event_payload = record.to_dict()
```

## Polyglot parity

The benchmark gate records Rust, Mojo, Julia, Go, and Python source-contract
slots. The current handoff path is evidence construction, not a numerical hot
loop, so the non-Python slots validate parity against the Python reference
contract. If native kernels are later added, they must preserve the same hashes
and fail-closed input boundaries.

```bash
uv run python benchmarks/pha_c_handoff_benchmark.py \
  --parity-gate \
  --calls 1 \
  --output benchmarks/results/pha_c_handoff.json
```

Committed benchmark JSON is local regression evidence only. It is not a
production timing claim unless rerun under the benchmark-isolation protocol.

## Failure boundaries

The handoff fails closed on:

- empty, non-finite, complex, object-dtype, or boolean phase/position vectors;
- mismatched phase and position lengths;
- non-finite timestamps or references;
- negative tolerances;
- invalid consecutive-sample controls.

::: scpn_phase_orchestrator.upde.pha_c_handoff.PHACHandoffRecord

::: scpn_phase_orchestrator.upde.pha_c_handoff.build_pha_c_handoff_record

::: scpn_phase_orchestrator.upde.pha_c_handoff.pha_c_handoff_record_to_dict
