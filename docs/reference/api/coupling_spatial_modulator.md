# Coupling — Spatial Modulator

`SpatialCouplingModulator` is the reusable distance-coupling primitive for
moving oscillator systems. It converts a reviewed base coupling matrix into a
position-aware coupling matrix before UPDE, Swarmalator, Doppler, or moving-frame
engines consume it.

## Mathematical contract

For the default moving-agent and MIF contract:

```text
K'_ij = K_base * K_ij / (1 + ||x_i - x_j||),   K'_ii = 0
```

The diagonal is always zero because self-coupling is not physical for the
standard UPDE contract. Inputs must be finite, real-valued, non-boolean arrays;
numeric-string aliases are rejected before float coercion. Base coupling
matrices must be square and zero diagonal; distance matrices from custom
distance functions must be symmetric, non-negative, and zero diagonal.

## Decay forms

| `decay_form` | Formula | Use case |
|---|---|---|
| `inverse_plus_one` | `1 / (1 + d)` | MIF/FRC kinematic merging, mobile sensors, robots, and moving power assets where coupling decays smoothly without a singularity |
| `exponential` | `exp(-d / ell)` | Media with characteristic propagation or attenuation length |
| `power_law` | `(1 + d / ell)^(-p)` | Scale-free or long-range mobile coupling |
| `inverse_distance` | `1 / sqrt(d^2 + epsilon)` | Exact compatibility with the existing Swarmalator phase-distance kernel |

## Pipeline wiring

```text
positions(t) + reviewed K_nm
        │
        ▼
SpatialCouplingModulator.modulate()
        │
        ▼
UPDEEngine / Swarmalator / future Doppler and MovingFrame engines
        │
        ▼
coherence, Lyapunov, entropy-production, and merge-window monitors
```

The Python reference is paired with Rust, Go, Julia, and Mojo accelerator
surfaces. The benchmark gate records each declared backend slot and accepts only
when available backends match the Python reference within the documented
tolerance.

The public dispatcher and direct accelerator wrappers share the same
spatial-modulator output validator. Public positions, base coupling matrices,
scalar decay controls, direct accelerator counts/forms/flat buffers, Rust
wrapper returns, optional backend outputs, and direct Julia raw returns are
validated before dtype coercion, so numeric-string aliases, boolean aliases,
complex aliases, non-finite values, wrong cardinality, and non-zero diagonals are
rejected instead of being widened into apparently valid float matrices. Public
dispatch still returns an `(n, n)` matrix to callers; backend fallback is
reserved for loader or runtime unavailability, not malformed backend physics
evidence.

## Why spatial modulation is an operations control

- In moving populations, coupling strength should not be treated as static.
  Distance-aware modulation makes control decisions sensitive to geometry drift
  before supervisory actions are emitted.
- The kernel-level invariant `K'_ii = 0` prevents self-loop amplification in both
  mobile and static modes.
- Row-wise decay choices are what lets teams keep a single coupling topology and
  switch physical assumptions per domain.

## API

```python
import numpy as np
from scpn_phase_orchestrator.coupling import SpatialCouplingModulator

knm = np.array([[0.0, 1.0], [1.0, 0.0]])
positions = np.array([[0.0], [3.0]])

modulator = SpatialCouplingModulator(K_base=0.5)
modulated = modulator.modulate(knm, positions)
# modulated[0, 1] == 0.5 / (1 + 3)
```

`jacobian_positions(positions)` returns the analytical derivative of the
modulation matrix with respect to the position array. The shape is
`(n, n, n, dim)`, where `J[i, j, a, d] = dM[i, j] / dx[a, d]`.

## Benchmark

```bash
PYTHONPATH=src python benchmarks/spatial_modulator_benchmark.py --parity-gate --sizes 16 --dim 2 --calls 3
```

Timing fields are local non-isolated regression evidence unless the benchmark
metadata records CPU/core isolation and host-load controls. They are not
production throughput claims.

::: scpn_phase_orchestrator.coupling.spatial_modulator
