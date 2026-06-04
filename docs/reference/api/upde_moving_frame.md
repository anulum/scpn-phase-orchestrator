# UPDE Moving-Frame Engine

`MovingFrameUPDEEngine` is the PHA-C.3 kinematic phase-dynamics surface for
systems where each oscillator has both a phase and an absolute axial coordinate
in a chamber-fixed reference frame. It composes two existing production
contracts:

- `SpatialCouplingModulator` computes distance-dependent `K_nm(z)` from the
  current axial positions.
- `DopplerEngine` computes velocity-dependent detuning from the active coupling
  graph.

The engine then advances phase and position as one coupled step. This is the
surface to use when phase synchronisation depends on moving geometry rather
than only on oscillator labels.

## Use cases

Use the moving-frame engine for domains where a fixed graph is physically
incorrect:

- counter-propagating plasmoid or particle packets approaching a chamber
  reference point;
- mobile sensor or clock swarms where distance and relative velocity alter
  synchronisation;
- acoustic, RF, or oscillator networks with line-of-flight detuning;
- digital-twin studies where collision or merge timing must be tied to the
  same integration clock as phase dynamics;
- MIF-style axial merger scenarios that require both `R(t)` and `z_i(t)`.

## Mathematical contract

For every outer integration step `s`, with axial positions `z_i(s)` and scalar
velocities `v_i(s)`:

```text
K_eff_ij(s) = K_base_ij * g(|z_i(s) - z_j(s)|)
D_i(s) = doppler_strength * sum_j |K_eff_ij| * (v_i - v_j) / (|v_i| + eps)
         / sum_j |K_eff_ij|
omega_eff_i(s) = omega_i(s) + D_i(s)
theta(s + dt) = UPDE(theta(s), omega_eff(s), K_eff(s), alpha, dt)
z_i(s + dt) = z_i(s) + v_i(s) * dt
```

`g(distance)` is the configured `SpatialCouplingModulator` decay kernel:
`inverse_plus_one`, `exponential`, `power_law`, or `inverse_distance`. The
position update is ballistic over each outer step, which is exact for fixed
velocity over that step and consistent with the row-major velocity schedule used
by the backend contract.

## Public API

```python
import numpy as np
from scpn_phase_orchestrator.coupling import SpatialCouplingModulator
from scpn_phase_orchestrator.upde import MovingFrameUPDEEngine

knm = np.array([[0.0, 0.4], [0.4, 0.0]])
engine = MovingFrameUPDEEngine(
    2,
    omega=np.array([0.1, -0.1]),
    k_nm=knm,
    alpha=0.0,
    dt=1.0e-9,
    positions_t0=np.array([-1.0e-6, 1.0e-6]),
    velocities=np.array([500.0, -500.0]),
    spatial_modulator=SpatialCouplingModulator(K_base=0.8),
    doppler_strength=0.01,
    solver="rk45",
)

phases = engine.run(n_steps=2)
positions = engine.positions
near_reference = engine.collision_imminent(threshold_m=1.0e-9)
```

`positions_t0` must be a finite axial vector with shape `(n,)`. `velocities`
may be a fixed vector or a callable `velocities(t) -> array`, matching the
Doppler schedule contract. `omega` may be fixed or callable, matching the
standard `UPDEEngine` time-varying frequency contract.

## Collision predicate

`collision_imminent(threshold_m=...)` checks current distance to the reference,
next-step distance under the currently resolved velocity, and sign-crossing of
the chamber reference. With `threshold_m=0.0`, exact crossing is still detected.
This makes it useful for merge-window monitors without forcing every caller to
manually inspect signed positions.

## Backend contract

The backend-neutral function `moving_frame_run(...)` accepts row-major schedules
and returns a flat vector:

```text
[final_phase_0, ..., final_phase_n-1, final_z_0, ..., final_z_n-1]
```

Python, Rust/PyO3, Go, Julia, and Mojo source surfaces share the same contract.
Optional accelerator runtimes are feature-detected; unavailable runtimes are
reported by the benchmark rather than hidden.

```bash
PYTHONPATH=src python benchmarks/upde_moving_frame_benchmark.py --parity-gate
```

Committed benchmark JSON is local regression and parity evidence only. It is
not a production throughput claim unless rerun under the repository benchmark
isolation protocol.

## Failure boundaries

The moving-frame engine fails closed on:

- non-finite, complex, object-dtype, or boolean phase, position, omega,
  velocity, `K_nm`, or `alpha` inputs;
- non-zero self-coupling diagonal in the base `K_nm`;
- malformed schedule shapes or mismatched oscillator counts;
- negative collision thresholds;
- non-positive spatial decay scale, spatial epsilon, or Doppler epsilon;
- backend output with non-finite positions or phases outside `[0, 2*pi)`.

::: scpn_phase_orchestrator.upde.moving_frame.MovingFrameUPDEEngine

::: scpn_phase_orchestrator.upde.moving_frame.moving_frame_run

::: scpn_phase_orchestrator.upde.moving_frame.moving_frame_run_python
