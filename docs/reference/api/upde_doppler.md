# UPDE Doppler Engine

`DopplerEngine` is the PHA-C.2 kinematic phase-dynamics surface for moving
oscillator systems where relative velocity shifts the effective natural
frequency before phase coupling is applied. It is intended for counter-moving
plasmoids, moving sensor swarms, acoustic/clock synchronisation under motion,
and any domain where phase lock depends on velocity-corrected detuning rather
than static `omega` alone.

## Mathematical contract

For each outer integration step the engine resolves `omega(t)` and scalarises
velocity into one value per oscillator. The effective frequency is

```text
omega_eff_i(t) = omega_i(t) + D_i(t)
D_i(t) = s * sum_j |K_ij| * (v_i - v_j) / (|v_i| + epsilon) / sum_j |K_ij|
```

where `s` is `doppler_strength`, `epsilon` is `doppler_epsilon`, and rows with
no active coupling receive zero Doppler correction. The row normalisation keeps
`doppler_strength` independent of graph degree while the active `K_nm` topology
still determines which relative velocities are physically coupled.

Scalar velocities use the signed values directly. Vector velocities are reduced
to scalar speeds by Euclidean norm unless `velocity_axis` is supplied, in which
case velocities are projected onto the normalised axis so counter-propagating
motion keeps its sign.

## Public API

```python
import numpy as np
from scpn_phase_orchestrator.upde import DopplerEngine

knm = np.array([[0.0, 5.0], [5.0, 0.0]])
velocities = np.array([300.0, -300.0])
omega = np.array([-2.0, 2.0])

engine = DopplerEngine(
    2,
    omega=omega,
    k_nm=knm,
    alpha=0.0,
    dt=1.0e-3,
    velocities=velocities,
    solver="euler",
)
phases = engine.run(n_steps=2_000)
print(phases, engine.doppler_term)
```

`velocities` may be a fixed array or a callable `velocities(t) -> array`.
`omega` may use the same fixed/callable forms supported by `UPDEEngine`.

## Backend contract

Doppler integration is implemented as a schedule-backed UPDE run:

1. Resolve `omega_schedule[step, i]`.
2. Resolve `velocity_schedule[step, i]`.
3. Compute graph-weighted `D_i`.
4. Integrate one UPDE outer step with `omega_eff = omega + D`.

The source contract exists for Python, Rust/PyO3, Go, Julia, and Mojo. The
benchmark gate records unavailable optional runtimes explicitly rather than
silently skipping parity evidence.

```bash
PYTHONPATH=src python benchmarks/upde_doppler_benchmark.py --parity-gate
```

Committed local benchmark artefacts are regression/parity evidence only unless
rerun with the repository benchmark-isolation protocol.

## Failure boundaries

`DopplerEngine` fails closed on:

- non-finite or non-real phases, omega schedules, velocities, `K_nm`, or `alpha`;
- boolean and object-dtype numeric aliases;
- non-zero self-coupling diagonal in `K_nm`;
- non-positive `doppler_epsilon`;
- malformed scalar or vector velocity shapes;
- backend outputs outside `[0, 2*pi)`.

::: scpn_phase_orchestrator.upde.doppler.DopplerEngine

::: scpn_phase_orchestrator.upde.doppler.doppler_term

::: scpn_phase_orchestrator.upde.doppler.doppler_run
