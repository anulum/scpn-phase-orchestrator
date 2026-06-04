# UPDE — Time-varying natural frequencies

`UPDEEngine` supports natural-frequency schedules for systems where each
oscillator's intrinsic angular frequency is a function of outer-step time:

```text
dtheta_i/dt = omega_i(t) + sum_j K_ij sin(theta_j - theta_i - alpha_ij)
              + zeta sin(Psi - theta_i)
```

This is the PHA-C.5 surface required by moving-frame and Doppler workflows. It
keeps the ordinary fixed-omega API intact while adding two production-safe forms:

| Form | API | Use case |
| --- | --- | --- |
| Fixed constructor omega | `UPDEEngine(..., omega=array)` | Reuse the same frequency vector across many `step()` or `run()` calls without passing it every time. |
| Callable omega | `UPDEEngine(..., omega=lambda t: omega0 + slope*t)` | Chirps, drifting clocks, moving-agent frequency shifts, thermal detuning, Doppler preparation, or measured frequency schedules. |

Existing calls such as `engine.step(phases, omegas, knm, zeta, psi, alpha)` and
`engine.run(phases, omegas, knm, zeta, psi, alpha, n_steps)` remain valid.
Configured omega is used only when the call does not provide an explicit
frequency vector.

## Example

```python
import numpy as np
from scpn_phase_orchestrator.upde import UPDEEngine

n = 4
dt = 1.0e-6
omega0 = np.array([1.0, 1.1, 0.9, 1.2])
slope = np.array([-0.01, 0.0, 0.02, -0.015])
knm = np.zeros((n, n))
alpha = np.zeros((n, n))
phases = np.zeros(n)

engine = UPDEEngine(
    n,
    dt=dt,
    method="euler",
    omega=lambda t: omega0 + slope * t,
)

final_phases = engine.run(phases, knm=knm, alpha=alpha, n_steps=100)
current_omega = engine.omega_current
current_time = engine.time
```

For zero coupling and no drive, a linear schedule has the analytic reference

```text
theta_i(T) = theta_i(0) + omega_i(0) T + 0.5 slope_i T^2.
```

The module-specific regression tests use a 1 microsecond step to keep the Euler
schedule error within the 1 ppm acceptance envelope.

## Backend contract

Callable frequencies are resolved into a finite real-valued schedule matrix with
shape `(n_steps, n_oscillators)`. The stateless schedule runner then dispatches
that matrix through the same backend chain as fixed-omega UPDE:

| Backend | Schedule surface |
| --- | --- |
| Rust | `PyUPDEStepper.run_omega_schedule()` |
| Go | `UPDERunOmegaSchedule` in `go/upde_engine.go` |
| Julia | `UPDEEngineJL.upde_run_omega_schedule()` |
| Mojo | `RUN_SCHEDULE` operation in `mojo/upde_engine.mojo` |
| Python | `upde_run_omega_schedule_python()` |

Each backend receives the same row-major schedule. Validation rejects boolean,
complex, non-finite, empty, wrong-rank, and wrong-width schedules before any
integration result is accepted.

## Benchmark gate

Run the local parity gate:

```bash
PYTHONPATH=src python benchmarks/upde_time_varying_omega_benchmark.py --parity-gate
```

The committed result is local regression evidence only. It records backend
availability and parity, but does not make production timing claims unless the
run is repeated under the repository benchmark-isolation requirements.

## Public API

:::: scpn_phase_orchestrator.upde.engine.upde_run_omega_schedule
