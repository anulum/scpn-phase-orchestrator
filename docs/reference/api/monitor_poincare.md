# Poincare section monitor

The Poincare monitor extracts recurrence sections from continuous trajectories.
For oscillator systems this turns a high-dimensional flow into a sequence of
section crossings, making periodic locking, quasi-periodicity, and chaotic
return-time spread measurable without assuming a particular phase model.

## Use cases

- Detect stable limit-cycle entrainment by checking that return times converge.
- Compare phase-oscillator regimes before and after coupling or control changes.
- Build low-dimensional recurrence maps for nonlinear trajectory diagnostics.
- Gate simulator regressions where crossing counts or interpolation times drift.

## Public boundary contract

`poincare_section(trajectory, normal, offset=0.0, direction="positive")`
accepts finite real 1-D or 2-D trajectories and a finite real section normal.
The normal length must match the state dimension. `direction` is limited to
`positive`, `negative`, or `both`.

`phase_poincare(phases, oscillator_idx=0, section_phase=0.0)` accepts finite
real phase histories with shape `(T, N)`, rejects boolean aliases, checks the
oscillator index, and computes crossings modulo `2*pi`.

## Direct accelerator boundary contract

The direct Go, Julia, and Mojo wrappers validate before loading their optional
runtimes:

- flattened trajectory and phase buffers must be real finite `float64` vectors;
- `t`, `d`, and `n` must be non-boolean positive integers;
- flattened buffer lengths must exactly match `t*d` or `t*n`;
- normals must be finite real vectors with length `d`;
- `direction_id` is limited to `0`, `1`, or `2`;
- `oscillator_idx` must be in `[0, n)`;
- `offset` and `section_phase` must be finite real scalars.

This keeps the Python, Go, Julia, and Mojo surfaces aligned: invalid physics
states fail deterministically in Python before optional runtime loading or FFI
marshalling.
