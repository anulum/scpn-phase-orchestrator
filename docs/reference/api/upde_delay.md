<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Delayed Kuramoto API -->

# Time-delayed Kuramoto integration

`DelayedEngine` integrates a fixed discrete delay with simultaneous explicit
Euler updates. `DelayBuffer` independently stores finite phase snapshots.
The implementation is a numerical model; physical signal interpretation and
validation belong to the consuming domain.

## Mathematical recurrence

For timestep `dt`, delay count $d$ and accepted step $k$, the source phase is

$$
\widetilde\theta_j^k =
\begin{cases}
\theta_j^k, & k < d, \\
\theta_j^{k-d}, & k \ge d.
\end{cases}
$$

The update uses row-major coupling from source $j$ into receiver $i$:

$$
\theta_i^{k+1} = \Pi\!\left(\theta_i^k + \Delta t
\left[\omega_i + \sum_j K_{ij}
\sin(\widetilde\theta_j^k-\theta_i^k-\alpha_{ij})
+ \zeta\sin(\Psi-\theta_i^k)\right]\right).
$$

The physical delay is $\tau=d\,\Delta t$. Early steps use their **current**
source phases while history accumulates. This differs from supplying a
prescribed continuous prehistory or holding the initial phase fixed throughout
warmup. The API does not accept a prehistory function or distributed delays.

All CPU owners use float64. The projection $\Pi$ returns phases in
`[0, 2*pi)` with positive zero for either sign of zero and for a negative
remainder rounded to the upper endpoint. Interior representable values,
including `nextafter(2*pi, 0)`, remain intact. Nonfinite numerical results are
refused. Very large unwrapped phases remain subject to float64/libm range
reduction and cancellation limits; range conformance alone does not establish
accurate phase differences for those inputs.

Delay can change synchrony and produce multiple dynamical regimes. A universal
cosine law for the order parameter is not provided by this solver. For the
particular noiseless, identical, globally coupled continuous model,
[Yeung and Strogatz](https://arxiv.org/pdf/chao-dyn/9807030), equations 5–6,
derive collective-frequency self-consistency and orbital stability conditions
for uniformly rotating synchrony. Those assumptions differ from arbitrary
matrices, forcing and the discrete warmup defined above.

## Public Python API

```python
import numpy as np
from scpn_phase_orchestrator.upde.delay import DelayedEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

phases = np.array([0.4, 1.1])
omegas = np.array([1.0, -0.7])
knm = np.array([[0.0, 0.5], [-0.2, 0.0]])
alpha = np.full((2, 2), 0.01)
engine = DelayedEngine(2, dt=0.01, delay_steps=3)
final = engine.run(
    phases, omegas, knm, zeta=0.2, psi=0.3, alpha=alpha,
    n_steps=40, backend="python",
)
r, mean_phase = compute_order_parameter(final)
```

The constructor requires positive oscillator and delay counts fitting a
platform index, and a positive finite timestep. Phases and frequencies have
shape `(n,)`; coupling and lag have shape `(n, n)`. Frequencies are in rad/s
when the timestep is in seconds. Coupling and drive carry angular-frequency
units, and phase lags/reference phases are in radians. Omitting `alpha` uses
a zero lag matrix. Signed frequencies and directed or signed coupling are
supported; no symmetry or zero-diagonal constraint is imposed here.

Inputs must be finite real measurements. Boolean, numeric-string, complex and
temporal aliases are refused before numerical dispatch. Results use independent
storage; callers retain their input arrays. Public `run()` requires a positive
`n_steps` count.

### Stateful steps and fresh batches

`step()` is the NumPy stateful interface. Pass each accepted output into the
next call to advance a sequence:

```python
current = phases.copy()
for _ in range(40):
    current = engine.step(current, omegas, knm, zeta=0.2, psi=0.3, alpha=alpha)
```

A refused step preserves the accepted history, so a later valid call can
continue it. The legacy `step_idx` argument is accepted for call compatibility;
accepted calls determine the history position.

Each `run()` begins a fresh batch and leaves the instance's `step()` history
unchanged. Splitting a stateless batch into two `run()` calls restarts warmup at
the second call. Use `step()` for continuous history across calls. The internal
Rust `DelayedStepper` is stateful, including across its own `run()` calls.

### Explicit runtime selection

`run(..., backend="python")` requires NumPy. The explicit names `rust`, `go`,
`julia` and `mojo` require the corresponding actual owner. An unavailable owner
raises `ImportError`; invalid numerical inputs/results raise `ValueError`.
Errors from an explicitly requested owner propagate to the caller.

The default `backend="auto"` retains the configured priority
`rust → mojo → julia → go → python`. Load-time unavailability proceeds through
that order. A refusal from the selected native batch falls back to NumPy.
`ACTIVE_BACKEND` and `AVAILABLE_BACKENDS` describe import-time admission;
explicit selection provides reproducible owner-specific execution.

| Owner | Native surface |
|---|---|
| Rust | PyO3 `spo_kernel.delayed_kuramoto_run_rust` |
| Go | `go/libdelay.so`, `DelayedKuramotoRun` C ABI |
| Julia | `julia/delay.jl`, `DelayJL.delayed_kuramoto_run` |
| Mojo | `mojo/delay_mojo`, `DELAY` stdin/stdout protocol |
| Python | Validated simultaneous Euler recurrence in `upde.delay` |

Julia admission requires a running `juliacall.Main`, a loadable side-file and
its numerical entry point. Installing Julia's Python bridge alone does not
provide that side-file. Missing, broken or incomplete Julia artifacts remain
unavailable; valid source-checkout assets enable the Julia owner.

Direct Go/Julia/Mojo bridge functions validate flattened matrices before
loading their runtimes. Their zero-step calls return independent validated
input copies, preserving unwrapped phases, without requiring a foreign
artifact. Positive batches require actual execution. Go refuses counts that
cannot fit its signed C count ABI. A delay longer than a finite batch is
internally shortened to that batch length, preserving its warmup semantics
and avoiding unused history allocation.
The C entry point also refuses matrix byte extents that exceed a platform index;
native callers remain responsible for supplying full valid buffers.

The Rust batch API returns `SpoResult<Vec<f64>>`; its PyO3 boundary maps
refusals to `ValueError`. The native argument names remain
`phases_init, omegas, knm_flat, alpha_flat, n, zeta, psi, dt, delay_steps, n_steps`.
Zero native steps validate and copy the supplied phases. Native consumers must
be qualified against the actual built artifact, beyond its version label.

## Standalone snapshot buffer

```python
from scpn_phase_orchestrator.upde.delay import DelayBuffer

buffer = DelayBuffer(n_oscillators=2, max_delay_steps=3)
buffer.push(np.array([0.4, 1.1]))
buffer.push(np.array([0.5, 1.2]))
latest = buffer.get_delayed(1)  # [0.5, 1.2]
previous = buffer.get_delayed(2)  # [0.4, 1.1]
```

Lookup uses a **one-based snapshot index**: one is the latest stored snapshot.
A lookup beyond available history returns `None`; zero and negative indexes
are invalid. Both insertion and retrieval copy their data. Mutating a returned
snapshot cannot corrupt retained history. `length` gives current occupancy;
`clear()` discards all snapshots. The buffer does not infer physical time
intervals from these indexes.

## Comparison and verification

Run the maintained comparison against actual provisioned owners:

```bash
python -m benchmarks.delay_benchmark --parity-gate \
  --n 8 --delay-steps 3 --n-steps 80 --calls 3 \
  --require-backends python rust go julia mojo
```

Each owner executes through explicit public selection. The comparison binds
both the NumPy result and an independent scalar saved-history recurrence,
checks pure rotation, half-open endpoints and positive zero, and records
repeated warmed-call timings. Missing required owners, runtime refusals or
failed contracts make the CLI exit nonzero. Every declared owner retains an
honest result slot; a failed owner receives no successful phase hash or timing.
The aggregate reference suite retains the `delay_polyglot` key.

The [recorded October 20 comparison](../data/delay_real_runtime_2026-10-20.json)
executes all five owners for 2, 16 and 257 oscillators, with five warmed repeated
calls per owner. Its Rust kernel uses a development build; these timings do not
qualify release throughput. The current CLI accepts the rebuilt kernel and
rejects the genuine original installed kernel at the corrected phase endpoint.
Source and binary hashes distinguish these same-version artifacts.

Configured parity tolerances are `1e-9` for Rust/Go/Julia and `1e-6` for Mojo's
text round trip. Shared-host diagnostic timings are not production throughput
or speedup claims. The batch costs $O(TN^2)$ arithmetic and needs history
proportional to $N\min(d,T)$, in addition to matrices and numerical scratch.
Native implementations retain different phase versus sine/cosine histories;
there is no single byte count common to every owner.

The owning tests cover public buffer/step/batch behaviour, actual compiled
execution, scalar/parallel numerical paths, source delay, measurement refusal,
recovery, caller ownership, and real Julia absent/broken/present assets.
Python bridge coverage and native source instrumentation are separate scopes.

## Technical reference

::: scpn_phase_orchestrator.upde.delay
    options:
        show_root_heading: true
        members_order: source

## Source

- Public engine and buffer: `src/scpn_phase_orchestrator/upde/delay.py`
- Direct bridges and validation: `src/scpn_phase_orchestrator/experimental/accelerators/upde/_delay_*.py`
- Rust: `spo-kernel/crates/spo-engine/src/delay.rs`
- PyO3 boundary: `spo-kernel/crates/spo-ffi/src/delayed_boundary.rs`
- Go: `go/delay.go`, owning contracts `go/delay_test.go`
- Julia: `julia/delay.jl`, owning contracts `julia/test_delay.jl`
- Mojo: `mojo/delay.mojo`
- Comparison: `benchmarks/delay_benchmark.py`
- Owning tests: `tests/test_delay.py`, `tests/test_delay_backends.py`,
  `tests/test_delay_real_runtime.py`, `tests/test_delay_julia_absence.py`,
  `tests/test_delay_measurement_inputs.py`, `tests/test_delay_config_validation.py`,
  `tests/test_delayed_engine_workflow.py` and `tests/test_delay_benchmark.py`.
