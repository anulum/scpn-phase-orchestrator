# Subsystem: `upde` — phase-ODE integrator family

The numerical core: integrates coupled phase dynamics. 74 files, ~18.3k LOC.

## Inputs

`step(phases, omegas, knm, zeta, psi, alpha)`:

- `phases` — `NDArray[float64]`, shape `(N,)`, radians.
- `omegas` — `(N,)`, natural frequencies (rad/s).
- `knm` — `(N, N)`, coupling matrix, zero diagonal (validated).
- `alpha` — `(N, N)`, Sakaguchi phase-lag matrix.
- `zeta` — scalar external-drive strength; `psi` — drive reference phase.
- `IntegrationConfig(dt, method ∈ {euler, rk4, rk45}, substeps, atol, rtol)`.

## Outputs

- New `phases`, wrapped to `[0, 2π)`.
- `UPDEState` — frozen diagnostic record: per-layer `LayerState`,
  `cross_layer_alignment`, `stability_proxy`.
- Order parameter `(R ∈ [0,1], ψ)` via `order_params.compute_order_parameter`.

## Processing model

Integration methods: forward Euler, classical RK4, and adaptive RK45
(Dormand–Prince 5/4 with PI step control). Geometric variants use an
exponential map on the unit circle; the stochastic layer uses Euler–Maruyama;
the Ott–Antonsen reduction is an O(1) mean-field predictor (Bessel `I0`, `I1`).

### Engine variants (14–15 implemented)

`UPDEEngine` (standard Kuramoto), `StuartLandauEngine` (phase + amplitude, Hopf),
`InertialKuramotoEngine` (2nd-order swing), `SwarmalatorEngine`,
`SimplicialEngine` (3-body), `TorusEngine` (symplectic), `DelayedEngine`,
`SplittingEngine` (operator splitting), `SheafUPDEEngine`, `SparseUPDEEngine`
(CSR), `HypergraphEngine`, `DopplerEngine`, `MovingFrameUPDEEngine`, and JAX
`JaxUPDEEngine` / `JaxStuartLandauEngine`. Papers cited in code (unvouched):
Acebrón 2005, Filatrella–Nielsen–Mallick 2008, and others.
The two JAX engines validate array source types on the host before JAX device
conversion: boolean, complex, and numeric-string aliases fail closed while
finite real numeric-object arrays remain compatible.

## Backends

Dispatched through `upde/_run.py` with the fastest-first chain
**Rust → WebGPU → Mojo → Julia → Go → Python**; the per-language forwarder
modules (`_engine_go.py`, `_engine_mojo.py`, …) point to
`experimental/accelerators/upde/`. Rust paths exist for the standard, sparse,
sheaf, geometric, inertial, and Ott–Antonsen engines.
The sparse engine has its own Python/optional-Rust selection, independent of
the stateless dense chain. Finite real input phases may lie outside the torus;
returned native phases must have exact oscillator cardinality and lie inside
`[0, 2*pi)`. A positive finite native timestep diagnostic is copied into public
`last_dt` only after output validation. RK45 reports a next-step proposal.
Readonly phase/frequency/CSR/lag inputs are snapshotted before mutable native
coupling admission; aliased inputs keep entry-time values during a run. The
native binding requires writable contiguous coupling, whereas the NumPy-only
fallback does not mutate it. Public strided inputs are copied to contiguous
buffers. See [the CSR contract](../../reference/api/upde.md#csr-buffer-ownership-and-timestep-diagnostics).
The sheaf boundary validates phase, frequency, restriction-map, and drive-target
source types before conversion, including temporal aliases. It selects its own
Python or PyO3/Rust implementation independently of the scalar dispatch chain.
Its RK45 uses proportional control of the maximum scaled component error and
traverses the complete configured outer interval. Its Rust result is accepted
only as a finite real flattened `N * D` torus state; `last_dt` reports a positive
finite next substep proposal bounded by the outer interval. Direct native
substeps partition that interval. Refusal preserves input storage and the
pre-call proposal across the entire step or batch. The solver returns a phase
matrix; consumers supply any `UPDEState` diagnostics. See the
[cellular-sheaf contract](../../reference/api/upde.md#cellular-sheaf-engine).
Finite-difference and optional JAX adjoint paths share one pre-execution state
contract for phase, frequency, coupling, and phase-lag arrays. Counts and
perturbation/timestep scalars are validated before arithmetic or optional JAX
imports, so backend availability cannot mask malformed public input.
Bayesian UPDE uses one source-type-aware array boundary for direct inputs,
Gaussian distribution parameters, fitted posterior data, and custom
distribution samples. Boolean, complex, and numeric-string aliases are rejected
before `float64` conversion, and finite real drive controls are validated before
Monte Carlo rollout.
The forward and variational prediction models apply the same source-type-aware
contract to phase, frequency, predicted-state, observed-state, and precision
vectors. Coercive aliases fail before prediction, free-energy arithmetic, or
online state updates, while finite real numeric object arrays remain compatible.
Public Strang-splitting phase, frequency, coupling, phase-lag, and optional-
backend phase arrays reject boolean, complex/object-complex, and numeric-string
aliases before conversion while preserving finite real numeric-object arrays.
Existing shape, zero-self-coupling, and torus-output checks remain authoritative.
Geometric direct Go, Julia, and Mojo bridges validate torus phase, frequency,
coupling, lag, scalar, count, and backend-output payloads before optional native
runtime loading; numeric-string aliases are rejected at the Python boundary.

## Wiring

Constructed by `api.Orchestrator`, `runtime/simulation.simulate`, and the
`server` simulation state. Output (`UPDEState`, order parameter) feeds `monitor/`
and, through it, `supervisor/`. `coupling/` supplies the `knm` each step.

## Scope boundaries

- JAX engines are marked `# pragma: no cover` (untested in standard CI).
- `bayesian.py` raises `NotImplementedError` for non-NumPy uncertainty backends.
- The splitting engine documents symplectic reversibility but has no test
  asserting it.
