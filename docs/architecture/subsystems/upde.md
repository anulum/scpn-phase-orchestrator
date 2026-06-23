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
`JaxUPDEEngine` / `JaxStuartLandauEngine`. (Root ARCHITECTURE.md says "12";
the verified count is higher.) Papers cited in code (unvouched): Acebrón 2005,
Filatrella–Nielsen–Mallick 2008, and others.

## Backends

Dispatched through `upde/_run.py` with the fastest-first chain
**Rust → WebGPU → Mojo → Julia → Go → Python**; the per-language forwarder
modules (`_engine_go.py`, `_engine_mojo.py`, …) point to
`experimental/accelerators/upde/`. Rust paths exist for the standard, sparse,
sheaf, geometric, inertial, and Ott–Antonsen engines.

## Wiring

Constructed by `api.Orchestrator`, `runtime/simulation.simulate`, and the
`server` simulation state. Output (`UPDEState`, order parameter) feeds `monitor/`
and, through it, `supervisor/`. `coupling/` supplies the `knm` each step.

## Scope boundaries

- JAX engines are marked `# pragma: no cover` (untested in standard CI).
- `bayesian.py` raises `NotImplementedError` for non-NumPy uncertainty backends.
- The splitting engine documents symplectic reversibility but has no test
  asserting it.
