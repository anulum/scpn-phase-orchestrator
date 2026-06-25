# Subsystem: `nn` — differentiable JAX/Equinox backend

A fully differentiable re-implementation of the oscillator dynamics for
gradient-based learning, inference, and control. 27 files, ~8.5k LOC. Optional
`[nn]` extra; a **parallel track**, not part of the default step loop.

## Inputs

`jax.Array` phases `(N,)`, omegas `(N,)`, coupling `K` `(N, N)`, `dt`, static
`n_steps`; for inverse inference, an observed trajectory `(T, N)`; PPO scenario
and batch types for the supervisor.

## Outputs

`(final_phases, trajectory (n_steps, N))`; inferred `(K, omegas)`; scalar order
parameters and losses; `SupervisorAction(deltas_K, deltas_omega, …)` and PPO loss
with auxiliaries.

## Processing model

JIT-compiled, `vmap`-friendly, autodiff throughout; RK4 via `lax.scan`. Layers:
`KuramotoLayer`, `SimplicialKuramotoLayer`, `StuartLandauLayer`,
`UDEKuramotoLayer` (physics + tanh-bounded neural residual), `ThetaNeuronLayer`,
`PhaseAutoencoder`. Functional kernels include order parameter / PLV, a spectral
alignment function (SAF) with `eigh` and conjugate-gradient paths, chimera
detection, an oscillator Ising machine (graph colouring / max-cut / QUBO), the
Balloon–Windkessel BOLD model, and a Kuramoto reservoir. Inverse coupling has
analytical (least-squares), hybrid (analytical + Adam), and pure-gradient
variants. `nn/supervisor/` is a **package** (11 modules) implementing a PPO-
trained `DifferentiableSupervisorPolicy` (it is not a single god-file).

## Backends

This *is* the JAX path; `HAS_JAX = find_spec("jax")`. Hardware-agnostic JIT
(GPU when present). No overlap with the Rust/NumPy `upde` engines — distinct
stacks. Optional import is lazy: accessing an `nn` symbol without JAX raises a
clear install error.

## Wiring

Reachable only through the CLI `verification` command (supervisor baseline
experiments, JAX-gated). Not used by the server, gRPC, or
`runtime.simulation`. An orthogonal research/ML harness.

## Scope boundaries

The four former 1.0-blocking validation gaps are resolved in the public xfail
register: UDE extrapolation stays finite, training preserves symmetric `K`, the
analytical inverse handles uncoupled `K=0` data, and the entropy-production test
uses the production dissipation estimator. Remaining `xfail`s are tracked as
non-blocking precision, finite-size, heuristic-hardness, or test-design
limitations.
