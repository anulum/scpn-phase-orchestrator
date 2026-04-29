<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Backend fallback policy -->

# Backend Fallback Chain

SCPN Phase Orchestrator has two maintained execution paths for production use:

- **Rust FFI** for hot numerical kernels, supervisor primitives, extractors,
  monitors, and deployment-facing acceleration.
- **JAX** for differentiable models, neural layers, inverse problems, and GPU or
  autodiff workflows.

Pure Python remains the reference fallback. Auxiliary Go, Julia, and Mojo shims
exist only for selected kernels and should be treated as experimental unless a
maintained workload proves they are worth the CI and maintenance cost.

## Runtime Selection Patterns

There are two backend-selection patterns in the current codebase.

### Rust-or-Python Delegation

Most production modules use simple Rust detection:

```python
from scpn_phase_orchestrator._compat import HAS_RUST
```

If `spo_kernel` is importable, the module uses the Rust FFI hot path. If not,
the module stays on the NumPy/Python implementation with the same public API.

Examples:

| Area | Rust path | Fallback |
| --- | --- | --- |
| UPDE and Stuart-Landau steppers | `spo_kernel` PyO3 classes | NumPy/Python engines |
| Coupling and order parameters | `spo_kernel` functions | NumPy/Python functions |
| Monitor and supervisor primitives | `spo_kernel` classes/functions | Python implementations |
| Physical, informational, symbolic extractors | `spo_kernel` functions | SciPy/NumPy/Python extractors |

Use this path when runtime stability matters.

### Five-Backend Experimental Dispatch

Some research kernels expose a broader chain:

```text
rust -> mojo -> julia -> go -> python
```

The modules publish `ACTIVE_BACKEND` and `AVAILABLE_BACKENDS`, then dispatch to
the first importable backend. Python remains the final fallback.

Examples include backend-dispatched analysis kernels such as basin stability,
fractal dimension, Lyapunov, order-parameter, spectral, market, and related
research paths.

These shims are useful for parity experiments, but they are not the default
production recommendation. Keep them behind explicit module boundaries and
avoid making application behaviour depend on an auxiliary toolchain being
installed.

## JAX Path

JAX is not a fallback for Rust. It is a separate execution mode for workloads
that need differentiability or accelerator compilation.

Use JAX when you need:

- gradients through phase dynamics,
- trainable coupling matrices,
- inverse modelling from observed phase traces,
- neural-network layers,
- GPU/TPU compilation for large batched experiments.

Install with:

```bash
pip install scpn-phase-orchestrator[nn]
```

JAX modules expose their own availability checks, such as `HAS_JAX` in the JAX
UPDE engine. If JAX is missing, those modules raise an import or construction
error instead of silently switching to NumPy, because autodiff semantics would
change.

## Recommended Order by Workload

| Workload | Recommended path | Reason |
| --- | --- | --- |
| CLI simulation, domainpack validation, audit replay | Rust if installed, Python fallback | Stable API and deterministic fallback. |
| Real-time adapters, supervisor, monitors | Rust FFI | Lower latency and parity-covered behaviour. |
| Differentiable training, inverse coupling, neural layers | JAX | Gradients and JIT compilation are the core contract. |
| Research comparison of analysis kernels | Rust, then auxiliary backend, then Python | Useful for parity and benchmarking experiments. |
| Production deployment with optional backends unavailable | Python fallback or Rust FFI | Avoid hidden dependence on Go, Julia, or Mojo toolchains. |

## Numerical Parity Expectations

Every backend path must preserve the public mathematical contract of the
module. The tolerance depends on the kernel:

| Kernel class | Expected parity rule |
| --- | --- |
| Direct algebraic metrics | Float64-close to Python/Rust reference. |
| ODE integration | Same integrator, same step order, bounded floating-point drift. |
| Randomised Monte Carlo estimators | Python owns the seed where possible; compare deterministic summaries, not scheduler order. |
| JAX training paths | Compare physical invariants and loss trends, not byte-for-byte arrays. |

If a backend requires different RNG ownership, accumulation order, or
precision, document that in the module docstring and tests.

## Adding a Backend

Do not add another language path just because it is easy to call. A backend is
worth keeping only if it satisfies all of these:

1. It implements the same mathematical contract as the reference path.
2. It has parity tests against Python and, where relevant, Rust.
3. It has an installation story that does not break normal users.
4. It has benchmark evidence for its target workload.
5. It has a maintainer willing to keep CI and toolchain drift under control.

The minimum public surface for a dispatched module is:

```python
ACTIVE_BACKEND: str
AVAILABLE_BACKENDS: list[str]
```

The Python path must always remain available.

## Demotion and Removal Criteria

Auxiliary backends should be demoted to experimental docs, disabled by default,
or removed when any of these are true:

- no maintained workload depends on the backend,
- CI or local setup cost is disproportionate to the value,
- measured speedup is below the maintenance threshold for the workload,
- the backend cannot preserve parity without special cases,
- the upstream toolchain is unstable for supported platforms,
- Rust or JAX now covers the same workload with less operational cost.

The roadmap target is simple: Rust and JAX stay first-class; Go, Julia, Mojo,
and future auxiliary paths must keep earning their place.

## Operator Checklist

Before relying on a backend in a deployment:

- Confirm the selected backend at startup or in logs.
- Run the module parity tests in the target environment.
- Keep `spo_kernel` version aligned with the Python package when using Rust.
- Pin JAX, accelerator runtime, and driver versions for JAX deployments.
- Keep auxiliary backends out of production images unless explicitly needed.
- Record the backend in benchmark and audit metadata when comparing runs.

## Related Pages

- [Rust FFI Acceleration](rust_ffi.md)
- [Differentiable Kuramoto Layer](differentiable_kuramoto.md)
- [Performance Tuning](performance.md)
- [Testing](testing.md)
