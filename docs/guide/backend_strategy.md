# Backend Strategy

SCPN Phase Orchestrator has several implementation backends. The supported
path is deliberately narrow:

1. Rust FFI for production hot paths.
2. JAX for differentiable and accelerator-oriented workflows.
3. NumPy/SciPy Python as the portable reference fallback.
4. Julia, Go, and Mojo as experimental parity or benchmark backends.

Use this page to decide which backend should carry a deployment.

## Support Tiers

| Tier | Backend | Status | Use for |
|------|---------|--------|---------|
| Primary | Rust FFI (`spo_kernel`) | production path | low-latency CPU stepping, monitors, coupling, supervisor hot paths |
| Primary | JAX | production path for differentiable work | gradient-based training, GPU/TPU experiments, learned policies |
| Reference | NumPy/SciPy Python | always available | correctness, portability, debugging, fallback execution |
| Experimental | Julia | benchmark/parity | numerical experiments that prove a clear advantage |
| Experimental | Go | benchmark/parity | deployment experiments that prove operational simplicity or speed |
| Experimental | Mojo | benchmark/parity | compiler/runtime experiments |

Experimental backends should not become the default path unless they show a
clear 5-10x production advantage for a real workload and have CI parity tests.

## Runtime Selection

The runtime preference is:

```text
Rust FFI available and supported for this module
    -> use Rust
else JAX workflow explicitly selected
    -> use JAX APIs
else
    -> use NumPy/SciPy Python fallback
```

Julia, Go, and Mojo are not automatic production fallbacks. Treat them as
opt-in experiments until their maintenance cost is justified by measured value.

## Rust Path

Rust is the primary acceleration path for CPU-bound production work.

Use it when:

- `spo_kernel` is installable on the deployment target,
- the workload is latency-sensitive,
- the hot path is already covered by Rust parity tests,
- deterministic CPU execution matters more than autodiff.

Build it with:

```bash
maturin develop --release -m spo-kernel/crates/spo-ffi/Cargo.toml
```

Python classes auto-detect `spo_kernel` and delegate supported hot paths with
no API change.

## JAX Path

JAX is the primary path for differentiable orchestration.

Use it when:

- coupling or policy parameters are learned,
- gradients need to flow through phase dynamics,
- GPU/TPU acceleration is part of the experiment,
- the workflow lives in `nn/`, UDE, or differentiable Kuramoto modules.

JAX should not replace the Rust path for ordinary CPU production stepping unless
measurements show it is better for that workload.

## Python Reference Path

The Python path is the correctness baseline. It is always available and should
remain readable, deterministic, and well tested.

Use it when:

- debugging a new model,
- reproducing a bug without native extensions,
- validating Rust or JAX parity,
- supporting platforms where native builds are not available.

The Python path is not a failure mode. It is the portable reference
implementation.

## Experimental Backends

Julia, Go, and Mojo backends stay experimental unless they meet all promotion
criteria:

- reproducible benchmark showing at least 5-10x improvement on a production
  workload, or a deployment capability Rust/JAX cannot provide,
- parity tests against the Python reference,
- CI coverage on the target platform,
- clear owner and maintenance story,
- documented fallback if the backend is unavailable.

If those criteria are not met, keep the backend under experiments, benchmarks,
or explicit optional code paths.

## Contribution Rules

New backend code should answer these questions in the PR:

- What production workload does it improve?
- Which primary backend does it replace or complement?
- What is the measured speed, memory, or deployment advantage?
- Which parity tests prove equivalent numerical behaviour?
- What happens when the backend is missing?

Without those answers, prefer improving Rust, JAX, or the Python reference path.
