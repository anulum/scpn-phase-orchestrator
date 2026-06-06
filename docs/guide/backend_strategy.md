# Backend Strategy

## Why this strategy matters

SPO exposes multiple numerical backends so teams can move between portability and
performance without changing orchestration code. This strategy is the control layer
that keeps that choice explicit and repeatable.

Without it, environment differences tend to become hidden defaults. With it, each
deployment state is mapped to one expected default and one explicit fallback path.

SCPN Phase Orchestrator has several implementation backends. The supported
path is deliberately narrow:

1. Rust FFI for production hot paths.
2. JAX for differentiable and accelerator-oriented workflows.
3. WebGPU for browser, mobile GPU, WebView, and edge deployments where native
   toolchains are unavailable.
4. NumPy/SciPy Python as the portable reference fallback.
5. Julia, Go, and Mojo as experimental parity or benchmark backends.

Use this page to decide which backend should carry a deployment.

## Support Tiers

| Tier | Backend | Status | Use for |
|------|---------|--------|---------|
| Primary | Rust FFI (`spo_kernel`) | production path | low-latency CPU stepping, monitors, coupling, supervisor hot paths |
| Primary | JAX | production path for differentiable work | gradient-based training, GPU/TPU experiments, learned policies |
| Portable accelerator | WebGPU | browser/edge package path | Euler UPDE stepping in WebGPU-capable browsers and edge runtimes |
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
else WebGPU bridge is explicitly configured or browser/edge package is selected
    -> use WebGPU bridge/package
else JAX workflow explicitly selected
    -> use JAX APIs
else
    -> use NumPy/SciPy Python fallback
```

This sequence is not only a runtime heuristic. It encodes the support boundary:
Rust and JAX carry production expectations; WebGPU and Python are constrained by
deployment context.

Julia, Go, and Mojo are not automatic production fallbacks. Treat them as
opt-in experiments until their maintenance cost is justified by measured value.

Use this as a decision chain for release and incident response: when a runtime
target cannot satisfy the first match, operators can predict which path will be
chosen and what observability/accuracy trade-offs follow.

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

This keeps optimization goals separate: use JAX for learning and gradient workflows,
Rust for deterministic production stepping and monitoring.

## WebGPU Path

WebGPU is the first browser/edge compute path. It is designed for deployments
where the host can expose a WebGPU runner or execute the generated browser
package, but cannot install native Rust, Julia, Go, or Mojo toolchains.

Use it when:

- the workload runs in a browser, WebView, mobile GPU, or edge JavaScript
  runtime,
- Python dispatch is wired through `SPO_WEBGPU_DISPATCH_BRIDGE=module:function`
  or the generated ES-module package is run directly,
- Euler UPDE stepping is sufficient for the deployment contract,
- `f32` WebGPU arithmetic is acceptable and documented,
- a native toolchain install is not acceptable.

Do not use it for certification-grade `f64` comparisons, formal safety
evidence, or differentiable training. Those remain Rust/JAX/NumPy duties.

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

## Decision workflow for production onboarding

When onboarding a new workload, use this order:

1. Confirm the deployment contract (latency, hardware, reproducibility).
2. Select the first backend path that satisfies both requirements and evidence.
3. Require parity validation for that workload before changing any default.
4. Document the fallback path for unsupported or transient backend outages.

This workflow prevents silent strategy drift between teams and environments.

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

Private shim modules whose filenames start with an underscore are not public
documentation targets. They exist to preserve the stable Python API while
isolating auxiliary language calls and backend availability checks. Document
the public dispatcher, parity expectation, and operational status instead of
duplicating one API page per `_go`, `_julia`, or `_mojo` implementation file.

## Contribution Rules

New backend code should answer these questions in the PR:

- What production workload does it improve?
- Which primary backend does it replace or complement?
- What is the measured speed, memory, or deployment advantage?
- Which parity tests prove equivalent numerical behaviour?
- What happens when the backend is missing?

Without those answers, prefer improving Rust, JAX, or the Python reference path.

## Practical deployment mapping

Use this quick mapping before switching backend defaults:

- **Latency-sensitive production API**: prefer Rust FFI and keep JAX as a separate
  research lane.
- **Differentiable research and autotune**: prefer JAX, then validate with Python
  reference parity for audit checks.
- **Browser or edge demo**: prefer WebGPU for deployment compatibility, and define
  explicit f32 constraints in the project notes.
- **Regulated review or offline investigation**: prefer Python reference or Rust
  with full replay gates enabled.

Each backend shift should include at least one documented parity command and one
documented replay command so teams can confirm both trajectory shape and replay
integrity after switching.

## Governance rule for backend experiments

Experimental backends (Julia, Go, Mojo) are only moved out of experiment status
after all three are true: measurable production benefit, parity evidence, and
maintenance ownership. Until then, treat their outputs as comparative artifacts,
not default execution policy.
