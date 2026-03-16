# Rust FFI Acceleration

The `spo-kernel` Rust workspace provides 5-10x speedup for UPDE integration,
coupling construction, imprint modulation, and supervisor logic. Python classes
auto-detect the compiled `spo_kernel` module and delegate transparently -- no
code changes needed.

## Prerequisites

- Rust 1.75+ (MSRV)
- maturin (`pip install maturin`)

## Building

```bash
cd spo-kernel
maturin develop --release -m crates/spo-ffi/Cargo.toml
```

This compiles all Rust crates and installs `spo_kernel` into the active Python
environment. Verify:

```python
import spo_kernel
print(spo_kernel.PyUPDEStepper)
```

## Auto-Delegation

Python classes check for `spo_kernel` at construction time. If present, hot
paths delegate to Rust with no API change:

```python
from scpn_phase_orchestrator import UPDEEngine

engine = UPDEEngine(n_oscillators=64, dt=0.01, method="rk4")
# engine._rust is a PyUPDEStepper if spo_kernel is installed
# engine._rust is None otherwise (pure numpy fallback)
```

The `_compat.HAS_RUST` flag controls delegation globally. Set it to `False`
in benchmarks to force the Python path.

## Accelerated Modules

| Python Class / Function | Rust FFI Class | Hot path |
|------------------------|----------------|----------|
| `UPDEEngine` | `PyUPDEStepper` | `step()`, `run()` |
| `StuartLandauEngine` | `PyStuartLandauStepper` | `step()`, `run()` |
| `CouplingBuilder` | `PyCouplingBuilder` | `build()`, `project()` |
| `ImprintModel` | `PyImprintModel` | `update()`, `modulate_coupling()`, `modulate_lag()` |
| `compute_order_parameter` | `order_parameter` | single call |
| `compute_plv` | `plv` | single call |
| `modulation_index` | `pac_modulation_index` | single call |
| `pac_matrix` | `pac_matrix_compute` | full NxN |
| `CoherenceMonitor` | `PyCoherenceMonitor` | `compute_r_good()`, `compute_r_bad()`, `detect_phase_lock()` |
| `RegimeManager` | `PyRegimeManager` | `evaluate()`, `transition()` |
| `ActionProjector` | `PyActionProjector` | `project()` |
| `BoundaryObserver` | `PyBoundaryObserver` | `observe()` |
| `SupervisorPolicy` | `PySupervisorPolicy` | `decide()` |
| `PhaseQualityScorer` | `PyPhaseQualityScorer` | `score()`, `is_collapsed()` |
| `LagModel` | `PyLagModel` | `estimate()` |
| Physical extractor | `physical_extract` | analytic signal extraction |
| Symbolic extractors | `ring_phase`, `graph_walk_phase`, `transition_quality` | single call |
| Informational extractor | `event_phase` | timestamp analysis |

## Benchmark Comparison

`bench/run_benchmarks.py` measures `UPDEEngine.step()` with RK4, averaged
over 1000 steps after 50 warmup iterations.

| N | Python (numpy) | Rust (spo_kernel) | Speedup |
|---|---------------|-------------------|---------|
| 16 | ~25 us/step | 7.3 us/step | 3.4x |
| 64 | ~180 us/step | 28 us/step | 6.4x |
| 256 | ~2.8 ms/step | 0.32 ms/step | 8.7x |
| 1024 | ~45 ms/step | 8.6 ms/step | 5.2x |

The speedup saturates at large N because both paths are O(N^2) in coupling
computation; the Rust advantage comes from avoiding Python interpreter overhead
and numpy dispatch per operation.

## Crate Structure

```
spo-kernel/
  Cargo.toml          # workspace root
  crates/
    spo-types/        # Shared types: UPDEState, LayerState, Regime, Knob, ControlAction
    spo-engine/       # UPDE stepper, Stuart-Landau stepper, coupling, imprint, lags, PAC
    spo-oscillators/  # Physical, informational, symbolic, quality extractors
    spo-supervisor/   # Boundaries, coherence, policy, projector, regime manager
    spo-ffi/          # PyO3 bindings (this is what maturin builds)
```

All pure-logic crates (`spo-types`, `spo-engine`, `spo-oscillators`,
`spo-supervisor`) have `#![no_std]` aspirations but currently use `std` for
`HashMap` and `Vec`. Only `spo-ffi` depends on PyO3 and numpy.

## Contributing Rust Code

Run all three before submitting:

```bash
cargo fmt --all
cargo clippy --workspace -- -D warnings
cargo test --workspace
```

The CI pipeline runs:

| Job | Matrix | What |
|-----|--------|------|
| `rust-check` | 3 OS (Linux, macOS, Windows) | `cargo fmt --check`, `clippy -D warnings`, `cargo test` |
| `ffi-test` | 3 OS x 2 Python (3.10, 3.12) | `maturin develop --release`, `pytest tests/` |
| `cargo-audit` | Linux | `cargo audit` for known vulnerabilities |
| `rust-msrv` | Linux | Verify builds on Rust 1.75.0 |

## Numerical Parity

The Rust and Python implementations produce identical results to float64
precision. The CI `ffi-test` job runs the full Python test suite with
`spo_kernel` installed, confirming parity across all integration methods
(euler, rk4, rk45) and both engines (UPDEEngine, StuartLandauEngine).
