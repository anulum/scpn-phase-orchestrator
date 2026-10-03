# Rust FFI Acceleration

The `spo-kernel` Rust workspace accelerates selected paths across the engine,
coupling, supervisor, SSGF, monitor, extraction, and autotune surfaces. Recorded
local speedups vary substantially by operation, size, build, and host. Python
classes auto-detect the compiled `spo_kernel` module and delegate
transparently—no application-code change is needed.

For runtime selection across Rust, Python, JAX, and auxiliary research
backends, see [Backend Fallback Chain](backend_fallbacks.md).

## Prerequisites

- Rust 1.89+ (workspace MSRV; `nalgebra` 0.35 requires 1.89)
- maturin (`pip install maturin`)

## Building

Use the repository helper so maturin runs through the Python interpreter that
owns the target environment:

```bash
python tools/install_spo_kernel.py --release
```

This compiles all Rust crates and installs `spo_kernel` into the active Python
environment. To target the repository virtual environment explicitly:

```bash
.venv/bin/python tools/install_spo_kernel.py --release
```

Verify the selected environment:

```bash
python tools/install_spo_kernel.py --check-only
```

The helper runs `python -m maturin develop --release` for the selected
interpreter, but not in the way a bare command would:

- `maturin develop` installs into the environment named by `VIRTUAL_ENV` (or
  `CONDA_PREFIX`, or a `.venv` found above the working directory), not into the
  interpreter that runs it. The helper sets `VIRTUAL_ENV` to the prefix of the
  interpreter given by `--python`.
- In an environment created by uv, maturin installs through uv, and uv applies
  the configuration of the project it runs in. This repository's
  `[tool.uv] exclude-dependencies` lists `spo-kernel`, so a bare
  `maturin develop` run from the checkout reports the kernel as installed while
  leaving the previous build (or none) in place. The helper runs maturin from an
  empty directory outside any project.
- After installing, the helper compares the SHA-256 of the extension the
  environment would load with the library cargo built, and fails if they
  differ. The JSON record reports both as `extension` and `extension_sha256`.

You can inspect the command without compiling Rust:

```bash
python tools/install_spo_kernel.py --dry-run --json
```

After installation, direct import should work:

```python
import spo_kernel
print(spo_kernel.PyUPDEStepper)
```

## Cellular-Sheaf Integration

The public `SheafUPDEEngine` selects `PySheafUPDEStepper` when the compiled
module is available, otherwise it uses NumPy. This selection is separate from
the scalar engine's auxiliary-language fallback chain.

Every successful step advances the configured outer `dt`, including adaptive
RK45. Native `n_substeps` divides that interval for all three methods. RK45 uses
the maximum scaled component error; its positive finite `last_dt` is the next
internal proposal, bounded by the outer interval. Fixed methods report `dt`.
Counts reject boolean aliases, overflowing geometry and invalid substep counts;
numerical controls are finite real values, with `rtol >= atol` for RK45.

Direct native inputs are readonly, one-dimensional contiguous float64 arrays
in `(i, d)` phase/frequency and `(i, j, d, k)` restriction-map order. They may
share storage. Step and batch outputs own independent storage, and invalid
buffers or numerical refusal preserve input values and the pre-call proposal.
Rounded upper torus endpoints map to equivalent zero, including valid tiny
negative phase crossings; non-finite arithmetic still refuses.
Zero batches still validate all buffers and return a copy preserving finite
unwrapped phases. The same native instance recovers on subsequent valid input.
The Python entry point also accepts finite real array-like inputs and copies
strided storage. See the [complete contract](../reference/api/upde.md#cellular-sheaf-engine).

Run `PYTHONPATH=.:src python -m benchmarks.sheaf_benchmark` in each actual
environment to compare all three methods with an independent SciPy DOP853
tensor ODE reference. The [September 30 diagnostic record](../reference/data/sheaf_benchmark_2026-09-30.json)
contains raw repeats, source and binary provenance, input fingerprints and
equation errors. Shared-host timings establish neither a production latency
guarantee nor a causal acceleration claim.

## Auto-Delegation

Python classes check for `spo_kernel` at construction time. If present, hot
paths delegate to Rust with no API change:

```python
from scpn_phase_orchestrator import UPDEEngine

engine = UPDEEngine(n_oscillators=64, dt=0.01, method="rk4")
# engine._rust is a PyUPDEStepper if spo_kernel is installed
# engine._rust is None otherwise (pure numpy fallback)
```

Use separate actual kernel-present and kernel-absent environments for
stateful comparisons. Changing `_compat.HAS_RUST` after import does not update
the engine module's imported flag and is not evidence of Python execution.

Sleep staging is an explicit exception: `classify_sleep_stage` and
`ultradian_phase` use Python by default even when the kernel is installed.
Select `backend="rust"` to exercise their native wrappers; an unavailable
kernel raises `RuntimeError`. Their dispatcher does not use
`_compat.HAS_RUST`. See the
[sleep-staging reference](../reference/api/monitor_sleep_staging.md#backend-selection)
for validation, parity tests and measured wrapper overhead.

## Coupling projection

`PyCouplingBuilder.project(values, n)` is an explicit native entry point.
`project_knm(values, [SymmetryConstraint(), NonNegativeConstraint()])` runs
NumPy and has matching projection semantics. Both preserve finite extreme
pair means and subnormal values, accept the empty matrix, and reject invalid
source types and non-finite coefficients. Native count extraction rejects
negative Python integers with `OverflowError`; boolean/text counts, product
overflow and wrong flat cardinality raise `ValueError`. No invalid request
mutates its input or prevents subsequent valid projection. Rust public
construction checks both `n*n` and binary64 byte capacity before allocating its
matrices, returning `ValueError` rather than a native capacity panic. The public
builder applies the same capacity bound before attempting either backend.
Finite extreme strength/decay values preserve the exponential equation and zero
diagonal; underflow rounds to zero. The amplitude and SCPN numerical paths remain
Python-only. For actual constructor parity/refusal/recovery cases use
`native-tests/test_coupling_builder.py` with the newly built wheel and
`tests/test_coupling_builder_finite.py` in both genuine environments.
See [construction measurements](../reference/data/coupling_builder_runtime_benchmark_2026-10-02.json)
for exact scalar bits, current source/binary pins and shared-host timings.

Run `native-tests/test_geometry_projection.py` against the newly built wheel;
run `tests/test_geometry_projection_finite.py` in both real native and genuinely
kernel-absent interpreters. Never emulate absence with an availability flag.
See [geometry constraints](../specs/geometry_constraints.md) and
[recorded measurements](../reference/data/geometry_projection_runtime_benchmark_2026-10-02.json).

## Accelerated Modules

| Python Class / Function | Rust FFI Class | Hot path |
|------------------------|----------------|----------|
| `UPDEEngine` | `PyUPDEStepper` | dense `step()`, `run()`, frequency/Doppler schedules and moving-frame schedule |
| `SheafUPDEEngine` | `PySheafUPDEStepper` | matrix phase `step()`, `run()` |
| `StuartLandauEngine` | `PyStuartLandauStepper` | `step()`, `run()` |
| `CouplingBuilder` / geometry projection | `PyCouplingBuilder` | `build()`; direct `project()` for symmetry, non-negativity and zero diagonal |
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
| `PhaseQualityScorer` | `PyPhaseQualityScorer` | `score()`, `is_collapsed()`, `downweight_mask()` |
| `LagModel` | `PyLagModel` | `estimate()` |
| `NeurocoreBridge` | `PyLIFEnsemble` | `step()` (LIF ensemble, 325x at N=10000) |
| Physical extractor | `physical_extract` | analytic phase/frequency and scaled envelope mean/CV; Hilbert preprocessing stays in SciPy |
| Symbolic extractors | `ring_phases_rust`, `graph_walk_phases_rust`, `transition_qualities_rust` | vector phases and linear graph qualities; cyclic ring quality stays in Python |
| Informational extractor | `event_phase` | timestamp analysis |
| `SimplicialEngine` | `simplicial_run` | 3-body coupling `run()` |
| `HypergraphEngine` | `hypergraph_run` | k-body coupling `run()` |
| `GeometricEngine` | `geometric_run` | SO(2) exp map `run()` |
| `extract_envelope` | `envelope_rms_rust` | cumulative-sum RMS |
| `OttAntonsenReduction` | `oa_run_rust` | `run()`, `steady_state_R()`, `predict_from_oscillators()` |
| `SplittingEngine` | `splitting_run_rust` | Strang split `run()` |
| `te_adapt_coupling` | `te_adapt_coupling_rust` | TE-directed coupling update |
| `UniversalPrior.log_probability` | `prior_log_probability_rust` | Bayesian log-density |
| `load_hcp_connectome` | `load_hcp_connectome_rust` | synthetic connectome generation |
| `GeometryCarrier.decode` | `carrier_decode_rust` | softplus(A·z) decode |
| `compute_ethical_cost` | `compute_ethical_cost_rust` | SEC + CBF ethical cost |
| `classify_sleep_stage` | `classify_sleep_stage_rust` | stage classification, explicit `backend="rust"` |
| `ultradian_phase` | `ultradian_phase_rust` | cycle phase, explicit `backend="rust"` |
| `EVSMonitor._frequency_specificity` | `frequency_specificity_rust` | target/control ITPC ratio |
| `PhaseSINDy.fit` | `sindy_fit_rust` | STLSQ sparse regression |
| `estimate_coupling` | (disabled) | normal equations (3x slower than LAPACK) |
| `extract_phases` | (disabled) | naive DFT (60x slower than SciPy FFT) |

### Symbolic input boundaries

The vector symbolic functions accept one-dimensional native-endian `int64` or
`uint64` NumPy arrays. Positive, negative and zero strides, read-only views and
empty arrays preserve logical observation order; a contiguous buffer is not
required. Direct FFI refuses unaligned views before Rust element access and
refuses other dtypes, ranks and Python lists. The public `SymbolicExtractor`
normalises narrower integer widths within their signedness and copies unaligned
input, so callers need not pre-align public observations.

Ring residues are computed as integers before conversion. Graph walk differences
and totals use `u128`, including the full signed/unsigned 64-bit span; the Python
fallback uses arbitrary-size integers. Output remains `float64`, not exact
rational arithmetic. Vector graph qualities use linear distances; cyclic ring
quality remains in Python.

Quality scoring uses the same finite-pair policy in Python and direct native
Rust: qualities are clamped to [0, 1], amplitudes have a 1e-12 floor and weights
are divided by their finite maximum before accumulation. The direct native
scorer retains matching-prefix behaviour for unequal sequences; the public
state-based API supplies equal-length quality/amplitude sequences. Per-call
collapse/mask thresholds differing from the configured values use Python.
Real native tests observe the installed scorer's C calls and verify public
outputs, without replacing the backend.

The [quality runtime snapshot](../reference/data/phase_quality_runtime_benchmark_2026-10-03.json)
records current public/native measurements in installed and genuinely absent
environments, plus the standalone Rust slice benchmark. These boundaries
include different validation costs and are shared-host regression evidence.

These symbolic vector contracts require `spo-kernel >= 0.5.11`; both the
`rust` and `scpn-all` extras enforce that floor. Rebuild the kernel when
updating a source checkout; older wheels do not accept unsigned or strided
arrays under this contract.

Direct scalar/count arguments are bounded by the target's `usize`; the public
extractor separately requires `n_states >= 2` and uses Python for larger-than-
`usize` counts instead of claiming native execution. Direct
`transition_qualities_rust` requires a finite initial quality; the public
extractor additionally bounds that configured quality to [0, 1].

`benchmarks/bench_symbolic.py` measures both installed-native and genuinely
kernel-absent environments through the public extractor. It records actual
C-call identity, source hashes, Python/NumPy versions and host load. Its
non-isolated median/P95 timings do not establish a cross-environment speedup.
The [2026-09-30 raw diagnostic](../reference/data/symbolic_extraction_diagnostic_2026-09-30.json)
also pins the loaded extension artefact by SHA-256.

## Benchmark Comparison

Local diagnostics on 2026-10-01 measured real `UPDEEngine.step()` through
the maintained stateful comparison: 16 nodes, 1000 public steps per method,
Linux x86-64, CPython 3.12.3 and the Rust 1.98.1 release build.

| Method | NumPy-only aggregate (ms) | Installed Rust aggregate (ms) |
|---|---:|---:|
| Euler | 120.9 | 138.8 |
| RK4 | 209.3 | 135.1 |
| RK45 | 198.6 | 138.8 |

Reproduce with `.venv/bin/python -m benchmarks.engine_comparison` and
`PYTHONPATH=src /usr/bin/python3 -m benchmarks.engine_comparison` in a genuinely
kernel-absent interpreter. The maintained CLI emits formatted aggregate times,
not per-step raw samples or P50 estimates. The
[raw runtime record](../reference/data/upde_phase_wrapping_benchmark_2026-10-01.json)
preserves both complete eight-variant outputs, source/artefact fingerprints and
separate stateless/CSR/JAX/browser measurements.
The environments use NumPy 2.2.6 and 2.5.3 respectively on a shared workstation;
these observations do not establish causal speedup, latency guarantees or
capacity. The ordinary UPDE slots select Rust/NumPy; other models retain their
own backend dispatch.

`benchmarks/engine_comparison.py` exercises the stateful variants.
`benchmarks/upde_engine_benchmark.py` compares the separate stateless runner
across Rust, Mojo, Julia, Go and Python; frequency/Doppler and moving-frame
comparators cover their own scheduled contracts. Do not use a stateless-only
comparison to infer the cost of the stateful NumPy borrow boundary.

## Dense stepper buffer ownership

The four mutable dense calls (`step`, `run`, `run_omega_schedule`,
`run_doppler_schedule`) copy readonly input values and release their NumPy
borrow guards before requesting writable coupling. Identical or overlapping
input views therefore retain entry-time values rather than conflicting with
that request. Native plasticity still updates the original coupling after
each step; lag and frequency/velocity snapshots do not change mid-run.

Coupling must remain writable and contiguous at this native boundary.
Outstanding external borrows, readonly coupling and non-contiguous required
buffers raise `ValueError`. A failed mutable borrow does not advance the
stepper. The moving-frame method has a distinct readonly coupling contract.
The public stateful engine and stateless Rust callers pass coupling through
this boundary: readonly coupling is refused there even when plasticity is off,
whereas the other CPU paths accept it. Supply a writable copy for portable
admission across backends.
See [the core engine reference](../reference/api/upde.md#shared-numpy-storage)
for public-engine behaviour.

## Sparse stepper buffer ownership

`PySparseUPDEStepper.step()` and `run()` snapshot readonly phase, frequency,
CSR row/column and lag inputs before acquiring writable coupling. This permits
shared storage while preserving entry-time readonly values across a run.
`set_plasticity()` still changes the original coupling; `disable_plasticity()`
keeps it fixed. The public `SparseUPDEEngine` constructs the native solver with
plasticity disabled and has no public plasticity setter.

The native `order_parameter()` reads cached derivative-stage phases:
Euler retains the input of its final substep, RK4 its final k4 stage, and RK45
its final y5 stage before wrapping. Compute the order parameter from the
returned phase vector when it must describe that wrapped output.
Integer phase, frequency, coupling and lag inputs are converted to float64
before integration in both environments; unsigned and narrow integer
subtraction therefore follows the real-valued phase equation.

Direct native calls require contiguous buffers and writable coupling.
Readonly coupling raises `ValueError` without advancing the solver; supplying a
writable copy lets the same instance recover. The public wrapper admits strided
inputs by copying them, but native readonly-coupling refusals propagate.
The public refusal applies to contiguous float64 coupling passed through without
copying; dtype conversion and strided copies can create writable native buffers
while leaving the original input unchanged. The actual NumPy-only fallback
accepts readonly coupling without mutation.
Public zero-step runs validate and return a copy without advancing the solver.
`last_dt` reports the configured fixed timestep or the RK45 next-step proposal,
not the elapsed time of an accepted adaptive step.
The public wrapper validates real results in both environments. Producers
canonicalise a remainder rounded to the excluded upper endpoint, and signed
zero, to positive zero; interior phases remain. Nonfinite computed output and
invalid adaptive proposals still raise `ValueError`. A refused native step
restores caller phases and its pre-step proposal before plasticity; subsequent
valid input can use the same solver. Public `last_dt` retains its pre-call
value on a failed run, but earlier accepted native steps are not rewound if a
later batch step refuses. Buffer-borrow recovery remains possible with writable
contiguous inputs. Native dense/CSR rollback snapshots retain phases and trigonometric caches, adding O(N) storage per step.
Recovery above concerns nonfinite phase computation. A separate invalid native
adaptive diagnostic, including zero-proposal underflow, retains the public
diagnostic but requires reconstruction with suitable numerical controls.

The maintained sparse chain is Python → PyO3 → Rust. The stateless dense
Go/Julia/Mojo accelerators are separate surfaces and do not implement CSR sparse
integration. See [the sparse API reference](../reference/api/upde.md#sparse-engine).

### Sparse diagnostic workloads (2026-10-01)

Run the real public Euler path against an independent SciPy CSR sine-difference
reference:

```bash
.venv/bin/python -m benchmarks.sparse_benchmark
PYTHONPATH=src /usr/bin/python3 -m benchmarks.sparse_benchmark
```

The second command requires an environment where `spo_kernel` is absent;
verify availability rather than selecting a synthetic backend. Both defaults
use seed 42, `dt=0.01` seconds, 100 steps and three freshly initialised repeats.
Duplicate random edges merge in CSR, so stored edge counts differ from requested
entry counts. JSON includes raw seconds, input SHA-256, runtime versions, order
parameter and the largest final circular phase error against the reference.

| Nodes | Stored edges | Installed Rust: median µs/step | Kernel absent: median µs/step | Largest reference error (rad) |
| --- | --- | --- | --- | --- |
| 1,000 | 9,963 | 240.8 | 33,648.2 | 1.0e-15 |
| 10,000 | 99,936 | 3,242.3 | 313,791.7 | 2.7e-15 |

Input SHA-256 agrees across both environments for each workload. Both use
Python 3.12.3; the installed-kernel environment uses NumPy 2.5.3/SciPy 1.18.1,
and the absent-kernel environment NumPy 2.2.6/SciPy 1.15.3. Measurements run on a
shared, non-isolated workstation with no reserved cores. Native per-step
observations vary across repeats (234–256 µs at 1,000 nodes);
the displayed median is not a stable latency estimate. They are local equation
and timing diagnostics, not causal speedup, production latency or capacity
claims. Raw repetitions, host load before/after, affinity, governor and source
fingerprints are in the public raw runtime record linked above.

### LIF Ensemble (NeurocoreBridge)

The `PyLIFEnsemble` accelerates the neurocore bridge's spiking neuron
simulation. Measured on Windows 11, Python 3.12, Rust 1.93.0 release build,
N=10000 neurons (10 layers × 1000), 100 substeps:

| Backend | Time | ns/neuron/substep | Speedup vs scalar |
|---------|------|-------------------|-------------------|
| Rust (`PyLIFEnsemble`) | 0.004 s | 3-6 ns | 325× |
| NumPy (vectorised) | 0.014 s | 14 ns | 93× |
| Scalar (sc-neurocore per-neuron) | 1.306 s | 1,306 ns | 1× |

The recorded 4 ms per 100-substep result is local throughput evidence. It does
not establish a 250 Hz control-loop deadline: end-to-end sensing, scheduling,
transport, actuation, jitter, and worst-case latency were not measured.

## Crate Structure

```
spo-kernel/
  Cargo.toml          # workspace root
  crates/
    spo-types/        # Shared types: UPDEState, LayerState, Regime, Knob, ControlAction
    spo-engine/       # 53 modules: UPDE (12 engines), coupling (11), monitors (14), SSGF (3), autotune (4), + support
    spo-oscillators/  # Physical, informational, symbolic, quality extractors
    spo-supervisor/   # Boundaries, coherence, policy, projector, regime manager
    spo-ffi/          # PyO3 bindings (this is what maturin builds)
```

All pure-logic crates (`spo-types`, `spo-engine`, `spo-oscillators`,
`spo-supervisor`) have `#![no_std]` aspirations but currently use `std` for
`HashMap` and `Vec`. Only `spo-ffi` depends on PyO3 and numpy.

## FFI Numeric Precision Contract

The Python/Rust boundary is a float64 contract:

- Python passes contiguous numeric arrays after shape, finite-value, and domain
  validation.
- Rust kernels return float64-compatible values and Python revalidates shape,
  finiteness, and physical bounds before accepting the result.
- Phase outputs are normalised to `[0, 2π)` unless the documented monitor
  contract returns a unitless score or matrix.
- Fixed-point formal manifests are separate review artefacts; they do not
  change the runtime float64 solver contract.

Any new FFI path must add a module-specific parity test that compares Python
and Rust on the same seeded input and documents the tolerated absolute or
relative error. Safety-critical gates must fail closed when either backend
emits NaN, infinity, shape drift, or a value outside the declared physical
range.

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
| `ffi-test` | 3 OS x 2 Python (3.11, 3.12) | `maturin develop --release`, `pytest tests/` |
| `cargo-audit` | Linux | `cargo audit` for known vulnerabilities |
| `cargo-deny` | Linux | RustSec advisories, banned wildcard dependencies, and source registry policy |
| `rust-miri` | Linux nightly | Miri smoke tests for pure-Rust type/supervisor crates |
| `rust-msrv` | Linux | Verify builds on Rust 1.89.0 |

## Numerical Parity

The Rust and Python implementations agree numerically at float64 precision,
not necessarily bitwise. Symbolic ring phases and graph qualities may differ
in their last bits above `2**53`: Rust divides converted operands while
Python rounds an integer quotient. The `2**53 + 1` regression preserves this
explicit finite-precision contract without changing parity tolerances.
The CI `ffi-test` job runs the full Python test suite with
`spo_kernel` installed, confirming parity across all integration methods
(euler, rk4, rk45) and both engines (UPDEEngine, StuartLandauEngine).
