# Performance Tuning

## Integration Method Selection

| Method | Cost per step | Accuracy | Use when |
|--------|--------------|----------|----------|
| `euler` | 1 derivative eval | O(dt) | Fastest. Sufficient when dt satisfies CFL bound |
| `rk4` | 4 derivative evals | O(dt^4) | Good default. Use for stiff coupling or when accuracy matters |
| `rk45` | 6 derivative evals (adaptive) | O(dt^5) | Oscillator frequencies vary by >10x or coupling transients create short stiff intervals |

For most applications, `rk4` with a dt satisfying the CFL bound is the right
choice. `euler` is acceptable for real-time applications where latency budgets
are tight and dt is small enough.

## CFL Stability Bound

```
dt < pi / (max(omega) + N * max(K) + zeta)
```

The coupling term contributes up to `N * max(K)` to the effective frequency.
Exceeding this bound causes phase jumps that break the wrapping invariant.

The binding spec `sample_period_s` sets dt. Validate at initialisation.

## Pre-Allocated Arrays

Both `UPDEEngine` and `StuartLandauEngine` pre-allocate all scratch arrays at
construction time:

| Array | Shape | Engine |
|-------|-------|--------|
| `_phase_diff` | (N, N) | Both |
| `_sin_diff` | (N, N) | Both |
| `_cos_diff` | (N, N) | StuartLandau only |
| `_scratch_dtheta` | (N,) | Both |
| `_scratch_dr` | (N,) | StuartLandau only |
| `_scratch_deriv` | (2N,) | StuartLandau only |

All operations use numpy `out=` parameters to write into pre-allocated buffers.
No heap allocation occurs during stepping. For RK4, intermediate `k` vectors
are copied since the scratch buffer is reused across stages.

## Rust FFI

When `spo_kernel` is installed, Python classes delegate hot paths to Rust
automatically. Typical speedup: 3-9x depending on N. See
[Rust FFI Acceleration](rust_ffi.md) for build instructions.

Benchmark numbers (RK4, 1000 steps, averaged):

| N | Python (numpy) | Rust (spo_kernel) | Speedup |
|---|---------------|-------------------|---------|
| 8 | ~12 us/step | ~4.2 us/step | 2.9x |
| 16 | ~25 us/step | ~7.3 us/step | 3.4x |
| 64 | ~180 us/step | ~28 us/step | 6.4x |
| 256 | ~2.8 ms/step | ~320 us/step | 8.7x |
| 1024 | ~45 ms/step | ~8.6 ms/step | 5.2x |

Both paths are O(N^2) due to the coupling matrix. The Rust advantage comes
from eliminating Python interpreter overhead and numpy dispatch per operation.

## Coupling Matrix Sparsity

The default coupling builder uses exponential decay: `K_ij = base * exp(-alpha * |i - j|)`.
For large N, many entries are negligible. The current implementation stores
the full dense (N, N) matrix. Sparse representations are planned but not
yet implemented.

For now, keep N manageable. Typical domain sizes:

| Domain | N | dt | Method | Budget per step |
|--------|---|-----|--------|----------------|
| EEG (8 channels) | 8 | 0.01s | rk4 | <10 us |
| Microservices (20 services) | 20 | 15s | euler | <50 us |
| Tokamak (16 SCPN layers) | 16 | 0.001s | rk4 | <10 us |
| Power grid (100 nodes) | 100 | 0.01s | rk4 | <100 us |

## Memory Footprint

- Coupling matrix: O(N^2) -- dominates for N > 32
- Phases, omegas, scratch vectors: O(N)
- Stuart-Landau adds O(N^2) for `_cos_diff` and O(N) for amplitude scratch
- Audit log: O(steps) if enabled

For N = 1024: coupling matrix alone is 8 MB (float64). Phase vectors are 8 KB.

## Profiling

Run `bench/run_benchmarks.py` for systematic measurement:

```bash
python bench/run_benchmarks.py --json > results.json
python bench/run_benchmarks.py --python-only  # force numpy path
```

Output includes per-N step times, total wall time, and backend identification
(numpy vs spo_kernel).

## Baseline Regression

CI compares benchmark results against `bench/baseline.json`. A step-time
increase of >20% over baseline fails the check. Update baselines after
intentional algorithm changes:

```bash
python bench/run_benchmarks.py --json > bench/baseline.json
```

## Tips

- Set `method="euler"` for real-time applications where dt is already small
- Use `engine.run(phases, ..., n_steps=N)` instead of a Python loop -- the
  Rust path batches all N steps in a single FFI call
- Profile with `--python-only` to isolate numpy performance from Rust FFI
- For N > 256, Rust FFI is strongly recommended
