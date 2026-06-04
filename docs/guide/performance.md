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

When `spo_kernel` is installed, 53 engine modules delegate hot paths to
Rust automatically. Speedups range from 2x to 96x depending on the module
and N. See [Rust FFI Acceleration](rust_ffi.md) for build instructions
and the full module table. For backend support tiers and fallback rules, see
[Backend Strategy](backend_strategy.md).

**Note:** Two modules (coupling_est, phase_extract) have Rust auto-select
disabled because LAPACK lstsq and SciPy FFT respectively outperform the
current Rust implementations.

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
```

Output includes per-N step times, total wall time, and backend identification
(`python`, `rust`, or `rust_batch`, depending on optional backend availability).

## Reference Benchmark Suite

Run the v1 reference suite when publishing benchmark snapshots against the
Kuramoto/Strogatz/Acebrón, Stuart-Landau/Pikovsky, and Petri-net reference
surfaces:

```bash
PYTHONPATH=src python benchmarks/reference_suite.py
```

The JSON output is written to `benchmarks/results/reference_suite.json` and
contains two top-level fields:

| Field | Purpose |
|-------|---------|
| `metadata` | Snapshot date, exact command, backend label, Python version, NumPy version, executable, platform string, and benchmark evidence boundary |
| `benchmarks` | Kuramoto, Stuart-Landau, Petri reachability, and frontier gate records with timings, acceptance flags, thresholds, and physical summary values |

The Kuramoto record is an acceptance benchmark, not only a timer. It checks
zero self-coupling, bounded order parameter, identical-oscillator
synchronisation, and the exact two-oscillator locked phase lag
`asin((omega_2 - omega_1) / (2K))` from the pairwise Kuramoto equations.
The Stuart-Landau record is also an acceptance benchmark: it checks positive
finite coupled amplitude, zero self-coupling, wrapped phase output, convergence
to the uncoupled Hopf limit-cycle radius `sqrt(mu)`, and decay of a
subcritical `mu < 0` amplitude trajectory.
The Petri-net record is the formal-state reference gate. It requires exact
four-marking reachability, one-token conservation, deterministic transition
order, and the expected final marking after `n_steps`.
The ITPC record is a polyglot physics gate for the Lachaux inter-trial
phase-coherence estimator. It records Rust/Mojo/Julia/Go/Python slot status,
vector parity, pause-persistence parity, aligned-trial unit coherence,
opposite-phase zero coherence, unit-interval bounds, and unavailable-toolchain
reasons without mocking the backend boundary.
The chimera record is a polyglot physics gate for the Kuramoto-Battogtokh
local-order vector. It records Rust/Mojo/Julia/Go/Python slot status, local
order parity, global phase-gauge invariance, synchronised unit local order,
disconnected zero local order, and the exact uniform-circle all-to-all
reference `1 / (N - 1)`.
The spectral record is a polyglot mathematics gate for the Dörfler-Bullo
combinatorial graph Laplacian. It records Rust/Mojo/Julia/Go/Python slot
status, algebraic-connectivity parity, Fiedler-vector direction parity,
non-negative spectral gap, zero row sums, positive semidefiniteness, and exact
uniform-path plus complete-graph spectra.
The Hodge record is a polyglot mathematics gate for the Jiang decomposition of
coupling flow into gradient, curl, and harmonic components. It records
Rust/Mojo/Julia/Go/Python slot status, component parity, exact reconstruction
of total phase-weighted flow, near-zero harmonic residual for the clean
symmetric plus antisymmetric split, global phase-shift invariance, symmetric
zero-curl and antisymmetric zero-gradient special cases, the two-node
antisymmetric closed form, scale covariance, and unavailable-toolchain reasons.
The embedding record is a polyglot mathematics gate for Takens
delay-coordinate reconstruction. It records Rust/Mojo/Julia/Go/Python slot
status, exact delay-indexing parity, Fraser-Swinney mutual-information parity
where the backend exposes that primitive, nearest-neighbour geometry parity
where exposed, public fallback-dispatch parity, constant-signal zero mutual
information, zero-lag information dominance over distant lags, time-shift row
consistency, and nearest-neighbour self-exclusion on a line lattice.
The transfer-entropy record is a polyglot information-physics gate for the
Schreiber directed-information estimator. It records Rust/Mojo/Julia/Go/Python
slot status, exact scalar parity, pairwise-matrix parity, scalar-matrix
consistency for `TE(i -> j)`, zero diagonal, non-negative entropy-bounded
scores, known causal-direction preservation, phase-wrapping invariance,
short-series zero behaviour, public fallback-dispatch parity, and
unavailable-toolchain reasons without mocking the backend boundary.
The entropy-production record is a polyglot thermodynamic gate for the Acebrón
overdamped-Kuramoto dissipation rate. It records Rust/Mojo/Julia/Go/Python
slot status, exact formula parity against `sum(dtheta_dt ** 2) * dt`,
non-negative rates, fixed-point and zero-timestep limits, linear timestep
scaling, quadratic global-coupling scaling in the zero-frequency case, global
phase-shift invariance, oscillator permutation invariance, public
fallback-dispatch parity, and unavailable-toolchain reasons without mocking
the backend boundary.

Treat the emitted `snapshot_date` as a historical measurement label. Do not
copy the timings into current documentation unless the command was rerun in the
same environment and the new JSON artefact is available for review.
Treat checked-in timing fields as local, non-isolated regression evidence
unless the metadata also records CPU/core isolation and host-load controls.
Without that isolation evidence, the timing fields prove reproducibility and
parity execution only; they are not production throughput claims.

## Baseline Regression

CI builds the Rust FFI backend, runs `bench/run_benchmarks.py`, and compares
the result against `bench/baseline.json`. The regression gate fails closed when:

- no comparable baseline records exist;
- the current run has no comparable records;
- the benchmark JSON contains duplicate object keys or non-finite JSON tokens;
- a benchmark identity field is malformed;
- a baseline or current benchmark key is duplicated;
- a checked-in baseline configuration is missing from the current run;
- any comparable `us_per_step` value is non-finite or non-positive;
- a step-time increase exceeds both the configured relative threshold, currently
  20%, and the configured absolute significance floor, currently 100 us.

The absolute floor is deliberate. Very small oscillator cases can move by large
percentages when a hosted CI runner adds fixed overhead, while larger cases in
the same run may improve. The gate reports those tiny-case movements as
tolerated noise, but it still fails closed for materially large slowdowns.

Use `--allow-missing-current` only for deliberate local partial comparisons,
not for CI. Update baselines after intentional algorithm changes:

```bash
python bench/run_benchmarks.py --json > bench/baseline.json
python bench/compare_baseline.py bench/baseline.json bench/current.json
```

## Tips

- Set `method="euler"` for real-time applications where dt is already small
- Use `engine.run(phases, ..., n_steps=N)` instead of a Python loop -- the
  Rust path batches all N steps in a single FFI call
- Profile with `--python-only` to isolate numpy performance from Rust FFI
- For N > 256, Rust FFI is strongly recommended

### Spatial coupling modulator gate

The PHA-C.1 spatial modulator has a dedicated polyglot parity benchmark:

```bash
PYTHONPATH=src python benchmarks/spatial_modulator_benchmark.py --parity-gate
```

The gate checks more than runtime. It verifies the closed-form inverse-plus-one kernel, translation invariance, permutation equivariance, zero self-coupling, symmetry preservation, and the physically required monotonic relation that nearer oscillator pairs receive stronger spatial coupling than farther pairs. Rust, Mojo, Julia, Go, and Python slots are reported separately so missing optional accelerators are visible without weakening the acceptance contract.

Timing fields from this benchmark are local, non-isolated regression evidence. They are not production throughput claims unless the surrounding run records isolation, CPU/GPU binding, dependency versions, and backend build metadata.
