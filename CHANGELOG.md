# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added (2026-04-18 — hypergraph multi-backend)
- `julia/hypergraph.jl`, `go/hypergraph.go`
  (→ `libhypergraph.so`), `mojo/hypergraph.mojo`
  (→ `hypergraph_mojo`) implementing the generalised k-body
  Kuramoto stepper (Tanaka-Aoyagi 2011, Skardal-Arenas 2019,
  Bick et al. 2023) with an optional dense pairwise ``K`` term
  and external ``(ζ, ψ)`` drive.
- Python bridges `upde/_hypergraph_{julia,go,mojo}.py`.
- `upde/hypergraph.py` upgraded to five-backend dispatcher
  operating on the ``run(phases, omegas, n_steps, …)`` entry
  point (composite-step batching matches the Rust FFI).
- Python reference realigned to the Rust sincos expansion
  ``sin(θ_j − θ_i) = s_j·c_i − c_j·s_i`` on the ``alpha == 0``
  fast path; nonzero alpha uses the direct ``sin(diff)`` form
  in all five backends, matching Rust bit-for-bit.
- 29 new tests — `tests/test_hypergraph_algorithm.py` (13 incl.
  Hypothesis: Hyperedge API, phase wrap, zero-coupling rotation,
  pairwise-only Kuramoto limit, triadic fixed-point preservation,
  near-sync local stability, ζ-forcing),
  `tests/test_hypergraph_backends.py` (11 cross-backend parity
  for pairwise / alpha≠0 / no-pairwise regimes with Hypothesis
  sweeps for Rust / Go),
  `tests/test_hypergraph_stability.py` (5 long-run invariants).
- Parity measured bit-exact (0.0) across Rust / Julia / Go /
  Python on pairwise+triadic+4-body ICs with alpha=0,
  alpha≠0, and ζ≠0; Mojo agrees within the subprocess
  text-round-trip epsilon.
- `benchmarks/hypergraph_benchmark.py` rewritten as a per-backend
  wall-clock harness at ``N ∈ {8, 32, 64}`` with 2·N random
  triadic edges. Measured numbers show Go outperforming Rust
  for small N: the kernel's inner loop is mostly sequential
  hyperedge accumulation, so the rayon fork-join overhead in
  the Rust ``par_iter_mut`` path dominates at sub-O(10⁴) work
  per step. The canonical Rust → Mojo → Julia → Go → Python
  ordering is retained per
  ``feedback_fallback_chain_ordering.md``; the benchmark
  records what actually happens on this workload.

### Added (2026-04-18 — inertial multi-backend)
- `julia/inertial.jl`, `go/inertial.go` (→ `libinertial.so`),
  `mojo/inertial.mojo` (→ `inertial_mojo`) implementing the
  second-order (swing-equation) Kuramoto RK4 stepper
  (Filatrella-Nielsen-Mallick 2008).
- Python bridges `upde/_inertial_{julia,go,mojo}.py`.
- `upde/inertial.py` upgraded to five-backend dispatcher. The
  Python reference was realigned to use the same
  ``sin(θ_j − θ_i) = sin(θ_j)·cos(θ_i) − cos(θ_j)·sin(θ_i)``
  expansion as the Rust kernel (``spo-engine/src/inertial.rs``),
  giving bit-exact parity across Rust / Julia / Go / Python. Mojo
  drifts only by the subprocess text-round-trip epsilon.
- 21 new tests — `tests/test_inertial_algorithm.py` (13 incl.
  Hypothesis: RK4 exactness under zero coupling + zero damping,
  exponential ω-decay under damping, phase wrap, shapes, helpers)
  and `tests/test_inertial_backends.py` (8 cross-backend +
  multi-step parity with Hypothesis sweeps for Rust / Go).
- Parity measured 0.0 bit-exact across Rust / Julia / Go / Python
  and 1.7e-18 on Mojo for one RK4 step on the canonical N=8
  all-to-all test problem.
- `benchmarks/inertial_benchmark.py` rewritten as a per-backend
  wall-clock harness at ``N ∈ {8, 32, 128}`` (the earlier single-
  backend stub is replaced; no historical numbers are lost since
  it printed only a single throughput figure).

### Added (2026-04-18 — basin_stability multi-backend)
- `julia/basin_stability.jl`, `go/basin_stability.go`
  (→ `libbasin_stability.so`), `mojo/basin_stability.mojo`
  (→ `basin_stability_mojo`) implementing the one-trial Kuramoto
  ``steady_state_r`` kernel (explicit Euler, transient discarded,
  time-averaged order parameter).
- Python bridges `upde/_basin_stability_{julia,go,mojo}.py`.
- `upde/basin_stability.py` upgraded to five-backend dispatcher on
  the single-trial kernel. Rust's ``steady_state_r_rust`` FFI is
  now the active path; ``basin_stability(...)`` owns the Monte
  Carlo loop + RNG in Python (``np.random.default_rng(seed)``) and
  calls the dispatched trial kernel once per IC. This is the
  ``dimension`` pattern — Python owns randomness so the compute
  primitive stays deterministic and parity-testable.
- 26 new tests — `tests/test_basin_stability_algorithm.py` (16
  algorithmic + Hypothesis incl. physics limits, threshold
  monotonicity, ``multi_basin_stability`` keys, shape / bounds /
  determinism), `tests/test_basin_stability_backends.py` (10
  cross-backend parity with Hypothesis sweeps for Rust / Go).
- Parity measured bit-exact (0.0) across all five backends for
  the canonical all-to-all test problem.
- `benchmarks/basin_stability_benchmark.py` — per-backend
  wall-clock harness at ``N ∈ {8, 32, 64}``, ``n_transient=200``,
  ``n_measure=100``.

### Added (2026-04-18 — swarmalator multi-backend)
- `julia/swarmalator.jl`, `go/swarmalator.go`
  (→ `libswarmalator.so`), `mojo/swarmalator.mojo`
  (→ `swarmalator_mojo`) implementing the O(N²·d) swarmalator
  step (O'Keeffe & Strogatz 2017) with position attraction /
  repulsion coupled to Kuramoto-style phase dynamics.
- Python bridges `upde/_swarmalator_{julia,go,mojo}.py`.
- `upde/swarmalator.py` upgraded to five-backend dispatcher.
  `SwarmalatorEngine` keeps ``(n_agents, dim, dt)`` state;
  the step itself stays stateless ``(pos, phases, omegas) →
  (new_pos, new_phases)``.
- Python reference fallback re-aligned to the Rust kernel:
  repulse uses ``b / (dist · d²ₛᵤₘ + eps)`` (pre-eps squared
  sum), not the earlier ``b / (dist³ + eps)`` variant. The two
  agree as ``eps → 0`` but drift at small distances; all four
  non-Rust backends now match Rust exactly.
- 19 new tests — `tests/test_swarmalator_algorithm.py` (11
  algorithmic + Hypothesis: phase wrap invariant, translation
  invariance of velocity, coincident-agents regularisation,
  ``k = 0`` decouples phases from positions),
  `tests/test_swarmalator_backends.py` (8 cross-backend parity
  with Hypothesis sweeps for Rust / Go).
- Parity: Rust 5.5e-17, Julia 0.0 exact, Go 2.8e-17, Mojo
  1.1e-16 (all under the 1e-9 tolerance).
- `benchmarks/swarmalator_benchmark.py` — per-backend wall-clock
  harness at ``N ∈ {8, 32, 128}``, ``dim=2``, 5 calls.

### Added (2026-04-18 — retrofitted per-backend benchmarks)
- `benchmarks/psychedelic_benchmark.py`,
  `benchmarks/hodge_benchmark.py`,
  `benchmarks/envelope_benchmark.py` — per-backend wall-clock
  harnesses for the three earlier migrations whose
  benchmark step had been skipped. Each sweeps a representative
  size range and records ``ms_per_call`` for every available
  backend.

### Added (2026-04-18 — envelope multi-backend)
- `julia/envelope.jl`, `go/envelope.go` (→ `libenvelope.so`),
  `mojo/envelope.mojo` (→ `envelope_mojo`) implementing
  sliding-window RMS (O(T) cumulative-sum form) + modulation depth.
- Python bridges `upde/_envelope_{julia,go,mojo}.py`.
- `upde/envelope.py` upgraded to five-backend dispatcher on the
  1-D path. The 2-D batched ``(T, N)`` path stays pure NumPy (Rust
  FFI is 1-D-only; vectorised NumPy is already near-optimal).
- Python fallback handles the ``window > T`` edge case (all-zero
  output to match Rust canonical behaviour); parity tests restrict
  Hypothesis to the physically meaningful ``window ≤ T`` regime.
- 20 new tests — `tests/test_envelope_algorithm.py` (12 algorithmic
  + Hypothesis), `tests/test_envelope_backends.py` (8 cross-backend
  parity with Hypothesis sweeps for Rust / Go).
- Parity bit-equivalent (0.0 exact) on Rust / Julia / Go; 3.3e-15
  on Mojo.

### Added (2026-04-18 — hodge multi-backend)
- `julia/hodge.jl`, `go/hodge.go` (→ `libhodge.so`),
  `mojo/hodge.mojo` (→ `hodge_mojo`) implementing the Hodge
  decomposition of coupling dynamics into symmetric (gradient),
  antisymmetric (curl), and harmonic (residual) per-oscillator
  components (Jiang et al. 2011).
- Python bridges `coupling/_hodge_{julia,go,mojo}.py`.
- `coupling/hodge.py` upgraded to five-backend dispatcher.
- 15 new tests — `tests/test_hodge_algorithm.py` (8 algorithmic +
  Hypothesis incl. symmetric-K → zero curl, antisymmetric-K → zero
  gradient, gradient+curl+harmonic reconstructs total),
  `tests/test_hodge_backends.py` (7 cross-backend parity with
  Hypothesis sweeps for Rust / Go).
- Parity measured bit-equivalent (4.4e-16) on Rust/Julia/Go.
  Mojo 8.9e-16.

### Added (2026-04-18 — psychedelic entropy multi-backend)
- `julia/psychedelic.jl`, `go/psychedelic.go`
  (→ `libpsychedelic.so`), `mojo/psychedelic.mojo`
  (→ `psychedelic_mojo`) implementing the circular-phase Shannon
  entropy kernel (wrap to ``[0, 2π)`` + histogram + entropy in nats).
- Python bridges `monitor/_psychedelic_{julia,go,mojo}.py`.
- `monitor/psychedelic.py` upgraded to five-backend dispatcher for
  `entropy_from_phases`. `reduce_coupling` and
  `simulate_psychedelic_trajectory` stay as they are —
  `reduce_coupling` is a scalar matrix multiplication, not a compute
  kernel, and `simulate_psychedelic_trajectory` is a wrapper that
  composes `UPDEEngine.run`, `compute_itpc`, `detect_chimera`,
  and the new `entropy_from_phases` dispatcher.
- 16 new tests — `tests/test_psychedelic_algorithm.py` (9
  algorithmic + Hypothesis incl. translation invariance,
  log(n_bins) upper bound, n_bins parameter effect),
  `tests/test_psychedelic_backends.py` (7 cross-backend parity,
  Hypothesis sweeps for Rust / Go).
- Parity measured at 4.4e-16 on Rust / Julia / Go and 1.2e-12 on
  Mojo (under the 1e-9 tolerance).

### Added (2026-04-18 — poincare multi-backend)
- `julia/poincare.jl`, `go/poincare.go` (→ `libpoincare.so`),
  `mojo/poincare.mojo` (→ `poincare_mojo`) implementing the
  generic hyperplane Poincaré section and the phase-oscillator
  variant (2π wrap detection) with linear interpolation.
- Python bridges `monitor/_poincare_julia.py`,
  `monitor/_poincare_go.py`, `monitor/_poincare_mojo.py`.
- `monitor/poincare.py` upgraded to five-backend dispatcher with a
  caller-preallocated output convention: both kernels return
  ``(crossings_flat, times, n_crossings)`` where the output
  buffers are sized to ``T * d`` / ``T * N`` (worst case) and the
  dispatcher reshapes the populated prefix.
- 19 new tests — `tests/test_poincare_algorithm.py` (10
  algorithmic + Hypothesis + direction filtering),
  `tests/test_poincare_backends.py` (5 cross-backend parity with
  array-exact crossing counts and 1e-9 coord tolerances),
  `tests/test_poincare_stability.py` (4 long-run invariants,
  `pytest.mark.slow`).
- Parity measured bit-exact across Rust / Julia / Go and 1.5e-18
  on Mojo (subprocess text round-trip).

### Added (2026-04-18 — embedding primitives multi-backend)
- `julia/embedding.jl`, `go/embedding.go` (→ `libembedding.so`),
  `mojo/embedding.mojo` (→ `embedding_mojo`) implementing three
  delay-embedding primitives: `delay_embed`, `mutual_information`
  (Fraser-Swinney 1986), `nearest_neighbor_distances` (k=1 brute
  kNN for FNN).
- Python bridges `monitor/_embedding_julia.py`,
  `monitor/_embedding_go.py`, `monitor/_embedding_mojo.py`.
- `monitor/embedding.py` upgraded to five-backend dispatcher.
  Rust has no standalone MI or kNN FFI — the dispatcher falls
  through to the next available backend for those two, while
  keeping Rust's native `optimal_delay_rust` and
  `optimal_dimension_rust` fast paths for the wrappers.
- 25 new tests — `tests/test_embedding_algorithm.py` (12
  algorithmic + Hypothesis), `tests/test_embedding_backends.py`
  (10 cross-backend parity including Rust-active MI/NN
  fall-through), `tests/test_embedding_stability.py` (3 long-run
  invariants, `pytest.mark.slow`).
- Parity: `delay_embed` array-exact, `nearest_neighbor_distances`
  within 3e-17 and array-exact indices, `mutual_information`
  within 1e-9 (histogram bin-edge rounding vs `np.histogram2d`).

### Added (2026-04-18 — recurrence multi-backend)
- `julia/recurrence.jl`, `go/recurrence.go`
  (→ `librecurrence.so`), `mojo/recurrence.mojo`
  (→ `recurrence_mojo`) implementing the Eckmann 1987 recurrence
  matrix and cross-recurrence matrix with both euclidean and
  angular (chord-distance) metrics.
- Python bridges `monitor/_recurrence_julia.py`,
  `monitor/_recurrence_go.py`, `monitor/_recurrence_mojo.py`.
- `monitor/recurrence.py` upgraded to five-backend dispatcher.
  `rqa` and `cross_rqa` now always use the dispatched matrix and
  a Python-side line-length analysis for uniform behaviour across
  backends.
- Rewrote `benchmarks/recurrence_benchmark.py` as a multi-backend
  wall-clock harness (was a Rust-only Criterion-style print).
- 29 new tests — `tests/test_recurrence_algorithm.py` (14
  algorithmic + Hypothesis incl. angular-metric correctness),
  `tests/test_recurrence_backends.py` (12 cross-backend parity
  with **array-exact** boolean equality; both metrics),
  `tests/test_recurrence_stability.py` (3 long-run invariants,
  `pytest.mark.slow`).
- `docs/reference/api/monitor_recurrence.md` (475 lines) with
  Eckmann / Marwan formalism, per-backend build notes, measured
  benchmarks, failure modes, and references.

### Added (2026-04-18 — winding multi-backend)
- `julia/winding.jl`, `go/winding.go` (→ `libwinding.so`),
  `mojo/winding.mojo` (→ `winding_mojo`) implementing the
  cumulative winding-number tracker (integer int64 output).
- Python bridges `monitor/_winding_julia.py`,
  `monitor/_winding_go.py`, `monitor/_winding_mojo.py`.
- `monitor/winding.py` upgraded to five-backend dispatcher.
- 20 new tests — `tests/test_winding_algorithm.py` (10
  algorithmic + Hypothesis), `tests/test_winding_backends.py` (7
  cross-backend parity with **array-exact** integer equality
  across every backend), `tests/test_winding_stability.py` (3
  long-run invariants incl. additivity across splits, N=64×T=10000
  stress, noise robustness; `pytest.mark.slow`).
- `benchmarks/winding_benchmark.py` multi-backend wall-clock
  harness.
- `docs/reference/api/monitor_winding.md` (395 lines) with winding
  formalism, per-backend build notes, measured benchmarks, failure
  modes, and references.

### Added (2026-04-18 — chimera multi-backend)
- `julia/chimera.jl`, `go/chimera.go` (→ `libchimera.so`),
  `mojo/chimera.mojo` (→ `chimera_mojo`) implementing the
  Kuramoto & Battogtokh 2002 local order parameter per oscillator.
- Python bridges `monitor/_chimera_julia.py`,
  `monitor/_chimera_go.py`, `monitor/_chimera_mojo.py`.
- `monitor/chimera.py` upgraded to five-backend dispatcher.
  Classification thresholds + partition stay Python-side.
- 24 new tests — `tests/test_chimera_algorithm.py` (13 algorithmic
  + Hypothesis + exact uniform-circle identity),
  `tests/test_chimera_backends.py` (8 cross-backend parity with
  Hypothesis sweeps for Rust / Go), `tests/test_chimera_stability.py`
  (3 long-run invariants incl. narrow-kernel ring chimera,
  `pytest.mark.slow`).
- `benchmarks/chimera_benchmark.py` multi-backend wall-clock
  harness.
- `docs/reference/api/monitor_chimera.md` (420 lines) with
  Kuramoto-Battogtokh formalism, per-backend build notes,
  measured benchmarks, failure modes, and references.

### Added (2026-04-18 — dimension multi-backend)
- `julia/dimension.jl`, `go/dimension.go`
  (→ `libdimension.so`), `mojo/dimension.mojo`
  (→ `dimension_mojo`) implementing the Grassberger-Procaccia 1983
  correlation integral and Kaplan-Yorke 1979 dimension.
- Python bridges `monitor/_dimension_julia.py`,
  `monitor/_dimension_go.py`, `monitor/_dimension_mojo.py`.
- `monitor/dimension.py` upgraded to five-backend dispatcher.
  Pair-subsampling RNG is now Python-owned and threaded through to
  every non-Rust backend for deterministic cross-backend parity on
  the full-pairs branch; Rust retains its in-kernel RNG for API
  stability.
- 28 new tests — `tests/test_dimension_algorithm.py` (14
  algorithmic + Hypothesis + analytic KY limits),
  `tests/test_dimension_backends.py` (11 cross-backend parity with
  Hypothesis sweeps for Rust / Go), `tests/test_dimension_stability.py`
  (3 long-run invariants, `pytest.mark.slow`).
- `benchmarks/dimension_benchmark.py` multi-backend wall-clock
  harness.
- `docs/reference/api/monitor_dimension.md` (457 lines) with
  Grassberger-Procaccia + Kaplan-Yorke formalism, per-backend build
  notes, measured benchmarks, failure modes, and references.

### Added (2026-04-18 — entropy_prod multi-backend)
- `julia/entropy_prod.jl`, `go/entropy_prod.go`
  (→ `libentropy_prod.so`), `mojo/entropy_prod.mojo`
  (→ `entropy_prod_mojo`) implementing the overdamped-Kuramoto
  dissipation rate Σ (dθ/dt)² · dt.
- Python bridges `monitor/_entropy_prod_julia.py`,
  `monitor/_entropy_prod_go.py`, `monitor/_entropy_prod_mojo.py`.
- `monitor/entropy_prod.py` upgraded to five-backend dispatcher
  (`ACTIVE_BACKEND`, `AVAILABLE_BACKENDS`).
- 21 new tests — `tests/test_entropy_prod_algorithm.py` (11
  algorithmic + Hypothesis + dispatcher), `tests/test_entropy_prod_backends.py`
  (7 cross-backend parity at 1e-12 / 1e-9),
  `tests/test_entropy_prod_stability.py` (3 long-run invariants
  including a UPDEEngine-coupled synchronisation trace,
  `pytest.mark.slow`).
- `benchmarks/entropy_prod_benchmark.py` multi-backend wall-clock
  harness across N ∈ {16, 64, 256, 1024}.
- `docs/reference/api/monitor_entropy_prod.md` (548 lines) with
  Acebrón 2005 formalism, per-backend build notes, measured
  benchmarks on the local host, failure-mode audit, and references.

### Added (2026-04-18 — itpc multi-backend)
- `julia/itpc.jl`, `go/itpc.go` (→ `libitpc.so`), `mojo/itpc.mojo`
  (→ `itpc_mojo`) implementing Lachaux 1999 inter-trial phase
  coherence (`compute_itpc` + `itpc_persistence`).
- Python bridges `monitor/_itpc_julia.py`, `monitor/_itpc_go.py`,
  `monitor/_itpc_mojo.py`.
- `monitor/itpc.py` upgraded to five-backend dispatcher
  (`ACTIVE_BACKEND`, `AVAILABLE_BACKENDS`).
- 31 new tests — `tests/test_itpc_algorithm.py` (18 algorithmic +
  Hypothesis), `tests/test_itpc_backends.py` (10 cross-backend
  parity at 1e-12 for Rust / Julia / Go, 1e-9 for Mojo),
  `tests/test_itpc_stability.py` (3 long-run invariants,
  `pytest.mark.slow`).
- `benchmarks/itpc_benchmark.py` multi-backend wall-clock harness.
- `docs/reference/api/monitor_itpc.md` (453 lines) with Lachaux
  1999 formalism, per-backend build notes, measured benchmarks on
  the local host, and failure-mode audit.

### Added (2026-04-18 — upde engine multi-backend)
- `julia/upde_engine.jl`, `go/upde_engine.go`
  (→ `libupde_engine.so`), `mojo/upde_engine.mojo`
  (→ `upde_engine_mojo`) implementing the Sakaguchi-Kuramoto UPDE
  batched integrator with Euler, RK4 and Dormand-Prince RK45 with
  adaptive step-size control.
- Python bridges `upde/_engine_julia.py`, `upde/_engine_go.py`,
  `upde/_engine_mojo.py`.
- Module-level `upde_run` stateless kernel in `upde/engine.py` with
  5-backend dispatcher (`ACTIVE_BACKEND`, `AVAILABLE_BACKENDS`).
  `UPDEEngine.run` now routes through `upde_run` so every available
  toolchain is exercised.
- Python reference `_upde_run_python` with RK4 / Euler substepping
  + inline Dormand-Prince tableau matching
  `spo-engine/src/upde.rs` bit-for-bit (verified against Rust to
  1e-12, Mojo to 1e-6).
- 42 new tests — `tests/test_upde_run_algorithm.py` (13
  algorithmic properties incl. Hypothesis),
  `tests/test_upde_run_backends.py` (26 cross-backend parity across
  3 methods × multiple seeds), `tests/test_upde_run_stability.py` (3
  long-run invariants, `pytest.mark.slow`).
- `benchmarks/upde_engine_benchmark.py` multi-backend wall-clock
  harness across sizes × methods.
- `docs/reference/api/upde_engine.md` extended with 5-backend
  section + measured benchmark table on the local host.

### Added (2026-04-18 — lyapunov spectrum multi-backend)
- `julia/lyapunov.jl`, `go/lyapunov.go` (→ `liblyapunov.so`),
  `mojo/lyapunov.mojo` (→ `lyapunov_mojo`) implementing the Benettin
  1980 / Shimada-Nagashima 1979 spectrum with RK4 integration and
  periodic row-oriented Modified Gram-Schmidt.
- Python bridges `monitor/_lyapunov_julia.py`, `monitor/_lyapunov_go.py`,
  `monitor/_lyapunov_mojo.py`.
- `monitor/lyapunov.py` upgraded to five-backend dispatcher
  (`ACTIVE_BACKEND`, `AVAILABLE_BACKENDS`). `LyapunovGuard` is
  preserved unchanged as a stateful single-backend observer.
- Reference Python kernel switched from forward Euler + coupling-only
  Jacobian to RK4 + driver-diagonal Jacobian + row-oriented QR so all
  backends (Rust, Mojo, Julia, Go, Python) agree bit-for-bit on the
  same problem instance.
- 32 new tests — `tests/test_lyapunov_algorithm.py` (14 algorithmic
  properties + Hypothesis), `tests/test_lyapunov_backends.py` (15
  cross-backend parity, including driver and phase-lag cases),
  `tests/test_lyapunov_stability.py` (3 long-run invariants, marked
  `pytest.mark.slow`).
- `benchmarks/lyapunov_benchmark.py` multi-backend wall-clock harness
  (warm-up + sized sweep at `N ∈ {4, 8, 16, 32}`).
- `docs/reference/api/monitor_lyapunov.md` (643 lines) covering the
  variational equation, Benettin algorithm, row-MGS convention,
  per-backend build notes, measured benchmarks on the local host,
  failure modes, and references.

### Added (2026-04-17 — transfer_entropy multi-backend)
- `julia/transfer_entropy.jl`, `go/transfer_entropy.go`
  (→ `libtransfer_entropy.so`), `mojo/transfer_entropy.mojo`
  (→ `transfer_entropy_mojo`) implementing Schreiber 2000 phase
  transfer entropy (pairwise and full matrix).
- Python bridges `monitor/_te_julia.py`, `monitor/_te_go.py`,
  `monitor/_te_mojo.py`.
- `monitor/transfer_entropy.py` upgraded to five-backend dispatcher.
- 14 new tests (8 per-backend parity + 6 stability/slow).
- `benchmarks/transfer_entropy_benchmark.py` multi-backend harness.
- `docs/reference/api/monitor_transfer_entropy.md` (600 lines)
  covering Schreiber TE formalism, 5-backend chain, measured
  benchmarks, physical invariants, comparisons with Granger / PLV /
  PID.

### Added (2026-04-17 — NPE multi-backend)
- `julia/npe.jl`, `go/npe.go` (→ `libnpe.so`), `mojo/npe.mojo`
  (→ `npe_mojo`) implementing the normalised persistent entropy
  and the circular phase-distance matrix.
- Python bridges `monitor/_npe_julia.py`, `monitor/_npe_go.py`,
  `monitor/_npe_mojo.py`.
- `monitor/npe.py` upgraded to five-backend dispatcher.
- 17 new tests (12 per-backend parity + 5 stability/slow).
- `benchmarks/npe_benchmark.py` multi-backend harness.

### Added (2026-04-17 — PAC multi-backend)
- Julia port `julia/pac.jl`, Go port `go/pac.go` (→ `libpac.so`),
  Mojo port `mojo/pac.mojo` (→ `pac_mojo` executable) for Tort 2010
  phase-amplitude coupling.
- Python bridges `upde/_pac_julia.py`, `upde/_pac_go.py`,
  `upde/_pac_mojo.py`.
- `upde/pac.py` now exposes `ACTIVE_BACKEND` / `AVAILABLE_BACKENDS`
  and dispatches `modulation_index` / `pac_matrix` fastest-first
  across the five backends.
- `tests/test_pac_backends.py` — per-backend parity (Rust/Julia/Go
  bit-exact, Mojo ≤ 1e-10).
- `tests/test_pac_stability.py` — MI bounded in [0, 1], monotonic in
  modulation depth, diagonal-vs-off for locked signals. Marked
  `pytest.mark.slow`.
- `benchmarks/pac_benchmark.py` — multi-backend wall-clock harness.

### Added (2026-04-17 — order_params multi-backend)
- Julia port `julia/order_params.jl`, Go port `go/order_params.go`
  (→ `liborder_params.so`), Mojo port `mojo/order_params.mojo`
  (→ `order_params_mojo` executable).
- PyO3 export for `compute_layer_coherence`.
- `upde/order_params.py` upgraded to five-backend dispatcher.
- 39 new tests (20 algorithm + 13 per-backend parity + 6 stability).

### Changed — AttnRes upgraded to full multi-head (2026-04-17)

Following the new ``feedback_no_simplistic_models.md`` rule, the
Phase-3 AttnRes spike was upgraded from a single-equation Hebbian
proxy to the full arXiv:2603.15031 Transformer architecture:

- **Full multi-head implementation** — Fourier-feature phase
  embedding (``d_model = 8``), ``H = 4`` attention heads with
  learnable Q/K/V projections (seeded Xavier init via new
  ``default_projections()``), scaled dot-product softmax attention
  (paper-faithful full-N scope; optional ``block_size`` local mask),
  output projection ``W_O``, symmetric cosine-similarity
  aggregation onto ``K_nm``.
- **All 5 backends re-ported** — Rust, Julia, Go, Mojo, Python all
  now carry the full multi-head kernel. Parity: bit-exact
  (5.55e-17) for Rust/Julia/Go; 1.55e-14 for Mojo (text-protocol
  rounding budget).
- **PyO3 signature extended** to carry the four projection buffers
  plus ``n_heads`` and a signed ``block_size`` (``-1`` = unbounded
  full attention).
- **Test coverage** — 20 algorithm tests, 14 per-backend parity
  tests, 3 stability tests (marked slow); 37 AttnRes-specific
  tests total, all pass.
- **Old single-head ports deleted** per the option-B clause in the
  new rule: the simplified kernels would have shipped as toys
  alongside the full model, so they were removed from main rather
  than left to decay.

### Added — AttnRes multi-language fallback chain (Phase-3 spike)
- `coupling/attention_residuals.py` — new ``attnres_modulate`` pure
  function plus multi-backend dispatcher following the global
  fastest-first rule (Rust → Mojo → Julia → Go → Python). Two public
  attributes ``ACTIVE_BACKEND`` and ``AVAILABLE_BACKENDS`` let callers
  see which backends loaded on the current host.
- `spo-kernel/crates/spo-engine/src/attnres.rs` — Rust implementation
  with single-scratch-buffer design, no Rayon (measured slower at
  SPO-realistic N ≤ 64), 11 pure-Rust unit tests, criterion bench in
  ``utility_bench``. PyO3 binding returns
  ``Bound<PyArray1<f64>>`` directly to avoid Vec → PyList overhead.
- `julia/attnres.jl` + `coupling/_attnres_julia.py` — Julia port with
  lazy ``juliacall`` bridge. Bit-exact parity with the NumPy reference.
- `go/attnres.go` + `coupling/_attnres_go.py` — Go port compiled to
  c-shared ``libattnres.so``; ctypes bridge. Bit-exact parity.
- `mojo/attnres.mojo` + `coupling/_attnres_mojo.py` — Mojo port with
  subprocess bridge using a single-line text protocol (Mojo 0.26
  ``UnsafePointer`` C-ABI is in transition; documented upgrade path
  to ``@export(ABI="C")`` + shared library once the pointer surface
  stabilises in 0.27+). Parity within 7.72e-15.
- `benchmarks/attnres_modulation_benchmark.py` — multi-backend
  overhead measurement against the baseline ``UPDEEngine.step``.
  Verified Rust speedup of 2.5–4.5× over the NumPy fallback on
  N ∈ {16, 64, 128, 256}.
- `tests/test_attention_residuals.py` (17 tests) + new
  `tests/test_attention_residuals_backends.py` (13 tests) —
  per-backend parity, symmetry, zero-diagonal, no-new-edges,
  block-window, lambda=0 identity, contract failures, R-within-5 %
  validation against the baseline.
- New global rule at
  `feedback_fallback_chain_ordering.md` — every multi-language
  compute dispatcher across GOTM orders backends fastest-first.

### Changed — type discipline (continued)
- `attention_residuals.py` dispatcher uses a canonical
  ``_BackendFn`` Callable alias so loader functions are strictly
  typed; the only remaining ``type: ignore`` is on the juliacall
  import (juliacall ships no py.typed marker) and is documented
  inline with its reason.

### Security
- `adapters/modbus_tls.py` no longer echoes the full private-key or
  certificate path in `ConnectionError` messages — only the filename.
- `binding/loader.py` scrubs paths from YAML / JSON / missing-file
  errors via `path.name` + `OSError.strerror`.
- `adapters/lsl_bci_bridge.py` stops echoing the configured
  `stream_name` in the "Could not connect" RuntimeError.
- `adapters/remanentia_bridge.py` stops echoing the offending URL
  when rejecting a non-http(s) scheme.
- `modbus_tls` enables `CERT_REQUIRED` + hostname verification when
  a CA bundle is configured (previously `CERT_NONE` by default).

### Added — amplitude metric chain in QueueWaves pipeline
- `PhaseComputePipeline.tick()` now populates `mean_amplitude`,
  `subcritical_fraction`, and `pac_max` on the emitted `UPDEState`
  using a 32-tick rolling window of Hilbert amplitudes. Policy rules
  referencing those metrics now fire as documented.
- `ServiceSnapshot.amplitude` reports the real Hilbert envelope
  instead of a hard-coded 1.0.

### Added — criterion bench suite expansion
- `spo-kernel/crates/spo-engine/benches/parallel_bench.rs` extended
  to cover all salvaged Rayon compute paths (chimera, bifurcation,
  dimension, poincare, market, coupling_est, sindy, kaplan_yorke).
- `spo-kernel/crates/spo-engine/benches/monitors_bench.rs` (new) —
  8 sequential compute paths (spectral, lyapunov, transfer_entropy,
  embedding, recurrence, hodge, entropy_prod, pid).
- `spo-kernel/crates/spo-engine/benches/utility_bench.rs` (new) —
  12 remaining compute paths (basin_stability, itpc, ei_balance,
  splitting, imprint, npe, evs, phase_extract, carrier, ethical,
  connectome, oa_run).
- Coverage across 4 bench binaries is now ~24 spo-engine modules.

### Changed — type discipline
- `src/scpn_phase_orchestrator/server.py` `_lifespan` annotated
  `AsyncIterator[None]` (was missing a return type).
- `tools/benchmark_summary.py` rewritten to use an argv list plus
  `env=` parameter in place of `shell=True` with a `noqa: S602`
  suppression; canonical SPDX header added (the file had none).
- `tools/coverage_guard.py` shebang moved to line 1 (was at line 9
  after SPDX, where the kernel does not honour it); `evaluate()`
  narrows the `dict[str, object]` global threshold via isinstance
  rather than casting directly.
- `tools/generate_header.py` modernised matplotlib calls:
  `add_axes` / `imshow extent` take float tuples; `plt.cm.cool`
  replaced with `plt.get_cmap("cool")`.
- `tools/gpu_benchmark.py` `BENCHMARKS` annotated
  `list[tuple[str, Callable[[], dict]]]`.
- Combined `mypy src/scpn_phase_orchestrator tools/` reaches 176
  files, 0 errors (from 16).

### Added — cross-platform tooling
- `tools/normalise_spdx_headers.py` — one-shot normaliser that
  converts the 6-line merged SPDX variant to the canonical 7-line
  form with `--dry-run` / `--apply` / `--verify` modes. Applied to
  632 files in this release.
- `tools/generate_grpc.py` — cross-platform Python port of
  `generate_grpc.sh`, callable from Windows without WSL.

### Added — FPGA synthesisable Verilog
- `KuramotoVerilogCompiler` now emits Q16.16 fixed-point Verilog that
  instantiates `cordic_sincos` from `spo-fpga/kuramoto_core.v` for
  every non-zero K_ij entry — replacing the previous simulation-only
  `$sin(...)` placeholder. 30-case test suite covers encoding,
  module structure, synthesisability, CORDIC instantiation and the
  summation chain.

### Added — observability
- Structured `logging.getLogger(__name__)` instrumentation on
  `server.py` (lifespan startup / shutdown, api.reset, api.step),
  `server_grpc.py` (Step / Reset RPCs) and `supervisor/policy.py`
  (regime + action count + knob list extras on every `decide()` call).

### Added — concurrent-safety primitives
- `threading.RLock()` around `UPDEEngine.step/run` and
  `StuartLandauEngine.step` so shared pre-allocated scratch arrays
  cannot be corrupted by concurrent callers (multi-client gRPC /
  WebSocket deployments).
- `SimulationState._lock` unified to `threading.Lock`; REST and gRPC
  now serialise against the same mutex.
- `GaianMeshNode` gained `__enter__` / `__exit__` so
  `with GaianMeshNode(...)` releases sockets on exit, including on
  exception paths. FastAPI `create_app` installs an
  `asynccontextmanager` lifespan that clears the event bus on
  shutdown.

### Added — constructor validation
- `UPDEEngine`, `SwarmalatorEngine`, `SimplicialEngine`, `DelayBuffer`,
  `DelayedEngine`, `InertialKuramotoEngine`, `OttAntonsenReduction`,
  `JaxUPDEEngine`, `JaxStuartLandauEngine` raise `ValueError` on
  non-positive `n` / `dt`, negative `sigma²`, zero dim, unknown
  integration method.
- `WebhookAlerter.cooldown_seconds`, `MetricBuffer.maxlen`,
  `PhaseSINDy.threshold / max_iter`, `LyapunovGuard.basin_threshold`,
  `EventBus.maxlen`, `PGBO.cost_weights` all gain up-front validation.

### Performance
- `reporting/plots.py` defers the matplotlib import to the first
  plot call. Importing `CoherencePlot` no longer triggers matplotlib
  backend init or font cache loading on CLI / server paths.
- Salvaged Rayon parallelisation for `chimera::local_order_parameter`,
  `bifurcation::trace_sync_transition`, `dimension`, `market`,
  `poincare`, `coupling_est`, `sindy`, `spo-wasm`, `active_inference`
  from the stalled perf branches. Tests + SPDX / ORCID / Contact
  headers preserved throughout. Dedicated criterion benchmarks added
  in `spo-engine/benches/parallel_bench.rs`.

### Fixed
- `pac_matrix_compute` docstring now matches the implementation
  (row-major, not column-major). Callers must pass `ravel(order="C")`.
- `spo-engine/benches/upde_bench.rs` failed to compile; three call
  sites now pass `&mut cs.knm` to match the `UPDEStepper.step/run`
  mutability contract.
- 26 ruff errors and 8 cargo warnings introduced by this session's
  commits — all cleaned. `cargo check --all` and
  `ruff check src/ tests/ tools/` are zero-warning.

### Changed
- SPDX headers normalised to canonical 7-line format across 632
  files (Python, Rust, YAML, TOML, Shell, Markdown). The previous
  6-line merged `SPDX-License-Identifier: ... | Commercial license
  available` variant is no longer present in-repo.
- `docs/reference/api/{coupling,monitor,upde}.md`, CHANGELOG and
  README no longer use internal quality tier names. Neutral
  descriptive language replaces them.
- Incidental `protoscience@anylum.li` email typo corrected to
  `protoscience@anulum.li` in 9 files.

### Tests — Phase-7 property-based invariants
- `test_property_monitors.py` (new, 6 tests) — PID non-negativity and
  empty-group zero-return; phase transfer entropy bounded by
  `log(n_bins)` for independent signals and for self-TE; Kuramoto
  supercritical monotonicity (R at K=1.0 is at least R at K=0.4 within
  a narrow frequency spread) and incoherent-state ceiling at K=0.
- `test_property_engine.py` (new, 4 tests) — UPDE ``step`` is
  equivariant under oscillator-index permutations (compared modulo
  2π on the torus); identity permutation is fixed bit-for-bit;
  Dense↔Sparse CSR parity for both Euler and RK4 across Hypothesis
  sparsity masks (10–90%).

### Tests — CI-tool defensive coverage
- `test_bench_compare_baseline.py` (new, 14 tests) — the CI
  benchmark-regression guard is now under test: PASS/FAIL branches,
  list + dict baseline layouts, missing-key / zero-baseline skip,
  CLI argv length and malformed JSON surfaces.
- `test_tools_normalise_spdx.py` (new, 23 tests) — SPDX split logic
  (`#` / `//` / bare), typo fix, `HEADER_SCAN_LIMIT`, excluded dirs
  and `.venv` prefixes, iter filter, dry-run vs apply atomicity,
  verify mode exit codes.
- `test_tools_coverage_guard.py` (new, 32 tests) — validator bounds
  (NaN/Inf rejected), domain parser (Windows-style paths included),
  Cobertura XML parser, thresholds schema, per-global / per-domain /
  per-file evaluation branches, main() integration.
- `test_tools_check_version_sync.py` (new, 9 tests) — pyproject /
  CITATION / Cargo version extraction; nested `[dependencies]
  version = ...` not mis-matched; mismatch and missing-file exits.
- `test_tools_check_module_linkage.py` (new, 15 tests) — module
  discovery (`__init__.py` excluded), import-path construction,
  linkage via dotted import or `test_<stem>` reference, allowlist
  schema validation, stale-entry detection, `--allow-stale-allowlist`
  bypass.

### Tests — S6 THIN file strengthening
- `test_ffi_parity.py` 4 → 15 (Sakaguchi lag, external drive,
  negative / asymmetric coupling, N=64 scale, degenerate zero state,
  run() batch, determinism, N=1 / antiphase / full-sync order
  parameters).
- `test_pac_parity.py` 3 → 14 (true Python ↔ Rust parity restored;
  parametrised bin counts; degenerate n_bins < 2; constant amplitude;
  fully synchronous; short series; matrix shape + diagonal).
- `test_sindy.py` 1 → 8 (zero-coupling sparsification; N=5 stability;
  threshold sparsifier contract; equation-dump format; empty input).
- `test_sparse_engine.py` 3 → 9 (zero CSR parity, RK45 50% density,
  fully-dense parity, Sakaguchi lag, invalid method, N=1 decouple).
- `test_sheaf_engine.py` 2 → 7 (zero restriction maps, D=1 long-run
  parity, per-dim external drive, single oscillator, wrap contract).
- `test_semantic_compiler.py` 2 → 12 (default layers / base freq;
  fusion / cell keyword routing; case-insensitive regex; decade
  scaling; eight oscillators per layer; empty prompt fallback).
- `test_quantum_bridge.py` 2 → 17 (constructor guards; import_artifact
  defaults and edges; import_knm validation; round-trip).
- `test_active_inference_agent.py` 2 → 11 (directionality,
  target_r bounds, learning rate effect, repeated-call stability).
- `test_viz_streamer.py` 2 → 11 (defaults, primitives, empty
  containers, multi-dim, list-of-arrays, numpy dtypes, broadcast
  no-client and before-start fast paths, deep nesting).
- `test_lsl_bridge.py` 3 → 12 (constructor defaults,
  HAS_LSL=False short-circuit, no-stream-found case, stop-before-start
  idempotence, pure / noisy / empty phase extraction, scrubbed
  error message).

### Added — Rust Path Expansion (36 → 53 spo-engine modules)
- 17 new Rust engine modules: simplicial, hypergraph, geometric, envelope, reduction, splitting, te_adaptive, prior, ethical, sleep_staging, evs, sindy, coupling_est, phase_extract, carrier, connectome, freq_id
- 17 new reference documentation pages (567+ lines each, 8 sections, verified benchmarks) for all new Rust modules
- Python `_HAS_RUST` auto-select wiring for 15 of 17 modules (coupling_est and phase_extract disabled — LAPACK/FFT faster)
- Rust test count: 243 → 567 (+324 tests across 17 modules)
- spo-ffi bindings: ~387 lines of new FFI wrapper functions
- Refactored 7 mega-functions (>50 lines) into composable helpers
- Benchmarked all 17 modules: speedups range from 2.4x (prior) to 96x (OA reduction), with 2 modules where Python/LAPACK is faster

### Added — nn/ Physics Validation Suite (194 tests, 13 phases)
- `tests/test_nn_physics_validation.py` through `_p13.py`: 194 automated physics tests validating the JAX nn/ module against analytical results
- `docs/reference/nn.md`: complete 677-line API reference for nn/ module (16 source files, 90+ symbols)
- `docs/reference/nn_physics_validation_plan.md`: validation plan with results, 14 findings register, full-suite verification
- `docs/guide/differentiable_kuramoto.md`: 7 new sections (Winfree, theta neuron, chimera, spectral, analytical inverse, training loop, GPU benchmarks)
- `benchmarks/results/gpu_benchmark_2026-03-29.json`: first local GPU validation (GTX 1060 6GB, JAX 0.9.2)
- 14 findings documented: K symmetry broken by training (#7, HIGH, confirmed by 3 codebases), UDE extrapolation NaN (#4, HIGH), BKT vs mean-field topology-dependent (#11, CRITICAL), inverse ill-conditioned at K=0 (#12), and 10 more
- First automated FIM (strange loop) validation: sync at K=0, gradient trainable, Lyapunov function V = -ΣK cos(Δθ) - λR²
- Cross-project sync with scpn-quantum-control (NB37-43) and sc-neurocore (v3.14.0)

### Added — Final Examples + Strategic Docs (23 → 25)
- `imprint_memory.py`: coupling that remembers past synchronisation (Hebbian memory)
- `petri_policy_demo.py`: Petri net FSM regime transitions + event bus
- `docs/guide/digital_twin.md`: how SPO fits into digital twin architectures
- `docs/guide/notebook_to_production.md`: exploration → deployment lifecycle
- `docs/competitive_comparison.md`: capability comparison table vs Brian2/TVB/neurolib/pyDSTool

### Added — Deep Physics Examples (17 → 23)
- `plasticity_learning.py`: Hebbian eligibility + TE-adaptive coupling evolving over time
- `ssgf_closure_loop.py`: geometry → dynamics → cost → gradient → geometry self-organisation
- `hodge_decomposition.py`: gradient vs curl vs harmonic coupling flow (3 cases)
- `stochastic_resonance.py`: noise improves synchronisation at optimal D*
- `multi_engine_comparison.py`: UPDE Euler/RK4 vs TorusEngine vs SplittingEngine
- `audit_replay_demo.py`: SHA-256 chain, deterministic replay, tamper detection

### Added — Interactive Tools & Media
- `tools/spo_studio.py`: Streamlit GUI — browse 33 domainpacks, tune K/ζ/Ψ knobs, live R(t) chart, regime timeline, per-layer breakdown (`streamlit run tools/spo_studio.py`)
- `docs/demo/index.html`: WASM interactive demo — 66KB Rust→WebAssembly Kuramoto in browser, real-time R(t) + phase portrait, sliders for N/K/spread/dt
- `docs/video_scripts.md`: 7 × 60-second demo scripts with voiceover text for Loom/OBS recording

### Added — Showcase Examples (12 → 17)
- `supervisor_advantage.py`: open-loop vs closed-loop coherence, quantified % improvement
- `failure_recovery.py`: inject coupling fault, detect R drop, boost remaining links, recover
- `cross_domain_universality.py`: same 4-line pattern across plasma, cardiac, power, traffic, neuro
- `scaling_showcase.py`: N=4 to N=1000 with wall-clock timing per step
- `inverse_coupling_demo.py`: learn hidden coupling matrix from observed phase trajectories

### Added — Adoption & Ecosystem
- `spo demo` CLI command: one-command full-stack demo for any domainpack
- 32-domainpack benchmark table in domainpack gallery (measured R values, Kaggle Linux)
- Kaggle reproducibility scripts: `tools/kaggle_demo_all32.py`, `tools/kaggle_mutation_test.py`
- 5 papers + 2 validation docs published on mkdocs site
- `.github/FUNDING.yml`: Polar.sh added for sponsorship
- 3 good first issues created (#27 OPC-UA, #28 ROS2, #29 theta neuron Equinox)
- Real-data ingestion examples: EEG file → Hilbert → phases, Prometheus → QueueWaves

### Added — Infrastructure Hardening
- `/api/health` deep health endpoint (engine + R + regime checks)
- `test_grpc_integration.py`: 6 in-process gRPC servicer tests (GetState, Step, Reset, GetConfig, layers)
- Trivy container security scanning in publish pipeline (blocks on CRITICAL/HIGH)
- GHCR image push (`ghcr.io/anulum/scpn-phase-orchestrator`) with version + latest tags
- Dockerfile HEALTHCHECK upgraded from import-only to `/api/health`
- Production guide: health check docs, GHCR registry, Trivy scanning, updated Dockerfile examples

### Added — Mutation Testing & Killer Tests
- `test_mutation_killers.py`: 32 tests targeting mutants that survived mutmut analysis (order_params.py: 16 survivors killed, numerics.py: 5 survivors killed)
- Mutation testing pipeline on Kaggle (mutmut 2.4.5, kernel: anulum/spo-mutmut-v2)
- Testing guide: mutation testing results, methodology, Kaggle/WSL instructions

### Added — Domain Examples (3 → 10)
- `examples/neuroscience_eeg.py`: 8-electrode EEG alpha-band synchronization with chimera detection
- `examples/cardiac_rhythm.py`: SA node pacemaker, AV block scenario, external drive recovery
- `examples/plasma_control.py`: tokamak MHD mode locking with Lyapunov guard
- `examples/traffic_flow.py`: 8-intersection green wave, link failure, coupling boost recovery
- `examples/epidemic_sir.py`: 6-region epidemic synchronization with transfer entropy causality
- `examples/swarmalator_dynamics.py`: phase-spatial coupling sweep (J=0 to J=2)
- `examples/stuart_landau_bifurcation.py`: Hopf bifurcation μ sweep, r → √μ analytical comparison

### Added — Cross-Engine Parity & Analytical Validation
- `test_engine_parity.py`: UPDE vs TorusEngine vs SplittingEngine vs Simplicial equivalence matrix; free rotation exact match across 3 engines; spectral K_c vs bifurcation simulation; Stuart-Landau r → √μ property-based proof
- `test_engine_rigor.py`: 27 dedicated tests for HypergraphEngine, market module, envelope solver, adjoint gradients, DelayBuffer/DelayedEngine
- `test_stress_scale.py`: N=1000 oscillators, T=50000 steps, OA analytical validation (K_c = 2Δ, R_ss formula, OA vs UPDE on Lorentzian)

### Added — Property-Based Test Suite (680+ new tests)
- 14 property-based test files (`test_prop_*.py`) proving mathematical invariants via hypothesis: Lyapunov spectrum bounds, Kaplan-Yorke dimension, basin stability, transfer entropy, Hodge decomposition, spectral graph theory, recurrence/RQA, chimera detection, winding numbers, Boltzmann weights, SSGF costs, delay embedding, EI balance, NPE, simplicial reduction, swarmalator/inertial dynamics, plasticity, stochastic injection
- `test_degenerate_edges.py`: 98 boundary tests (N=1, dt=0, zero coupling) across all 5 engine types
- `test_roundtrip_consistency.py`: 86 cross-module mathematical consistency proofs
- 6 new module test files: `test_ssgf_modules.py`, `test_upde_math.py`, `test_coupling_modules.py`, `test_drivers_oscillators.py`, `test_supervisor_modules.py`, `test_imprint_actuation.py`
- Expanded coverage-gap tests for bifurcation, dimension, embedding, recurrence, predictive supervisor
- Total: 2,420 → 3,008 tests (24.3% increase), 99.33% coverage

### Added — Testing Documentation
- `docs/guide/testing.md`: testing guide with hypothesis profiles, test architecture, invariant catalogue, contribution patterns

### Added — Differentiable Phase Dynamics (`nn/` module)
- `nn/functional.py`: `kuramoto_step`, `kuramoto_rk4_step`, `kuramoto_forward` — JAX differentiable Kuramoto with JIT, vmap, autodiff
- `nn/functional.py`: `simplicial_step`, `simplicial_rk4_step`, `simplicial_forward` — first differentiable 3-body Kuramoto (Gambuzza 2023)
- `nn/functional.py`: `stuart_landau_step`, `stuart_landau_rk4_step`, `stuart_landau_forward` — differentiable phase + amplitude dynamics
- `nn/functional.py`: `saf_order_parameter`, `saf_loss`, `coupling_laplacian` — spectral alignment function for topology optimization (Skardal & Taylor 2016)
- `nn/kuramoto_layer.py`: `KuramotoLayer` — equinox.Module with learnable K and ω
- `nn/stuart_landau_layer.py`: `StuartLandauLayer` — equinox.Module with learnable K, K_r, ω, μ
- `nn/bold.py`: `bold_from_neural`, `bold_signal` — Balloon-Windkessel BOLD generator (Friston 2000)
- `nn/reservoir.py`: `reservoir_drive`, `ridge_readout`, `reservoir_predict` — Kuramoto reservoir computing
- `nn/ude.py`: `UDEKuramotoLayer`, `CouplingResidual` — physics backbone + learned neural residual (UDE)
- `nn/inverse.py`: `infer_coupling`, `inverse_loss`, `coupling_correlation` — gradient-based inverse Kuramoto
- `nn/oim.py`: `oim_forward`, `extract_coloring`, `coloring_energy` — oscillator Ising machine for graph coloring

### Added — NumPy Dynamics Engines
- `upde/inertial.py`: `InertialKuramotoEngine` — second-order swing equation for power grids (Filatrella 2008)
- `upde/market.py`: `extract_phase`, `market_order_parameter`, `detect_regimes`, `sync_warning` — financial market regime detection
- `upde/swarmalator.py`: `SwarmalatorEngine` — coupled spatial + phase dynamics (O'Keeffe 2017)

### Added — Documentation
- 4 guide pages: Advanced Dynamics, Control Systems, Analysis Toolkit, Hardware & Deployment
- API reference page for nn/ module (mkdocstrings)
- Usage guide with code examples for all nn/ modules
- README capabilities section with full feature inventory

### Added — Rich API Documentation for Pre-existing Modules
- API reference pages rewritten for: stochastic engine, geometric engine,
  delay engine, Ott-Antonsen reduction, variational predictor, adjoint
  gradients, Hodge decomposition, three-factor plasticity, TE adaptive
  coupling, HCP connectome, MPC supervisor, chimera detection, EVS,
  PID, Lyapunov, entropy production, winding number, ITPC, transfer entropy
- Each page includes theory background, equations, usage examples, paper refs

### Added — Documentation Audit (2026-03-25)
- ARCHITECTURE.md rewritten: 9 UPDE engines, 15 monitors, nn/ module, ssgf/,
  autotune/, visualization/ — all 14 subpackages + 5 top-level modules documented
- Competitive analysis updated: JAX GPU, Lyapunov, AKOrN/XGI comparison, 10 use cases
- FAQ expanded: nn/ module, 9 engines, SSGF, inverse Kuramoto, OIM, stochastic
  resonance, Ott-Antonsen reduction (8 new entries)
- docs/index.md: 4 new feature cards (Differentiable, 9 Engines, 16 Monitors, Inverse)
- Gallery: notebooks table expanded 7 → 19 entries with descriptions
- Test/domainpack counts corrected across all docs (2200+ Python, 211 Rust, 32 packs)
- identity_coherence domainpack added to gallery, README, and all count references

### Changed
- Preflight excludes JAX nn/ tests (CPU XLA too slow; tests run on GPU or in CI)
- Preflight excludes test_quantum_bridge_live.py (Qiskit Aer segfault on Windows)
- `pyproject.toml`: nn extras (`jax>=0.4, equinox>=0.11`), per-file E402 ignores
- Coverage/mypy: nn/ excluded from CI gate (JAX not available in CI environment)
- pip-audit: ignore CVE-2026-4539 (pygments transitive dep, no fix available)
- Ruff bumped 0.15.6 → 0.15.7
- Pre-commit ruff pinned to v0.15.7 (was v0.15.6, caused CI format divergence)
- GitHub Actions bumped: codeql-action 4.34.1, rust-toolchain, actions/cache 5.0.4, action-gh-release 2.6.1

## [0.5.0] - 2026-03-22

### Added

- Auto-tune pipeline: Hilbert phase extraction, DMD frequency ID, coupling estimation
- Visualization: D3 network graph, coupling heatmap, Three.js torus, phase wheel
- 7 new domainpacks (32 total): financial_markets, gene_oscillator, vortex_shedding, robotic_cpg, sleep_architecture, musical_acoustics, brain_connectome
- Information theory: transfer entropy, entropy production, PID synergy/redundancy
- Topology: Hodge decomposition, winding numbers, NPE
- SSGF: free energy (Langevin noise, Boltzmann weight)
- Safety: STL runtime monitor (rtamt), Modbus TLS adapter, Kani proof stubs
- Production: Helm chart, docker-compose (Redis+Prometheus+Grafana), Prometheus metrics exporter, Redis state store, gRPC streaming, WASM crate
- Publication: 4 paper drafts (JOSS, dual R+p_h1, SSGF identity, safety cert)
- Hardware: FPGA Verilog kuramoto_core.v (Zynq-7020, CORDIC)
- Clinical: ITPC, sleep staging, validation study protocol
- Consciousness model: chimera detection, 3-factor plasticity, psychedelic simulation, HCP connectome
- `bench/bench_stuart_landau.py` — Stuart-Landau engine benchmark harness
- `bench/baseline.json` — UPDE + SL baseline timing data (Python + Rust)
- `tests/test_geometry_walk.py` — geometry_walk domainpack spec/run/policy tests
- `tests/test_queuewaves_pipeline.py` — QueueWaves ConfigCompiler + PhaseComputePipeline tests
- `tests/test_coverage_gaps.py` — 119 tests covering CLI, PAC, physical oscillator, binding, validator, supervisor, order_params, audit, Stuart-Landau RK45 edge cases
- `tests/apps/queuewaves/test_server_coverage.py` — async FastAPI route tests (health, state, history, anomalies, WebSocket)
- `tests/apps/queuewaves/test_alerter_coverage.py` — Slack webhook formatting, cooldown edge cases
- `tests/apps/queuewaves/test_collector_coverage.py` — httpx error handling paths
- `# pragma: no cover` on Rust-only FFI import branches (12 source files)
- `ARCHITECTURE.md` — system overview, pipeline diagram, module map
- `SUPPORT.md` — help channels, security advisories link
- `GOVERNANCE.md` — decision process, project lead
- `CONTRIBUTORS.md` — contributor attribution
- `NOTICE.md` — AGPL compliance boundary table, third-party attribution
- `REUSE.toml` — REUSE 3.0 license compliance
- `Makefile` — 20 convenience targets (test, lint, fmt, docs, bench, bridge, etc.)
- `requirements.txt` / `requirements-dev.txt` — pinned dependency files
- `.gitattributes` — LF normalization, linguist overrides
- `_typos.toml` — domain-specific allowlist for typos checker
- `.github/FUNDING.yml` — sponsorship links
- `.github/workflows/pre-commit.yml` — pre-commit hook enforcement in CI
- `.github/workflows/codeql.yml` — CodeQL semantic security analysis (weekly + PR)
- `.github/workflows/scorecard.yml` — OpenSSF Scorecard (weekly + push)
- `.github/workflows/stale.yml` — auto-close stale issues/PRs (60+14 day lifecycle)
- `.github/workflows/release.yml` — automated GitHub Release with changelog extraction and SBOM
- `src/scpn_phase_orchestrator/exceptions.py` — centralized exception hierarchy (`SPOError`, `BindingError`, `ValidationError`, `ExtractorError`, `EngineError`, `PolicyError`, `AuditError`)
- `resolve_extractor_type()` in `binding/types.py` — maps aliases (`physical`/`informational`/`symbolic`) to algorithm names (`hilbert`/`event`/`ring`)
- `Discussions` link in `[project.urls]`
- `spo report` command: coherence summary from audit log (text + `--json-out`)
- `spo replay --verify` for Stuart-Landau logs (chained phase-amplitude replay)
- `ReplayEngine.verify_determinism_sl_chained()` for SL audit log verification
- `linux-aarch64` wheel target in `publish.yml`

### Fixed

- **Contract drift**: Loader now resolves extractor_type aliases to algorithm names at parse time. Both `physical` and `hilbert` are valid input; internally normalized to algorithm names.
- **Audit hash chain**: `_write_record` strips `_hash` key before digest to prevent user-data collision
- **Stuart-Landau coupling**: clamp `r >= 0` in `_derivative` to prevent sign-flip in RK intermediate stages
- **CLI audit logger**: simulation loop wrapped in `try/finally` to guarantee `close()` on exception
- **Hilbert guard**: `PhysicalExtractor.extract()` validates signal shape before transform
- **Lag model**: `build_alpha_matrix` uses `carrier_freq_hz` for correct seconds→radians conversion
- **Exception swallowing**: `except Exception` → `except ImportError` in stuart_landau.py and pac.py
- **Coupling diagonal**: layer-scoped K update zeros `knm[idx,idx]` after row/column scaling
- **Regime state machine**: CRITICAL must pass through RECOVERY before NOMINAL
- **Geometry projection**: `project_knm` zeros diagonal after constraint projection (Rust parity)
- **Zeta lower bound**: `cli.py` zeta accumulation clamped to `max(0.0, ...)` to prevent negative drive
- **Control period**: `control_period_s` from binding spec now gates supervisor/policy evaluation interval
- **Gallery doc**: corrected "all notebooks validated in CI" to reflect actual 5/24 coverage

### Changed

- `build_nengo_network()` replaced with pure-NumPy `build_numpy_network()` (old name kept as alias)
- `nengo` optional extra is now empty (nengo 4.x incompatible with NumPy 2.x)
- Coverage guard thresholds raised to 95% global and per-domain
- CI installs `plot` extra for full matplotlib coverage
- `SECURITY.md` — updated supported versions, added Security Advisories link
- `__init__.py` public API expanded: +`AuditLogger`, `BoundaryObserver`, `CouplingBuilder`, `RegimeManager`, `SPOError`, `SupervisorPolicy`
- FFI: `PyStuartLandauStepper.run()` exposed via PyO3
- Coverage guard: reporting threshold raised from 40% to 90%
- `system_overview.md`: added RK45 to methods list, 5 missing data structures to key structures table
- `CONTRIBUTING.md`: added `boundaries:`, `actuators:`, and `policy.yaml` examples
- `ROADMAP.md`: corrected v0.4.1 test counts (1011 Python, 180 Rust)
- `ASSUMPTIONS.md`: corrected CFL and max_dt line references
- `upde_numerics.md`: CFL formula corrected from `1` to `pi` (matching code and ASSUMPTIONS.md)
- `policy_dsl.md`: domainpack count corrected (12/17 → 24/24), added 5 amplitude metrics to fields table
- `index.md`: Policy DSL label changed from "planned v0.2" to "v0.2+"
- `01_new_domain_checklist.md`: policy.yaml example updated to current schema
- `README.md`: 3 missing domainpacks added (autonomous_vehicles, network_security, satellite_constellation)
- `VALIDATION.md`: test counts and domain count updated (1011/180/24)
- `CITATION.cff`: added `abstract` field
- `bio_stub/README.md`: corrected boundary severity (lower: soft, not hard)
- `safety_tier` runtime warning for non-research tiers
- `pyproject.toml`: `scpn-all` extra now requires `spo-kernel>=0.2.0` (was stale `>=0.1.1`)
- `docs/index.md`: added 6 missing nav entries to match `mkdocs.yml` (concepts, specs, gallery)
- `mkdocs.yml`: added `references.bib` to Reference nav section
- `README.md`: quickstart now includes `spo scaffold` example
- `ruff.lint.ignore`: added `S603` (subprocess in tests is expected)
- `CHANGELOG.md`: corrected QueueWaves path from `apps/queuewaves/` to full module path

## [0.4.1] - 2026-03-04

### Added

- **Rust `StuartLandauStepper`** — phase-amplitude ODE integrator in `spo-engine/src/stuart_landau.rs` with Euler/RK4/RK45, zero-alloc scratch, 12 inline tests
- **Rust PAC** — `modulation_index` and `pac_matrix` in `spo-engine/src/pac.rs` (Tort et al. 2010), 5 inline tests
- **FFI `PyStuartLandauStepper`** — PyO3 wrapper delegating to Rust Stuart-Landau stepper
- **FFI `pac_modulation_index` / `pac_matrix_compute`** — PAC functions exposed to Python
- Python `StuartLandauEngine` auto-delegates to Rust when `spo_kernel` available
- Python `modulation_index()` auto-delegates to Rust when `spo_kernel` available
- PAC-driven policy rules: `pac_max`, `mean_amplitude`, `subcritical_fraction`, `amplitude_spread`, `mean_amplitude_layer` metrics in `_extract_metric`
- Amplitude configs (`amplitude:` YAML block) for 6 domainpacks: neuroscience_eeg, cardiac_rhythm, plasma_control, firefly_swarm, rotating_machinery, power_grid
- `pac_gating_alert` and `subcritical_recovery` policy rules for all 6 amplitude domainpacks
- `CoherencePlot` matplotlib implementations: `plot_r_timeline`, `plot_regime_timeline`, `plot_action_audit`, `plot_amplitude_timeline`, `plot_pac_heatmap`
- 4 Rust benchmarks: `sl_euler_step_n64`, `sl_rk4_step_n64`, `sl_1000steps_n64`, `pac_mi_n1000`
- ~30 new Python tests across 5 test files (total ~895)

## [0.4.0] - 2026-03-04

### Added

- **Stuart-Landau amplitude engine** — `StuartLandauEngine` integrates coupled phase-amplitude ODEs (Acebrón et al. 2005, Rev. Mod. Phys.): `dr_i/dt = (μ - r²)r + ε Σ K^r_ij r_j cos(θ_j - θ_i)`. Euler/RK4/RK45 methods, pre-allocated scratch arrays, amplitude clamping, weighted order parameter.
- **Phase-amplitude coupling (PAC)** — `modulation_index()` (Tort et al. 2010), `pac_matrix()`, `pac_gate()` in `upde/pac.py`
- **Modulation envelopes** — `extract_envelope()` (sliding-window RMS), `envelope_modulation_depth()`, `EnvelopeState` in `upde/envelope.py`
- `AmplitudeSpec` dataclass in binding types; `amplitude:` YAML block activates Stuart-Landau mode
- `CouplingState.knm_r` — amplitude coupling matrix alongside phase coupling
- `CouplingBuilder.build_with_amplitude()` for joint phase + amplitude coupling
- `ImprintModel.modulate_mu()` — imprint-dependent bifurcation parameter modulation
- `LayerState.mean_amplitude`, `LayerState.amplitude_spread` (backward-compatible defaults)
- `UPDEState.mean_amplitude`, `UPDEState.pac_max`, `UPDEState.subcritical_fraction`
- CLI `run` command branches on amplitude mode: builds `StuartLandauEngine`, computes PAC, tracks envelope metrics
- Audit logger records `amplitude_mode` in header; replay reconstructs correct engine type
- ~80 new tests across 7 new test files (total ~860)

### Changed

- `AuditLogger.log_header()` accepts `amplitude_mode` parameter
- `ReplayEngine.build_engine()` returns `StuartLandauEngine` when header has `amplitude_mode=True`
- Binding validator rejects `amplitude.epsilon < 0` and non-finite `amplitude.mu`

## [0.3.0] - 2026-03-04

### Added

- **Petri net regime FSM** — `PetriNet`, `Place`, `Arc`, `Transition`, `Marking`, `Guard` for multi-phase protocol sequencing
- **`PetriNetAdapter`** — maps Petri net markings to `Regime` values with highest-severity-wins priority
- **`ProtocolNetSpec`** — binding spec `protocol_net:` key for declarative protocol sequencing in YAML
- **Event-driven transitions** — `EventBus` + `RegimeEvent` pub/sub system with bounded history
- **`RegimeManager.force_transition()`** — bypasses cooldown and hysteresis hold
- **`RegimeManager.transition_history`** — deque of (step, prev, new) tuples (maxlen=100)
- **`hysteresis_hold_steps`** — consecutive-step requirement for soft downward transitions
- **`BoundaryObserver` event wiring** — posts `boundary_breach` events to EventBus
- **SNN controller bridge** (`SNNControllerBridge`) — pure-numpy LIF rate model + Nengo/Lava optional backends
- `nengo` and `lava` optional dependency groups
- Event kinds: `boundary_breach`, `r_threshold`, `regime_transition`, `manual`, `petri_transition`
- CLI wires EventBus, BoundaryObserver events, and Petri net when binding spec declares `protocol_net:`
- Rust `RegimeManager.force_transition()` and `transition_log` for FFI contract parity
- ~90 new tests across 5 new test files

### Changed

- `SupervisorPolicy` accepts optional `petri_adapter` argument; when present, `decide()` delegates regime to Petri net
- `BoundaryObserver.observe()` accepts optional `step` kwarg for event attribution
- `RegimeManager` constructor accepts `event_bus` and `hysteresis_hold_steps` params
- `adapters/__init__.py` exports `SNNControllerBridge`

## [0.2.0] - 2026-03-04

### Added

- **Compound policy DSL** — `CompoundCondition` with AND/OR logic over multiple `PolicyCondition` triggers
- **Action chains** — `PolicyRule.actions` accepts a list of `PolicyAction` items fired on a single trigger
- **Rule rate-limiting** — per-rule `cooldown_s` and `max_fires` fields
- **`stability_proxy` metric** in policy conditions (global mean R)
- **OpenTelemetry export** — `OTelExporter` with span instrumentation, gauge metrics (`spo.r_global`, `spo.stability_proxy`), step counter; no-op fallback when `opentelemetry-api` is absent
- `otel` optional dependency group (`opentelemetry-api>=1.20`, `opentelemetry-sdk>=1.20`)
- Pre-commit hook for version consistency check across pyproject.toml, CITATION.cff, Cargo.toml
- **QueueWaves** — real-time microservice cascade failure detector (`src/scpn_phase_orchestrator/apps/queuewaves/`)
  - PrometheusCollector with persistent async httpx client and ring buffers
  - PhaseComputePipeline wrapping UPDE engine for Kuramoto phase analysis
  - AnomalyDetector: retry-storm, cascade-propagation, chronic-degradation
  - WebhookAlerter with deduplication and Slack/generic webhook formats
  - FastAPI server with REST API, WebSocket streaming, Prometheus exposition
  - Single-file HTML dashboard (R timeline, phase wheel, alert table)
  - CLI subcommands: `spo queuewaves serve`, `spo queuewaves check`
  - Graceful shutdown with task cancellation and resource cleanup
  - 60 tests, coverage >90%
- 12 new domainpacks: cardiac_rhythm, circadian_biology, chemical_reactor, epidemic_sir, firefly_swarm, laser_array, manufacturing_spc (upgraded), neuroscience_eeg, pll_clock, power_grid, rotating_machinery, swarm_robotics (total: 21)
- 3 adapter bridges: FusionCoreBridge, PlasmaControlBridge, QuantumControlBridge
- RK45 adaptive integration with configurable tolerance and max-step limits
- PolicyEngine: declarative YAML rules with regime/metric triggers
- ActionProjector wiring for supervisor → actuation pipeline
- BindingLoadError exception and validator guards for malformed specs
- Phase-synchronization control theory docs (scope-of-competence, hardware pipeline)
- Synchronization manifold header image
- Queuewaves retry-storm demo notebook
- **Deterministic replay** from audit.jsonl with chained phase-vector verification
  - `AuditLogger.log_header()` writes engine config (n, dt, method, seed)
  - `AuditLogger.log_step()` now records full UPDE inputs (phases, omegas, knm, alpha, zeta, psi)
  - `ReplayEngine.verify_determinism_chained()` replays logged steps and compares output phases
  - `ReplayEngine.build_engine()` reconstructs UPDEEngine from header record
  - CLI `spo replay --verify` validates reproducibility within tolerance (atol=1e-6)
  - CLI `spo run --seed` makes initial RNG seed configurable and logged

### Fixed

- **[P0]** Rust `ImprintModel.modulate_lag` added row-wise `m[i]` offset; now uses `m[i] - m[j]` preserving antisymmetry
- **[P0]** CLI `run` silently dropped `K` and `Psi` supervisor actions; now applies coupling scaling and target phase
- **[P0]** CLI `stability_proxy` used only first layer R; now uses mean R across all layers
- **[P1]** Rust `PhysicalExtractor` quality hardcoded to 1.0; now computes envelope coefficient-of-variation
- **[P1]** `compute_plv` silently truncated mismatched arrays; now raises `ValueError` / `SpoError::InvalidDimension`
- All domainpack binding specs use semver (`0.1.0` not `0.1`)
- 9 mypy type errors in bridges and CLI resolved
- CI: queuewaves optional deps installed for test coverage
- Ruff format violations in audit/logger.py and tests/test_audit_replay.py

### Changed

- `PolicyRule` now uses `actions: list[PolicyAction]` instead of top-level knob/scope/value/ttl_s fields
- `PolicyRule.condition` accepts both `PolicyCondition` and `CompoundCondition`
- `PolicyEngine` tracks per-rule fire counts and cooldown timestamps
- `OTelAdapter` stub replaced by production `OTelExporter` class
- FFI `PyUPDEStepper` accepts `n_substeps` parameter
- FFI `PyCoherenceMonitor` exposes `detect_phase_lock` with full CLA matrix
- `ImprintState` and `CouplingState` are frozen dataclasses
- CI: PRs to `develop` branch trigger CI; FFI test job runs full test suite
- CI: install `.[dev,queuewaves]` in all jobs for full coverage
- Previous sprints 1–9 entries moved to [0.1.1] section below
- Repo hygiene: PEP 639 classifier fix, pre-commit pins, stub exports

## [0.1.1] - 2026-03-02

### Fixed

- **[P0]** `verify_determinism` compared global R against mean-of-layer-R (different quantities); now compares against `stability_proxy`
- **[P0]** `UPDEEngine.step()` accepted shape-mismatched arrays silently; now validates all input shapes
- **[P0]** Rust `UPDEStepper` validated `n_substeps` but ignored it (always 1); now loops `n_substeps` iterations at `sub_dt = dt / n_substeps`
- **[P0]** Rust `LagModel` propagated NaN distances into alpha matrix; now rejects NaN/Inf distances with `IntegrationDiverged`
- **[P0]** `RegimeManager.transition()` took redundant `current` param that could diverge from `self._current`; removed for Python↔Rust parity
- **[P0]** Merge duplicate `validate_binding_spec` — canonical version now in `validator.py` with all checks merged
- **[P1]** `InformationalExtractor` theta always cancelled to ~0; use median freq × total time
- **[P1]** Python `ImprintModel.modulate_lag` added row-wise offset; use `m[i] - m[j]` (antisymmetric)
- **[P1]** CLI scaffold generated `version: '0.1'` (fails validator); now `'0.1.0'`
- **[P1]** CLI `run` did not compute `cross_layer_alignment`; now uses `compute_plv` between layer pairs
- **[P1]** CLI zeta had no TTL expiry; now decrements TTL counter and resets to 0 on expiry
- **[P1]** Rust UPDE stepper did not check omegas/knm for NaN/Inf; now rejects with `IntegrationDiverged`
- **[P1]** `PhysicalExtractor._snr_estimate` always returned ~1.0; replaced with envelope-CV metric
- **[P1]** `UPDEEngine.compute_order_parameter` reimplemented inline; now delegates to canonical implementation
- **[P1]** `AuditLogger` file writes never flushed; switched to line-buffered I/O
- **[P1]** CLI `run` command ignored spec drivers, boundaries, and actuators; rewritten to wire supervisor

### Changed

- `BoundaryState.soft_warnings` renamed to `soft_violations` for Rust parity
- 7 source files import `TWO_PI`/`HAS_RUST` from `_compat` instead of redefining
- `RegimeManager.transition()` takes only `proposed` param (breaking: callers updated)
- Rust `event_phase` uses median freq × total time (matches Python fix)
- ROADMAP domainpack names match actual directory names
- CONTRIBUTING import path corrected: `extractors/` → `oscillators/`
- `spo-types`: `serde_json` moved to `[dev-dependencies]`
- CHANGELOG heading `Improved` → `Changed` per Keep a Changelog spec
- CLI import path uses canonical `from scpn_phase_orchestrator.binding import validate_binding_spec`
- `oscillators/__init__.py` exports `PhysicalExtractor`, `InformationalExtractor`, `SymbolicExtractor`, `PhaseQualityScorer`
- `adapters/__init__.py` exports `SCPNControlBridge`
- Rust: `Debug` impl for `UPDEStepper`, `#[derive(Debug)]` for `ImprintModel`, `LagModel`
- Rust: doc comments on all public types
- Migrate remaining probe-imports to `importlib.util.find_spec` with lazy imports
- Remove dead `_HAS_RUST` assignments from regimes.py, coherence.py
- Add `#[must_use]` to all pure public Rust functions
- Add crate-level `//!` doc comments to all crates
- Refactor physical.rs Pass 1 to `iter_mut().zip()` iterators
- Make `LockSignature`, `LayerState`, `UPDEState` frozen dataclasses
- CI: add pip + cargo caching, replace manual `cargo-audit` install with `rustsec/audit-check` action
- Add `Documentation` and `Changelog` URLs to `project.urls` (PyPI sidebar)

### Added

- `src/scpn_phase_orchestrator/_compat.py` — shared `TWO_PI`, `HAS_RUST` constants
- `src/scpn_phase_orchestrator/py.typed` — PEP 561 marker
- `tools/check_version_sync.py` — version sync check across pyproject.toml, CITATION.cff, Cargo.toml
- `.dockerignore` — excludes .git, target, caches, site
- CI lint job runs `check_version_sync.py`
- Publish preflight runs Rust clippy + tests
- PyPI and docs badges in README
- `repository` field in Cargo workspace metadata
- 4 new validator tests, 5 hypothesis property tests, 4 coupling lags tests, 4 coupling templates tests, 2 physical extractor tests

## [0.1.0] - 2026-03-01

### Added

- UPDE engine with Euler/RK4 integration and pre-allocated scratch arrays
- 3-channel oscillator model: Physical, Informational, Symbolic (P/I/S)
- Coupling matrix (Knm) management with exponential decay and template switching
- Supervisor with RegimeManager (NOMINAL/DEGRADED/CRITICAL/RECOVERY) and policy actions
- Actuation mapper and ActionProjector for domain-agnostic output binding
- Memory imprint model with exponential decay and coupling/lag modulation
- Boundary observer (soft/hard violations) feeding regime decisions
- CLI entry point (`spo`) with init, run, replay, status commands
- 4 domainpacks: minimal_domain, queuewaves, geometry_walk, bio_stub
- Binding spec JSON Schema for domainpack configuration and validation
- PhaseExtractor base class for domain-specific signal intake
- Assumption registry (`docs/ASSUMPTIONS.md`) documenting all empirical thresholds
- Bibliography (`docs/references.bib`) with 10 canonical references (Kuramoto, Acebrón, Sakaguchi, Strogatz, Dörfler, Lachaux, Gabor, Pikovsky, Hairer, Courant)
- Citation metadata (`CITATION.cff`) for Zenodo and academic databases
- Coverage regression guard (`tools/coverage_guard.py`) enforcing 90% line coverage
- Module linkage guard (`tools/check_test_module_linkage.py`) requiring test files for all source modules
- Rust kernel (`spo-kernel/`) with PyO3 bindings for UPDEEngine, RegimeManager, CoherenceMonitor

[Unreleased]: https://github.com/anulum/scpn-phase-orchestrator/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/anulum/scpn-phase-orchestrator/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/anulum/scpn-phase-orchestrator/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/anulum/scpn-phase-orchestrator/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/anulum/scpn-phase-orchestrator/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/anulum/scpn-phase-orchestrator/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/anulum/scpn-phase-orchestrator/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/anulum/scpn-phase-orchestrator/releases/tag/v0.1.0
