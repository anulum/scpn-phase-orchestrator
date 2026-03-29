# nn/ Module — Physics Validation Plan

Validation tests that confirm or falsify the physical correctness of the
JAX nn/ module. Each test has a known analytical result. A failure means
the implementation is wrong, not that the test is too strict.

Existing tests (`test_physics_benchmarks.py`, `test_ott_antonsen.py`,
`test_bifurcation.py`) validate the NumPy engine. These tests validate
the JAX nn/ module independently, then cross-validate both backends.

---

## Phase 1 Results (2026-03-29)

**25 passed, 0 failed, 1 xfail.** File: `tests/test_nn_physics_validation.py`

| # | Test | Result | Detail |
|---|---|---|---|
| V1 | RK4 convergence order | **PASS** | Error ratio 15.2 (expected ~16 for O(dt^4)). Requires float64. |
| V2 | N=2 analytical solution | **PASS** | K_eff = 2·K_ij for phase-difference equation. All 3 K values match. |
| V3 | Lyapunov monotonicity | **PASS** | 0 violations in 500 steps. Energy never increases. |
| V4 | R(K) transition vs Ott-Antonsen | **PASS** | Max |R_sim - R_OA| = 0.175 for N=512. Within finite-N tolerance. |
| V5 | Stuart-Landau Hopf bifurcation | **PASS** | 6/6 parametric cases: r → sqrt(mu) for mu>0, r → 0 for mu<0. |
| V6 | Gradient vs finite difference | **PASS** | 4/4 functions: order_parameter, kuramoto_forward, coloring_energy pass. SAF passes with non-degenerate K. |
| V7 | Simplicial hysteresis | **XFAIL** | No hysteresis detected at sigma2=3, N=64. See Finding #2. |
| V8 | BOLD HRF impulse response | **PASS** | Peak at 3.1s. See Finding #3. |
| V9 | analytical_inverse accuracy | **PASS** | N=4: corr>0.99, N=8: corr>0.95, N=16: corr>0.90. |
| V10 | Gradient stability vs n_steps | **PASS** | Finite gradients confirmed to n_steps=1000. |
| V11 | Winfree → Kuramoto weak limit | **PASS** | |R_winfree - R_kuramoto| < 0.15 at K=0.01. |
| V12 | OIM impossible colouring | **PASS** | K4 with 3 colours: >0 violations. K4 with 4 colours: 0 violations. |

### Finding #1: SAF eigh gradient degeneracy

`saf_order_parameter` gradient is NaN when K is uniform (all entries equal)
because `jnp.linalg.eigh` backward pass is undefined at repeated eigenvalues.
Known JAX limitation. Workaround: use non-uniform K, or add small random
perturbation before differentiating. Does not affect forward evaluation.

### Finding #2: Simplicial hysteresis absent at tested parameters

At sigma2=3.0, N=64, no hysteresis loop was detected in the R(K) sweep.
Possible reasons: (a) sigma2 too small for this N, (b) N too small for
the collective effect, (c) the 3-body implementation uses mean-field
factorisation S·C instead of explicit triplet sums, which may weaken the
effect. Needs investigation with larger N (256+) and stronger sigma2.

### Finding #3: BOLD HRF peak timing

Balloon-Windkessel with Stephan et al. 2007 parameters produces HRF peak
at ~3.1s, not the canonical ~5s from SPM (which uses Friston et al. 2003
parameters with different kappa/gamma). This is a correct implementation
of Stephan 2007, not a bug. Document the parameter-dependence of peak
timing in the BOLD API reference.

---

## Phase 1 Test Matrix

| # | Test | Falsifies | Priority | Analytical reference |
|---|---|---|---|---|
| V1 | RK4 convergence order | Integrator | P0 | dt^4 error scaling |
| V2 | N=2 analytical solution | Kuramoto ODE | P0 | 2·arctan(c·exp(-K_eff·t)) |
| V3 | Lyapunov monotonicity | Energy conservation | P0 | V = -Sum K cos(Delta_theta) |
| V4 | R(K) transition vs Ott-Antonsen | Entire Kuramoto physics | P0 | R = sqrt(1 - K_c/K) |
| V5 | Stuart-Landau Hopf bifurcation | Amplitude dynamics | P0 | r → sqrt(mu) |
| V6 | Gradient vs finite difference | Entire training pipeline | P0 | Numerical derivative |
| V7 | Simplicial hysteresis | "Explosive sync" claim | P1 | First-order transition |
| V8 | BOLD HRF impulse response | Hemodynamic model | P1 | Stephan 2007 |
| V9 | analytical_inverse accuracy | ">0.95 correlation" claim | P1 | Ground truth coupling |
| V10 | Gradient stability vs n_steps | Practical training limit | P1 | Gradient norm trend |
| V11 | Winfree → Kuramoto weak limit | Cross-model consistency | P1 | Equivalence theorem |
| V12 | OIM impossible colouring | Solver correctness | P2 | chi(K4) = 4 |

## Phase 2 Test Matrix

| # | Test | Falsifies | Priority | Analytical reference |
|---|---|---|---|---|
| V13 | Theta neuron SNIPER period | Excitability model | P1 | T ~ pi/sqrt(eta) |
| V14 | SAF accuracy boundary | SAF applicability range | P1 | Error vs K/K_c |
| V15 | Asymmetric K inverse | Inverse limitations | P1 | Correlation must drop |
| V16 | Large-N Ott-Antonsen convergence | Finite-size scaling | P1 | O(1/sqrt(N)) |
| V17 | UDE overfitting on noise | UDE usefulness | P1 | Test > train error |
| V18 | NumPy ↔ JAX engine parity | Backend consistency | P0 | Identical R within 1e-3 |
| V19 | Masked Kuramoto topology | Sparse coupling | P1 | Disconnected components desync |
| V20 | Reservoir edge-of-bifurcation | Theory prediction | P2 | Max performance at K~K_c |
| V21 | Chimera on ring (JAX) | Chimera detection | P1 | 0 < chimera_index, partial R |
| V22 | Training convergence | Training pipeline | P0 | Loss strictly decreasing |
| V23 | Eigenratio vs known graphs | Spectral metrics | P1 | Star < ring < complete |
| V24 | Phase-locking value symmetry | PLV metric | P0 | PLV symmetric, diagonal = 1 |

## Implementation

- Phase 1: `tests/test_nn_physics_validation.py` (done, 26 tests)
- Phase 2: `tests/test_nn_physics_validation_p2.py` (planned, ~24 tests)

GPU optional — all tests run on CPU. Phase 1 completes in ~80s.

## Acceptance Criteria

- P0 tests: MUST pass. Failure blocks release.
- P1 tests: SHOULD pass. Failure documented as known limitation.
- P2 tests: NICE TO HAVE. Failure informs future work.
