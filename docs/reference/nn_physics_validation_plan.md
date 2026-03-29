# nn/ Module — Physics Validation Plan

Validation tests that confirm or falsify the physical correctness of the
JAX nn/ module. Each test has a known analytical result. A failure means
the implementation is wrong, not that the test is too strict.

Existing tests (`test_physics_benchmarks.py`, `test_ott_antonsen.py`,
`test_bifurcation.py`) validate the NumPy engine. These tests validate
the JAX nn/ module independently, then cross-validate both backends.

## Test Matrix

| # | Test | Falsifies | Priority | Analytical reference |
|---|---|---|---|---|
| V1 | RK4 convergence order | Integrator | P0 | dt^4 error scaling |
| V2 | N=2 analytical solution | Kuramoto ODE | P0 | 2·arctan(c·exp(-K·t)) |
| V3 | Lyapunov monotonicity | Energy conservation | P0 | V = -Sum K cos(Delta_theta) |
| V4 | R(K) transition vs Ott-Antonsen | Entire Kuramoto physics | P0 | R = sqrt(1 - K_c/K) |
| V5 | Stuart-Landau Hopf bifurcation | Amplitude dynamics | P0 | r → sqrt(mu) |
| V6 | Gradient vs finite difference | Entire training pipeline | P0 | Numerical derivative |
| V7 | Simplicial hysteresis | "Explosive sync" claim | P1 | First-order transition |
| V8 | BOLD HRF impulse response | Hemodynamic model | P1 | Friston 2000 Fig. 2 |
| V9 | analytical_inverse accuracy | ">0.95 correlation" claim | P1 | Ground truth coupling |
| V10 | Gradient stability vs n_steps | Practical training limit | P1 | Gradient norm trend |
| V11 | Winfree → Kuramoto weak limit | Cross-model consistency | P1 | Equivalence theorem |
| V12 | OIM impossible colouring | Solver correctness | P2 | chi(K4) = 4 |
| V13 | Theta neuron SNIPER period | Excitability model | P2 | T ~ pi/sqrt(eta) |
| V14 | SAF accuracy boundary | SAF applicability range | P2 | Error vs K/K_c |
| V15 | Asymmetric K inverse | Inverse limitations | P2 | Must fail gracefully |
| V16 | Large-N Ott-Antonsen convergence | Finite-size scaling | P2 | O(1/sqrt(N)) |
| V17 | UDE overfitting on noise | UDE usefulness | P2 | Test > train error |

## Implementation

All tests in `tests/test_nn_physics_validation.py`. Require JAX
(`pytest.importorskip("jax")`). GPU optional — all tests run on CPU
within 60 seconds total.

## Acceptance Criteria

- P0 tests: MUST pass. Failure blocks release.
- P1 tests: SHOULD pass. Failure documented as known limitation.
- P2 tests: NICE TO HAVE. Failure informs future work.
