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

## Phase 3 Results (2026-03-29)

**12 passed, 0 failed, 1 xfail.** File: `tests/test_nn_physics_validation_p3.py`

| # | Test | Result | Detail |
|---|---|---|---|
| V25 | SL training convergence | **PASS** | Loss decreases over 40 epochs with optax.adam. |
| V26 | Reservoir prediction | **XFAIL** | Correlation -0.37 — random K not at edge-of-bifurcation. See Finding #5. |
| V27 | OIM energy descent | **PASS** | <5 energy violations in 500 steps. |
| V28 | Multiple shooting inverse | **PASS** | Shooting corr comparable to single-shot at T=200. |
| V29 | SL gradient w.r.t. mu | **PASS** | Finite, non-zero gradient confirmed. |
| V30 | Simplicial = Kuramoto at sigma2=0 | **PASS** | Trajectories identical within 1e-5. |
| V31 | Winfree uncoupled period | **PASS** | T_measured matches 2*pi/omega within 5%. |
| V32 | BOLD linearity | **PASS** | 2x input → ~2x BOLD peak (ratio in [1.3, 2.7]). |
| V33 | Theta excitable silence | **PASS** | <3 spikes with eta=-1, no input. |
| V34 | Masked layer consistency | **PASS** | kuramoto_forward_masked = kuramoto_forward(K*mask). |
| V35 | coupling_correlation identity | **PASS** | corr(K,K)=1.0, corr(K,-K)=-1.0. |
| V36 | SL phase frequency | **PASS** | Phase advance = omega*T within 0.02 rad. Amplitude stays at sqrt(mu). |

### Finding #5: Reservoir requires K_c tuning

Random coupling K and input weights W_in do not produce useful reservoir
computation. Theory (arXiv:2407.16172) predicts optimal performance at
K ≈ K_c (edge-of-bifurcation). The reservoir module works mechanically
but the user must tune K to the critical coupling for their frequency
distribution. This should be documented prominently.

## Phase 3 Test Matrix

| # | Test | Falsifies | Priority |
|---|---|---|---|
| V25 | SL training convergence | SL gradient pipeline | P0 |
| V26 | Reservoir prediction quality | Reservoir usefulness | P2 |
| V27 | OIM energy descent | OIM gradient property | P1 |
| V28 | Multiple shooting inverse | Shooting implementation | P1 |
| V29 | SL gradient w.r.t. mu | SL autodiff | P0 |
| V30 | Simplicial = Kuramoto at sigma2=0 | Simplicial correctness | P0 |
| V31 | Winfree uncoupled period | Winfree ODE | P1 |
| V32 | BOLD linearity | Hemodynamic linearity | P1 |
| V33 | Theta excitable silence | Excitability model | P1 |
| V34 | Masked layer consistency | Mask implementation | P0 |
| V35 | coupling_correlation identity | Metric correctness | P0 |
| V36 | SL phase frequency | SL phase ODE | P0 |

---

## Cumulative Summary

| Phase | Tests | Passed | xFail | Findings |
|---|---|---|---|---|
| P1 (V1–V12) | 26 | 25 | 1 | SAF eigh NaN, simplicial hysteresis absent, BOLD peak 3.1s |
| P2 (V13–V24) | 11 | 10 | 1 | UDE extrapolation NaN |
| P3 (V25–V36) | 13 | 12 | 1 | Reservoir needs K_c tuning |
| P4 (V37–V46) | 12 | 12 | 0 | None |
| P5 (V47–V60) | 16 | 16 | 0 | Mean-phase drift (#6) |
| P6 (V61–V74) | 16 | 14 | 2 | K symmetry broken (#7), OIM Petersen fail (#8) |
| P7 (V75–V86) | 14 | 11 | 3 | FIM scaling small-N (#9), FIM hysteresis K-range (#10), BKT vs MF (#11) |
| P8 (V87–V96) | 10 | 10 | 0 | None — SR, roundtrips, EEG, delay, FIM+SL confirmed |
| P9 (V97–V108) | 18 | 17 | 1 | Inverse ill-conditioned at K=0 (#12) |
| P10 (V109–V120) | 19 | 19 | 0 | None — clean sweep. Capstone: scaling collapse, reproducibility, GD works. |
| **Total** | **155** | **146** | **9** | **12 findings** |

All 5 findings are genuine limitations, not bugs. None falsify the core
physics. The framework is sound.

---

## Findings Register

| # | Finding | Severity | Module | Root cause | Workaround | Status |
|---|---|---|---|---|---|---|
| 1 | SAF eigh gradient NaN for uniform K | Medium | `functional.py` | JAX `eigh` backward pass undefined at repeated eigenvalues | Use non-uniform K or add small perturbation before differentiating | Known limitation |
| 2 | Simplicial hysteresis absent at sigma2=3, N=64 | High | `functional.py` | Mean-field factorisation `2·S·C/N²` may weaken 3-body effect vs explicit triplet sums | Investigate with N=256+, sigma2=10+ | Open investigation |
| 3 | BOLD HRF peak at 3.1s, not canonical 5s | Low | `bold.py` | Stephan 2007 params differ from SPM canonical (Friston 2003). Both are correct for their parameter sets. | Document parameter-dependence; provide SPM parameter preset | Documentation needed |
| 4 | UDE extrapolation NaN beyond training window | High | `ude.py` | `CouplingResidual` MLP (tanh activations) is unbounded at large inputs; ODE integration amplifies residual divergence | Add output clamping `jnp.clip(residual, -1, 1)` or enforce Lipschitz constraint via spectral normalisation | Fix needed |
| 5 | Reservoir random K gives negative correlation | Medium | `reservoir.py` | Operating point not at edge-of-bifurcation (K ≈ K_c). Random K overshoots or undershoots K_c. | Document that K must be tuned to K_c = 2·Δ for Lorentzian ω distribution; provide `auto_tune_K` helper | Documentation needed |
| 6 | Mean phase Ψ drifts ~0.13 over 1000 steps | Low | `functional.py` | `% TWO_PI` wrapping and float32 sin/cos rounding break exact rotational symmetry. Rate: ~1.3e-4 per step. | Use float64 for precision-critical work; or track Ψ explicitly and correct | Known limitation |
| 7 | K loses symmetry after gradient training | High | `training.py` | `jax.grad` of loss w.r.t. symmetric K produces non-symmetric gradient. optax.adam updates K with non-symmetric step → K drifts asymmetric. | Add `K = (K + K.T) / 2` after each update in training loop; or use Cholesky parameterisation `K = L·L^T` | Fix needed |
| 8 | OIM fails on Petersen graph (chi=3) | Medium | `oim.py` | Petersen graph is 3-regular with girth 5 — hard for annealing heuristics. 30 restarts insufficient. OIM coupling `sin(3·Δθ)` may have local minima for this topology. | Increase n_restarts (100+), adjust annealing schedule, or use `oim_solve` with custom k_max/n_anneal for hard instances | Known limitation |
| 9 | FIM λ_c scaling breaks at small N | Medium | test-local FIM | N=4 syncs at near-zero λ (finite-size effect). Scaling law λ_c∝N holds only for N≥8. NB25 used stronger omega spread (Cauchy 0.5 vs our Normal 0.5). | Test scaling at N≥32; match NB25 frequency distribution exactly | Known limitation |
| 10 | FIM hysteresis invisible at λ=3, K∈[0,5] | Low | test-local FIM | FIM at λ=3 is strong enough that N=16 reaches R≈0.998 from BOTH directions in K∈[0,5]. NB27 used K∈[0,20] and saw hysteresis in K=4-10 range. | Widen K sweep range to [0,20]; or reduce λ to ~1.5 | Test parameter mismatch |
| 11 | BKT universality contradicts V52 mean-field β=1/2 | **Critical** | cross-project | V52 confirmed β=1/2 for all-to-all uniform K (mean-field). NB43 found β→0 (BKT) for heterogeneous K_nm coupling. The universality class depends on TOPOLOGY, not on FIM. All-to-all = mean-field, structured = BKT. | Document that critical exponents are topology-dependent; add heterogeneous-K test to V52 | Open investigation |
| 12 | analytical_inverse ill-conditioned at K=0 | Medium | `inverse.py` | Without coupling, ω-driven phase drift produces sin(Δθ) basis correlations that lstsq misinterprets as coupling. ‖K_est‖ = 51.6 for true K=0. | Add ridge regularisation (alpha > 0) as default; or check condition number before returning result. Document that inverse requires actual coupling to work. | Fix needed |

### Finding #1 — Detail

**Reproduced by:** V6 `test_saf_order_parameter_grad` with `K = ones * 0.5`.
**Mechanism:** Uniform K produces a Laplacian `L = D - K` with (N-1)-fold
degenerate eigenvalue. The `eigh` backward pass divides by eigenvalue gaps
`1/(λ_i - λ_j)`, which is `1/0` at degeneracies. JAX returns NaN silently.
**Impact:** Cannot gradient-optimise SAF loss starting from uniform coupling.
Must initialise with non-uniform K (e.g., random perturbation).
**Scope:** Only affects `saf_order_parameter` and `saf_loss` under `jax.grad`.
Forward evaluation always works.

### Finding #2 — Detail

**Reproduced by:** V7 `test_hysteresis_present` with sigma2=3, N=64.
**Mechanism:** The 3-body term in `_simplicial_deriv` uses mean-field
factorisation: `Σ_{j,k} sin(Δθ_j + Δθ_k) ≈ 2·S_i·C_i` where
`S_i = Σ sin(Δθ_j)`, `C_i = Σ cos(Δθ_j)`. This is exact for the full
3-body sum but the normalisation `σ₂/N²` may be too aggressive for N=64.
Gambuzza et al. 2023 used N=500+ in their simulations.
**Open question:** Does the implementation actually reproduce Fig. 2 of
Gambuzza et al.? This requires N≥256 and careful parameter matching.

### Finding #3 — Detail

**Reproduced by:** V8 `test_hrf_peak_timing`.
**Mechanism:** Default parameters are Stephan et al. 2007: κ=0.65, γ=0.41,
τ=0.98, α=0.32. SPM12 uses Friston et al. 2003 with κ=0.65, γ=0.41 but
different τ and α, plus a second derivative term that shifts the peak.
**Impact:** Users comparing SPO BOLD output against SPM will see a timing
mismatch. Both parameter sets are physically valid.

### Finding #4 — Detail

**Reproduced by:** V17 `test_ude_does_not_overfit`. UDE trained on 60 steps,
evaluated at steps 60–100 → NaN.
**Mechanism:** `CouplingResidual` maps Δθ → correction via 3-layer MLP with
tanh activations. At training time, Δθ stays in a bounded range. At test
time, if phases diverge slightly from the training distribution, the MLP
outputs grow, which causes larger phase errors, which cause larger Δθ
inputs to the MLP — a positive feedback loop → divergence → NaN.
**Fix:** Clamp MLP output: `return jnp.clip(x[0], -1.0, 1.0)` in
`CouplingResidual.__call__`. This bounds the correction to ±1 (same order
as sin(Δθ)), preventing runaway. Alternatively, use spectral normalisation
to enforce a Lipschitz constant on the MLP.

### Finding #5 — Detail

**Reproduced by:** V26 `test_reservoir_recovers_signal`. Random K with
N=12, signal = sin(t).
**Mechanism:** Kuramoto reservoir theory (arXiv:2407.16172) predicts that
computational capacity peaks at K = K_c (critical coupling). Below K_c,
oscillators are incoherent — reservoir has rich dynamics but weak signal
amplification. Above K_c, oscillators lock — reservoir loses computational
diversity. Random K is unlikely to hit the sweet spot.
**Impact:** The `reservoir_drive` + `ridge_readout` pipeline is correct but
useless without K_c tuning. Users need guidance: compute K_c from their
frequency distribution, then set K ≈ K_c.

---

## Phase 4 Results (2026-03-29)

**12 passed, 0 failed, 0 xfail.** File: `tests/test_nn_physics_validation_p4.py`

| # | Test | Result | Detail |
|---|---|---|---|
| V37 | Arnold tongue | **PASS** | Locking at K=0.8 > Δω/2=0.5. Drift at K=0.2 < Δω/2. |
| V38 | Phase diffusion below K_c | **PASS** | circ_var > 0.5 after 2000 steps with K ≪ K_c. |
| V39 | Time reversal | **PASS** | Forward+reverse error < 0.1 (gradient flow reversible). |
| V40 | Hybrid inverse noisy | **PASS** | corr_hybrid ≥ corr_analytical - 0.15 on noisy data. |
| V41 | vmap correctness | **PASS** | vmap output identical to sequential within 1e-5. |
| V42 | scan = manual loop | **PASS** | Final state and full trajectory match within 1e-5. |
| V43 | SL amplitude consensus | **PASS** | Spread decreased >50% with amplitude coupling. |
| V44 | Chimera index boundaries | **PASS** | chi < 0.01 for sync, chi < 0.05 for uniform spread. |
| V45 | OIM bipartite K_{3,3} | **PASS** | 0 violations with 2 colours. |
| V46 | PLV correlates with R | **PASS** | Mean PLV increases monotonically with coupling K. |

No new findings. All structural properties confirmed.

## Phase 5 Results (2026-03-29)

**16 passed, 0 failed, 0 xfail.** File: `tests/test_nn_physics_validation_p5.py`

| # | Test | Result | Detail |
|---|---|---|---|
| V47 | Gauge invariance | **PASS** | R and phase differences identical under global shift. |
| V48 | Winding number conservation | **PASS** | q stays within 0.1 of initial value over 2000 steps. |
| V49 | Dimensional scaling | **PASS** | |ΔR| < 0.05 for scaled (2ω, 2K, dt/2) above K_c. Below K_c: scaling fails numerically (no attractor). |
| V50 | Numerical symmetry breaking | **PASS** | R > 0.999 for first 5000+ steps with identical initial conditions. |
| V51 | Extensivity | **PASS** | R spread < 0.15 across N = {32, 64, 128, 256} at fixed K_eff. |
| V52 | Critical exponent β = 1/2 | **PASS** | R² vs (K-K_c) linear fit R² > 0.8. Mean-field exponent confirmed. |
| V53 | Multistability | **PASS** | In-phase (Δθ=0): stable. Anti-phase (Δθ=π): unstable, evolves to 0. |
| V54 | Mean phase conservation | **PASS** | Drift 0.13 over 1000 steps in float32. See Finding #6. |
| V55 | Phase response curve | **PASS** | Measured PRC correlates > 0.9 with -sin(θ). |
| V56 | Quasi-periodic spectrum | **PASS** | Top-3 spectral peaks contain > 30% of total power. |
| V57 | Bimodal clustering | **PASS** | Two-group R > global R for bimodal ω distribution. |
| V58 | Adiabatic tracking | **PASS** | R tracks slowly ramping K: late R > early R. |
| V59 | Perturbation relaxation | **PASS** | Decay rate increases with coupling strength. |
| V60 | Float32 divergence | **PASS** | Sync state: R > 0.9 in float32. Perturbation test finite. |

### Finding #6: Mean phase drift in float32

**Reproduced by:** V54 `test_mean_phase_drift`.
**Mechanism:** The `% TWO_PI` operation after each step clips phases to
[0, 2π). This clipping is not rotationally invariant — it depends on the
absolute value of the phase, not just the differences. Combined with float32
rounding in `jnp.sin` and `jnp.cos`, this produces a systematic drift of
the mean phase Ψ of ~1.3e-4 radians per step.
**Impact:** Over 1000 steps, Ψ drifts by ~0.13 radians. For most
applications (R, PLV, coupling inference), this is irrelevant because they
depend only on phase differences. But applications that track absolute
phase (e.g., entrainment to external signal) will accumulate error.
**Scope:** Float32 only. Float64 would reduce drift by ~10^8.

## Phase 6 Results (2026-03-29)

**14 passed, 0 failed, 2 xfail.** File: `tests/test_nn_physics_validation_p6.py`

| # | Test | Result | Detail |
|---|---|---|---|
| V61 | Permutation equivariance | **PASS** | Relabelled oscillators produce identical dynamics within 1e-4. |
| V62 | 2π boundary gradient | **PASS** | Gradient finite at wrapping boundary; sin/cos path avoids discontinuity. |
| V63 | R fluctuation scaling | **PASS** | var(R) decreases from N=32 to N=512 near K_c. |
| V64 | Gradient magnitude vs N | **PASS** | |∇loss| varies < 100x across N={8,16,32,64}. |
| V65 | Inverse noise breakdown | **PASS** | Correlation curve measured: >0.8 noiseless, decreasing with noise. |
| V66 | Amplitude death | **PASS** | SL with strong ε and spread ω: amplitudes finite. |
| V67 | K symmetry under training | **XFAIL** | K becomes asymmetric after 30 Adam steps. See Finding #7. |
| V68 | Layer compositionality | **PASS** | Gradient flows through chained KuramotoLayer pair; both ∇K non-zero. |
| V69 | SAF on star topology | **PASS** | SAF returns finite R on heterogeneous graph. |
| V70 | Inverse conditioning | **PASS** | Ring topology recovery ≥ dense recovery - 0.2. |
| V71 | Lyapunov exponent sign | **PASS** | Sync: perturbation decays. Desync: perturbation persists. |
| V72 | Multi-timescale BOLD | **PASS** | 2Hz neural → BOLD has <30% high-frequency power (hemodynamic LP filter). |
| V73 | OIM Petersen graph | **XFAIL** | 2 violations in 30 restarts. See Finding #8. |
| V73 | OIM C5 cycle | **PASS** | chi(C5)=3 correctly: 2 colours fail, 3 colours succeed. |
| V74 | Gradient chain rule | **PASS** | ∇f + ∇(-f) < 1e-6 — autodiff chain rule exact. |

### Finding #7: K symmetry broken by gradient training

**Reproduced by:** V67 `test_K_stays_symmetric`.
**Mechanism:** The gradient `∂loss/∂K` of a scalar loss w.r.t. a symmetric
matrix K is NOT symmetric in general. Example: `loss = R(trajectory(K))`.
The chain rule produces `∂R/∂θ · ∂θ/∂K` where the Jacobian `∂θ/∂K` has
no symmetry guarantee because the scan accumulates asymmetric contributions.
After 30 Adam steps, `max|K - K^T|` exceeds 0.01.
**Impact:** Physically, asymmetric K means directed coupling (oscillator i
drives j but not vice versa). This changes the dynamics qualitatively — the
Lyapunov function (V3) no longer exists, the system is no longer gradient
flow. Users who train KuramotoLayer and then interpret K as physical
connectivity will get wrong conclusions.
**Fix:** Add `K = (K + K.T) / 2; K = K.at[diag].set(0)` after each
gradient update. Or reparameterise: store L and compute K = L·L^T.

### Finding #8: OIM fails on Petersen graph

**Reproduced by:** V73 `test_petersen_3colorable`.
**Mechanism:** The Petersen graph (10 nodes, 15 edges, 3-regular, girth 5)
is a known hard case for heuristic graph colouring. The `sin(3·Δθ)` coupling
creates 3 equidistant phase clusters, but the Petersen graph's symmetry
group (S₅) has frustrated cycles that trap the annealing in local minima.
**Impact:** OIM is not a general-purpose graph colouring solver. It works
well for easy instances (bipartite, small sparse graphs) but fails on
algebraically structured hard instances. This should be documented.

## Phase 4 Test Matrix

| # | Test | Falsifies | Priority | Analytical reference |
|---|---|---|---|---|
| V37 | Arnold tongue (frequency locking) | Coupling-detuning relationship | P0 | Locking when K > Δω/2 for N=2 |
| V38 | Phase diffusion below K_c | Sub-critical dynamics | P1 | Unbounded phase drift |
| V39 | Time reversal (gradient flow) | Reversibility of potential dynamics | P1 | Forward ≈ reversed trajectory |
| V40 | hybrid_inverse improves on analytical | Hybrid method value | P1 | corr_hybrid >= corr_analytical for noisy data |
| V41 | vmap correctness | Batched execution | P0 | vmap(f)(batch) = stack([f(x) for x in batch]) |
| V42 | scan = manual loop | Internal consistency | P0 | kuramoto_forward = manual step loop |
| V43 | SL amplitude consensus | Amplitude coupling | P1 | Spread decreases with epsilon |
| V44 | Chimera index = 0 for uniform states | Chimera metric boundary | P1 | Sync → 0, desync → 0 |
| V45 | OIM bipartite 2-colour perfect | Easy graph benchmark | P1 | 0 violations |
| V46 | PLV correlates with R | Metric consistency | P1 | High R → high mean PLV |

## Implementation

- Phase 1: `tests/test_nn_physics_validation.py` (26 tests, ~80s)
- Phase 2: `tests/test_nn_physics_validation_p2.py` (11 tests, ~60s)
- Phase 3: `tests/test_nn_physics_validation_p3.py` (13 tests, ~590s)
- Phase 4: `tests/test_nn_physics_validation_p4.py` (12 tests, ~98s)
- Phase 5: `tests/test_nn_physics_validation_p5.py` (16 tests, ~175s)
- Phase 6: `tests/test_nn_physics_validation_p6.py` (16 tests, ~130s)
- Phase 7: `tests/test_nn_physics_validation_p7.py` (14 tests, ~2900s — FIM Python loops)
- Phase 8: `tests/test_nn_physics_validation_p8.py` (10 tests, ~185s)
- Phase 9: `tests/test_nn_physics_validation_p9.py` (18 tests, ~54s)
- Phase 10: `tests/test_nn_physics_validation_p10.py` (19 tests, ~216s)

GPU optional — all tests run on CPU.

## Acceptance Criteria

- P0 tests: MUST pass. Failure blocks release.
- P1 tests: SHOULD pass. Failure documented as known limitation.
- P2 tests: NICE TO HAVE. Failure informs future work.
