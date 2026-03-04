# Assumptions & Empirical Constants

Every threshold, default, and rate limit in SPO is listed here with its
provenance. Constants are either **derived** (from cited equation),
**calibrated** (fitted to simulation data), or **empirical** (engineering
judgement, subject to domain tuning).

See [references.bib](references.bib) for full bibliographic entries.

## Regime Thresholds

| Constant | Value | Provenance | Used in |
|----------|-------|------------|---------|
| `_R_CRITICAL` | 0.3 | Empirical. Below this, mean-field theory predicts incoherence for finite-N Kuramoto populations [acebron2005, §2.3]. | `supervisor/regimes.py:23` |
| `_R_DEGRADED` | 0.6 | Empirical. Partial synchronisation threshold; finite-N populations show bistability in this range [acebron2005, §2.3]. | `supervisor/regimes.py:24` |
| `hysteresis` | 0.05 | Empirical. Reserved for future threshold offset. | `supervisor/regimes.py:35` |
| `cooldown_steps` | 10 | Empirical. Prevents regime oscillation; 10 steps chosen to exceed typical transient duration. | `supervisor/regimes.py:35` |

## Supervisor Policy

| Constant | Value | Provenance | Used in |
|----------|-------|------------|---------|
| `_K_BUMP` | 0.05 | Empirical. Coupling increment per DEGRADED step; small enough to avoid overshoot. | `supervisor/policy.py:16` |
| `_ZETA_BUMP` | 0.1 | Empirical. Driver strength increment during CRITICAL; twice `_K_BUMP` for faster damping. | `supervisor/policy.py:17` |
| `_K_REDUCE` | −0.03 | Empirical. Coupling reduction on worst layer during CRITICAL. | `supervisor/policy.py:18` |
| `_RESTORE_FRACTION` | 0.5 | Empirical. Recovery restores at half the DEGRADED bump rate. | `supervisor/policy.py:19` |

## Quality Gating

| Constant | Value | Provenance | Used in |
|----------|-------|------------|---------|
| `min_quality` | 0.3 | Empirical. Below this, phase estimate is unreliable (SNR < −5 dB equivalent). | `oscillators/quality.py:36` |
| collapse `threshold` | 0.1 | Empirical. Quality floor for declaring oscillator collapse. | `oscillators/quality.py:28` |
| stall quality | 0.2 | Empirical. Quality assigned to repeated-state symbolic oscillators. | `oscillators/symbolic.py:83` |
| PLV lock `threshold` | 0.9 | Empirical. Consistent with Lachaux et al. (1999) convention for significant phase locking [lachaux1999]. | `monitor/coherence.py:36` |

## Coupling Defaults

| Constant | Value | Provenance | Used in |
|----------|-------|------------|---------|
| `base_strength` | 0.45 | Empirical. Typical starting coupling for 4–16 oscillator populations; yields R ≈ 0.6–0.8 in default topology. | `coupling/knm.py`, binding specs |
| `decay_alpha` | 0.3 | Empirical. Exponential distance falloff; nearest neighbours dominate at this rate. | `coupling/knm.py`, binding specs |

## Rate Limits (Actuation)

| Constant | Value | Provenance | Used in |
|----------|-------|------------|---------|
| K rate limit | 0.1/step | Empirical. Prevents coupling shock; documented in `knobs_K_alpha_zeta_Psi.md`. | `actuation/constraints.py` |
| alpha rate limit | 0.05/step | Empirical. Lag changes beyond this destabilise transients. | `actuation/constraints.py` |
| zeta rate limit | 0.2/step | Empirical. Driver strength can change faster than coupling. | `actuation/constraints.py` |
| K range | [0, 5.0] | Empirical. Upper bound avoids numerical overflow in coupling sum. | `actuation/constraints.py` |
| alpha range | [−π, π] | Derived. Full Sakaguchi–Kuramoto lag range [sakaguchi1986]. | `actuation/constraints.py` |
| zeta range | [0, 2.0] | Empirical. Driver beyond 2.0 suppresses intrinsic dynamics. | `actuation/constraints.py` |

## Imprint Model

| Constant | Value | Provenance | Used in |
|----------|-------|------------|---------|
| `decay_rate` | 0.01 (default) | Empirical. Exponential forgetting; analogous to Ebbinghaus decay but applied to coupling modulation. Original to SPO. | `imprint/update.py`, binding specs |
| `saturation` | 2.0 (default) | Empirical. Caps imprint at 2× baseline, preventing runaway coupling amplification. | `imprint/update.py`, binding specs |

## Evaluation Protocol

| Constant | Value | Provenance | Used in |
|----------|-------|------------|---------|
| R_good target | > 0.8 | Empirical. Healthy synchronisation floor for evaluation pass. | `specs/eval_protocol.md` |
| R_bad ceiling | < 0.3 | Empirical. Matches `_R_CRITICAL` — suppressed bad-layer coherence. | `specs/eval_protocol.md` |
| convergence R threshold | 0.7 | Empirical. Speed metric: steps until R_good first exceeds 0.7. | `specs/eval_protocol.md` |
| default eval steps | 100 | Empirical. Sufficient for convergence in 4–16 oscillator populations at dt=0.01. | `specs/eval_protocol.md` |
| initial seed | 42 | Arbitrary. Fixed for deterministic replay. | `specs/eval_protocol.md` |

## Numerical Stability

| Constant | Value | Provenance | Used in |
|----------|-------|------------|---------|
| CFL-like bound | `dt < π / (max_ω + N·max_K + ζ)` | Derived. Phase change per step must stay below half-cycle; analogous to CFL condition [courant1928]. Euler-specific. | `upde/numerics.py:21` |
| `max_dt` | 0.01 | Empirical. Default upper bound for integration timestep. | `upde/numerics.py:18` |

## RK45 Adaptive Integration

| Constant | Value | Provenance | Used in |
|----------|-------|------------|---------|
| DP tableau | B5, B4, A-matrix | Derived. Dormand–Prince 5(4) pair [dormand1980]. | `upde/engine.py` |
| safety factor | 0.9 | Empirical. Standard PI step-size controller safety [hairer1993, §II.4]. | `upde/engine.py` |
| shrink bound | 0.2 | Empirical. Minimum dt scale factor per rejection. | `upde/engine.py` |
| growth bound | 5.0 | Empirical. Maximum dt scale factor per accepted step. | `upde/engine.py` |
| max rejections | 3 | Empirical. Caps retries per step before accepting with min dt. | `upde/engine.py` |
| `atol` | 1e-6 | Empirical. Absolute error tolerance for step acceptance. | `upde/numerics.py` |
| `rtol` | 1e-3 | Empirical. Relative error tolerance for step acceptance. | `upde/numerics.py` |

## Stuart-Landau Amplitude (v0.4)

| Constant | Value | Provenance | Used in |
|----------|-------|------------|---------|
| `mu` | (binding spec) | Bifurcation parameter. `mu > 0` → supercritical (limit cycle at `r = sqrt(mu)`); `mu < 0` → subcritical (decay to 0). Acebrón et al. 2005, Rev. Mod. Phys. 77, §3. | `upde/stuart_landau.py`, binding specs |
| `epsilon` | (binding spec) | Amplitude coupling strength. Scales `K^r_ij` contribution. Must be >= 0. | `upde/stuart_landau.py`, binding specs |
| `amp_coupling_strength` | 0.0 (default) | Empirical. Base strength for amplitude coupling matrix `K^r`. Zero disables amplitude coupling (phase-amplitude decoupled). | `coupling/knm.py`, binding specs |
| `amp_coupling_decay` | 0.3 (default) | Empirical. Exponential distance falloff for `K^r`, same convention as phase coupling `decay_alpha`. | `coupling/knm.py`, binding specs |
| Amplitude clamp | `max(0, r_i)` | Derived. Amplitude non-negativity enforced after every integration step; physically r represents oscillation envelope magnitude. | `upde/stuart_landau.py` |
| Initial amplitude | `sqrt(max(mu, 0))` | Derived. Equilibrium of uncoupled Stuart-Landau at `mu > 0`. | `cli.py` |
| Subcritical threshold | 0.1 | Empirical. Oscillators with `r < 0.1` counted as subcritical in `subcritical_fraction`. | `cli.py` |

## Phase-Amplitude Coupling (v0.4)

| Constant | Value | Provenance | Used in |
|----------|-------|------------|---------|
| `n_bins` | 18 (default) | Empirical. Tort et al. 2010, J. Neurophysiol.: 18 bins (20° each) standard for MI computation. | `upde/pac.py` |
| `pac_gate threshold` | 0.3 (default) | Empirical. MI above 0.3 indicates significant phase-amplitude coupling. | `upde/pac.py` |
| MI normalisation | `log(n_bins)` | Derived. KL divergence normalised by max entropy for uniform distribution. | `upde/pac.py` |

## Policy Engine

| Constant | Value | Provenance | Used in |
|----------|-------|------------|---------|
| Comparison operators | `>`, `>=`, `<`, `<=`, `==` | Standard. Evaluate policy rule conditions. | `supervisor/policy_rules.py` |
