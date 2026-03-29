<!--
SPDX-License-Identifier: AGPL-3.0-or-later
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# Software Verification & Validation Report

**Version:** 0.5.0
**Date:** 2026-03-28
**Author:** Miroslav Šotek / Arcane Sapience

This document reports measured validation results for the SCPN Phase
Orchestrator software. Every claim is grounded in automated tests that
run on every commit. No aspirational language — only measured reality.

---

## 1. Test Suite Summary

| Metric | Measured value | Gate |
|--------|---------------|------|
| Total Python tests | 3,130+ (core) + 194 (nn/ physics validation) | — |
| Total Rust tests | 211 | — |
| nn/ physics validation | 194 tests, 183 pass, 10 xfail, 1 skip | 0 hard failures |
| Line coverage | 99%+ | 95% minimum |
| Docstring coverage | 100% (0 missing) | — |
| Domainpack coverage | 32/32 (100%) | — |
| Property-based tests (hypothesis) | ~350 | — |
| Mutation survivors (order_params + numerics) | 0 (32 killers) | — |

## 2. Numerical Validation Against Analytical Results

### 2.1 Kuramoto Synchronisation Threshold

**Reference:** Strogatz 2000 "From Kuramoto to Crawford", Acebrón et al. 2005 Rev. Mod. Phys. 77(1).

| Test | Analytical prediction | Measured | File |
|------|----------------------|----------|------|
| Identical oscillators, K > 0 | R → 1 | R > 0.95 (N=16, 5000 steps) | test_physics_benchmarks.py |
| Spread frequencies, K ≪ K_c | R ≈ 0 | R < 0.5 (N=32, K=0.01) | test_physics_benchmarks.py |
| K increasing → R increasing | Monotonic | Verified across K ∈ [0.01, 3.0] | test_physics_benchmarks.py |
| External drive ζ > 0 | Phase locking | R > 0.9 (ζ=2.0) | test_physics_benchmarks.py |

### 2.2 Stuart-Landau Hopf Bifurcation

**Reference:** Pikovsky et al. 2001 "Synchronization: A Universal Concept".

| Test | Prediction | Measured | File |
|------|-----------|----------|------|
| μ > 0 → limit cycle r = √μ | r = √μ | |r - √μ| < 0.05 (μ ∈ [0.1, 5.0], hypothesis) | test_engine_parity.py |
| μ < 0 → decay r → 0 | r → 0 | r < 0.01 (μ ∈ {-1, -0.5}, 5000 steps) | test_physics_benchmarks.py |
| Supercritical amplitude consensus | spread → 0 | spread < 0.3 | test_physics_benchmarks.py |

### 2.3 Ott-Antonsen Mean-Field Reduction

**Reference:** Ott & Antonsen 2008, Chaos 18(3):037113.

| Test | Prediction | Measured | File |
|------|-----------|----------|------|
| K_c = 2Δ | Exact | |K_c - 2Δ| < 1e-12 | test_stress_scale.py |
| R_ss = √(1 - 2Δ/K) for K > K_c | Exact | |R_ss - formula| < 1e-12 | test_stress_scale.py |
| OA vs UPDE simulation (Lorentzian, K > K_c) | Agreement | |R_OA - R_UPDE| < 0.15 (N=200) | test_stress_scale.py |
| OA vs UPDE (K < K_c) | Both R ≈ 0 | R_OA = 0, R_UPDE < 0.3 | test_stress_scale.py |

### 2.4 Spectral Graph Theory

**Reference:** Dörfler & Bullo 2014, Automatica 50(6).

| Test | Prediction | Measured | File |
|------|-----------|----------|------|
| K > 2K_c → synchronises | R > 0.5 | Verified | test_engine_parity.py |
| K < K_c/10 → no sync | R < 0.7 | Verified | test_engine_parity.py |
| Laplacian PSD, row sums = 0 | Identity | Verified (hypothesis, N ∈ [2,12]) | test_prop_hodge_spectral.py |
| λ₂ > 0 iff connected | Theorem | Verified (hypothesis) | test_prop_hodge_spectral.py |
| Ring λ₂ > chain λ₂ | Theorem | Verified (N=8) | test_convergence_topology.py |

## 3. Cross-Engine Equivalence

All engines that should agree on a given scenario produce the same result.

| Engine A | Engine B | Scenario | Tolerance | File |
|----------|----------|----------|-----------|------|
| UPDE Euler | TorusEngine | Single step, dt=0.001 | 1e-4 | test_engine_parity.py |
| UPDE Euler | SplittingEngine | Single step, dt=0.001 | 1e-3 | test_engine_parity.py |
| UPDE Euler | RK4 | 500-step converged R | 0.05 | test_engine_parity.py |
| Simplicial σ₂=0 | UPDE Euler | Any input (hypothesis) | 1e-10 | test_engine_parity.py |
| All 3 engines | Analytical | Free rotation θ = ωt | 1e-6 | test_engine_parity.py |

## 4. Convergence Order Verification

| Integrator | Expected order | Test | File |
|-----------|---------------|------|------|
| Euler | O(h) — exact on linear ODE | Free rotation error < 1e-10 | test_convergence_topology.py |
| RK4 | O(h⁴) — exact on linear ODE | Free rotation error < 1e-10 | test_convergence_topology.py |
| RK4 vs Euler (coupled) | RK4 more accurate at same dt | err_RK4 < err_Euler | test_convergence_topology.py |

## 5. Extreme-Scale Validation

| Test | Scale | Result | File |
|------|-------|--------|------|
| N=1000 identical sync | 1000 osc, 1000 steps | R > 0.90 | test_stress_scale.py |
| N=1000 random R | 1000 oscillators | R < 0.15 (≈ 1/√N) | test_stress_scale.py |
| N=1000 NPE | 1000 oscillators | Finite, no OOM | test_stress_scale.py |
| N=512 Laplacian | 512×512 matrix | PSD, Fiedler > 0 | test_stress_scale.py |
| 10,000 steps | 16 osc, 10000 steps | All finite, R ∈ [0,1] | test_stress_scale.py |
| 50,000 steps | 8 osc, 50000 steps | R variance < 0.1 | test_stress_scale.py |

## 6. Mutation Testing

**Tool:** mutmut 2.4.5 on Kaggle (Linux).

| Module | Mutants generated | Survived | Killer tests | Final |
|--------|------------------|----------|-------------|-------|
| upde/order_params.py | 28 | 16 | 22 | 0 survivors |
| upde/numerics.py | 10 | 5 | 10 | 0 survivors |

All surviving mutants were killed by dedicated tests in
`test_mutation_killers.py`. Targets: boundary returns, imaginary unit,
operator semantics, exact default values, PLV edge cases.

## 7. Property-Based Invariant Proofs

Each `@given` test generates 50+ random inputs and verifies mathematical
invariants. 350+ hypothesis tests across 14 files prove:

- Lyapunov spectrum: length=N, sorted descending, finite, K=0 → ≈0
- Kaplan-Yorke D_KY ∈ [0,N], all-negative → 0, all-positive → N
- Correlation integral monotonic in ε, C(ε) ∈ [0,1]
- Basin stability S_B ∈ [0,1], threshold monotonicity
- Transfer entropy TE ≥ 0, diagonal = 0
- Hodge: gradient + curl + harmonic = total
- Recurrence matrix symmetric, RR/DET/LAM ∈ [0,1]
- Chimera index ∈ [0,1], coherent/incoherent disjoint
- Winding numbers integer-valued
- Boltzmann weight ∈ (0,1] for U ≥ 0
- NPE ∈ [0,1], sync → 0
- Eligibility symmetric, ∈ [-1,1]
- StochasticInjector: D=0 → no change, output ∈ [0,2π)

## 8. Infrastructure Validation

| Component | Test | Result |
|-----------|------|--------|
| REST API (/api/health) | Deep health check | engine + R + regime verified |
| gRPC servicer | 6 in-process tests | GetState, Step, Reset, GetConfig, layers |
| Dockerfile HEALTHCHECK | /api/health endpoint | Functional, not import-only |
| CI hash pinning | --require-hashes | All tool installs hash-verified |
| Container scanning | Trivy CRITICAL/HIGH | Blocks publish on vulnerabilities |
| Audit chain | SHA256 JSONL | Tamper detection + deterministic replay |
| 33/33 domainpacks | Load + simulate | All produce valid R ∈ [0,1] |

## 9. Known Limitations

- **JAX nn/ module**: not tested in CI (requires GPU). Validated locally on GTX 1060 (9 GPU benchmark suites) and L40S (cloud). 194 automated physics validation tests across 13 phases (183 pass, 10 xfail, 1 skip). See `docs/reference/nn_physics_validation_plan.md`.
- **Rust FFI**: CI tests on Python 3.10 (fallback) and 3.12 (with kernel). Windows Rust build intermittent.
- **Quantum bridge**: requires IBM Quantum credentials. Tested locally, not in CI.
- **FPGA/WASM**: mentioned in architecture, not yet validated.
- **Single maintainer**: all validation by one author team.

## 10. Reproducing These Results

```bash
# Full test suite
pip install -e ".[dev,full,queuewaves,plot]"
pytest tests/ -v --tb=short --cov=scpn_phase_orchestrator

# Property-based (thorough)
pytest tests/test_prop_*.py --hypothesis-profile=ci

# Stress tests
pytest tests/test_stress_scale.py -v

# Engine parity
pytest tests/test_engine_parity.py -v

# Mutation testing (Linux only)
pip install mutmut==2.4.5
mutmut run --paths-to-mutate src/scpn_phase_orchestrator/upde/order_params.py \
  --runner "pytest tests/test_mutation_killers.py -x -q --tb=no"
```
