<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — nn validation xfail/skip governance -->

# nn Validation xFail/Skip Register

Purpose: satisfy v1.0 release governance for remaining `nn/` validation
exceptions by tracking each with an issue reference, owner, and release
decision.

Last reviewed: 2026-05-01

## Ownership and policy

- Owner: Arcane Sapience
- Scope: `tests/test_nn_physics_validation*.py`
- Rule: no untracked xfail/skip remains in this suite.

## Exception register

| Ref | Test location | Marker | Summary | Owner | v1.0 blocking? | Disposition |
|---|---|---|---|---|---|---|
| NNVAL-001 | `test_nn_physics_validation.py:205` | `@xfail` | CPU-JAX float32 diverges from GPU at tight tolerance | Arcane Sapience | No | Keep xfail until deterministic cross-device tolerance envelope is defined. |
| NNVAL-002 | `test_nn_physics_validation.py:389` | `pytest.xfail` | SAF gradient NaN at eigendegenerate spectra | Arcane Sapience | No | Keep xfail; known `eigh` backward degeneracy path. |
| NNVAL-003 | `test_nn_physics_validation.py:477` | `pytest.xfail` | Simplicial hysteresis not detected at current N/sigma2 | Arcane Sapience | No | Keep xfail pending larger-N parameter sweep evidence. |
| NNVAL-004 | `test_nn_physics_validation_p2.py:276` | `assert` | UDE extrapolation NaN outside train window | Arcane Sapience | No | RESOLVED 2026-06-15 — `CouplingResidual` now tanh-bounds its output so the learned correction stays in `[-1, 1]` (matching the bounded `sin` backbone) and forward integration past the training window stays finite. The test also genuinely extrapolates (100-step run with trained params) instead of slicing past the trajectory end, and asserts finite test loss with `test/train < 10`. |
| NNVAL-005 | `test_nn_physics_validation_p3.py:114` | `pytest.xfail` | Reservoir correlation below threshold without tuned K_c | Arcane Sapience | No | Keep xfail; expected operating-point sensitivity. |
| NNVAL-006 | `test_nn_physics_validation_p5.py:389` | `@xfail` | CPU-JAX float32 phase drift at strict tolerance | Arcane Sapience | No | Keep xfail with documented float64 recommendation for strict checks. |
| NNVAL-007 | `test_nn_physics_validation_p6.py:345` | `assert` | K symmetry breaks during gradient training | Arcane Sapience | No | RESOLVED 2026-06-15 — `KuramotoLayer.coupling` integrates the symmetric part `(K+Kᵀ)/2`, so the loss gradient w.r.t. `K` is symmetric and trained `K` stays exactly symmetric (`max\|K−Kᵀ\|=0`). The xfail is now a hard symmetry assertion. |
| NNVAL-008 | `test_nn_physics_validation_p6.py:677` | `pytest.xfail` | OIM Petersen graph residual violations | Arcane Sapience | No | Keep xfail; heuristic hardness case, not release blocker. |
| NNVAL-009 | `test_nn_physics_validation_p7.py:150` | `@xfail` | FIM small-N scaling non-monotonic | Arcane Sapience | No | Keep xfail; finite-size regime note. |
| NNVAL-010 | `test_nn_physics_validation_p7.py:180` | `pytest.xfail` | FIM λ_c(4) near zero finite-size effect | Arcane Sapience | No | Keep xfail with explicit small-N caveat. |
| NNVAL-011 | `test_nn_physics_validation_p7.py:292` | `pytest.xfail` | FIM hysteresis not visible in current λ/K range | Arcane Sapience | No | Keep xfail; requires expanded sweep window. |
| NNVAL-012 | `test_nn_physics_validation_p9.py:320` | `@xfail` | MI ordering fragile on CPU-JAX float32 | Arcane Sapience | No | Keep xfail; precision/device sensitivity case. |
| NNVAL-013 | `test_nn_physics_validation_p9.py:539` | `assert` | `analytical_inverse` ill-conditioned at K=0 | Arcane Sapience | No | RESOLVED 2026-06-15 — `analytical_inverse` now fits ω jointly via an intercept column, so uncoupled data is no longer confounded by ω-drift and recovers `‖K‖≈0` (0.001 vs 51.6 before). Coupled recovery unchanged (corr 1.000). The xfail is now a hard `‖K‖<0.5` assertion. |
| NNVAL-014 | `test_nn_physics_validation_p11.py:160` | `pytest.xfail` | Critical slowing metric fails to capture expected behaviour | Arcane Sapience | No | Keep xfail; test-design refinement item. |
| NNVAL-015 | `test_nn_physics_validation_p11.py:550` | `@xfail` | CPU-JAX float32 diverges from GPU at N=512 | Arcane Sapience | No | Keep xfail until large-N cross-device tolerance policy lands. |
| NNVAL-016 | `test_nn_physics_validation_p12.py:48` | `assert` | Entropy-production formula mismatch/theoretical gap | Arcane Sapience | No | RESOLVED 2026-06-15 — the inline `Σ coupling·dθ/dt` formula was wrong (signed). The production `monitor.entropy_prod.entropy_production_rate` already uses the correct non-negative dissipation `Σ(dθ/dt)²·dt` (Acebrón 2005); the test now validates that estimator (non-negative across the trajectory, σ>0 under incoherent drive, σ≈0 when frequency-locked). |
| NNVAL-017 | `test_nn_physics_validation.py:48` | `pytest.skip` | JAX x64 unavailable on some platforms | Arcane Sapience | No | Conditional skip accepted; environment capability gate. |
| NNVAL-018 | `test_nn_physics_validation_p13.py:270` | `pytest.skip` | Rust FFI not available if `spo-kernel` not compiled | Arcane Sapience | No | Conditional skip accepted; build capability gate. |

## Release gate summary

Blocking exceptions for v1.0 closure:

- None. All four previously blocking exceptions are resolved (below).

Resolved blockers (kept for history):

- NNVAL-004 (`UDE` extrapolation NaN) — resolved 2026-06-15 via tanh-bounded
  residual plus a genuine extrapolation test.
- NNVAL-016 (entropy-production contract gap) — resolved 2026-06-15 by testing
  the correct non-negative production estimator instead of a signed inline
  formula.
- NNVAL-007 (training-induced `K` asymmetry) — resolved 2026-06-15 via symmetric
  coupling parametrisation.
- NNVAL-013 (`analytical_inverse` at `K=0`) — resolved 2026-06-15 via joint
  intercept (ω) estimation removing the ω/coupling confounding.

All other listed exceptions are explicitly non-blocking with current evidence.

## Governance interpretation

This register has two operational roles:

1. **Release safety**: entries with `v1.0 blocking = Yes` are release-blockers
   until fixed or reclassified.
2. **Predictive planning**: non-blocking exceptions are tracked as controlled
   technical debt so teams can forecast risk by release train.

When a blocking item is resolved, re-run the linked test file and record the
evidence in this table before removing the row or flipping disposition.
For large sweeps, keep the owner and rationale so the same operational decision can
be audited during release review.

## Practical rollout guidance

Treat this register as a living risk ledger:

- Before release, verify each `v1.0 blocking = Yes` row is either fixed or reclassified.
- Before CI-heavy milestones, run this file’s linked tests so blockers are surfaced
  before documentation or roadmap milestones mask execution signals.
- Keep owner and rationale fields synchronized with ticket status and code comments so
  no blocker is resolved without shared evidence.

This format is intentionally strict because `nn` parity and safety surfaces are
most useful when every known exception has a defined control path.

## Expected evidence cadence

- Review quarterly or before any production tag.
- Keep `NNVAL-*` links stable even after fixes; they provide historical context for
  why the contract required that tolerance or implementation policy.
- Escalate conditional skips that are repeatedly hit in release CI into explicit
  runtime capability notes.
