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
| NNVAL-004 | `test_nn_physics_validation_p2.py:287` | `pytest.xfail` | UDE extrapolation NaN outside train window | Arcane Sapience | **Yes** | Must be fixed or hard-gated before v1.0 freeze. |
| NNVAL-005 | `test_nn_physics_validation_p3.py:114` | `pytest.xfail` | Reservoir correlation below threshold without tuned K_c | Arcane Sapience | No | Keep xfail; expected operating-point sensitivity. |
| NNVAL-006 | `test_nn_physics_validation_p5.py:389` | `@xfail` | CPU-JAX float32 phase drift at strict tolerance | Arcane Sapience | No | Keep xfail with documented float64 recommendation for strict checks. |
| NNVAL-007 | `test_nn_physics_validation_p6.py:351` | `pytest.xfail` | K symmetry breaks during gradient training | Arcane Sapience | **Yes** | Must be fixed via symmetry projection or constrained parametrisation pre-v1.0. |
| NNVAL-008 | `test_nn_physics_validation_p6.py:677` | `pytest.xfail` | OIM Petersen graph residual violations | Arcane Sapience | No | Keep xfail; heuristic hardness case, not release blocker. |
| NNVAL-009 | `test_nn_physics_validation_p7.py:150` | `@xfail` | FIM small-N scaling non-monotonic | Arcane Sapience | No | Keep xfail; finite-size regime note. |
| NNVAL-010 | `test_nn_physics_validation_p7.py:180` | `pytest.xfail` | FIM λ_c(4) near zero finite-size effect | Arcane Sapience | No | Keep xfail with explicit small-N caveat. |
| NNVAL-011 | `test_nn_physics_validation_p7.py:292` | `pytest.xfail` | FIM hysteresis not visible in current λ/K range | Arcane Sapience | No | Keep xfail; requires expanded sweep window. |
| NNVAL-012 | `test_nn_physics_validation_p9.py:320` | `@xfail` | MI ordering fragile on CPU-JAX float32 | Arcane Sapience | No | Keep xfail; precision/device sensitivity case. |
| NNVAL-013 | `test_nn_physics_validation_p9.py:548` | `pytest.xfail` | `analytical_inverse` ill-conditioned at K=0 | Arcane Sapience | **Yes** | Add regularisation/conditioning gate before v1.0 freeze. |
| NNVAL-014 | `test_nn_physics_validation_p11.py:160` | `pytest.xfail` | Critical slowing metric fails to capture expected behaviour | Arcane Sapience | No | Keep xfail; test-design refinement item. |
| NNVAL-015 | `test_nn_physics_validation_p11.py:550` | `@xfail` | CPU-JAX float32 diverges from GPU at N=512 | Arcane Sapience | No | Keep xfail until large-N cross-device tolerance policy lands. |
| NNVAL-016 | `test_nn_physics_validation_p12.py:83` | `pytest.xfail` | Entropy-production formula mismatch/theoretical gap | Arcane Sapience | **Yes** | Resolve formula contract or downgrade claim pre-v1.0 freeze. |
| NNVAL-017 | `test_nn_physics_validation.py:48` | `pytest.skip` | JAX x64 unavailable on some platforms | Arcane Sapience | No | Conditional skip accepted; environment capability gate. |
| NNVAL-018 | `test_nn_physics_validation_p13.py:270` | `pytest.skip` | Rust FFI not available if `spo-kernel` not compiled | Arcane Sapience | No | Conditional skip accepted; build capability gate. |

## Release gate summary

Blocking exceptions for v1.0 closure:

- NNVAL-004 (`UDE` extrapolation NaN)
- NNVAL-007 (training-induced `K` asymmetry)
- NNVAL-013 (`analytical_inverse` at `K=0`)
- NNVAL-016 (entropy-production contract gap)

All other listed exceptions are explicitly non-blocking with current evidence.
