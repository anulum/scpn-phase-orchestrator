# Gemini Final Audit Report — scpn-phase-orchestrator

**Date:** 2026-03-19
**Repo:** `03_CODE/scpn-phase-orchestrator/`
**Status:** **NO-GO** (Blocking findings identified)

---

## 1. Fix Verification Matrix (A1–A30)

| ID | Description | Result | Evidence / Note |
|:---|:---|:---:|:---|
| **A1** | RK45 7-stage DP54 Coefficients | **PASS** | `upde/engine.py:28`, `dp_tableau.rs:12`. Matches Dormand-Prince (1980). |
| **A2** | Audit replay state mismatch | **PASS** | `cli.py:273`. `logged_zeta` and `logged_psi` captured before action loop. |
| **A3** | SL replay — amplitude fields | **PASS** | `audit/replay.py:126`, `cli.py:421`. Log entries include full SL state. |
| **A4** | NaN validation (All engines) | **PASS** | Verified in `engine.py:100`, `stuart_landau.py:145`, `upde.rs:88`, `stuart_landau.rs:124`. |
| **A5** | Empty array guards | **PASS** | `upde/order_params.py:22`, `upde/order_params.py:44`. |
| **A6** | PAC n_bins guard | **PASS** | `upde/pac.py:20`. Returns 0.0 for `n_bins < 2`. |
| **A7** | JAX method validation | **PASS** | `upde/jax_engine.py:117`. Raises `ValueError` for `rk45`. |
| **A8** | Server regime evaluation | **PASS** | `server.py:117`. Evaluates and transitions before returning snapshot. |
| **A9** | `metaphysics_demo` policy syntax | **PASS** | `domainpacks/metaphysics_demo/policy.yaml:3`. `regime` is a list. |
| **A10** | Domainpack `run.py` SL mode | **PASS** | `cardiac_rhythm/run.py:133`, `neuroscience_eeg/run.py:130`. |
| **A11** | `init_phases` family resolution | **PASS** | `oscillators/init_phases.py:103`. Correctly resolves from explicit family or RR. |
| **A12** | Coverage pragmas | **PASS** | Verified on JAX, BrainFlow, Modbus imports and adapters. |
| **A13** | Docker digest pinning | **PASS** | `Dockerfile:7,20,32`. All stages use `@sha256`. |
| **A14** | `get_omegas` validation | **PASS** | `binding/types.py:165`. Raises `ValueError` on length mismatch. |
| **A15** | `SimulatedBoard` timestamps | **PASS** | `hardware_io.py:160`. Uses `np.arange` and advances `self._t`. |
| **A16** | Server `asyncio.Lock` | **PASS** | `server.py:157`. Lock created and used in async endpoints. |
| **A17** | `mu` imprint modulation | **PASS** | `cardiac_rhythm/run.py:134`, `neuroscience_eeg/run.py:131`. |
| **A18** | Replay zero-fill guard | **PASS** | `audit/replay.py:136`. Skips SL verification with warning if fields missing. |
| **A27** | SL docstring lag term | **PASS** | `upde/stuart_landau.py:29`. Equation updated to include `- α_ij`. |
| **A28** | `knm_r` imprint modulation | **FAIL** | **Missing.** `ImprintModel` does not modulate `knm_r`. |
| **A29** | Family round-robin sort | **PASS** | `oscillators/init_phases.py:110`. Uses `sorted(families.keys())`. |
| **A30** | Server XSS | **FAIL** | **Risk.** `DASHBOARD_HTML` uses `innerHTML` with `l.name`. |

---

## 2. NaN Validation Matrix (A4)

| Parameter | Python UPDE | Python SL | Rust UPDE | Rust SL |
|-----------|:-----------:|:---------:|:---------:|:-------:|
| phases/state | CHECKED | CHECKED | CHECKED | CHECKED |
| omegas | CHECKED | CHECKED | CHECKED | CHECKED |
| mu | N/A | CHECKED | N/A | CHECKED |
| knm | CHECKED | CHECKED | CHECKED | CHECKED |
| knm_r | N/A | CHECKED | N/A | CHECKED |
| alpha | CHECKED | CHECKED | CHECKED | CHECKED |
| zeta | CHECKED | CHECKED | CHECKED | CHECKED |
| psi | CHECKED | CHECKED | CHECKED | CHECKED |
| epsilon | N/A | CHECKED | N/A | CHECKED |

---

## 3. Fresh Audit Findings (B1–B15)

| ID | File:Line | Severity | Description | Recommendation |
|:---|:---|:---:|:---|:---|
| **B1** | `domainpacks/metaphysics_demo/policy.yaml`:3 | **CRITICAL** | Invalid regime names `desynchronised`/`synchronised` (not in `Regime` enum). | Update to `NOMINAL`/`DEGRADED`. |
| **B2** | `src/.../server.py`:150 | **IMPORTANT** | `/api/state` endpoint not locked; risk of inconsistent data snapshot. | Wrap in `async with sim._lock:`. |
| **B3** | `src/.../upde/stuart_landau.py`:29 | **MINOR** | Docstring missing `- α_ij` in amplitude ODE equation (Code is correct). | Synchronize docstring with code. |
| **B4** | `VALIDATION.md`:147 | **MINOR** | Documentation missing `- α_ij` in amplitude ODE equation. | Update documentation. |
| **B5** | `src/.../audit/logger.py`:34 | **MINOR** | `json.dumps` without `sort_keys=True` risks non-canonical hashes. | Add `sort_keys=True`. |

---

## 4. Test Gap Analysis (C)

1. **RK45 Tableau (Critical)**: No tests verify the literal values of `_DP_A`, `_DP_B4`, `_DP_B5` against the Dormand-Prince reference.
2. **Audit Pre-action State (Important)**: No test confirms that `zeta`/`psi` recorded in the audit log are the values *before* policy modification.
3. **Audit SL Fields (Important)**: No test verifies that SL-mode logs contain `amplitudes`, `mu`, `knm_r`, and `epsilon`.
4. **Hardware Continuity (Important)**: `SimulatedBoardAdapter` lacks a test ensuring no timestamp overlap/gap between consecutive reads.

---

## 5. Overall Verdict

**Verdict:** **NO-GO**

**Blocking Conditions:**
- Fix `metaphysics_demo` policy regime names (**B1**).
- Lock the `/api/state` endpoint in `server.py` (**B2**).
- Implement `knm_r` modulation in `ImprintModel` (**A28**) or document its intentional exclusion.
- Add RK45 tableau coefficient verification tests (**C1**).
