# Safety Analysis — SCPN Phase Orchestrator

SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available

First-of-its-kind safety analysis for Kuramoto-based oscillator control
in industrial settings.

---

## 1. Scope

This document covers the safety-relevant properties of the SCPN Phase
Orchestrator when deployed as a supervisory control layer for coupled
oscillator systems. Target domains include plasma stabilisation (tokamak),
power grid synchronisation, and neuromorphic process control.

Applicable standards:

| Standard | Scope |
|----------|-------|
| IEC 62443 | Industrial automation security (zones, conduits, SL targets) |
| NERC CIP | Critical infrastructure protection (bulk electric systems) |
| IEC 61508 | Functional safety (SIL classification) |
| IEC 61511 | Safety instrumented systems (process sector) |

---

## 2. Hazard Analysis

### 2.1 Identified Hazards

| ID | Hazard | Cause | Severity | Likelihood |
|----|--------|-------|----------|------------|
| H-1 | Loss of synchronisation | Coupling gain (K) exceeds stability boundary | High | Medium |
| H-2 | Unbounded control action | ActionProjector bounds misconfigured | High | Low |
| H-3 | Regime misclassification | Order parameter R corrupted or delayed | Medium | Medium |
| H-4 | Stuck in Critical regime | Cooldown prevents recovery transition | Medium | Low |
| H-5 | TLS credential compromise | Modbus adapter certificate leak | High | Low |
| H-6 | Rate-limit violation | Control output changes faster than actuator can track | High | Medium |

### 2.2 Hazard Mitigations

| Hazard | Mitigation | Implementation |
|--------|------------|----------------|
| H-1 | STL runtime monitor: `always (R >= 0.3)` | `monitor/stl.py` with rtamt |
| H-2 | ActionProjector value clamping | `spo-supervisor/projector.rs` |
| H-2 | Kani formal proof of clamp correctness | `spo-supervisor/src/formal_safety.rs` |
| H-3 | Hysteresis and hold-step filtering | `spo-supervisor/regime.rs` |
| H-4 | Critical always bypasses cooldown | `RegimeManager::transition()` |
| H-5 | Mutual TLS with certificate validation | `adapters/modbus_tls.py` |
| H-6 | Rate-limit enforcement in ActionProjector | `spo-supervisor/projector.rs` |

---

## 3. Safety Requirements

### 3.1 Control Bounds (SR-1)

All control actions output by the ActionProjector shall satisfy:

    lo <= action.value <= hi

for the configured bounds `(lo, hi)` of each Knob. This is enforced by
`project_value()` in the Rust kernel and verified by Kani proof
`action_projector_value_clipping_contract`.

**Status: Verified by crate-owned Kani harness** (unit tests pass; CI fails on proof failure)

### 3.2 Rate Limiting (SR-2)

The absolute change in any control output between consecutive steps shall
not exceed the configured rate limit:

    |action(t) - action(t-1)| <= rate_limit

The floating supervisor path enforces this in `ActionProjector::project()`.
Certification-oriented actuator paths use the exact fixed-point helpers
`compute_adaptive_rate_limit_fixed()` and `project_fixed_point_value()`, where
the adaptive limit is bounded by:

    min_limit <= rate_limit(t) <= max_limit

and the projected actuator movement satisfies:

    |action(t) - action(t-1)| <= rate_limit(t)

Kani proofs: `adaptive_rate_limit_contract` and
`action_projector_adaptive_fixed_point_rate_limit_contract`.

**Status: Verified for the adaptive fixed-point actuator contract** (floating path covered by unit tests; CI fails on proof failure)

### 3.3 Regime FSM Ordering (SR-3)

The RegimeManager shall never transition directly from Critical to Nominal.
The path must pass through Recovery:

    Critical -> Recovery -> Nominal

This prevents abrupt removal of safety interlocks. Kani proof:
`critical_never_evaluates_directly_to_nominal`.

**Status: Verified by crate-owned Kani harness** (unit tests pass; CI fails on proof failure)

### 3.3.1 Nominal Safe Envelope (SR-3a)

From a Nominal current regime with no hard boundary violations, a finite
mean coherence summary satisfying:

    R_CRITICAL <= mean_R <= 1.0

shall not classify directly as Critical. The verified proof harness is
`nominal_safe_summary_never_classifies_critical`.

Coupling-bounds guarantees are enforced at the action-projection layer. A
full continuous-time proof that bounded `K` preserves the coherence premise
for every oscillator topology remains a separate Lyapunov/reachability task.

**Status: Verified for the discrete supervisor classification contract**

### 3.4 Critical Bypass (SR-4)

Transition to Critical regime shall always succeed, regardless of cooldown
state. Safety-critical state changes must never be delayed.

**Status: Verified by unit tests; Kani transition-state proof remains future work**

### 3.5 Transition Log Bounded (SR-5)

The transition log shall never exceed `MAX_LOG_LEN` (100) entries. Unbounded
growth would constitute a memory safety violation in long-running deployments.

**Status: Verified by unit tests; Kani bounded-log proof remains future work**

### 3.6 TLS Authentication (SR-6)

All Modbus connections to SCADA endpoints shall use TLS mutual authentication.
Plaintext Modbus TCP is not permitted in production deployments (IEC 62443 SL-2+).

**Status: Implemented** (`adapters/modbus_tls.py`), not yet penetration-tested

### 3.7 STL Runtime Monitoring (SR-7)

Deployed systems shall run continuous STL monitoring of the order parameter R
and coupling gain K against their safety specifications. Violation (negative
robustness) shall trigger an immediate regime transition to Critical.

**Status: Implemented** (`monitor/stl.py`), integration with supervisor pending

---

## 4. Verification Methods

### 4.1 Currently Verified

| Property | Method | Location |
|----------|--------|----------|
| Value clipping correctness | Kani harness + function contract + Lean fixed-point proof + unit tests | `formal_safety.rs`, `projector.rs`, `formal/lean/SPOFormal/Projector.lean` |
| Adaptive fixed-point rate limiting correctness | Kani harness + function contract + Lean fixed-point proof + unit tests | `formal_safety.rs`, `projector.rs`, `formal/lean/SPOFormal/Projector.lean` |
| Nominal safe-envelope classification | Kani harness + function contract + Lean fixed-point proof + unit tests | `formal_safety.rs`, `regime.rs`, `formal/lean/SPOFormal/Regime.lean` |
| Critical never evaluates directly to Nominal | Kani harness + function contract + Lean fixed-point proof + unit tests | `formal_safety.rs`, `regime.rs`, `formal/lean/SPOFormal/Regime.lean` |
| Degraded-band classification from Nominal | Kani harness + Lean fixed-point proof + unit tests | `formal_safety.rs`, `regime.rs`, `formal/lean/SPOFormal/Regime.lean` |
| Cooldown bypass for Critical | Unit tests | `regime.rs` tests |
| Log boundedness | Unit tests | `regime.rs` tests |
| Boundary observer triggers | Unit tests | `tests/test_supervisor_regimes.py` |
| Hysteresis correctness | Unit tests + property tests | `tests/test_regime_hysteresis.py` |

### 4.2 Not Yet Verified

| Property | Required Method | Estimated Effort |
|----------|----------------|------------------|
| Full Lyapunov stability of coupled Kuramoto under bounded `K` | Mathematical proof + numerical validation | Major research effort |
| Nonlinear stability under parameter perturbation | Bifurcation analysis (continuation methods) | Medium |
| Real-time deadline guarantees | WCET analysis on target hardware | Medium |
| TLS implementation security | Penetration testing, certificate rotation testing | Medium |
| STL monitor integration with supervisor | Integration tests with supervisor loop | Low |
| Multi-domain coupling stability | Cross-domain Lyapunov analysis | Major research effort |
| Kani proof for transition-log boundedness on actual `VecDeque` state | Kani harness over `RegimeManager::transition` | Low |
| Kani proof for cooldown bypass on actual `RegimeManager` state | Kani harness over `RegimeManager::transition` | Low |

### 4.3 Kani Integration Plan

The Kani proof harnesses live in `spo-kernel/crates/spo-supervisor/src/` and
are compiled with `cfg(kani)`. To run them:

```bash
cd spo-kernel
cargo kani -p spo-supervisor -Z function-contracts
```

The GitHub Actions Kani workflow runs the same harnesses and no longer marks
proof failures as allowed failures.

### 4.4 Lean Integration Plan

The Lean proof lane lives in `formal/lean/` and mirrors discrete fixed-point
supervisor contracts independently of the Rust/Kani implementation. To run it:

```bash
cd formal/lean
lake build
```

The GitHub Actions Lean workflow builds the Lake project whenever Lean proofs,
projector contracts, or regime-classification contracts change. The Lean lane is
not a continuous-time Kuramoto stability proof; it proves the integer/fixed-point
contract boundary for projection and finite-input regime decisions.

---

## 5. IEC 62443 Compliance Notes

### Zone Model

The Phase Orchestrator operates as a **Level 2** (supervisory control)
component. Modbus/TLS connections to Level 1 (basic control) devices
must traverse a conduit with:

- SL-2 minimum: authentication, integrity, confidentiality
- Certificate-based mutual TLS (implemented in `modbus_tls.py`)
- Network segmentation enforced by external firewall rules (not in scope)

### Security Levels

| Requirement | SL-1 | SL-2 | SL-3 | Status |
|-------------|------|------|------|--------|
| Unique user identification | -- | Yes | Yes | Implemented (TLS cert CN) |
| Integrity of communicated data | -- | Yes | Yes | Implemented (TLS) |
| Data confidentiality | -- | -- | Yes | Implemented (TLS encryption) |
| Audit trail | Yes | Yes | Yes | Implemented (`audit/` subsystem) |
| Denial-of-service resistance | -- | -- | Yes | Not yet addressed |

---

## 6. NERC CIP Applicability

For bulk electric system deployments, the following CIP standards apply:

- **CIP-005**: Electronic Security Perimeters — Modbus/TLS satisfies
  the encrypted communication requirement for routable protocols.
- **CIP-007**: System Security Management — patch management and
  malware prevention are out of scope for the orchestrator itself.
- **CIP-010**: Configuration Change Management — the audit subsystem
  and transition log support change tracking requirements.
- **CIP-013**: Supply Chain Risk Management — SBOM generation via
  CI (`wheels.yml`) and REUSE compliance address provenance.

---

## 7. Revision History

| Date | Version | Change |
|------|---------|--------|
| 2026-03-21 | 1.0 | Initial safety analysis |
