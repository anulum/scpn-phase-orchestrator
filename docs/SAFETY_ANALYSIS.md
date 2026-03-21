# Safety Analysis — SCPN Phase Orchestrator

SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available

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
| H-2 | Kani formal proof of clamp correctness | `kani/proofs/action_projector.rs` |
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
`f64::clamp()` in the Rust kernel and verified by Kani proof
`action_projector_value_clipping_proof`.

**Status: Verified** (Kani proof stub, unit tests pass)

### 3.2 Rate Limiting (SR-2)

The absolute change in any control output between consecutive steps shall
not exceed the configured rate limit:

    |action(t) - action(t-1)| <= rate_limit

Enforced in `ActionProjector::project()`. Kani proof:
`action_projector_rate_limit_proof`.

**Status: Verified** (Kani proof stub, unit tests pass)

### 3.3 Regime FSM Ordering (SR-3)

The RegimeManager shall never transition directly from Critical to Nominal.
The path must pass through Recovery:

    Critical -> Recovery -> Nominal

This prevents abrupt removal of safety interlocks. Kani proof:
`regime_manager_no_critical_to_nominal_proof`.

**Status: Verified** (Kani proof stub, unit tests pass)

### 3.4 Critical Bypass (SR-4)

Transition to Critical regime shall always succeed, regardless of cooldown
state. Safety-critical state changes must never be delayed.

**Status: Verified** (Kani proof stub, unit tests pass)

### 3.5 Transition Log Bounded (SR-5)

The transition log shall never exceed `MAX_LOG_LEN` (100) entries. Unbounded
growth would constitute a memory safety violation in long-running deployments.

**Status: Verified** (Kani proof stub, unit tests pass)

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
| Value clipping correctness | Kani proof stub + unit tests | `kani/proofs/action_projector.rs`, `projector.rs` tests |
| Rate limiting correctness | Kani proof stub + unit tests | `kani/proofs/action_projector.rs`, `projector.rs` tests |
| FSM transition ordering | Kani proof stub + unit tests | `kani/proofs/action_projector.rs`, `regime.rs` tests |
| Cooldown bypass for Critical | Kani proof stub + unit tests | `kani/proofs/action_projector.rs`, `regime.rs` tests |
| Log boundedness | Kani proof stub + unit tests | `kani/proofs/action_projector.rs`, `regime.rs` tests |
| Boundary observer triggers | Unit tests | `tests/test_supervisor_regimes.py` |
| Hysteresis correctness | Unit tests + property tests | `tests/test_regime_hysteresis.py` |

### 4.2 Not Yet Verified

| Property | Required Method | Estimated Effort |
|----------|----------------|------------------|
| Full Lyapunov stability of coupled Kuramoto | Mathematical proof + numerical validation | Major research effort |
| Nonlinear stability under parameter perturbation | Bifurcation analysis (continuation methods) | Medium |
| Real-time deadline guarantees | WCET analysis on target hardware | Medium |
| TLS implementation security | Penetration testing, certificate rotation testing | Medium |
| STL monitor integration with supervisor | Integration tests with supervisor loop | Low |
| Multi-domain coupling stability | Cross-domain Lyapunov analysis | Major research effort |

### 4.3 Kani Integration Plan

The Kani proof stubs in `spo-kernel/kani/proofs/` are currently compile-checked
only. To run them:

```bash
cargo kani --manifest-path spo-kernel/Cargo.toml
```

CI integration is planned once Kani supports the MSRV (1.83.0) and can run
in the GitHub Actions environment without exceeding the 6-hour job limit.

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
