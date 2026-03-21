# Toward Safety Certification of Kuramoto-Based Control Systems

**Status: First-of-its-kind analysis. No published safety certification
exists for oscillator-based control systems. This paper identifies what
has been verified, what has not, and what would be required for
certification under IEC 62443 and IEC 61508.**

Miroslav Šotek (ORCID: 0009-0009-3560-0851)
Anulum Research

---

## Abstract

Kuramoto-type coupled oscillator networks are increasingly proposed for
control applications — power grid frequency regulation, plasma
stabilization, neuromorphic process control. No published work addresses
the safety certification of such systems under industrial standards.
This paper presents the first hazard analysis, verification inventory,
and standards mapping for a Kuramoto-based control system (the SCPN Phase
Orchestrator). We document which safety properties have been formally
verified (control-action bounding, rate limiting, FSM transition ordering
via Kani proofs), which have been tested but not proven (Lyapunov
stability monitoring, STL runtime checking), and which remain open
(full nonlinear stability proof, real-time deadline guarantees, multi-domain
coupling stability). The gap between current verification status and
IEC 62443 SL-3 / IEC 61508 SIL-2 certification is analyzed concretely.

## 1. Introduction

The Kuramoto model (Kuramoto, 1975) describes phase dynamics of coupled
oscillators:

dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j − θ_i)

Originally a theoretical tool for studying synchronization transitions
(Acebrón et al., 2005; Strogatz, 2000), it has been applied to power
grid frequency regulation (Dörfler & Bullo, 2014), cardiac pacemaker
modeling (Michaels et al., 1987), and neural synchrony (Breakspear et
al., 2010). The SCPN Phase Orchestrator (SPO) generalizes this by
treating the Kuramoto equation as a control substrate: domain-specific
problems are compiled into oscillator networks, and synchronization state
drives control actions fed back to the physical process.

When such a system is deployed for safety-critical control, it must
satisfy functional safety requirements. No published work addresses
this. The Kuramoto model's nonlinear dynamics, the potential for
desynchronization cascades, and the coupling between the oscillator
state and physical actuators create a hazard landscape with no precedent
in safety certification literature.

This paper is a gap analysis, not a certification claim. We inventory
what has been verified in SPO as of v0.4.1 and what remains.

## 2. Hazard Model

### 2.1 System Boundary

SPO operates as a Level 2 supervisory control component in the IEC 62443
zone model. It receives sensor data from Level 1 basic control devices,
computes phase dynamics, and outputs bounded control actions via an
ActionProjector. Communication uses Modbus/TLS to Level 1 and gRPC/TLS
to Level 3 (operations management).

### 2.2 Identified Hazards

Six hazards were identified through structured analysis:

| ID | Hazard | Root Cause | Severity | Likelihood |
|----|--------|------------|----------|------------|
| H-1 | Loss of synchronization | K exceeds stability boundary; coupling topology fragments | High | Medium |
| H-2 | Unbounded control action | ActionProjector bounds misconfigured or bypassed | High | Low |
| H-3 | Regime misclassification | R computation corrupted; sensor noise; race condition in FSM | Medium | Medium |
| H-4 | Stuck in Critical regime | Cooldown timer prevents legitimate recovery | Medium | Low |
| H-5 | Communication integrity loss | TLS certificate compromise; man-in-the-middle on Modbus | High | Low |
| H-6 | Rate-limit violation | Control output exceeds actuator tracking capability | High | Medium |

### 2.3 Failure Modes

Beyond the six primary hazards, the following failure modes apply to
Kuramoto-based control specifically:

**Chimera states.** Partial synchronization where some oscillator groups
lock while others drift (Abrams & Strogatz, 2004). In a control context,
chimera states produce conflicting control signals from different
oscillator subpopulations. SPO detects chimeras via cluster coherence
metrics but does not formally prevent them.

**Desynchronization cascades.** Removal of a high-degree oscillator can
trigger cascading desynchronization analogous to cascading failures in
power grids (Motter & Lai, 2002). No formal analysis of cascade
propagation exists for SPO's coupling estimation algorithm.

**Metastability traps.** The SSGF optimizer may converge to a local
minimum producing a coupling topology that sustains a metastable state —
neither fully synchronized nor incoherent — for extended periods. Whether
such states produce safe control output depends on the domain.

## 3. Verification Methods

### 3.1 Formally Verified Properties (Kani)

The Rust kernel (`spo-kernel/`) contains Kani proof stubs for five
safety properties. Kani is a model checker for Rust that explores all
possible executions of a function up to a bounded depth.

| Property | Kani Proof | Status |
|----------|-----------|--------|
| Control action value within [lo, hi] for all inputs | `action_projector_value_clipping_proof` | Compile-checked; unit tests pass |
| Rate-limited: \|a(t) − a(t−1)\| ≤ rate_limit | `action_projector_rate_limit_proof` | Compile-checked; unit tests pass |
| FSM never transitions Critical → Nominal directly | `regime_manager_no_critical_to_nominal_proof` | Compile-checked; unit tests pass |
| Critical transition bypasses cooldown | `regime_manager_critical_bypass_proof` | Compile-checked; unit tests pass |
| Transition log length ≤ MAX_LOG_LEN (100) | `regime_manager_log_bounded_proof` | Compile-checked; unit tests pass |

**Honest status.** These proofs are currently Kani proof *stubs* that
compile and are checked at build time. Full Kani verification (exhaustive
bounded model checking) requires Kani to support the project's MSRV
(1.83.0) and to complete within CI time limits. The proofs have not yet
been run to completion under Kani's model checker. They represent the
intended proof obligations, not completed verification.

### 3.2 Lyapunov Stability Monitoring

The `LyapunovGuard` module evaluates the Lyapunov function:

V(θ) = −(K/2N) Σ_{i,j} K_ij cos(θ_i − θ_j)

and monitors dV/dt ≤ 0 (non-increasing energy) and the basin condition
max|θ_i − θ_j| < π/2 for connected oscillator pairs
(van Hemmen & Wreszinski, 1993).

**What this verifies.** When dV/dt ≤ 0 and the basin condition holds,
the system is in the basin of attraction of the synchronized state for
the all-to-all Kuramoto model.

**What this does not verify.** (a) The Lyapunov function is computed
from the current state, not proven to hold for all future states.
Runtime monitoring detects violations after they occur. (b) The basin
condition (max phase difference < π/2) is sufficient for all-to-all
coupling; for sparse or heterogeneous coupling, tighter conditions
apply (Dörfler & Bullo, 2014, Theorem 4.1) that are not checked.
(c) The Sakaguchi-Kuramoto extension (phase frustration α_ij ≠ 0)
breaks the gradient structure; V is no longer a true Lyapunov function
in this case. SPO still computes V when frustration is enabled, but the
stability guarantee does not hold.

### 3.3 Signal Temporal Logic (STL) Runtime Monitoring

The `STLMonitor` wraps the rtamt library to evaluate STL specifications
against numeric traces at runtime:

- `always (R >= 0.3)` — synchronization floor
- `always (K <= 10.0)` — coupling gain ceiling

Negative robustness (violation) triggers an immediate transition to
Critical regime.

**What this verifies.** Runtime violations are detected within one
integration step of occurrence. The regime transition is immediate
(Critical bypasses cooldown, per Kani proof stub).

**What this does not verify.** STL monitoring is reactive, not
predictive. It cannot guarantee that R will remain above 0.3 in the
next step — only that the violation will be caught when it happens.
Predictive STL (Donzé et al., 2010) using a model of future dynamics
would provide stronger guarantees but is not implemented.

### 3.4 Unit and Integration Tests

SPO has 1305+ tests covering:
- Regime FSM transitions and hysteresis
- ActionProjector bounding and rate limiting
- Order parameter computation
- SSGF cost function correctness
- STL monitor activation
- Lyapunov guard basin detection
- All 32 domainpack compilation

Tests are executed in CI on Python 3.10-3.13 with full coverage.

## 4. Standards Mapping

### 4.1 IEC 62443 (Industrial Automation Security)

SPO operates at Security Level 2 (SL-2) with partial SL-3 coverage:

| Requirement | SL-2 | SL-3 | SPO Status |
|-------------|------|------|------------|
| Unique identification | Required | Required | Implemented (TLS cert CN) |
| Data integrity | Required | Required | Implemented (TLS) |
| Data confidentiality | — | Required | Implemented (TLS encryption) |
| Audit trail | Required | Required | Implemented (audit subsystem) |
| Denial-of-service resistance | — | Required | **Not addressed** |
| Restrict data flow | Required | Required | Implemented (zone boundaries) |
| Timely response to events | — | Required | Implemented (Critical regime) |

**Gap: DoS resistance.** SPO does not implement rate limiting on inbound
gRPC or Modbus/TLS connections. A sustained connection flood could prevent
legitimate sensor data from reaching the phase dynamics engine. Mitigation
requires network-layer defenses (firewall rules, connection limits) outside
SPO's scope.

### 4.2 IEC 61508 (Functional Safety)

For SIL-2 classification (target for supervisory control), IEC 61508
requires:

| Requirement | SIL-2 | SPO Status |
|-------------|-------|------------|
| Formal specification of safety requirements | Required | Partially met (SR-1 through SR-7 documented) |
| Semi-formal design methods | Highly recommended | Met (typed interfaces, FSM diagrams) |
| Formal verification of critical modules | Recommended | Partially met (Kani proof stubs, not yet run to completion) |
| Diagnostic coverage ≥ 90% | Required | Met for regime FSM and ActionProjector; not measured system-wide |
| Hardware fault tolerance (HFT ≥ 0) | Required | **Not applicable** (software component; relies on HW redundancy) |
| Common-cause failure analysis | Required | **Not performed** |
| Systematic capability ≥ SIL 2 | Required | **Not assessed** by accredited body |

**Gaps.** (a) Kani proofs must run to completion, not just compile.
(b) Common-cause failure analysis for coupled oscillator dynamics
has no precedent — a methodology would need to be developed.
(c) SIL-2 systematic capability assessment requires an accredited
assessor.

### 4.3 NERC CIP (Bulk Electric System)

For power grid deployments:

| CIP Standard | Requirement | SPO Status |
|-------------|-------------|------------|
| CIP-005 | Encrypted communication | Met (Modbus/TLS) |
| CIP-007 | Patch management | Out of scope (OS-level) |
| CIP-010 | Configuration change management | Met (audit log, transition log) |
| CIP-013 | Supply chain risk management | Met (SBOM, REUSE, SPDX compliance) |

## 5. Gap Analysis

### 5.1 Verified

| Property | Method | Confidence |
|----------|--------|------------|
| Control action bounding | Kani proof stub + tests | Medium (stub, not full proof) |
| Rate limiting | Kani proof stub + tests | Medium |
| FSM transition ordering | Kani proof stub + tests | Medium |
| Cooldown bypass for Critical | Kani proof stub + tests | Medium |
| Log boundedness | Kani proof stub + tests | Medium |
| Runtime synchronization monitoring | STL + Lyapunov + tests | Medium (reactive only) |
| Communication encryption | TLS implementation + tests | Medium (not pen-tested) |

### 5.2 Not Verified

| Property | Required Method | Estimated Effort | Blocking Certification? |
|----------|----------------|------------------|------------------------|
| Full Lyapunov stability of coupled Kuramoto under arbitrary topology | Mathematical proof + numerical continuation | Major research | Yes (SIL-2) |
| Nonlinear stability under parameter perturbation | Bifurcation analysis | Medium | Yes (SIL-2) |
| Real-time deadline guarantees (WCET) | WCET analysis on target hardware | Medium | Yes (for hard-RT deployments) |
| Desynchronization cascade propagation bounds | Graph-theoretic analysis + simulation | Medium | Domain-dependent |
| Chimera state prevention or detection guarantees | Open research problem | Major research | Domain-dependent |
| TLS implementation security | Penetration testing + cert rotation testing | Medium | Yes (SL-3) |
| Common-cause failure analysis | Novel methodology development | Major | Yes (SIL-2) |
| Multi-domain coupling stability | Cross-domain Lyapunov analysis | Major research | Yes (multi-domain deployments) |

### 5.3 Summary Assessment

SPO has verified the "last mile" of control safety — the ActionProjector
provably bounds outputs and limits rates, and the regime FSM provably
maintains transition ordering. These are necessary but insufficient for
system-level safety certification.

The primary gaps are upstream: the stability of the Kuramoto dynamics
themselves, the behavior under topology changes (SSGF updates W at each
outer cycle), and the absence of WCET analysis. These are not
engineering oversights; they are open research problems specific to
oscillator-based control. No other project has addressed them either.

A realistic path to SIL-2 certification would require:

1. Run Kani proofs to completion (engineering effort, not research).
2. Prove Lyapunov stability for the specific coupling topologies
   produced by each domainpack (per-domain effort).
3. Perform WCET analysis on the Rust kernel for target hardware.
4. Commission penetration testing of the TLS implementation.
5. Develop a common-cause failure methodology for coupled oscillator
   systems (novel contribution).
6. Engage an accredited assessor for SIL-2 systematic capability.

Steps 1, 3, and 4 are straightforward engineering. Steps 2, 5, and 6
require research or external engagement that does not yet exist in the
literature.

## References

- Abrams, D.M. & Strogatz, S.H. (2004). Chimera states for coupled oscillators. *Phys. Rev. Lett.* 93(17), 174102.
- Acebrón, J.A. et al. (2005). The Kuramoto model: A simple paradigm for synchronization phenomena. *Rev. Mod. Phys.* 77(1), 137-185.
- Ames, A.D. et al. (2017). Control barrier function based quadratic programs for safety critical systems. *IEEE Trans. Autom. Control* 62(8), 3861-3876.
- Breakspear, M. et al. (2010). Generative models of cortical oscillations. *Neuroinformatics* 8(3), 159-167.
- Donzé, A. et al. (2010). Robust satisfaction of temporal logic over real-valued signals. *FORMATS 2010*, LNCS 6246, 92-106.
- Dörfler, F. & Bullo, F. (2014). Synchronization in complex networks of phase oscillators: A survey. *Automatica* 50(6), 1539-1564.
- IEC 61508 (2010). Functional safety of electrical/electronic/programmable electronic safety-related systems. International Electrotechnical Commission.
- IEC 62443 (2018). Security for industrial automation and control systems. International Electrotechnical Commission.
- Kuramoto, Y. (1975). Self-entrainment of a population of coupled non-linear oscillators. *Lecture Notes in Physics* 39, 420-422.
- Michaels, D.C. et al. (1987). Mechanisms of biological pacemaker periodicity. *Circ. Res.* 61(5), 704-714.
- Motter, A.E. & Lai, Y.-C. (2002). Cascade-based attacks on complex networks. *Phys. Rev. E* 66(6), 065102.
- Strogatz, S.H. (2000). From Kuramoto to Crawford. *Physica D* 143(1-4), 1-20.
- van Hemmen, J.L. & Wreszinski, W.F. (1993). Lyapunov function for the Kuramoto model of nonlinearly coupled oscillators. *J. Stat. Phys.* 72, 145-166.
