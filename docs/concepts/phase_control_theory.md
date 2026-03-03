# Phase-Synchronization Control Theory

## Abstract

This document presents the theoretical basis for using Kuramoto phase
dynamics as a **universal control abstraction** — a domain-agnostic
framework where any system with coupled cycles is controlled by
manipulating synchronization topology rather than tracking setpoints.
We demonstrate three concrete instantiations: tokamak fusion plasma,
qubit register coherence, and multi-scale plasma MHD hierarchy, and
argue that the approach constitutes a structural advance over classical
control methods (PID, MPC, LQR) by operating on the geometric invariant
(phase coherence) rather than on domain-specific state variables.

---

## 1. Core Claim

**Synchronization is a universal control variable.**

Every physical system with coupled cyclical degrees of freedom —
rotating machinery, oscillating plasmas, precessing qubits, rhythmic
biological processes, queued computational workloads — admits a
description in terms of phase angles θ_i(t) evolving on the N-torus
T^N.  The Kuramoto order parameter R = |⟨e^{iθ}⟩| then measures
the *degree of coherence* of those phases.

The key insight: instead of designing a bespoke controller for each
domain's state variables (temperatures, pressures, currents, gate
fidelities), we:

1. **Map** domain observables → phase angles via principled extraction
   (Hilbert transform, event frequency, state-ring encoding, or
   analytic formulas).
2. **Couple** phases through a Knm matrix whose topology encodes the
   desired interaction structure.
3. **Measure** coherence R per layer and partition layers into
   *good* (synchrony to promote) and *bad* (synchrony to suppress).
4. **Act** on four universal knobs — coupling K, phase lag α,
   entrainment ζ, reference phase Ψ — which the domain binding
   translates back into physical actuators.

This four-step pipeline is the **SCPN Phase Orchestrator (SPO)**.

---

## 2. Mathematical Foundation

### 2.1 The Unified Phase Dynamics Equation (UPDE)

```
dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j − θ_i − α_ij) + ζ sin(Ψ − θ_i)
```

| Symbol | Meaning | Controllable? |
|--------|---------|---------------|
| ω_i | Natural frequency of oscillator i | No (intrinsic) |
| K_ij | Coupling strength between i and j | Yes (knob K) |
| α_ij | Phase lag between i and j | Yes (knob α) |
| ζ | External entrainment drive | Yes (knob ζ) |
| Ψ | Reference phase | Yes (knob Ψ) |

This is the Sakaguchi–Kuramoto equation [sakaguchi1986] with an
external drive term.  The coupling matrix K_ij, lag matrix α_ij,
drive ζ, and reference Ψ are the four degrees of freedom available
to the controller.

### 2.2 Order Parameter as Health Metric

For a group G of oscillators:

```
R_G e^{iψ_G} = (1/|G|) Σ_{i∈G} e^{iθ_i}
```

R_G ∈ [0, 1] measures coherence.  R = 1 means perfect phase-locking;
R = 0 means uniform scatter.  This is a **topological invariant** — it
depends only on the relative phase distribution, not on the absolute
values or the physical meaning of the oscillators.

### 2.3 Dual Objective Partition

The binding spec partitions layers into two sets:

- **good_layers**: R_good should be maximised (healthy coordination).
- **bad_layers**: R_bad should be suppressed (pathological lock-in).

This captures a pattern universal across domains:

| Domain | good (maximise R) | bad (suppress R) |
|--------|-------------------|------------------|
| Tokamak plasma | Transport barrier, equilibrium | ELM cascade, tearing modes |
| Cloud queues | Throughput coordination | Retry storm synchrony |
| Manufacturing | Line yield coherence | Sensor drift correlation |
| Neuroscience | Functional connectivity | Seizure-like hypersynchrony |
| Quantum computing | Logical qubit coherence | Decoherence crosstalk |

No classical control formulation natively represents this dual objective
without domain-specific cost function engineering.

### 2.4 Regime State Machine

The supervisor maps (R_good, R_bad, boundary violations) to four regimes:

```
NOMINAL ←→ DEGRADED ←→ CRITICAL ←→ RECOVERY
```

- NOMINAL: R_good ≥ 0.6, no hard violations.
- DEGRADED: R_good ∈ [0.3, 0.6), or R_bad rising.
- CRITICAL: R_good < 0.3, or any hard boundary violation.
- RECOVERY: exiting CRITICAL, restoring coupling.

Transitions have hysteresis (0.05) and cooldown (10 steps) to prevent
oscillation.  This is a finite-state supervisory controller
[ramadge1987] realised on top of Kuramoto observables.

---

## 3. Fusion Plasma Control

### 3.1 The Control Problem

A tokamak confines plasma (T > 10^8 K) magnetically.  The plasma
exhibits coupled oscillatory phenomena across timescales spanning
six orders of magnitude:

| Phenomenon | Timescale | Character |
|-----------|-----------|-----------|
| Micro-turbulence (ITG/TEM) | ~μs | Broadband fluctuation |
| Zonal flows | ~ms | Shear oscillation |
| MHD tearing modes | ~ms | Rotating magnetic islands |
| Sawtooth crashes | ~10–100 ms | Periodic core relaxation |
| ELM cycles | ~1–10 ms | Edge pressure collapse |
| Transport barrier | ~100 ms | Steady-state H-mode pedestal |
| Current profile (q) | ~1 s | Slow diffusive evolution |
| Global equilibrium | ~1–10 s | Grad-Shafranov force balance |

Existing tokamak control uses separate PID loops for each phenomenon:
a shape controller for equilibrium, a density controller for Greenwald
limit, NTM controllers for tearing modes, ELM pace-making, etc.
These loops are designed independently, with ad-hoc priority rules
when they conflict.

### 3.2 SPO Formulation

The `plasma_control` domainpack maps these phenomena to 8 Kuramoto
layers (16 oscillators, 2 per layer).  The observable-to-phase mapping:

| Observable | Phase formula | Physical rationale |
|-----------|---------------|-------------------|
| q_profile | 2π(q − q_min)/(q_max − q_min) | Safety factor MHD proximity |
| β_N | 2πβ_N/β_limit | Troyon stability margin |
| τ_E | 2πτ_E/τ_ref | Confinement quality fraction |
| Sawtooth count | count·π mod 2π | π-phase kick per crash event |
| ELM count | count·π mod 2π | π-phase kick per ELM event |
| MHD amplitude | 2π·amplitude/threshold | Mode activity normalised |

Each formula maps the observable's physics-relevant range onto [0, 2π).
Event-driven phenomena (sawteeth, ELMs) produce discrete phase kicks
rather than continuous evolution — this naturally represents their
crash-then-recover dynamics.

**Physics boundaries** enforce hard limits:

- q_min ≥ 1.0 (Kruskal–Shafranov MHD stability)
- β_N ≤ 2.8 (Troyon no-wall ideal limit)
- Greenwald fraction ≤ 1.2 (density limit)

These map to SPO hard boundaries → immediate CRITICAL regime if violated.

**Coupling topology**: the Knm matrix uses exponential distance decay
(K_ij = K_base · exp(−α|i−j|)) between layers, encoding the physical
reality that adjacent-timescale phenomena couple more strongly.

**Dual objective**: layers 4–6 (transport barrier, current profile,
equilibrium) are *good* — their coherence indicates stable H-mode.
Layers 0, 2, 3 (micro-turbulence, tearing, sawtooth/ELM) are *bad* —
their synchronisation indicates confinement-degrading MHD activity.

**Policy rules** implement two key responses:
1. `suppress_elm_storm`: when R_bad on the sawtooth/ELM layer exceeds
   0.7, inject phase lag α on the turbulence layer to decouple the
   cascade.
2. `restore_transport_barrier`: when R_good on the transport barrier
   drops below 0.3, boost global coupling K.

### 3.3 What This Replaces

Conventional tokamak control stacks:

| Layer | Conventional | SPO equivalent |
|-------|-------------|----------------|
| Shape control | MIMO PID on gaps/currents | Equilibrium layer R + ζ drive |
| Density | PID on line-integrated density | Greenwald boundary (hard) |
| NTM suppression | ECCD aimed at island O-point | MHD layer R_bad + α decouple |
| ELM mitigation | Pellet/RMP pacing | ELM layer R_bad + policy rule |
| q-profile | Feedforward + slow PID | Current profile layer R_good |
| Disruption avoidance | Expert-system exception handler | CRITICAL regime + Lyapunov boundary |

The SPO formulation replaces ~6 independent SISO/MIMO PID loops and
a disruption handler with a single coherence-aware supervisory
controller that sees the *relationships* between phenomena, not just
individual setpoints.

### 3.4 The Kronecker Expansion

The `PlasmaControlBridge.import_knm_spec` method performs a critical
step: it takes an (L×L) *layer-level* coupling matrix (describing how
fast turbulence couples to slow equilibrium) and expands it to an
(N×N) *oscillator-level* matrix via Kronecker product:

```
K_osc = K_layer ⊗ 1_{n×n}
```

where n is the number of oscillators per layer.  This preserves the
inter-layer coupling topology while giving each oscillator within a
layer uniform intra-block coupling (with zero self-coupling on the
diagonal).

---

## 4. Quantum Coherence Control

### 4.1 The Control Problem

A register of N qubits evolving under an XY Hamiltonian:

```
H = Σ_{i<j} J_ij (X_i X_j + Y_i Y_j) + Σ_i h_i Z_i
```

can be interpreted as N coupled oscillators in the XY plane.
The Bloch-sphere azimuthal angles φ_i play the role of Kuramoto
phases, and the coupling constants J_ij play the role of K_ij.

### 4.2 SPO Formulation

The `quantum_simulation` domainpack maps this:

- **qubit_register layer** (4 oscillators): raw XY-plane phases φ_i
  of individual qubits, extracted from the statevector.
- **logical_coherence layer** (4 oscillators): logical qubit phases
  after error correction, treated as informational events (entanglement
  success/failure timestamps → frequency → phase).

The order parameter R_qubit measures how well the physical qubits
maintain phase coherence (≈ entanglement fidelity for the XY
Hamiltonian).  R_logical measures logical error rate stability.

**Boundary**: fidelity ≥ 0.5 (hard) — below this, the quantum state
is no better than random.

The `QuantumControlBridge` provides:
- `import_artifact` / `export_artifact`: pure dict conversion, no
  quantum library needed.
- `build_quantum_circuit`: constructs a Trotterised circuit from the
  Knm coupling matrix (requires scpn-quantum-control).
- `extract_phases_from_statevector`: extracts per-qubit Bloch-sphere
  phases from a Qiskit Statevector.

### 4.3 What This Adds

Standard quantum optimal control (GRAPE, Krotov, DRAG) optimises
pulse waveforms to achieve a target unitary.  SPO offers a
complementary view: instead of optimising for a specific gate, it
monitors *ongoing coherence* as the system evolves, and adjusts
coupling topology to maintain synchrony.  This is relevant for:

- Analog quantum simulation (maintaining collective coherence in
  Hamiltonian evolution)
- Quantum error correction monitoring (tracking logical qubit phase
  stability across correction cycles)
- Variational quantum circuits (adjusting entangler structure based
  on coherence feedback)

---

## 5. The Fusion Core Bridge: Equilibrium-to-Phase Mapping

### 5.1 The Grad-Shafranov Gap

The Grad-Shafranov equation governs tokamak equilibrium:

```
Δ*ψ = −μ_0 R² dp/dψ − F dF/dψ
```

It produces continuous field solutions (ψ, q, pressure, current),
not oscillatory dynamics.  There is no natural "phase" in an
equilibrium solver output.

The `FusionCoreBridge` resolves this gap by defining six analytic
phase-mapping formulas that convert equilibrium observables into
[0, 2π) phases (§3.2 table).  Each formula maps the observable's
physical range onto the circle:

- **Linear normalisation** for continuous observables (q, β_N, τ_E,
  MHD amplitude): the ratio of current value to its physical limit
  sets the phase.  A phase near 2π means the system is near its limit.
- **Event-driven kicks** for discrete phenomena (sawtooth crashes,
  ELMs): each event advances the phase by π, producing a half-cycle
  jump.  Pairs of rapid events produce near-2π advances, which wrap
  back near zero — naturally encoding the periodic crash-recovery cycle.

This mapping is invertible (`phases_to_feedback` recovers the mean
phase and coherence), allowing closed-loop operation: equilibrium
solver → phases → UPDE evolution → feedback → equilibrium solver.

---

## 6. Novelty Analysis

### 6.1 What Exists in the Literature

| Approach | Scope | Phase-based? | Multi-scale? | Domain-agnostic? |
|----------|-------|-------------|-------------|-----------------|
| Kuramoto model [kuramoto1975] | Descriptive | Yes | No | No (physics only) |
| PID control | Control | No | No | Partially |
| MPC [rawlings2017] | Control | No | Partially | No |
| LQR/LQG | Control | No | No | Partially |
| DIII-D PCS [ferron2006] | Tokamak control | No | Partially | No |
| GRAPE [khaneja2005] | Quantum control | No | No | No |
| Synchrophasor (PMU) | Power grid | Yes | No | No |
| **SPO** | **Control** | **Yes** | **Yes** | **Yes** |

The Kuramoto model has been used to *describe* synchronisation in
power grids [dorfler2014], neural networks [breakspear2010], and
chemical oscillators.  But it has never been used as a *control
abstraction* — a framework where:

1. Arbitrary domain observables are systematically mapped into phases.
2. The Kuramoto coupling matrix K_ij becomes a **control variable**.
3. The order parameter R becomes a **health metric**.
4. A supervisory policy operates on (R_good, R_bad) to drive the
   system between regimes.

This four-part combination — observable→phase mapping + coupling
as actuator + R as metric + dual-objective supervision — is, to the
authors' knowledge, not present in any published control framework.

### 6.2 Specific Novelties

1. **Dual R_good/R_bad objective**: no standard cost function
   formulation natively distinguishes "synchrony to promote" from
   "synchrony to suppress" as structurally independent objectives.
   MPC/LQR penalise deviation from a setpoint; they do not partition
   the state space into "good coherence" and "bad coherence."

2. **Observable-to-phase mapping as a formal bridge**: the analytic
   formulas (§3.2, §5.1) constitute a systematic procedure for
   embedding non-oscillatory observables (equilibrium fields,
   event counts) into the Kuramoto state space.  This has not been
   formalised as a general control methodology.

3. **Kronecker coupling expansion** (§3.4): the technique of
   specifying inter-layer coupling at the hierarchy level and
   expanding to oscillator level via Kronecker product is a practical
   contribution that preserves multi-scale topology while keeping
   the configuration compact.

4. **Regime-based supervisory control on Kuramoto observables**:
   integrating a finite-state supervisor (NOMINAL/DEGRADED/CRITICAL/
   RECOVERY) with Kuramoto R thresholds and physics-derived hard
   boundaries (q > 1.0, β_N < 2.8) in a single framework.

5. **Domain-agnostic binding spec**: the same engine, integrator,
   supervisor, and policy rules control tokamak plasma, qubit
   registers, cloud queue systems, manufacturing lines, and
   biological oscillators — with only a YAML configuration change.

---

## 7. Why Phase-Synchronization Control Supersedes Classical Methods

### 7.1 PID: Single-Variable, No Structure

PID tracks a single variable against a setpoint.  It has no concept
of:

- Multi-scale coupling between phenomena
- Phase relationships (lead/lag) between oscillators
- Coherence as a collective property
- Dual objectives (promote some sync, suppress other sync)

A tokamak running 6 PID loops has 6 independent controllers that
can (and do) fight each other.  SPO sees the 6 phenomena as one
coupled dynamical system and acts on the coupling structure.

### 7.2 MPC: Domain-Specific, Computationally Expensive

Model Predictive Control solves an optimisation problem at each
timestep using a plant model.  It is powerful but:

- Requires a domain-specific model (cannot reuse across domains)
- Scales poorly with state dimension (quadratic or worse in N)
- Solves for actuator trajectories, not for coupling topology
- Cannot express "suppress this coherence" without manual cost
  function engineering

SPO's per-step cost is O(N²) for the Kuramoto coupling sum —
dominated by the sin(θ_j − θ_i) computation — and O(N) for the
order parameter.  No optimisation problem is solved online.

### 7.3 LQR: Linear, Assumes Quadratic Cost

Linear-Quadratic Regulator requires linearisation around an
operating point.  Phase dynamics are inherently nonlinear
(sin(·) coupling).  Linearising destroys the wrap-around topology
of the N-torus and cannot represent phase-locking transitions.

### 7.4 Expert Systems: Brittle, Non-Transferable

Current tokamak disruption avoidance uses expert-rule exception
handlers (e.g., DIII-D's PCS exception handling system).  These
rules are:

- Written by domain experts for specific machines
- Not transferable between tokamaks (let alone to other domains)
- Triggered by absolute thresholds, not relational coherence
- Unable to detect emergent pathological synchronisation

SPO's policy rules operate on *relational* observables (R_bad
on a specific layer exceeding a threshold), which transfers
across machines with the same phenomenological hierarchy.

### 7.5 The Structural Advantage

The fundamental advantage is dimensional: SPO operates on one
scalar per layer (R), plus one regime state, plus four knobs.
This is a **constant-dimension** control space regardless of how
many raw state variables the domain has.  A tokamak with 10⁴
measurement channels collapses to 8 R-values + 4 boundaries +
4 knobs.  The controller complexity does not scale with plant
complexity.

---

## 8. Practical Implications

### 8.1 For Fusion Energy

- A single SPO instance replaces the 6+ independent PID loops in a
  tokamak plasma control system.
- Physics boundaries (Kruskal–Shafranov, Troyon, Greenwald) are
  first-class hard constraints, not afterthought exception handlers.
- The `plasma_control` policy rules encode ELM suppression and
  transport barrier recovery as declarative YAML — transferable
  between machines (ITER, SPARC, DEMO) by adjusting the binding spec.
- The Lyapunov verdict import allows integration with existing
  disruption prediction systems as a soft boundary signal.

### 8.2 For Quantum Computing

- Ongoing coherence monitoring during analog quantum simulation,
  complementing gate-level optimal control.
- Entanglement topology (the Knm matrix) as a control variable,
  adjusted based on real-time R feedback.
- The fidelity hard boundary (≥ 0.5) triggers regime escalation
  before the quantum state becomes irrecoverable.

### 8.3 For General Process Control

- Cloud/queue systems: retry storm detection via R_bad, automatic
  decoupling via α lag injection (queuewaves domainpack).
- Manufacturing: sensor drift correlation detection, yield coherence
  monitoring (manufacturing_spc domainpack).
- Biology: multi-scale oscillator monitoring from cellular to
  systemic (bio_stub domainpack).

All use the *same engine* with different YAML binding specs.

---

## 9. Limitations and Open Questions

1. **Phase extraction fidelity**: the Hilbert transform, event
   frequency, and ring-encoding methods have finite bandwidth and
   quality limits.  Not all domain signals have a clean oscillatory
   component.  The quality gating system (§ oscillators_PIS.md)
   mitigates this but does not eliminate it.

2. **Coupling matrix design**: the exponential-decay Knm is a
   reasonable default but may not match the true coupling topology
   of a specific system.  Domain calibration is needed.

3. **Observable-to-phase mapping invertibility**: the analytic formulas
   (§3.2) are forward-only surjections, not bijections.  The
   `phases_to_feedback` method recovers aggregate statistics (R, mean
   phase) but cannot reconstruct the original observables.

4. **Computational limits**: the N² coupling computation limits
   real-time operation to ~10³ oscillators at ~1 kHz on current
   hardware.  The Rust FFI backend extends this to ~10⁴.

5. **Validation**: the framework is demonstrated on synthetic
   simulations.  Validation against real tokamak data (DIII-D,
   EAST, JET) and real quantum hardware (IBM Eagle/Heron) remains
   future work.

---

## 10. References

- **[kuramoto1975]** Y. Kuramoto (1975). Self-entrainment of a population of coupled non-linear oscillators. *Lecture Notes in Physics* 39, 420–422.
- **[sakaguchi1986]** H. Sakaguchi & Y. Kuramoto (1986). A soluble active rotater model showing phase transitions via mutual entertainment. *Prog. Theor. Phys.* 76, 576–581.
- **[acebron2005]** J. A. Acebrón et al. (2005). The Kuramoto model: a simple paradigm for synchronization phenomena. *Rev. Mod. Phys.* 77, 137–185.
- **[dorfler2014]** F. Dörfler & F. Bullo (2014). Synchronization in complex networks of phase oscillators: a survey. *Automatica* 50, 1539–1564.
- **[breakspear2010]** M. Breakspear, S. Heitmann & A. Daffertshofer (2010). Generative models of cortical oscillations. *Front. Hum. Neurosci.* 4, 190.
- **[rawlings2017]** J. B. Rawlings, D. Q. Mayne & M. M. Diehl (2017). *Model Predictive Control: Theory, Computation, and Design*. 2nd ed., Nob Hill Publishing.
- **[ferron2006]** J. R. Ferron et al. (2006). Real time equilibrium reconstruction for tokamak discharge control. *Nucl. Fusion* 38, 1055.
- **[khaneja2005]** N. Khaneja et al. (2005). Optimal control of coupled spin dynamics: design of NMR pulse sequences by gradient ascent. *J. Magn. Reson.* 172, 296–305.
- **[ramadge1987]** P. J. Ramadge & W. M. Wonham (1987). Supervisory control of a class of discrete event processes. *SIAM J. Control Optim.* 25(1), 206–230.
- **[dormand1980]** J. R. Dormand & P. J. Prince (1980). A family of embedded Runge-Kutta formulae. *J. Comput. Appl. Math.* 6(1), 19–26.
- **[troyon1984]** F. Troyon et al. (1984). MHD-limits to plasma confinement. *Plasma Phys. Control. Fusion* 26, 209.
- **[kruskal1958]** M. D. Kruskal & M. Schwarzschild (1954). Some instabilities of a completely ionized plasma. *Proc. R. Soc. Lond. A* 223, 348–360.
- **[greenwald2002]** M. Greenwald (2002). Density limits in toroidal plasmas. *Plasma Phys. Control. Fusion* 44, R27.
